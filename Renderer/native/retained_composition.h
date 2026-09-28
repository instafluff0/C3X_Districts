#pragma once
#include "gpu_image_compositor.h"
#include <map>
#include <memory>
#include <functional>

namespace c3x_gpu_images {
// Versioned rectangular writes preserve the native composition order. A copy
// captures its source version, not a mutable image handle. Opaque writes cut
// away covered history; transparent writes retain only the affected underlay.
// All execution uses the production packed/full-color compositor.
class RetainedComposition {
public:
    struct Work {unsigned operations=0,assemblies=0,copies=0;std::uint64_t copied_pixels=0,assembly_pixels=0;};
    using Texture=ComPtr<ID3D11Texture2D>;
    struct SampledImage {
        enum class Kind { unchanged, immutable, bgra, frozen };
        Kind kind=Kind::unchanged;Texture texture;Rect area{};
        SampledImage()=default;
        // The source owner has moved to another camera. Preserve this exact
        // completed image, but retire its animation callback and dependencies.
        static SampledImage frozen(){SampledImage result;result.kind=Kind::frozen;return result;}
        SampledImage(Texture value):kind(Kind::immutable),texture(std::move(value)){}
        // Borrow a working BGRA surface only until this sample is consumed.
        // The retained node imports it into its own reusable packed output.
        static SampledImage bgra(ID3D11Texture2D* source,Rect region){
            SampledImage result;result.kind=Kind::bgra;result.texture=source;result.area=region;return result;
        }
    };
    using Sample=std::function<SampledImage(long long,long long)>;
    struct Direct {
        // Sample copied state once per frame, then execute against assembled
        // resident underlays in native order. No finished unit source image.
        bool animated=false;
        std::uint64_t input_bytes=0; // immutable direct-pass payload, charged with retained outputs
        std::function<std::uint64_t(long long,long long)> revision;
        std::function<bool(Compositor&,Command const&)> draw;
    };
private:
    struct Node;
    struct Patch {Rect area;std::shared_ptr<Node> node;unsigned output=0;};
    struct Picture {unsigned width=0,height=0;Format format=Format::bgra32;std::vector<Patch> patches;bool partitioned=true;std::uint64_t version=0;Rect required{};};
    struct Node {
        Rect area{};Command command{};bool operation=false,dynamic=false,map_dynamic=false,constant=false;
        Picture inputs[6];Id original[6]={};
        Texture output[2];std::uint64_t bytes[2]={},revision=0,seen=0,sampled=0,direct_revision=0;
        ComPtr<ID3D11ShaderResourceView> output_view[2];
        std::vector<std::uint64_t> dependencies;
        Sample sample;Direct direct;Compositor::ImportTarget sample_target;
    };
    ID3D11Device* device;ID3D11DeviceContext* context;
    Compositor replay;
    std::map<Id,Picture> images;
    Picture front;
    std::uint64_t serial=0,frame=0,front_revision=0,drawn_revision=0;
    std::vector<std::uint64_t> drawn_dependencies;
    bool admitted=true;
    std::size_t nodes=0;std::uint64_t resident_bytes=0;
    std::uint64_t sample_allocations=0,sample_imports=0,source_views=0;
    Work work;
    // Match the bounded live native family, including saved packed/full-color
    // versions and old/new immutable map overlap at fullscreen.
    constexpr static std::uint64_t resident_budget=256u*1024u*1024u;
    Rect intersect(Rect a,Rect b)const{return {std::max(a.left,b.left),std::max(a.top,b.top),std::min(a.right,b.right),std::min(a.bottom,b.bottom)};}
    bool empty(Rect r)const{return r.left>=r.right||r.top>=r.bottom;}
    Rect extent(Picture const& p)const{return {0,0,int(p.width),int(p.height)};}
    std::shared_ptr<Node> node(){
        if(nodes>=32768)throw std::runtime_error("retained composition node budget");
        auto value=new Node;++nodes;return std::shared_ptr<Node>(value,[this](Node* p){
            resident_bytes-=p->bytes[0]+p->bytes[1]+p->direct.input_bytes;delete p;--nodes;});
    }
    void output(Node& n,unsigned index,Texture texture){
        D3D11_TEXTURE2D_DESC d={};if(texture)texture->GetDesc(&d);
        auto bytes=std::uint64_t(d.Width)*d.Height*4;
        if(bytes>resident_budget-(resident_bytes-n.bytes[index]))throw std::runtime_error("retained composition texture budget");
        if(n.output[index].Get()!=texture.Get())n.output_view[index].Reset();
        resident_bytes=resident_bytes-n.bytes[index]+bytes;n.bytes[index]=bytes;n.output[index]=std::move(texture);
    }
    Texture crop(ID3D11Texture2D* source,Rect r){
        if(empty(r))return {};D3D11_TEXTURE2D_DESC d={};source->GetDesc(&d);
        d.Width=unsigned(r.right-r.left);d.Height=unsigned(r.bottom-r.top);
        d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        d.Usage=D3D11_USAGE_DEFAULT;d.CPUAccessFlags=0;d.MiscFlags=0;
        Texture out;checked(device->CreateTexture2D(&d,nullptr,&out));
        D3D11_BOX b={unsigned(r.left),unsigned(r.top),0,unsigned(r.right),unsigned(r.bottom),1};
        context->CopySubresourceRegion(out.Get(),0,0,0,0,source,0,&b);return out;
    }
    void capture_output(Node& n,unsigned index,ID3D11Texture2D* source,Rect r){
        // This node owns a versioned recipe, not a published texture handle.
        // Re-evaluation changes that recipe's result in GPU order. Other native
        // versions have distinct nodes, so matching storage can stay allocated.
        D3D11_TEXTURE2D_DESC desc={};if(n.output[index])n.output[index]->GetDesc(&desc);
        if(desc.Width!=unsigned(r.right-r.left)||desc.Height!=unsigned(r.bottom-r.top)){
            // Reject optional history before allocating another texture. A
            // full budget must not transiently consume the game's remaining VA.
            auto bytes=std::uint64_t(r.right-r.left)*(r.bottom-r.top)*4;
            if(bytes>resident_budget-(resident_bytes-n.bytes[index]))throw std::runtime_error("retained composition texture budget");
            output(n,index,crop(source,r));return;
        }
        D3D11_BOX box={unsigned(r.left),unsigned(r.top),0,unsigned(r.right),unsigned(r.bottom),1};
        context->CopySubresourceRegion(n.output[index].Get(),0,0,0,0,source,0,&box);
        ++work.copies;work.copied_pixels+=std::uint64_t(r.right-r.left)*(r.bottom-r.top);
    }
    Picture read(Id id,Rect region){
        auto it=images.find(id);if(it==images.end())throw std::runtime_error("retained source missing");
        Picture out{it->second.width,it->second.height,it->second.format,{},it->second.partitioned};
        out.required=intersect(region,extent(it->second));
        for(auto const& patch:it->second.patches){auto r=intersect(region,patch.area);if(!empty(r))out.patches.push_back({r,patch.node,patch.output});}
        return out;
    }
    void replace(Picture& image,Rect region,std::vector<Patch> const& replacement){region=intersect(region,extent(image));if(empty(region))return;
        std::vector<Patch> next;
        for(auto const& p:image.patches){
            auto cut=intersect(region,p.area);if(empty(cut)){next.push_back(p);continue;}
            Rect pieces[4]={{p.area.left,p.area.top,p.area.right,cut.top},{p.area.left,cut.bottom,p.area.right,p.area.bottom},
                {p.area.left,cut.top,cut.left,cut.bottom},{cut.right,cut.top,p.area.right,cut.bottom}};
            for(auto r:pieces)if(!empty(r))next.push_back({r,p.node,p.output});
        }
        next.insert(next.end(),replacement.begin(),replacement.end());
        if(next.size()>8192)throw std::runtime_error("retained composition region budget");
        image.patches=std::move(next);image.version=++serial;
    }
    void write(Picture& image,Rect region,std::shared_ptr<Node> const& value,unsigned output){
        region=intersect(region,extent(image));if(!empty(region))replace(image,region,{{region,value,output}});
    }
    void collect(std::shared_ptr<Node> const& n,long long ticks,long long frequency,unsigned depth){
        if(n->sampled==frame)return;
        if(depth>256)throw std::runtime_error("retained composition dependency depth");
        // Collect authoritative direct samples before any map rendering or pose
        // joins. Revision callbacks may offer immutable CPU inputs to workers;
        // actual GPU execution remains in the original native command order.
        if(n->direct.revision)n->direct_revision=n->direct.revision(ticks,frequency);
        for(auto const& input:n->inputs)for(auto const& patch:input.patches)
            collect(patch.node,ticks,frequency,depth+1);
        n->sampled=frame;
    }
    void evaluate(std::shared_ptr<Node> const& n,long long ticks,long long frequency,unsigned depth){
        if(n->seen==frame)return;
        if(depth>256)throw std::runtime_error("retained composition dependency depth");
        if(n->sample){
            auto sampled=n->sample(ticks,frequency);
            if(sampled.kind==SampledImage::Kind::frozen){
                n->sample={};n->sample_target={};n->dynamic=n->map_dynamic=false;
            }else if(sampled.kind==SampledImage::Kind::bgra){
                auto r=sampled.area;unsigned w=unsigned(n->area.right-n->area.left),h=unsigned(n->area.bottom-n->area.top);
                if(r.left<0||r.top<0||r.right-r.left!=int(w)||r.bottom-r.top!=int(h))
                    throw std::runtime_error("retained sample extent changed");
                if(!n->sample_target.texture){
                    auto bytes=std::uint64_t(w)*h*4;
                    if(bytes>resident_budget-(resident_bytes-n->bytes[0]))throw std::runtime_error("retained composition texture budget");
                    auto canvas=replay.create(w,h,Format::bgra32,false);
                    if(!canvas)throw std::runtime_error("retained sample admission failed");
                    n->sample_target=replay.release_import_target(canvas);++sample_allocations;
                }
                if(!replay.import_bgra(n->sample_target,sampled.texture.Get(),r.left,r.top))
                    throw std::runtime_error("retained sample import failed");
                ++sample_imports;output(*n,0,n->sample_target.texture);n->revision=++serial;
            }else if(sampled.kind==SampledImage::Kind::immutable){
                if(!sampled.texture)throw std::runtime_error("retained visual selection retired");
                if(sampled.texture.Get()!=n->output[0].Get()){
                    n->sample_target={};
                    output(*n,0,std::move(sampled.texture));n->revision=++serial;
                }
            }
        }else if(n->operation){
            std::vector<std::uint64_t> versions;
            if(n->direct.revision)versions.push_back(n->direct_revision);
            n->dynamic=n->direct.animated;n->map_dynamic=false;
            for(auto const& p:n->inputs)for(auto const& patch:p.patches){
                evaluate(patch.node,ticks,frequency,depth+1);versions.push_back(patch.node->revision);
                n->dynamic|=patch.node->dynamic;n->map_dynamic|=patch.node->map_dynamic;
            }
            if(!n->output[0]||versions!=n->dependencies){
                ++work.operations;
                Id temporary[6]={};
                auto const& original=n->command;
                // Ordinary passes need only their affected underlay rectangle.
                // Cross-position self-copy keeps the full coordinate domain.
                bool tint=original.kind==Kind::native_blend&&original.color==2;
                bool local=tint||!(original.source&&(original.source==original.destination||original.source==original.background||
                    original.source==original.detail||original.source==original.background_detail));
                // These native operations write every pixel of their local
                // result. Their discarded before-image needs storage, not a
                // clear followed by an immediate overwrite of the same pixels.
                bool overwrite=local&&!n->direct.draw&&(original.kind==Kind::fill||original.kind==Kind::copy||
                    original.kind==Kind::quantize||(original.kind==Kind::expand&&original.color==65536)||
                    (original.kind==Kind::native_image&&original.color==65536));
                bool tight_source=local&&original.kind==Kind::native_image&&
                    original.source_width==original.area.right-original.area.left&&original.source_height==original.area.bottom-original.area.top;
                Rect source_area={original.source_x+n->area.left-original.area.left,original.source_y+n->area.top-original.area.top,
                    original.source_x+n->area.right-original.area.left,original.source_y+n->area.bottom-original.area.top};
                int x=local?n->area.left:0,y=local?n->area.top:0;
                try{
                    for(unsigned i=0;i<6;++i)if(n->original[i]){
                        for(unsigned prior=0;prior<i;++prior)if(n->original[prior]==n->original[i])temporary[i]=temporary[prior];
                        if(!temporary[i]){
                            bool target=i==0||i==3||(i==2&&original.kind!=Kind::native_text)||(i==4&&original.kind!=Kind::native_image);
                            bool writable=n->original[i]==n->original[0]||n->original[i]==n->original[3];
                            temporary[i]=assemble(n->inputs[i],ticks,frequency,depth+1,
                                tight_source&&(i==1||i==4)?source_area:local&&target?n->area:Rect{},!writable,
                                !(overwrite&&(i==0||i==3)));
                        }
                    }
                    auto c=n->command;c.destination=temporary[0];c.source=temporary[1];c.background=temporary[2];
                    c.detail=temporary[3];c.background_detail=temporary[4];c.program=temporary[5];
                    c.area={c.area.left-x,c.area.top-y,c.area.right-x,c.area.bottom-y};
                    c.clip={c.clip.left-x,c.clip.top-y,c.clip.right-x,c.clip.bottom-y};
                    Rect result={n->area.left-x,n->area.top-y,n->area.right-x,n->area.bottom-y};
                    if(tight_source){c.area=c.clip=result;c.source_x=c.source_y=0;
                        c.source_width=result.right-result.left;c.source_height=result.bottom-result.top;}
                    if(!(n->direct.draw?n->direct.draw(replay,c):replay.submit(&c,1)))throw std::runtime_error("retained operation rejected");
                    capture_output(*n,0,replay.texture(c.destination),result);
                    if(c.detail)capture_output(*n,1,replay.texture(c.detail),result);
                    n->dependencies=std::move(versions);n->revision=++serial;
                }catch(...){for(unsigned i=0;i<6;++i)if(temporary[i]&&std::find(temporary,temporary+i,temporary[i])==temporary+i)replay.recycle(temporary[i]);throw;}
                for(unsigned i=0;i<6;++i)if(temporary[i]&&std::find(temporary,temporary+i,temporary[i])==temporary+i)replay.recycle(temporary[i]);
            }
            // A frozen source can leave the output revision unchanged. Retire
            // its recipe even then, so HUD holes do not keep every old camera.
            if(!n->dynamic){for(auto& input:n->inputs)input={};n->dependencies.clear();n->operation=false;}
        }
        n->seen=frame;
    }
    Id assemble(Picture const& p,long long ticks,long long frequency,unsigned depth,Rect region={},bool readonly=false,bool initialize=true){
        // Evaluate children before reserving full-canvas scratch, so dependency
        // depth does not multiply the working-surface allocation.
        for(auto const& part:p.patches)evaluate(part.node,ticks,frequency,depth);
        if(empty(region))region=extent(p);region=intersect(region,extent(p));
        // A read records its actual footprint separately from the canvas's
        // coordinate domain. One source version may cover that complete read
        // even when only a glyph-sized region of a large atlas was captured.
        // Bind it directly; sparse holes and merged aliased reads still assemble.
        if(readonly&&p.partitioned&&p.patches.size()==1){auto const& part=p.patches[0];auto a=part.area,b=part.node->area;
            auto needed=empty(p.required)?region:intersect(region,p.required);
            if(!empty(needed)&&a.left<=needed.left&&a.top<=needed.top&&a.right>=needed.right&&a.bottom>=needed.bottom&&
               b.left==region.left&&b.top==region.top&&b.right==region.right&&b.bottom==region.bottom&&part.node->output[part.output]){
                auto& view=part.node->output_view[part.output];
                if(!view){checked(device->CreateShaderResourceView(part.node->output[part.output].Get(),nullptr,&view));++source_views;}
                auto id=replay.attach_source_unrecorded(part.node->output[part.output].Get(),p.format,view.Get());if(id)return id;
            }
        }
        // The oldest full-size underlay is often fragmented by hundreds of
        // later UI writes. Copy it once, then overlay their disjoint results.
        // A complete partition proves every overdrawn pixel is overwritten;
        // sparse canvases and aliased input unions keep exact rectangle copies.
        Patch const* base=nullptr;
        if(p.partitioned){
            std::uint64_t covered=0;bool complete=true;
            for(auto const& part:p.patches){auto r=intersect(part.area,region);if(empty(r))continue;
                if(!part.node->output[part.output])complete=false;
                if(!base)base=&part;covered+=std::uint64_t(r.right-r.left)*(r.bottom-r.top);}
            if(base){auto r=base->node->area;
                if(!complete||covered!=std::uint64_t(region.right-region.left)*(region.bottom-region.top)||
                   r.left>region.left||r.top>region.top||r.right<region.right||r.bottom<region.bottom||
                   !base->node->output[base->output])base=nullptr;
            }
        }
        Id out=replay.create(region.right-region.left,region.bottom-region.top,p.format,initialize&&!base);if(!out)throw std::runtime_error("retained composition scratch budget");
        ++work.assemblies;
        work.assembly_pixels+=std::uint64_t(region.right-region.left)*(region.bottom-region.top);
        try{
            if(base){auto r=base->node->area;
                D3D11_BOX box={unsigned(region.left-r.left),unsigned(region.top-r.top),0,
                    unsigned(region.right-r.left),unsigned(region.bottom-r.top),1};
                context->CopySubresourceRegion(replay.texture(out),0,0,0,0,base->node->output[base->output].Get(),0,&box);
                ++work.copies;work.copied_pixels+=std::uint64_t(region.right-region.left)*(region.bottom-region.top);
            }
            for(auto const& part:p.patches){
                if(base&&part.node==base->node&&part.output==base->output)continue;
                auto& source=part.node->output[part.output];if(!source)continue; // pristine zero canvas
                auto area=intersect(part.area,region);if(empty(area))continue;
                D3D11_BOX box={unsigned(area.left-part.node->area.left),unsigned(area.top-part.node->area.top),0,
                    unsigned(area.right-part.node->area.left),unsigned(area.bottom-part.node->area.top),1};
                context->CopySubresourceRegion(replay.texture(out),0,area.left-region.left,area.top-region.top,0,source.Get(),0,&box);
                ++work.copies;work.copied_pixels+=std::uint64_t(area.right-area.left)*(area.bottom-area.top);

            }
        }catch(...){replay.recycle(out);throw;}
        return out;
    }
public:
    RetainedComposition(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),replay(d,c,128u*1024u*1024u){}
    ~RetainedComposition(){front={};images.clear();}
    void clear(){front={};images.clear();replay.clear_recycled();admitted=true;}
    void discard(){front={};images.clear();replay.clear_recycled();admitted=false;}
    void uncommit(){front={};}
    std::uint64_t bytes()const{return resident_bytes;}
    Counts replay_stats()const{return replay.stats();}
    std::uint64_t sampling_allocations()const{return sample_allocations;}
    std::uint64_t sampling_imports()const{return sample_imports;}
    std::uint64_t source_view_creations()const{return source_views;}
    std::size_t node_count()const{return nodes;}
    bool accepting()const{return admitted;}
    std::size_t sampled_sources()const{
        std::vector<Node const*> visited;std::vector<Node const*> pending;
        for(auto const& patch:front.patches)pending.push_back(patch.node.get());
        std::size_t count=0;
        while(!pending.empty()){
            auto n=pending.back();pending.pop_back();
            if(std::find(visited.begin(),visited.end(),n)!=visited.end())continue;
            visited.push_back(n);if(n->sample)++count;
            for(auto const& input:n->inputs)for(auto const& patch:input.patches)pending.push_back(patch.node.get());
        }return count;
    }
    bool animated()const{for(auto const& p:front.patches)if(p.node->dynamic)return true;return false;}
    bool animated_map()const{for(auto const& p:front.patches)if(p.node->map_dynamic)return true;return false;}
    bool ready()const{return admitted&&front.width!=0;}
    void create(Id id,unsigned w,unsigned h,Format format){if(admitted){images[id]={w,h,format,{}};images[id].version=++serial;}}
    void destroy(Id id){images.erase(id);} // committed versions retain their own source data
    void source(Id id,ID3D11Texture2D* texture,Sample sample={},bool immutable=false,bool map_source=false){
        if(!admitted)return;auto& p=images.at(id);auto n=node();n->area=extent(p);n->revision=++serial;
        output(*n,0,(sample||immutable)?Texture(texture):crop(texture,n->area));n->dynamic=bool(sample);n->map_dynamic=map_source&&n->dynamic;n->sample=std::move(sample);p.patches={{n->area,n,0}};p.version=++serial;
    }
    void record(Command const& c,Direct direct={}){
        if(!admitted)return;auto target=images.find(c.destination);if(target==images.end())throw std::runtime_error("retained target missing");
        auto area=intersect(intersect(c.area,c.clip),extent(target->second));if(empty(area))return;
        // Equal-coordinate copies select an immutable source version. Keeping
        // its patches shares both samples and outputs instead of allocating a
        // full-canvas replay result for every map/screen/save transfer. Read
        // before replacing, including self-copies; later source writes cannot
        // alter this version. Shifted/converted/direct passes retain execution.
        if(c.kind==Kind::copy&&!c.detail&&!direct.draw&&!direct.revision&&!direct.animated&&!direct.input_bytes&&
           c.source_x==c.area.left&&c.source_y==c.area.top&&images.at(c.source).format==target->second.format){
            auto selected=read(c.source,area);replace(target->second,area,selected.patches);return;
        }
        // JGL's unscaled opaque transfer is the same version selection for
        // both its native words and full-color companion. Replaying it as a
        // stretch shader rebuilt entire animated map canvases every frame.
        if(c.kind==Kind::native_image&&c.color==65536&&!direct.draw&&!direct.revision&&!direct.animated&&!direct.input_bytes&&
           c.source_x==c.area.left&&c.source_y==c.area.top&&c.source_width==c.area.right-c.area.left&&
           c.source_height==c.area.bottom-c.area.top&&images.at(c.source).format==target->second.format&&
           (!c.detail||c.background_detail)){
            auto words=read(c.source,area);
            Picture detail;if(c.detail)detail=read(c.background_detail,area);
            replace(target->second,area,words.patches);
            if(c.detail)replace(images.at(c.detail),area,detail.patches);
            return;
        }
        // A native form commonly clears a full-window canvas to its color key,
        // then paints a few HUD rectangles. Proven key-only patches do not
        // affect either destination. Keep only the actual painted rectangles;
        // no texture readback or guess about uploaded source pixels is needed.
        if(c.kind==Kind::native_image&&c.color!=65536&&c.source!=c.destination&&
           c.source_width==c.area.right-c.area.left&&c.source_height==c.area.bottom-c.area.top&&
           !direct.draw&&!direct.revision&&!direct.animated&&!direct.input_bytes){
            int dx=c.area.left-c.source_x,dy=c.area.top-c.source_y;
            auto selected=read(c.source,{area.left-dx,area.top-dy,area.right-dx,area.bottom-dy});
            std::uint64_t covered=0;bool transparent=false,painted=false;Rect ink={};
            for(auto const& patch:selected.patches){covered+=std::uint64_t(patch.area.right-patch.area.left)*(patch.area.bottom-patch.area.top);
                bool keyed=patch.node->constant&&patch.node->command.color==c.color;transparent|=keyed;
                if(!keyed){if(!painted)ink=patch.area;else ink={std::min(ink.left,patch.area.left),std::min(ink.top,patch.area.top),
                    std::max(ink.right,patch.area.right),std::max(ink.bottom,patch.area.bottom)};painted=true;}}
            if(transparent&&selected.partitioned&&covered==std::uint64_t(area.right-area.left)*(area.bottom-area.top)){
                if(!painted)return;
                ink={ink.left+dx,ink.top+dy,ink.right+dx,ink.bottom+dy};
                // One bounded pass is cheaper than one pass per glyph/patch.
                // Keyed holes inside the painted envelope remain shader work.
                if(ink.left!=area.left||ink.top!=area.top||ink.right!=area.right||ink.bottom!=area.bottom){
                    auto bounded=c;bounded.clip=ink;record(bounded);return;
                }
            }
        }
        if(direct.input_bytes>resident_budget-resident_bytes)throw std::runtime_error("retained direct input budget");
        auto n=node();resident_bytes+=direct.input_bytes;n->operation=true;n->area=area;n->command=c;n->command.clip=area;n->direct=std::move(direct);n->dynamic=n->direct.animated;
        n->constant=c.kind==Kind::fill&&!n->direct.draw&&!n->direct.revision&&!n->direct.animated;
        Id ids[6]={c.destination,c.source,c.background,c.detail,c.background_detail,c.program};
        bool opaque=c.kind==Kind::fill||c.kind==Kind::copy||c.kind==Kind::quantize||
            (c.kind==Kind::expand&&c.color==65536)||(c.kind==Kind::native_image&&c.color==65536);
        for(unsigned i=0;i<6;++i)if(ids[i]){
            n->original[i]=ids[i];auto& p=images.at(ids[i]);Rect region=extent(p);
            if(i==0||i==3)region=opaque?Rect{0,0,0,0}:area;
            else if(i==1 && c.kind!=Kind::native_lookup){
                region={c.source_x+area.left-c.area.left,c.source_y+area.top-c.area.top,
                    c.source_x+area.right-c.area.left,c.source_y+area.bottom-c.area.top};
                if(c.kind==Kind::native_image&&(c.source_width!=c.area.right-c.area.left||c.source_height!=c.area.bottom-c.area.top))
                    region={c.source_x,c.source_y,c.source_x+c.source_width,c.source_y+c.source_height};
                if(c.kind==Kind::native_blend&&c.color==2)region=area;
            }else if(i==4&&c.kind==Kind::native_image&&c.source_width==c.area.right-c.area.left&&c.source_height==c.area.bottom-c.area.top){
                region={c.source_x+area.left-c.area.left,c.source_y+area.top-c.area.top,c.source_x+area.right-c.area.left,c.source_y+area.bottom-c.area.top};
            }else if((i==2&&c.kind!=Kind::native_text)||(i==4&&c.kind!=Kind::native_image))region=area;
            n->inputs[i]=read(ids[i],region);
        }
        // Aliased operands must refer to one assembled before-image. Merge the
        // needed regions; duplicate patches are identical and harmless copies.
        for(unsigned i=0;i<6;++i)if(ids[i])for(unsigned j=i+1;j<6;++j)if(ids[j]==ids[i]){
            n->inputs[i].patches.insert(n->inputs[i].patches.end(),n->inputs[j].patches.begin(),n->inputs[j].patches.end());n->inputs[i].partitioned=false;n->inputs[j]=n->inputs[i];
        }
        for(auto const& input:n->inputs)for(auto const& p:input.patches){n->dynamic|=p.node->dynamic;n->map_dynamic|=p.node->map_dynamic;}
        write(images.at(c.destination),area,n,0);if(c.detail)write(images.at(c.detail),area,n,1);
    }
    void commit(Id image,Rect area){
        if(!admitted)return;auto const& p=images.at(image);area=intersect(area,extent(p));if(empty(area))return;
        if(area.left==0&&area.top==0&&area.right==int(p.width)&&area.bottom==int(p.height)){
            // The native screen may be offered again without any intervening
            // write. Preserve its completed visual revision so the autonomous
            // presenter can skip an identical full-screen GPU composition.
            bool same=front.version==p.version&&front.width==p.width&&front.height==p.height&&front.format==p.format;
            if(!same){front=p;++front_revision;}
            return;
        }
        ++front_revision;
        if(front.width!=p.width||front.height!=p.height||front.format!=p.format){
            if(area.left||area.top||area.right!=int(p.width)||area.bottom!=int(p.height)){front={};return;}
            front=p;return;
        }
        // A native partial transfer changes exactly its rectangle. Preserve
        // displayed pixels outside it, including versions no longer in p.
        auto selected=read(image,area);
        auto zero=node();zero->area=area;write(front,area,zero,0);
        for(auto const& part:selected.patches)write(front,part.area,part.node,part.output);
    }
    Texture sample(long long ticks,long long frequency){
        work={};
        if(!ready())return {};++frame;
        for(auto const& part:front.patches)collect(part.node,ticks,frequency,0);
        auto image=assemble(front,ticks,frequency,0,{},true);
        Texture result;
        try{result=crop(replay.texture(image),extent(front));}
        catch(...){replay.recycle(image);throw;}replay.recycle(image);return result;
    }
    // Caller has supplied a completed native transfer. Rendering only touches
    // private scratch and the existing presenter's retained display.
    int draw(long long ticks,long long frequency,ID3D11RenderTargetView* target,ID3D11Texture2D* display,ID3D11Texture2D* buffer){
        work={};
        if(!ready())return 0;++frame;
        for(auto const& part:front.patches)collect(part.node,ticks,frequency,0);
        std::vector<std::uint64_t> versions;
        for(auto const& part:front.patches){evaluate(part.node,ticks,frequency,0);versions.push_back(part.node->revision);}
        if(drawn_revision==front_revision&&versions==drawn_dependencies)return 2; // no new source sample
        auto image=assemble(front,ticks,frequency,0,{},true);
        bool ok=false;
        try{ok=replay.display(image,target,front.width,front.height,extent(front));}
        catch(...){replay.recycle(image);throw;}replay.recycle(image);
        if(ok){context->CopyResource(buffer,display);context->Flush();drawn_revision=front_revision;drawn_dependencies=std::move(versions);}return ok?1:0;
    }
    Work last_work()const{return work;}
    template<class Report> void describe(Report report)const{
        std::vector<Node const*> ordered;
        auto add=[&](Node const* n){if(std::find(ordered.begin(),ordered.end(),n)==ordered.end()&&ordered.size()<256)ordered.push_back(n);};
        for(auto const& p:front.patches)add(p.node.get());
        for(unsigned i=0;i<ordered.size();++i)for(auto const& input:ordered[i]->inputs)for(auto const& p:input.patches)add(p.node.get());
        for(unsigned i=0;i<ordered.size();++i){auto n=ordered[i];auto const& c=n->command;char text[256];
            std::snprintf(text,sizeof(text),"id=%u kind=%d operation=%u dynamic=%u sampled=%u area=%d,%d,%d,%d source=%d,%d,%d,%d color=%u",
                i,int(c.kind),unsigned(n->operation),unsigned(n->dynamic),unsigned(bool(n->sample)),
                n->area.left,n->area.top,n->area.right,n->area.bottom,c.source_x,c.source_y,c.source_width,c.source_height,c.color);
            std::string line=text;
            for(unsigned input=0;input<6;++input)if(!n->inputs[input].patches.empty()){
                std::snprintf(text,sizeof(text)," input%u=",input);line+=text;
                std::vector<unsigned> unique;
                for(auto const& p:n->inputs[input].patches){auto id=unsigned(std::find(ordered.begin(),ordered.end(),p.node.get())-ordered.begin());
                    if(std::find(unique.begin(),unique.end(),id)!=unique.end())continue;unique.push_back(id);
                    std::snprintf(text,sizeof(text),"%u.%u,",id,p.output);line+=text;}
            }
            report(line.c_str());
        }
    }
};
}
