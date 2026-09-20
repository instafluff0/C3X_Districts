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
    using Texture=ComPtr<ID3D11Texture2D>;
    using Sample=std::function<Texture(long long,long long)>;
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
    struct Picture {unsigned width=0,height=0;Format format=Format::bgra32;std::vector<Patch> patches;};
    struct Node {
        Rect area{};Command command{};bool operation=false,dynamic=false;
        Picture inputs[6];Id original[6]={};
        Texture output[2];std::uint64_t bytes[2]={},revision=0,seen=0;
        std::vector<std::uint64_t> dependencies;
        Sample sample;Direct direct;
    };
    ID3D11Device* device;ID3D11DeviceContext* context;
    Compositor replay;
    std::map<Id,Picture> images;
    Picture front;
    std::uint64_t serial=0,frame=0,front_revision=0,drawn_revision=0;
    std::vector<std::uint64_t> drawn_dependencies;
    bool admitted=true;
    std::size_t nodes=0;std::uint64_t resident_bytes=0;
    constexpr static std::uint64_t resident_budget=128u*1024u*1024u;
    Rect intersect(Rect a,Rect b)const{return {std::max(a.left,b.left),std::max(a.top,b.top),std::min(a.right,b.right),std::min(a.bottom,b.bottom)};}
    bool empty(Rect r)const{return r.left>=r.right||r.top>=r.bottom;}
    Rect extent(Picture const& p)const{return {0,0,int(p.width),int(p.height)};}
    std::shared_ptr<Node> node(){
        if(nodes>=32768)throw std::runtime_error("retained composition node budget");
        ++nodes;return std::shared_ptr<Node>(new Node,[this](Node* p){resident_bytes-=p->bytes[0]+p->bytes[1]+p->direct.input_bytes;delete p;--nodes;});
    }
    void output(Node& n,unsigned index,Texture texture){
        D3D11_TEXTURE2D_DESC d={};if(texture)texture->GetDesc(&d);
        auto bytes=std::uint64_t(d.Width)*d.Height*4;
        if(bytes>resident_budget-(resident_bytes-n.bytes[index]))throw std::runtime_error("retained composition texture budget");
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
    Picture read(Id id,Rect region){
        auto it=images.find(id);if(it==images.end())throw std::runtime_error("retained source missing");
        Picture out{it->second.width,it->second.height,it->second.format,{}};
        for(auto const& patch:it->second.patches){auto r=intersect(region,patch.area);if(!empty(r))out.patches.push_back({r,patch.node,patch.output});}
        return out;
    }
    void write(Picture& image,Rect region,std::shared_ptr<Node> const& value,unsigned output){region=intersect(region,extent(image));if(empty(region))return;
        std::vector<Patch> next;
        for(auto const& p:image.patches){
            auto cut=intersect(region,p.area);if(empty(cut)){next.push_back(p);continue;}
            Rect pieces[4]={{p.area.left,p.area.top,p.area.right,cut.top},{p.area.left,cut.bottom,p.area.right,p.area.bottom},
                {p.area.left,cut.top,cut.left,cut.bottom},{cut.right,cut.top,p.area.right,cut.bottom}};
            for(auto r:pieces)if(!empty(r))next.push_back({r,p.node,p.output});
        }
        next.push_back({region,value,output});
        if(next.size()>8192)throw std::runtime_error("retained composition region budget");
        image.patches=std::move(next);
    }
    void evaluate(std::shared_ptr<Node> const& n,long long ticks,long long frequency,unsigned depth){
        if(n->seen==frame)return;
        if(depth>256)throw std::runtime_error("retained composition dependency depth");
        if(n->sample){
            auto sampled=n->sample(ticks,frequency);
            if(!sampled)throw std::runtime_error("retained visual selection retired");
            if(sampled.Get()!=n->output[0].Get()){output(*n,0,std::move(sampled));n->revision=++serial;}
        }else if(n->operation){
            std::vector<std::uint64_t> versions;
            if(n->direct.revision)versions.push_back(n->direct.revision(ticks,frequency));
            for(auto const& p:n->inputs)for(auto const& patch:p.patches){evaluate(patch.node,ticks,frequency,depth+1);versions.push_back(patch.node->revision);}
            if(!n->output[0]||versions!=n->dependencies){
                Id temporary[6]={};
                auto const& original=n->command;
                // Ordinary passes need only their affected underlay rectangle.
                // Cross-position self-copy keeps the full coordinate domain.
                bool tint=original.kind==Kind::native_blend&&original.color==2;
                bool local=tint||!(original.source&&(original.source==original.destination||original.source==original.background||
                    original.source==original.detail||original.source==original.background_detail));
                int x=local?n->area.left:0,y=local?n->area.top:0;
                try{
                    for(unsigned i=0;i<6;++i)if(n->original[i]){
                        for(unsigned prior=0;prior<i;++prior)if(n->original[prior]==n->original[i])temporary[i]=temporary[prior];
                        if(!temporary[i]){
                            bool target=i==0||i==3||(i==2&&original.kind!=Kind::native_text)||(i==4&&original.kind!=Kind::native_image);
                            bool writable=n->original[i]==n->original[0]||n->original[i]==n->original[3];
                            temporary[i]=assemble(n->inputs[i],ticks,frequency,depth+1,local&&target?n->area:Rect{},!writable);
                        }
                    }
                    auto c=n->command;c.destination=temporary[0];c.source=temporary[1];c.background=temporary[2];
                    c.detail=temporary[3];c.background_detail=temporary[4];c.program=temporary[5];
                    c.area={c.area.left-x,c.area.top-y,c.area.right-x,c.area.bottom-y};
                    c.clip={c.clip.left-x,c.clip.top-y,c.clip.right-x,c.clip.bottom-y};
                    Rect result={n->area.left-x,n->area.top-y,n->area.right-x,n->area.bottom-y};
                    if(!(n->direct.draw?n->direct.draw(replay,c):replay.submit(&c,1)))throw std::runtime_error("retained operation rejected");
                    output(*n,0,crop(replay.texture(c.destination),result));
                    if(c.detail)output(*n,1,crop(replay.texture(c.detail),result));
                    n->dependencies=std::move(versions);n->revision=++serial;
                    if(!n->dynamic){for(auto& input:n->inputs)input={};n->dependencies.clear();n->operation=false;}
                }catch(...){for(unsigned i=0;i<6;++i)if(temporary[i]&&std::find(temporary,temporary+i,temporary[i])==temporary+i)replay.destroy(temporary[i]);throw;}
                for(unsigned i=0;i<6;++i)if(temporary[i]&&std::find(temporary,temporary+i,temporary[i])==temporary+i)replay.destroy(temporary[i]);
            }
        }
        n->seen=frame;
    }
    Id assemble(Picture const& p,long long ticks,long long frequency,unsigned depth,Rect region={},bool readonly=false){
        // Evaluate children before reserving full-canvas scratch, so dependency
        // depth does not multiply the working-surface allocation.
        for(auto const& part:p.patches)evaluate(part.node,ticks,frequency,depth);
        if(empty(region))region=extent(p);region=intersect(region,extent(p));
        // Shared immutable sources bind directly. No viewport reconstruction is
        // needed to sample a map, unit pose, sprite, or lookup table texture.
        if(readonly&&p.patches.size()==1){auto const& part=p.patches[0];auto a=part.area,b=part.node->area;
            if(a.left==0&&a.top==0&&a.right==int(p.width)&&a.bottom==int(p.height)&&a.left==b.left&&a.top==b.top&&a.right==b.right&&a.bottom==b.bottom&&
               region.left==0&&region.top==0&&region.right==int(p.width)&&region.bottom==int(p.height)&&part.node->output[part.output]){
                auto id=replay.attach_source(part.node->output[part.output].Get(),p.format);if(id)return id;
            }
        }
        Id out=replay.create(region.right-region.left,region.bottom-region.top,p.format);if(!out)throw std::runtime_error("retained composition scratch budget");
        try{
            for(auto const& part:p.patches){
                auto& source=part.node->output[part.output];if(!source)continue; // pristine zero canvas
                auto area=intersect(part.area,region);if(empty(area))continue;
                D3D11_BOX box={unsigned(area.left-part.node->area.left),unsigned(area.top-part.node->area.top),0,
                    unsigned(area.right-part.node->area.left),unsigned(area.bottom-part.node->area.top),1};
                context->CopySubresourceRegion(replay.texture(out),0,area.left-region.left,area.top-region.top,0,source.Get(),0,&box);

            }
        }catch(...){replay.destroy(out);throw;}
        return out;
    }
public:
    RetainedComposition(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),replay(d,c,128u*1024u*1024u){}
    ~RetainedComposition(){front={};images.clear();}
    void clear(){front={};images.clear();admitted=true;}
    void discard(){front={};images.clear();admitted=false;}
    void uncommit(){front={};}
    std::uint64_t bytes()const{return resident_bytes;}
    std::size_t node_count()const{return nodes;}
    bool accepting()const{return admitted;}
    bool animated()const{for(auto const& p:front.patches)if(p.node->dynamic)return true;return false;}
    bool ready()const{return admitted&&front.width!=0;}
    void create(Id id,unsigned w,unsigned h,Format format){if(admitted)images[id]={w,h,format,{}};}
    void destroy(Id id){images.erase(id);} // committed versions retain their own source data
    void source(Id id,ID3D11Texture2D* texture,Sample sample={},bool immutable=false){
        if(!admitted)return;auto& p=images.at(id);auto n=node();n->area=extent(p);n->revision=++serial;
        output(*n,0,(sample||immutable)?Texture(texture):crop(texture,n->area));n->dynamic=bool(sample);n->sample=std::move(sample);p.patches={{n->area,n,0}};
    }
    void record(Command const& c,Direct direct={}){
        if(!admitted)return;auto target=images.find(c.destination);if(target==images.end())throw std::runtime_error("retained target missing");
        auto area=intersect(intersect(c.area,c.clip),extent(target->second));if(empty(area))return;
        if(direct.input_bytes>resident_budget-resident_bytes)throw std::runtime_error("retained direct input budget");
        auto n=node();resident_bytes+=direct.input_bytes;n->operation=true;n->area=area;n->command=c;n->command.clip=area;n->direct=std::move(direct);n->dynamic=n->direct.animated;
        Id ids[6]={c.destination,c.source,c.background,c.detail,c.background_detail,c.program};
        bool opaque=c.kind==Kind::fill||c.kind==Kind::copy||c.kind==Kind::quantize||
            (c.kind==Kind::expand&&c.color==65536)||(c.kind==Kind::native_image&&c.color==65536);
        for(unsigned i=0;i<6;++i)if(ids[i]){
            n->original[i]=ids[i];auto& p=images.at(ids[i]);Rect region=extent(p);
            if(i==0||i==3)region=opaque?Rect{0,0,0,0}:area;
            else if(i==1 && c.kind!=Kind::native_lookup){
                region={c.source_x+area.left-c.area.left,c.source_y+area.top-c.area.top,
                    c.source_x+area.right-c.area.left,c.source_y+area.bottom-c.area.top};
                if(c.kind==Kind::native_image)region={c.source_x,c.source_y,c.source_x+c.source_width,c.source_y+c.source_height};
                if(c.kind==Kind::native_blend&&c.color==2)region=area;
            }else if((i==2&&c.kind!=Kind::native_text)||(i==4&&c.kind!=Kind::native_image))region=area;
            n->inputs[i]=read(ids[i],region);
        }
        // Aliased operands must refer to one assembled before-image. Merge the
        // needed regions; duplicate patches are identical and harmless copies.
        for(unsigned i=0;i<6;++i)if(ids[i])for(unsigned j=i+1;j<6;++j)if(ids[j]==ids[i]){
            n->inputs[i].patches.insert(n->inputs[i].patches.end(),n->inputs[j].patches.begin(),n->inputs[j].patches.end());n->inputs[j]=n->inputs[i];
        }
        for(auto const& input:n->inputs)for(auto const& p:input.patches)n->dynamic|=p.node->dynamic;
        write(images.at(c.destination),area,n,0);if(c.detail)write(images.at(c.detail),area,n,1);
    }
    void commit(Id image,Rect area){
        if(!admitted)return;auto const& p=images.at(image);area=intersect(area,extent(p));if(empty(area))return;++front_revision;
        if(area.left==0&&area.top==0&&area.right==int(p.width)&&area.bottom==int(p.height)){front=p;return;}
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
    Texture snapshot_bgra(ID3D11Texture2D* source,int x,int y,unsigned w,unsigned h){
        auto image=replay.create(w,h,Format::bgra32);if(!image)throw std::runtime_error("retained map import budget");
        Texture result;
        try{if(!replay.import_bgra(image,source,x,y))throw std::runtime_error("retained map import failed");result=crop(replay.texture(image),{0,0,int(w),int(h)});}
        catch(...){replay.destroy(image);throw;}replay.destroy(image);return result;
    }
    Texture sample(long long ticks,long long frequency){
        if(!ready())return {};++frame;auto image=assemble(front,ticks,frequency,0);
        auto result=crop(replay.texture(image),extent(front));replay.destroy(image);return result;
    }
    // Caller has supplied a completed native transfer. Rendering only touches
    // private scratch and the existing presenter's retained display.
    int draw(long long ticks,long long frequency,ID3D11RenderTargetView* target,ID3D11Texture2D* display,ID3D11Texture2D* buffer){
        if(!ready())return 0;++frame;
        std::vector<std::uint64_t> versions;
        for(auto const& part:front.patches){evaluate(part.node,ticks,frequency,0);versions.push_back(part.node->revision);}
        if(drawn_revision==front_revision&&versions==drawn_dependencies)return 2; // no new source sample
        auto image=assemble(front,ticks,frequency,0);
        bool ok=replay.display(image,target,front.width,front.height,extent(front));replay.destroy(image);
        if(ok){context->CopyResource(buffer,display);context->Flush();drawn_revision=front_revision;drawn_dependencies=std::move(versions);}return ok?1:0;
    }
};
}
