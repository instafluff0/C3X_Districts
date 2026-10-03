#pragma once
#include "gpu_image_compositor.h"
#include "zoom_transition.h"
#include "scene_projection.h"
#include "gpu_projected_layer.h"
#include <array>
#include <map>
#include <memory>
#include <functional>
#include <unordered_set>

namespace c3x_gpu_images {
// Versioned rectangular writes preserve the native composition order. A copy
// captures its source version, not a mutable image handle. Opaque writes cut
// away covered history; transparent writes retain only the affected underlay.
// All execution uses the production packed/full-color compositor.
class RetainedComposition {
public:
    struct Work {unsigned operations=0,assemblies=0,copies=0,selected_borrows=0,selected_owned=0,direct_native_images=0;
        std::uint64_t copied_pixels=0,assembly_pixels=0,avoided_copy_pixels=0;};
    struct RecipeReuse {std::uint64_t eligible=0,probed=0,reused=0;};
    struct PlanReuse {std::uint64_t builds=0,reuses=0,source_binds=0,source_reuses=0,batch_builds=0,batch_reuses=0;std::size_t nodes=0;};
    using Texture=ComPtr<ID3D11Texture2D>;
    struct Placed {Command command;int x=0,y=0;};
    struct SampledImage {
        enum class Kind { unchanged, immutable, bgra, frozen, held };
        Kind kind=Kind::unchanged;Texture texture;Rect area{};float sharpness=0.f;
        std::uint64_t generation=0;
        SampledImage()=default;
        // The source owner has moved to another camera. Preserve this exact
        // completed image, but retire its animation callback and dependencies.
        static SampledImage frozen(){SampledImage result;result.kind=Kind::frozen;return result;}
        // Preparation is still pending. Keep completed pixels and the live
        // dependency, allowing native UI and future preparation to progress.
        static SampledImage held(){SampledImage result;result.kind=Kind::held;return result;}
        SampledImage(Texture value):kind(Kind::immutable),texture(std::move(value)){}
        // Borrow a working BGRA surface only until this sample is consumed.
        // The retained node imports it into its own reusable packed output.
        static SampledImage bgra(ID3D11Texture2D* source,Rect region,float sharpness=0.f,std::uint64_t generation=0){
            SampledImage result;result.kind=Kind::bgra;result.texture=source;result.area=region;result.sharpness=sharpness;result.generation=generation;return result;
        }
    };
    struct Sample {
        std::function<SampledImage(long long,long long)> canonical;
        std::function<SampledImage(long long,long long,float)> projected;
        std::function<void(long long,long long,float)> prepare;
        std::uint64_t source_generation=0;
        Sample()=default;
        template<class F,typename std::enable_if<!std::is_same<typename std::decay<F>::type,Sample>::value,int>::type=0>
        Sample(F&& function):canonical(std::forward<F>(function)){}
        explicit operator bool()const{return bool(canonical);}
        SampledImage operator()(long long ticks,long long frequency)const{return canonical(ticks,frequency);}
    };
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
    struct BatchOp {Placed placed;Picture inputs[6];};
    struct BatchSource {Picture picture;Texture texture;CompositionStorage::Lease owned,physical;std::vector<std::uint64_t> revisions,pending;};
    struct BatchPreparation {
        // This exact HUD program owns only external immutable operands. Each
        // native publication keeps its own before-images and output pair.
        std::vector<BatchOp> batch;
        Id original[2]={};unsigned width[2]={},height[2]={};Format formats[2]={};Rect area{};
        std::weak_ptr<c3x_renderer::ZoomTransition> placement;
        // Operand versions are immutable. Bind each distinct picture once,
        // rather than assembling the same glyph/table for every HUD command.
        std::vector<BatchSource> batch_sources;
        std::vector<Compositor::SpatialSource> batch_views;
        std::vector<std::array<unsigned,6>> batch_operands;
        std::vector<Command> batch_commands;
        std::vector<std::uint64_t> batch_offsets;
        Compositor::SpatialPlan spatial_plan;
        std::uint64_t spatial_allowance=0,binding_allowance=0;
        bool batch_compiled=false,batch_bound=false,binding_refused=false,spatial_attempted=false,spatial_ready=false;
    };
    struct Node {
        Rect area{},recipe_clip{};Command command{};bool selected_world=false,operation=false,dynamic=false,map_dynamic=false,constant=false,view_dependent=false,retired=false;
        Picture inputs[6];Id original[6]={};
        std::vector<BatchOp> batch;
        std::shared_ptr<BatchPreparation> batch_preparation;
        std::vector<std::uint64_t> pending_dependencies;
        CompositionStorage::Lease storage[2],owned_storage[2];
        Texture output[2];bool borrowed_output[2]={};Format selected_format[2]={};std::uint64_t bytes[2]={},revision=0,seen=0,sampled=0,direct_revision=0;
        std::uint64_t publication=0,source_generation=0;bool map_source=false;
        ComPtr<ID3D11ShaderResourceView> output_view[2];
        std::vector<std::uint64_t> dependencies;
        Sample sample;Direct direct;Compositor::ImportTarget sample_target;
        std::shared_ptr<c3x_renderer::ZoomTransition> view;
        std::shared_ptr<c3x_renderer::ZoomTransition> placement;
        int anchor_x=0,anchor_y=0;
        Compositor::ImportTarget view_words;
        unsigned view_native_format=0;
        float view_scale=0.f;
        std::shared_ptr<Node> projected;
        std::uint64_t projected_frame=0,prepared_frame=0,prepared_projected_frame=0;
        bool projects_scene=false;
    };
    ID3D11Device* device;ID3D11DeviceContext* context;
    Compositor replay;
    std::map<Id,Picture> images;
    Picture front;
    std::shared_ptr<Node> world_selection;
    std::uint64_t serial=0,frame=0,front_revision=0,drawn_revision=0;
    std::vector<std::uint64_t> drawn_dependencies;
    std::vector<std::uint64_t> pending_drawn_dependencies;
    // One optional assembled front, never a native version or recipe input.
    // Weak patch identities cannot keep a retired camera alive.
    struct FrontPatch {Rect area{};std::weak_ptr<Node> node;unsigned output=0;std::uint64_t revision=0;};
    Texture assembled_front;CompositionStorage::Lease front_owned,front_physical;
    unsigned assembled_width=0,assembled_height=0;Format assembled_format=Format::bgra32;
    std::uint64_t assembled_revision=0;std::vector<FrontPatch> assembled_patches;
    struct PlanNode {std::weak_ptr<Node> node;bool project=false;};
    struct PrepareNode {std::weak_ptr<Node> node;bool project=false;std::weak_ptr<c3x_renderer::ZoomTransition> scale;};
    std::vector<PlanNode> collect_plan;
    std::vector<PrepareNode> prepare_plan;
    std::uint64_t topology_revision=0,planned_front=~std::uint64_t(0),planned_topology=~std::uint64_t(0);
    PlanReuse plan_counts;bool compiled_enabled=true;
    // Reuse the latest live exact HUD program across native world-end nodes.
    // A weak index neither pins an old base nor adds saved recipe history.
    std::weak_ptr<BatchPreparation> recent_batch;
    unsigned batch_inventories=0;
    bool admitted=true;
    std::size_t nodes=0;std::uint64_t direct_bytes=0;
    CompositionStorage storage,owned_storage;
    std::uint64_t sample_allocations=0,sample_imports=0,source_views=0;
    Work work;double selected_view_scale=1.;
    ProjectedLayer projected_layer;
    // Index only the last eight eligible live recipes. Weak entries cannot
    // extend an image/version lifetime, and proof traversal has a fixed cap.
    static constexpr unsigned recipe_limit=8,recipe_patch_limit=256;
    std::array<std::weak_ptr<Node>,recipe_limit> recent_recipes{};
    unsigned recipe_cursor=0;
    RecipeReuse recipe_counts;
    static bool same_rect(Rect a,Rect b){return a.left==b.left&&a.top==b.top&&a.right==b.right&&a.bottom==b.bottom;}
    static bool same_picture(Picture const& a,Picture const& b,bool require_version=true){
        if(a.width!=b.width||a.height!=b.height||a.format!=b.format||a.partitioned!=b.partitioned||
           (require_version&&a.version!=b.version)||!same_rect(a.required,b.required)||a.patches.size()!=b.patches.size())return false;
        for(std::size_t i=0;i<a.patches.size();++i){auto const& x=a.patches[i];auto const& y=b.patches[i];
            if(x.node!=y.node||x.output!=y.output||!same_rect(x.area,y.area))return false;
        }return true;
    }
    bool eligible_recipe(Node const& n)const{
        auto const& c=n.command;
        if(!n.operation||n.retired||n.direct.draw||n.direct.revision||n.direct.input_bytes||n.direct.animated||n.placement||
           !same_rect(n.area,extent(n.inputs[0]))||!same_rect(c.area,n.area)||c.source_x||c.source_y||
           n.inputs[1].width!=n.inputs[0].width||n.inputs[1].height!=n.inputs[0].height)return false;
        bool keyed=c.kind==Kind::native_image&&c.color<=65535&&
            c.source_width==int(n.inputs[0].width)&&c.source_height==int(n.inputs[0].height);
        bool expanded=c.kind==Kind::expand&&c.color==65536&&n.inputs[0].format==Format::bgra32&&n.inputs[1].format!=Format::bgra32;
        if(!keyed&&!expanded)return false;
        std::size_t patches=0;for(auto const& input:n.inputs){patches+=input.patches.size();if(patches>recipe_patch_limit)return false;}
        return true;
    }
    static bool same_recipe(Node const& a,Node const& b){
        auto const& x=a.command;auto const& y=b.command;
        if(x.kind!=y.kind||!same_rect(x.area,y.area)||!same_rect(x.clip,y.clip)||!same_rect(a.area,b.area)||
           !same_rect(a.recipe_clip,b.recipe_clip)||x.source_x!=y.source_x||x.source_y!=y.source_y||x.color!=y.color||
           x.source_width!=y.source_width||x.source_height!=y.source_height)return false;
        for(unsigned i=0;i<6;++i){
            if(bool(a.original[i])!=bool(b.original[i])||!same_picture(a.inputs[i],b.inputs[i]))return false;
            for(unsigned j=0;j<6;++j)if((a.original[i]==a.original[j])!=(b.original[i]==b.original[j]))return false;
        }return true;
    }
    std::shared_ptr<Node> reuse_recipe(std::shared_ptr<Node> const& next){
        if(!eligible_recipe(*next))return next;++recipe_counts.eligible;
        auto selected=next;
        for(unsigned i=0;i<recipe_limit;++i){unsigned slot=(recipe_cursor+recipe_limit-1-i)%recipe_limit;
            auto old=recent_recipes[slot].lock();if(!old||!eligible_recipe(*old))continue;++recipe_counts.probed;
            if(same_recipe(*old,*next)){selected=std::move(old);recent_recipes[slot].reset();++recipe_counts.reused;break;}
        }
        recent_recipes[recipe_cursor]=selected;recipe_cursor=(recipe_cursor+1)%recipe_limit;return selected;
    }
    // Match the bounded live native family, including saved packed/full-color
    // versions and old/new immutable map overlap at fullscreen.
    constexpr static std::uint64_t resident_budget=256u*1024u*1024u;
    Rect intersect(Rect a,Rect b)const{return {std::max(a.left,b.left),std::max(a.top,b.top),std::min(a.right,b.right),std::min(a.bottom,b.bottom)};}
    bool empty(Rect r)const{return r.left>=r.right||r.top>=r.bottom;}
    Rect extent(Picture const& p)const{return {0,0,int(p.width),int(p.height)};}
    bool directly_bindable(Picture const& p)const{
        if(!p.partitioned||p.patches.size()!=1)return false;
        auto const& patch=p.patches.front();auto needed=empty(p.required)?extent(p):intersect(extent(p),p.required);
        return !empty(needed)&&patch.area.left<=needed.left&&patch.area.top<=needed.top&&patch.area.right>=needed.right&&patch.area.bottom>=needed.bottom&&
            same_rect(patch.node->area,extent(p))&&bool(patch.node->output[patch.output]);
    }
    std::shared_ptr<Node> node(){
        if(nodes>=32768)throw std::runtime_error("retained composition node budget");
        auto value=new Node;++nodes;return std::shared_ptr<Node>(value,[this](Node* p){
            direct_bytes-=p->direct.input_bytes;delete p;--nodes;});
    }
    std::uint64_t resident_bytes()const{return owned_storage.bytes()+direct_bytes+replay.spatial_bytes();}
    void release_front(){assembled_patches.clear();assembled_front.Reset();front_owned={};front_physical={};assembled_width=assembled_height=0;assembled_revision=0;}
    void reserve(std::uint64_t bytes,char const* site){
        if(bytes<=resident_budget-resident_bytes())return;
        // Optional front reuse cannot displace an authoritative native output.
        release_front();
        if(bytes<=resident_budget-resident_bytes())return;
        char line[384];std::snprintf(line,sizeof(line),
            "[C3X renderer] stage=retained-admission-rejected site=%s requested=%llu resident=%llu cap=%llu physical=%llu peak=%llu recipe_eligible=%llu recipe_probed=%llu recipe_reused=%llu\n",
            site,bytes,resident_bytes(),resident_budget,storage.bytes(),storage.peak(),recipe_counts.eligible,recipe_counts.probed,recipe_counts.reused);OutputDebugStringA(line);
        throw std::runtime_error("retained composition texture budget");
    }
    void output(Node& n,unsigned index,Texture texture){
        D3D11_TEXTURE2D_DESC d={};if(texture)texture->GetDesc(&d);
        auto bytes=std::uint64_t(d.Width)*d.Height*4;
        if(texture&&!owned_storage.contains(texture.Get()))reserve(bytes,"output");
        // Acquire before dropping the old allocation: peak includes overlap.
        auto owned=owned_storage.retain(texture.Get()),physical=storage.retain(texture.Get());
        if(n.output[index].Get()!=texture.Get())n.output_view[index].Reset();
        n.bytes[index]=bytes;n.output[index]=std::move(texture);
        n.owned_storage[index]=std::move(owned);n.storage[index]=std::move(physical);
        n.borrowed_output[index]=false;
    }
    bool exact_plane(Picture const& p,Rect area)const{
        if(!p.partitioned||p.patches.size()!=1||!same_rect(extent(p),area))return false;
        auto const& patch=p.patches.front();
        if(!same_rect(patch.area,area)||!same_rect(patch.node->area,area)||!patch.node->output[patch.output])return false;
        D3D11_TEXTURE2D_DESC desc={};patch.node->output[patch.output]->GetDesc(&desc);
        return desc.Format==DXGI_FORMAT_R32_UINT&&desc.Width==unsigned(area.right-area.left)&&desc.Height==unsigned(area.bottom-area.top);
    }
    void admit_owned_outputs(Node& n,unsigned count){
        // Admission and allocation of both planes precede either write. A
        // borrowed plane is never a target, even if its dimensions match.
        auto bytes=std::uint64_t(n.area.right-n.area.left)*(n.area.bottom-n.area.top)*4;
        Texture next[2];unsigned replace=0;
        for(unsigned i=0;i<count;++i){D3D11_TEXTURE2D_DESC desc={};if(n.output[i])n.output[i]->GetDesc(&desc);
            if(!n.borrowed_output[i]&&desc.Width==unsigned(n.area.right-n.area.left)&&desc.Height==unsigned(n.area.bottom-n.area.top))continue;
            replace|=1u<<i;
        }
        reserve(bytes*((replace&1u)+((replace>>1)&1u)),"owned-pair");
        D3D11_TEXTURE2D_DESC desc={};desc.Width=unsigned(n.area.right-n.area.left);desc.Height=unsigned(n.area.bottom-n.area.top);
        desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;desc.Format=DXGI_FORMAT_R32_UINT;
        desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        for(unsigned i=0;i<count;++i)if(replace&(1u<<i))checked(device->CreateTexture2D(&desc,nullptr,&next[i]));
        for(unsigned i=0;i<count;++i)if(replace&(1u<<i))output(n,i,std::move(next[i]));
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
        if(n.borrowed_output[index]||desc.Width!=unsigned(r.right-r.left)||desc.Height!=unsigned(r.bottom-r.top)){
            // Reject optional history before allocating another texture. A
            // full budget must not transiently consume the game's remaining VA.
            auto bytes=std::uint64_t(r.right-r.left)*(r.bottom-r.top)*4;
            reserve(bytes,"capture-output");
            output(n,index,crop(source,r));++work.copies;work.copied_pixels+=bytes/4;return;
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
    void invalidate_plan(){++topology_revision;}
    void compile_plan(){
        if(planned_front==front_revision&&planned_topology==topology_revision){++plan_counts.reuses;return;}
        collect_plan.clear();prepare_plan.clear();
        std::map<Node const*,unsigned> collected,prepared;
        std::function<void(std::shared_ptr<Node> const&,unsigned,bool)> gather;
        gather=[&](std::shared_ptr<Node> const& n,unsigned depth,bool project){
            if(depth>256)throw std::runtime_error("retained composition dependency depth");
            project|=n->projects_scene;unsigned bit=project?2:1;auto& seen=collected[n.get()];
            if(seen&bit)return;seen|=bit;
            if(collected.size()>32768)throw std::runtime_error("retained composition plan budget");
            collect_plan.push_back({n,project});
            for(auto const& input:n->inputs)for(auto const& patch:input.patches)gather(patch.node,depth+1,project);
            for(auto const& draw:n->batch)for(auto const& input:draw.inputs)for(auto const& patch:input.patches)gather(patch.node,depth+1,project);
        };
        for(auto const& patch:front.patches)gather(patch.node,0,false);
        std::function<void(std::shared_ptr<Node> const&,unsigned,bool,std::shared_ptr<c3x_renderer::ZoomTransition>)> preparation;
        preparation=[&](std::shared_ptr<Node> const& n,unsigned depth,bool project,std::shared_ptr<c3x_renderer::ZoomTransition> scale){
            if(depth>256)throw std::runtime_error("retained preparation dependency depth");
            if(n->view&&n->projects_scene){project=true;scale=n->view;}
            unsigned bit=project?2:1;auto& seen=prepared[n.get()];if(seen&bit)return;seen|=bit;
            prepare_plan.push_back({n,project,scale});
            for(auto const& input:n->inputs)for(auto const& patch:input.patches)preparation(patch.node,depth+1,project,scale);
            for(auto const& draw:n->batch)for(auto const& input:draw.inputs)for(auto const& patch:input.patches)preparation(patch.node,depth+1,project,scale);
        };
        for(auto const& patch:front.patches)preparation(patch.node,0,false,{});
        planned_front=front_revision;planned_topology=topology_revision;
        ++plan_counts.builds;plan_counts.nodes=collected.size();
    }
    void prepare_front(long long ticks,long long frequency){
        if(!compiled_enabled){
            for(auto const& part:front.patches)collect(part.node,ticks,frequency,0);
            for(auto const& part:front.patches)prepare(part.node,ticks,frequency,0);
            return;
        }
        compile_plan();
        // Mark all projection paths before any source callback, just as the
        // interpreter's complete collect pass does before preparation.
        for(auto const& entry:collect_plan)if(entry.project)if(auto n=entry.node.lock())n->projected_frame=frame;
        for(auto const& entry:collect_plan)if(auto n=entry.node.lock())if(n->sampled!=frame){
            if(n->direct.revision)n->direct_revision=n->direct.revision(ticks,frequency);
            n->sampled=frame;
        }
        for(auto const& entry:prepare_plan)if(auto n=entry.node.lock()){
            auto& visited=entry.project?n->prepared_projected_frame:n->prepared_frame;
            if(visited==frame)continue;visited=frame;
            if(n->sample.prepare&&(!n->sample.projected||n->projected_frame!=frame||entry.project)){
                auto owner=entry.scale.lock();float scale=owner?float(owner->sample(ticks,frequency)):1.f;
                n->sample.prepare(ticks,frequency,scale);
            }
        }
    }
    static bool same_batch(BatchPreparation const& old,Node const& next){
        if(!same_rect(old.area,next.area)||old.placement.owner_before(next.placement)||next.placement.owner_before(old.placement)||old.batch.size()!=next.batch.size())return false;
        for(unsigned i=0;i<2;++i)if(old.width[i]!=next.inputs[i].width||old.height[i]!=next.inputs[i].height||old.formats[i]!=next.inputs[i].format)return false;
        for(std::size_t index=0;index<old.batch.size();++index){auto const& a=old.batch[index];auto const& b=next.batch[index];
            auto const& x=a.placed.command;auto const& y=b.placed.command;
            if(a.placed.x!=b.placed.x||a.placed.y!=b.placed.y||x.kind!=y.kind||!same_rect(x.area,y.area)||
               !same_rect(x.clip,y.clip)||x.source_x!=y.source_x||x.source_y!=y.source_y||x.color!=y.color||
               x.source_width!=y.source_width||x.source_height!=y.source_height)return false;
            Id left[6]={x.destination,x.source,x.background,x.detail,x.background_detail,x.program};
            Id right[6]={y.destination,y.source,y.background,y.detail,y.background_detail,y.program};
            for(unsigned i=0;i<6;++i){
                unsigned l=!left[i]?0:left[i]==old.original[0]?1:left[i]==old.original[1]?2:3;
                unsigned r=!right[i]?0:right[i]==next.original[0]?1:right[i]==next.original[1]?2:3;
                if(l!=r||(l==3&&!same_picture(a.inputs[i],b.inputs[i],false)))return false;
                for(unsigned j=0;j<i;++j)if((left[i]==left[j])!=(right[i]==right[j]))return false;
            }
        }return true;
    }
    void prepare_batch(Node& n){
        if(n.batch_preparation)return;
        if(auto old=recent_batch.lock())if(same_batch(*old,n)){
            n.batch_preparation=std::move(old);++plan_counts.batch_reuses;return;
        }
        auto next=std::make_shared<BatchPreparation>();next->batch=n.batch;
        for(auto& draw:next->batch){auto const& c=draw.placed.command;
            Id ids[6]={c.destination,c.source,c.background,c.detail,c.background_detail,c.program};
            for(unsigned i=0;i<6;++i)if(!ids[i]||ids[i]==n.original[0]||ids[i]==n.original[1])draw.inputs[i]={};
        }
        next->original[0]=n.original[0];next->original[1]=n.original[1];
        for(unsigned i=0;i<2;++i){next->width[i]=n.inputs[i].width;next->height[i]=n.inputs[i].height;next->formats[i]=n.inputs[i].format;}
        next->area=n.area;next->placement=n.placement;
        n.batch_preparation=next;recent_batch=next;++plan_counts.batch_builds;
    }
    void compile_batch(Node& n){
        prepare_batch(n);
        auto& b=*n.batch_preparation;
        if(b.batch_compiled)return;
        b.batch_operands.reserve(n.batch.size());b.batch_commands.reserve(n.batch.size());b.batch_offsets.resize(n.batch.size(),~std::uint64_t(0));
        for(auto const& draw:n.batch){
            auto const& c=draw.placed.command;Id original[6]={c.destination,c.source,c.background,c.detail,c.background_detail,c.program};
            std::array<unsigned,6> operands={};
            for(unsigned i=0;i<6;++i)if(original[i]){
                if(original[i]==n.original[0])operands[i]=1;
                else if(original[i]==n.original[1])operands[i]=2;
                else{
                    // Snapshot wrappers have fresh serials, but exact node/output/
                    // patch identities still certify the same immutable pixels.
                    // Deduplication changes no
                    // read-before-write relation: these inputs are read-only.
                    unsigned source=0;for(;source<b.batch_sources.size();++source)
                        if(same_picture(b.batch_sources[source].picture,draw.inputs[i],false))break;
                    if(source==b.batch_sources.size())b.batch_sources.push_back({draw.inputs[i]});
                    operands[i]=source+3;
                }
            }
            b.batch_operands.push_back(operands);b.batch_commands.push_back(c);
        }
        b.batch_views.resize(b.batch_sources.size());
        for(unsigned index=0;index<b.batch_sources.size();++index){auto const& p=b.batch_sources[index].picture;
            b.batch_views[index]={Compositor::spatial_source_id(index),{},p.width,p.height,p.format};
        }
        b.batch_compiled=true;
        if(n.batch.size()>=256&&!batch_inventories){batch_inventories=1;
            std::array<unsigned,12> kinds={};std::vector<unsigned> readers(b.batch_sources.size());
            std::uint64_t affected=0,source_pixels=0;unsigned aliases=0,cross_reads=0,dynamic_sources=0,fanout=0;
            for(std::size_t index=0;index<n.batch.size();++index){auto const& c=n.batch[index].placed.command;
                if(unsigned(c.kind)<kinds.size())++kinds[unsigned(c.kind)];
                auto r=intersect(intersect(c.area,c.clip),n.area);if(!empty(r))affected+=std::uint64_t(r.right-r.left)*(r.bottom-r.top);
                if(c.source==n.original[0]||c.source==n.original[1]){
                    ++aliases;if(!(c.kind==Kind::native_blend&&c.color==2)&&
                        (c.source_x!=c.area.left||c.source_y!=c.area.top))++cross_reads;
                }
                for(unsigned binding:b.batch_operands[index])if(binding>=3)++readers[binding-3];
            }
            for(auto const& source:b.batch_sources){source_pixels+=std::uint64_t(source.picture.width)*source.picture.height;
                bool dynamic=false;for(auto const& patch:source.picture.patches)dynamic|=patch.node->dynamic;
                dynamic_sources+=dynamic;
            }
            for(auto count:readers)fanout=std::max(fanout,count);
            char line[512];std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=retained-batch-inventory commands=%zu sources=%zu source_pixels=%llu affected_pixels=%llu max_source_fanout=%u pair_source_aliases=%u cross_position_reads=%u dynamic_sources=%u area=%d,%d,%d,%d kinds=%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u\n",
                n.batch.size(),b.batch_sources.size(),source_pixels,affected,fanout,aliases,cross_reads,dynamic_sources,
                n.area.left,n.area.top,n.area.right,n.area.bottom,kinds[0],kinds[1],kinds[2],kinds[3],kinds[4],kinds[5],kinds[6],kinds[7],kinds[8],kinds[9],kinds[10],kinds[11]);
            OutputDebugStringA(line);
        }
    }
    void release_batch(Node& n){n.batch_preparation.reset();}
    bool bind_batch(Node& n,long long ticks,long long frequency,unsigned depth,Id const (&pair)[2],double scale){
        compile_batch(n);auto& b=*n.batch_preparation;bool changed=!b.spatial_attempted;
        // Sources use private plan identities, never hundreds of live image
        // handles. Only the current assembly or fallback operation binds the
        // compositor's existing bounded handle table.
        if(b.batch_sources.size()>8192){b.batch_bound=false;return false;}
        bool revisions_changed=false;
        for(auto& source:b.batch_sources){
            source.pending.clear();source.pending.push_back(source.picture.version);
            for(auto const& patch:source.picture.patches)source.pending.push_back(patch.node->revision);
            revisions_changed|=source.pending!=source.revisions;
        }
        auto available=resident_bytes()<resident_budget?resident_budget-resident_bytes():0;
        if(b.binding_refused&&!revisions_changed&&available==b.binding_allowance){b.batch_bound=false;return false;}
        if(revisions_changed){
            // Generation changes retire copied atlas inputs before recycling
            // their exact source handles. Unchanged sources keep their binds.
            b.spatial_plan={};b.spatial_ready=b.spatial_attempted=false;
            for(unsigned index=0;index<b.batch_sources.size();++index){auto& source=b.batch_sources[index];
                if(source.texture&&source.pending!=source.revisions){
                    b.batch_views[index].texture.Reset();source.texture.Reset();source.owned={};source.physical={};
                }
            }
        }
        std::uint64_t needed=0;
        for(auto const& source:b.batch_sources)if(!source.texture&&!directly_bindable(source.picture))
            needed+=std::uint64_t(source.picture.width)*source.picture.height*4;
        available=resident_bytes()<resident_budget?resident_budget-resident_bytes():0;
        if(needed>available){
            // Binding is optional. Roll it back as a unit and use the existing
            // transient interpreter rather than rejecting the native front.
            b.spatial_plan={};b.spatial_ready=b.spatial_attempted=b.batch_bound=false;
            for(auto& view:b.batch_views)view.texture.Reset();
            for(auto& source:b.batch_sources){source.texture.Reset();source.owned={};source.physical={};source.revisions.swap(source.pending);}
            b.binding_refused=true;b.binding_allowance=resident_bytes()<resident_budget?resident_budget-resident_bytes():0;return false;
        }
        b.binding_refused=false;
        for(unsigned index=0;index<b.batch_sources.size();++index){auto& source=b.batch_sources[index];
            if(source.texture&&source.pending==source.revisions){++plan_counts.source_reuses;continue;}
            auto image=assemble(source.picture,ticks,frequency,depth+1,{},true);
            try{source.texture=replay.texture(image);source.owned=owned_storage.retain(source.texture.Get());source.physical=storage.retain(source.texture.Get());}
            catch(...){replay.destroy(image);throw;}
            // An assembled sparse source can own replay scratch. Destroy its
            // handle rather than recycling it: the captured texture must not
            // enter a writable pool while this plan can still read it.
            replay.destroy(image);b.batch_views[index].texture=source.texture;
            source.revisions.swap(source.pending);++plan_counts.source_binds;changed=true;
        }
        auto resident=resident_bytes();auto allowance=resident<resident_budget?resident_budget-resident:0;
        // Failed admission is an exact plan state too. Retry when its command
        // placement, immutable operands or real available budget changes.
        if(!b.spatial_ready&&allowance!=b.spatial_allowance)changed=true;
        for(std::size_t index=0;index<n.batch.size();++index){
            auto const& placed=n.batch[index].placed;
            int dx=int(std::lround((placed.x-int(n.inputs[0].width/2))*(scale-1.)));
            int dy=int(std::lround((placed.y-int(n.inputs[0].height/2))*(scale-1.)));
            auto offset=(std::uint64_t(unsigned(dx))<<32)|unsigned(dy);
            if(offset!=b.batch_offsets[index])changed=true;
        }
        if(changed)for(std::size_t index=0;index<n.batch.size();++index){
            auto const& placed=n.batch[index].placed;
            int dx=int(std::lround((placed.x-int(n.inputs[0].width/2))*(scale-1.)));
            int dy=int(std::lround((placed.y-int(n.inputs[0].height/2))*(scale-1.)));
            auto offset=(std::uint64_t(unsigned(dx))<<32)|unsigned(dy);
            auto c=placed.command;Id ids[6]={};
            for(unsigned i=0;i<6;++i){auto binding=b.batch_operands[index][i];
                ids[i]=binding==0?0:binding<=2?pair[binding-1]:b.batch_views[binding-3].id;
            }
            c.destination=ids[0];c.source=ids[1];c.background=ids[2];c.detail=ids[3];c.background_detail=ids[4];c.program=ids[5];
            c.area={c.area.left+dx,c.area.top+dy,c.area.right+dx,c.area.bottom+dy};
            c.clip={c.clip.left+dx,c.clip.top+dy,c.clip.right+dx,c.clip.bottom+dy};
            b.batch_commands[index]=c;b.batch_offsets[index]=offset;
        }
        if(changed){
            // Offset/zoom updates retain the atlas of immutable glyphs/tables.
            // The compositor replaces command/tile metadata atomically while
            // checking exact source descriptors; source changes cleared above.
            b.spatial_ready=replay.compile_spatial_sources(b.spatial_plan,b.batch_commands.data(),b.batch_commands.size(),pair[0],pair[1],
                allowance,b.batch_views,true);
            b.spatial_attempted=true;b.spatial_allowance=allowance;
        }
        b.batch_bound=true;
        return b.spatial_ready;
    }
    void collect(std::shared_ptr<Node> const& n,long long ticks,long long frequency,unsigned depth,bool project=false){
        project|=n->projects_scene;
        bool marked=project&&n->projected_frame!=frame;
        if(project)n->projected_frame=frame;
        if(n->sampled==frame&&!marked)return;
        if(depth>256)throw std::runtime_error("retained composition dependency depth");
        // Collect authoritative direct samples before any map rendering or pose
        // joins. Revision callbacks may offer immutable CPU inputs to workers;
        // actual GPU execution remains in the original native command order.
        if(n->direct.revision)n->direct_revision=n->direct.revision(ticks,frequency);
        for(auto const& input:n->inputs)for(auto const& patch:input.patches)
            collect(patch.node,ticks,frequency,depth+1,project);
        for(auto const& draw:n->batch)for(auto const& input:draw.inputs)for(auto const& patch:input.patches)
            collect(patch.node,ticks,frequency,depth+1,project);
        n->sampled=frame;
    }
    // Run preparation before evaluating the stable native graph. This callback
    // may adopt ready CPU content and render private frame scratch, but must
    // never recursively execute native commands or mutate this graph.
    void prepare(std::shared_ptr<Node> const& n,long long ticks,long long frequency,unsigned depth,
            bool project=false,float scale=1.f){
        if(depth>256)throw std::runtime_error("retained preparation dependency depth");
        if(n->view&&n->projects_scene){project=true;scale=float(n->view->sample(ticks,frequency));}
        auto& visited=project?n->prepared_projected_frame:n->prepared_frame;
        if(visited==frame)return;visited=frame;
        if(n->sample.prepare && (!n->sample.projected || n->projected_frame!=frame || project)){
            n->sample.prepare(ticks,frequency,scale);
        }
        for(auto const& input:n->inputs)for(auto const& patch:input.patches)
            prepare(patch.node,ticks,frequency,depth+1,project,scale);
        for(auto const& draw:n->batch)for(auto const& input:draw.inputs)for(auto const& patch:input.patches)
            prepare(patch.node,ticks,frequency,depth+1,project,scale);
    }
    void evaluate(std::shared_ptr<Node> const& n,long long ticks,long long frequency,unsigned depth){
        if(n->seen==frame)return;
        if(depth>256)throw std::runtime_error("retained composition dependency depth");
        bool had_map=n->map_dynamic;
        if(n->sample){
            // The displayed scene is sampled at its projection below. Native
            // save/restore images retain the canonical publication, so they
            // cannot trigger a second geometry draw at the old zoom.
            auto sampled=n->sample.projected&&n->projected_frame==frame?SampledImage{}:n->sample(ticks,frequency);
            if(sampled.kind==SampledImage::Kind::frozen){
                n->sample={};n->sample_target={};n->dynamic=n->map_dynamic=false;n->retired=true;
                invalidate_plan();
            }else if(sampled.kind==SampledImage::Kind::bgra){
                auto r=sampled.area;unsigned w=unsigned(n->area.right-n->area.left),h=unsigned(n->area.bottom-n->area.top);
                if(r.left<0||r.top<0||r.right-r.left!=int(w)||r.bottom-r.top!=int(h))
                    throw std::runtime_error("retained sample extent changed");
                if(!n->sample_target.texture){
                    auto bytes=std::uint64_t(w)*h*4;
                    reserve(bytes,"sample-view-output");
                    auto canvas=replay.create(w,h,Format::bgra32,false);
                    if(!canvas)throw std::runtime_error("retained sample admission failed");
                    n->sample_target=replay.release_import_target(canvas);++sample_allocations;
                }
                if(!replay.import_bgra(n->sample_target,sampled.texture.Get(),r.left,r.top,sampled.sharpness))
                    throw std::runtime_error("retained sample import failed");
                ++sample_imports;output(*n,0,n->sample_target.texture);n->source_generation=sampled.generation;n->revision=++serial;
            }else if(sampled.kind==SampledImage::Kind::immutable){
                if(!sampled.texture)throw std::runtime_error("retained visual selection retired");
                n->source_generation=sampled.generation;
                if(sampled.texture.Get()!=n->output[0].Get()){
                    n->sample_target={};
                    output(*n,0,std::move(sampled.texture));n->revision=++serial;
                }
            }
        }else if(n->selected_world){
            auto& versions=n->pending_dependencies;versions.clear();versions.push_back(n->inputs[0].version);versions.push_back(n->inputs[1].version);
            n->map_dynamic=false;
            for(unsigned i=0;i<2;++i)for(auto const& patch:n->inputs[i].patches){
                evaluate(patch.node,ticks,frequency,depth+1);versions.push_back(patch.node->revision);
                n->map_dynamic|=patch.node->map_dynamic;
            }
            if(!n->output[0]||versions!=n->dependencies){
                for(unsigned i=0;i<2;++i){
                    if(compiled_enabled&&n->inputs[i].format==n->selected_format[i]&&exact_plane(n->inputs[i],n->area)){
                        auto const& patch=n->inputs[i].patches.front();
                        // The full dependency proof above includes both the
                        // picture version and source revision. Pointer reuse
                        // alone cannot leave a newly sampled plane unchanged.
                        output(*n,i,patch.node->output[patch.output]);n->borrowed_output[i]=true;
                        ++work.selected_borrows;work.avoided_copy_pixels+=std::uint64_t(n->area.right-n->area.left)*(n->area.bottom-n->area.top);
                        continue;
                    }
                    // Acquire private storage before supplying an assembly
                    // target; a previous exact selection may share its source.
                    if(n->borrowed_output[i]){
                        auto bytes=std::uint64_t(n->area.right-n->area.left)*(n->area.bottom-n->area.top)*4;
                        reserve(bytes,"selected-owned-output");
                        auto texture=n->output[i];D3D11_TEXTURE2D_DESC desc={};texture->GetDesc(&desc);
                        Texture target;checked(device->CreateTexture2D(&desc,nullptr,&target));output(*n,i,std::move(target));
                    }
                    ++work.selected_owned;
                    auto source=assemble(n->inputs[i],ticks,frequency,depth+1,{},true,true,n->output[i].Get());
                    try{if(replay.texture(source)!=n->output[i].Get())capture_output(*n,i,replay.texture(source),n->area);}
                    catch(...){replay.recycle(source);throw;}replay.recycle(source);
                }
                n->dependencies.swap(versions);n->revision=++serial;
            }
        }else if(n->view&&n->projects_scene){
            float scale=float(n->view->sample(ticks,frequency));
            auto source=assemble_projected(n->inputs[0],ticks,frequency,depth+1,scale,extent(n->inputs[0]),n->inputs[0].width,n->inputs[0].height);
            try{
                for(unsigned i=0;i<(n->view_native_format?2u:1u);++i){
                    auto& target=i?n->view_words:n->sample_target;
                    if(!target.texture){
                        auto id=replay.create(n->inputs[0].width,n->inputs[0].height,Format::bgra32,false);
                        if(!id)throw std::runtime_error("projected world admission");
                        target=replay.release_import_target(id);output(*n,i,target.texture);
                    }
                }
                if(!replay.transform_view(n->sample_target,source,1.f,n->view_native_format?&n->view_words:nullptr,
                    n->view_native_format==2?Format::rgb565:Format::rgb555))throw std::runtime_error("projected world rejected");
            }catch(...){replay.recycle(source);throw;}
            replay.recycle(source);n->view_scale=scale;n->revision=++serial;selected_view_scale=scale;
            n->map_dynamic=false;
            for(auto const& patch:n->inputs[0].patches)n->map_dynamic|=patch.node->map_dynamic;
        }else if(n->view){
            auto& versions=n->pending_dependencies;versions.clear();
            n->map_dynamic=false;
            for(auto const& patch:n->inputs[0].patches){
                evaluate(patch.node,ticks,frequency,depth+1);versions.push_back(patch.node->revision);
                n->map_dynamic|=patch.node->map_dynamic;
            }
            float scale=float(n->view->sample(ticks,frequency));
            if(!n->output[0]||n->view_scale!=scale||versions!=n->dependencies){
                auto source=assemble(n->inputs[0],ticks,frequency,depth+1,{},true);
                try{
                    for(unsigned i=0;i<(n->view_native_format?2u:1u);++i){
                        auto& target=i?n->view_words:n->sample_target;
                        if(!target.texture){
                            auto bytes=std::uint64_t(n->inputs[0].width)*n->inputs[0].height*4;
                            reserve(bytes,"sample-view-output");
                            auto id=replay.create(n->inputs[0].width,n->inputs[0].height,Format::bgra32,false);
                            if(!id)throw std::runtime_error("retained view output admission failed");
                            target=replay.release_import_target(id);output(*n,i,target.texture);
                        }
                    }
                    if(!replay.transform_view(n->sample_target,source,scale,n->view_native_format?&n->view_words:nullptr,
                        n->view_native_format==2?Format::rgb565:Format::rgb555))throw std::runtime_error("retained world view rejected");
                }catch(...){replay.recycle(source);throw;}
                replay.recycle(source);
                n->view_scale=scale;n->dependencies.swap(versions);n->revision=++serial;
            }
            selected_view_scale=n->view_scale;
        }else if(!n->batch.empty()){
            auto& versions=n->pending_dependencies;versions.clear();
            n->map_dynamic=false;bool dynamic=false;
            auto visit=[&](Picture const& input){
                versions.push_back(input.version);
                for(auto const& patch:input.patches){evaluate(patch.node,ticks,frequency,depth+1);
                    versions.push_back(patch.node->revision);n->map_dynamic|=patch.node->map_dynamic;dynamic|=patch.node->dynamic;}
            };
            visit(n->inputs[0]);visit(n->inputs[1]);
            double scale=n->placement->sample(ticks,frequency);
            if(compiled_enabled){compile_batch(*n);for(auto const& source:n->batch_preparation->batch_sources)visit(source.picture);}
            for(auto const& draw:n->batch){
                int dx=int(std::lround((draw.placed.x-int(n->inputs[0].width/2))*(scale-1.)));
                int dy=int(std::lround((draw.placed.y-int(n->inputs[0].height/2))*(scale-1.)));
                versions.push_back((std::uint64_t(unsigned(dx))<<32)|unsigned(dy));
                if(!compiled_enabled)for(auto const& input:draw.inputs)if(input.width)visit(input);
            }
            if((!n->output[0]||versions!=n->dependencies)&&!(had_map&&!n->map_dynamic&&n->output[0])){
                // One pair per immutable HUD generation. Reserve both growths
                // before initializing either result; old/new overlap counts.
                auto bytes=std::uint64_t(n->inputs[0].width)*n->inputs[0].height*4;
                reserve((!n->output[0]?bytes:0)+(!n->output[1]?bytes:0),"hud-pair");
                for(unsigned i=0;i<2;++i)if(!n->output[i]){
                    auto id=replay.create(n->inputs[i].width,n->inputs[i].height,Format::bgra32,false);
                    if(!id)throw std::runtime_error("retained HUD pair admission");
                    auto target=replay.release_import_target(id);output(*n,i,target.texture);
                }
                Id pair[2]={};
                try{
                    for(unsigned i=0;i<2;++i){
                        pair[i]=replay.attach_target_unrecorded(n->output[i].Get(),n->inputs[i].format);
                        if(!pair[i])throw std::runtime_error("retained HUD target admission");
                        // Complete detached before-images can assemble into
                        // their admitted output directly. Sparse/aliased reads
                        // retain the ordinary scratch path.
                        auto base=assemble(n->inputs[i],ticks,frequency,depth+1,{},true,true,
                            compiled_enabled?n->output[i].Get():nullptr);
                        if(replay.texture(base)!=n->output[i].Get()){
                            context->CopyResource(n->output[i].Get(),replay.texture(base));
                            ++work.copies;work.copied_pixels+=bytes/4;
                        }
                        replay.recycle(base);
                    }
                    bool spatial=compiled_enabled&&bind_batch(*n,ticks,frequency,depth,pair,scale);
                    if(spatial){
                        work.operations+=unsigned(n->batch.size());
                        if(!replay.submit_spatial_sources(n->batch_preparation->spatial_plan,pair[0],pair[1],n->batch_preparation->batch_views))throw std::runtime_error("retained spatial HUD operation rejected");
                    }else if(compiled_enabled&&n->batch_preparation->batch_bound){
                        work.operations+=unsigned(n->batch_preparation->batch_commands.size());
                        for(std::size_t index=0;index<n->batch_preparation->batch_commands.size();++index){
                            auto c=n->batch_preparation->batch_commands[index];Id ids[6]={};
                            for(unsigned i=0;i<6;++i){auto binding=n->batch_preparation->batch_operands[index][i];
                                ids[i]=binding==0?0:binding<=2?pair[binding-1]:n->batch_preparation->batch_views[binding-3].id;
                            }
                            c.destination=ids[0];c.source=ids[1];c.background=ids[2];c.detail=ids[3];c.background_detail=ids[4];c.program=ids[5];
                            if(!replay.submit_source_commands(&c,1,n->batch_preparation->batch_views))throw std::runtime_error("retained bound HUD operation rejected");
                        }
                    }else for(auto const& draw:n->batch){
                        auto c=draw.placed.command;
                        Id original[6]={c.destination,c.source,c.background,c.detail,c.background_detail,c.program},ids[6]={};
                        try{
                            for(unsigned i=0;i<6;++i)if(original[i]){
                                if(original[i]==n->original[0])ids[i]=pair[0];
                                else if(original[i]==n->original[1])ids[i]=pair[1];
                                else{
                                    for(unsigned prior=0;prior<i;++prior)if(original[prior]==original[i])ids[i]=ids[prior];
                                    if(!ids[i])ids[i]=assemble(draw.inputs[i],ticks,frequency,depth+1,{},true);
                                }
                            }
                            c.destination=ids[0];c.source=ids[1];c.background=ids[2];
                            c.detail=ids[3];c.background_detail=ids[4];c.program=ids[5];
                            int dx=int(std::lround((draw.placed.x-int(n->inputs[0].width/2))*(scale-1.)));
                            int dy=int(std::lround((draw.placed.y-int(n->inputs[0].height/2))*(scale-1.)));
                            c.area={c.area.left+dx,c.area.top+dy,c.area.right+dx,c.area.bottom+dy};
                            c.clip={c.clip.left+dx,c.clip.top+dy,c.clip.right+dx,c.clip.bottom+dy};
                            ++work.operations;
                            if(!replay.submit(&c,1))throw std::runtime_error("retained HUD operation rejected");
                        }catch(...){for(unsigned i=0;i<6;++i)if(ids[i]&&ids[i]!=pair[0]&&ids[i]!=pair[1]&&
                            std::find(ids,ids+i,ids[i])==ids+i)replay.recycle(ids[i]);throw;}
                        for(unsigned i=0;i<6;++i)if(ids[i]&&ids[i]!=pair[0]&&ids[i]!=pair[1]&&
                            std::find(ids,ids+i,ids[i])==ids+i)replay.recycle(ids[i]);
                    }
                }catch(...){for(auto id:pair)if(id)replay.recycle(id);throw;}
                for(auto id:pair)replay.recycle(id);
                n->dependencies.swap(versions);n->revision=++serial;
            }
            n->dynamic=dynamic||bool(n->placement);
            if(had_map&&!n->map_dynamic){
                // Saved generations keep their completed pixels, never a
                // mutable live selection or a future camera's zoom placement.
                release_batch(*n);n->batch.clear();n->inputs[0]={};n->inputs[1]={};n->placement.reset();
                n->dependencies.clear();n->dynamic=n->view_dependent=false;n->retired=true;
                invalidate_plan();
            }
        }else if(n->operation){
            auto& versions=n->pending_dependencies;versions.clear();
            if(n->direct.revision)versions.push_back(n->direct_revision);
            int dx=0,dy=0;
            if(n->placement){
                auto scale=n->placement->sample(ticks,frequency);
                dx=int(std::lround((n->anchor_x-int(n->inputs[0].width/2))*(scale-1.)));
                dy=int(std::lround((n->anchor_y-int(n->inputs[0].height/2))*(scale-1.)));
                versions.push_back((std::uint64_t(unsigned(dx))<<32)|unsigned(dy));
            }
            n->dynamic=n->direct.animated||bool(n->placement);n->map_dynamic=false;
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
                bool overwrite=!n->placement&&local&&!n->direct.draw&&(original.kind==Kind::fill||original.kind==Kind::copy||
                    original.kind==Kind::quantize||(original.kind==Kind::expand&&original.color==65536)||
                    (original.kind==Kind::native_image&&original.color==65536));
                bool tight_source=!n->placement&&local&&original.kind==Kind::native_image&&
                    original.source_width==original.area.right-original.area.left&&original.source_height==original.area.bottom-original.area.top;
                bool owned_native=compiled_enabled&&tight_source&&!n->direct.draw&&!n->direct.revision&&!n->direct.animated&&!n->direct.input_bytes&&
                    !original.background&&!original.program&&n->inputs[0].format!=Format::bgra32&&n->inputs[1].format==n->inputs[0].format&&
                    (!original.detail||(original.detail!=original.destination&&n->inputs[3].format==Format::bgra32));
                // Preserve the interpreter's aliased-before-image rules. An
                // independent source can bind readonly only after all of its
                // reads have been resolved, before either output is written.
                for(unsigned i:{1u,4u})if(n->original[i]&&(n->original[i]==n->original[0]||n->original[i]==n->original[3]))owned_native=false;
                Rect source_area={original.source_x+n->area.left-original.area.left,original.source_y+n->area.top-original.area.top,
                    original.source_x+n->area.right-original.area.left,original.source_y+n->area.bottom-original.area.top};
                int x=local?n->area.left:0,y=local?n->area.top:0;
                try{
                    if(owned_native){
                        // Retain every before-image first. They may share a
                        // live selection; none is mutated by pair admission.
                        Id before[2]={};
                        try{
                            for(unsigned i=0;i<6;++i)if(n->original[i]){
                                if(i==0||i==3){if(!overwrite)before[i==3]=assemble(n->inputs[i],ticks,frequency,depth+1,n->area,true);}
                                else temporary[i]=assemble(n->inputs[i],ticks,frequency,depth+1,
                                    i==1||i==4?source_area:Rect{},true);
                            }
                            admit_owned_outputs(*n,original.detail?2u:1u);
                            for(unsigned i=0;i<(original.detail?2u:1u);++i){
                                unsigned slot=i?3:0;
                                temporary[slot]=replay.attach_target_unrecorded(n->output[i].Get(),n->inputs[slot].format);
                                if(!temporary[slot])throw std::runtime_error("retained native owned target admission");
                            }
                            for(unsigned i=0;i<(original.detail?2u:1u);++i){
                                if(before[i]){context->CopyResource(n->output[i].Get(),replay.texture(before[i]));
                                    ++work.copies;work.copied_pixels+=std::uint64_t(n->area.right-n->area.left)*(n->area.bottom-n->area.top);}
                            }
                        }catch(...){for(auto id:before)if(id)replay.recycle(id);throw;}
                        for(auto id:before)if(id)replay.recycle(id);
                    }else
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
                    if(n->placement){
                        c.area={c.area.left+dx,c.area.top+dy,c.area.right+dx,c.area.bottom+dy};
                        c.clip={c.clip.left+dx,c.clip.top+dy,c.clip.right+dx,c.clip.bottom+dy};
                    }
                    Rect result={n->area.left-x,n->area.top-y,n->area.right-x,n->area.bottom-y};
                    if(tight_source){c.area=c.clip=result;c.source_x=c.source_y=0;
                        c.source_width=result.right-result.left;c.source_height=result.bottom-result.top;}
                    if(!(n->direct.draw?n->direct.draw(replay,c):replay.submit(&c,1)))throw std::runtime_error("retained operation rejected");
                    if(owned_native){++work.direct_native_images;work.avoided_copy_pixels+=
                        std::uint64_t(result.right-result.left)*(result.bottom-result.top)*(c.detail?2u:1u);}
                    else{capture_output(*n,0,replay.texture(c.destination),result);
                        if(c.detail)capture_output(*n,1,replay.texture(c.detail),result);}
                    n->dependencies.swap(versions);n->revision=++serial;
                }catch(...){for(unsigned i=0;i<6;++i)if(temporary[i]&&std::find(temporary,temporary+i,temporary[i])==temporary+i)replay.recycle(temporary[i]);throw;}
                for(unsigned i=0;i<6;++i)if(temporary[i]&&std::find(temporary,temporary+i,temporary[i])==temporary+i)replay.recycle(temporary[i]);
            }
            // A frozen source can leave the output revision unchanged. Retire
            // its recipe even then, so HUD holes do not keep every old camera.
            if(!n->dynamic){for(auto& input:n->inputs)input={};n->dependencies.clear();n->operation=false;invalidate_plan();}
        }
        if(n->view&&had_map&&!n->map_dynamic){
            // Native HUD/save copies keep these completed pixels after the
            // camera retires. Release both projected and ordinary view recipes
            // so they cannot retain old scenes or follow subsequent zooms.
            n->inputs[0]={};n->view.reset();n->sample_target={};n->view_words={};
            n->dynamic=n->view_dependent=n->projects_scene=false;
            invalidate_plan();
        }
        n->seen=frame;
    }
    bool projectable(Picture const& p,bool& scene,unsigned depth){
        if(depth>256||p.format!=Format::bgra32)return false;
        for(auto const& patch:p.patches){auto const& n=patch.node;
            if(n->sample.projected){scene=true;continue;}
            if(!n->map_dynamic)continue; // Immutable native overlay, same pixels and order.
            if(n->operation&&n->command.kind==Kind::native_image&&patch.output==1&&!n->direct.draw&&
               n->command.color<=65535&&n->inputs[1].width&&n->inputs[4].width&&
               n->inputs[1].format!=Format::bgra32&&n->inputs[4].format==Format::bgra32&&
               projectable(n->inputs[3],scene,depth+1))continue;
            if(!n->operation||n->command.kind!=Kind::unit_over||patch.output!=1||n->direct.draw||
               n->inputs[1].format!=Format::bgra32||
               !projectable(n->inputs[3],scene,depth+1))return false;
            // The overlay source must be independent of its world underlay.
            for(auto const& source:n->inputs[1].patches)if(source.node->map_dynamic)return false;
        }
        return true;
    }
    Rect project(Rect r,float scale,unsigned width,unsigned height)const{
        c3x_renderer::SceneProjection p(width,height,scale);
        return intersect({int(std::lround(p.x(float(r.left)))),int(std::lround(p.y(float(r.top)))),
            int(std::lround(p.x(float(r.right)))),int(std::lround(p.y(float(r.bottom))))},
            {0,0,int(width),int(height)});
    }
    void projected_output(Node& n,Rect area){
        unsigned w=unsigned(area.right-area.left),h=unsigned(area.bottom-area.top);
        if(n.sample_target.width!=w||n.sample_target.height!=h){
            auto bytes=std::uint64_t(w)*h*4;
            reserve(bytes,"projected-output");
            auto id=replay.create(w,h,Format::bgra32,false);
            if(!id)throw std::runtime_error("projected scene allocation");
            n.sample_target=replay.release_import_target(id);output(n,0,n.sample_target.texture);
        }
        n.area=area;
    }
    std::shared_ptr<Node> evaluate_projected(Patch const& patch,long long ticks,long long frequency,
            unsigned depth,float scale,unsigned width,unsigned height){
        if(depth>256)throw std::runtime_error("projected scene depth");
        auto const& original=patch.node;
        if(!original->projected)original->projected=node();
        auto n=original->projected;
        if(n->seen==frame)return n;
        Rect area=project(original->area,scale,width,height);
        if(empty(area)){n->area=area;n->seen=frame;return n;}
        if(original->sample.projected||original->retired){
            auto sampled=original->retired?SampledImage::frozen():original->sample.projected(ticks,frequency,scale);
            if(sampled.kind==SampledImage::Kind::frozen||sampled.kind==SampledImage::Kind::held){
                if(sampled.kind==SampledImage::Kind::frozen){
                    original->retired=true;original->sample={};original->dynamic=original->map_dynamic=false;
                    invalidate_plan();
                }
                // A camera can retire before its first display. Preserve its
                // completed publication rather than returning an empty image.
                // If a prior projected pose exists, reproject that exact pose
                // during the short handoff instead of rewinding its animation.
                if(!n->output[0]||n->view_scale!=scale){
                    if(!n->output[0]){n->publication=original->publication;n->source_generation=original->source_generation;n->map_source=original->map_source;}
                    auto source=n->output[0]?n->output[0]:original->output[patch.output];
                    auto source_area=n->output[0]?n->area:original->area;
                    float relative=n->output[0]?scale/n->view_scale:scale;
                    auto id=replay.create(area.right-area.left,area.bottom-area.top,Format::bgra32,false);
                    if(!id)throw std::runtime_error("retired projected scene admission");
                    auto target=replay.release_import_target(id);
                    auto input=replay.attach_source_unrecorded(source.Get(),Format::bgra32);
                    try{projected_layer.draw(device,context,replay.view(input),nullptr,target.write.Get(),area,
                        relative,float(width/2),float(height/2),-float(source_area.left),-float(source_area.top));}
                    catch(...){replay.recycle(input);throw;}replay.recycle(input);
                    n->sample_target=std::move(target);output(*n,0,n->sample_target.texture);
                    n->area=area;n->view_scale=scale;n->revision=++serial;
                }
                n->seen=frame;return n;
            }
            if(sampled.kind!=SampledImage::Kind::bgra)throw std::runtime_error("projected scene unavailable");
            projected_output(*n,area);
            if(!replay.import_bgra(n->sample_target,sampled.texture.Get(),sampled.area.left,sampled.area.top,sampled.sharpness))
                throw std::runtime_error("projected scene import");
            n->publication=original->publication;n->source_generation=sampled.generation;n->map_source=original->map_source;
        }else if(original->operation&&original->map_dynamic){
            // Project the retained underlay recursively. Native image ordering
            // and version ownership are unchanged. For a keyed native image,
            // use its native words as a mask over its full-color source. The
            // pathfinder can draw into that source without resampling the map.
            auto below=assemble_projected(original->inputs[3],ticks,frequency,depth+1,scale,area,width,height);
            Id source=0,key=0;
            try{
                projected_output(*n,area);auto const& c=original->command;
                if(c.kind==Kind::native_image){
                    source=assemble(original->inputs[4],ticks,frequency,depth+1,{},true);
                    key=assemble(original->inputs[1],ticks,frequency,depth+1,{},true);
                }else source=assemble(original->inputs[1],ticks,frequency,depth+1,{},true);
                projected_layer.draw(device,context,replay.view(source),replay.view(below),n->sample_target.write.Get(),
                    area,scale,float(width/2),float(height/2),float(c.source_x-c.area.left),float(c.source_y-c.area.top),
                    key?replay.view(key):nullptr,c.color);
            }catch(...){if(source)replay.recycle(source);if(key)replay.recycle(key);replay.recycle(below);throw;}
            replay.recycle(source);if(key)replay.recycle(key);replay.recycle(below);
            original->map_dynamic=false;
            for(auto const& input:original->inputs)for(auto const& part:input.patches)
                original->map_dynamic|=part.node->map_dynamic;
        }else{
            evaluate(original,ticks,frequency,depth+1);
            auto source_identity=static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(original->output[patch.output].Get()));
            auto viewport=(std::uint64_t(width)<<32)|height;
            auto source_origin=(std::uint64_t(unsigned(original->area.left))<<32)|unsigned(original->area.top);
            // Independent overlays do not change merely because the live map
            // below them advanced its clock. Zoom/source changes still redraw.
            if(compiled_enabled&&n->output[0]&&same_rect(n->area,area)&&n->view_scale==scale&&
               n->dependencies.size()==5&&n->dependencies[0]==original->revision&&n->dependencies[1]==patch.output&&n->dependencies[2]==source_identity&&
               n->dependencies[3]==viewport&&n->dependencies[4]==source_origin){
                n->seen=frame;return n;
            }
            projected_output(*n,area);
            auto texture=original->output[patch.output];
            if(texture){
                auto source=replay.attach_source_unrecorded(texture.Get(),Format::bgra32);
                try{projected_layer.draw(device,context,replay.view(source),nullptr,n->sample_target.write.Get(),area,
                    scale,float(width/2),float(height/2),-float(original->area.left),-float(original->area.top));}
                catch(...){replay.recycle(source);throw;}replay.recycle(source);
            }else{unsigned zero[4]={};context->ClearUnorderedAccessViewUint(n->sample_target.write.Get(),zero);}
            n->dependencies={original->revision,patch.output,source_identity,viewport,source_origin};
        }
        n->view_scale=scale;n->seen=frame;n->revision=++serial;return n;
    }
    Id assemble_projected(Picture const& picture,long long ticks,long long frequency,unsigned depth,
            float scale,Rect region,unsigned width,unsigned height){
        std::vector<Patch> selected;
        for(auto const& patch:picture.patches){
            auto area=intersect(project(patch.area,scale,width,height),region);if(empty(area))continue;
            auto n=evaluate_projected(patch,ticks,frequency,depth+1,scale,width,height);
            if(n->output[0])selected.push_back({intersect(area,n->area),n,0});
        }
        if(selected.size()==1){auto const& p=selected.front();
            if(p.area.left==region.left&&p.area.top==region.top&&p.area.right==region.right&&p.area.bottom==region.bottom&&
               p.node->area.left==region.left&&p.node->area.top==region.top&&p.node->area.right==region.right&&p.node->area.bottom==region.bottom)
                return replay.attach_source_unrecorded(p.node->output[0].Get(),Format::bgra32);
        }
        auto out=replay.create(region.right-region.left,region.bottom-region.top,Format::bgra32,true);
        if(!out)throw std::runtime_error("projected assembly admission");
        for(auto const& part:selected){auto a=part.area;if(empty(a))continue;
            auto b=part.node->area;D3D11_BOX box={unsigned(a.left-b.left),unsigned(a.top-b.top),0,
                unsigned(a.right-b.left),unsigned(a.bottom-b.top),1};
            context->CopySubresourceRegion(replay.texture(out),0,a.left-region.left,a.top-region.top,0,part.node->output[0].Get(),0,&box);
            ++work.copies;work.copied_pixels+=std::uint64_t(a.right-a.left)*(a.bottom-a.top);
        }
        ++work.assemblies;work.assembly_pixels+=std::uint64_t(region.right-region.left)*(region.bottom-region.top);
        return out;
    }
    Id assemble(Picture const& p,long long ticks,long long frequency,unsigned depth,Rect region={},bool readonly=false,bool initialize=true,ID3D11Texture2D* destination=nullptr){
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
        // A complete partition can assemble directly into a live selection's
        // persistent output. Its inputs never reference that selection, so
        // these copies cannot overwrite a source version. Sparse reads still
        // use cleared scratch; ordinary immutable native nodes are unchanged.
        // A mutable selected owner may already share an input plane. Only a
        // detached target can replace the intermediate assembly allocation.
        if(destination)for(auto const& part:p.patches)if(part.node->output[part.output].Get()==destination){destination=nullptr;break;}
        Id out=destination&&base?replay.attach_source_unrecorded(destination,p.format):
            replay.create(region.right-region.left,region.bottom-region.top,p.format,initialize&&!base);
        if(!out)throw std::runtime_error("retained composition scratch budget");
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
    bool assemble_front(Id& image,Rect& damage){
        if(!compiled_enabled||!front.partitioned||front.patches.size()<2){release_front();return false;}
        std::uint64_t covered=0;
        for(auto const& patch:front.patches){auto area=intersect(patch.area,extent(front)),source=patch.node->area;
            if(empty(area)||!same_rect(area,patch.area)||!patch.node->output[patch.output]||
               source.left>area.left||source.top>area.top||source.right<area.right||source.bottom<area.bottom||
               patch.node->output[patch.output].Get()==assembled_front.Get()){release_front();return false;}
            covered+=std::uint64_t(area.right-area.left)*(area.bottom-area.top);
        }
        auto pixels=std::uint64_t(front.width)*front.height;
        if(covered!=pixels){release_front();return false;} // sparse canvases keep cleared scratch
        if(!assembled_front||assembled_width!=front.width||assembled_height!=front.height||assembled_format!=front.format){
            release_front();auto used=resident_bytes();
            if(used>resident_budget||pixels*4>resident_budget-used)return false;
            auto id=replay.create(front.width,front.height,Format::bgra32,false);if(!id)return false;
            auto target=replay.release_import_target(id);if(!target.texture)return false;
            assembled_front=std::move(target.texture);front_owned=owned_storage.retain(assembled_front.Get());front_physical=storage.retain(assembled_front.Get());
            assembled_width=front.width;assembled_height=front.height;assembled_format=front.format;
        }
        bool complete=assembled_revision!=front_revision||assembled_patches.size()!=front.patches.size();
        if(!complete)for(std::size_t index=0;index<front.patches.size();++index){auto const& old=assembled_patches[index];auto const& patch=front.patches[index];
            if(old.node.lock()!=patch.node||old.output!=patch.output||!same_rect(old.area,patch.area)){complete=true;break;}
        }
        damage={};std::uint64_t changed_pixels=0;
        for(std::size_t index=0;index<front.patches.size();++index){auto const& patch=front.patches[index];
            if(!complete&&assembled_patches[index].revision==patch.node->revision)continue;
            auto area=patch.area,source=patch.node->area;
            D3D11_BOX box={unsigned(area.left-source.left),unsigned(area.top-source.top),0,
                unsigned(area.right-source.left),unsigned(area.bottom-source.top),1};
            context->CopySubresourceRegion(assembled_front.Get(),0,area.left,area.top,0,patch.node->output[patch.output].Get(),0,&box);
            damage=empty(damage)?area:Rect{std::min(damage.left,area.left),std::min(damage.top,area.top),std::max(damage.right,area.right),std::max(damage.bottom,area.bottom)};
            ++work.copies;changed_pixels+=std::uint64_t(area.right-area.left)*(area.bottom-area.top);
        }
        if(changed_pixels){++work.assemblies;work.copied_pixels+=changed_pixels;work.assembly_pixels+=changed_pixels;}
        assembled_patches.clear();assembled_patches.reserve(front.patches.size());
        for(auto const& patch:front.patches)assembled_patches.push_back({patch.area,patch.node,patch.output,patch.node->revision});
        assembled_revision=front_revision;
        // An unsuccessful prior display may already have assembled these
        // pixels. Retry the complete display rather than declaring it drawn.
        if(empty(damage))damage=extent(front);
        image=replay.attach_source_unrecorded(assembled_front.Get(),front.format);
        if(!image){release_front();return false;}return true;
    }
public:
    RetainedComposition(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),replay(d,c,128u*1024u*1024u){replay.share_storage(storage);}
    bool prepare_assets(std::function<bool()> cancelled={}){
        if(!replay.prepare_assets(cancelled) || (cancelled && cancelled()))return false;
        projected_layer.prepare_assets(device);
        return !cancelled || !cancelled();
    }
    ~RetainedComposition(){front={};images.clear();world_selection.reset();}
    void clear(){release_front();recent_batch.reset();recent_recipes={};recipe_cursor=0;recipe_counts={};collect_plan.clear();prepare_plan.clear();plan_counts={};batch_inventories=0;invalidate_plan();front={};images.clear();world_selection.reset();replay.clear_working();admitted=true;}
    void discard(){release_front();recent_batch.reset();recent_recipes={};recipe_cursor=0;recipe_counts={};collect_plan.clear();prepare_plan.clear();plan_counts={};batch_inventories=0;invalidate_plan();front={};images.clear();world_selection.reset();replay.clear_working();admitted=false;}
    void uncommit(){release_front();collect_plan.clear();prepare_plan.clear();invalidate_plan();front={};}
    // The interpreter remains executable at the same clock for pixel oracles.
    // This switch changes execution only, never captured native semantics.
    void set_compiled_enabled(bool value){compiled_enabled=value;if(!value)release_front();invalidate_plan();}
    std::uint64_t bytes()const{return resident_bytes();}
    std::uint64_t allocation_bytes()const{return storage.bytes()+direct_bytes;}
    std::uint64_t allocation_peak()const{return storage.peak();}
    void share_storage(CompositionStorage const& tracker){storage=tracker;replay.share_storage(tracker);}
    Counts replay_stats()const{return replay.stats();}
    std::uint64_t sampling_allocations()const{return sample_allocations;}
    std::uint64_t sampling_imports()const{return sample_imports;}
    std::uint64_t source_view_creations()const{return source_views;}
    std::size_t node_count()const{return nodes;}
    RecipeReuse recipe_reuse()const{return recipe_counts;}
    PlanReuse plan_reuse()const{return plan_counts;}
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
            for(auto const& draw:n->batch)for(auto const& input:draw.inputs)for(auto const& patch:input.patches)pending.push_back(patch.node.get());
        }return count;
    }
    // Optional route witness: identify copied source versions reachable from
    // the actual composed front. Mixed or absent map sources do not certify a
    // destination. This inspection neither samples nor prepares any source.
    std::pair<std::uint64_t,std::uint64_t> front_publication()const{
        std::unordered_set<Node const*> visited;std::vector<Node const*> pending;
        for(auto const& patch:front.patches)pending.push_back(patch.node.get());
        std::pair<std::uint64_t,std::uint64_t> found{};
        while(!pending.empty()){
            auto n=pending.back();pending.pop_back();
            if(n->projected_frame==frame && n->projected && n->projected->map_source)n=n->projected.get();
            if(!visited.insert(n).second)continue;
            if(visited.size()>32768)return {};
            if(n->map_source && (!n->publication||!n->source_generation))return {};
            if(n->publication){auto proof=std::make_pair(n->publication,n->source_generation);
                if(found.first&&found!=proof)return {};found=proof;}
            for(auto const& input:n->inputs)for(auto const& patch:input.patches)pending.push_back(patch.node.get());
            for(auto const& draw:n->batch)for(auto const& input:draw.inputs)for(auto const& patch:input.patches)pending.push_back(patch.node.get());
        }return found;
    }
    bool animated()const{for(auto const& p:front.patches)if(p.node->dynamic)return true;return false;}
    bool animated_map()const{for(auto const& p:front.patches)if(p.node->map_dynamic)return true;return false;}
    bool ready()const{return admitted&&front.width!=0;}
    void create(Id id,unsigned w,unsigned h,Format format){if(admitted){images[id]={w,h,format,{}};images[id].version=++serial;}}
    void snapshot(Id destination,Id source){images[destination]=images.at(source);images[destination].version=++serial;}
    void destroy(Id id){images.erase(id);} // committed versions retain their own source data
    void source(Id id,ID3D11Texture2D* texture,Sample sample={},bool immutable=false,bool map_source=false,std::uint64_t publication=0,std::uint64_t generation=0){
        if(!admitted)return;auto& p=images.at(id);auto n=node();n->area=extent(p);n->revision=++serial;
        output(*n,0,(sample||immutable)?Texture(texture):crop(texture,n->area));n->dynamic=bool(sample);n->map_dynamic=map_source&&n->dynamic;n->map_source=map_source;n->publication=map_source?publication:0;n->source_generation=generation;n->sample=std::move(sample);p.patches={{n->area,n,0}};p.version=++serial;
    }
    // Select a complete world version at an explicit composition boundary.
    // Later native writes to the source cannot change this selection. Fixed
    // UI writes recorded over destination remain in screen coordinates.
    // Native working textures are untouched: only the display recipe changes.
    void view(Id destination,Id source,std::shared_ptr<c3x_renderer::ZoomTransition> transition,Id words=0){
        if(!admitted)return;
        auto& target=images.at(destination);auto const& input=images.at(source);
        if(!transition||input.format!=Format::bgra32||target.format!=Format::bgra32||
           input.width!=target.width||input.height!=target.height)
            throw std::invalid_argument("retained world view extent or format");
        for(auto const& patch:input.patches)if(patch.node->view_dependent)
            throw std::invalid_argument("world view already transformed");
        auto n=node();n->area=extent(input);n->view=std::move(transition);n->dynamic=n->view_dependent=true;
        if(words){auto& native=images.at(words);
            if(native.format==Format::bgra32||native.width!=input.width||native.height!=input.height)
                throw std::invalid_argument("retained native view extent or format");
            n->view_native_format=native.format==Format::rgb565?2:1;write(native,n->area,n,1);
        }
        n->inputs[0]=read(source,extent(input));
        bool scene=false;n->projects_scene=projectable(n->inputs[0],scene,0)&&scene;
        for(auto const& patch:n->inputs[0].patches)n->map_dynamic|=patch.node->map_dynamic;
        write(target,n->area,n,0);
    }
    // Native UI versions retain their own pixels and dirty rectangles, but
    // every map underlay samples the latest complete world and map HUD. This
    // explicit live selection prevents stale labels without repainting or
    // erasing native panels that Civ III did not redraw this time.
    void select_world(Id words,Id detail,Id source_words,Id source_detail){
        if(!admitted)return;
        auto const& source=images.at(source_detail);
        auto bounds=extent(source);
        if(!world_selection||world_selection->area.right!=bounds.right||world_selection->area.bottom!=bounds.bottom)
            world_selection=node();
        auto n=world_selection;n->area=bounds;n->selected_world=n->dynamic=n->view_dependent=true;
        n->inputs[0]=images.at(source_words);n->inputs[1]=source;
        n->selected_format[0]=images.at(words).format;n->selected_format[1]=images.at(detail).format;
        invalidate_plan();
        n->map_dynamic=false;
        for(unsigned i=0;i<2;++i)for(auto const& patch:n->inputs[i].patches)n->map_dynamic|=patch.node->map_dynamic;
        write(images.at(words),bounds,n,0);write(images.at(detail),bounds,n,1);
    }
    void placed_batch(Id words,Id detail,std::vector<Placed> const& commands,
                      std::shared_ptr<c3x_renderer::ZoomTransition> placement){
        if(!admitted||commands.empty())return;
        auto n=node();n->area=extent(images.at(words));n->placement=std::move(placement);
        n->original[0]=words;n->original[1]=detail;
        n->inputs[0]=images.at(words);n->inputs[1]=images.at(detail);
        n->dynamic=n->view_dependent=true;
        for(auto const& placed:commands){
            if(placed.command.destination!=words&&placed.command.destination!=detail)
                throw std::invalid_argument("retained HUD destination outside pair");
            BatchOp draw;draw.placed=placed;auto const& c=placed.command;
            Id ids[6]={c.destination,c.source,c.background,c.detail,c.background_detail,c.program};
            for(unsigned i=0;i<6;++i)if(ids[i]&&ids[i]!=words&&ids[i]!=detail){
                draw.inputs[i]=images.at(ids[i]); // immutable operand version
                for(auto const& patch:draw.inputs[i].patches)n->map_dynamic|=patch.node->map_dynamic;
            }
            n->batch.push_back(std::move(draw));
        }
        // Match before replacing the old published pair, while its weak
        // preparation can still be locked. Before-images never enter the key.
        if(compiled_enabled)prepare_batch(*n);
        for(unsigned i=0;i<2;++i)for(auto const& patch:n->inputs[i].patches)n->map_dynamic|=patch.node->map_dynamic;
        write(images.at(words),n->area,n,0);write(images.at(detail),n->area,n,1);
    }
    void record(Command const& c,Direct direct={},std::shared_ptr<c3x_renderer::ZoomTransition> placement={},int anchor_x=0,int anchor_y=0){
        if(!admitted)return;auto target=images.find(c.destination);if(target==images.end())throw std::runtime_error("retained target missing");
        auto area=intersect(intersect(c.area,c.clip),extent(target->second));if(empty(area))return;
        // Equal-coordinate copies select an immutable source version. Keeping
        // its patches shares both samples and outputs instead of allocating a
        // full-canvas replay result for every map/screen/save transfer. Read
        // before replacing, including self-copies; later source writes cannot
        // alter this version. Shifted/converted/direct passes retain execution.
        if(!placement&&c.kind==Kind::copy&&!c.detail&&!direct.draw&&!direct.revision&&!direct.animated&&!direct.input_bytes&&
           c.source_x==c.area.left&&c.source_y==c.area.top&&images.at(c.source).format==target->second.format){
            auto selected=read(c.source,area);replace(target->second,area,selected.patches);return;
        }
        // JGL's unscaled opaque transfer is the same version selection for
        // both its native words and full-color companion. Replaying it as a
        // stretch shader rebuilt entire animated map canvases every frame.
        if(!placement&&c.kind==Kind::native_image&&c.color==65536&&!direct.draw&&!direct.revision&&!direct.animated&&!direct.input_bytes&&
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
        if(!placement&&c.kind==Kind::native_image&&c.color!=65536&&c.source!=c.destination&&
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
        reserve(direct.input_bytes,"direct-input");
        auto paint=area;
        if(placement){
            int dx=int(std::lround((anchor_x-int(target->second.width/2))*(c3x_renderer::ZoomTransition::maximum-1.)));
            int dy=int(std::lround((anchor_y-int(target->second.height/2))*(c3x_renderer::ZoomTransition::maximum-1.)));
            area=intersect({area.left+std::min(0,dx),area.top+std::min(0,dy),area.right+std::max(0,dx),area.bottom+std::max(0,dy)},extent(target->second));
        }
        auto n=node();direct_bytes+=direct.input_bytes;n->operation=true;n->area=area;n->command=c;n->recipe_clip=c.clip;n->command.clip=paint;n->direct=std::move(direct);n->dynamic=n->direct.animated||bool(placement);
        n->placement=std::move(placement);n->anchor_x=anchor_x;n->anchor_y=anchor_y;n->view_dependent=bool(n->placement);
        n->constant=!n->placement&&c.kind==Kind::fill&&!n->direct.draw&&!n->direct.revision&&!n->direct.animated;
        Id ids[6]={c.destination,c.source,c.background,c.detail,c.background_detail,c.program};
        bool opaque=!n->placement&&(c.kind==Kind::fill||c.kind==Kind::copy||c.kind==Kind::quantize||
            (c.kind==Kind::expand&&c.color==65536)||(c.kind==Kind::native_image&&c.color==65536));
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
        for(auto const& input:n->inputs)for(auto const& p:input.patches){n->dynamic|=p.node->dynamic;n->map_dynamic|=p.node->map_dynamic;n->view_dependent|=p.node->view_dependent;}
        n=reuse_recipe(n);
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
        prepare_front(ticks,frequency);
        auto image=assemble(front,ticks,frequency,0,{},true);
        Texture result;
        try{result=crop(replay.texture(image),extent(front));}
        catch(...){replay.recycle(image);throw;}replay.recycle(image);return result;
    }
    // Caller has supplied a completed native transfer. Rendering only touches
    // private scratch and the existing presenter's retained display.
    int draw(long long ticks,long long frequency,ID3D11RenderTargetView* target,ID3D11Texture2D* display,ID3D11Texture2D* buffer){
        work={};selected_view_scale=1.;
        if(!ready())return 0;++frame;
        prepare_front(ticks,frequency);
        auto& versions=pending_drawn_dependencies;versions.clear();
        for(auto const& part:front.patches){evaluate(part.node,ticks,frequency,0);versions.push_back(part.node->revision);}
        if(drawn_revision==front_revision&&versions==drawn_dependencies)return 2; // no new source sample
        Id image=0;Rect damage=extent(front);
        if(!assemble_front(image,damage))image=assemble(front,ticks,frequency,0,{},true);
        bool ok=false;
        try{ok=empty(damage)||replay.display(image,target,front.width,front.height,damage);}
        catch(...){assembled_revision=0;replay.recycle(image);throw;}replay.recycle(image);
        if(!ok)assembled_revision=0;
        // The caller publishes with Present or a keyed release followed by
        // Flush. Keep the retained copy in that same submission batch.
        if(ok){context->CopyResource(buffer,display);drawn_revision=front_revision;drawn_dependencies.swap(versions);}return ok?1:0;
    }
    double view_scale()const{return selected_view_scale;}
    Work last_work()const{return work;}
    // Native completed-front identity; visual clock samples do not advance it.
    std::uint64_t committed_revision()const{return front_revision;}
    template<class Report> void describe(Report report,bool all_images=false)const{
        std::vector<Node const*> ordered;
        auto add=[&](Node const* n){if(std::find(ordered.begin(),ordered.end(),n)==ordered.end()&&ordered.size()<(all_images?1024u:256u))ordered.push_back(n);};
        for(auto const& p:front.patches)add(p.node.get());
        if(all_images)for(auto const& image:images)for(auto const& p:image.second.patches)add(p.node.get());
        for(unsigned i=0;i<ordered.size();++i){
            for(auto const& input:ordered[i]->inputs)for(auto const& p:input.patches)add(p.node.get());
            if(ordered[i]->projected)add(ordered[i]->projected.get());
            for(auto const& draw:ordered[i]->batch)for(auto const& input:draw.inputs)for(auto const& p:input.patches)add(p.node.get());
        }
        for(unsigned i=0;i<ordered.size();++i){auto n=ordered[i];auto const& c=n->command;char text[384];
            std::snprintf(text,sizeof(text),"id=%u bytes=%llu kind=%d operation=%u dynamic=%u map_dynamic=%u retired=%u sampled=%u projected=%u view=%u seen=%llu batch=%u area=%d,%d,%d,%d source=%d,%d,%d,%d color=%u",
                i,n->bytes[0]+n->bytes[1]+n->direct.input_bytes,int(c.kind),unsigned(n->operation),unsigned(n->dynamic),unsigned(n->map_dynamic),unsigned(n->retired),unsigned(bool(n->sample)),unsigned(n->projects_scene),unsigned(bool(n->view)),n->seen,
                unsigned(n->batch.size()),n->area.left,n->area.top,n->area.right,n->area.bottom,c.source_x,c.source_y,c.source_width,c.source_height,c.color);
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
