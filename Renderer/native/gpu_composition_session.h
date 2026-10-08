#pragma once
#include "gpu_frame_api.h"
#include "gpu_image_compositor.h"
#include "retained_composition.h"
#include "pan_transition.h"
#include <array>
namespace c3x_gpu_images {
// Lives exclusively on RendererWorker, with its existing immediate context.
// A map is immutable; native composition writes separately owned images.
class Session {
    static constexpr unsigned live_image_budget=256u*1024u*1024u;
    ID3D11Device* device;ID3D11DeviceContext* context;
    CompositionStorage storage;
    Compositor gpu;RetainedComposition layers;Id map=0;std::int64_t ticket=0,identity=0;std::uint64_t readbacks=0;
    Id resident_unit=0;ID3D11Texture2D* resident_unit_texture=nullptr;
    bool map_animation_expected=false;
    std::shared_ptr<c3x_renderer::ZoomTransition> zoom=std::make_shared<c3x_renderer::ZoomTransition>();
    // A published camera step is shown as an image-space slide once its
    // native world arrives (see PanTransition and RetainedComposition::set_pan).
    c3x_renderer::PanTransition pan;
    ComPtr<ID3D11Texture2D> pan_under[2];
    int pan_step_x=0,pan_step_y=0;bool pan_pending=false,pan_armed=false;int presented_pan_packed=0;
    void start_pan(){
        if(!pan_armed)return;pan_armed=false;
        LARGE_INTEGER now={},frequency={};QueryPerformanceCounter(&now);QueryPerformanceFrequency(&frequency);
        pan.begin(pan_step_x,pan_step_y,now.QuadPart,frequency.QuadPart);
    }
    // Private retained IDs do not cross the image transport or own exact GPU
    // working textures. Each frame starts from the complete canonical map.
    static constexpr Id world_words=Id(1)<<63,world_detail=world_words+1,world_view_words=world_words+2,world_view_detail=world_words+3;
    unsigned world_width=0,world_height=0;Id world_destination=0;
    double rendered_zoom=1.;
    Id fixed_words=0,fixed_detail=0;
    std::vector<Command> fixed_shadows;
    struct Hud {Id canvas=0,detail=0;unsigned identity=0;int x=0,y=0;unsigned key=0,key_detail=0;int layout_x=0,layout_y=0;int unit_id=-1;
        std::vector<Command> draws;std::vector<Id> snapshots;};
    std::vector<Hud> hud;
    std::shared_ptr<c3x_renderer::render_core::UnitHudAnchors> unit_anchors;
    Id hud_canvas=0,hud_detail=0,next_snapshot=world_detail+4096;
    void erase_hud(std::size_t at){for(auto id:hud[at].snapshots)layers.destroy(id);hud.erase(hud.begin()+at);}
    // Native image execution cost, summarized every 2 s at trace level 2.
    struct ExecuteProfile {
        LARGE_INTEGER reported{},frequency{};std::uint64_t calls=0,commands=0,uploads=0,upload_pixels=0;
        double submit_ms=0,record_ms=0,resource_ms=0,total_ms=0,max_ms=0;bool enabled=false;
        ExecuteProfile(){char value[4]={};enabled=GetEnvironmentVariableA("C3X_RENDERER_TRACE",value,sizeof(value))==1&&value[0]=='2';
            QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&reported);}
        double since(LARGE_INTEGER& mark)const{LARGE_INTEGER now={};QueryPerformanceCounter(&now);
            double ms=1000.*double(now.QuadPart-mark.QuadPart)/double(frequency.QuadPart);mark=now;return ms;}
        void report(){
            LARGE_INTEGER now={};QueryPerformanceCounter(&now);
            if(now.QuadPart-reported.QuadPart<2*frequency.QuadPart)return;
            char line[320];std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=native-image-execution calls=%llu commands=%llu uploads=%llu upload_pixels=%llu total_ms=%.1f submit_ms=%.1f record_ms=%.1f resource_ms=%.1f max_ms=%.2f window_ms=%.0f\n",
                (unsigned long long)calls,(unsigned long long)commands,(unsigned long long)uploads,(unsigned long long)upload_pixels,
                total_ms,submit_ms,record_ms,resource_ms,max_ms,1000.*double(now.QuadPart-reported.QuadPart)/double(frequency.QuadPart));
            OutputDebugStringA(line);auto keep=enabled;auto rate=frequency;*this=ExecuteProfile{};enabled=keep;frequency=rate;reported=now;
        }
    } execute_profile;
    void record(Command const& command){
        if(!layers.accepting())return;
        if(hud_canvas&&(command.destination==hud_canvas||command.destination==hud_detail)){
            auto c=command;auto& item=hud.back();
            if(c.kind==Kind::fill&&c.destination==item.canvas&&c.color==item.key)return;
            if(c.kind==Kind::fill&&c.destination==item.detail&&c.color==item.key_detail)return;
            auto snapshot=[&](Id& id){if(!id||id==hud_canvas||id==hud_detail)return;
                auto next=++next_snapshot;layers.snapshot(next,id);item.snapshots.push_back(next);id=next;};
            snapshot(c.source);snapshot(c.program);
            if(c.kind==Kind::native_text)snapshot(c.background);
            if(c.kind==Kind::native_image)snapshot(c.background_detail);
            item.draws.push_back(c);return;
        }
        // A canvas rebuild retires its old labels. Each subsequent lexical
        // draw scope supplies the current visible item and its native anchor.
        if(command.kind==Kind::fill||command.kind==Kind::copy||command.kind==Kind::quantize||
           (command.kind==Kind::native_image&&command.color==65536)){
            auto erase=intersection(command.area,command.clip);
            for(std::size_t at=hud.size();at-->0;)if(hud[at].canvas==command.destination){
                auto& item=hud[at];std::vector<Command> remaining;
                for(auto c:item.draws){auto a=intersection(c.area,c.clip),cut=intersection(a,erase);
                    if(cut.left>=cut.right||cut.top>=cut.bottom){remaining.push_back(c);continue;}
                    Rect pieces[]={{a.left,a.top,a.right,cut.top},{a.left,cut.bottom,a.right,a.bottom},
                        {a.left,cut.top,cut.left,cut.bottom},{cut.right,cut.top,a.right,cut.bottom}};
                    for(auto r:pieces)if(r.left<r.right&&r.top<r.bottom){c.clip=r;remaining.push_back(c);}
                }
                item.draws=std::move(remaining);if(item.draws.empty())erase_hud(at);
            }
        }
        if(fixed_words&&(command.destination==fixed_words||command.destination==fixed_detail)){
            // Notification text is in the GUI form, but Civ III writes its
            // shadow into Units_Control. Keep that lexical scope out of the
            // canonical world and replay each shadow over the displayed world.
            if(command.kind==Kind::native_lookup){
                auto copy=command;copy.source=world_detail+1024+fixed_shadows.size();
                layers.snapshot(copy.source,command.source);fixed_shadows.push_back(copy);
            }else if(command.kind!=Kind::fill)throw std::runtime_error("unexpected fixed notification canvas operation");
            return;
        }
        layers.record(command);
    }
    void world(Command const& input){
        if(!layers.accepting())return;
        auto c=input;unsigned w=unsigned(c.area.right),h=unsigned(c.area.bottom);
        if(!c.destination||!c.detail||!c.source||c.area.left||c.area.top||
           !w||!h||w>2240||h>1260)throw std::runtime_error("invalid world composition boundary");
        if(c.kind==Kind::world_begin){
            auto format=gpu.format(c.destination);
            if(world_width!=w||world_height!=h){
                layers.destroy(world_words);layers.destroy(world_detail);layers.destroy(world_view_words);layers.destroy(world_view_detail);
                layers.create(world_words,w,h,format);layers.create(world_detail,w,h,Format::bgra32);
                layers.create(world_view_words,w,h,format);layers.create(world_view_detail,w,h,Format::bgra32);
                world_width=w;world_height=h;
            }
            world_destination=c.destination;
        }else if(world_destination!=c.destination||world_width!=w||world_height!=h)
            return; // Loading can compose the unit form before its map form.
        c.kind=Kind::native_image;c.destination=world_words;c.detail=world_detail;
        layers.record(c);
        if(input.kind==Kind::world_end){
            // The displayed world planes still belong to the previous camera.
            if(pan_pending){pan_pending=false;pan_armed=!zoom->moving()&&layers.copy_selected_world(pan_under);}
            layers.view(world_view_detail,world_detail,zoom,world_view_words);
            std::vector<RetainedComposition::Placed> placed;
            for(auto const& item:hud)for(auto draw:item.draws){
                draw.destination=draw.destination==item.detail?world_view_detail:world_view_words;
                if(draw.detail)draw.detail=world_view_detail;
                if(draw.source==item.canvas)draw.source=world_view_words;
                else if(draw.source==item.detail)draw.source=world_view_detail;
                if(draw.background&&draw.kind!=Kind::native_text)draw.background=world_view_words;
                if(draw.background_detail&&draw.kind!=Kind::native_image)draw.background_detail=world_view_detail;
                draw.area={draw.area.left-item.layout_x,draw.area.top-item.layout_y,
                    draw.area.right-item.layout_x,draw.area.bottom-item.layout_y};
                draw.clip={draw.clip.left-item.layout_x,draw.clip.top-item.layout_y,
                    draw.clip.right-item.layout_x,draw.clip.bottom-item.layout_y};
                placed.push_back({draw,item.x,item.y,item.unit_id>=0?unit_anchors:nullptr,item.unit_id});
            }
            layers.placed_batch(world_view_words,world_view_detail,placed,zoom);
            for(auto shadow:fixed_shadows){
                shadow.destination=shadow.background=world_view_words;
                shadow.detail=shadow.background_detail=world_view_detail;
                layers.record(shadow);
            }
            layers.select_world(input.destination,input.detail,world_view_words,world_view_detail);
        }
    }
public:
    Session(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c),gpu(d,c,live_image_budget,true),layers(d,c){gpu.share_storage(storage);layers.share_storage(storage);}
    // Existing live/replay owners retain their shared programs before a real
    // view creates its targets; no map identity or native image is admitted.
    bool prepare_assets(std::function<bool()> cancelled={}){
        return gpu.prepare_assets(cancelled) && layers.prepare_assets(cancelled);
    }
    // Eight fullscreen packed/full-color native pairs and old/new immutable
    // maps require about 194 MiB at 2240x1260. Bound live images at 256 MiB,
    // including small UI sources; retained replay has its separate budget.
    bool publish_source(ID3D11Texture2D* texture,std::int64_t serial,int x,int y,int width,int height,
                        RetainedComposition::Sample sample,bool shared_source){
        if(!texture||serial<=ticket)return false;
        D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
        if(!width)width=int(d.Width);if(!height)height=int(d.Height);
        if(shared_source && (x||y||width!=int(d.Width)||height!=int(d.Height)))return false;
        auto next=shared_source?gpu.attach_source(texture):gpu.create(width,height,Format::bgra32);
        if(!next){char message[224];auto counts=gpu.stats();
            sprintf_s(message,"[C3X renderer] stage=map-publication-rejected reason=canvas-admission width=%d height=%d resident_bytes=%llu cap_bytes=%u\n",
                width,height,counts.resident_bytes,live_image_budget);OutputDebugStringA(message);return false;}

        if(!shared_source&&!gpu.import_bgra(next,texture,x,y)){gpu.destroy(next);return false;}
        // Admission failure leaves the previous immutable map and UI handles
        // usable. Publish the new identity only after its import succeeds.
        if(map){layers.destroy(map);gpu.destroy(map);}map=next;map_animation_expected=bool(sample);
        try{
            // A rejected history is rebuilt only at fresh authoritative map
            // demand. Current native images become immutable static inputs;
            // subsequent map writes restore their dynamic dependencies.
            if(!layers.accepting()){
                layers.clear();world_width=world_height=0;world_destination=0;
                hud.clear();hud_canvas=hud_detail=0;fixed_shadows.clear();
                gpu.visit_images([&](Id id,unsigned w,unsigned h,Format format,ID3D11Texture2D* source){
                    layers.create(id,w,h,format);layers.source(id,source);
                });
            }
            unit_anchors=sample.unit_anchors;
            auto generation=sample.source_generation;
            layers.create(map,width,height,Format::bgra32);layers.source(map,gpu.texture(map),std::move(sample),true,true,std::uint64_t(serial),generation);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        ticket=serial;if(!identity)identity=serial;return true;
    }
    bool publish(ID3D11Texture2D* texture,std::int64_t serial,int x=0,int y=0,int width=0,int height=0,RetainedComposition::Sample sample={}){
        return publish_source(texture,serial,x,y,width,height,std::move(sample),false);
    }
    // A map produced by the x64 core is immutable R32_UINT scene data. The
    // x86 native composition owner borrows its shared allocation directly;
    // neither a CPU readback nor a second full-size map texture is required.
    bool publish_shared(ID3D11Texture2D* texture,std::int64_t serial,RetainedComposition::Sample sample={}){
        return publish_source(texture,serial,0,0,0,0,std::move(sample),true);
    }
    std::int64_t session_identity()const{return identity;}
    Id map_image()const{return map;}
#ifdef C3X_HELPER_TRIAL
    ID3D11Texture2D* map_texture(){return gpu.texture(map);}
    bool display_map_to(ID3D11RenderTargetView* target,unsigned width,unsigned height){
        return gpu.display(map,target,width,height,{0,0,int(width),int(height)});
    }
#endif
    double visual_scale()const{return layers.view_scale();}
    unsigned presented_zoom()const{return unsigned(zoom->last_presented()*65536.+.5);}
    void did_present(){zoom->did_present(rendered_zoom);}
    std::uint64_t upload_count()const{return gpu.stats().uploads;}
    std::int64_t current_ticket()const{return ticket;}
    // Publish an immutable native screen version. The independent cadence
    // samples it later; accepting a UI transfer does not render another map.
    // Partial transfers retain exactly the previously committed outside area.
    // Screen-pixel camera step (new minus previous) for the next native world.
    void camera_step(int dx,int dy){pan_pending=dx||dy;pan_step_x=dx;pan_step_y=dy;if(!pan_pending)pan_armed=false;}
    bool panning()const{return pan.moving();}
    int presented_pan()const{return presented_pan_packed;}
    bool commit_display(std::int64_t requested,Id image,unsigned w,unsigned h,Rect area){
        if(requested!=ticket||!gpu.displayable(image,w,h))return false;
        layers.commit(image,area);start_pan();
        // A discarded optional history still has valid native GPU canvases.
        // Keep transport alive while visual_ready() requests a fresh map;
        // rejecting this transfer would prevent that recovery from arriving.
        return !layers.accepting()||layers.ready();
    }
    bool display_to(std::int64_t requested,Id image,ID3D11RenderTargetView* target,ID3D11Texture2D* retained,ID3D11Texture2D* buffer,unsigned w,unsigned h,Rect area,long long ticks=0,long long frequency=0,std::array<LONGLONG,4>* phase_ticks=nullptr,std::array<LONGLONG,8>* draw_ticks=nullptr){
        if(requested!=ticket||!target||!retained||!buffer||!gpu.displayable(image,w,h))return false;
        LARGE_INTEGER mark={};
        if(phase_ticks)QueryPerformanceCounter(&mark);
        try{layers.commit(image,area);start_pan();}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        if(phase_ticks){LARGE_INTEGER next={};QueryPerformanceCounter(&next);(*phase_ticks)[0]=next.QuadPart-mark.QuadPart;mark=next;}
        // Native draws update the scene recipe, not its visual time. Both native
        // transfers and autonomous frames sample that same committed recipe.
        // Otherwise every native unit/UI transfer restores the old map sample.
        // A retired camera becomes static, but its completed retained image
        // still contains the latest unit poses. Drawing the original native
        // canvas here would rewind them until the next map is adopted.
        if(frequency>0 && layers.ready()){
            if(visual_frame(ticks,frequency,target,retained,buffer)!=0)return true;
            // Optional clock sampling may run out of scratch while the completed
            // native transfer is still valid. Discarded history can be rebuilt
            // at the next map publication; do not reject current native pixels.
            if(FAILED(device->GetDeviceRemovedReason()))return false;
        }
        if(!gpu.display(image,target,w,h,area,draw_ticks))return false;
        if(phase_ticks){LARGE_INTEGER next={};QueryPerformanceCounter(&next);(*phase_ticks)[1]=next.QuadPart-mark.QuadPart;mark=next;}
        context->CopyResource(buffer,retained);
        if(phase_ticks){LARGE_INTEGER next={};QueryPerformanceCounter(&next);(*phase_ticks)[2]=next.QuadPart-mark.QuadPart;mark=next;}
        // Submission belongs to the caller's Present/keyed-surface boundary.
        if(phase_ticks){LARGE_INTEGER next={};QueryPerformanceCounter(&next);(*phase_ticks)[3]=next.QuadPart-mark.QuadPart;}
        return true;
    }
    int compose_resident_unit(c3x_renderer_gpu_unit_v1 const& request,ID3D11Texture2D* texture,unsigned width,unsigned height,int x,int y,RetainedComposition::Sample sample={}){
        if(request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(request.destination==std::int64_t(map)||request.detail==std::int64_t(map))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(!texture||!width||!height||width>1024||height>1024)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(texture!=resident_unit_texture){
            if(resident_unit){layers.destroy(resident_unit);gpu.destroy(resident_unit);}resident_unit=0;resident_unit_texture=nullptr;
            resident_unit=gpu.attach_source(texture);if(!resident_unit)return C3X_RENDERER_RESULT_BAD_ARGUMENT;resident_unit_texture=texture;
        }else gpu.record_external(resident_unit);
        try{layers.create(resident_unit,width,height,Format::bgra32);layers.source(resident_unit,texture,std::move(sample),true);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        Command draw={Kind::unit_over,Id(request.destination),resident_unit,{x,y,x+int(width),y+int(height)},
            {request.clip[0],request.clip[1],request.clip[2],request.clip[3]},0,0,0,Id(request.background),Id(request.detail),Id(request.background_detail)};
        bool ok=gpu.submit(&draw,1);if(ok)try{layers.record(draw);}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
    // Static route/grid art has no dependency on the animated map beneath it.
    // Snapshot its small packed texture once, then replay only the blend when
    // that underlay changes. The retained source owns its version and budget.
    int draw_overlay(c3x_renderer_gpu_unit_v1 const& request,ID3D11Texture2D* texture,unsigned width,unsigned height,int x,int y){
        if(request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(request.destination==std::int64_t(map)||request.detail==std::int64_t(map)||!texture)
            return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        auto source=gpu.attach_source(texture);if(!source)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        bool ok=false;
        try{
            layers.create(source,width,height,Format::bgra32);layers.source(source,texture);
            Command draw={Kind::unit_over,Id(request.destination),source,{x,y,x+int(width),y+int(height)},
                {request.clip[0],request.clip[1],request.clip[2],request.clip[3]},0,0,0,
                Id(request.background),Id(request.detail),Id(request.background_detail)};
            ok=gpu.submit(&draw,1);if(ok)record(draw);
        }catch(...){layers.destroy(source);gpu.destroy(source);throw;}
        layers.destroy(source);gpu.destroy(source);
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
    int draw_dynamic(c3x_renderer_gpu_unit_v1 const& request,unsigned width,unsigned height,int x,int y,RetainedComposition::Direct operation){
        if(request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(request.destination==std::int64_t(map)||request.detail==std::int64_t(map)||!operation.draw)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        Command draw={Kind::unit_over,Id(request.destination),0,{x,y,x+int(width),y+int(height)},
            {request.clip[0],request.clip[1],request.clip[2],request.clip[3]},0,0,0,Id(request.background),Id(request.detail),Id(request.background_detail)};
        bool ok=operation.draw(gpu,draw);
        if(ok)try{layers.record(draw,std::move(operation));}catch(std::exception const& e){OutputDebugStringA(e.what());layers.discard();}
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
    std::uint64_t committed_revision()const{return layers.committed_revision();}
    std::uint64_t visual_sample_allocations()const{return layers.sampling_allocations();}
    std::uint64_t visual_sample_imports()const{return layers.sampling_imports();}
    std::uint64_t visual_bytes()const{return layers.bytes();}
    std::size_t allocation_bytes()const{return std::size_t(layers.allocation_bytes());}
    std::uint64_t allocation_peak()const{return storage.peak();}
    std::size_t visual_nodes()const{return layers.node_count();}
    std::size_t visual_sources()const{return layers.sampled_sources();}
    RetainedComposition::Work visual_work()const{return layers.last_work();}
    RetainedComposition::RecipeReuse visual_recipe_reuse()const{return layers.recipe_reuse();}
    RetainedComposition::PlanReuse visual_plan_reuse()const{return layers.plan_reuse();}
    Counts visual_gpu_counts()const{return layers.replay_stats();}
    std::pair<std::uint64_t,std::uint64_t> visual_publication()const{return layers.front_publication();}
    template<class Report> void describe_visual(Report report)const{layers.describe(report);}
    // Correct static pixels alone do not certify ambient delivery. A CPU
    // snapshot can sever map samples while unit animation remains reachable.
    // Let the existing native recovery demand run until map writes restore it.
    bool visual_ready()const{return layers.ready()&&(!map_animation_expected||layers.animated_map());}
    bool visual_active()const{return layers.ready()&&layers.animated();}
    void stop_visuals(){layers.uncommit();}
    int visual_frame(long long ticks,long long frequency,ID3D11RenderTargetView* target,ID3D11Texture2D* display,ID3D11Texture2D* buffer){
        auto step=pan.sample(ticks,frequency);
        if(zoom->moving()){pan.cancel();step={};}
        if(!step.active){pan_under[0].Reset();pan_under[1].Reset();}
        layers.set_pan(step.x,step.y,step.under_x,step.under_y,step.active?pan_under:nullptr);
        try{auto result=layers.draw(ticks,frequency,target,display,buffer);
            if(result==1)presented_pan_packed=step.active?int((unsigned(step.x)&0xffffu)|(unsigned(step.y)<<16)):0;
            c3x_recording::event(c3x_recording::visual,0,[&](auto& b){using namespace c3x_recording;u64(b,std::uint64_t(ticks));u64(b,std::uint64_t(frequency));u32(b,unsigned(result));u64(b,layers.bytes());u64(b,layers.node_count());u64(b,layers.sampled_sources());u32(b,visual_ready());});
            if(result==1)rendered_zoom=layers.view_scale();
            return result;}catch(std::exception const& e){
            // A failed recipe cannot produce a frame. Release its outputs now;
            // the last completed display stays intact, and the next native map
            // rebuilds retained history from authoritative current images.
            MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);GlobalMemoryStatusEx(&memory);
            char status[224];sprintf_s(status,"[C3X renderer] stage=visual-failure-memory device_reason=0x%08lx available_virtual=%llu available_pagefile=%llu\n",
                device->GetDeviceRemovedReason(),memory.ullAvailVirtual,memory.ullAvailPageFile);
            OutputDebugStringA(status);OutputDebugStringA(e.what());OutputDebugStringA("\n");
            layers.describe([](char const* line){OutputDebugStringA("[C3X renderer] stage=failed-retained-node ");OutputDebugStringA(line);OutputDebugStringA("\n");},true);
            layers.discard();return false;}
    }
    int execute(c3x_renderer_gpu_images_v1 const& request,std::vector<Command> const& commands,
                std::vector<unsigned> const& pixels,c3x_renderer_gpu_result_v1& result,std::vector<unsigned>& output){
        output.clear();result={sizeof(result)};
        if(!ticket||request.ticket!=ticket)return C3X_RENDERER_RESULT_SUPERSEDED;
        auto& profile=execute_profile;LARGE_INTEGER began={},mark={};
        if(profile.enabled){QueryPerformanceCounter(&began);mark=began;++profile.calls;profile.commands+=commands.size();}
        // Draws and the immediately following resource boundary share one
        // owner handoff. Submission order and explicit CPU readback stay exact.
        if(!commands.empty()){
            for(auto const& c:commands)if(c.destination==map||c.detail==map)return request.action==C3X_GPU_SUBMIT?C3X_RENDERER_RESULT_BAD_ARGUMENT:C3X_RENDERER_RESULT_ERROR;
            for(std::size_t at=0;at<commands.size();){
                auto const& c=commands[at];
                if(c.kind>=Kind::world_begin){
                    if(c.kind==Kind::zoom_target){LARGE_INTEGER now={},frequency={};
                        QueryPerformanceCounter(&now);QueryPerformanceFrequency(&frequency);
                        zoom->target(double(c.color)/65536.,now.QuadPart,frequency.QuadPart);
                    }else if(c.kind==Kind::world_begin||c.kind==Kind::world_end)world(c);
                    else if(c.kind==Kind::hud_begin){
                        for(std::size_t i=hud.size();i-->0;)if(hud[i].canvas==c.destination&&hud[i].identity==c.color&&hud[i].unit_id==c.source_height-1&&
                            (c.source_height||c.color||(hud[i].x==c.source_x&&hud[i].y==c.source_y)))erase_hud(i);
                        hud.push_back({c.destination,c.detail,c.color,c.source_x,c.source_y,unsigned(c.source_width)});
                        auto& item=hud.back();item.unit_id=c.source_height-1;item.layout_x=c.area.left;item.layout_y=c.area.top;unsigned k=item.key,r,g,b=((k&31)<<3)|((k&31)>>2);
                        if(gpu.format(c.destination)==Format::rgb565){g=((k>>3)&252)|((k>>9)&3);r=((k>>8)&248)|((k>>13)&7);}
                        else{g=((k>>2)&248)|((k>>7)&7);r=((k>>7)&248)|((k>>12)&7);}
                        item.key_detail=0xff000000u|(r<<16)|(g<<8)|b;
                        hud_canvas=c.destination;hud_detail=c.detail;
                    }else if(c.kind==Kind::hud_end){hud_canvas=hud_detail=0;}
                    else if(c.kind==Kind::fixed_ui_begin){
                        for(auto const& shadow:fixed_shadows)layers.destroy(shadow.source);
                        fixed_shadows.clear();fixed_words=c.destination;fixed_detail=c.detail;
                    }else if(c.kind==Kind::fixed_ui_end){fixed_words=fixed_detail=0;}
                    else return C3X_RENDERER_RESULT_BAD_ARGUMENT;
                    ++at;continue;
                }
                auto end=at+1;while(end<commands.size()&&commands[end].kind<Kind::world_begin)++end;
                if(profile.enabled)profile.since(mark);
                if(!gpu.submit(commands.data()+at,end-at))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
                if(profile.enabled)profile.submit_ms+=profile.since(mark);
                for(;at<end;++at)try{record(commands[at]);}catch(std::exception const& e){
                    OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}
                if(profile.enabled)profile.record_ms+=profile.since(mark);
            }
        }
        bool ok=false;Id image=Id(request.image);
        if(profile.enabled){profile.since(mark);if(request.action==C3X_GPU_UPLOAD){++profile.uploads;profile.upload_pixels+=pixels.size();}}
        if(request.action==C3X_GPU_CREATE){image=gpu.create(request.width,request.height,request.format==C3X_GPU_RGB555?Format::rgb555:request.format==C3X_GPU_RGB565?Format::rgb565:Format::bgra32);ok=image!=0;if(ok)layers.create(image,request.width,request.height,request.format==C3X_GPU_RGB555?Format::rgb555:request.format==C3X_GPU_RGB565?Format::rgb565:Format::bgra32);}
        else if(request.action==C3X_GPU_UPLOAD){auto before=gpu.stats().uploads;ok=image!=map&&request.revision>0&&gpu.upload(image,request.revision,pixels.data(),pixels.size());
            if(ok&&gpu.stats().uploads!=before)try{layers.source(image,gpu.texture(image));}catch(std::exception const& e){OutputDebugStringA("[C3X renderer] retained admission: ");OutputDebugStringA(e.what());OutputDebugStringA("\n");layers.discard();}}
        else if(request.action==C3X_GPU_DESTROY){ok=image!=map&&gpu.destroy(image);if(ok)layers.destroy(image);}
        else if(request.action==C3X_GPU_SUBMIT)ok=true;
        else if(request.action==C3X_GPU_READBACK){
            auto texture=gpu.texture(image);if(!texture)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
            if(std::uint64_t(d.Width)*d.Height>request.pixel_count)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            output.resize(std::size_t(d.Width)*d.Height);
            d.BindFlags=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
            D3D11_MAPPED_SUBRESOURCE data={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&data));
            for(unsigned y=0;y<d.Height;++y)std::memcpy(output.data()+std::size_t(y)*d.Width,static_cast<char*>(data.pData)+std::size_t(y)*data.RowPitch,d.Width*4);
            context->Unmap(stage.Get(),0);++readbacks;ok=true;gpu.record_readback(image,output.data(),output.size());
        }
        if(profile.enabled){profile.resource_ms+=profile.since(mark);auto total=profile.since(began);
            profile.total_ms+=total;profile.max_ms=std::max(profile.max_ms,total);profile.report();}
        auto counts=gpu.stats();result.image=std::int64_t(image);result.pixel_count=unsigned(output.size());
        result.resident_bytes=std::int64_t(counts.resident_bytes);result.uploads=std::int64_t(counts.uploads);
        result.commands=std::int64_t(counts.commands);result.readbacks=std::int64_t(readbacks);
        return ok?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_BAD_ARGUMENT;
    }
};
}
