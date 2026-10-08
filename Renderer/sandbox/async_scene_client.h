#pragma once
#include "async_publication.h"
#include "../native/remote_scene_output.h"
#include "../native/ordered_image_batch.h"
#include <map>
#include <memory>

namespace c3x_remote_scene {
// Caller-visible identities are reserved locally. The transport thread resolves
// them in command order, including creates that have not reached x64 yet.
// A ready camera is only inspected by the consumer. Adoption is a later ordered
// command, so old image operations cannot straddle a premature map retirement.
template<class Transport>class AsyncSceneClient {
    Transport transport;
    bool enabled;
    c3x_async::Publication publication;
    using Id=c3x_renderer_i64;
    Id next_image=0,next_camera=0,session=1;
    std::map<Id,Id> image_ids,ticket_ids; // transport thread only
    Id remote_camera=0,worker_camera=0,worker_map=0;
    std::atomic<unsigned> adopted_cameras{0};
    std::function<void(Id,Id,c3x_renderer_gpu_camera_view_v1 const&)> camera_adoption_observer;
    struct Camera {
        Id ticket=0;
        std::atomic<Id> remote_ticket{0};
        std::mutex mutex;
        bool query=false,adopted=false; // local reservation; execution counted separately
        int code=C3X_RENDERER_RESULT_PENDING;
        std::unique_ptr<CameraOutput> ready;
    };
    std::shared_ptr<Camera> camera;
    std::shared_ptr<c3x_inputs::Frame> published_frame;
    c3x_renderer_camera_identity_v1 published_identity={};
    std::unique_ptr<CameraOutput> displayed;
    std::atomic<int> policy{0};
    // Last posted ambient permission (0/1); a reset forgets it (transport thread).
    std::atomic<unsigned> requested_policy{~0u};
    struct Page {
        std::mutex mutex;bool query=false,ready=false;int code=C3X_RENDERER_RESULT_PENDING;
        c3x_renderer_world_page_v1 value={};
    } world_page;
    struct Status {
        std::mutex mutex;bool query=false;
        c3x_renderer_world_status_v1 value={sizeof(value)};
    } world_progress;
    static void require_result(int code,char const* operation){
        if(code!=C3X_RENDERER_RESULT_OK)
            throw std::runtime_error(std::string("asynchronous renderer ")+operation+" failed: "+std::to_string(code));
    }
    static void accept_state(int code,char const* operation){
        if(code!=C3X_RENDERER_RESULT_SUPERSEDED)require_result(code,operation);
    }
    Id image(Id local)const{
        if(!local)return 0;
        auto found=image_ids.find(local);
        if(found==image_ids.end())throw std::runtime_error("asynchronous image used outside its lifetime");
        return found->second;
    }
    Id ticket(Id local)const{
        auto found=ticket_ids.find(local);
        if(found==ticket_ids.end())throw std::runtime_error("asynchronous map used before adoption");
        return found->second;
    }
    void target(c3x_renderer_gpu_unit_v1& value)const{
        if(!value.ticket)return; // scene observation without a composed canvas
        value.ticket=ticket(value.ticket);value.destination=image(value.destination);
        value.background=image(value.background);value.detail=image(value.detail);
        value.background_detail=image(value.background_detail);
    }
    static bool same_camera_source(c3x_renderer_frame_v1 const& a,c3x_renderer_camera_identity_v1 const& ai,
                                   c3x_renderer_frame_v1 const& b,c3x_renderer_camera_identity_v1 const& bi){
        auto left=a,right=b;left.tiles=right.tiles=nullptr;left.world_topology=right.world_topology=nullptr;
        left.presentation_time_ticks=right.presentation_time_ticks=0;
        left.visible_animation_count=right.visible_animation_count=0;
        return !std::memcmp(&ai,&bi,sizeof(ai))&&!std::memcmp(&left,&right,sizeof(left))&&
            (!a.tile_count||!std::memcmp(a.tiles,b.tiles,std::size_t(a.tile_count)*sizeof(*a.tiles)))&&
            (!a.world_topology_count||!std::memcmp(a.world_topology,b.world_topology,std::size_t(a.world_topology_count)*sizeof(*a.world_topology)));
    }
    void execute_images(ImageBatch& batch){
        std::vector<ImageBatch::Operation> mapped;std::map<Id,bool> created_here;
        auto resolve=[&](Id local){return created_here.count(local)?-local:image(local);};
        for(auto const& operation:batch.operations){
            mapped.push_back(operation);auto& packet=mapped.back().image;auto& value=packet.value;
            value.ticket=ticket(value.ticket);value.image=resolve(value.image);
            for(auto& draw:packet.commands){draw.destination=resolve(draw.destination);draw.source=resolve(draw.source);
                draw.background=resolve(draw.background);draw.detail=resolve(draw.detail);
                draw.background_detail=resolve(draw.background_detail);draw.program=resolve(draw.program);}
            packet.bind();if(operation.created)created_here[operation.created]=true;
        }
        std::vector<ImageBatch::Reply> replies;require_result(transport.images_batch(mapped,replies),"image-batch-transport");
        for(std::size_t i=0;i<replies.size();++i){auto const& operation=batch.operations[i];
            require_result(replies[i].code,"image-batch-operation");
            if(operation.created)image_ids[operation.created]=replies[i].value.image;
            if(operation.image.value.action==C3X_GPU_DESTROY)image_ids.erase(operation.image.value.image);
        }
        c3x_inputs::require(replies.size()==batch.operations.size(),"image batch lost reliable suffix");
    }
    template<class Work>int post(std::size_t bytes,Work work,unsigned replace_key=0,char const* label=nullptr,
                                 bool (*passes)(char const*)=nullptr){
        bool accepted=publication.post(bytes,std::move(work),replace_key,label?label:"state",1,passes);
        transport.publication_pressure(publication.status().records);
        return accepted?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_DEVICE_ERROR;
    }
    // Copied unit facts arrive at hundreds per second. One synchronous IPC per
    // fact (~0.5 ms each in the VM) saturated this transport thread and backed
    // up every later camera/image/present command. Scene-only facts can pass
    // queued canvas work and join their preceding ordered batch. Canvas-bound
    // facts retain their image dependencies. Each is encoded on this transport
    // thread at send time (after earlier image identities resolve), and its
    // own result keeps its original strict/superseded acceptance rule.
    // Local fact identities; the real transport maps them to wire messages.
    enum FactId:unsigned {fact_unit,fact_forget,fact_visual,fact_animation,fact_motion,fact_move,fact_spawn,fact_state};
    struct PendingFact {unsigned id=0;bool strict=false;char const* operation="state";
        std::function<void(c3x_inputs::Writer&)> write;};
    struct FactGroup {std::vector<PendingFact> facts;};
    template<class T>static auto fact_batching(int)->decltype(std::declval<T&>().facts(
        std::declval<std::vector<typename T::Fact> const&>(),std::declval<std::vector<unsigned>&>()),std::true_type{});
    template<class T>static std::false_type fact_batching(long);
    void execute_facts(FactGroup& group){
        if constexpr(decltype(fact_batching<Transport>(0))::value){
            std::vector<typename Transport::Fact> wire;wire.reserve(group.facts.size());
            for(auto& fact:group.facts){c3x_inputs::Writer out;fact.write(out);typename Transport::Fact next;
                Transport::fact_route(fact.id,next.kind,next.subtype);next.bytes=std::move(out.bytes);wire.push_back(std::move(next));}
            std::vector<unsigned> codes;
            if(!transport.facts(wire,codes))throw std::runtime_error("asynchronous renderer unit-fact-batch failed");
            for(std::size_t n=0;n<codes.size();++n){
                if(group.facts[n].strict)require_result(int(codes[n]),group.facts[n].operation);
                else accept_state(int(codes[n]),group.facts[n].operation);
            }
        }else (void)group;
    }
    static bool independent_canvas(char const* label){
        return label&&(!std::strcmp(label,"images")||!std::strcmp(label,"tactical")||!std::strcmp(label,"present"));
    }
    int post_fact(std::size_t bytes,PendingFact fact,std::function<void()> single,bool scene_only=true){
        if constexpr(decltype(fact_batching<Transport>(0))::value){
            auto group=std::make_shared<FactGroup>();group->facts.push_back(std::move(fact));
            bool accepted=publication.post_group(bytes,1,3,group,[this](FactGroup& value){execute_facts(value);},
                [](FactGroup& target,FactGroup& incoming){for(auto& f:incoming.facts)target.facts.push_back(std::move(f));},
                std::size_t(512)*1024,std::size_t(4096),std::size_t(512),"state",scene_only?independent_canvas:nullptr);
            transport.publication_pressure(publication.status().records);
            return accepted?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_DEVICE_ERROR;
        }else{(void)fact;return post(bytes,std::move(single),0,"state",scene_only?independent_canvas:nullptr);}
    }
    static std::shared_ptr<c3x_inputs::Frame> copy_frame(c3x_renderer_frame_v1 const& source){
        auto result=std::make_shared<c3x_inputs::Frame>();result->value=source;
        if(source.tile_count)result->tiles.assign(source.tiles,source.tiles+source.tile_count);
        if(source.world_topology_count)result->topology.assign(source.world_topology,source.world_topology+source.world_topology_count);
        result->bind();return result;
    }
    static std::unique_ptr<CameraOutput> copy_view(c3x_renderer_gpu_camera_view_v1 const& source){
        auto result=std::make_unique<CameraOutput>();result->value=source;
        auto frame=copy_frame(source.camera.frame);result->frame=std::move(*frame);result->frame.bind();
        auto const& output=source.camera.output;result->output.value=output;result->output.gpu=source.image;
        if(output.fallback_tile_count)result->output.fallbacks.assign(output.fallback_tile_indices,output.fallback_tile_indices+output.fallback_tile_count);
        if(output.replacement_tile_count)result->output.replacements.assign(output.replacement_tile_flags,output.replacement_tile_flags+output.replacement_tile_count);
        c3x_inputs::require(!output.bgra_pixels,"asynchronous map contains CPU pixels");
        result->output.bind();result->bind();return result;
    }
    // Transport fixtures and older optional helpers retain ordered RPC
    // inspection. The production helper supplies an independent ready slot.
    template<class T>static auto prepare_receipt(T& value,int)->decltype(value.prepare_camera_receipt(),void()){
        value.prepare_camera_receipt();
    }
    template<class T>static void prepare_receipt(T&,long){}
    template<class T>static auto completion(T& value,Id ticket,CameraOutput& output,int& code,int)
        ->decltype(value.camera_completion(ticket,output,code)){
        return value.camera_completion(ticket,output,code);
    }
    template<class T>static bool completion(T&,Id,CameraOutput&,int&,long){return false;}
    int query_page(Page& slot,c3x_renderer_world_page_v1& result,bool delta){
        std::lock_guard<std::mutex> lock(slot.mutex);
        if(slot.ready){slot.ready=false;result=slot.value;return slot.code;}
        if(!slot.query){
            slot.query=true;
            if(post(sizeof(Page),[this,&slot,delta]{
                c3x_renderer_world_page_v1 value={};
                int code=delta?transport.world_delta_scope(value):transport.world_query(value);
                std::lock_guard<std::mutex> finish(slot.mutex);
                slot.value=value;slot.value.tiles=nullptr;slot.value.frame.tiles=nullptr;
                slot.value.frame.world_topology=nullptr;slot.code=code;slot.ready=true;slot.query=false;
            })!=C3X_RENDERER_RESULT_OK)return C3X_RENDERER_RESULT_DEVICE_ERROR;
        }
        return publication.healthy()?C3X_RENDERER_RESULT_PENDING:C3X_RENDERER_RESULT_DEVICE_ERROR;
    }
    int submit_page(c3x_renderer_world_page_v1 value,int code,bool delta){
        // Until a map is displayed, Civ III deliberately defers world capture.
        // Return that status to the caller so it can capture again later; no
        // owned snapshot exists to publish and no transport failure occurred.
        if(code!=C3X_RENDERER_RESULT_OK)return code;
        auto tiles=std::make_shared<std::vector<c3x_renderer_tile_v1>>();
        if(value.count)tiles->assign(value.tiles,value.tiles+value.count);
        value.tiles=nullptr;value.frame.tiles=nullptr;value.frame.world_topology=nullptr;
        return post(sizeof(value)+tiles->size()*sizeof((*tiles)[0]),[this,value,code,delta,tiles]()mutable{
            value.tiles=tiles->data();
            accept_state(delta?transport.world_delta_submit(value,code):transport.world_submit(value,code),
                delta?"world-delta":"world-page");
        },0,delta?"world-delta":"world-page",delta?independent_canvas:nullptr);
    }
    void clear(){
        image_ids.clear();ticket_ids.clear();worker_camera=remote_camera=worker_map=0;policy=0;requested_policy=~0u;
        {std::lock_guard<std::mutex> lock(world_page.mutex);
            world_page.query=world_page.ready=false;world_page.code=C3X_RENDERER_RESULT_PENDING;world_page.value={};}
        {std::lock_guard<std::mutex> lock(world_progress.mutex);
            world_progress.query=false;world_progress.value={sizeof(world_progress.value)};}
    }
public:
    template<class... Args>AsyncSceneClient(bool asynchronous,std::function<void(char const*)> report,Args&&... args):
        transport(std::forward<Args>(args)...),enabled(asynchronous),publication(std::move(report)){
        observe_publication({});
    }
    ~AsyncSceneClient(){publication.stop();}
    bool asynchronous()const{return enabled;}
    unsigned presented_zoom()const{return transport.presented_zoom();}
    void observe_publication(std::function<void(char const*,double,double)> observer){
        publication.observe([this,observer=std::move(observer)](char const* label,double queued,double service){
            transport.publication_pressure(publication.status().records);
            if(observer)observer(label,queued,service);
        });
    }
    // Executed on the transport thread after adoption; the view is borrowed
    // only during the callback. Local tickets are aliases, not helper sources.
    void observe_camera_adoption(std::function<void(Id,Id,c3x_renderer_gpu_camera_view_v1 const&)> observer){
        camera_adoption_observer=std::move(observer);
    }
    bool alive()const{return publication.healthy()&&transport.alive();}
    void progress(unsigned& accepted,unsigned& completed,unsigned& frames)const{
        accepted=publication.accepted();completed=publication.completed();frames=transport.frames();
    }
    auto publication_status()const{return publication.status();}
    unsigned executed_adoptions()const{return adopted_cameras.load(std::memory_order_acquire);}
    unsigned presented_frames()const{return transport.frames();}
    auto stats(){return enabled?publication.setup([this]{return transport.stats();}):transport.stats();}
    int definitions(char const* root,char const* fallback,char const* scenario,char const* custom){
        camera.reset();displayed.reset();published_frame.reset();++session;
        if(!enabled)return transport.definitions(root,fallback,scenario,custom);
        // Admission lets native setup forms progress. The existing loading-scope
        // setup barrier joins actual source readiness before world capture.
        auto input=transport.definition_input(root,fallback,scenario,custom);
        auto bytes=input.capacity();
        return post(bytes,[this,input=std::move(input)]{
            clear();require_result(transport.definitions(input),"definition-preparation");
        },0,"definition-preparation");
    }
    int pack(char const* path){
        camera.reset();displayed.reset();published_frame.reset();++session;
        return enabled?publication.setup([&]{clear();return transport.pack(path);}):transport.pack(path);
    }
#ifdef C3X_RENDERER_BENCHMARK_ORACLE
    int benchmark_reset(unsigned mode,c3x_renderer_benchmark_oracle_trim_v1& result){
        camera.reset();displayed.reset();published_frame.reset();++session;
        return enabled?publication.reconcile([&]{int code=transport.benchmark_reset(mode,result);clear();return code;}):transport.benchmark_reset(mode,result);
    }
#endif
    int reset(){
        camera.reset();displayed.reset();published_frame.reset();++session;
        return enabled?publication.reconcile([&]{int code=transport.reset();require_result(code,"reset");clear();return code;}):transport.reset();
    }
    int set_units(int value){return enabled?post(sizeof(value),[this,value]{require_result(transport.set_units(value),"unit-configuration");}):transport.set_units(value);}
    int visual_policy(unsigned value){
        if(!enabled)return transport.visual_policy(value);
        if(value>=2)return policy.load(std::memory_order_acquire);
        // Civ III repeats the same permission on every present. Each post cost
        // two helper round trips on the shared transport; every present
        // already refreshes `policy`, and the helper's clock accumulates
        // elapsed time whenever it is next sampled.
        if(requested_policy.exchange(value)==value)return C3X_RENDERER_RESULT_OK;
        return post(sizeof(value),[this,value]{transport.visual_policy(value);policy=transport.visual_policy(3);});
    }
    int camera_begin(c3x_renderer_camera_request_v1 const& request,Id& result){
        if(!enabled)return transport.camera_begin(request,result);
        // Clock/UI repetitions do not cancel an immutable pending destination.
        if(camera&&published_frame&&same_camera_source(*request.frame,request.identity,published_frame->value,published_identity)){
            result=camera->ticket;return C3X_RENDERER_RESULT_PENDING;
        }
        transport.supersede_pending_camera();
        prepare_receipt(transport,0);
        auto slot=std::make_shared<Camera>();slot->ticket=++next_camera;
        auto frame=copy_frame(*request.frame);auto identity=request.identity;
        int code=post(sizeof(*frame)+frame->tiles.size()*sizeof(frame->tiles[0])+frame->topology.size()*4,
            [this,slot,frame,identity]{
                c3x_renderer_camera_request_v1 value={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(value),&frame->value,identity};
                Id actual=0;int accepted=transport.camera_begin(value,actual);
                if(accepted!=C3X_RENDERER_RESULT_PENDING)require_result(accepted,"camera-begin");
                worker_camera=slot->ticket;remote_camera=actual;
                slot->remote_ticket.store(actual,std::memory_order_release);
            },1,"camera-begin",
            // Latest camera wins. It may run ahead of queued native UI drawing:
            // that work uses the displayed ticket, which only the later ordered
            // adoption retires, and the worker serves it at job checkpoints.
            // Facts, scene state and camera commands keep their order.
            independent_canvas);
        if(code!=C3X_RENDERER_RESULT_OK)return code;
        camera=slot;published_frame=frame;published_identity=identity;
        result=slot->ticket;return C3X_RENDERER_RESULT_PENDING;
    }
    int camera_poll(Id wanted,c3x_renderer_gpu_camera_view_v1& result){
        if(!enabled)return transport.camera_poll(wanted,result);
        if(!publication.healthy())return C3X_RENDERER_RESULT_DEVICE_ERROR;
        auto slot=camera;if(!slot||slot->ticket!=wanted)return C3X_RENDERER_RESULT_SUPERSEDED;
        std::lock_guard<std::mutex> lock(slot->mutex);
        if(slot->adopted){result=displayed->value;return C3X_RENDERER_RESULT_OK;}
        bool mailbox=false;
        if(!slot->ready&&slot->code==C3X_RENDERER_RESULT_PENDING){
            auto remote=slot->remote_ticket.load(std::memory_order_acquire);
            CameraOutput ready;int code=C3X_RENDERER_RESULT_PENDING;
            mailbox=completion(transport,remote,ready,code,0);
            if(mailbox&&!remote)code=C3X_RENDERER_RESULT_PENDING;
            if(mailbox){
                if(code==C3X_RENDERER_RESULT_OK){
                    // GPU publications deliberately omit topology storage. Its
                    // captured revision and exact ticket still bind the source.
                    auto captured=published_frame?published_frame->value:c3x_renderer_frame_v1{};
                    captured.world_topology=nullptr;captured.world_topology_count=0;
                    if(!published_frame||!same_camera_source(ready.value.camera.frame,ready.value.camera.identity,
                        captured,published_identity))return C3X_RENDERER_RESULT_SUPERSEDED;
                    slot->ready=std::make_unique<CameraOutput>(std::move(ready));
                }
                slot->code=code;
            }
        }
        if(slot->ready){
            Id map=++next_image;
            // This command precedes every operation using the returned ticket.
            int code=post(sizeof(CameraOutput),[this,wanted,map,observer=camera_adoption_observer]{
                if(worker_camera!=wanted)throw std::runtime_error("camera changed before ordered adoption");
                c3x_renderer_gpu_camera_view_v1 actual={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(actual)};
                int accepted=transport.camera_poll(remote_camera,actual);
                require_result(accepted,"camera-adopt");
                if(observer)observer(wanted,wanted,actual);
                if(worker_map)image_ids.erase(worker_map);
                ticket_ids.clear();ticket_ids[wanted]=actual.image.ticket;
                image_ids[map]=actual.image.map_image;worker_map=map;
                adopted_cameras.fetch_add(1,std::memory_order_release);
            },0,"camera-adopt");
            if(code!=C3X_RENDERER_RESULT_OK)return code;
            displayed=std::move(slot->ready);slot->adopted=true;
            displayed->output.gpu.ticket=wanted;displayed->output.gpu.map_image=map;
            displayed->output.gpu.session=session;displayed->value.camera.ticket=wanted;
            displayed->bind();result=displayed->value;return C3X_RENDERER_RESULT_OK;
        }
        if(slot->code!=C3X_RENDERER_RESULT_PENDING)return slot->code;
        if(!mailbox&&!slot->query){
            slot->query=true;
            int code=post(sizeof(Camera),[this,slot]{
                c3x_renderer_gpu_camera_view_v1 value={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(value)};
                int ready=worker_camera==slot->ticket?transport.camera_ready(remote_camera,value):C3X_RENDERER_RESULT_SUPERSEDED;
                auto copied=ready==C3X_RENDERER_RESULT_OK?copy_view(value):nullptr;
                std::lock_guard<std::mutex> finish(slot->mutex);
                slot->code=ready;slot->ready=std::move(copied);slot->query=false;
            },0,"camera-ready");
            if(code!=C3X_RENDERER_RESULT_OK)return code;
        }
        return C3X_RENDERER_RESULT_PENDING;
    }
    int camera_cancel(Id wanted){
        if(!enabled)return transport.camera_cancel(wanted);
        if(camera&&camera->ticket==wanted){transport.supersede_pending_camera();camera.reset();}
        return post(sizeof(wanted),[this,wanted]{if(worker_camera==wanted){
            accept_state(transport.camera_cancel(remote_camera),"camera-cancel");worker_camera=remote_camera=0;}});
    }
    int images(c3x_renderer_gpu_images_v1 const& request,c3x_renderer_gpu_result_v1& result,unsigned* pixels,unsigned capacity){
        if(!enabled)return transport.images(request,result,pixels,capacity);
        // A game-thread CPU lease cannot be satisfied asynchronously. Report the
        // unsupported ownership transition; never read the custom map back.
        if(request.action==C3X_GPU_READBACK)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        auto bytes=sizeof(c3x_inputs::Images)+std::size_t(request.pixel_count)*4+
            std::size_t(request.command_count)*sizeof(c3x_renderer_gpu_command_v1);
        Id created=0;
        bool accepted=publication.post_group_wait<ImageBatch>(bytes,ImageBatch::work(request),2,[&]{
            auto batch=std::make_shared<ImageBatch>();batch->operations.resize(1);
            auto& operation=batch->operations[0];auto& packet=operation.image;packet.value=request;
            if(request.pixel_count)packet.pixels.assign(request.pixels,request.pixels+request.pixel_count);
            if(request.command_count)packet.commands.assign(request.commands,request.commands+request.command_count);
            packet.bind();created=request.action==C3X_GPU_CREATE?next_image+1:0;operation.created=created;
            return batch;
        },
            [this](ImageBatch& value){execute_images(value);},
            [](ImageBatch& target,ImageBatch& incoming){target.operations.push_back(std::move(incoming.operations[0]));},
            ImageBatch::join_bytes,ImageBatch::work_limit,ImageBatch::operation_limit,"images");
        transport.publication_pressure(publication.status().records);
        if(accepted&&created)next_image=created;
        result={sizeof(result)};result.image=created?created:request.image;
        return accepted?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_DEVICE_ERROR;
    }

    int unit(c3x_renderer_unit_v1 value,c3x_renderer_gpu_unit_v1 destination,int* bounds){
        if(!enabled)return transport.unit(value,destination,bounds);
        // Bodies live in the retained 3D scene. Native animation must not erase
        // or redraw their canvas while the renderer advances its own clock.
        bounds[0]=bounds[2]=value.body_x;bounds[1]=bounds[3]=value.body_y;
        return post_fact(sizeof(value)+sizeof(destination),{fact_unit,true,"unit-observation",
            [this,value,destination](c3x_inputs::Writer& out)mutable{target(destination);c3x_inputs::unit(out,value);c3x_inputs::target_fields(out,destination);}},
            [this,value,destination]()mutable{target(destination);int unused[4]={};require_result(transport.unit(value,destination,unused),"unit-observation");},destination.ticket==0);
    }
    void forget_unit(int id){
        if(!enabled){transport.forget_unit(id);return;}
        post_fact(sizeof(id),{fact_forget,false,"unit-forget",
            [id](c3x_inputs::Writer& out){out(std::int32_t(id));}},[this,id]{transport.forget_unit(id);});
    }
    int unit_visual(c3x_renderer_unit_visual_v1 value){return enabled?post_fact(sizeof(value),{fact_visual,false,"unit-visual",
        [value](c3x_inputs::Writer& out)mutable{c3x_inputs::unit_visual_fields(out,value);}},
        [this,value]{accept_state(transport.unit_visual(value),"unit-visual");}):transport.unit_visual(value);}
    int unit_animation(c3x_renderer_unit_animation_v1 value){return enabled?post_fact(sizeof(value),{fact_animation,false,"unit-animation",
        [value](c3x_inputs::Writer& out)mutable{c3x_inputs::unit_animation_fields(out,value);}},
        [this,value]{accept_state(transport.unit_animation(value),"unit-animation");}):transport.unit_animation(value);}
    int unit_motion(c3x_renderer_unit_move_v1 value){return enabled?post_fact(sizeof(value),{fact_motion,false,"unit-motion",
        [value](c3x_inputs::Writer& out)mutable{c3x_inputs::unit_move_fields(out,value);}},
        [this,value]{accept_state(transport.unit_motion(value),"unit-motion");}):transport.unit_motion(value);}
    int unit_move(c3x_renderer_unit_move_v1 value){return enabled?post_fact(sizeof(value),{fact_move,false,"unit-move",
        [value](c3x_inputs::Writer& out)mutable{c3x_inputs::unit_move_fields(out,value);}},
        [this,value]{accept_state(transport.unit_move(value),"unit-move");}):transport.unit_move(value);}
    int unit_spawn(c3x_renderer_unit_spawn_v1 value){return enabled?post_fact(sizeof(value),{fact_spawn,false,"unit-spawn",
        [value](c3x_inputs::Writer& out)mutable{c3x_inputs::unit_spawn_fields(out,value);}},
        [this,value]{accept_state(transport.unit_spawn(value),"unit-spawn");}):transport.unit_spawn(value);}
    int unit_state(c3x_renderer_unit_state_v1 value){return enabled?post_fact(sizeof(value),{fact_state,false,"unit-state",
        [value](c3x_inputs::Writer& out)mutable{c3x_inputs::unit_state_fields(out,value);}},
        [this,value]{accept_state(transport.unit_state(value),"unit-state");}):transport.unit_state(value);}
    // Consecutive native strokes into the same canvas (borders, routes) join
    // the last queued tactical record: each line was its own helper round
    // trip, ~150 a second on a busy map. The helper draws a capture's
    // primitives in order with premultiplied blending in one pass, so the
    // joined record composes as the separate ones did. Animated and static
    // captures never join (an animated one is redrawn every frame).
    struct TacticalGroup {c3x_renderer::tactical::Input capture;c3x_renderer_gpu_unit_v1 destination;};
    c3x_renderer_gpu_unit_v1 tactical_target={};bool tactical_animated=false;unsigned tactical_key=0,tactical_serial=0;
    static bool same_target(c3x_renderer_gpu_unit_v1 const& a,c3x_renderer_gpu_unit_v1 const& b){
        return a.ticket==b.ticket&&a.destination==b.destination&&a.background==b.background&&a.detail==b.detail&&
            a.background_detail==b.background_detail&&a.playback_flags==b.playback_flags&&
            a.clip[0]==b.clip[0]&&a.clip[1]==b.clip[1]&&a.clip[2]==b.clip[2]&&a.clip[3]==b.clip[3];
    }
    int tactical(c3x_renderer::tactical::Input capture,c3x_renderer_gpu_unit_v1 destination){
        if(!enabled)return transport.tactical(capture,destination);
        auto size=sizeof(destination)+capture.primitives.size()*sizeof(capture.primitives[0]);
        if(!tactical_key||!same_target(destination,tactical_target)||capture.animated!=tactical_animated){
            tactical_target=destination;tactical_animated=capture.animated;
            tactical_key=0x80000000u|(++tactical_serial&0x7fffffffu); // disjoint from fact/image groups
        }
        // One unit per record, as before joining: a route or grid capture can
        // carry thousands of primitives, and counting each one exhausted the
        // queue's work budget on the busy map. Joined primitives stay below the
        // 16384-primitive capture limit through the 256 KB join bound.
        auto group=std::make_shared<TacticalGroup>(TacticalGroup{std::move(capture),destination});
        bool accepted=publication.post_group(size,1,tactical_key,group,
            [this](TacticalGroup& value){target(value.destination);require_result(transport.tactical(value.capture,value.destination),"tactical");},
            [](TacticalGroup& into,TacticalGroup& incoming){auto& p=into.capture.primitives;
                p.insert(p.end(),incoming.capture.primitives.begin(),incoming.capture.primitives.end());},
            std::size_t(256)*1024,std::size_t(4096),std::size_t(4096),"tactical");
        transport.publication_pressure(publication.status().records);
        return accepted?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_DEVICE_ERROR;
    }
    // Loading admission stays ordered, but returns only after ownership transfer.
    // The native callback itself remains on the caller/game thread between pages.
    int seed_world_scope(c3x_renderer_camera_request_v1 const& request){
        auto frame=copy_frame(*request.frame);auto identity=request.identity;
        return enabled?publication.setup([this,frame,identity]{
            c3x_renderer_camera_request_v1 value={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(value),&frame->value,identity};
            return transport.seed_world_scope(value);
        }):transport.seed_world_scope(request);
    }
    int seed_world_loading_scope(c3x_renderer_camera_request_v1 const& request){
        auto frame=copy_frame(*request.frame);auto identity=request.identity;
        return enabled?publication.setup([this,frame,identity]{
            c3x_renderer_camera_request_v1 value={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(value),&frame->value,identity};
            return transport.seed_world_scope(value,true);
        }):transport.seed_world_scope(request,true);
    }
    int prepare_world_loading(c3x_renderer_camera_identity_v1 const& identity){
        // World preparation retires the helper's animation sampler, even when
        // the next visible scene is identical. It cannot reuse this camera's
        // ready receipt. Keep displayed pixels and image aliases until ordered
        // adoption of a fresh camera, including after a failed preparation.
        if(enabled)camera.reset();
        return enabled?publication.setup([this,identity]{return transport.prepare_world_loading(identity);}):transport.prepare_world_loading(identity);
    }
    int world_seed_query(c3x_renderer_world_page_v1& page){
        return enabled?publication.setup([&]{return transport.world_query(page,true);}):transport.world_query(page,true);
    }
    int world_seed_submit(c3x_renderer_world_page_v1 const& page,int code){
        return enabled?publication.setup([&]{return transport.world_submit(page,code);}):transport.world_submit(page,code);
    }
    int arm_world_changes(c3x_renderer_camera_identity_v1 const& identity){
        return enabled?publication.setup([this,identity]{return transport.arm_world_changes(identity);}):transport.arm_world_changes(identity);
    }
    int world_query(c3x_renderer_world_page_v1& page){return enabled?query_page(world_page,page,false):transport.world_query(page);}
    int world_delta_scope(c3x_renderer_world_page_v1& page){
        if(!enabled)return transport.world_delta_scope(page);
        if(!published_frame)return C3X_RENDERER_RESULT_PENDING;
        page={};page.struct_size=sizeof(page);page.first=UINT32_MAX;page.capacity=128;
        page.frame=published_frame->value;page.identity=published_identity;
        page.frame.tiles=nullptr;page.frame.tile_count=0;page.frame.world_topology=nullptr;
        return C3X_RENDERER_RESULT_OK;
    }
    int world_submit(c3x_renderer_world_page_v1 const& page,int code){return enabled?submit_page(page,code,false):transport.world_submit(page,code);}
    int world_delta_submit(c3x_renderer_world_page_v1 const& page,int code){return enabled?submit_page(page,code,true):transport.world_delta_submit(page,code);}
    int world_status(c3x_renderer_world_status_v1& value){
        if(!enabled)return transport.world_status(value);
        std::lock_guard<std::mutex> lock(world_progress.mutex);value=world_progress.value;
        if(!world_progress.query){world_progress.query=true;
            if(post(sizeof(value),[this]{c3x_renderer_world_status_v1 next={sizeof(next)};
                int code=transport.world_status(next);
                std::lock_guard<std::mutex> finish(world_progress.mutex);
                if(code==C3X_RENDERER_RESULT_OK)world_progress.value=next;world_progress.query=false;
            })!=C3X_RENDERER_RESULT_OK)return C3X_RENDERER_RESULT_DEVICE_ERROR;}
        return value.total?C3X_RENDERER_RESULT_OK:C3X_RENDERER_RESULT_PENDING;
    }
    template<class Handle>int bind_surface(Handle handle,unsigned width,unsigned height){
        if(!enabled)return transport.bind_surface(handle,width,height);
        auto retained=transport.retain_surface(handle);
        return post(sizeof(handle)+8,[this,retained,width,height]{require_result(transport.bind_surface(retained.get(),width,height),"surface-bind");});
    }
    template<class Shared>int present(c3x_renderer_gpu_present_v1 value,Shared& frame){
        if(!enabled)return transport.present(value,frame);
        frame={};return post(sizeof(value),[this,value]()mutable{
            if(!value.action){value.ticket=ticket(value.ticket);value.image=image(value.image);}
            Shared unused;require_result(transport.present(value,unused),"present");policy=transport.visual_policy(3);
        },0,"present");
    }
    // Startup joins the same reliable publication prefix. The transport owns
    // this copied request until the exact committed front has been presented.
    template<class Shared>int present_required(c3x_renderer_gpu_present_v1 value,Shared& frame){
        if(!enabled)return transport.present_required(value,frame);
        frame={};return publication.setup([this,value,&frame]()mutable{
            value.ticket=ticket(value.ticket);value.image=image(value.image);
            int code=transport.present_required(value,frame);policy=transport.visual_policy(3);
            return code;
        });
    }
    template<class Shared>int visual(std::int64_t ticks,std::int64_t frequency,Shared& frame){
        if(!enabled)return transport.visual(ticks,frequency,frame);
        frame={};return C3X_RENDERER_RESULT_PENDING; // x64 owns the independent cadence
    }
    bool surface_pixels(std::vector<unsigned>& pixels,unsigned& width,unsigned& height){
        // Explicit diagnostic witness / final window handoff, outside draws.
        return enabled?publication.setup([&]{return transport.surface_pixels(pixels,width,height);}):transport.surface_pixels(pixels,width,height);
    }
    int render(c3x_renderer_camera_request_v1 const& request,c3x_renderer_gpu_frame_v1& gpu,c3x_renderer_output_v1& output){
        return enabled?C3X_RENDERER_RESULT_BAD_ARGUMENT:transport.render(request,gpu,output);
    }
    int render_cpu(c3x_renderer_frame_v1 const& frame,c3x_renderer_camera_identity_v1 const* identity,c3x_renderer_output_v1& output){
        return enabled?C3X_RENDERER_RESULT_BAD_ARGUMENT:transport.render_cpu(frame,identity,output);
    }
    int unit_cpu(c3x_renderer_unit_v1 const& unit,unsigned flags,int* bounds,std::vector<std::uint32_t>& pixels,int& x,int& y,unsigned& width,unsigned& height){
        return enabled?C3X_RENDERER_RESULT_BAD_ARGUMENT:transport.unit_cpu(unit,flags,bounds,pixels,x,y,width,height);
    }
};
}
