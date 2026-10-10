#pragma once
#include "native_image_adapter.h"
#include "gpu_image_worker_client.h"
#include <memory>
#include "tactical_overlay.h"
#include "native_navigation.h"
#include "scene_projection.h"
#include <functional>
namespace c3x_native_images {
// Caller-thread owner for the native map/copy/save/display family. The existing
// renderer worker owns GPU work; this object never retains game request pointers.
class CompositionOwner {
    c3x_renderer_gpu_render_fn render;
    c3x_renderer_gpu_camera_begin_fn camera_begin=nullptr;
    c3x_renderer_gpu_camera_poll_view_fn camera_poll=nullptr;
    c3x_renderer_camera_cancel_fn camera_cancel=nullptr;
    c3x_renderer_i64 camera_ticket=0;void* camera_image=nullptr;int camera_width=0,camera_height=0;
    c3x_renderer_frame_v1 camera_capture={};
    c3x_renderer_camera_identity_v1 camera_identity={};
    std::vector<c3x_renderer_tile_v1> camera_tiles;
    std::vector<c3x_renderer_u32> camera_topology;
    c3x_renderer_gpu_images_fn images;
    c3x_renderer_gpu_present_fn present;
    c3x_renderer_gpu_unit_fn unit;
    c3x_renderer_native_lifetime_fn lifetime;
    void* bits;void* release;
    DWORD thread=GetCurrentThreadId();
    std::unique_ptr<c3x_gpu_images::WorkerClient> client;
    std::unique_ptr<Adapter<c3x_gpu_images::WorkerClient>> adapter;
    c3x_renderer_gpu_frame_v1 frame={sizeof(frame)};
    bool scene_units=false,trace_success=false;
    void* front_native=nullptr;void* display_native=nullptr;unsigned surface_copy_reports=0,surface_fill_reports=0,cold_stroke_reports=0,world_reports=0;
    using Tactical=c3x_renderer::tactical::Input;
    std::function<int(Tactical const&,c3x_renderer_gpu_unit_v1 const&)> tactical;
    Tactical route;void* route_image=nullptr;c3x_renderer_tactical_view_v1 route_view={};
    c3x_renderer::tactical::RouteAnchors route_anchors;
    std::array<float,2> destination={};std::string route_text;
    float projected(int v,bool y)const{return float(double(v)*route_view.tile_width/route_view.native_tile_width+
        double(y?route_view.translate_y_fp:route_view.translate_x_fp)/65536.);}
    std::array<float,2> route_point(int x,int y)const{
        return route_anchors.resolve(x,y,{projected(x,false),projected(y,true)},field(route_image,0x38),field(route_image,0x3c));
    }
    // C3X_RENDERER_NATIVE_MAP_HUD=1 declines the map HUD the renderer draws
    // itself (stage 4.3), so Civ III draws it: the side-by-side reference.
    bool native_map_hud=[]{char value[4]={};return GetEnvironmentVariableA("C3X_RENDERER_NATIVE_MAP_HUD",value,sizeof(value))==1&&value[0]=='1';}();
    std::array<std::uint64_t,4> status_counts{};
    void note_status(){auto total=status_counts[0]+status_counts[1]+status_counts[2]+status_counts[3];
        if(total>3&&total%4096)return;char line[192];std::snprintf(line,sizeof(line),
            "[C3X renderer] stage=unit-status-reports accepted=%llu refused_facts=%llu refused_canvas=%llu refused_led=%llu\n",
            (unsigned long long)status_counts[0],(unsigned long long)status_counts[1],(unsigned long long)status_counts[2],(unsigned long long)status_counts[3]);
        OutputDebugStringA(line);}
    int tactical_draw(void* image,Tactical const& capture,void* background=nullptr){
        if(capture.primitives.empty())return 1;
        if(!tactical)return 0;
        return adapter->draw_tactical([&](auto const& target){return tactical(capture,target);},frame.ticket,image,background)?1:0;
    }
    Navigation navigation;
    void* pending=nullptr;Rect area={};int phase_x=0,phase_y=0;
    void check_thread(){if(GetCurrentThreadId()!=thread)throw std::runtime_error("native composition caller changed");}
    void release_window(){c3x_renderer_gpu_present_v1 r={sizeof(r)};r.action=2;
        if(present(&r)!=C3X_RENDERER_RESULT_OK)throw std::runtime_error("native display handoff failed");}
    static int field(void* p,unsigned offset){return c3x_native_access::field(p,offset);}
    void clear_camera_capture(){camera_capture={};camera_tiles.clear();camera_topology.clear();}
    void trace_map(char const* phase,int result,void* image,c3x_renderer_camera_request_v1 const* request=nullptr)const{
        if(!trace_success && (result==C3X_RENDERER_RESULT_OK ||
                result==C3X_RENDERER_RESULT_PENDING || result==C3X_RENDERER_RESULT_SUPERSEDED ||
                result==C3X_RENDERER_RESULT_BUSY))return;
        char line[320];int anchor_x=0,anchor_y=0;
        if(request&&request->frame&&request->frame->tile_count&&request->frame->tiles){
            anchor_x=request->frame->tiles[0].anchor_x;anchor_y=request->frame->tiles[0].anchor_y;
        }
        std::snprintf(line,sizeof(line),
            "[C3X renderer] stage=native-map-transaction phase=%s result=%d camera_ticket=%lld front_ticket=%lld pending=%u image_matches_camera=%u image_matches_pending=%u anchor=%d,%d tiles=%u\n",
            phase,result,static_cast<long long>(camera_ticket),static_cast<long long>(frame.ticket),
            unsigned(pending!=nullptr),unsigned(image==camera_image),unsigned(image==pending),
            anchor_x,anchor_y,request&&request->frame?request->frame->tile_count:0u);
        OutputDebugStringA(line);
    }
    bool same_camera_capture(c3x_renderer_camera_request_v1 const& request)const{
        if(!camera_capture.struct_size || std::memcmp(&camera_identity,&request.identity,sizeof(camera_identity)))return false;
        auto current=*request.frame,retained=camera_capture;
        current.tiles=retained.tiles=nullptr;
        current.world_topology=retained.world_topology=nullptr;
        current.presentation_time_ticks=retained.presentation_time_ticks=0;
        current.dirty_flags=retained.dirty_flags=0;
        current.visible_animation_count=retained.visible_animation_count=0;
        return !std::memcmp(&current,&retained,sizeof(current)) &&
            (!current.tile_count || !std::memcmp(request.frame->tiles,camera_tiles.data(),
                camera_tiles.size()*sizeof(camera_tiles[0]))) &&
            (!current.world_topology_count || !std::memcmp(request.frame->world_topology,
                camera_topology.data(),camera_topology.size()*sizeof(camera_topology[0])));
    }
public:
    bool eligible(void* image,c3x_renderer_frame_v1 const& demand){
        return lifetime(C3X_NATIVE_MAP,image,0) && field(image,0x24)==16 && !field(image,0x4c4) && !field(image,0x4c8) &&
            field(image,0x38)==demand.target_width && field(image,0x3c)==demand.target_height;
    }
    int prepare_image(void* image,c3x_renderer_gpu_frame_v1 const& next,c3x_renderer_output_v1 const& output,int x,int y){
        if(client){if(next.ticket!=frame.ticket || next.session!=frame.session)client->advance(next);}
        else {
            auto next_client=std::make_unique<c3x_gpu_images::WorkerClient>(images,next,scene_units);
            auto next_adapter=std::make_unique<Adapter<c3x_gpu_images::WorkerClient>>(*next_client,bits,release,lifetime);
            client=std::move(next_client);adapter=std::move(next_adapter);
        }
        frame=next;
        if(!adapter->admit(image))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        pending=image;area={output.clip_left,output.clip_top,output.clip_right,output.clip_bottom};
        phase_x=x;phase_y=y;return C3X_RENDERER_RESULT_OK;
    }
public:
    CompositionOwner(c3x_renderer_gpu_render_fn r,c3x_renderer_gpu_images_fn i,c3x_renderer_gpu_present_fn p,
        c3x_renderer_gpu_unit_fn u,c3x_renderer_native_lifetime_fn l,void* b,void* end,bool direct_scene_units=false,bool diagnostic_trace=false):render(r),images(i),present(p),unit(u),lifetime(l),bits(b),release(end),scene_units(direct_scene_units),trace_success(diagnostic_trace){}
    // Wheel zoom requests: each is sequenced, ordered in the native stream and
    // also published at once through `zoom_hint` (review, section 42).
    std::function<void(unsigned,unsigned)> zoom_hint;unsigned zoom_sequence=0;
    void set_zoom_hint(std::function<void(unsigned,unsigned)> hint){zoom_hint=std::move(hint);}
    void set_camera(c3x_renderer_gpu_camera_begin_fn begin,c3x_renderer_gpu_camera_poll_view_fn poll,c3x_renderer_camera_cancel_fn cancel){
        camera_begin=begin;camera_poll=poll;camera_cancel=cancel;
    }
    int request_camera(void* image,c3x_renderer_camera_request_v1 const& request,c3x_renderer_i64& ticket){
        check_thread();
        if(!camera_begin || pending || !eligible(image,*request.frame))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        // Civ III clears/draws native surfaces before requesting its next map.
        // Publish those commands under the old ticket before the camera request.
        // Renderer64 flush only enqueues copied work; it never waits for x64.
        if(client&&!client->flushed()){
            if(!scene_units)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            client->flush();
        }
        c3x_renderer_i64 next=0;int result=camera_begin(&request,&next);
        if(result==C3X_RENDERER_RESULT_PENDING){
            camera_ticket=next;camera_image=image;camera_width=request.frame->target_width;camera_height=request.frame->target_height;ticket=next;
            // Capture-only movement requests and ordinary redraws share the
            // same copied identity. The next redraw polls this work instead
            // of cancelling it merely because it was submitted before drawing.
            try{
                camera_capture=*request.frame;camera_identity=request.identity;
                camera_tiles.clear();camera_topology.clear();
                if(camera_capture.tile_count)camera_tiles.assign(request.frame->tiles,request.frame->tiles+camera_capture.tile_count);
                if(camera_capture.world_topology_count)camera_topology.assign(request.frame->world_topology,
                    request.frame->world_topology+camera_capture.world_topology_count);
                camera_capture.tiles=nullptr;camera_capture.world_topology=nullptr;
            }catch(...){
                if(camera_cancel)camera_cancel(camera_ticket);
                camera_ticket=0;camera_image=nullptr;clear_camera_capture();throw;
            }
        }
        return result;
    }
    int poll_camera(void* image,c3x_renderer_i64 ticket,c3x_renderer_gpu_camera_view_v1& view){
        check_thread();
        if(!camera_poll || ticket<=0 || ticket!=camera_ticket || image!=camera_image)return C3X_RENDERER_RESULT_SUPERSEDED;
        if(pending)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        // Native lifetime validation precedes any adoption; an escaped/deleted
        // destination cannot redirect a ready result to a different surface.
        if(!lifetime(C3X_NATIVE_MAP,image,0) || field(image,0x38)!=camera_width || field(image,0x3c)!=camera_height) return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        // Adoption changes the active ticket. All earlier native commands must
        // precede it in the same publication queue, including draws made while
        // this camera was pending.
        if(client&&!client->flushed()){
            if(!scene_units)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            client->flush();
        }
        c3x_renderer_gpu_camera_view_v1 next={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(next)};
        int result=camera_poll(ticket,&next);
        if(result!=C3X_RENDERER_RESULT_OK){trace_map("poll",result,image);return result;}
        if(!eligible(image,next.camera.frame))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        result=prepare_image(image,next.image,next.camera.output,next.pixel_phase_x,next.pixel_phase_y);
        if(result==C3X_RENDERER_RESULT_OK){view=next;camera_ticket=0;camera_image=nullptr;}
        return result;
    }
    int navigate(int action,void* image,custom_renderer_native_view& view,c3x_renderer_camera_request_v1 const* request){
        check_thread();
        if(action==C3X_NAV_REQUEST || action==C3X_NAV_REQUEST_SCROLL)
            return navigation.request(*this,image,view,*request,action==C3X_NAV_REQUEST_SCROLL);
        if(action==C3X_NAV_PENDING)
            return navigation.active()&&!navigation.available()?C3X_RENDERER_RESULT_PENDING:C3X_RENDERER_RESULT_OK;
        return navigation.poll(*this,action,image,view);
    }
    void retire_image(int operation,void* image){
        // Cross-thread observation invalidates admission in Lifetimes. It must
        // not mutate this caller-thread owner or throw across the C hook.
        if(GetCurrentThreadId()!=thread)return;
        if(operation!=C3X_NATIVE_INIT && operation!=C3X_NATIVE_DESTROY && operation!=C3X_NATIVE_IMAGE_REINIT &&
            !(operation==C3X_NATIVE_VERIFY && !image))return;
        check_thread();
        if(!image || image==camera_image){
            if(camera_ticket&&camera_cancel)camera_cancel(camera_ticket);
            camera_ticket=0;camera_image=nullptr;clear_camera_capture();navigation.clear();
        }
        if(!image || image==pending){pending=nullptr;navigation.clear();}
        if(!image || image==front_native)front_native=nullptr;
        if(!image || image==display_native)display_native=nullptr;
        if(!image || image==route_image){route={};route_image=nullptr;route_text.clear();route_anchors.points.clear();}
    }
    bool defer_cold_stroke(void* image,void const* stroke){
        // The first copied camera frame has not created its GPU adapter yet.
        // A line drawn into its future full-screen destination would be
        // overwritten by the map copy; acquiring a native DC for that line
        // would instead disqualify the destination for its entire lifetime.
        if(!scene_units||adapter||camera_ticket<=0||!image||!stroke||!tactical||
            camera_width<640||camera_height<480||field(image,0x24)!=16||
            field(image,0x38)!=camera_width||field(image,0x3c)!=camera_height||
            !lifetime(C3X_NATIVE_MAP,image,0))return false;
        auto p=static_cast<c3x_renderer_native_stroke const*>(stroke);
        if(p->width<1||p->width>128||p->dash<0||p->dash>2||
            p->x1<-32768||p->x1>32767||p->y1<-32768||p->y1>32767||
            p->x2<-32768||p->x2>32767||p->y2<-32768||p->y2>32767)return false;
        if(cold_stroke_reports++<8){char line[192];std::snprintf(line,sizeof(line),
            "[C3X renderer] stage=native-cold-stroke-deferred ticket=%lld image=%p size=%d,%d\n",
            static_cast<long long>(camera_ticket),image,camera_width,camera_height);
            OutputDebugStringA(line);}
        return true;
    }
    void set_tactical(std::function<int(Tactical const&,c3x_renderer_gpu_unit_v1 const&)> draw){tactical=std::move(draw);}
    bool active()const{return adapter!=nullptr;}
    c3x_renderer_i64 sample_ticks()const{return frame.presentation_time_ticks;}
    // Diagnostic identity of the image actually prepared by this caller-thread
    // owner. A navigation fallback/barrier never grants an offered GPU source.
    // The async bridge reserves this ticket locally; it is not an x64 source
    // serial. Only its ordered adoption knows the remote map identity.
    c3x_renderer_i64 local_image_ticket()const{return frame.ticket;}
    c3x_renderer_i64 requested_ticket()const{return camera_ticket;}
    bool offered_navigation()const{return navigation.available();}
    // Native validation occurs between prepare and commit. Preparation never
    // inserts pixels or claims category replacement on the game's behalf.
    int map(int action,void* image,c3x_renderer_camera_request_v1 const* request,c3x_renderer_output_v1* output){
        check_thread();
        if(action==C3X_NATIVE_MAP_CANCEL){
            trace_map("cancel",C3X_RENDERER_RESULT_OK,image);
            if(camera_ticket&&camera_cancel)camera_cancel(camera_ticket);
            camera_ticket=0;camera_image=nullptr;clear_camera_capture();pending=nullptr;navigation.clear();return C3X_RENDERER_RESULT_OK;
        }
        if(action==C3X_NATIVE_MAP_COMMIT){
            if(navigation.available()||!pending||image!=pending||!adapter)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            pending=nullptr;
            if(!adapter->insert_map(image,Id(frame.map_image),area,area.left,area.top,phase_x,phase_y))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            front_native=image;client->flush();trace_map("commit",C3X_RENDERER_RESULT_OK,image);return C3X_RENDERER_RESULT_OK;
        }
        if(action==C3X_NATIVE_MAP_PREPARE && request && request->frame && output && navigation.available()){
            if(pending==image && eligible(image,*request->frame) && navigation.take(image,*request,*output))return C3X_RENDERER_RESULT_OK;
            // Fresh authoritative capture changed while this view was pending.
            // Reject its old coverage and request a new copied camera scene.
            pending=nullptr;navigation.clear();
        }
        if(action!=C3X_NATIVE_MAP_PREPARE||!request||!request->frame||!output||pending)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        trace_map("prepare",C3X_RENDERER_RESULT_PENDING,image,request);
        auto const& demand=*request->frame;
        if(!eligible(image,demand))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(adapter&&!adapter->admit(image))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(scene_units){
            // The game thread only submits or polls a copied camera demand.
            // A cold destination may remain unpresented until its scene is
            // ready; this branch never calls the exact GPU renderer.
            if(camera_ticket && (camera_image!=image || !same_camera_capture(*request))){
                if(camera_cancel)camera_cancel(camera_ticket);
                camera_ticket=0;camera_image=nullptr;clear_camera_capture();
            }
            if(camera_ticket){
                c3x_renderer_gpu_camera_view_v1 ready={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(ready)};
                int result=poll_camera(image,camera_ticket,ready);
                if(result==C3X_RENDERER_RESULT_OK){
                    *output=ready.camera.output;clear_camera_capture();trace_map("adopt",result,image,request);return result;
                }
                if(result!=C3X_RENDERER_RESULT_PENDING){camera_ticket=0;camera_image=nullptr;clear_camera_capture();}
                trace_map("poll-return",result,image,request);
                return result;
            }
            c3x_renderer_i64 next=0;
            int result=request_camera(image,*request,next);
            trace_map("begin",result,image,request);
            if(result!=C3X_RENDERER_RESULT_PENDING)return result;
            return C3X_RENDERER_RESULT_PENDING;
        }
        if(client)client->flush();
        c3x_renderer_gpu_frame_v1 next={sizeof(next)};
        int result=render(request,&next,output);if(result!=C3X_RENDERER_RESULT_OK)return result;
        camera_ticket=0;camera_image=nullptr;
        return prepare_image(image,next,*output,demand.tile_count?demand.tiles[0].anchor_x:0,demand.tile_count?demand.tiles[0].anchor_y:0);
    }
    int operation(int op,void* image,void* source,void const* from,void const* to,unsigned color){
        check_thread();
        // Native unit state/visual capture already ran before this call. Until
        // the first fresh map owns its canvas, there is no map body to compose
        // into; admitting the old raster unit path here would resurrect it.
        if(op==C3X_NATIVE_UNIT_DRAW&&scene_units&&(!adapter||!adapter->owns(image))){
            char line[160];std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=native-unit-capture result=1 accepted=0 reason=unowned-canvas front_ticket=%lld adapter=%u\n",
                static_cast<long long>(frame.ticket),unsigned(bool(adapter)));
            OutputDebugStringA(line);return 1;
        }
        if(!adapter)return 0;
        if(op==C3X_NATIVE_ZOOM_TARGET){
            if(color<c3x_renderer::SceneProjection::minimum_q16||color>c3x_renderer::SceneProjection::maximum_q16)return -1;
            auto sequence=++zoom_sequence;
            Command command={Kind::zoom_target,0,0,{}, {},int(sequence),0,color};
            client->submit(&command,1);client->flush();
            if(zoom_hint)zoom_hint(color,sequence);
            return 1;
        }
        if(op==C3X_NATIVE_HUD_BEGIN||op==C3X_NATIVE_HUD_END||op==C3X_NATIVE_UNIT_HUD_BEGIN){
            Command command={op==C3X_NATIVE_HUD_END?Kind::hud_end:Kind::hud_begin};
            if(op!=C3X_NATIVE_HUD_END){
                if(!from||(!adapter->owns(image)&&!adapter->admit(image)))return 0;
                auto anchor=static_cast<int const*>(from);
                command.detail=adapter->display_image(image);command.destination=adapter->image(image);
                command.source_x=anchor[0];command.source_y=anchor[1];command.color=color;
                if(op==C3X_NATIVE_UNIT_HUD_BEGIN){
                    if(anchor[2]<0||anchor[2]==INT_MAX)return -1;
                    command.source_height=anchor[2]+1;
                }
                if(to){auto offset=static_cast<int const*>(to);command.area.left=offset[0];command.area.top=offset[1];}
                command.source_width=int(adapter->transparency(image));
            }
            client->submit(&command,1);return 1;
        }
        if(op==C3X_NATIVE_UNIT_STATUS){
            if(native_map_hud)return 0;
            // Stage 4.3: the renderer draws this unit status in its open HUD
            // scope from draw_status's facts; nothing reaches the canvas.
            auto s=static_cast<c3x_renderer_unit_status_v1 const*>(from);
            // Accepted and refused reports by reason (1 facts, 2 canvas, 3 LED).
            auto refuse=[&](unsigned reason){++status_counts[reason];note_status();return 0;};
            if(!s||s->struct_size!=sizeof(*s)||s->max_hp<1||s->max_hp>100000||s->damage<0||s->damage>100000||
               (s->flags&~3u)||s->stack<0||s->stack>8||s->x<-32768||s->x>32767||s->y<-32768||s->y>32767)return refuse(1);
            if(!adapter->owns(image)&&!adapter->admit(image))return refuse(2);
            unsigned led_width=0,led_height=0;Id led=0;
            if(to&&!(led=adapter->ordinary_sprite(const_cast<void*>(to),image,led_width,led_height)))return refuse(3);
            ++status_counts[0];note_status();
            Command command={Kind::unit_status,adapter->image(image),led,{s->x,s->y,s->x,s->y},{},s->max_hp,s->damage,
                s->flags|(unsigned(s->stack)<<8)};
            command.source_width=int(led_width);command.source_height=int(led_height);
            client->submit(&command,1);return 1;
        }
        if(op==C3X_NATIVE_FIXED_UI_BEGIN||op==C3X_NATIVE_FIXED_UI_END){
            Command command={op==C3X_NATIVE_FIXED_UI_BEGIN?Kind::fixed_ui_begin:Kind::fixed_ui_end};
            if(op==C3X_NATIVE_FIXED_UI_BEGIN){
                if(!adapter->owns(image)&&!adapter->admit(image))return 0;
                command.detail=adapter->display_image(image);command.destination=adapter->image(image);
            }
            client->submit(&command,1);return 1;
        }
        if(op==C3X_NATIVE_WORLD_BEGIN||op==C3X_NATIVE_WORLD_END){
            int result=adapter->world_transfer(op,image,source)?1:0;
            if(world_reports++<8){char line[192];std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=world-view-boundary operation=%d accepted=%d source_owned=%u destination_owned=%u\n",
                op,result,unsigned(adapter->owns(source)),unsigned(adapter->owns(image)));OutputDebugStringA(line);}
            return result;
        }
        if(op==C3X_NATIVE_HIT_EXEMPT){
            // Civ III's form hit test never reads these canvases (injected
            // code states why). An unowned one has no coverage to drop yet;
            // the next declaration finds it once owned.
            for(void* p:{image,source})if(p&&adapter->owns(p))client->hit_exempt(adapter->image(p));
            return 1;
        }
        if(op==C3X_NATIVE_HIT_PIXEL){
            if(!scene_units||!adapter->owns(image))return 0;
            if(!from||!to)throw std::runtime_error("missing form input query");
            auto point=static_cast<int const*>(from);unsigned value=0;
            if(!client->hit_pixel(adapter->image(image),point[0],point[1],value))
                throw std::runtime_error("missing owned form input coverage");
            *const_cast<unsigned*>(static_cast<unsigned const*>(to))=value;return 1;
        }
        if(op==C3X_NATIVE_LINE_TARGET)return tactical&&adapter->owns(image)?1:0;
        if(op==C3X_NATIVE_STROKE){
            auto p=static_cast<c3x_renderer_native_stroke const*>(from);
            if(!tactical||!p||p->width<1||p->width>128||p->dash<0||p->dash>2||
                p->x1<-32768||p->x1>32767||p->y1<-32768||p->y1>32767||
                p->x2<-32768||p->x2>32767||p->y2<-32768||p->y2>32767){
                adapter->operation(C3X_NATIVE_DC,image,nullptr,nullptr,nullptr,0);return 0;
            }
            // The first actual stroke is destination demand, just like a
            // copy or HUD blend. Waiting for an earlier GPU write would send
            // an eligible future map canvas through native DC acquisition and
            // permanently sever its later animated map copies.
            if(!adapter->owns(image)&&!adapter->admit(image))return 0;
            // A solid one-pixel axis-aligned stroke is exactly the half-open
            // pixel run from its start toward its end (the tactical native
            // line: +.5 centres, hard caps, full coverage on its own row).
            // Fill it rather than rasterize a tactical capture; every city
            // label draws four (review 49). Exact for black and white.
            unsigned rgb=p->argb&0xffffffu;
            if(p->width==1&&!p->dash&&(p->argb>>24)==255&&(rgb==0||rgb==0xffffffu)&&(p->x1==p->x2)!=(p->y1==p->y2)){
                RECT run=p->y1==p->y2?RECT{p->x1<p->x2?p->x1:p->x2+1,p->y1,p->x1<p->x2?p->x2:p->x1+1,p->y1+1}:
                    RECT{p->x1,p->y1<p->y2?p->y1:p->y2+1,p->x1+1,p->y1<p->y2?p->y2:p->y1+1};
                if(adapter->operation(C3X_NATIVE_FILL,image,nullptr,nullptr,&run,rgb?0x80007fffu:0x80000000u)==1)return 1;
            }
            Tactical capture;capture.native_line(float(p->x1),float(p->y1),float(p->x2),float(p->y2),p->width,p->dash,p->argb);
            if(tactical_draw(image,capture))return 1;
            adapter->operation(C3X_NATIVE_DC,image,nullptr,nullptr,nullptr,0);return 0;
        }
        if(op==C3X_NATIVE_TACTICAL_ROUTE_BEGIN){
            if(!from||route_image||!tactical)return 0;
            route_view=*static_cast<c3x_renderer_tactical_view_v1 const*>(from);
            if(route_view.native_tile_width<=0||route_view.tile_width<64||route_view.tile_width>192)return 0;
            route_anchors.assign(to,color);
            route={};route_text.clear();route_image=image;destination={};return 1;
        }
        if(route_image==image && op==C3X_NATIVE_LINE){
            auto p=static_cast<int const*>(from);if(!p)throw std::runtime_error("route endpoints missing");
            auto a=route_point(p[0],p[1]),b=route_point(p[2],p[3]);
            if(trace_success){char line[256];std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=route-line native=%d,%d,%d,%d projected=%.2f,%.2f,%.2f,%.2f\n",
                p[0],p[1],p[2],p[3],a[0],a[1],b[0],b[1]);
                OutputDebugStringA(line);}
            route.line(a[0],a[1],b[0],b[1]);return 1;
        }
        if(route_image==image && op==C3X_NATIVE_TEXT){
            if(!source||color>32)throw std::runtime_error("route text missing/oversized");
            route_text.assign(static_cast<char const*>(source),color);return 1;
        }
        if(op==C3X_NATIVE_TACTICAL_TARGET){
            if(!from||route_image!=image)return 0;auto p=static_cast<int const*>(from);
            destination=route_point(p[0],p[1]);
            route.ring(destination[0],destination[1],float(route_view.native_tile_width),false);return 1;
        }
        if(op==C3X_NATIVE_TACTICAL_ROUTE_END){
            if(image!=route_image)return 0;route_image=nullptr;
            if(!route_text.empty())route.label(destination[0],destination[1],route_text,20.f);
            route.world_overlay=true;
            int result=tactical_draw(image,route,source);route={};route_text.clear();route_anchors.points.clear();return result;
        }
        if(op==C3X_NATIVE_TACTICAL_RING){
            // Resident units own their cursor in the same current scene, below
            // the body. A retained native ring would float over it and survive
            // old native erase rectangles after movement/selection changes.
            if(scene_units)return 1;
            if(!from)return 0;auto p=static_cast<int const*>(from);Tactical capture;
            capture.ring(float(p[0]),float(p[1]),float(p[2]),p[3]!=0);return tactical_draw(image,capture,source);
        }
        if(op==C3X_NATIVE_TACTICAL_GRID){
            if(!color)return 1;if(!from)return 0;
            auto const& view=*static_cast<c3x_renderer_frame_v1 const*>(from);Tactical capture;
            for(unsigned i=0;i<view.tile_count;++i){auto const& tile=view.tiles[i];
                if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER)||((tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_KNOWN)&&!(tile.tile_flags&C3X_RENDERER_TILE_EXPLORED)))continue;
                float x=float(tile.anchor_x),y=float(tile.anchor_y),w=float(view.tile_width),h=float(view.tile_height);
                // Each native diamond owns its top two edges, so shared edges
                // are drawn once. Hidden cells cannot introduce grid geometry.
                capture.line(x,y+h*.5f,x+w*.5f,y,.9f,true);
                capture.line(x+w*.5f,y,x+w,y+h*.5f,.9f,true);
            }
            return tactical_draw(image,capture);
        }
        if(op==C3X_NATIVE_IMAGE_PRESENT){
            if(color>1)throw std::runtime_error("invalid native presentation requirement");
            display_native=image;
            auto id=adapter->display_image(image);
            if(!id){
                // Full-color allocation can fail even for an owned surface.
                // Materialize it before the caller's private CPU snapshot;
                // ordinary CPU UI sources simply pass through this barrier.
                adapter->operation(C3X_NATIVE_BITS,image,nullptr,nullptr,nullptr,0);
                if(color)throw std::runtime_error("required native display image unavailable");
                OutputDebugStringA("[C3X renderer] stage=native-ui-present result=0 reason=missing-display-image\n");return 0;
            }
            if(!source)throw std::runtime_error("native transfer has no Graphsy owner");
            auto window=c3x_native_access::window(source);
            int width=field(image,0x38),height=field(image,0x3c);
            RECT rect=from?*static_cast<RECT const*>(from):RECT{0,0,width,height};
            c3x_renderer_gpu_present_v1 r={sizeof(r)};r.ticket=frame.ticket;r.image=std::int64_t(id);r.window=window;
            r.width=width;r.height=height;r.area[0]=rect.left;r.area[1]=rect.top;r.area[2]=rect.right;r.area[3]=rect.bottom;
            r.action=color?3:0;
            client->flush();auto result=present(&r);
            if(trace_success || result!=C3X_RENDERER_RESULT_OK){
                char line[256];std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=native-ui-present result=%d ticket=%lld map_image=%lld display_image=%lld area=%ld,%ld,%ld,%ld\n",
                result,static_cast<long long>(frame.ticket),static_cast<long long>(frame.map_image),
                static_cast<long long>(id),rect.left,rect.top,rect.right,rect.bottom);
                OutputDebugStringA(line);
            }
            if(result==C3X_RENDERER_RESULT_OK)return 1;
            // Renderer64 owns these pixels. A window admission failure must
            // not request a forbidden map readback and poison the image queue.
            if(scene_units)throw std::runtime_error("asynchronous native presentation failed");
            // Admission rejection is safe only after the actual display and
            // current native source have separately returned to CPU ownership.
            release_window();adapter->operation(C3X_NATIVE_DC,image,nullptr,nullptr,nullptr,0);return 0;
        }
        if(op==C3X_NATIVE_UNIT_DRAW){
            if(!from||!to)return 0;
            if(scene_units){
                // The fresh scene owns the map body and depth. Forward only the
                // copied native identity/pose; pending UI commands keep their
                // normal order and are flushed at the display boundary.
                c3x_renderer_gpu_unit_v1 target={sizeof(target)};
                target.ticket=frame.ticket;target.destination=frame.map_image;
                target.background=frame.map_image;target.clip[2]=frame.width;
                target.clip[3]=frame.height;target.playback_flags=color;
                int result=unit(static_cast<c3x_renderer_unit_v1 const*>(from),&target,
                    const_cast<int*>(static_cast<int const*>(to)));
                if(trace_success || result!=C3X_RENDERER_RESULT_OK){
                    char line[160];std::snprintf(line,sizeof(line),
                    "[C3X renderer] stage=native-unit-capture result=%d accepted=%u front_ticket=%lld\n",
                    result,unsigned(result==C3X_RENDERER_RESULT_OK),static_cast<long long>(frame.ticket));
                    OutputDebugStringA(line);
                }
                if(result!=C3X_RENDERER_RESULT_OK)
                    throw std::runtime_error("fresh map unit capture failed");
                return 1;
            }
            return adapter->draw_unit(unit,frame.ticket,*static_cast<c3x_renderer_unit_v1 const*>(from),image,source,
                const_cast<int*>(static_cast<int const*>(to)),color)?1:0;
        }
        bool full_copy=op==C3X_NATIVE_COPY&&from&&to&&source&&image&&
            field(source,0x38)>=640&&field(source,0x3c)>=480&&
            field(image,0x38)>=640&&field(image,0x3c)>=480;
        bool source_owned=full_copy&&adapter->owns(source),destination_owned=full_copy&&adapter->owns(image);
        int result=adapter->operation(op,image,source,from,to,color);
        if(full_copy&&surface_copy_reports++<32){
            char line[320];std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=native-surface-copy result=%d front_ticket=%lld source_front=%u destination_front=%u source_display=%u destination_display=%u source_owned=%u destination_owned=%u from=%ld,%ld,%ld,%ld to=%ld,%ld,%ld,%ld\n",
                result,static_cast<long long>(frame.ticket),unsigned(source==front_native),unsigned(image==front_native),
                unsigned(source==display_native),unsigned(image==display_native),unsigned(source_owned),unsigned(destination_owned),
                static_cast<RECT const*>(from)->left,static_cast<RECT const*>(from)->top,
                static_cast<RECT const*>(from)->right,static_cast<RECT const*>(from)->bottom,
                static_cast<RECT const*>(to)->left,static_cast<RECT const*>(to)->top,
                static_cast<RECT const*>(to)->right,static_cast<RECT const*>(to)->bottom);
            OutputDebugStringA(line);
        }
        if(op==C3X_NATIVE_FILL&&image==front_native&&to&&surface_fill_reports++<16){
            auto const* bounds=static_cast<RECT const*>(to);char line[192];
            std::snprintf(line,sizeof(line),
                "[C3X renderer] stage=native-front-fill result=%d front_ticket=%lld owned=%u color=%u area=%ld,%ld,%ld,%ld\n",
                result,static_cast<long long>(frame.ticket),unsigned(adapter->owns(image)),color,
                bounds->left,bounds->top,bounds->right,bounds->bottom);
            OutputDebugStringA(line);
        }
        return result;
    }
    void drain(){
        check_thread();
        // Retire all unpublished state even if preserving the native display
        // fails. Retry may release ownership, never revive the cancelled view.
        map(C3X_NATIVE_MAP_CANCEL,nullptr,nullptr,nullptr);
        route={};route_image=nullptr;route_text.clear();
        if(client){client->flush();release_window();
            // Renderer64 retires its GPU surfaces; config-off/menu transitions
            // repaint native content. They never read the custom map into JGL.
            if(!scene_units)adapter->drain();
            adapter.reset();client.reset();}
        front_native=nullptr;display_native=nullptr;
    }
    void abandon(){
        check_thread();
        if(adapter){adapter->abandon();adapter.reset();}
        client.reset();pending=nullptr;camera_ticket=0;camera_image=nullptr;front_native=nullptr;display_native=nullptr;
        route={};route_image=nullptr;route_text.clear();navigation.clear();
    }
};
}
