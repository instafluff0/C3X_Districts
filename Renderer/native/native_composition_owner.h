#pragma once
#include "native_image_adapter.h"
#include "gpu_image_worker_client.h"
#include <memory>
#include "tactical_overlay.h"
#include <functional>
namespace c3x_native_images {
// Caller-thread owner for the native map/copy/save/display family. The existing
// renderer worker owns GPU work; this object never retains game request pointers.
class CompositionOwner {
    c3x_renderer_gpu_render_fn render;
    c3x_renderer_gpu_images_fn images;
    c3x_renderer_gpu_present_fn present;
    c3x_renderer_gpu_unit_fn unit;
    c3x_renderer_native_lifetime_fn lifetime;
    void* bits;void* release;
    DWORD thread=GetCurrentThreadId();
    std::unique_ptr<c3x_gpu_images::WorkerClient> client;
    std::unique_ptr<Adapter<c3x_gpu_images::WorkerClient>> adapter;
    c3x_renderer_gpu_frame_v1 frame={sizeof(frame)};
    using Tactical=c3x_renderer::tactical::Input;
    std::function<int(Tactical const&,c3x_renderer_gpu_unit_v1 const&)> tactical;
    Tactical route;void* route_image=nullptr;c3x_renderer_tactical_view_v1 route_view={};
    std::array<float,2> destination={};std::string route_text;
    float projected(int v,bool y)const{return float(double(v)*route_view.tile_width/route_view.native_tile_width+
        double(y?route_view.translate_y_fp:route_view.translate_x_fp)/65536.);}
    int tactical_draw(void* image,Tactical const& capture){
        if(capture.primitives.empty())return 1;
        if(!tactical)return 0;
        return adapter->draw_tactical([&](auto const& target){return tactical(capture,target);},frame.ticket,image)?1:0;
    }
    void* pending=nullptr;Rect area={};int phase_x=0,phase_y=0;
    void check_thread(){if(GetCurrentThreadId()!=thread)throw std::runtime_error("native composition caller changed");}
    void release_window(){c3x_renderer_gpu_present_v1 r={sizeof(r)};r.action=2;
        if(present(&r)!=C3X_RENDERER_RESULT_OK)throw std::runtime_error("native display handoff failed");}
    static int field(void* p,unsigned offset){return *reinterpret_cast<int*>(static_cast<char*>(p)+offset);}
public:
    CompositionOwner(c3x_renderer_gpu_render_fn r,c3x_renderer_gpu_images_fn i,c3x_renderer_gpu_present_fn p,
        c3x_renderer_gpu_unit_fn u,c3x_renderer_native_lifetime_fn l,void* b,void* end):render(r),images(i),present(p),unit(u),lifetime(l),bits(b),release(end){}
    void set_tactical(std::function<int(Tactical const&,c3x_renderer_gpu_unit_v1 const&)> draw){tactical=std::move(draw);}
    bool active()const{return adapter!=nullptr;}
    c3x_renderer_i64 sample_ticks()const{return frame.presentation_time_ticks;}
    // Native validation occurs between prepare and commit. Preparation never
    // inserts pixels or claims category replacement on the game's behalf.
    int map(int action,void* image,c3x_renderer_camera_request_v1 const* request,c3x_renderer_output_v1* output){
        check_thread();
        if(action==C3X_NATIVE_MAP_CANCEL){pending=nullptr;return C3X_RENDERER_RESULT_OK;}
        if(action==C3X_NATIVE_MAP_COMMIT){
            if(!pending||image!=pending||!adapter)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            pending=nullptr;
            if(!adapter->insert_map(image,Id(frame.map_image),area,area.left,area.top,phase_x,phase_y))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            client->flush();return C3X_RENDERER_RESULT_OK;
        }
        if(action!=C3X_NATIVE_MAP_PREPARE||!request||!request->frame||!output||pending)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(!lifetime(C3X_NATIVE_MAP,image,0)||field(image,0x24)!=16||field(image,0x4c4)||field(image,0x4c8))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        auto const& demand=*request->frame;
        if(field(image,0x38)!=demand.target_width||field(image,0x3c)!=demand.target_height)return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(adapter&&!adapter->admit(image))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(client)client->flush();
        c3x_renderer_gpu_frame_v1 next={sizeof(next)};
        int result=render(request,&next,output);if(result!=C3X_RENDERER_RESULT_OK)return result;
        if(client)client->advance(next);
        else {
            client=std::make_unique<c3x_gpu_images::WorkerClient>(images,next);
            adapter=std::make_unique<Adapter<c3x_gpu_images::WorkerClient>>(*client,bits,release,lifetime);
        }
        frame=next;
        if(!adapter->admit(image))return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        pending=image;area={output->clip_left,output->clip_top,output->clip_right,output->clip_bottom};
        phase_x=demand.tile_count?demand.tiles[0].anchor_x:0;phase_y=demand.tile_count?demand.tiles[0].anchor_y:0;
        return C3X_RENDERER_RESULT_OK;
    }
    int operation(int op,void* image,void* source,void const* from,void const* to,unsigned color){
        check_thread();if(!adapter)return 0;
        if(op==C3X_NATIVE_TACTICAL_ROUTE_BEGIN){
            if(!from||route_image||!tactical)return 0;
            route_view=*static_cast<c3x_renderer_tactical_view_v1 const*>(from);
            if(route_view.native_tile_width<=0||route_view.tile_width<64||route_view.tile_width>192)return 0;
            route={};route_text.clear();route_image=image;destination={};return 1;
        }
        if(route_image==image && op==C3X_NATIVE_LINE){
            auto p=static_cast<int const*>(from);if(!p)throw std::runtime_error("route endpoints missing");
            route.line(projected(p[0],false),projected(p[1],true),projected(p[2],false),projected(p[3],true));return 1;
        }
        if(route_image==image && op==C3X_NATIVE_TEXT){
            if(!source||color>32)throw std::runtime_error("route text missing/oversized");
            route_text.assign(static_cast<char const*>(source),color);return 1;
        }
        if(op==C3X_NATIVE_TACTICAL_TARGET){
            if(!from||route_image!=image)return 0;auto p=static_cast<int const*>(from);
            destination={projected(p[0],false),projected(p[1],true)};
            route.ring(destination[0],destination[1],float(route_view.tile_width),false);return 1;
        }
        if(op==C3X_NATIVE_TACTICAL_ROUTE_END){
            if(image!=route_image)return 0;route_image=nullptr;
            if(!route_text.empty())route.label(destination[0],destination[1],route_text,std::max(18.f,float(route_view.tile_width)*.26f));
            int result=tactical_draw(image,route);route={};route_text.clear();return result;
        }
        if(op==C3X_NATIVE_TACTICAL_RING){
            if(!from)return 0;auto p=static_cast<int const*>(from);Tactical capture;
            capture.ring(float(p[0]),float(p[1]),float(p[2]),p[3]!=0);return tactical_draw(image,capture);
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
            auto id=adapter->display_image(image);
            if(!id)return 0; // Caller uploads this CPU UI source into the same presenter.
            if(!source)throw std::runtime_error("native transfer has no Graphsy owner");
            auto dc=*reinterpret_cast<HDC*>(static_cast<char*>(source)+0x138);
            int width=field(image,0x38),height=field(image,0x3c);
            RECT rect=from?*static_cast<RECT const*>(from):RECT{0,0,width,height};
            c3x_renderer_gpu_present_v1 r={sizeof(r)};r.ticket=frame.ticket;r.image=std::int64_t(id);r.window=WindowFromDC(dc);
            r.width=width;r.height=height;r.area[0]=rect.left;r.area[1]=rect.top;r.area[2]=rect.right;r.area[3]=rect.bottom;
            client->flush();auto result=present(&r);
            if(result==C3X_RENDERER_RESULT_OK)return 1;
            // Admission rejection is safe only after the actual display and
            // current native source have separately returned to CPU ownership.
            release_window();adapter->operation(C3X_NATIVE_DC,image,nullptr,nullptr,nullptr,0);return 0;
        }
        if(op==C3X_NATIVE_UNIT_DRAW){
            if(!from||!to)return 0;
            return adapter->draw_unit(unit,frame.ticket,*static_cast<c3x_renderer_unit_v1 const*>(from),image,source,
                const_cast<int*>(static_cast<int const*>(to)),color)?1:0;
        }
        return adapter->operation(op,image,source,from,to,color);
    }
    void drain(){check_thread();pending=nullptr;if(client){client->flush();release_window();adapter->drain();adapter.reset();client.reset();}}
};
}
