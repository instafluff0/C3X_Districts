#pragma once
#include "gpu_frame_api.h"
#include <cstring>
#include <vector>

namespace c3x_native_images {
// One caller-thread navigation transaction. Only copied scene values travel to
// the worker; the native camera remains the displayed camera until poll succeeds.
class Navigation {
    custom_renderer_native_view requested={};
    c3x_renderer_frame_v1 captured={};
    c3x_renderer_camera_identity_v1 identity={};
    std::vector<c3x_renderer_tile_v1> tiles;
    std::vector<c3x_renderer_u32> topology;
    c3x_renderer_gpu_camera_view_v1 ready={};
    c3x_renderer_i64 ticket=0;
    void* destination=nullptr;
    bool offered=false;
    bool matches(c3x_renderer_camera_request_v1 const& demand)const {
        auto a=*demand.frame,b=captured;
        a.tiles=b.tiles=nullptr;a.world_topology=b.world_topology=nullptr;
        a.presentation_time_ticks=b.presentation_time_ticks=0;
        a.dirty_flags=b.dirty_flags=0;
        a.visible_animation_count=b.visible_animation_count=0;
        return !std::memcmp(&identity,&demand.identity,sizeof(identity)) &&
            !std::memcmp(&a,&b,sizeof(a)) &&
            (!a.tile_count || !std::memcmp(demand.frame->tiles,tiles.data(),tiles.size()*sizeof(tiles[0]))) &&
            (!a.world_topology_count || !std::memcmp(demand.frame->world_topology,topology.data(),topology.size()*sizeof(topology[0])));
    }
public:
    bool active()const{return ticket!=0;}
    bool available()const{return offered;}
    static bool same_projection(custom_renderer_native_view const& a,custom_renderer_native_view const& b){
        return a.width==b.width && a.height==b.height && a.tile_width==b.tile_width &&
            a.native_width==b.native_width && a.translate_x==b.translate_x && a.translate_y==b.translate_y;
    }
    void clear(){ticket=0;destination=nullptr;offered=false;ready={};tiles.clear();topology.clear();captured={};}
    template<class Owner> int request(Owner& owner,void* image,custom_renderer_native_view const& view,c3x_renderer_camera_request_v1 const& demand){
        if(active()&&!offered&&destination==image && !std::memcmp(&requested,&view,sizeof(view))&&matches(demand))
            return C3X_RENDERER_RESULT_PENDING; // Repeated edge-scroll demand must not starve the worker.
        c3x_renderer_i64 next=0;
        int result=owner.request_camera(image,demand,next);
        if(result!=C3X_RENDERER_RESULT_PENDING)return result;
        requested=view;captured=*demand.frame;identity=demand.identity;ticket=next;destination=image;offered=false;
        tiles.clear();if(captured.tile_count)tiles.assign(captured.tiles,captured.tiles+captured.tile_count);
        topology.clear();if(captured.world_topology_count)topology.assign(captured.world_topology,captured.world_topology+captured.world_topology_count);
        captured.tiles=nullptr;captured.world_topology=nullptr;
        return result;
    }
    template<class Owner> int poll(Owner& owner,int action,void* image,custom_renderer_native_view& current){
        if(!active())return C3X_RENDERER_RESULT_SUPERSEDED;
        if(action==C3X_NAV_DISCARD || image!=destination || !same_projection(current,requested)){
            owner.map(C3X_NATIVE_MAP_CANCEL,image,nullptr,nullptr);return C3X_RENDERER_RESULT_SUPERSEDED;
        }
        auto next=requested;
        if(action==C3X_NAV_BARRIER){
            owner.map(C3X_NATIVE_MAP_CANCEL,image,nullptr,nullptr);current=next;return C3X_RENDERER_RESULT_OK;
        }
        if(offered)return C3X_RENDERER_RESULT_SUPERSEDED;
        c3x_renderer_gpu_camera_view_v1 result={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(result)};
        int code=owner.poll_camera(image,ticket,result);
        if(code==C3X_RENDERER_RESULT_PENDING)return code;
        if(code==C3X_RENDERER_RESULT_OK){ready=result;offered=true;current=next;return code;}
        // Admission loss or supersession needs the exact native path at the
        // intended destination. It never grants ready coverage.
        owner.map(C3X_NATIVE_MAP_CANCEL,image,nullptr,nullptr);current=next;return C3X_RENDERER_RESULT_OK;
    }
    bool take(void* image,c3x_renderer_camera_request_v1 const& demand,c3x_renderer_output_v1& output){
        if(!offered)return false;
        bool valid=image==destination && matches(demand);
        if(valid)output=ready.camera.output;
        clear();return valid;
    }
};
}
