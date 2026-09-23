#pragma once
#include "helper_trial/scene_client.h"
#include "input_recording/journal.h"
#include "input_recording/runtime.h"
#include "remote_scene_output.h"

namespace c3x_remote_scene {
struct SharedFrame {std::uint64_t handle=0;unsigned width=0,height=0;};

// The x86 side retains values and native image ownership; all renderer scene
// state and GPU image identities are owned by the x64 helper. Borrowed output
// arrays remain valid until the next successful scene or camera adoption.
class Client {
    c3x_helper_trial::SceneClient transport;
    Output scene_result;
    CameraOutput camera_result;
    static c3x_inputs::Bytes reply(c3x_helper_trial::Wire const& wire){
        c3x_inputs::require(wire.reply_size<=c3x_helper_trial::wire_capacity,"remote reply limit");
        return {wire.payload,wire.payload+wire.reply_size};
    }
    c3x_helper_trial::Wire const& invoke(unsigned kind,unsigned subtype,unsigned char const* bytes,unsigned count,
                                        bool shared_frame=false,bool raw_shared=false,
                                        std::int64_t ticks=0,std::int64_t frequency=0){
        bool replay_override=false;
        auto* clock=c3x_inputs::replay_clock();
        if(!frequency&&clock&&!c3x_inputs::replay_execution().performance&&
           !c3x_inputs::realtime_replay().enabled&&clock->at<clock->values.size()){
            replay_override=clock->sample(ticks,frequency);
        }
        return transport.call_live(kind,subtype,bytes,count,shared_frame,raw_shared,ticks,frequency,replay_override);
    }
public:
    Client(std::wstring const& helper,std::wstring const& dll):transport(helper,dll){}
    c3x_helper_trial::SceneClient::Stats stats()const{return transport.stats();}
    int definitions(char const* root,char const* fallback,char const* scenario,char const* custom){
        c3x_inputs::Writer input;
        input.string(root,32768);input.string(fallback,32768);
        input.string(scenario,32768);input.string(custom,32768);
        return int(invoke(unsigned(c3x_inputs::Kind::native_bridge),8,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int pack(char const* path){
        c3x_inputs::Writer input;input.string(path,32768);
        return int(invoke(unsigned(c3x_inputs::Kind::native_bridge),7,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int reset(){
        return int(invoke(unsigned(c3x_inputs::Kind::native_bridge),6,nullptr,0).code);
    }
    int set_units(int enabled){
        c3x_inputs::Writer input;input(std::int32_t(enabled));
        return int(invoke(unsigned(c3x_inputs::Kind::configuration),3,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int visual_policy(unsigned policy){
        c3x_inputs::Writer input;input(std::int32_t(C3X_NATIVE_VISUAL_POLICY));
        input.u32(0);input.u32(0);input.u32(policy);
        return int(invoke(unsigned(c3x_inputs::Kind::native_bridge),1,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int render(c3x_renderer_camera_request_v1 const& request,c3x_renderer_gpu_frame_v1& gpu,
               c3x_renderer_output_v1& output){
        c3x_inputs::Writer input;auto identity=request.identity;
        c3x_inputs::c3x_renderer_camera_identity_v1_fields(input,identity);
        c3x_inputs::frame(input,*request.frame);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::scene),3,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        auto bytes=reply(response);c3x_inputs::Reader reader{bytes};decode(reader,scene_result);
        gpu=scene_result.gpu;output=scene_result.value;return C3X_RENDERER_RESULT_OK;
    }
    int render_cpu(c3x_renderer_frame_v1 const& frame,c3x_renderer_camera_identity_v1 const* identity,
                   c3x_renderer_output_v1& output){
        c3x_inputs::Writer input;
        if(identity){auto value=*identity;c3x_inputs::c3x_renderer_camera_identity_v1_fields(input,value);}
        c3x_inputs::frame(input,frame);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::scene),identity?2:1,
            input.bytes.data(),unsigned(input.bytes.size()));
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        auto bytes=reply(response);c3x_inputs::Reader reader{bytes};decode(reader,scene_result);
        output=scene_result.value;return C3X_RENDERER_RESULT_OK;
    }
    int camera_begin(c3x_renderer_camera_request_v1 const& request,c3x_renderer_i64& ticket){
        c3x_inputs::Writer input;auto identity=request.identity;
        c3x_inputs::c3x_renderer_camera_identity_v1_fields(input,identity);
        c3x_inputs::frame(input,*request.frame);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::camera),1,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code==C3X_RENDERER_RESULT_PENDING)ticket=response.recorded_ticket;
        return int(response.code);
    }
    int camera_poll(c3x_renderer_i64 ticket,c3x_renderer_gpu_camera_view_v1& view){
        c3x_inputs::Writer input;input(ticket);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::camera),3,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        auto bytes=reply(response);c3x_inputs::Reader reader{bytes};decode_camera(reader,camera_result);
        view=camera_result.value;return C3X_RENDERER_RESULT_OK;
    }
    int camera_cancel(c3x_renderer_i64 ticket){
        c3x_inputs::Writer input;input(ticket);
        return int(invoke(unsigned(c3x_inputs::Kind::camera),4,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int world_query(c3x_renderer_world_page_v1& page){
        auto const& response=invoke(unsigned(c3x_inputs::Kind::world_page),1,nullptr,0);
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        auto bytes=reply(response);c3x_inputs::Reader input{bytes};
        page={};page.struct_size=sizeof(page);page.frame.api_version=C3X_RENDERER_API_VERSION;
        page.frame.struct_size=sizeof(page.frame);
        input(page.first);input(page.capacity);
        c3x_inputs::c3x_renderer_camera_identity_v1_fields(input,page.identity);
        c3x_inputs::frame_fields(input,page.frame);input.done();
        c3x_inputs::require(page.capacity==128,"remote world page capacity");
        return C3X_RENDERER_RESULT_OK;
    }
    int world_submit(c3x_renderer_world_page_v1 const& page,int callback_result){
        c3x_inputs::require(page.count<=page.capacity&&page.count<=128&&
            (page.count==0||page.tiles),"remote world page bounds");
        c3x_inputs::Writer input;input(page.first);input(page.capacity);input(page.count);
        auto identity=page.identity;
        auto frame=page.frame;
        c3x_inputs::c3x_renderer_camera_identity_v1_fields(input,identity);
        c3x_inputs::frame_fields(input,frame);input(std::int32_t(callback_result));
        for(unsigned n=0;n<page.count;++n){auto tile=page.tiles[n];
            c3x_inputs::c3x_renderer_tile_v1_fields(input,tile);}
        return int(invoke(unsigned(c3x_inputs::Kind::world_page),2,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int images(c3x_renderer_gpu_images_v1 const& request,c3x_renderer_gpu_result_v1& result,
               unsigned* pixels,unsigned capacity){
        c3x_inputs::Writer input;c3x_inputs::images(input,request);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::image_commands),0,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        result={sizeof(result)};result.image=response.result_image;result.pixel_count=response.result_pixels;
        result.resident_bytes=response.resident_bytes;result.uploads=response.uploads;
        result.commands=response.commands;result.readbacks=response.readbacks;
        if(request.action==C3X_GPU_READBACK){
            c3x_inputs::require(pixels&&capacity>=result.pixel_count&&
                response.reply_size==result.pixel_count*4,"remote readback extent");
            std::memcpy(pixels,response.payload,response.reply_size);
        }
        return C3X_RENDERER_RESULT_OK;
    }
    int unit(c3x_renderer_unit_v1 const& unit,c3x_renderer_gpu_unit_v1 const& target,int* bounds){
        c3x_inputs::Writer input;c3x_inputs::unit(input,unit);auto value=target;
        c3x_inputs::target_fields(input,value);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::unit),1,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code==C3X_RENDERER_RESULT_OK)for(unsigned n=0;n<4;++n)bounds[n]=response.bounds[n];
        return int(response.code);
    }
    int unit_cpu(c3x_renderer_unit_v1 const& unit,unsigned flags,int* bounds,
                 std::vector<std::uint32_t>& pixels,int& x,int& y,unsigned& width,unsigned& height){
        c3x_inputs::Writer input;c3x_inputs::unit(input,unit);input(flags);input.u32(bounds?1:0);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::unit),2,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        auto bytes=reply(response);c3x_inputs::Reader reader{bytes};
        reader(x);reader(y);width=reader.u32();height=reader.u32();
        c3x_inputs::require(width<=1024&&height<=1024,"remote CPU unit extent");
        int returned[4]={};for(auto& value:returned)reader(value);
        pixels.resize(std::size_t(width)*height);for(auto& value:pixels)reader(value);
        reader.done();if(bounds)std::copy(std::begin(returned),std::end(returned),bounds);
        return C3X_RENDERER_RESULT_OK;
    }
    void forget_unit(int id){
        c3x_inputs::Writer input;input(std::int32_t(id));
        invoke(unsigned(c3x_inputs::Kind::unit_forget),0,input.bytes.data(),unsigned(input.bytes.size()));
    }
    int tactical(c3x_renderer::tactical::Input const& capture,c3x_renderer_gpu_unit_v1 const& target){
        c3x_inputs::Writer input;auto value=target;c3x_inputs::target_fields(input,value);
        c3x_inputs::tactical(input,capture);
        return int(invoke(unsigned(c3x_inputs::Kind::tactical),0,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int present(c3x_renderer_gpu_present_v1 const& value,SharedFrame& frame){
        c3x_inputs::Writer input;input(value.action);input(value.ticket);input(value.image);
        input(value.width);input(value.height);for(auto x:value.area)input(x);input.u32(1);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::presentation),0,input.bytes.data(),
            unsigned(input.bytes.size()),value.action==0);
        frame={response.shared_handle,response.width,response.height};return int(response.code);
    }
    int visual(std::int64_t ticks,std::int64_t frequency,SharedFrame& frame){
        c3x_inputs::Writer input;input.u32(1);
        auto const& response=invoke(unsigned(c3x_inputs::Kind::visual),1,input.bytes.data(),
            unsigned(input.bytes.size()),true,false,ticks,frequency);
        frame={response.shared_handle,response.width,response.height};return int(response.code);
    }
};
}
