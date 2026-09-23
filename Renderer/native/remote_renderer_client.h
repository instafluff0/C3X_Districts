#pragma once
#include "helper_trial/scene_client.h"
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
public:
    Client(std::wstring const& helper,std::wstring const& dll):transport(helper,dll){}
    int definitions(char const* root,char const* fallback,char const* scenario,char const* custom){
        c3x_inputs::Writer input;
        input.string(root,32768);input.string(fallback,32768);
        input.string(scenario,32768);input.string(custom,32768);
        return int(transport.call_live(unsigned(c3x_inputs::Kind::native_bridge),8,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int reset(){
        return int(transport.call_live(unsigned(c3x_inputs::Kind::native_bridge),6,nullptr,0).code);
    }
    int set_units(int enabled){
        c3x_inputs::Writer input;input(std::int32_t(enabled));
        return int(transport.call_live(unsigned(c3x_inputs::Kind::configuration),3,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int visual_policy(unsigned policy){
        c3x_inputs::Writer input;input(std::int32_t(C3X_NATIVE_VISUAL_POLICY));
        input.u32(0);input.u32(0);input.u32(policy);
        return int(transport.call_live(unsigned(c3x_inputs::Kind::native_bridge),1,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int render(c3x_renderer_camera_request_v1 const& request,c3x_renderer_gpu_frame_v1& gpu,
               c3x_renderer_output_v1& output){
        c3x_inputs::Writer input;auto identity=request.identity;
        c3x_inputs::c3x_renderer_camera_identity_v1_fields(input,identity);
        c3x_inputs::frame(input,*request.frame);
        auto const& response=transport.call_live(unsigned(c3x_inputs::Kind::scene),3,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        auto bytes=reply(response);c3x_inputs::Reader reader{bytes};decode(reader,scene_result);
        gpu=scene_result.gpu;output=scene_result.value;return C3X_RENDERER_RESULT_OK;
    }
    int camera_begin(c3x_renderer_camera_request_v1 const& request,c3x_renderer_i64& ticket){
        c3x_inputs::Writer input;auto identity=request.identity;
        c3x_inputs::c3x_renderer_camera_identity_v1_fields(input,identity);
        c3x_inputs::frame(input,*request.frame);
        auto const& response=transport.call_live(unsigned(c3x_inputs::Kind::camera),1,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code==C3X_RENDERER_RESULT_PENDING)ticket=response.recorded_ticket;
        return int(response.code);
    }
    int camera_poll(c3x_renderer_i64 ticket,c3x_renderer_gpu_camera_view_v1& view){
        c3x_inputs::Writer input;input(ticket);
        auto const& response=transport.call_live(unsigned(c3x_inputs::Kind::camera),3,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code!=C3X_RENDERER_RESULT_OK)return int(response.code);
        auto bytes=reply(response);c3x_inputs::Reader reader{bytes};decode_camera(reader,camera_result);
        view=camera_result.value;return C3X_RENDERER_RESULT_OK;
    }
    int camera_cancel(c3x_renderer_i64 ticket){
        c3x_inputs::Writer input;input(ticket);
        return int(transport.call_live(unsigned(c3x_inputs::Kind::camera),4,input.bytes.data(),
            unsigned(input.bytes.size())).code);
    }
    int images(c3x_renderer_gpu_images_v1 const& request,c3x_renderer_gpu_result_v1& result,
               unsigned* pixels,unsigned capacity){
        c3x_inputs::Writer input;c3x_inputs::images(input,request);
        auto const& response=transport.call_live(unsigned(c3x_inputs::Kind::image_commands),0,input.bytes.data(),
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
        auto const& response=transport.call_live(unsigned(c3x_inputs::Kind::unit),1,input.bytes.data(),
            unsigned(input.bytes.size()));
        if(response.code==C3X_RENDERER_RESULT_OK)for(unsigned n=0;n<4;++n)bounds[n]=response.bounds[n];
        return int(response.code);
    }
    int present(c3x_renderer_gpu_present_v1 const& value,SharedFrame& frame){
        c3x_inputs::Writer input;input(value.action);input(value.ticket);input(value.image);
        input(value.width);input(value.height);for(auto x:value.area)input(x);input.u32(1);
        auto const& response=transport.call_live(unsigned(c3x_inputs::Kind::presentation),0,input.bytes.data(),
            unsigned(input.bytes.size()),value.action==0);
        frame={response.shared_handle,response.width,response.height};return int(response.code);
    }
    int visual(std::int64_t ticks,std::int64_t frequency,SharedFrame& frame){
        c3x_inputs::Writer input;input.u32(1);
        auto const& response=transport.call_live(unsigned(c3x_inputs::Kind::visual),1,input.bytes.data(),
            unsigned(input.bytes.size()),true,false,ticks,frequency);
        frame={response.shared_handle,response.width,response.height};return int(response.code);
    }
};
}
