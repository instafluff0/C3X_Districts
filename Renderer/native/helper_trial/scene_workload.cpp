#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <d3d11.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "../input_recording/journal.h"
#include "../asset_content_hash.h"
#include "../gpu_composition_session.h"
#include "../remote_scene_output.h"
#include "../visual_cadence.h"
#include "scene_wire.h"

using namespace c3x_inputs;
using namespace c3x_helper_trial;
namespace {
static_assert(sizeof(void*)==4||sizeof(void*)==8,"unsupported process architecture");
double milliseconds(){LARGE_INTEGER q={},f={};QueryPerformanceCounter(&q);QueryPerformanceFrequency(&f);return 1000.0*double(q.QuadPart)/double(f.QuadPart);}
std::uint64_t private_bytes(){PROCESS_MEMORY_COUNTERS_EX m={};m.cb=sizeof(m);if(!GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof(m)))throw std::runtime_error("memory sample failed");return m.PrivateUsage;}
std::wstring object_name(std::wstring const& base,wchar_t const* suffix){return base+suffix;}
struct Core {
    HMODULE module=nullptr;
    c3x_renderer_render_fn render=nullptr;
    c3x_renderer_render_view_fn render_view=nullptr;
    using GpuRender=int(*)(c3x_renderer_camera_request_v1 const*,c3x_renderer_gpu_frame_v1*,c3x_renderer_output_v1*);
    GpuRender gpu_render=nullptr;
    using CameraBegin=int(*)(c3x_renderer_camera_request_v1 const*,c3x_renderer_i64*);
    using CameraPoll=int(*)(c3x_renderer_i64,c3x_renderer_gpu_camera_view_v1*);
    using CameraCancel=int(*)(c3x_renderer_i64);
    CameraBegin camera_begin=nullptr;CameraPoll camera_poll=nullptr;CameraCancel camera_cancel=nullptr;
    using Definitions=int(*)(char const*,char const*,char const*,char const*);
    Definitions definitions=nullptr;
    using Pack=int(*)(char const*);Pack pack=nullptr;
    using Reset=void(*)();Reset reset=nullptr;
    c3x_renderer_gpu_images_fn images=nullptr;
    c3x_renderer_native_image_fn native_image=nullptr;
    using GpuUnit=int(*)(c3x_renderer_unit_v1 const*,c3x_renderer_gpu_unit_v1 const*,int*);
    GpuUnit unit_gpu=nullptr;
    using CpuUnit=int(*)(c3x_renderer_unit_v1 const*,unsigned,int,int*,int*,int*,unsigned*,unsigned*,std::uint32_t*,unsigned);
    CpuUnit unit_cpu=nullptr;
    using UnitForget=void(*)(int);UnitForget unit_forget=nullptr;
    using Tactical=int(*)(c3x_renderer::tactical::Input const*,c3x_renderer_gpu_unit_v1 const*);
    Tactical tactical_gpu=nullptr;
    using WorldQuery=int(*)(c3x_renderer_world_page_v1*);
    using WorldSubmit=int(*)(c3x_renderer_world_page_v1 const*,int);
    using WorldStatus=int(*)(c3x_renderer_world_status_v1*);
    WorldQuery world_query=nullptr;WorldSubmit world_submit=nullptr;
    WorldStatus world_status=nullptr;
    using SetUnits=int(*)(int);
    SetUnits set_units=nullptr;
    using TrialClock=void(*)(std::int64_t,std::int64_t);
    TrialClock set_clock=nullptr;
    using Shared=int(*)(c3x_renderer_i64,c3x_renderer_i64,DWORD,std::uint64_t*,unsigned*,unsigned*);
    Shared shared=nullptr,shared_raw=nullptr;bool verify_pixels=false;
    using PresentShared=int(*)(c3x_renderer_gpu_present_v1 const*,DWORD,std::uint64_t*,unsigned*,unsigned*);
    PresentShared present_shared=nullptr;
    using VisualShared=int(*)(std::int64_t,std::int64_t,DWORD,std::uint64_t*,unsigned*,unsigned*);
    VisualShared visual_shared=nullptr;
    using BindSurface=int(*)(std::uint64_t,unsigned,unsigned);
    BindSurface bind_surface=nullptr;bool direct_surface_bound=false,direct_display_ready=false;
    c3x_renderer::VisualCadence direct_cadence{
        std::chrono::microseconds(16667),std::chrono::milliseconds(2)};
    using SurfacePixels=int(*)(unsigned*,unsigned,unsigned*,unsigned*);
    SurfacePixels surface_pixels=nullptr;
    std::map<std::int64_t,std::int64_t> ticket_ids,image_ids;
    explicit Core(wchar_t const* dll,bool verify=false):verify_pixels(verify){module=LoadLibraryW(dll);require(module!=nullptr,"renderer DLL load failed");
        render=reinterpret_cast<c3x_renderer_render_fn>(GetProcAddress(module,"c3x_renderer_render"));
        render_view=reinterpret_cast<c3x_renderer_render_view_fn>(GetProcAddress(module,"c3x_renderer_render_view"));
        gpu_render=reinterpret_cast<GpuRender>(GetProcAddress(module,"c3x_renderer_gpu_render"));
        camera_begin=reinterpret_cast<CameraBegin>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
        camera_poll=reinterpret_cast<CameraPoll>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll_view"));
        camera_cancel=reinterpret_cast<CameraCancel>(GetProcAddress(module,"c3x_renderer_camera_cancel"));
        definitions=reinterpret_cast<Definitions>(GetProcAddress(module,"c3x_renderer_set_definition_paths"));
        pack=reinterpret_cast<Pack>(GetProcAddress(module,"c3x_renderer_set_pack_path"));
        reset=reinterpret_cast<Reset>(GetProcAddress(module,"c3x_renderer_reset"));
        images=reinterpret_cast<c3x_renderer_gpu_images_fn>(GetProcAddress(module,"c3x_renderer_gpu_images"));
        native_image=reinterpret_cast<c3x_renderer_native_image_fn>(GetProcAddress(module,"c3x_renderer_native_image"));
        unit_gpu=reinterpret_cast<GpuUnit>(GetProcAddress(module,"c3x_renderer_gpu_unit"));
        unit_cpu=reinterpret_cast<CpuUnit>(GetProcAddress(module,"c3x_renderer_trial_unit_pixels"));
        unit_forget=reinterpret_cast<UnitForget>(GetProcAddress(module,"c3x_renderer_unit_forget"));
        tactical_gpu=reinterpret_cast<Tactical>(GetProcAddress(module,"c3x_renderer_trial_tactical"));
        world_query=reinterpret_cast<WorldQuery>(GetProcAddress(module,"c3x_renderer_trial_world_query"));
        world_submit=reinterpret_cast<WorldSubmit>(GetProcAddress(module,"c3x_renderer_trial_world_submit"));
        world_status=reinterpret_cast<WorldStatus>(GetProcAddress(module,"c3x_renderer_world_status"));
        set_units=reinterpret_cast<SetUnits>(GetProcAddress(module,"c3x_renderer_set_unit_rendering"));
        set_clock=reinterpret_cast<TrialClock>(GetProcAddress(module,"c3x_renderer_trial_set_clock"));
        shared=reinterpret_cast<Shared>(GetProcAddress(module,"c3x_renderer_trial_export_shared"));
        shared_raw=reinterpret_cast<Shared>(GetProcAddress(module,"c3x_renderer_trial_export_shared_raw"));
        present_shared=reinterpret_cast<PresentShared>(GetProcAddress(module,"c3x_renderer_trial_present_shared"));
        visual_shared=reinterpret_cast<VisualShared>(GetProcAddress(module,"c3x_renderer_trial_visual_shared"));
        bind_surface=reinterpret_cast<BindSurface>(GetProcAddress(module,"c3x_renderer_trial_bind_surface"));
        surface_pixels=reinterpret_cast<SurfacePixels>(GetProcAddress(module,"c3x_renderer_trial_surface_pixels"));
        require(render&&render_view&&gpu_render&&camera_begin&&camera_poll&&camera_cancel&&definitions&&reset,"renderer DLL entries missing");}
    ~Core(){direct_cadence.stop();if(module){reset();auto trace_flush=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_trial_trace_flush"));
        if(trace_flush)trace_flush();FreeLibrary(module);}}
    void start_direct_cadence(){
        if(!direct_surface_bound||!direct_display_ready||!visual_shared||!native_image)return;
        char manual[4]={};
        if(GetEnvironmentVariableA("C3X_RENDERER_MANUAL_VISUAL",manual,sizeof(manual))==1&&manual[0]=='1')return;
        if(native_image(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,2)<=0)return;
        direct_cadence.enable([this]{
            LARGE_INTEGER now={},frequency={};
            if(!QueryPerformanceCounter(&now)||!QueryPerformanceFrequency(&frequency))return;
            std::uint64_t handle=0;unsigned width=0,height=0;
            int code=visual_shared(now.QuadPart,frequency.QuadPart,0,&handle,&width,&height);
            if(code==C3X_RENDERER_RESULT_ERROR||code==C3X_RENDERER_RESULT_DEVICE_ERROR){
                OutputDebugStringA("[C3X renderer] Renderer64 visual surface unavailable\n");
                direct_cadence.disable();
            }
        });
    }
    void stop_direct_cadence(){direct_cadence.stop();direct_display_ready=false;}
    void execute(Wire& wire){
        wire.status=0;wire.code=0;wire.executed=1;wire.reply_size=0;
        wire.width=wire.height=wire.rendered=wire.fallback=0;wire.shared_handle=0;
        wire.result_image=0;wire.result_pixels=0;
        wire.resident_bytes=wire.uploads=wire.commands=wire.readbacks=0;
        std::fill(std::begin(wire.bounds),std::end(wire.bounds),0);
        std::fill(std::begin(wire.hash),std::end(wire.hash),0);std::fill(std::begin(wire.gpu_hash),std::end(wire.gpu_hash),0);wire.gpu_hash_valid=0;
        try{
            require(wire.magic==wire_magic&&wire.version==wire_version&&wire.size<=wire_capacity&&
                wire.live<=1&&wire.replay_clock<=1,"invalid scene wire header");
            if(wire.replay_clock){require(set_clock!=nullptr,"helper lacks recorded clock entry");
                set_clock(wire.clock_ticks,wire.clock_frequency);}
            else if(wire.live&&set_clock)set_clock(0,0);
            Bytes bytes(wire.payload,wire.payload+wire.size);Reader in{bytes};auto started=milliseconds();
            if(wire.kind==unsigned(Kind::configuration)&&wire.subtype==3){
                int enabled=0;in(enabled);in.done();require(set_units!=nullptr,"helper lacks unit configuration");
                wire.code=unsigned(set_units(enabled));
            }else if(wire.kind==unsigned(Kind::native_bridge)&&wire.subtype==1){
                int operation=0;unsigned image=0,source=0,color=0;
                in(operation);in(image);in(source);in(color);in.done();
                require(operation==C3X_NATIVE_VISUAL_POLICY&&!image&&!source&&native_image,
                    "unsupported native policy message");
                if(color==0)stop_direct_cadence();
                wire.code=unsigned(native_image(operation,nullptr,nullptr,nullptr,nullptr,color));
                if(color==1&&wire.code>0)start_direct_cadence();
            }else if(wire.kind==unsigned(Kind::native_bridge)&&wire.subtype==6){
                in.done();stop_direct_cadence();reset();ticket_ids.clear();image_ids.clear();direct_surface_bound=false;wire.code=1;
            }else if(wire.kind==unsigned(Kind::native_bridge)&&wire.subtype==8){
                stop_direct_cadence();
                bool present[4]={};std::string paths[4];for(unsigned n=0;n<4;++n)paths[n]=in.string(32768,&present[n]);in.done();
                wire.code=unsigned(definitions(present[0]?paths[0].c_str():nullptr,present[1]?paths[1].c_str():nullptr,
                    present[2]?paths[2].c_str():nullptr,present[3]?paths[3].c_str():nullptr));
                direct_surface_bound=false;
                ticket_ids.clear();image_ids.clear();
            }else if(wire.live&&wire.kind==unsigned(Kind::native_bridge)&&wire.subtype==7){
                stop_direct_cadence();
                require(pack!=nullptr,"helper lacks pack configuration");
                bool present=false;auto path=in.string(32768,&present);in.done();
                wire.code=unsigned(pack(present?path.c_str():nullptr));
                direct_surface_bound=false;
                ticket_ids.clear();image_ids.clear();
            }else if(wire.live&&wire.kind==unsigned(Kind::camera)&&wire.subtype==1){
                c3x_renderer_camera_identity_v1 identity={};c3x_renderer_camera_identity_v1_fields(in,identity);
                Frame frame_value;frame(in,frame_value);in.done();
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame_value.value,identity};
                c3x_renderer_i64 ticket=0;wire.code=unsigned(camera_begin(&request,&ticket));
                wire.recorded_ticket=ticket;
            }else if(wire.live&&wire.kind==unsigned(Kind::camera)&&wire.subtype==3){
                c3x_renderer_i64 ticket=0;in(ticket);in.done();
                c3x_renderer_gpu_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
                wire.code=unsigned(camera_poll(ticket,&view));
                if(wire.code==C3X_RENDERER_RESULT_OK){
                    Writer response;
                    response(view.camera.ticket);
                    auto identity=view.camera.identity;c3x_renderer_camera_identity_v1_fields(response,identity);
                    frame(response,view.camera.frame);
                    c3x_remote_scene::encode(response,view.image,view.camera.output);
                    response(view.pixel_phase_x);response(view.pixel_phase_y);
                    require(response.bytes.size()<=wire_capacity,"remote camera result exceeds slot");
                    wire.reply_size=unsigned(response.bytes.size());
                    std::memcpy(wire.payload,response.bytes.data(),wire.reply_size);
                }
            }else if(wire.live&&wire.kind==unsigned(Kind::camera)&&wire.subtype==4){
                c3x_renderer_i64 ticket=0;in(ticket);in.done();wire.code=unsigned(camera_cancel(ticket));
            }else if(wire.live&&wire.kind==unsigned(Kind::world_page)&&wire.subtype==1){
                in.done();require(world_query!=nullptr,"helper lacks world query entry");
                c3x_renderer_world_page_v1 page={};page.struct_size=sizeof(page);
                wire.code=unsigned(world_query(&page));
                if(wire.code==C3X_RENDERER_RESULT_OK){
                    Writer response;response(page.first);response(page.capacity);
                    c3x_inputs::c3x_renderer_camera_identity_v1_fields(response,page.identity);
                    c3x_inputs::frame_fields(response,page.frame);
                    wire.reply_size=unsigned(response.bytes.size());
                    std::memcpy(wire.payload,response.bytes.data(),wire.reply_size);
                }
            }else if(wire.live&&wire.kind==unsigned(Kind::world_page)&&wire.subtype==2){
                require(world_submit!=nullptr,"helper lacks world submit entry");
                c3x_renderer_world_page_v1 page={};page.struct_size=sizeof(page);
                in(page.first);in(page.capacity);in(page.count);
                c3x_inputs::c3x_renderer_camera_identity_v1_fields(in,page.identity);
                c3x_inputs::frame_fields(in,page.frame);
                int callback_result=0;in(callback_result);
                require(page.count<=128,"remote world page limit");
                std::vector<c3x_renderer_tile_v1> tiles(page.count);
                for(auto& tile:tiles)c3x_inputs::c3x_renderer_tile_v1_fields(in,tile);
                in.done();page.tiles=tiles.data();
                wire.code=unsigned(world_submit(&page,callback_result));
            }else if(wire.live&&wire.kind==unsigned(Kind::world_page)&&wire.subtype==3){
                in.done();require(world_status!=nullptr,"helper lacks world status entry");
                c3x_renderer_world_status_v1 status={sizeof(status)};
                wire.code=unsigned(world_status(&status));
                if(wire.code==C3X_RENDERER_RESULT_OK){
                    Writer response;response(status.total);response(status.authoritative);
                    response(status.capture_cursor);response(status.regions);
                    response(status.prepared_regions);response(status.unavailable_regions);
                    response(status.capture_passes);response(status.appearance_sequence);
                    response(status.preparation_sequence);
                    wire.reply_size=unsigned(response.bytes.size());
                    std::memcpy(wire.payload,response.bytes.data(),wire.reply_size);
                }
            }else if(wire.kind==unsigned(Kind::scene)&&wire.subtype>=1&&wire.subtype<=3){
                c3x_renderer_camera_identity_v1 identity={};if(wire.subtype!=1)c3x_renderer_camera_identity_v1_fields(in,identity);
                Frame frame_value;frame(in,frame_value);in.done();
                c3x_renderer_output_v1 output={C3X_RENDERER_API_VERSION,sizeof(output)};
                c3x_renderer_gpu_frame_v1 gpu={sizeof(gpu)};
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame_value.value,identity};
                if(wire.subtype==1)wire.code=unsigned(render(&frame_value.value,&output));
                else if(wire.subtype==2)wire.code=unsigned(render_view(&request,&output));
                else{wire.code=unsigned(gpu_render(&request,&gpu,&output));
                    if(!wire.live&&wire.code==C3X_RENDERER_RESULT_OK&&wire.recorded_ticket&&wire.recorded_image){
                        ticket_ids[wire.recorded_ticket]=gpu.ticket;image_ids[wire.recorded_image]=gpu.map_image;}
                    if(wire.code==C3X_RENDERER_RESULT_OK&&verify_pixels){
                        require(images!=nullptr&&gpu.width>0&&gpu.height>0,"GPU readback entry or extent missing");
                        auto count=unsigned(gpu.width)*unsigned(gpu.height);std::vector<unsigned> pixels(count);
                        c3x_renderer_gpu_images_v1 read={};read.struct_size=sizeof(read);read.action=C3X_GPU_READBACK;
                        read.ticket=gpu.ticket;read.image=gpu.map_image;read.pixel_count=count;
                        c3x_renderer_gpu_result_v1 result={sizeof(result)};
                        require(images(&read,&result,pixels.data(),count)==C3X_RENDERER_RESULT_OK&&result.pixel_count==count,"GPU map diagnostic readback failed");
                        auto hash=c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(pixels.data()),std::size_t(count)*4);
                        std::copy(hash.begin(),hash.end(),wire.gpu_hash);wire.gpu_hash_valid=1;
                    }
                    if(wire.code==C3X_RENDERER_RESULT_OK&&wire.consumer_pid){
                        auto export_map=wire.shared_raw?shared_raw:shared;
                        require(export_map!=nullptr,"helper lacks requested shared map export");
                        unsigned w=0,h=0;int code=export_map(gpu.ticket,gpu.map_image,wire.consumer_pid,&wire.shared_handle,&w,&h);
                        require(code==C3X_RENDERER_RESULT_OK&&wire.shared_handle&&w==unsigned(gpu.width)&&h==unsigned(gpu.height),"shared map export failed");
                    }}
                if(wire.live&&wire.code==C3X_RENDERER_RESULT_OK){
                    wire.recorded_ticket=gpu.ticket;wire.recorded_image=gpu.map_image;
                }
                wire.width=unsigned(std::max(0,output.width));wire.height=unsigned(std::max(0,output.height));
                wire.rendered=output.rendered_tile_count;wire.fallback=output.fallback_tile_count;
                if(wire.code==C3X_RENDERER_RESULT_OK&&output.bgra_pixels&&output.width>0&&output.height>0&&
                    output.stride_bytes==output.width*4&&output.width<=2240&&output.height<=1260){
                    auto hash=c3x_renderer::asset_content_hash(static_cast<unsigned char const*>(output.bgra_pixels),
                        std::size_t(output.stride_bytes)*unsigned(output.height));
                    std::copy(hash.begin(),hash.end(),wire.hash);
                }
                if(wire.code==C3X_RENDERER_RESULT_OK){
                    Writer response;c3x_remote_scene::encode(response,gpu,output);
                    require(response.bytes.size()<=wire_capacity,"remote scene response exceeds slot");
                    wire.reply_size=unsigned(response.bytes.size());
                    std::memcpy(wire.payload,response.bytes.data(),wire.reply_size);
                }
            }else if(wire.kind==unsigned(Kind::image_commands)&&wire.subtype==0){
                require(images!=nullptr,"helper lacks GPU image entry");
                Images owned;c3x_inputs::images(in,owned);in.done();
                auto map_id=[&](auto const& table,std::int64_t old){
                    if(!old)return std::int64_t(0);
                    auto found=table.find(old);if(found!=table.end())return found->second;
                    require(wire.expected_code!=C3X_RENDERER_RESULT_OK,"missing successful GPU image identity");
                    return std::int64_t(INT64_MAX);
                };
                auto old_image=owned.value.image;
                if(!wire.live){
                    owned.value.ticket=map_id(ticket_ids,owned.value.ticket);
                    owned.value.image=map_id(image_ids,owned.value.image);
                    for(auto& command:owned.commands){
                        command.destination=map_id(image_ids,command.destination);
                        command.source=map_id(image_ids,command.source);
                        command.background=map_id(image_ids,command.background);
                        command.detail=map_id(image_ids,command.detail);
                        command.background_detail=map_id(image_ids,command.background_detail);
                        command.program=map_id(image_ids,command.program);
                    }
                }
                std::vector<unsigned> readback;
                if(owned.value.action==C3X_GPU_READBACK)readback.resize(owned.value.pixel_count);
                c3x_renderer_gpu_result_v1 result={sizeof(result)};
                wire.code=unsigned(images(&owned.value,&result,readback.empty()?nullptr:readback.data(),
                    unsigned(readback.size())));
                wire.result_image=result.image;wire.result_pixels=result.pixel_count;
                wire.resident_bytes=result.resident_bytes;wire.uploads=result.uploads;
                wire.commands=result.commands;wire.readbacks=result.readbacks;
                if(wire.code==C3X_RENDERER_RESULT_OK){
                    if(!wire.live&&owned.value.action==C3X_GPU_CREATE&&wire.recorded_image)
                        image_ids[wire.recorded_image]=result.image;
                    else if(!wire.live&&owned.value.action==C3X_GPU_DESTROY)image_ids.erase(old_image);
                    if(owned.value.action==C3X_GPU_READBACK&&result.pixel_count<=readback.size()){
                        auto hash=c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(readback.data()),
                            std::size_t(result.pixel_count)*4);
                        std::copy(hash.begin(),hash.end(),wire.gpu_hash);wire.gpu_hash_valid=1;
                        if(wire.live){
                            require(std::size_t(result.pixel_count)*4<=wire_capacity,"remote readback slot limit");
                            wire.reply_size=result.pixel_count*4;
                            std::memcpy(wire.payload,readback.data(),wire.reply_size);
                        }
                    }
                }
            }else if(wire.live&&wire.kind==unsigned(Kind::presentation)&&wire.subtype==1){
                stop_direct_cadence();
                require(bind_surface!=nullptr,"helper lacks direct-surface entry");
                auto handle=in.u64();auto width=in.u32(),height=in.u32();in.done();
                wire.code=unsigned(bind_surface(handle,width,height));
                direct_surface_bound=handle&&wire.code==C3X_RENDERER_RESULT_OK;
            }else if(wire.live&&wire.kind==unsigned(Kind::presentation)&&wire.subtype==2){
                in.done();require(surface_pixels&&direct_surface_bound,"direct surface readback unavailable");
                wire.code=unsigned(surface_pixels(reinterpret_cast<unsigned*>(wire.payload),
                    wire_capacity/4,&wire.width,&wire.height));
                if(wire.code==C3X_RENDERER_RESULT_OK){
                    require(std::uint64_t(wire.width)*wire.height*4<=wire_capacity,"direct surface readback exceeded slot");
                    wire.reply_size=wire.width*wire.height*4;
                }
            }else if(wire.kind==unsigned(Kind::presentation)&&wire.subtype==0){
                require(present_shared!=nullptr,"helper lacks final-image shared export");
                c3x_renderer_gpu_present_v1 value={sizeof(value)};
                in(value.action);in(value.ticket);in(value.image);in(value.width);in(value.height);
                for(auto& x:value.area)in(x);auto owner=in.u32();in.done();
                require(owner<=1,"invalid presentation role");
                if(value.action)stop_direct_cadence();
                if(!wire.live&&wire.expected_code!=C3X_RENDERER_RESULT_OK){
                    // Native admission rejected this offer before a renderer
                    // frame existed; do not mutate the x64 retained display.
                    wire.code=wire.expected_code;wire.executed=0;
                }else{
                    if(!wire.live&&value.ticket){auto found=ticket_ids.find(value.ticket);
                        value.ticket=found!=ticket_ids.end()?found->second:INT64_MAX;}
                    if(!wire.live&&value.image){auto found=image_ids.find(value.image);
                        value.image=found!=image_ids.end()?found->second:INT64_MAX;}
                    wire.code=unsigned(present_shared(&value,wire.consumer_pid,
                        &wire.shared_handle,&wire.width,&wire.height));
                    if(value.action)direct_surface_bound=false;
                    else if(wire.code==C3X_RENDERER_RESULT_OK&&direct_surface_bound){
                        direct_display_ready=true;start_direct_cadence();
                    }
                }
            }else if(wire.kind==unsigned(Kind::visual)&&wire.subtype==1){
                require(visual_shared!=nullptr,"helper lacks visual shared export");
                auto automatic=in.u32();in.done();require(automatic<=1,"invalid visual offer");
                if(!wire.clock_frequency&&!wire.live){wire.code=wire.expected_code;wire.executed=0;}
                else{require(wire.consumer_pid||direct_surface_bound,"sampled visual frame has no consumer");
                    auto rendered=unsigned(visual_shared(wire.clock_ticks,wire.clock_frequency,wire.consumer_pid,
                        &wire.shared_handle,&wire.width,&wire.height));
                    wire.rendered=rendered;
                    // A submitted frame can be pending at the x86 window
                    // despite its x64 pixels already being ready.
                    wire.code=!wire.live&&wire.expected_code==C3X_RENDERER_RESULT_PENDING&&rendered==C3X_RENDERER_RESULT_OK?
                        C3X_RENDERER_RESULT_PENDING:rendered;}
            }else if(wire.live&&wire.kind==unsigned(Kind::unit)&&wire.subtype==2){
                require(unit_cpu!=nullptr,"helper lacks CPU unit pixel entry");
                c3x_renderer_unit_v1 value={};c3x_inputs::unit(in,value);
                unsigned flags=in.u32(),with_bounds=in.u32();in.done();
                require(with_bounds<=1,"invalid CPU unit bounds request");
                std::vector<std::uint32_t> pixels(1024u*1024u);unsigned width=0,height=0;int x=0,y=0;
                wire.code=unsigned(unit_cpu(&value,flags,int(with_bounds),wire.bounds,&x,&y,&width,&height,
                    pixels.data(),unsigned(pixels.size())));
                if(wire.code==C3X_RENDERER_RESULT_OK){
                    require(width<=1024&&height<=1024,"CPU unit output extent exceeded");
                    Writer reply;reply(x);reply(y);reply(width);reply(height);
                    for(unsigned n=0;n<4;++n)reply(wire.bounds[n]);
                    for(unsigned n=0;n<width*height;++n)reply(pixels[n]);
                    require(reply.bytes.size()<=wire_capacity,"CPU unit output exceeded IPC slot");
                    wire.reply_size=unsigned(reply.bytes.size());
                    std::memcpy(wire.payload,reply.bytes.data(),wire.reply_size);
                }
            }else if(wire.kind==unsigned(Kind::unit)&&wire.subtype==1){
                require(unit_gpu!=nullptr,"helper lacks GPU unit entry");
                c3x_renderer_unit_v1 value={};c3x_inputs::unit(in,value);
                c3x_renderer_gpu_unit_v1 target={};target.struct_size=sizeof(target);
                c3x_inputs::target_fields(in,target);in.done();
                auto map_id=[&](auto const& table,std::int64_t old){
                    if(!old)return std::int64_t(0);
                    auto found=table.find(old);if(found!=table.end())return found->second;
                    require(wire.expected_code!=C3X_RENDERER_RESULT_OK,"missing successful GPU unit identity");
                    return std::int64_t(INT64_MAX);
                };
                if(!wire.live){
                    target.ticket=map_id(ticket_ids,target.ticket);
                    target.destination=map_id(image_ids,target.destination);
                    target.background=map_id(image_ids,target.background);
                    target.detail=map_id(image_ids,target.detail);
                    target.background_detail=map_id(image_ids,target.background_detail);
                }
                wire.code=unsigned(unit_gpu(&value,&target,wire.bounds));
            }else if(wire.live&&wire.kind==unsigned(Kind::unit_forget)&&wire.subtype==0){
                require(unit_forget!=nullptr,"helper lacks unit retirement entry");
                int id=0;in(id);in.done();unit_forget(id);wire.code=1;
            }else if(wire.live&&wire.kind==unsigned(Kind::tactical)&&wire.subtype==0){
                require(tactical_gpu!=nullptr,"helper lacks tactical renderer entry");
                c3x_renderer_gpu_unit_v1 target={sizeof(target)};c3x_inputs::target_fields(in,target);
                c3x_renderer::tactical::Input capture;c3x_inputs::tactical(in,capture);in.done();
                wire.code=unsigned(tactical_gpu(&capture,&target));
            }else throw std::runtime_error("unsupported scene wire operation");
            wire.service_us=std::uint64_t((milliseconds()-started)*1000.0);
            wire.private_bytes=private_bytes();
        }catch(std::exception const& error){wire.status=1;strncpy_s(wire.error,error.what(),_TRUNCATE);}
    }
};
struct Importer {
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11Device1> device1;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> target;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> buffer;
    Microsoft::WRL::ComPtr<ID3D11RenderTargetView> target_view;
    std::unique_ptr<c3x_gpu_images::Session> composition;
    std::int64_t serial=0;
    unsigned width=0,height=0;
    Importer(){D3D_FEATURE_LEVEL feature={};HRESULT hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context);
        require(SUCCEEDED(hr)&&SUCCEEDED(device.As(&device1)),"x86 shared import D3D device failed");}
    double import(Wire& wire,bool verify_pixels){
        if(!wire.shared_handle)return 0;
        double started=milliseconds();HANDLE handle=reinterpret_cast<HANDLE>(std::uintptr_t(wire.shared_handle));
        Microsoft::WRL::ComPtr<ID3D11Texture2D> source;HRESULT hr=device1->OpenSharedResource1(handle,IID_PPV_ARGS(&source));CloseHandle(handle);
        require(SUCCEEDED(hr),"x86 shared map import failed");
        D3D11_TEXTURE2D_DESC desc={};source->GetDesc(&desc);
        auto expected_format=wire.shared_raw?DXGI_FORMAT_R32_UINT:DXGI_FORMAT_B8G8R8A8_UNORM;
        if(desc.Width!=wire.width||desc.Height!=wire.height||desc.Format!=expected_format){
            char detail[160];std::snprintf(detail,sizeof(detail),"x86 imported map description differs actual=%ux%u format=%u expected=%ux%u",
                desc.Width,desc.Height,unsigned(desc.Format),wire.width,wire.height);throw std::runtime_error(detail);}
        D3D11_TEXTURE2D_DESC display_desc=desc;
        display_desc.MiscFlags=0;display_desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;
        display_desc.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
        if(!target||width!=desc.Width||height!=desc.Height){
            hr=device->CreateTexture2D(&display_desc,nullptr,&target);require(SUCCEEDED(hr),"x86 map destination allocation failed");
            if(wire.shared_raw){
                hr=device->CreateTexture2D(&display_desc,nullptr,&buffer);require(SUCCEEDED(hr),"x86 map history allocation failed");
                hr=device->CreateRenderTargetView(target.Get(),nullptr,&target_view);require(SUCCEEDED(hr),"x86 map render target failed");
            }
            width=desc.Width;height=desc.Height;
        }
        Microsoft::WRL::ComPtr<IDXGIKeyedMutex> keyed;require(SUCCEEDED(source.As(&keyed)),"x86 shared map lacks keyed mutex");
        hr=keyed->AcquireSync(1,1000);require(hr==S_OK,"x86 shared map acquire failed");
        if(wire.shared_raw){
            if(!composition)composition=std::make_unique<c3x_gpu_images::Session>(device.Get(),context.Get());
            require(composition->publish_shared(source.Get(),++serial),"x86 shared map admission failed");
            require(composition->display_to(composition->current_ticket(),composition->map_image(),
                target_view.Get(),target.Get(),buffer.Get(),width,height,{0,0,int(width),int(height)}),
                "x86 shared map composition failed");
        }else context->CopyResource(target.Get(),source.Get());
        hr=keyed->ReleaseSync(0);context->Flush();
        require(SUCCEEDED(hr),"x86 shared map release failed");
        if(verify_pixels){
            D3D11_TEXTURE2D_DESC stage_desc=display_desc;stage_desc.MiscFlags=0;stage_desc.BindFlags=0;
            stage_desc.Usage=D3D11_USAGE_STAGING;stage_desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            Microsoft::WRL::ComPtr<ID3D11Texture2D> stage;
            hr=device->CreateTexture2D(&stage_desc,nullptr,&stage);require(SUCCEEDED(hr),"x86 diagnostic stage allocation failed");
            context->CopyResource(stage.Get(),target.Get());D3D11_MAPPED_SUBRESOURCE mapped={};
            hr=context->Map(stage.Get(),0,D3D11_MAP_READ,0,&mapped);require(SUCCEEDED(hr),"x86 diagnostic map readback failed");
            std::vector<unsigned> pixels(std::size_t(desc.Width)*desc.Height);
            for(unsigned y=0;y<desc.Height;++y)std::memcpy(pixels.data()+std::size_t(y)*desc.Width,
                static_cast<unsigned char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,std::size_t(desc.Width)*4);
            context->Unmap(stage.Get(),0);
            auto hash=c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(pixels.data()),pixels.size()*4);
            std::copy(hash.begin(),hash.end(),wire.gpu_hash);wire.gpu_hash_valid=1;
        }
        return milliseconds()-started;
    }
};
struct ChildGuard {
    HANDLE process=nullptr;
    ~ChildGuard(){if(process){if(WaitForSingleObject(process,0)==WAIT_TIMEOUT){TerminateProcess(process,1);WaitForSingleObject(process,5000);}CloseHandle(process);}}
};
void apply_settings(Reader& in){
    require(in.u32()==protocol_version&&in.u32()==C3X_RENDERER_API_VERSION,"recording protocol mismatch");
    in.u64();ClockOrigin origin;clock_origin(in,origin);
    auto count=in.u32();require(count<=512,"settings limit");
    for(unsigned n=0;n<count;++n){auto key=in.string(256),value=in.string(32768);
        require(key.rfind("C3X_RENDERER_",0)==0&&key.rfind("C3X_RENDERER_INPUT_",0)!=0,"invalid recorded setting");
        if(key.find("TRACE")==std::string::npos&&key.find("RECORD")==std::string::npos)SetEnvironmentVariableA(key.c_str(),value.c_str());}
    in.done();SetEnvironmentVariableA("C3X_RENDERER_MANUAL_VISUAL","1");
}
void report(std::ofstream& file,std::uint64_t sequence,Wire const& wire,double roundtrip_ms,double import_ms,std::uint64_t driver_private_bytes){
    file<<"{\"sequence\":"<<sequence<<",\"family\":"<<wire.kind<<",\"subtype\":"<<wire.subtype
        <<",\"bytes\":"<<wire.size<<",\"result\":"<<wire.code<<",\"width\":"<<wire.width
        <<",\"height\":"<<wire.height<<",\"rendered\":"<<wire.rendered<<",\"fallback\":"<<wire.fallback
        <<",\"hash\":\""<<std::hex<<std::setfill('0');for(auto part:wire.hash)file<<std::setw(8)<<part;
    file<<std::dec<<"\",\"service_ms\":"<<double(wire.service_us)/1000.0<<",\"roundtrip_ms\":"<<roundtrip_ms
        <<",\"private_bytes\":"<<wire.private_bytes<<",\"driver_private_bytes\":"<<driver_private_bytes
        <<",\"shared_map\":"<<(wire.shared_handle?"true":"false")<<",\"import_ms\":"<<import_ms
        <<",\"gpu_hash_valid\":"<<(wire.gpu_hash_valid?"true":"false")<<",\"gpu_hash\":\""<<std::hex<<std::setfill('0');
    for(auto part:wire.gpu_hash)file<<std::setw(8)<<part;
    file<<std::dec<<"\"}\n";
    require(bool(file),"scene report write failed");
}
}

int wmain(int argc,wchar_t** argv){
    try{
#if defined(_WIN64)
        require(argc==5&&std::wstring(argv[1])==L"--child","helper usage: --child NAME DLL PARENT_PID");
        auto parent_pid=std::stoull(std::wstring(argv[4]));
        require(parent_pid>0&&parent_pid<=MAXDWORD,"invalid helper parent PID");
        struct ParentGuard {HANDLE handle;~ParentGuard(){if(handle)CloseHandle(handle);}} parent{
            OpenProcess(SYNCHRONIZE,FALSE,DWORD(parent_pid))};
        require(parent.handle!=nullptr,"helper parent unavailable");
        // The bridge and helper may both construct a trace sink. Keep their
        // files distinct so the x86 endpoint cannot truncate x64 evidence.
        wchar_t trace_path[MAX_PATH]={};
        auto trace_size=GetEnvironmentVariableW(L"C3X_RENDERER_TRACE_FILE",trace_path,MAX_PATH);
        if(trace_size && trace_size<MAX_PATH-4){
            std::wstring helper_trace(trace_path,trace_size);helper_trace+=L".x64";
            require(SetEnvironmentVariableW(L"C3X_RENDERER_TRACE_FILE",helper_trace.c_str())!=FALSE,
                "x64 trace path setup failed");
        }
        std::wstring base=argv[2];HANDLE mapping=OpenFileMappingW(FILE_MAP_ALL_ACCESS,FALSE,object_name(base,L"_map").c_str());
        HANDLE request=OpenEventW(SYNCHRONIZE,FALSE,object_name(base,L"_request").c_str());
        HANDLE response=OpenEventW(EVENT_MODIFY_STATE,FALSE,object_name(base,L"_response").c_str());
        require(mapping&&request&&response,"scene IPC objects missing");
        auto* wire=static_cast<Wire*>(MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(Wire)));require(wire!=nullptr,"scene IPC map failed");
        Core core(argv[3]);require(SetEvent(response)!=FALSE,"helper ready signal failed");
        unsigned last=wire->sequence;
        HANDLE active[2]={request,parent.handle};
        while(WaitForMultipleObjects(2,active,FALSE,120000)==WAIT_OBJECT_0){
            require(wire->magic==wire_magic&&wire->version==wire_version&&wire->sequence==last+1,"scene IPC sequence mismatch");
            last=wire->sequence;if(wire->kind==0){wire->status=0;SetEvent(response);break;}
            core.execute(*wire);require(SetEvent(response)!=FALSE,"helper response signal failed");
        }
        UnmapViewOfFile(wire);CloseHandle(response);CloseHandle(request);CloseHandle(mapping);return 0;
#else
        require(argc>=5&&argc<=12,"driver usage: --local DLL CAPTURE REPORT | --remote DLL CAPTURE REPORT HELPER [--verify-pixels] [--raw-shared] [--crash-after N] [--reserve-mib N]");
        bool remote=std::wstring(argv[1])==L"--remote",verify_pixels=false,raw_shared=false;unsigned crash_after=0,reserve_mib=0;
        require(remote||std::wstring(argv[1])==L"--local","invalid scene driver mode");
        for(int n=remote?6:5;n<argc;++n){std::wstring flag=argv[n];
            if(flag==L"--verify-pixels")verify_pixels=true;
            else if(flag==L"--raw-shared"&&remote)raw_shared=true;
            else if(flag==L"--crash-after"&&remote&&n+1<argc){crash_after=unsigned(std::stoul(argv[++n]));require(crash_after>1&&crash_after<54,"invalid helper crash point");}
            else if(flag==L"--reserve-mib"&&n+1<argc){reserve_mib=unsigned(std::stoul(argv[++n]));require(reserve_mib<=2048,"invalid x86 address-space reservation");}
            else throw std::runtime_error("unknown scene driver option");}
        void* reservation=reserve_mib?VirtualAlloc(nullptr,std::size_t(reserve_mib)*1024*1024,MEM_RESERVE,PAGE_NOACCESS):nullptr;
        require(!reserve_mib||reservation!=nullptr,"x86 address-space reservation failed");
        std::ofstream output(argv[4]);require(bool(output),"cannot create scene report");
        SegmentReader reader(argv[3]);Event event;require(reader.next(event)&&event.kind==Kind::manifest,"scene recording manifest missing");Reader settings{event.payload};apply_settings(settings);
        std::unique_ptr<Core> local;if(!remote)local=std::make_unique<Core>(argv[2],verify_pixels);
        std::unique_ptr<Importer> importer;if(remote)importer=std::make_unique<Importer>();
        HANDLE mapping=nullptr,request=nullptr,response=nullptr;Wire* shared=nullptr;PROCESS_INFORMATION child={};ChildGuard child_guard;
        std::wstring base;std::vector<unsigned char> last_config;
        auto start_child=[&]{
            std::wstring command=L"\""+std::wstring(argv[5])+L"\" --child \""+base+L"\" \""+std::wstring(argv[2])+L"\" "+std::to_wstring(GetCurrentProcessId());
            std::vector<wchar_t> writable(command.begin(),command.end());writable.push_back(0);STARTUPINFOW startup={};startup.cb=sizeof(startup);
            require(CreateProcessW(argv[5],writable.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW,nullptr,nullptr,&startup,&child)!=FALSE,"x64 helper start failed");
            child_guard.process=child.hProcess;CloseHandle(child.hThread);
            HANDLE ready[2]={response,child_guard.process};require(WaitForMultipleObjects(2,ready,FALSE,120000)==WAIT_OBJECT_0,"x64 helper startup failed");
        };
        auto receive=[&]{HANDLE ready[2]={response,child_guard.process};
            require(SetEvent(request)!=FALSE&&WaitForMultipleObjects(2,ready,FALSE,120000)==WAIT_OBJECT_0,"x64 scene response failed");};
        if(remote){
            base=std::wstring(L"Local\\C3XGate2_")+std::to_wstring(GetCurrentProcessId())+L"_"+std::to_wstring(GetTickCount64());
            mapping=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,sizeof(Wire),object_name(base,L"_map").c_str());
            request=CreateEventW(nullptr,FALSE,FALSE,object_name(base,L"_request").c_str());
            response=CreateEventW(nullptr,FALSE,FALSE,object_name(base,L"_response").c_str());
            require(mapping&&request&&response,"scene IPC creation failed");
            shared=static_cast<Wire*>(MapViewOfFile(mapping,FILE_MAP_ALL_ACCESS,0,0,sizeof(Wire)));require(shared!=nullptr,"scene IPC view failed");
            start_child();
        }
        auto local_wire=std::make_unique<Wire>();unsigned sent=0,scene_count=0;std::map<std::uint64_t,Event> pending;bool truncated=false;
        while(!reader.footer&&reader.next_verified(event,false,truncated)){
            if(event.kind==Kind::scene||(event.kind==Kind::native_bridge&&event.flags==8)){
                Reader in{event.payload};auto token=in.u64();in.u64();require(token&&pending.emplace(token,std::move(event)).second,"duplicate scene token");continue;}
            if(event.kind!=Kind::result)continue;
            Reader result{event.payload};auto token=result.u64();auto found=pending.find(token);if(found==pending.end())continue;
            auto& input=found->second;Reader in{input.payload};in.u64();in.u64();
            Wire* wire=remote?shared:local_wire.get();wire->magic=wire_magic;wire->version=wire_version;wire->sequence=++sent;
            wire->kind=unsigned(input.kind);wire->subtype=input.flags;wire->size=unsigned(input.payload.size()-in.at);
            wire->live=0;wire->reply_size=0;
            wire->shared_raw=remote&&raw_shared?1u:0u;
            wire->consumer_pid=remote?GetCurrentProcessId():0;
            require(wire->size<=wire_capacity,"scene value payload exceeds IPC capacity");
            std::copy(input.payload.begin()+in.at,input.payload.end(),wire->payload);
            if(input.kind==Kind::native_bridge)last_config.assign(input.payload.begin()+in.at,input.payload.end());
            double started=milliseconds();if(remote)receive();
            else local->execute(*wire);
            double roundtrip=milliseconds()-started;require(wire->status==0,wire->error);
            double import_ms=remote?importer->import(*wire,verify_pixels):0;
            if(wire->kind==unsigned(Kind::scene))++scene_count;
            report(output,input.sequence,*wire,roundtrip,import_ms,private_bytes());pending.erase(found);
            if(remote&&crash_after&&sent==crash_after){
                require(!last_config.empty(),"cannot restart helper without copied definition inputs");
                TerminateProcess(child_guard.process,9);require(WaitForSingleObject(child_guard.process,5000)==WAIT_OBJECT_0,"helper crash fixture did not exit");
                CloseHandle(child_guard.process);child_guard.process=nullptr;start_child();
                shared->magic=wire_magic;shared->version=wire_version;shared->sequence=++sent;
                shared->kind=unsigned(Kind::native_bridge);shared->subtype=8;shared->size=unsigned(last_config.size());
                shared->consumer_pid=GetCurrentProcessId();std::copy(last_config.begin(),last_config.end(),shared->payload);
                receive();require(shared->status==0&&shared->code==C3X_RENDERER_RESULT_OK,"helper definition restore failed");
            }
        }
        require(reader.footer&&pending.empty()&&scene_count>0,"scene workload incomplete");
        if(remote){shared->magic=wire_magic;shared->version=wire_version;shared->sequence=++sent;shared->kind=0;
            receive();
            require(WaitForSingleObject(child.hProcess,120000)==WAIT_OBJECT_0,"x64 helper exit timed out");
            DWORD code=1;GetExitCodeProcess(child.hProcess,&code);require(code==0,"x64 helper exited with failure");
            UnmapViewOfFile(shared);CloseHandle(response);CloseHandle(request);CloseHandle(mapping);}
        if(reservation)VirtualFree(reservation,0,MEM_RELEASE);
        std::printf("PASS scene workload: %u copied scene records, %u ordered wire operations\n",scene_count,sent-unsigned(remote));return 0;
#endif
    }catch(std::exception const& error){std::fprintf(stderr,"FAIL scene workload: %s\n",error.what());return 1;}
}
