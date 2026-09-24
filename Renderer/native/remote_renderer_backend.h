#pragma once
#include "remote_renderer_client.h"
#include "gpu_native_presenter.h"
#include "remote_direct_surface.h"
#include "native_screen_bridge.h"
#include "visual_cadence.h"
#include "input_recording/display.h"
#include <cstdio>
#include <map>
#include <mutex>

namespace c3x_remote_scene {
// Single x86 endpoint for the native composition owner. Scene/image work stays
// in x64; only the final shared BGRA texture crosses to the Civ III window.
class Backend {
    Client client;
    std::mutex gate;
    c3x_gpu_images::ComPtr<ID3D11Device> device;
    c3x_gpu_images::ComPtr<ID3D11Device1> device1;
    c3x_gpu_images::ComPtr<ID3D11DeviceContext> context;
    c3x_gpu_images::NativePresenter presenter;
    DirectSurface direct_surface;
    bool direct_active=false;
    bool direct_unavailable=false;
    bool direct_requested=false;
    c3x_renderer::VisualCadence cadence;
    HWND active_window=nullptr;
    bool visual_active=false;
    // Only changed authoritative unit facts need an IPC roundtrip. Native
    // redraw ticks can repeat the same action/HP for many visual frames.
    std::map<int,c3x_renderer_unit_state_v1> unit_facts;
    bool detach_direct(bool paint_native,std::vector<unsigned>* retained=nullptr,
                       unsigned* retained_width=nullptr,unsigned* retained_height=nullptr){
        if(!direct_active)return true;
        std::vector<unsigned> pixels;unsigned w=0,h=0;
        bool have=(!paint_native&&!retained)||client.surface_pixels(pixels,w,h);
        HWND hwnd=active_window;
        client.bind_surface(nullptr,0,0);
        direct_surface.reset();direct_active=false;
        if(retained&&have){*retained=std::move(pixels);
            if(retained_width)*retained_width=w;if(retained_height)*retained_height=h;}
        else if(paint_native&&have&&hwnd&&IsWindow(hwnd)){
            HDC dc=GetDC(hwnd);
            if(dc){BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
                info.bmiHeader.biWidth=LONG(w);info.bmiHeader.biHeight=-LONG(h);
                info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
                SetDIBitsToDevice(dc,0,0,w,h,0,0,0,h,pixels.data(),&info,DIB_RGB_COLORS);
                GdiFlush();ReleaseDC(hwnd,dc);}
        }
        return have;
    }
    bool graphics(){
        if(device)return true;
        D3D_FEATURE_LEVEL feature={};
        auto result=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context);
        return SUCCEEDED(result)&&SUCCEEDED(device.As(&device1));
    }
public:
    Backend(std::wstring const& helper,std::wstring const& dll,bool direct=false):client(helper,dll),direct_requested(direct){}
    ~Backend(){cadence.stop();}
    bool healthy()const{return client.alive();}
    void abandon(){
        cadence.stop();std::lock_guard<std::mutex> lock(gate);
        direct_surface.reset();direct_active=false;visual_active=false;active_window=nullptr;
        presenter.reset();unit_facts.clear();
    }
    int definitions(char const* root,char const* fallback,char const* scenario,char const* custom){
        std::lock_guard<std::mutex> lock(gate);
        if(!detach_direct(true))return C3X_RENDERER_RESULT_DEVICE_ERROR;
        visual_active=false;active_window=nullptr;cadence.disable();
        unit_facts.clear();
        direct_unavailable=false;
        return client.definitions(root,fallback,scenario,custom);
    }
    int pack(char const* path){std::lock_guard<std::mutex> lock(gate);
        if(!detach_direct(true))return C3X_RENDERER_RESULT_DEVICE_ERROR;
        visual_active=false;active_window=nullptr;cadence.disable();
        unit_facts.clear();
        direct_unavailable=false;
        return client.pack(path);}
    int set_units(int enabled){std::lock_guard<std::mutex> lock(gate);if(!enabled)unit_facts.clear();return client.set_units(enabled);}
    int visual_policy(unsigned policy){std::lock_guard<std::mutex> lock(gate);return client.visual_policy(policy);}
    int render(c3x_renderer_camera_request_v1 const& request,c3x_renderer_gpu_frame_v1& gpu,
               c3x_renderer_output_v1& output){
        std::lock_guard<std::mutex> lock(gate);return client.render(request,gpu,output);
    }
    int render_cpu(c3x_renderer_frame_v1 const& frame,c3x_renderer_camera_identity_v1 const* identity,
                   c3x_renderer_output_v1& output){
        std::lock_guard<std::mutex> lock(gate);return client.render_cpu(frame,identity,output);
    }
    int camera_begin(c3x_renderer_camera_request_v1 const& request,c3x_renderer_i64& ticket){
        std::lock_guard<std::mutex> lock(gate);return client.camera_begin(request,ticket);
    }
    int camera_poll(c3x_renderer_i64 ticket,c3x_renderer_gpu_camera_view_v1& view){
        std::lock_guard<std::mutex> lock(gate);return client.camera_poll(ticket,view);
    }
    int camera_cancel(c3x_renderer_i64 ticket){
        std::lock_guard<std::mutex> lock(gate);return client.camera_cancel(ticket);
    }
    int world_query(c3x_renderer_world_page_v1& page){
        std::lock_guard<std::mutex> lock(gate);return client.world_query(page);
    }
    int world_submit(c3x_renderer_world_page_v1 const& page,int callback_result){
        std::lock_guard<std::mutex> lock(gate);return client.world_submit(page,callback_result);
    }
    int world_delta_scope(c3x_renderer_world_page_v1& page){
        std::lock_guard<std::mutex> lock(gate);return client.world_delta_scope(page);
    }
    int world_delta_submit(c3x_renderer_world_page_v1 const& page,int callback_result){
        std::lock_guard<std::mutex> lock(gate);return client.world_delta_submit(page,callback_result);
    }
    int world_status(c3x_renderer_world_status_v1& status){
        std::lock_guard<std::mutex> lock(gate);return client.world_status(status);
    }
    int images(c3x_renderer_gpu_images_v1 const& request,c3x_renderer_gpu_result_v1& result,
               unsigned* pixels,unsigned capacity){
        std::lock_guard<std::mutex> lock(gate);return client.images(request,result,pixels,capacity);
    }
    int unit(c3x_renderer_unit_v1 const& unit,c3x_renderer_gpu_unit_v1 const& target,int* bounds){
        std::lock_guard<std::mutex> lock(gate);return client.unit(unit,target,bounds);
    }
    int unit_cpu(c3x_renderer_unit_v1 const& unit,unsigned flags,int* bounds,
                 std::vector<std::uint32_t>& pixels,int& x,int& y,unsigned& width,unsigned& height){
        std::lock_guard<std::mutex> lock(gate);return client.unit_cpu(unit,flags,bounds,pixels,x,y,width,height);
    }
    void forget_unit(int id){std::lock_guard<std::mutex> lock(gate);unit_facts.erase(id);client.forget_unit(id);}
    int unit_visual(c3x_renderer_unit_visual_v1 const& value){
        std::lock_guard<std::mutex> lock(gate);return client.unit_visual(value);
    }
    int unit_move(c3x_renderer_unit_move_v1 const& value){
        std::lock_guard<std::mutex> lock(gate);unit_facts.erase(value.unit_id);return client.unit_move(value);
    }
    int unit_spawn(c3x_renderer_unit_spawn_v1 const& value){
        std::lock_guard<std::mutex> lock(gate);unit_facts.erase(value.unit_id);return client.unit_spawn(value);
    }
    int unit_state(c3x_renderer_unit_state_v1 const& value){
        std::lock_guard<std::mutex> lock(gate);
        auto found=unit_facts.find(value.unit_id);
        if(found!=unit_facts.end()){
            auto const& old=found->second;
            if(old.kind==value.kind&&old.tile_x==value.tile_x&&old.tile_y==value.tile_y&&
               old.unit_type_id==value.unit_type_id&&old.owner_id==value.owner_id&&
               old.action==value.action&&old.damage==value.damage&&old.max_hp==value.max_hp&&
               old.visible==value.visible&&old.map_epoch==value.map_epoch&&
               old.viewer_epoch==value.viewer_epoch&&old.presentation_frequency==value.presentation_frequency)
                return C3X_RENDERER_RESULT_OK;
        }
        c3x_inputs::Call input(c3x_inputs::Kind::unit_state,0,[&](auto& out){
            auto copied=value;c3x_inputs::unit_state_fields(out,copied);
        });
        int code=C3X_RENDERER_RESULT_ERROR;
        try{code=input.result(client.unit_state(value));}
        catch(...){input.result(C3X_RENDERER_RESULT_ERROR);throw;}
        if(code==C3X_RENDERER_RESULT_OK){
            if(unit_facts.size()>=8192&&found==unit_facts.end())unit_facts.clear();
            unit_facts[value.unit_id]=value;
        }
        return code;
    }
    int tactical(c3x_renderer::tactical::Input const& capture,c3x_renderer_gpu_unit_v1 const& target){
        std::lock_guard<std::mutex> lock(gate);return client.tactical(capture,target);
    }
    int present(c3x_renderer_gpu_present_v1 const& request){
        std::lock_guard<std::mutex> lock(gate);
        char phase_option[4]={};bool phase_probe=GetEnvironmentVariableA("C3X_RENDERER_PRESENT_PHASES",phase_option,sizeof(phase_option))==1&&phase_option[0]=='1';
        LARGE_INTEGER phase_begin={},phase_remote={},phase_adopt={},phase_end={},phase_rate={};
        if(phase_probe){QueryPerformanceCounter(&phase_begin);QueryPerformanceFrequency(&phase_rate);}
        if(!presenter.caller_thread()||!direct_surface.caller_thread())return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(request.action&&direct_active&&!detach_direct(request.action==2))
            return C3X_RENDERER_RESULT_DEVICE_ERROR;
        if(request.action==0){
            DWORD process=0;
            if(GetWindowThreadProcessId(static_cast<HWND>(request.window),&process)!=GetCurrentThreadId()||
               process!=GetCurrentProcessId())return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            if(!graphics())return C3X_RENDERER_RESULT_DEVICE_ERROR;
            bool full=request.area[0]==0&&request.area[1]==0&&request.area[2]==request.width&&request.area[3]==request.height;
            if(direct_active&&!direct_surface.matches(static_cast<HWND>(request.window),
               unsigned(request.width),unsigned(request.height))){
                if(!detach_direct(true))return C3X_RENDERER_RESULT_DEVICE_ERROR;
            }
            char trial[4]={};bool requested=direct_requested||
                (GetEnvironmentVariableA("C3X_RENDERER_DIRECT_SURFACE_TRIAL",trial,sizeof(trial))==1&&trial[0]=='1');
            char strict_option[4]={};bool strict=GetEnvironmentVariableA("C3X_RENDERER_DIRECT_SURFACE_STRICT_TRIAL",strict_option,sizeof(strict_option))==1&&strict_option[0]=='1';
            if(requested&&full&&!direct_active&&!direct_unavailable){
                direct_active=direct_surface.prepare(static_cast<HWND>(request.window),device.Get(),
                    unsigned(request.width),unsigned(request.height))&&
                    client.bind_surface(direct_surface.handle(),unsigned(request.width),unsigned(request.height))==C3X_RENDERER_RESULT_OK;
                if(!direct_active)direct_unavailable=true;
            }
            if(!direct_active){
                direct_surface.reset();
                if(requested&&full&&strict)return C3X_RENDERER_RESULT_DEVICE_ERROR;
                if(!presenter.prepare(static_cast<HWND>(request.window),device.Get(),
                                      unsigned(request.width),unsigned(request.height),full))
                    return C3X_RENDERER_RESULT_BAD_ARGUMENT;
            }
        }
        SharedFrame frame;int code=client.present(request,frame);
        if(code==C3X_RENDERER_RESULT_OK&&direct_active&&request.action==0&&!direct_surface.activate())
            code=C3X_RENDERER_RESULT_DEVICE_ERROR;
        if(code==C3X_RENDERER_RESULT_DEVICE_ERROR&&direct_active&&request.action==0){
            char strict_option[4]={};if(GetEnvironmentVariableA("C3X_RENDERER_DIRECT_SURFACE_STRICT_TRIAL",strict_option,sizeof(strict_option))==1&&strict_option[0]=='1')
                return code;
            client.bind_surface(nullptr,0,0);direct_surface.reset();direct_active=false;direct_unavailable=true;
            bool full=request.area[0]==0&&request.area[1]==0&&request.area[2]==request.width&&request.area[3]==request.height;
            if(presenter.prepare(static_cast<HWND>(request.window),device.Get(),
                                 unsigned(request.width),unsigned(request.height),full))code=client.present(request,frame);
        }
        auto helper_present_service=phase_probe?client.stats().service_us:0;
        if(phase_probe)QueryPerformanceCounter(&phase_remote);
        if(code!=C3X_RENDERER_RESULT_OK){if(frame.handle)CloseHandle(reinterpret_cast<HANDLE>(std::uintptr_t(frame.handle)));return code;}
        if(request.action==0){
            int displayed=direct_active?C3X_RENDERER_RESULT_OK:
                presenter.adopt_shared(device1.Get(),context.Get(),frame.handle,frame.width,frame.height);
            if(phase_probe)QueryPerformanceCounter(&phase_adopt);
            if(displayed==C3X_RENDERER_RESULT_OK){
                active_window=static_cast<HWND>(request.window);
                visual_active=client.visual_policy(3)!=0;
                char manual[4]={};bool manual_replay=GetEnvironmentVariableA("C3X_RENDERER_MANUAL_VISUAL",manual,sizeof(manual))==1&&manual[0]=='1';
                if(visual_active&&!manual_replay&&!direct_active)cadence.enable([this]{
                    try{LARGE_INTEGER now={},frequency={};QueryPerformanceCounter(&now);QueryPerformanceFrequency(&frequency);
                        visual(now.QuadPart,frequency.QuadPart);}catch(...){OutputDebugStringA("[C3X renderer] x64 visual frame unavailable\n");}
                });
                else cadence.disable();
            }
            if(phase_probe){QueryPerformanceCounter(&phase_end);
                std::fprintf(stderr,"PRESENT_PHASE ticket=%lld remote_ms=%.3f helper_service_ms=%.3f adopt_ms=%.3f policy_ms=%.3f route=%s result=%d\n",
                    static_cast<long long>(request.ticket),1000.*double(phase_remote.QuadPart-phase_begin.QuadPart)/double(phase_rate.QuadPart),
                    double(helper_present_service)/1000.,
                    1000.*double(phase_adopt.QuadPart-phase_remote.QuadPart)/double(phase_rate.QuadPart),
                    1000.*double(phase_end.QuadPart-phase_adopt.QuadPart)/double(phase_rate.QuadPart),direct_active?"direct":"shared",displayed);}
            return displayed;
        }
        if(frame.handle)CloseHandle(reinterpret_cast<HANDLE>(std::uintptr_t(frame.handle)));
        visual_active=false;active_window=nullptr;cadence.disable();
        if(request.action==2){
            if(presenter.initialized&&!presenter.preserve_display(context.Get()))return C3X_RENDERER_RESULT_DEVICE_ERROR;
            presenter.release_native();
        }else presenter.reset();
        return C3X_RENDERER_RESULT_OK;
    }
    int visual(std::int64_t ticks,std::int64_t frequency){
        std::unique_lock<std::mutex> lock(gate,std::try_to_lock);
        char manual[4]={};bool manual_replay=GetEnvironmentVariableA("C3X_RENDERER_MANUAL_VISUAL",manual,sizeof(manual))==1&&manual[0]=='1';
        if(!lock.owns_lock()||!visual_active||!active_window||
           (!manual_replay&&(!IsWindowVisible(active_window)||IsIconic(active_window))))
            return C3X_RENDERER_RESULT_PENDING;
        SharedFrame frame;int code=client.visual(ticks,frequency,frame);
        if(code!=C3X_RENDERER_RESULT_OK){if(frame.handle)CloseHandle(reinterpret_cast<HANDLE>(std::uintptr_t(frame.handle)));return code;}
        if(direct_active)return code;
        if(!frame.handle)return C3X_RENDERER_RESULT_PENDING;
        return presenter.adopt_shared(device1.Get(),context.Get(),frame.handle,frame.width,frame.height,true);
    }
    int screen(c3x_native_images::ScreenSnapshot const* source){
        std::lock_guard<std::mutex> lock(gate);
        visual_active=false;cadence.disable();
        std::vector<unsigned> previous;unsigned previous_width=0,previous_height=0;
        if(direct_active){
            if(source){if(!detach_direct(false,&previous,&previous_width,&previous_height))return C3X_RENDERER_RESULT_DEVICE_ERROR;}
            else if(!detach_direct(true))return C3X_RENDERER_RESULT_DEVICE_ERROR;
        }
        active_window=nullptr;
        if(!source){
            if(presenter.initialized){
                if(!presenter.preserve_display(context.Get()))return C3X_RENDERER_RESULT_DEVICE_ERROR;
                presenter.release_native();
            }
            return C3X_RENDERER_RESULT_OK;
        }
        if(!graphics())return C3X_RENDERER_RESULT_DEVICE_ERROR;
        bool full=source->area.left==0&&source->area.top==0&&
            source->area.right==source->width&&source->area.bottom==source->height;
        if(!full&&!previous.empty()){
            if(previous_width!=unsigned(source->width)||previous_height!=unsigned(source->height)||
               !presenter.prepare(source->window,device.Get(),previous_width,previous_height,true)||
               !presenter.seed_bgra(context.Get(),previous.data(),previous_width,previous_height))
                return C3X_RENDERER_RESULT_DEVICE_ERROR;
        }
        if(!presenter.prepare(source->window,device.Get(),unsigned(source->width),unsigned(source->height),full)||
           !presenter.upload_screen(context.Get(),source->pixels.data(),unsigned(source->width),
               unsigned(source->height),source->area,source->native_format))
            return C3X_RENDERER_RESULT_DEVICE_ERROR;
        return presenter.present();
    }
    int reset(){
        cadence.stop();
        std::lock_guard<std::mutex> lock(gate);
        visual_active=false;
        unit_facts.clear();
        if(!detach_direct(true))return C3X_RENDERER_RESULT_DEVICE_ERROR;
        active_window=nullptr;
        direct_unavailable=false;
        if(presenter.initialized){
            if(!presenter.preserve_display(context.Get()))return C3X_RENDERER_RESULT_DEVICE_ERROR;
            presenter.release_native();
        }else presenter.reset();
        return client.reset();
    }
    bool replay_display(std::vector<unsigned>& pixels,unsigned& width,unsigned& height){
        std::lock_guard<std::mutex> lock(gate);
        if(direct_active)return client.surface_pixels(pixels,width,height);
        return c3x_inputs::display_pixels(device.Get(),context.Get(),presenter.retained(),pixels,width,height);
    }
    c3x_helper_trial::SceneClient::Stats replay_stats(){
        std::lock_guard<std::mutex> lock(gate);return client.stats();
    }
};
}
