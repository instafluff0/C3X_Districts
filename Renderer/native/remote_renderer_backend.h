#pragma once
#include "remote_renderer_client.h"
#include "gpu_native_presenter.h"
#include "native_screen_bridge.h"
#include "visual_cadence.h"
#include "input_recording/display.h"
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
    c3x_renderer::VisualCadence cadence;
    HWND active_window=nullptr;
    bool visual_active=false;
    bool graphics(){
        if(device)return true;
        D3D_FEATURE_LEVEL feature={};
        auto result=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context);
        return SUCCEEDED(result)&&SUCCEEDED(device.As(&device1));
    }
public:
    Backend(std::wstring const& helper,std::wstring const& dll):client(helper,dll){}
    ~Backend(){cadence.stop();}
    int definitions(char const* root,char const* fallback,char const* scenario,char const* custom){
        std::lock_guard<std::mutex> lock(gate);return client.definitions(root,fallback,scenario,custom);
    }
    int pack(char const* path){std::lock_guard<std::mutex> lock(gate);return client.pack(path);}
    int set_units(int enabled){std::lock_guard<std::mutex> lock(gate);return client.set_units(enabled);}
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
    void forget_unit(int id){std::lock_guard<std::mutex> lock(gate);client.forget_unit(id);}
    int tactical(c3x_renderer::tactical::Input const& capture,c3x_renderer_gpu_unit_v1 const& target){
        std::lock_guard<std::mutex> lock(gate);return client.tactical(capture,target);
    }
    int present(c3x_renderer_gpu_present_v1 const& request){
        std::lock_guard<std::mutex> lock(gate);
        if(!presenter.caller_thread())return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        if(request.action==0){
            if(!graphics())return C3X_RENDERER_RESULT_DEVICE_ERROR;
            bool full=request.area[0]==0&&request.area[1]==0&&request.area[2]==request.width&&request.area[3]==request.height;
            if(!presenter.prepare(static_cast<HWND>(request.window),device.Get(),
                                  unsigned(request.width),unsigned(request.height),full))
                return C3X_RENDERER_RESULT_BAD_ARGUMENT;
        }
        SharedFrame frame;int code=client.present(request,frame);
        if(code!=C3X_RENDERER_RESULT_OK){if(frame.handle)CloseHandle(reinterpret_cast<HANDLE>(std::uintptr_t(frame.handle)));return code;}
        if(request.action==0){
            int displayed=presenter.adopt_shared(device1.Get(),context.Get(),frame.handle,frame.width,frame.height);
            if(displayed==C3X_RENDERER_RESULT_OK){
                active_window=static_cast<HWND>(request.window);
                visual_active=client.visual_policy(2)!=0;
                char manual[4]={};bool manual_replay=GetEnvironmentVariableA("C3X_RENDERER_MANUAL_VISUAL",manual,sizeof(manual))==1&&manual[0]=='1';
                if(visual_active&&!manual_replay)cadence.enable([this]{
                    try{LARGE_INTEGER now={},frequency={};QueryPerformanceCounter(&now);QueryPerformanceFrequency(&frequency);
                        visual(now.QuadPart,frequency.QuadPart);}catch(...){OutputDebugStringA("[C3X renderer] x64 visual frame unavailable\n");}
                });
            }
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
        if(!frame.handle)return C3X_RENDERER_RESULT_PENDING;
        return presenter.adopt_shared(device1.Get(),context.Get(),frame.handle,frame.width,frame.height,true);
    }
    int screen(c3x_native_images::ScreenSnapshot const* source){
        std::lock_guard<std::mutex> lock(gate);
        visual_active=false;active_window=nullptr;cadence.disable();
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
        if(!presenter.prepare(source->window,device.Get(),unsigned(source->width),unsigned(source->height),full)||
           !presenter.upload_screen(context.Get(),source->pixels.data(),unsigned(source->width),
               unsigned(source->height),source->area,source->native_format))
            return C3X_RENDERER_RESULT_DEVICE_ERROR;
        return presenter.present();
    }
    int reset(){
        cadence.stop();
        std::lock_guard<std::mutex> lock(gate);
        visual_active=false;active_window=nullptr;
        if(presenter.initialized){
            if(!presenter.preserve_display(context.Get()))return C3X_RENDERER_RESULT_DEVICE_ERROR;
            presenter.release_native();
        }else presenter.reset();
        return client.reset();
    }
    bool replay_display(std::vector<unsigned>& pixels,unsigned& width,unsigned& height){
        std::lock_guard<std::mutex> lock(gate);
        return c3x_inputs::display_pixels(device.Get(),context.Get(),presenter.retained(),pixels,width,height);
    }
};
}
