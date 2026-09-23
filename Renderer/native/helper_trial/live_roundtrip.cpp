#define NOMINMAX
#include <windows.h>
#include <d3d11_1.h>
#include <wrl/client.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include "../input_recording/journal.h"
#include "../remote_scene_output.h"
#include "../remote_renderer_client.h"
#include "../gpu_native_presenter.h"

using namespace c3x_inputs;
using namespace c3x_helper_trial;
using Microsoft::WRL::ComPtr;

int wmain(int argc,wchar_t** argv){
    HWND window=nullptr;
    try{
        require(argc==4,"usage: live_roundtrip.exe HELPER DLL CAPTURE");
        SegmentReader journal(argv[3]);Event event;
        require(journal.next(event)&&event.kind==Kind::manifest,"recording manifest missing");
        c3x_remote_scene::Client helper(argv[1],argv[2]);
        bool definitions=false,submitted=false;unsigned polls=0;
        c3x_remote_scene::CameraOutput adopted;
        while(journal.next(event)){
            if(event.kind!=Kind::native_bridge&&event.kind!=Kind::scene)continue;
            Reader recorded{event.payload};recorded.u64();recorded.u64();
            if(event.kind==Kind::native_bridge&&event.flags==8){
                bool present[4]={};std::string path[4];
                for(unsigned n=0;n<4;++n)path[n]=recorded.string(32768,&present[n]);recorded.done();
                require(helper.definitions(present[0]?path[0].c_str():nullptr,
                    present[1]?path[1].c_str():nullptr,present[2]?path[2].c_str():nullptr,
                    present[3]?path[3].c_str():nullptr)==C3X_RENDERER_RESULT_OK,"definition publication failed");
                definitions=true;
            }else if(event.kind==Kind::scene&&event.flags==3&&definitions){
                Bytes values(event.payload.begin()+recorded.at,event.payload.end());Reader input{values};
                c3x_renderer_camera_identity_v1 identity={};c3x_renderer_camera_identity_v1_fields(input,identity);
                Frame frame_value;frame(input,frame_value);input.done();
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame_value.value,identity};
                c3x_renderer_i64 ticket=0;
                require(helper.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING&&ticket>0,"camera begin failed");
                for(;polls<120000;++polls){
                    auto code=helper.camera_poll(ticket,adopted.value);
                    if(code==C3X_RENDERER_RESULT_PENDING){Sleep(1);continue;}
                    require(code==C3X_RENDERER_RESULT_OK,"camera poll failed");
                    require(adopted.value.camera.ticket==ticket&&adopted.value.image.ticket>0&&
                        adopted.value.image.map_image>0&&adopted.value.camera.frame.tile_count==frame_value.value.tile_count,
                        "adopted camera identity or occurrences changed");
                    break;
                }
                require(polls<120000,"camera readiness timed out");submitted=true;break;
            }
        }
        require(submitted,"recording has no GPU scene following definitions");
        auto const& view=adopted.value;auto width=view.image.width,height=view.image.height;
        require(width>0&&height>0&&width<=2240&&height<=1260,"invalid adopted extent");
        window=CreateWindowExW(0,L"STATIC",L"C3X live helper roundtrip",WS_POPUP,0,0,width,height,
            nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
        require(window!=nullptr,"x86 presentation window failed");
        ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level={};
        require(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context)),"x86 device failed");
        ComPtr<ID3D11Device1> device1;require(SUCCEEDED(device.As(&device1)),"x86 device1 failed");
        c3x_gpu_images::NativePresenter presenter;
        require(presenter.prepare(window,device.Get(),unsigned(width),unsigned(height),true),"x86 presenter failed");
        c3x_renderer_gpu_present_v1 offer={sizeof(offer)};offer.ticket=view.image.ticket;
        offer.image=view.image.map_image;offer.width=width;offer.height=height;
        offer.area[2]=width;offer.area[3]=height;
        c3x_remote_scene::SharedFrame shared;
        require(helper.present(offer,shared)==C3X_RENDERER_RESULT_OK&&shared.handle&&
            shared.width==unsigned(width)&&shared.height==unsigned(height),"x64 final shared frame failed");
        require(presenter.adopt_shared(device1.Get(),context.Get(),shared.handle,shared.width,shared.height)==
            C3X_RENDERER_RESULT_OK,"x86 native composition of x64 frame failed");
        presenter.reset();DestroyWindow(window);
        std::printf("PASS live x64 camera to x86 native presenter: %u polls, %dx%d frame\n",polls+1,width,height);
        return 0;
    }catch(std::exception const& error){
        if(window)DestroyWindow(window);
        std::fprintf(stderr,"FAIL live helper roundtrip: %s\n",error.what());return 1;
    }
}
