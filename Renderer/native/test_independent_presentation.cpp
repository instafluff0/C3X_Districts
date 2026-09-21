#define NOMINMAX
#include <windows.h>
#include <thread>
#include <cstdio>
#include "gpu_native_presenter.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
using namespace c3x_gpu_images;
int test_independent_presentation(){
    WNDCLASSA wc={};wc.lpfnWndProc=DefWindowProcA;wc.hInstance=GetModuleHandleA(nullptr);wc.lpszClassName="C3XIndependentPresentationProbe";
    if(!RegisterClassA(&wc))return 1;
    HWND window=CreateWindowExA(WS_EX_TOPMOST,wc.lpszClassName,"Renderer presentation test",WS_POPUP|WS_VISIBLE,30,30,320,240,nullptr,nullptr,wc.hInstance,nullptr);
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context));
    NativePresenter presenter;if(!presenter.prepare(window,device.Get(),320,240,true))return 2;
    float black[4]={0,0,0,1};context->ClearRenderTargetView(presenter.view(),black);context->CopyResource(presenter.buffer(),presenter.retained());context->Flush();
    if(presenter.present()!=C3X_RENDERER_RESULT_OK)return 3;
    auto library=LoadLibraryA("dwmapi.dll");auto finish=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(library,"DwmFlush"));
    HANDLE done=CreateEventA(nullptr,TRUE,FALSE,nullptr);unsigned presents=0,visible_changes=0;int failure=0;
    std::thread render([&]{
        auto desktop=GetDC(nullptr);COLORREF previous=GetPixel(desktop,80,80);
        for(unsigned i=0;i<30;++i){float color[]={i%2?1.f:0.f,i%2?0.f:1.f,0.f,1.f};
            context->ClearRenderTargetView(presenter.view(),color);context->CopyResource(presenter.buffer(),presenter.retained());context->Flush();
            int code=presenter.present(true);if(code==C3X_RENDERER_RESULT_OK)++presents;else if(code!=C3X_RENDERER_RESULT_PENDING){failure=4;break;}
            if(FAILED(finish())){failure=5;break;}Sleep(33);
            auto pixel=GetPixel(desktop,80,80);if(pixel!=CLR_INVALID&&pixel!=previous)++visible_changes;previous=pixel;
        }
        ReleaseDC(nullptr,desktop);SetEvent(done);
    });
    // Deliberately do not PeekMessage, DispatchMessage or call a renderer API.
    // A HWND-bound Present that waits on this UI thread cannot pass the probe.
    if(WaitForSingleObject(done,10000)!=WAIT_OBJECT_0){std::fprintf(stderr,"FAIL presenter waits on blocked window thread\n");ExitProcess(6);}
    render.join();CloseHandle(done);
    if(!failure&&(presents<10||visible_changes<10))failure=7;
    presenter.release_native();
    auto dc=GetDC(window);RECT area={0,0,320,240};auto brush=CreateSolidBrush(RGB(0,0,255));FillRect(dc,&area,brush);GdiFlush();DeleteObject(brush);ReleaseDC(window,dc);
    finish();Sleep(50);dc=GetDC(nullptr);auto native=GetPixel(dc,80,80);ReleaseDC(nullptr,dc);
    if(native!=RGB(0,0,255))failure=8;
    std::printf("%s independent presentation: presents=%u visible_changes=%u blocked_UI_ms>=990 native_GDI_restored=%d failure=%d\n",failure?"FAIL":"PASS",presents,visible_changes,int(native==RGB(0,0,255)),failure);
    FreeLibrary(library);DestroyWindow(window);UnregisterClassA(wc.lpszClassName,wc.hInstance);return failure;
}
