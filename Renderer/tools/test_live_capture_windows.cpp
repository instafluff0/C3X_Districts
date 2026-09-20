// Collector integration probe. It never loads or launches Civ III.
#include <windows.h>
#include <d3d11.h>
#include <cstdio>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"user32.lib")

int main() {
    WNDCLASSA wc={};wc.lpfnWndProc=DefWindowProcA;wc.hInstance=GetModuleHandleA(nullptr);wc.lpszClassName="C3XCaptureProbe";
    if(!RegisterClassA(&wc))return 1;
    HWND window=CreateWindowA(wc.lpszClassName,"Renderer capture check",WS_OVERLAPPEDWINDOW,
        40,40,320,240,nullptr,nullptr,wc.hInstance,nullptr);
    if(!window)return 2;
    DXGI_SWAP_CHAIN_DESC desc={};desc.BufferDesc.Width=320;desc.BufferDesc.Height=240;
    desc.BufferDesc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;desc.SampleDesc.Count=1;
    desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=1;desc.OutputWindow=window;desc.Windowed=TRUE;
    ID3D11Device* device=nullptr;ID3D11DeviceContext* context=nullptr;IDXGISwapChain* swap=nullptr;
    if(FAILED(D3D11CreateDeviceAndSwapChain(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,
        &desc,&swap,&device,nullptr,&context)))return 3;
    ID3D11Texture2D* back=nullptr;ID3D11RenderTargetView* target=nullptr;
    if(FAILED(swap->GetBuffer(0,__uuidof(ID3D11Texture2D),reinterpret_cast<void**>(&back))) ||
       FAILED(device->CreateRenderTargetView(back,nullptr,&target)))return 4;
    ShowWindow(window,SW_SHOWNOACTIVATE);
    for(unsigned i=0;i<120;++i){
        MSG message;while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
        float color[]={i%2?.1f:.2f,.2f,.3f,1};context->ClearRenderTargetView(target,color);
        if(FAILED(swap->Present(0,0)))return 5;
        if(i%30==0)OutputDebugStringA("[C3X renderer] capture-selftest hardware-present\n");
        Sleep(20);
    }
    context->ClearState();target->Release();back->Release();swap->Release();context->Release();device->Release();
    DestroyWindow(window);UnregisterClassA(wc.lpszClassName,wc.hInstance);
    std::puts("PASS collector probe: 120 hardware presents; no game launched");return 0;
}
