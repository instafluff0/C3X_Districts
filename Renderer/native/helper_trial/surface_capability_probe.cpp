// Isolated driver/API capability probe. Does not attach to Civ III or stage a renderer.
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_3.h>
#include <dcomp.h>
#include <wrl/client.h>
#include <cstdio>

using Microsoft::WRL::ComPtr;

int main() {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL level{};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
        &device, &level, &context);
    std::printf("d3d_device=%08lx feature_level=%04x\n", static_cast<unsigned long>(hr),
        static_cast<unsigned>(level));
    if (FAILED(hr)) return 2;

    ComPtr<IDXGIDevice> dxgi;
    hr = device.As(&dxgi);
    if (FAILED(hr)) return 3;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<IDXGIFactoryMedia> media;
    if (SUCCEEDED(dxgi->GetAdapter(&adapter))) {
        ComPtr<IDXGIFactory2> factory;
        if (SUCCEEDED(adapter->GetParent(IID_PPV_ARGS(&factory)))) factory.As(&media);
    }
    std::printf("dxgi_factory_media=%d\n", media ? 1 : 0);
    bool bgra_supported = false;
    if (media) {
        HANDLE surface = nullptr;
        HRESULT surface_hr = DCompositionCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS,
            nullptr, &surface);
        std::printf("composition_surface_handle=%08lx\n",
            static_cast<unsigned long>(surface_hr));
        if (SUCCEEDED(surface_hr)) {
            DXGI_SWAP_CHAIN_DESC1 swap_desc{};
            swap_desc.Width = 320;
            swap_desc.Height = 240;
            swap_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            swap_desc.SampleDesc.Count = 1;
            swap_desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            swap_desc.BufferCount = 2;
            swap_desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
            swap_desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
            ComPtr<IDXGISwapChain1> surface_swap;
            surface_hr = media->CreateSwapChainForCompositionSurfaceHandle(device.Get(),
                surface, &swap_desc, nullptr, &surface_swap);
            std::printf("bgra_surface_swapchain=%08lx\n",
                static_cast<unsigned long>(surface_hr));
            bgra_supported = SUCCEEDED(surface_hr);
            CloseHandle(surface);
        }
    }
    return bgra_supported ? 0 : 4;
}
