// x64 half of the isolated cross-process composition-surface trial.
#define NOMINMAX
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_3.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using Microsoft::WRL::ComPtr;

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    unsigned width = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    unsigned height = static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10));
    char command[128]{};
    if (!std::fgets(command, sizeof(command), stdin)) return 2;
    unsigned long long raw_handle = 0;
    if (sscanf_s(command, "SURFACE %llu", &raw_handle) != 1) return 2;
    HANDLE surface = reinterpret_cast<HANDLE>(static_cast<ULONG_PTR>(raw_handle));
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL level{};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
        &device, &level, &context);
    if (FAILED(hr)) { std::printf("FAIL device %08lx\n", static_cast<unsigned long>(hr)); return 3; }
    ComPtr<IDXGIDevice> dxgi;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<IDXGIFactoryMedia> media;
    hr = device.As(&dxgi);
    if (SUCCEEDED(hr)) hr = dxgi->GetAdapter(&adapter);
    if (SUCCEEDED(hr)) hr = adapter->GetParent(IID_PPV_ARGS(&media));
    if (FAILED(hr)) { std::printf("FAIL factory_media %08lx\n", static_cast<unsigned long>(hr)); return 3; }
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = 2;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    ComPtr<IDXGISwapChain1> swap;
    hr = media->CreateSwapChainForCompositionSurfaceHandle(device.Get(), surface,
        &desc, nullptr, &swap);
    std::printf("SURFACE_SWAP %08lx\n", static_cast<unsigned long>(hr));
    std::fflush(stdout);
    if (FAILED(hr)) return 4;
    auto frame = [&](unsigned color) -> HRESULT {
        ComPtr<ID3D11Texture2D> buffer;
        ComPtr<ID3D11RenderTargetView> target;
        HRESULT result = swap->GetBuffer(0, IID_PPV_ARGS(&buffer));
        if (SUCCEEDED(result)) result = device->CreateRenderTargetView(buffer.Get(), nullptr, &target);
        if (FAILED(result)) return result;
        float rgba[4]{color & 1u ? 1.f : 0.f, color & 2u ? 1.f : 0.f,
                      color & 4u ? 1.f : 0.f, 1.f};
        context->ClearRenderTargetView(target.Get(), rgba);
        return swap->Present(0, 0);
    };
    if (!std::fgets(command, sizeof(command), stdin)) return 5;
    if (std::strcmp(command, "FRAME\n") != 0) return 5;
    hr = frame(1);
    std::printf("FIRST_PRESENT %08lx\n", static_cast<unsigned long>(hr));
    std::fflush(stdout);
    if (FAILED(hr)) return 5;
    if (!std::fgets(command, sizeof(command), stdin)) return 6;
    if (std::strcmp(command, "AUTO\n") != 0) return 6;
    LARGE_INTEGER begin{}, end{}, frequency{};
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&begin);
    for (unsigned i = 0; i < 10; ++i) {
        hr = frame(i & 1u ? 1u : 4u);
        if (FAILED(hr)) {
            std::printf("FAIL auto_present_%u %08lx\n", i, static_cast<unsigned long>(hr));
            std::fflush(stdout);
            return 6;
        }
        Sleep(75);
    }
    QueryPerformanceCounter(&end);
    std::printf("AUTO_COMPLETE frames=10 elapsed_ms=%.3f\n",
        1000. * double(end.QuadPart - begin.QuadPart) / double(frequency.QuadPart));
    std::fflush(stdout);
    if (!std::fgets(command, sizeof(command), stdin)) return 0;
    CloseHandle(surface);
    return 0;
}
