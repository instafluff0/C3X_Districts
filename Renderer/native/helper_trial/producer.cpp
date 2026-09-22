#define NOMINMAX
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

using Microsoft::WRL::ComPtr;

namespace {
std::size_t largest_free_region() {
    std::uintptr_t cursor = 0;
    std::size_t largest = 0;
    MEMORY_BASIC_INFORMATION info = {};
    while (VirtualQuery(reinterpret_cast<void*>(cursor), &info, sizeof(info)) == sizeof(info)) {
        if (info.State == MEM_FREE && info.RegionSize > largest) largest = info.RegionSize;
        std::uintptr_t next = reinterpret_cast<std::uintptr_t>(info.BaseAddress) + info.RegionSize;
        if (next <= cursor) break;
        cursor = next;
    }
    return largest;
}
struct Metrics {
    unsigned long long available_virtual = 0, largest_free = 0, private_bytes = 0;
    unsigned long long gpu_budget = 0, gpu_usage = 0;
};
Metrics sample_metrics(IDXGIAdapter* adapter) {
    Metrics result;
    MEMORYSTATUSEX memory = {}; memory.dwLength = sizeof(memory);
    GlobalMemoryStatusEx(&memory);
    result.available_virtual = memory.ullAvailVirtual;
    result.largest_free = largest_free_region();
    PROCESS_MEMORY_COUNTERS_EX usage = {};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&usage), sizeof(usage));
    result.private_bytes = usage.PrivateUsage;
    ComPtr<IDXGIAdapter3> adapter3;
    if (SUCCEEDED(adapter->QueryInterface(IID_PPV_ARGS(&adapter3)))) {
        DXGI_QUERY_VIDEO_MEMORY_INFO video = {};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &video))) {
            result.gpu_budget = video.Budget;
            result.gpu_usage = video.CurrentUsage;
        }
    }
    return result;
}

double milliseconds(LARGE_INTEGER begin, LARGE_INTEGER end) {
    LARGE_INTEGER frequency = {};
    QueryPerformanceFrequency(&frequency);
    return 1000.0 * double(end.QuadPart - begin.QuadPart) / double(frequency.QuadPart);
}

void fail(char const* operation, HRESULT result) {
    std::printf("FAIL %s 0x%08lx\n", operation, static_cast<unsigned long>(result));
    std::fflush(stdout);
}

bool write_frame(ID3D11DeviceContext* context, ID3D11RenderTargetView* target,
                 IDXGIKeyedMutex* mutex, unsigned sequence, unsigned color) {
    HRESULT acquired = mutex->AcquireSync(0, 0);
    if (acquired == static_cast<HRESULT>(WAIT_TIMEOUT)) {
        std::printf("DROP %u busy\n", sequence);
        std::fflush(stdout);
        return true;
    }
    if (acquired != S_OK) {
        fail("AcquireSync", acquired);
        return false;
    }
    float rgba[4] = {0, 0, 0, 1};
    rgba[0] = (color & 1u) ? 1.0f : 0.0f;
    rgba[1] = (color & 2u) ? 1.0f : 0.0f;
    rgba[2] = (color & 4u) ? 1.0f : 0.0f;
    LARGE_INTEGER begin = {}, end = {};
    QueryPerformanceCounter(&begin);
    context->ClearRenderTargetView(target, rgba);
    HRESULT released = mutex->ReleaseSync(1);
    context->Flush(); // Submits the GPU work; this is not a completion fence.
    QueryPerformanceCounter(&end);
    if (released != S_OK) {
        fail("ReleaseSync", released);
        return false;
    }
    std::printf("FRAME %u %.6f\n", sequence, milliseconds(begin, end));
    std::fflush(stdout);
    return true;
}
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fail("arguments", E_INVALIDARG);
        return 2;
    }
    DWORD consumer_pid = static_cast<DWORD>(std::strtoul(argv[1], nullptr, 10));
    unsigned width = static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10));
    unsigned height = static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10));
    if (!consumer_pid || !width || !height || width > 2240 || height > 1260) {
        fail("dimensions", E_INVALIDARG);
        return 2;
    }
    LARGE_INTEGER begin = {}, end = {};
    QueryPerformanceCounter(&begin);
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL feature = {};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0,
                                   D3D11_SDK_VERSION, &device, &feature, &context);
    if (FAILED(hr)) { fail("D3D11CreateDevice", hr); return 3; }
    ComPtr<ID3D11Device1> device1;
    hr = device.As(&device1);
    if (FAILED(hr)) { fail("ID3D11Device1", hr); return 3; }
    ComPtr<IDXGIDevice> dxgi;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC adapter_desc = {};
    hr = device.As(&dxgi);
    if (SUCCEEDED(hr)) hr = dxgi->GetAdapter(&adapter);
    if (SUCCEEDED(hr)) hr = adapter->GetDesc(&adapter_desc);
    if (FAILED(hr)) { fail("adapter", hr); return 3; }
    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = width;
    desc.Height = height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    HANDLE consumer = OpenProcess(PROCESS_DUP_HANDLE | PROCESS_QUERY_LIMITED_INFORMATION,
                                  FALSE, consumer_pid);
    if (!consumer) { fail("OpenProcess", HRESULT_FROM_WIN32(GetLastError())); return 3; }
    ComPtr<ID3D11Texture2D> texture[2];
    ComPtr<ID3D11RenderTargetView> target[2];
    ComPtr<IDXGIKeyedMutex> mutex[2];
    HANDLE shared[2] = {nullptr, nullptr};
    HANDLE duplicated[2] = {nullptr, nullptr};
    for (unsigned slot = 0; slot < 2; ++slot) {
        hr = device->CreateTexture2D(&desc, nullptr, &texture[slot]);
        if (SUCCEEDED(hr)) hr = device->CreateRenderTargetView(texture[slot].Get(), nullptr, &target[slot]);
        if (SUCCEEDED(hr)) hr = texture[slot].As(&mutex[slot]);
        ComPtr<IDXGIResource1> resource;
        if (SUCCEEDED(hr)) hr = texture[slot].As(&resource);
        if (SUCCEEDED(hr)) hr = resource->CreateSharedHandle(nullptr,
                               DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                               nullptr, &shared[slot]);
        if (SUCCEEDED(hr) && !DuplicateHandle(GetCurrentProcess(), shared[slot], consumer,
                                             &duplicated[slot], 0, FALSE, DUPLICATE_SAME_ACCESS))
            hr = HRESULT_FROM_WIN32(GetLastError());
        if (FAILED(hr)) {
            fail("CreateSharedHandle/DuplicateHandle", hr);
            for (HANDLE handle : shared) if (handle) CloseHandle(handle);
            for (HANDLE handle : duplicated) if (handle) {
                HANDLE local = nullptr;
                DuplicateHandle(consumer, handle, GetCurrentProcess(), &local, 0, FALSE,
                                DUPLICATE_CLOSE_SOURCE | DUPLICATE_SAME_ACCESS);
                if (local) CloseHandle(local);
            }
            CloseHandle(consumer);
            return 4;
        }
    }
    Metrics metrics = sample_metrics(adapter.Get());
    QueryPerformanceCounter(&end);
    std::printf("READY %llu %llu %ld %lu %u %.6f %llu %llu %llu %llu %llu\n",
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(duplicated[0])),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(duplicated[1])),
                adapter_desc.AdapterLuid.HighPart, adapter_desc.AdapterLuid.LowPart,
                static_cast<unsigned>(feature), milliseconds(begin, end),
                metrics.available_virtual, metrics.largest_free, metrics.private_bytes,
                metrics.gpu_budget, metrics.gpu_usage);
    std::fflush(stdout);
    char line[128] = {};
    bool protocol_ok = true;
    while (std::fgets(line, sizeof(line), stdin)) {
        if (std::strncmp(line, "EXIT", 4) == 0) break;
        if (std::strncmp(line, "METRIC", 6) == 0) {
            Metrics current = sample_metrics(adapter.Get());
            std::printf("METRIC %llu %llu %llu %llu %llu\n",
                        current.available_virtual, current.largest_free, current.private_bytes,
                        current.gpu_budget, current.gpu_usage);
            std::fflush(stdout);
            continue;
        }
        unsigned sequence = 0, color = 0;
        if (sscanf_s(line, "FRAME %u %u", &sequence, &color) != 2) {
            fail("command", E_INVALIDARG);
            protocol_ok = false;
            break;
        }
        if (!write_frame(context.Get(), target[sequence & 1u].Get(),
                         mutex[sequence & 1u].Get(), sequence, color)) { protocol_ok = false; break; }
    }
    for (HANDLE handle : shared) if (handle) CloseHandle(handle);
    CloseHandle(consumer);
    return protocol_ok ? 0 : 5;
}
