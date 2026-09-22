#define NOMINMAX
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_4.h>
#include <psapi.h>
#include <wrl/client.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../gpu_native_presenter.h"

using Microsoft::WRL::ComPtr;
using c3x_gpu_images::NativePresenter;

namespace {
void require(bool value, char const* what) {
    if (!value) throw std::runtime_error(what);
}
void check(HRESULT value, char const* what) {
    if (FAILED(value)) {
        char text[160];
        std::snprintf(text, sizeof(text), "%s: 0x%08lx", what, static_cast<unsigned long>(value));
        throw std::runtime_error(text);
    }
}
double now_ms() {
    LARGE_INTEGER value = {}, frequency = {};
    QueryPerformanceCounter(&value);
    QueryPerformanceFrequency(&frequency);
    return 1000.0 * double(value.QuadPart) / double(frequency.QuadPart);
}
double percentile(std::vector<double> values, double rank) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end());
    std::size_t count = std::max<std::size_t>(1, std::size_t(std::ceil(rank * values.size())));
    return values[std::min(values.size(), count) - 1];
}
double average(std::vector<double> const& values) {
    double total = 0;
    for (double value : values) total += value;
    return values.empty() ? 0 : total / values.size();
}
std::size_t largest_free_region() {
    std::uintptr_t cursor = 0;
    std::size_t largest = 0;
    MEMORY_BASIC_INFORMATION information = {};
    while (VirtualQuery(reinterpret_cast<void*>(cursor), &information, sizeof(information)) == sizeof(information)) {
        if (information.State == MEM_FREE) largest = std::max(largest, std::size_t(information.RegionSize));
        std::uintptr_t next = reinterpret_cast<std::uintptr_t>(information.BaseAddress) + information.RegionSize;
        if (next <= cursor) break;
        cursor = next;
    }
    return largest;
}
struct Memory {
    unsigned long long available_virtual = 0, largest_free = 0, private_bytes = 0;
    unsigned long long gpu_budget = 0, gpu_usage = 0;
};
Memory memory_sample(IDXGIAdapter* adapter) {
    Memory result;
    MEMORYSTATUSEX status = {}; status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    result.available_virtual = status.ullAvailVirtual;
    result.largest_free = largest_free_region();
    PROCESS_MEMORY_COUNTERS_EX counters = {};
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters), sizeof(counters));
    result.private_bytes = counters.PrivateUsage;
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
DWORD expected_color(unsigned bits) {
    return RGB(bits & 1 ? 255 : 0, bits & 2 ? 255 : 0, bits & 4 ? 255 : 0);
}
DWORD expected_gpu_pixel(unsigned bits) {
    return 0xff000000u | (bits & 1 ? 0x00ff0000u : 0u) |
           (bits & 2 ? 0x0000ff00u : 0u) | (bits & 4 ? 0x000000ffu : 0u);
}
DWORD pixel(ID3D11Device* device, ID3D11DeviceContext* context, ID3D11Texture2D* source, unsigned x, unsigned y) {
    D3D11_TEXTURE2D_DESC desc = {};
    source->GetDesc(&desc);
    require(x < desc.Width && y < desc.Height, "pixel bounds");
    desc.Width = desc.Height = 1;
    desc.BindFlags = desc.MiscFlags = 0;
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> stage;
    check(device->CreateTexture2D(&desc, nullptr, &stage), "create verification stage");
    D3D11_BOX box = {x, y, 0, x + 1, y + 1, 1};
    context->CopySubresourceRegion(stage.Get(), 0, 0, 0, 0, source, 0, &box);
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    check(context->Map(stage.Get(), 0, D3D11_MAP_READ, 0, &mapped), "map verification pixel");
    DWORD result = *reinterpret_cast<DWORD const*>(mapped.pData);
    context->Unmap(stage.Get(), 0);
    return result;
}
LRESULT CALLBACK window_proc(HWND hwnd, UINT message, WPARAM wp, LPARAM lp) {
    if (message == WM_DESTROY) return 0;
    return DefWindowProcA(hwnd, message, wp, lp);
}
LRESULT CALLBACK child_proc(HWND hwnd, UINT message, WPARAM wp, LPARAM lp) {
    if (message == WM_ERASEBKGND || message == WM_PAINT) {
        PAINTSTRUCT paint = {};
        HDC dc = message == WM_PAINT ? BeginPaint(hwnd, &paint) : reinterpret_cast<HDC>(wp);
        RECT area = {}; GetClientRect(hwnd, &area);
        HBRUSH brush = CreateSolidBrush(RGB(0, 0, 255));
        FillRect(dc, &area, brush);
        DeleteObject(brush);
        if (message == WM_PAINT) EndPaint(hwnd, &paint);
        return 1;
    }
    return DefWindowProcA(hwnd, message, wp, lp);
}
struct Window {
    HWND hwnd = nullptr, child = nullptr;
    bool visible;
    Window(unsigned width, unsigned height, bool show) : visible(show) {
        HINSTANCE instance = GetModuleHandleA(nullptr);
        WNDCLASSA cls = {}; cls.lpfnWndProc = window_proc; cls.hInstance = instance;
        cls.lpszClassName = "C3XHelperTrialWindow";
        require(RegisterClassA(&cls) != 0, "register trial window");
        WNDCLASSA ui = {}; ui.lpfnWndProc = child_proc; ui.hInstance = instance;
        ui.lpszClassName = "C3XHelperTrialNativeUI";
        require(RegisterClassA(&ui) != 0, "register native UI window");
        hwnd = CreateWindowExA(show ? WS_EX_TOPMOST : 0, cls.lpszClassName, "C3X helper trial",
                               WS_POPUP | WS_CLIPCHILDREN, 0, 0, int(width), int(height), nullptr, nullptr, instance, nullptr);
        require(hwnd != nullptr, "create trial window");
        child = CreateWindowExA(0, ui.lpszClassName, "Native UI", WS_CHILD | WS_VISIBLE,
                                16, 16, 80, 40, hwnd, nullptr, instance, nullptr);
        require(child != nullptr, "create native UI child");
        if (show) ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        UpdateWindow(child);
        MSG message = {};
        while (PeekMessageA(&message, nullptr, 0, 0, PM_REMOVE)) {
            TranslateMessage(&message); DispatchMessageA(&message);
        }
    }
    ~Window() {
        if (child) DestroyWindow(child);
        if (hwnd) DestroyWindow(hwnd);
        HINSTANCE instance = GetModuleHandleA(nullptr);
        UnregisterClassA("C3XHelperTrialNativeUI", instance);
        UnregisterClassA("C3XHelperTrialWindow", instance);
    }
    void resize(unsigned width, unsigned height) {
        require(SetWindowPos(hwnd, nullptr, 0, 0, int(width), int(height),
                             SWP_NOACTIVATE | SWP_NOZORDER) != FALSE, "resize trial window");
        RECT client = {}; GetClientRect(hwnd, &client);
        require(client.right == int(width) && client.bottom == int(height), "exact resized client");
    }
};
struct TrialDevice {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11Device1> device1;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGIAdapter> adapter;
    DXGI_ADAPTER_DESC description = {};
    TrialDevice() {
        D3D_FEATURE_LEVEL feature = {};
        check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
                                &device, &feature, &context), "x86 hardware D3D11 device");
        check(device.As(&device1), "x86 D3D11.1 interface");
        ComPtr<IDXGIDevice> dxgi;
        check(device.As(&dxgi), "x86 DXGI device");
        check(dxgi->GetAdapter(&adapter), "x86 adapter");
        check(adapter->GetDesc(&description), "x86 adapter description");
    }
};
std::string producer_path() {
    char path[MAX_PATH] = {};
    require(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0, "consumer module path");
    std::string result(path);
    std::size_t slash = result.find_last_of("\\/");
    require(slash != std::string::npos, "consumer directory");
    return result.substr(0, slash + 1) + "producer.exe";
}
struct Producer {
    HANDLE process = nullptr, input = nullptr, output = nullptr;
    HANDLE handles[2] = {nullptr, nullptr};
    ComPtr<ID3D11Texture2D> texture[2];
    ComPtr<IDXGIKeyedMutex> mutex[2];
    Memory memory;
    double startup_ms = 0;
    LUID luid = {};
    Producer(TrialDevice& graphics, unsigned width, unsigned height) {
        SECURITY_ATTRIBUTES security = {sizeof(security), nullptr, TRUE};
        HANDLE child_input = nullptr, child_output = nullptr;
        require(CreatePipe(&child_input, &input, &security, 0) != FALSE, "producer input pipe");
        require(SetHandleInformation(input, HANDLE_FLAG_INHERIT, 0) != FALSE, "input inheritance");
        require(CreatePipe(&output, &child_output, &security, 0) != FALSE, "producer output pipe");
        require(SetHandleInformation(output, HANDLE_FLAG_INHERIT, 0) != FALSE, "output inheritance");
        std::string executable = producer_path();
        std::string command = "\"" + executable + "\" " + std::to_string(GetCurrentProcessId()) +
                              " " + std::to_string(width) + " " + std::to_string(height);
        std::vector<char> command_line(command.begin(), command.end()); command_line.push_back('\0');
        STARTUPINFOA startup = {}; startup.cb = sizeof(startup); startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = child_input; startup.hStdOutput = child_output; startup.hStdError = child_output;
        PROCESS_INFORMATION created = {};
        BOOL launched = CreateProcessA(executable.c_str(), command_line.data(), nullptr, nullptr,
                                       TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &created);
        CloseHandle(child_input); CloseHandle(child_output);
        require(launched != FALSE, "launch x64 producer");
        process = created.hProcess; CloseHandle(created.hThread);
        try {
        std::string line = read_line();
        std::istringstream ready(line);
        std::string tag;
        unsigned long long handle0 = 0, handle1 = 0;
        long high = 0; unsigned long low = 0; unsigned feature = 0;
        ready >> tag >> handle0 >> handle1 >> high >> low >> feature >> startup_ms
              >> memory.available_virtual >> memory.largest_free >> memory.private_bytes
              >> memory.gpu_budget >> memory.gpu_usage;
        require(tag == "READY" && !ready.fail(), "producer READY/metrics");
        luid.HighPart = high; luid.LowPart = low;
        require(luid.HighPart == graphics.description.AdapterLuid.HighPart &&
                luid.LowPart == graphics.description.AdapterLuid.LowPart, "adapter LUID mismatch");
        handles[0] = reinterpret_cast<HANDLE>(std::uintptr_t(handle0));
        handles[1] = reinterpret_cast<HANDLE>(std::uintptr_t(handle1));
        for (unsigned slot = 0; slot < 2; ++slot) {
            check(graphics.device1->OpenSharedResource1(handles[slot], IID_PPV_ARGS(&texture[slot])),
                  "OpenSharedResource1 on x86");
            check(texture[slot].As(&mutex[slot]), "x86 keyed mutex");
        }
        } catch (...) {
            stop(true);
            throw;
        }
    }
    ~Producer() { stop(false); }
    std::string read_line() {
        std::string line;
        double deadline = now_ms() + 10000.0;
        while (now_ms() < deadline) {
            DWORD available = 0;
            if (!PeekNamedPipe(output, nullptr, 0, nullptr, &available, nullptr)) {
                throw std::runtime_error("producer pipe closed before reply");
            }
            if (available) {
                char character = 0; DWORD got = 0;
                require(ReadFile(output, &character, 1, &got, nullptr) != FALSE && got == 1,
                        "read producer reply");
                if (character == '\n') return line;
                if (line.size() >= 511) throw std::runtime_error("oversized producer reply");
                line.push_back(character);
            } else {
                require(WaitForSingleObject(process, 0) == WAIT_TIMEOUT,
                        "producer exited before reply");
                Sleep(1);
            }
        }
        throw std::runtime_error("producer reply timeout");
    }
    std::string frame(unsigned sequence, unsigned color) {
        std::string command = "FRAME " + std::to_string(sequence) + " " + std::to_string(color) + "\n";
        DWORD written = 0;
        require(WriteFile(input, command.data(), DWORD(command.size()), &written, nullptr) != FALSE &&
                written == command.size(), "write producer frame command");
        return read_line();
    }
    Memory metric() {
        char const command[] = "METRIC\n"; DWORD written = 0;
        require(WriteFile(input, command, DWORD(sizeof(command) - 1), &written, nullptr) != FALSE &&
                written == sizeof(command) - 1, "write producer metric command");
        std::istringstream response(read_line());
        std::string kind; Memory result;
        response >> kind >> result.available_virtual >> result.largest_free >> result.private_bytes
                 >> result.gpu_budget >> result.gpu_usage;
        require(kind == "METRIC" && !response.fail(), "producer metric reply");
        return result;
    }
    void stop(bool crash) {
        if (!process) return;
        if (crash) TerminateProcess(process, 0);
        else if (input) {
            char const command[] = "EXIT\n"; DWORD written = 0;
            WriteFile(input, command, DWORD(sizeof(command) - 1), &written, nullptr);
        }
        if (WaitForSingleObject(process, 3000) == WAIT_TIMEOUT) {
            TerminateProcess(process, 0);
            WaitForSingleObject(process, 3000);
        }
        for (unsigned slot = 0; slot < 2; ++slot) {
            mutex[slot].Reset(); texture[slot].Reset();
            if (handles[slot]) { CloseHandle(handles[slot]); handles[slot] = nullptr; }
        }
        CloseHandle(process); process = nullptr;
        if (input) { CloseHandle(input); input = nullptr; }
        if (output) { CloseHandle(output); output = nullptr; }
    }
};
struct Timings {
    std::vector<double> ipc, acquire, copy, present, desktop, total, producer_submit, local_copy;
    unsigned frames = 0, dropped = 0, desktop_changes = 0;
    unsigned desktop_samples = 0, desktop_map_matches = 0, desktop_ui_matches = 0;
    bool desktop_seen = false;
};
void write_report(std::string const& path, bool success, std::string const& error,
                  bool visible, bool desktop_seen, bool desktop_pass, Timings const& timing,
                  Memory const& x86, Memory const& x64, Memory const& x64_end,
                  LUID luid, unsigned width, unsigned height) {
    std::ofstream stream(path, std::ios::binary);
    require(bool(stream), "open native report");
    auto verdict = [&](bool pass) { return pass ? "pass" : "fail"; };
    stream << "{\n"
           << "  \"status\": \"" << verdict(success) << "\",\n"
           << "  \"presentation_backend\": \"NativePresenter-DComp\",\n"
           << "  \"cross_process_shared_import\": " << (timing.frames ? "true" : "false") << ",\n"
           << "  \"desktop_witness\": \"" << (visible ? verdict(desktop_seen && desktop_pass) : "unavailable") << "\",\n"
           << "  \"checks\": {\n"
           << "    \"import\": \"" << verdict(timing.frames > 0) << "\",\n"
           << "    \"exact_frames\": \"" << verdict(success && timing.frames > 0) << "\",\n"
           << "    \"partial_native\": \"" << verdict(success && timing.frames > 0) << "\",\n"
           << "    \"resize\": \"" << verdict(success && timing.frames > 0) << "\",\n"
           << "    \"helper_restart\": \"" << verdict(success && timing.frames > 0) << "\",\n"
           << "    \"blocked_ui_presentation\": \"" << verdict(success && timing.frames > 0) << "\",\n"
           << "    \"native_gdi_restore\": \"" << verdict(success && timing.frames > 0) << "\",\n"
           << "    \"backpressure\": \"" << verdict(success && timing.dropped > 0) << "\"\n"
           << "  },\n"
           << "  \"adapter_luid\": {\"match\": " << (timing.frames ? "true" : "false")
           << ", \"high\": " << luid.HighPart << ", \"low\": " << luid.LowPart << "},\n"
           << "  \"environment\": {\"viewport_width\": " << width << ", \"viewport_height\": " << height
           << ", \"frames\": " << timing.frames << ", \"dropped\": " << timing.dropped
           << ", \"desktop_samples\": " << timing.desktop_samples
           << ", \"desktop_map_matches\": " << timing.desktop_map_matches
           << ", \"desktop_ui_matches\": " << timing.desktop_ui_matches << "},\n"
           << "  \"timing\": {\n"
           << "    \"ipc_ms_mean\": " << average(timing.ipc) << ", \"ipc_ms_p95\": " << percentile(timing.ipc, .95) << ",\n"
           << "    \"acquire_ms_mean\": " << average(timing.acquire) << ", \"acquire_ms_p95\": " << percentile(timing.acquire, .95) << ",\n"
           << "    \"gpu_copy_submit_ms_mean\": " << average(timing.copy) << ", \"gpu_copy_submit_ms_p95\": " << percentile(timing.copy, .95) << ",\n"
           << "    \"present_call_ms_mean\": " << average(timing.present) << ", \"present_call_ms_p95\": " << percentile(timing.present, .95) << ",\n"
           << "    \"desktop_completion_ms_mean\": " << average(timing.desktop) << ", \"desktop_completion_ms_p95\": " << percentile(timing.desktop, .95) << ",\n"
           << "    \"request_to_desktop_ms_mean\": " << average(timing.total) << ", \"request_to_desktop_ms_p95\": " << percentile(timing.total, .95) << ",\n"
           << "    \"producer_submit_ms_mean\": " << average(timing.producer_submit) << ", \"local_copy_submit_ms_mean\": " << average(timing.local_copy) << "\n"
           << "  },\n"
           << "  \"memory\": {\n"
           << "    \"x86_available_virtual\": " << x86.available_virtual << ", \"x86_largest_free\": " << x86.largest_free
           << ", \"x86_private_bytes\": " << x86.private_bytes << ", \"x86_gpu_budget\": " << x86.gpu_budget
           << ", \"x86_gpu_usage\": " << x86.gpu_usage << ",\n"
           << "    \"x64_available_virtual\": " << x64.available_virtual << ", \"x64_largest_free\": " << x64.largest_free
           << ", \"x64_private_bytes\": " << x64.private_bytes << ", \"x64_gpu_budget\": " << x64.gpu_budget
           << ", \"x64_gpu_usage\": " << x64.gpu_usage << ",\n"
           << "    \"x64_end_available_virtual\": " << x64_end.available_virtual
           << ", \"x64_end_largest_free\": " << x64_end.largest_free
           << ", \"x64_end_private_bytes\": " << x64_end.private_bytes
           << ", \"x64_end_gpu_usage\": " << x64_end.gpu_usage << "\n"
           << "  },\n"
           << "  \"error\": \"" << error << "\"\n"
           << "}\n";
}
void seed_presenter(TrialDevice& graphics, NativePresenter& presenter) {
    float black[4] = {0, 0, 0, 1};
    graphics.context->ClearRenderTargetView(presenter.view(), black);
    graphics.context->CopyResource(presenter.buffer(), presenter.retained());
    graphics.context->Flush();
    require(presenter.present() == C3X_RENDERER_RESULT_OK, "initial owner-thread present");
}
void invoke_desktop_flush() {
    HMODULE library = LoadLibraryA("dwmapi.dll");
    if (!library) throw std::runtime_error("dwmapi unavailable");
    auto flush = reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(library, "DwmFlush"));
    require(flush != nullptr, "DwmFlush unavailable");
    HRESULT result = flush();
    FreeLibrary(library);
    check(result, "DwmFlush");
}
DWORD desktop_pixel(int x, int y) {
    HDC dc = GetDC(nullptr);
    if (!dc) return CLR_INVALID;
    DWORD value = GetPixel(dc, x, y);
    ReleaseDC(nullptr, dc);
    return value;
}
struct FrameRunner {
    TrialDevice& graphics;
    NativePresenter& presenter;
    Window& window;
    unsigned width, height;
    bool visible;
    Timings& timing;
    Memory& x64;
    Memory& x64_end;
    LUID& luid;
    FrameRunner(TrialDevice& g, NativePresenter& p, Window& w, unsigned iw, unsigned ih,
                bool show, Timings& t, Memory& producer_memory, Memory& producer_end, LUID& producer_luid)
        : graphics(g), presenter(p), window(w), width(iw), height(ih), visible(show),
          timing(t), x64(producer_memory), x64_end(producer_end), luid(producer_luid) {}
    void display(Producer& producer, unsigned sequence, unsigned color, bool verify_pixel) {
        double start = now_ms();
        std::string response = producer.frame(sequence, color);
        double ready = now_ms();
        std::istringstream line(response);
        std::string kind; unsigned reply_sequence = 0; double submit = 0;
        line >> kind >> reply_sequence;
        require(reply_sequence == sequence, "producer sequence identity");
        require(kind != "DROP", "unexpected normal-frame backpressure");
        require(kind == "FRAME", "producer frame reply");
        line >> submit;
        unsigned slot = sequence & 1u;
        double acquired_begin = now_ms();
        HRESULT acquisition = WAIT_TIMEOUT;
        double deadline = acquired_begin + 3000;
        while (now_ms() < deadline) {
            acquisition = producer.mutex[slot]->AcquireSync(1, 0);
            if (acquisition == S_OK) break;
            require(acquisition == static_cast<HRESULT>(WAIT_TIMEOUT), "keyed acquire failed/abandoned");
            Sleep(1);
        }
        require(acquisition == S_OK, "keyed acquire timed out");
        double acquired = now_ms();
        graphics.context->CopyResource(presenter.retained(), producer.texture[slot].Get());
        graphics.context->CopyResource(presenter.buffer(), presenter.retained());
        graphics.context->Flush();
        double copied = now_ms();
        check(producer.mutex[slot]->ReleaseSync(0), "keyed release");
        if (verify_pixel) {
            DWORD found = pixel(graphics.device.Get(), graphics.context.Get(), presenter.retained(),
                                std::min(width - 1, 150u), std::min(height - 1, 100u));
            DWORD expected = expected_gpu_pixel(color);
            require((found & 0xffffffu) == (expected & 0xffffffu), "exact shared frame pixel");
        }
        int result = C3X_RENDERER_RESULT_PENDING;
        double present_start = now_ms();
        for (unsigned attempt = 0; attempt < 1000 && result == C3X_RENDERER_RESULT_PENDING; ++attempt) {
            result = presenter.present(true);
            if (result == C3X_RENDERER_RESULT_PENDING) Sleep(1);
        }
        require(result == C3X_RENDERER_RESULT_OK, "independent DComp present");
        double presented = now_ms();
        invoke_desktop_flush();
        double desktop = now_ms();
        if (visible) {
            DWORD found = desktop_pixel(150, 100);
            DWORD ui = desktop_pixel(40, 35);
            ++timing.desktop_samples;
            bool map_match = found != CLR_INVALID && (found & 0xffffffu) == (DWORD(expected_color(color)) & 0xffffffu);
            bool ui_match = ui != CLR_INVALID && (ui & 0xffffffu) == (DWORD(RGB(0, 0, 255)) & 0xffffffu);
            timing.desktop_map_matches += map_match;
            timing.desktop_ui_matches += ui_match;
            if (timing.desktop_samples <= 4 && (!map_match || !ui_match))
                std::printf("DESKTOP_DIAGNOSTIC sequence=%u map=%06lx expected=%06lx ui=%06lx\n",
                            sequence, static_cast<unsigned long>(found & 0xffffffu),
                            static_cast<unsigned long>(expected_color(color) & 0xffffffu),
                            static_cast<unsigned long>(ui & 0xffffffu));
            if (map_match && ui_match) {
                ++timing.desktop_changes; timing.desktop_seen = true;
            }
        }
        timing.ipc.push_back(ready - start);
        timing.acquire.push_back(acquired - acquired_begin);
        timing.copy.push_back(copied - acquired);
        timing.present.push_back(presented - present_start);
        timing.desktop.push_back(desktop - presented);
        timing.total.push_back(desktop - start);
        timing.producer_submit.push_back(submit);
        ++timing.frames;
    }
    void local_control(unsigned color) {
        D3D11_TEXTURE2D_DESC desc = {};
        presenter.retained()->GetDesc(&desc);
        desc.MiscFlags = 0;
        ComPtr<ID3D11Texture2D> local;
        ComPtr<ID3D11RenderTargetView> target;
        check(graphics.device->CreateTexture2D(&desc, nullptr, &local), "local control texture");
        check(graphics.device->CreateRenderTargetView(local.Get(), nullptr, &target), "local control target");
        float rgba[4] = {color & 1 ? 1.f : 0.f, color & 2 ? 1.f : 0.f, color & 4 ? 1.f : 0.f, 1.f};
        graphics.context->ClearRenderTargetView(target.Get(), rgba);
        double start = now_ms();
        graphics.context->CopyResource(presenter.retained(), local.Get());
        graphics.context->CopyResource(presenter.buffer(), presenter.retained());
        graphics.context->Flush();
        timing.local_copy.push_back(now_ms() - start);
        int result = C3X_RENDERER_RESULT_PENDING;
        for (unsigned attempt = 0; attempt < 1000 && result == C3X_RENDERER_RESULT_PENDING; ++attempt) {
            result = presenter.present(true);
            if (result == C3X_RENDERER_RESULT_PENDING) Sleep(1);
        }
        require(result == C3X_RENDERER_RESULT_OK, "local control present");
        invoke_desktop_flush();
    }
    void run(unsigned count) {
        Producer producer(graphics, width, height);
        x64 = producer.memory; luid = producer.luid;
        for (unsigned i = 0; i < 8; ++i) local_control(i & 7u);
        // Occupy both slots, then prove the third command is dropped rather than
        // overwriting a frame that has not been adopted by the x86 presenter.
        std::string first = producer.frame(0, 1);
        std::string second = producer.frame(1, 2);
        require(first.rfind("FRAME 0 ", 0) == 0 && second.rfind("FRAME 1 ", 0) == 0,
                "initial bounded shared slots");
        std::string third = producer.frame(2, 3);
        require(third.rfind("DROP 2 ", 0) == 0, "bounded backpressure drop");
        ++timing.dropped;
        auto consume_held = [&](unsigned sequence, unsigned color) {
            unsigned slot = sequence & 1u;
            HRESULT result = WAIT_TIMEOUT;
            double deadline = now_ms() + 3000;
            while (now_ms() < deadline) {
                result = producer.mutex[slot]->AcquireSync(1, 0);
                if (result == S_OK) break;
                require(result == static_cast<HRESULT>(WAIT_TIMEOUT), "held keyed acquire failure");
                Sleep(1);
            }
            require(result == S_OK, "held keyed acquire timeout");
            graphics.context->CopyResource(presenter.retained(), producer.texture[slot].Get());
            graphics.context->CopyResource(presenter.buffer(), presenter.retained());
            graphics.context->Flush();
            check(producer.mutex[slot]->ReleaseSync(0), "held keyed release");
            DWORD found = pixel(graphics.device.Get(), graphics.context.Get(), presenter.retained(), 150, 100);
            require((found & 0xffffffu) == (expected_gpu_pixel(color) & 0xffffffu),
                    "held frame was overwritten");
            require(presenter.present(true) == C3X_RENDERER_RESULT_OK, "held frame present");
            invoke_desktop_flush();
            ++timing.frames;
        };
        consume_held(0, 1);
        consume_held(1, 2);
        for (unsigned sequence = 3; sequence < count; ++sequence) {
            display(producer, sequence, sequence & 7u,
                    sequence == 3 || sequence == count / 2 || sequence == count - 1);
            Sleep(16);
        }
        // A native 16-bit partial transfer must change only its rectangle.
        DWORD prior = pixel(graphics.device.Get(), graphics.context.Get(), presenter.retained(), 150, 100);
        unsigned pitch = (width + 1u) & ~1u;
        std::vector<unsigned short> native(std::size_t(pitch) * height, 0);
        RECT patch = {120, 80, 180, 140};
        for (int y = patch.top; y < patch.bottom; ++y)
            for (int x = patch.left; x < patch.right; ++x)
                native[std::size_t(y) * pitch + x] = 0xf800;
        require(presenter.upload_screen(graphics.context.Get(), native.data(), width, height, patch, 2),
                "partial native screen upload");
        DWORD inside = pixel(graphics.device.Get(), graphics.context.Get(), presenter.retained(), 150, 100);
        DWORD outside = pixel(graphics.device.Get(), graphics.context.Get(), presenter.retained(), 200, 150);
        require((inside & 0xffffffu) == (expected_gpu_pixel(1) & 0xffffffu) && outside == prior,
                "partial native transfer damaged map");
        require(presenter.present(true) == C3X_RENDERER_RESULT_OK, "partial native present");
        invoke_desktop_flush();
        if (visible) {
            DWORD desktop = desktop_pixel(150, 100);
            require((desktop & 0xffffffu) == DWORD(RGB(255, 0, 0)),
                    "partial native desktop pixel");
        }
        x64_end = producer.metric();
        // Destroy the producer while retaining the displayed x86 frame. A new
        // process must provide a fresh handle/session without a stale publication.
        producer.stop(true);
        Producer restarted(graphics, width, height);
        require(restarted.luid.HighPart == luid.HighPart && restarted.luid.LowPart == luid.LowPart,
                "restart adapter changed");
        display(restarted, count + 1, 5, true);
    }
};
}

int main(int argc, char** argv) {
    std::string report;
    unsigned width = 2240, height = 1260, frames = 120;
    bool visible = false, self_test = false, success = false;
    std::string error;
    Timings timing;
    Memory x86, x64, x64_end;
    LUID luid = {};
    try {
        for (int i = 1; i < argc; ++i) {
            std::string option = argv[i];
            if (option == "--self-test") self_test = true;
            else if (option == "--visible") visible = true;
            else if (i + 1 < argc && option == "--report") report = argv[++i];
            else if (i + 1 < argc && option == "--frames") frames = unsigned(std::strtoul(argv[++i], nullptr, 10));
            else if (i + 1 < argc && option == "--width") width = unsigned(std::strtoul(argv[++i], nullptr, 10));
            else if (i + 1 < argc && option == "--height") height = unsigned(std::strtoul(argv[++i], nullptr, 10));
            else throw std::runtime_error("unknown or incomplete argument");
        }
        require(self_test && !report.empty() && frames >= 8 && frames <= 1000 &&
                width >= 320 && width <= 2240 && height >= 240 && height <= 1260, "trial arguments");
        Window window(width, height, visible);
        TrialDevice graphics;
        NativePresenter presenter;
        require(presenter.prepare(window.hwnd, graphics.device.Get(), width, height, true),
                "NativePresenter preparation");
        seed_presenter(graphics, presenter);
        FrameRunner runner(graphics, presenter, window, width, height, visible, timing, x64, x64_end, luid);
        std::string worker_error;
        std::thread worker([&] {
            try { runner.run(frames); }
            catch (std::exception const& e) { worker_error = e.what(); }
        });
        // Deliberately block the window thread. The worker must import, render,
        // compose, and reach DWM without this message pump.
        double blocked_begin = now_ms();
        worker.join();
        double blocked_ms = now_ms() - blocked_begin;
        require(worker_error.empty(), worker_error.c_str());
        require(blocked_ms >= 1000 && timing.frames >= 8, "blocked UI presentation witness");
        bool desktop_pass = !visible || (timing.desktop_samples >= 4 &&
                                         timing.desktop_changes == timing.desktop_samples);
        require(desktop_pass, "desktop pixel witness");
        presenter.release_native();
        window.resize(std::max(320u, width / 2), std::max(240u, height / 2));
        unsigned resized_width = std::max(320u, width / 2), resized_height = std::max(240u, height / 2);
        require(presenter.prepare(window.hwnd, graphics.device.Get(), resized_width, resized_height, true),
                "resized NativePresenter preparation");
        seed_presenter(graphics, presenter);
        FrameRunner resized(graphics, presenter, window, resized_width, resized_height,
                            visible, timing, x64, x64_end, luid);
        std::thread second([&] {
            try {
                Producer producer(graphics, resized_width, resized_height);
                resized.display(producer, frames + 3, 6, true);
            } catch (std::exception const& e) { worker_error = e.what(); }
        });
        second.join();
        require(worker_error.empty(), worker_error.c_str());
        presenter.release_native();
        HDC dc = GetDC(window.hwnd);
        require(dc != nullptr, "GDI restoration DC");
        RECT full = {0, 0, LONG(resized_width), LONG(resized_height)};
        HBRUSH blue = CreateSolidBrush(RGB(0, 0, 255));
        FillRect(dc, &full, blue); GdiFlush();
        DeleteObject(blue); ReleaseDC(window.hwnd, dc);
        invoke_desktop_flush();
        if (visible) require(desktop_pixel(150, 100) == DWORD(RGB(0, 0, 255)),
                             "native GDI restoration desktop pixel");
        x86 = memory_sample(graphics.adapter.Get());
        success = true;
    } catch (std::exception const& e) {
        error = e.what();
        std::fprintf(stderr, "FAIL %s\n", error.c_str());
    }
    try {
        if (!report.empty()) write_report(report, success, error, visible,
                                          timing.desktop_seen, timing.desktop_samples >= 4 &&
                                          timing.desktop_changes == timing.desktop_samples,
                                          timing, x86, x64, x64_end, luid, width, height);
    } catch (std::exception const& e) {
        std::fprintf(stderr, "FAIL report: %s\n", e.what());
        return 2;
    }
    std::printf("%s helper trial frames=%u dropped=%u desktop_changes=%u\n",
                success ? "PASS" : "FAIL", timing.frames, timing.dropped, timing.desktop_changes);
    return success ? 0 : 1;
}
