// x86 window owner for the isolated x64 cross-process composition-surface trial.
#define NOMINMAX
#include <windows.h>
#include <d3d11_1.h>
#include <dcomp.h>
#include <dwmapi.h>
#include <wrl/client.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

static LRESULT CALLBACK paint(HWND hwnd, UINT message, WPARAM wp, LPARAM lp) {
    if (message == WM_PAINT) {
        PAINTSTRUCT state{};
        HDC dc = BeginPaint(hwnd, &state);
        RECT bounds{};
        GetClientRect(hwnd, &bounds);
        HBRUSH brush = CreateSolidBrush(GetParent(hwnd) ? RGB(0, 0, 255) : RGB(0, 255, 0));
        FillRect(dc, &bounds, brush);
        DeleteObject(brush);
        EndPaint(hwnd, &state);
        return 0;
    }
    return DefWindowProcA(hwnd, message, wp, lp);
}
static bool read_line(HANDLE pipe, std::string& line) {
    line.clear();
    char value = 0;
    DWORD count = 0;
    while (ReadFile(pipe, &value, 1, &count, nullptr) && count == 1) {
        if (value == '\n') {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            return true;
        }
        if (line.size() > 256) return false;
        line.push_back(value);
    }
    return false;
}
static bool send(HANDLE pipe, std::string const& command) {
    DWORD sent = 0;
    return WriteFile(pipe, command.data(), static_cast<DWORD>(command.size()), &sent, nullptr) &&
        sent == command.size();
}
static COLORREF pixel(int x, int y) {
    HDC dc = GetDC(nullptr);
    COLORREF value = GetPixel(dc, x, y);
    ReleaseDC(nullptr, dc);
    return value;
}
int main(int argc, char** argv) {
    unsigned width = argc > 1 ? static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10)) : 320;
    unsigned height = argc > 2 ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 240;
    if (width < 320 || height < 240 || width > 2240 || height > 1260) return 2;
    WNDCLASSA klass{};
    klass.lpfnWndProc = paint;
    klass.hInstance = GetModuleHandleA(nullptr);
    klass.lpszClassName = "C3XCompositionSurfaceProbe";
    if (!RegisterClassA(&klass)) return 2;
    HWND window = CreateWindowExA(WS_EX_TOPMOST, klass.lpszClassName, "Surface probe",
        WS_POPUP | WS_CLIPCHILDREN, 0, 0, int(width), int(height), nullptr, nullptr, klass.hInstance, nullptr);
    HWND child = CreateWindowExA(0, klass.lpszClassName, "Native child", WS_CHILD | WS_VISIBLE,
        16, 16, 40, 40, window, nullptr, klass.hInstance, nullptr);
    if (!window || !child) return 2;
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    UpdateWindow(child);
    std::printf("native_hdc_before=%06lx native_child_before=%06lx\n",
        static_cast<unsigned long>(pixel(100, 100) & 0xffffff),
        static_cast<unsigned long>(pixel(30, 30) & 0xffffff));

    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL level{};
    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, D3D11_SDK_VERSION,
        &device, &level, &context);
    if (FAILED(hr)) { std::printf("FAIL device %08lx\n", static_cast<unsigned long>(hr)); return 3; }
    ComPtr<IDXGIDevice> dxgi;
    device.As(&dxgi);
    ComPtr<IDCompositionDevice> composition;
    hr = DCompositionCreateDevice(dxgi.Get(), IID_PPV_ARGS(&composition));
    if (FAILED(hr)) { std::printf("FAIL composition %08lx\n", static_cast<unsigned long>(hr)); return 3; }
    HANDLE surface = nullptr;
    hr = DCompositionCreateSurfaceHandle(COMPOSITIONOBJECT_ALL_ACCESS, nullptr, &surface);
    if (FAILED(hr)) { std::printf("FAIL surface_handle %08lx\n", static_cast<unsigned long>(hr)); return 3; }
    ComPtr<IUnknown> wrapper;
    hr = composition->CreateSurfaceFromHandle(surface, &wrapper);
    std::printf("surface_wrapper=%08lx\n", static_cast<unsigned long>(hr));
    if (FAILED(hr)) return 3;
    ComPtr<IDCompositionTarget> target;
    ComPtr<IDCompositionVisual> visual;
    hr = composition->CreateTargetForHwnd(window, FALSE, &target);
    if (SUCCEEDED(hr)) hr = composition->CreateVisual(&visual);
    if (SUCCEEDED(hr)) hr = visual->SetContent(wrapper.Get());
    if (SUCCEEDED(hr)) hr = target->SetRoot(visual.Get());
    if (SUCCEEDED(hr)) hr = composition->Commit();
    std::printf("bind_surface=%08lx\n", static_cast<unsigned long>(hr));
    if (FAILED(hr)) return 3;

    SECURITY_ATTRIBUTES security{sizeof(security), nullptr, TRUE};
    HANDLE child_input = nullptr, input = nullptr, output = nullptr, child_output = nullptr;
    if (!CreatePipe(&child_input, &input, &security, 0) ||
        !CreatePipe(&output, &child_output, &security, 0)) return 4;
    SetHandleInformation(input, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(output, HANDLE_FLAG_INHERIT, 0);
    char module[MAX_PATH]{};
    GetModuleFileNameA(nullptr, module, MAX_PATH);
    std::string path(module);
    path.resize(path.find_last_of("\\/") + 1);
    path += "surface_producer.exe";
    std::string command = "\"" + path + "\" " + std::to_string(width) + " " + std::to_string(height);
    std::vector<char> command_line(command.begin(), command.end());
    command_line.push_back(0);
    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = child_input;
    startup.hStdOutput = startup.hStdError = child_output;
    PROCESS_INFORMATION process{};
    if (!CreateProcessA(path.c_str(), command_line.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process)) return 4;
    CloseHandle(child_input);
    CloseHandle(child_output);
    HANDLE remote_surface = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), surface, process.hProcess, &remote_surface,
        0, FALSE, DUPLICATE_SAME_ACCESS)) return 4;
    if (!send(input, "SURFACE " + std::to_string(reinterpret_cast<ULONG_PTR>(remote_surface)) + "\n")) return 4;
    std::string line;
    if (!read_line(output, line)) return 4;
    std::printf("producer_%s\n", line.c_str());
    if (line != "SURFACE_SWAP 00000000") return 4;
    if (!send(input, "FRAME\n") || !read_line(output, line)) return 4;
    std::printf("producer_%s\n", line.c_str());
    if (line != "FIRST_PRESENT 00000000") return 4;
    COLORREF initial = CLR_INVALID, ui = CLR_INVALID;
    for (unsigned attempt = 0; attempt < 100; ++attempt) {
        DwmFlush();
        initial = pixel(100, 100);
        ui = pixel(30, 30);
        if (initial == RGB(255, 0, 0) && ui == RGB(0, 0, 255)) break;
        Sleep(20);
    }
    std::printf("first_desktop_map=%06lx child_ui=%06lx\n",
        static_cast<unsigned long>(initial & 0xffffff),
        static_cast<unsigned long>(ui & 0xffffff));
    if (!send(input, "AUTO\n")) return 5;
    unsigned red = 0, blue = 0, changes = 0;
    COLORREF previous = CLR_INVALID;
    for (unsigned attempt = 0; attempt < 25; ++attempt) {
        DwmFlush();
        COLORREF current = pixel(100, 100);
        if (current == RGB(255, 0, 0)) ++red;
        if (current == RGB(0, 0, 255)) ++blue;
        if (previous != CLR_INVALID && previous != current) ++changes;
        previous = current;
        Sleep(40); // No HWND message pumping while x64 advances ten frames.
    }
    if (!read_line(output, line)) return 5;
    std::printf("producer_%s\n", line.c_str());
    std::printf("blocked_window_thread_samples red=%u blue=%u changes=%u\n", red, blue, changes);
    target->SetRoot(nullptr);
    composition->Commit();
    composition->WaitForCommitCompletion();
    send(input, "EXIT\n");
    WaitForSingleObject(process.hProcess, 5000);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(input);
    CloseHandle(output);
    CloseHandle(surface);
    DestroyWindow(child);
    DestroyWindow(window);
    return initial == RGB(255, 0, 0) && ui == RGB(0, 0, 255) &&
        red && blue && changes >= 2 ? 0 : 6;
}
