#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>
#include "../native/c3x_renderer_api.h"
#include "exchange.h"

int sandbox_reference_main(int argc, char** argv);

LRESULT CALLBACK sandbox_window_proc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProcA(window, message, wparam, lparam);
}

int sandbox_client_run(HMODULE module, c3x_renderer_frame_v1 const& prepared_frame) {
    using namespace sandbox_exchange;
    auto draw = reinterpret_cast<int(*)(c3x_renderer_frame_v1 const*, char const*,
        int,int,int,int,int,int,int)>(
        GetProcAddress(module, "c3x_sandbox_draw_fresh"));
    auto present = reinterpret_cast<int(*)(HWND, c3x_renderer_frame_v1 const*,
        int,int,int,int,int,int,int)>(
        GetProcAddress(module, "c3x_sandbox_present"));
    auto observed = reinterpret_cast<LONG(*)()>(GetProcAddress(module, "c3x_sandbox_swapchain_observed"));
    auto metrics = reinterpret_cast<void(*)(double*,unsigned*,unsigned*,unsigned*,
        std::size_t*,float*)>(
        GetProcAddress(module,"c3x_sandbox_fresh_metrics"));
    auto present_metrics = reinterpret_cast<void(*)(double*)>(
        GetProcAddress(module,"c3x_sandbox_present_metrics"));
    auto cache_metrics=reinterpret_cast<void(*)(unsigned*,unsigned*,unsigned*,unsigned*,unsigned*,unsigned*)>(
        GetProcAddress(module,"c3x_sandbox_cache_metrics"));
    char flat_option[8]{};
    bool flat_present = GetEnvironmentVariableA("C3X_SANDBOX_FLAT_PRESENT",
        flat_option,sizeof(flat_option)) && std::strcmp(flat_option,"1")==0;
    if (!draw || !present || !observed) return 2;
    auto inspect = reinterpret_cast<void(*)()>(GetProcAddress(module, "c3x_sandbox_inspect_scene"));
    if (inspect) inspect();
    WNDCLASSA window_class{};
    window_class.lpfnWndProc = sandbox_window_proc;
    window_class.hInstance = GetModuleHandleA(nullptr);
    window_class.lpszClassName = "C3XSandboxClientWindow";
    if (!RegisterClassA(&window_class)) return 3;
    HWND window = CreateWindowExA(0, window_class.lpszClassName, "C3X Sandbox x64",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
        prepared_frame.target_width / 2, prepared_frame.target_height / 2,
        nullptr, nullptr, window_class.hInstance, nullptr);
    if (!window) return 4;
    ShowWindow(window, SW_SHOW);

    HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
        sizeof(Exchange), mapping_name);
    if (!mapping) return 5;
    bool fresh_mapping = GetLastError() != ERROR_ALREADY_EXISTS;
    auto* exchange = static_cast<Exchange*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Exchange)));
    if (!exchange) { CloseHandle(mapping); return 6; }
    if (fresh_mapping) {
        ZeroMemory(exchange, sizeof(Exchange));
        InterlockedExchange(&exchange->ready, ready_magic);
    }
    char program[MAX_PATH]{};
    if (!GetModuleFileNameA(nullptr, program, MAX_PATH)) return 7;
    std::string host_path(program);
    host_path.resize(host_path.find_last_of("\\/") + 1);
    host_path += "synthetic_host_x86.exe";
    STARTUPINFOA startup{}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION host{};
    std::string command = '"' + host_path + '"';
    std::vector<char> command_line(command.begin(), command.end());
    command_line.push_back(0);
    if (!CreateProcessA(host_path.c_str(), command_line.data(), nullptr, nullptr,
        TRUE, CREATE_NO_WINDOW, nullptr, nullptr, &startup, &host)) {
        std::printf("CLIENT_HOST_START_ERROR code=%lu\n", GetLastError());
        return 8;
    }

    auto start = GetTickCount64();
    LONG host_before_pause = 0, host_after_pause = 0;
    LONG last_generation = 0, frames_during_host_pause = 0;
    int camera_x = 0, camera_y = 0;
    int input_x=0,input_y=0;
    bool client_paused = false, action_sent = false;
    double camera_frame_ms = 0;
    LONG hidden_frames=0,returned_visible_frames=0;
    unsigned visible_min=UINT_MAX,visible_max=0,shadow_builds=0,resident_builds=0;
    std::vector<double> frame_times, steady_times, draw_times, present_times;
    std::array<std::vector<double>,4> present_stages;
    std::array<std::vector<double>,5> phase_times;
    std::array<std::vector<double>,6> scene_stages;
    while (GetTickCount64() - start < 18000) {
        MSG message{};
        while (PeekMessageA(&message, nullptr, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) break;
            TranslateMessage(&message); DispatchMessageA(&message);
        }
        if (!IsWindow(window)) break;
        auto elapsed = GetTickCount64() - start;
        if (elapsed >= 12000 && !client_paused) {
            client_paused = true;
            host_before_pause = acquire(&exchange->host_heartbeat);
            std::printf("CLIENT_PAUSE_BEGIN frames=%ld host_generations=%ld\n",
                acquire(&exchange->client_heartbeat), host_before_pause);
            Sleep(2000);
            host_after_pause = acquire(&exchange->host_heartbeat);
            std::printf("CLIENT_PAUSE_END frames=%ld host_generations=%ld\n",
                acquire(&exchange->client_heartbeat), host_after_pause);
        }
        Snapshot snapshot{};
        if (observe(exchange, snapshot)) last_generation = snapshot.generation;
        int next_x=0,next_y=0;
        if (elapsed>=2000 && elapsed<4000) {
            next_x=int((elapsed-2000)*128/2000);
            next_y=int((elapsed-2000)*64/2000);
        } else if (elapsed>=4000 && elapsed<6200) {
            next_x=128;next_y=64;
        } else if (elapsed>=6200 && elapsed<8200) {
            next_x=128+int((elapsed-6200)*128/2000);
            next_y=64+int((elapsed-6200)*64/2000);
        } else if (elapsed>=8200 && elapsed<10000) {
            next_x=-640;next_y=-256;
        } else if (elapsed>=15000 && elapsed<16500) {
            next_x=6400;
        }
        if (GetAsyncKeyState(VK_LEFT) & 0x8000) input_x+=4;
        if (GetAsyncKeyState(VK_RIGHT) & 0x8000) input_x-=4;
        if (GetAsyncKeyState(VK_UP) & 0x8000) input_y+=4;
        if (GetAsyncKeyState(VK_DOWN) & 0x8000) input_y-=4;
        next_x+=input_x;next_y+=input_y;
        if (next_x!=camera_x || next_y!=camera_y) {
            camera_x=next_x;camera_y=next_y;
            InterlockedIncrement(&exchange->camera_sequence);
            exchange->camera_x = camera_x; exchange->camera_y = camera_y;
            InterlockedIncrement(&exchange->camera_sequence);
        }
        if (elapsed >= 6100 && !action_sent && snapshot.generation) {
            Action action{1, snapshot.generation, snapshot.viewer,
                snapshot.unit_incarnation, snapshot.unit_x + 1, snapshot.unit_y + 1};
            action_sent = send_action(exchange, action);
        }
        auto frame = prepared_frame;
        frame.presentation_frequency = 1000;
        frame.presentation_time_ticks = c3x_renderer_i64(GetTickCount64() - start);
        auto begin = GetTickCount64();
        int result = flat_present ? 0 : draw(&frame, nullptr, camera_x, camera_y,
            snapshot.unit_x,snapshot.unit_y,snapshot.unit_incarnation,
            snapshot.viewer,snapshot.unit_visible);
        auto drawn_at = GetTickCount64();
        if (!result) result = present(window, &frame, snapshot.unit_x, snapshot.unit_y,
            snapshot.unit_incarnation,snapshot.viewer,snapshot.unit_visible,camera_x,camera_y);
        if (result) {
            std::printf("CLIENT_FRAME_ERROR code=%d frame=%zu\n", result, frame_times.size());
            break;
        }
        if (present_metrics) {
            double stages[4]={};present_metrics(stages);
            for (unsigned i=0;i<4;++i) present_stages[i].push_back(stages[i]);
        }
        if (metrics) {
            double phases[6]={};unsigned visible=0;std::size_t gpu_bytes=0;float box[4]={};
            metrics(phases,&visible,&shadow_builds,&resident_builds,&gpu_bytes,box);
            visible_min=std::min(visible_min,visible);visible_max=std::max(visible_max,visible);
            if(frame_times.size()>=3)
                for(unsigned i=0;i<6;++i)scene_stages[i].push_back(phases[i]);
            if (frame_times.size()<3) {
            std::printf("CLIENT_FRESH_PHASE frame=%zu visible=%u shadows=%u target_mib=%.1f shadow_box=%.1f,%.1f,%.1f,%.1f scene_ms=%.1f reflection_ms=%.1f static_ms=%.1f water_ms=%.1f direct_units_ms=%.1f post_ms=%.1f\n",
                frame_times.size(),visible,shadow_builds,double(gpu_bytes)/(1024*1024),box[0],box[1],box[2],box[3],
                phases[0],phases[1],phases[2],phases[3],phases[4],phases[5]);
            }
        }
        double duration = double(GetTickCount64() - begin);
        if(frame_times.size()>=3)steady_times.push_back(duration);
        frame_times.push_back(duration);
        unsigned phase=elapsed<2000?0:elapsed<8200?1:elapsed<10000?2:
            elapsed<15000?3:4;
        phase_times[phase].push_back(duration);
        draw_times.push_back(double(drawn_at-begin));
        present_times.push_back(duration-draw_times.back());
        if (phase==2 && camera_frame_ms==0) camera_frame_ms=duration;
        if (!snapshot.unit_visible) ++hidden_frames;
        if (elapsed>=15000 && snapshot.unit_visible) ++returned_visible_frames;
        InterlockedIncrement(&exchange->client_heartbeat);
        if (elapsed >= 4000 && elapsed < 6000) ++frames_during_host_pause;
    }
    std::sort(frame_times.begin(), frame_times.end());
    std::sort(steady_times.begin(), steady_times.end());
    std::sort(draw_times.begin(), draw_times.end());
    std::sort(present_times.begin(), present_times.end());
    for (auto& samples:present_stages) std::sort(samples.begin(),samples.end());
    for (auto& samples:scene_stages) std::sort(samples.begin(),samples.end());
    auto quantile = [&](std::vector<double> const& samples, double fraction) {
        return samples.empty() ? 0.0 : samples[std::min(samples.size()-1,
            std::size_t(fraction * double(samples.size()-1)))];
    };
    char const* phase_names[]={"idle","scroll_and_move","jump",
        "resident_return","wrap_and_return"};
    for (unsigned i=0;i<phase_times.size();++i) {
        auto& samples=phase_times[i];std::sort(samples.begin(),samples.end());
        std::printf("CLIENT_PHASE name=%s frames=%zu median_ms=%.1f p95_ms=%.1f worst_ms=%.1f\n",
            phase_names[i],samples.size(),quantile(samples,.5),
            quantile(samples,.95),quantile(samples,1));
    }
    char const* stage_names[]={"backbuffer_composition","units","capture","Present"};
    for (unsigned i=0;i<present_stages.size();++i)
        std::printf("CLIENT_STAGE name=%s median_ms=%.2f p95_ms=%.2f worst_ms=%.2f\n",
            stage_names[i],quantile(present_stages[i],.5),
            quantile(present_stages[i],.95),quantile(present_stages[i],1));
    char const* scene_names[]={"selection_and_shadows","reflection","static_cache",
        "water_and_waves","animated_units","hdr_and_bloom"};
    for(unsigned i=0;i<scene_stages.size();++i)
        std::printf("CLIENT_SCENE_STAGE name=%s median_ms=%.2f p95_ms=%.2f\n",
            scene_names[i],quantile(scene_stages[i],.5),quantile(scene_stages[i],.95));
    std::printf("CLIENT_STEADY frames=%zu median_ms=%.1f p95_ms=%.1f\n",
        steady_times.size(),quantile(steady_times,.5),quantile(steady_times,.95));
    std::printf("CLIENT_SCENE resident_builds=%u shadow_builds=%u visible_min=%u visible_max=%u\n",
        resident_builds,shadow_builds,visible_min==UINT_MAX?0:visible_min,visible_max);
    if(cache_metrics){
        unsigned depth_copies=0,scrolls=0,full_draws=0,reflect_reuses=0,reflect_draws=0,pose_builds=0;
        cache_metrics(&depth_copies,&scrolls,&full_draws,&reflect_reuses,&reflect_draws,&pose_builds);
        std::printf("CLIENT_STATIC_CACHE depth_copies=%u scrolls=%u full_draws=%u reflection_reuses=%u reflection_draws=%u pose_builds=%u\n",
            depth_copies,scrolls,full_draws,reflect_reuses,reflect_draws,pose_builds);
    }
    WaitForSingleObject(host.hProcess, 3000);
    std::printf("CLIENT_RESULT successful_present_calls=%zu swapchain_frame_stat=%ld host_generation=%ld host_advanced_during_client_pause=%ld frames_during_host_pause=%ld action_ack=%ld action_result=%ld hidden_frames=%ld visible_after_identity_change=%ld camera=%d,%d median_ms=%.1f p95_ms=%.1f worst_ms=%.1f draw_median_ms=%.1f present_median_ms=%.1f jump_frame_ms=%.1f host_publish_p95_us=%ld host_publish_worst_us=%ld\n",
        frame_times.size(), observed(), last_generation, host_after_pause-host_before_pause,
        frames_during_host_pause, acquire(&exchange->action_ack),
        acquire(&exchange->action_result), hidden_frames,returned_visible_frames,camera_x,camera_y,
        quantile(frame_times,.5), quantile(frame_times,.95), quantile(frame_times,1),
        quantile(draw_times,.5), quantile(present_times,.5), camera_frame_ms,
        acquire(&exchange->host_publication_p95_us), acquire(&exchange->host_publication_worst_us));
    CloseHandle(host.hThread); CloseHandle(host.hProcess);
    UnmapViewOfFile(exchange); CloseHandle(mapping);
    DestroyWindow(window);
    return frame_times.empty() ? 9 : 0;
}

int main(int argc, char** argv) {
    return sandbox_reference_main(argc, argv);
}
