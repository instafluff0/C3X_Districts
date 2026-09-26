#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <cstdlib>
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

void sandbox_camera_path(ULONGLONG elapsed,int& x,int& y){
    x=y=0;
    if(elapsed>=23500 && elapsed<27500){
        x=int((elapsed-23500)*256/4000);
        y=int((elapsed-23500)*128/4000);
    }else if(elapsed>=27500 && elapsed<29000){x=-640;y=-256;}
    else if(elapsed>=33500 && elapsed<35000)x=6400;
}

float sandbox_zoom_path(ULONGLONG elapsed){
    if(elapsed>=29500 && elapsed<31000)
        return 1.f+.2f*float(elapsed-29500)/1500.f;
    if(elapsed>=31000 && elapsed<32500)
        return 1.2f-.2f*float(elapsed-31000)/1500.f;
    return 1.f;
}

int sandbox_client_run(HMODULE module, c3x_renderer_frame_v1 const& prepared_frame) {
    using namespace sandbox_exchange;
    auto draw = reinterpret_cast<int(*)(c3x_renderer_frame_v1 const*, char const*,
        int,int,int,int,int,int,int,float)>(
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
    auto prewarm = reinterpret_cast<int(*)(int,int)>(
        GetProcAddress(module,"c3x_sandbox_prewarm_units"));
    auto combat = reinterpret_cast<void(*)(int,c3x_renderer_i64)>(
        GetProcAddress(module,"c3x_sandbox_combat_event"));
    if(!prewarm||!combat)return 2;
    auto prewarm_begin=GetTickCount64();
    int prewarm_result=prewarm(prepared_frame.hour,prepared_frame.season);
    std::printf("CLIENT_PREWARM units_ms=%llu result=%d\n",
        static_cast<unsigned long long>(GetTickCount64()-prewarm_begin),prewarm_result);
    if(prewarm_result)return 2;
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
    Snapshot opening{0,1,1,7,19,47,1,0};
    char move_start_tile[32]{};
    int diagnostic_x=0,diagnostic_y=0;
    if(GetEnvironmentVariableA("C3X_SANDBOX_MOVE_START_TILE",move_start_tile,
            sizeof(move_start_tile)) &&
            sscanf_s(move_start_tile,"%d,%d",&diagnostic_x,&diagnostic_y)==2 &&
            diagnostic_x>=0 && diagnostic_y>=0){
        opening.unit_x=diagnostic_x;
        opening.unit_y=diagnostic_y;
    }
    auto prime_frame=prepared_frame;
    prime_frame.presentation_frequency=1000;
    prime_frame.presentation_time_ticks=0;
    auto prime_begin=GetTickCount64();
    int prime_result=draw(&prime_frame,nullptr,0,0,opening.unit_x,opening.unit_y,
        opening.unit_incarnation,opening.viewer,opening.unit_visible,1.f);
    if(!prime_result)prime_result=present(window,&prime_frame,opening.unit_x,opening.unit_y,
        opening.unit_incarnation,opening.viewer,opening.unit_visible,0,0);
    std::printf("CLIENT_PRIME scene_and_swapchain_ms=%llu result=%d\n",
        static_cast<unsigned long long>(GetTickCount64()-prime_begin),prime_result);
    if(prime_result)return 4;
    char clip_option[8]{};
    if(GetEnvironmentVariableA("C3X_SANDBOX_REPLAY_CLIP",clip_option,sizeof(clip_option)) &&
            std::strcmp(clip_option,"1")==0){
        char day_night_option[8]{};
        bool day_night=GetEnvironmentVariableA("C3X_SANDBOX_DAY_NIGHT",day_night_option,
            sizeof(day_night_option)) && std::strcmp(day_night_option,"1")==0;
        std::vector<double> cycle_frames,cycle_draws,cycle_presents;
        std::array<std::vector<double>,6> cycle_stages;
        LARGE_INTEGER cycle_frequency{};
        QueryPerformanceFrequency(&cycle_frequency);
        char frame_limit_text[16]{};
        int frame_limit=day_night?900:1080;
        if(GetEnvironmentVariableA("C3X_SANDBOX_CLIP_FRAMES",frame_limit_text,
                sizeof(frame_limit_text)))
            frame_limit=std::clamp(std::atoi(frame_limit_text),1,frame_limit);
        char move_at_text[16]{};
        int move_at=15000;
        if(!day_night && GetEnvironmentVariableA("C3X_SANDBOX_MOVE_AT_MS",
                move_at_text,sizeof(move_at_text)))
            move_at=std::clamp(std::atoi(move_at_text),0,32000);
        // Every source frame is rendered at 30 Hz. The first 15 seconds keep
        // the camera fixed so water and authored ambient clips are inspectable.
        for(int index=0;index<frame_limit;++index){
            int t=int((std::int64_t(index)*1000+15)/30);
            int x=0,y=0;
            if(!day_night)sandbox_camera_path(t,x,y);
            Snapshot actor=opening;
            if(!day_night){
                if(t>=move_at){actor.unit_x=opening.unit_x+1;actor.unit_y=opening.unit_y+1;}
                if(t>=17500)combat(1,t);
            }
            auto frame=prepared_frame;
            frame.presentation_frequency=1000;
            frame.presentation_time_ticks=t;
            LARGE_INTEGER started{},drawn{},finished{};
            QueryPerformanceCounter(&started);
            int result=flat_present?0:draw(&frame,nullptr,x,y,actor.unit_x,actor.unit_y,
                actor.unit_incarnation,actor.viewer,actor.unit_visible,
                day_night?1.f:sandbox_zoom_path(t));
            QueryPerformanceCounter(&drawn);
            if(!result)result=present(window,&frame,actor.unit_x,actor.unit_y,
                actor.unit_incarnation,actor.viewer,actor.unit_visible,x,y);
            QueryPerformanceCounter(&finished);
            if(result)return result;
            if(day_night && index>=3){
                double divisor=double(cycle_frequency.QuadPart)/1000.;
                cycle_frames.push_back(double(finished.QuadPart-started.QuadPart)/divisor);
                cycle_draws.push_back(double(drawn.QuadPart-started.QuadPart)/divisor);
                cycle_presents.push_back(double(finished.QuadPart-drawn.QuadPart)/divisor);
                if(metrics){
                    double phases[6]={};unsigned selected=0,shadow_builds=0,resident_builds=0;
                    std::size_t bytes=0;float box[4]={};
                    metrics(phases,&selected,&shadow_builds,&resident_builds,&bytes,box);
                    for(unsigned stage=0;stage<6;++stage)cycle_stages[stage].push_back(phases[stage]);
                }
            }
            if(index%30==0)std::printf("CLIENT_CLIP_FRAME time_ms=%d hour=%.2f camera=%d,%d zoom=%.3f\n",
                t,day_night?12.f+24.f*float(t)/30000.f:float(frame.hour),x,y,
                day_night?1.f:sandbox_zoom_path(t));
        }
        if(day_night){
            auto report=[](char const* name,std::vector<double>& values){
                std::sort(values.begin(),values.end());
                auto at=[&](double fraction){return values.empty()?0.:values[std::min(
                    values.size()-1,std::size_t(fraction*double(values.size()-1)))];};
                std::printf("CLIENT_CYCLE name=%s frames=%zu median_ms=%.2f p95_ms=%.2f\n",
                    name,values.size(),at(.5),at(.95));
            };
            report("total",cycle_frames);report("draw",cycle_draws);
            report("present",cycle_presents);
            char const* names[]={"selection_and_shadows","reflection","static_cache",
                "water_and_waves","animated_units","hdr_and_bloom"};
            for(unsigned stage=0;stage<6;++stage)report(names[stage],cycle_stages[stage]);
        }
        DestroyWindow(window);
        return 0;
    }

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
    LONG last_generation = 0, frames_during_host_pause = 0, last_combat=0;
    int camera_x = 0, camera_y = 0;
    int input_x=0,input_y=0;
    bool client_paused = false, action_sent = false;
    double camera_frame_ms = 0;
    LONG hidden_frames=0,returned_visible_frames=0;
    unsigned visible_min=UINT_MAX,visible_max=0,shadow_builds=0,resident_builds=0;
    std::vector<double> frame_times, steady_times, draw_times, present_times;
    std::array<std::vector<double>,4> present_stages;
    std::array<std::vector<double>,10> phase_times;
    std::vector<double> zoom_times;
    std::array<std::vector<double>,2> zoom_legs;
    double first_zoom_ms=0,return_zoom_ms=0;
    std::array<std::vector<double>,6> scene_stages;
    bool frame_failed=false;
    while (GetTickCount64() - start < 36000) {
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
        sandbox_camera_path(elapsed,next_x,next_y);
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
        if(snapshot.combat_serial>last_combat){
            last_combat=snapshot.combat_serial;
            combat(last_combat,c3x_renderer_i64(elapsed));
        }
        if (elapsed >= 15000 && !action_sent && snapshot.generation) {
            Action action{1, snapshot.generation, snapshot.viewer,
                snapshot.unit_incarnation, snapshot.unit_x + 1, snapshot.unit_y + 1};
            action_sent = send_action(exchange, action);
        }
        auto frame = prepared_frame;
        frame.presentation_frequency = 1000;
        frame.presentation_time_ticks = c3x_renderer_i64(GetTickCount64() - start);
        float zoom=sandbox_zoom_path(elapsed);
        auto begin = GetTickCount64();
        int result = flat_present ? 0 : draw(&frame, nullptr, camera_x, camera_y,
            snapshot.unit_x,snapshot.unit_y,snapshot.unit_incarnation,
            snapshot.viewer,snapshot.unit_visible,zoom);
        auto drawn_at = GetTickCount64();
        if (!result) result = present(window, &frame, snapshot.unit_x, snapshot.unit_y,
            snapshot.unit_incarnation,snapshot.viewer,snapshot.unit_visible,camera_x,camera_y);
        if (result) {
            std::printf("CLIENT_FRAME_ERROR code=%d frame=%zu\n", result, frame_times.size());
            frame_failed=true;
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
        unsigned phase=elapsed<15000?0:elapsed<16500?1:elapsed<23500?2:
            elapsed<27500?3:elapsed<29000?4:elapsed<29500?5:
            elapsed<32500?6:elapsed<33500?7:elapsed<35000?8:9;
        phase_times[phase].push_back(duration);
        if(elapsed>=29500 && elapsed<32500){
            zoom_times.push_back(duration);
            zoom_legs[elapsed<31000?0:1].push_back(duration);
            if(first_zoom_ms==0)first_zoom_ms=duration;
        } else if(elapsed>=32500 && return_zoom_ms==0 && first_zoom_ms!=0)
            return_zoom_ms=duration;
        draw_times.push_back(double(drawn_at-begin));
        present_times.push_back(duration-draw_times.back());
        if (phase==4 && camera_frame_ms==0) camera_frame_ms=duration;
        if (!snapshot.unit_visible) ++hidden_frames;
        if (elapsed>=35000 && snapshot.unit_visible) ++returned_visible_frames;
        InterlockedIncrement(&exchange->client_heartbeat);
        if (elapsed >= 9000 && elapsed < 11000) ++frames_during_host_pause;
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
    char const* phase_names[]={"idle","moving_unit","combat","scroll","jump",
        "resident_return","zoom","hold","wrap","wrap_return"};
    for (unsigned i=0;i<phase_times.size();++i) {
        auto& samples=phase_times[i];std::sort(samples.begin(),samples.end());
        std::printf("CLIENT_PHASE name=%s frames=%zu median_ms=%.1f p95_ms=%.1f worst_ms=%.1f\n",
            phase_names[i],samples.size(),quantile(samples,.5),
            quantile(samples,.95),quantile(samples,1));
    }
    std::sort(zoom_times.begin(),zoom_times.end());
    std::printf("CLIENT_ZOOM frames=%zu first_ms=%.1f median_ms=%.1f p95_ms=%.1f worst_ms=%.1f return_ms=%.1f\n",
        zoom_times.size(),first_zoom_ms,quantile(zoom_times,.5),
        quantile(zoom_times,.95),quantile(zoom_times,1),return_zoom_ms);
    char const* zoom_leg_names[]={"zoom_in","zoom_out"};
    for(unsigned i=0;i<zoom_legs.size();++i){
        auto& samples=zoom_legs[i];std::sort(samples.begin(),samples.end());
        std::printf("CLIENT_ZOOM_LEG name=%s frames=%zu median_ms=%.1f p95_ms=%.1f worst_ms=%.1f\n",
            zoom_leg_names[i],samples.size(),quantile(samples,.5),
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
    return frame_failed ? 10 : frame_times.empty() ? 9 : 0;
}

int main(int argc, char** argv) {
    return sandbox_reference_main(argc, argv);
}
