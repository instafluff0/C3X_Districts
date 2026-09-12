#pragma once

// Diagnostics are enabled by default, bounded, and serialized by the existing worker call
// gate and a short trace mutex. Normal gameplay uses OutputDebugStringA only.
// File output requires an explicit standalone-verification override.
// No tile loop performs I/O. QPC timestamps match injected capture logs.
#include <cstdio>

struct RendererTrace {
    std::mutex write_mutex;
    int level = 2;
    FILE * file = nullptr;
    std::size_t bytes = 0;
    bool buffered = false;
    std::string pending;
    std::size_t dropped = 0;
    std::size_t file_limit = 8u * 1024u * 1024u;
    std::atomic<std::uint64_t> sequence{0};
    std::atomic<std::uint64_t> usage_sequence{0};
    LARGE_INTEGER frequency = {};
    LARGE_INTEGER last_summary = {};

    RendererTrace() {
        char value[16] = {};
        if (GetEnvironmentVariableA("C3X_RENDERER_TRACE", value, sizeof(value)))
            level = std::clamp(std::atoi(value), 0, 2);
        if (GetEnvironmentVariableA("C3X_RENDERER_REGION_DIAGNOSTICS", value, sizeof(value)) && std::strcmp(value,"1")==0)
            file_limit = 32u * 1024u * 1024u;
        buffered=GetEnvironmentVariableA("C3X_RENDERER_TRACE_BUFFERED",value,sizeof(value)) && std::strcmp(value,"1")==0;
        if(buffered)pending.reserve(file_limit);
        QueryPerformanceFrequency(&frequency);
        char path[MAX_PATH] = {};
        DWORD length = GetEnvironmentVariableA("C3X_RENDERER_TRACE_FILE", path, sizeof(path));
        if (level && length > 0 && length < sizeof(path))
            fopen_s(&file, path, "wb");
        if(level) {
            FILETIME utc={};GetSystemTimeAsFileTime(&utc);
            std::uint64_t ticks=(std::uint64_t(utc.dwHighDateTime)<<32)|utc.dwLowDateTime;
            char detail[192];
            std::snprintf(detail,sizeof(detail),"schema=1 qpc_frequency=%lld utc_unix_ms=%llu endpoint=renderer_and_native_map_completion",
                static_cast<long long>(frequency.QuadPart),
                static_cast<unsigned long long>(ticks/10000u-11644473600000ull));
            write("usage-session",detail,true);
        }
    }

    ~RendererTrace() {
        if(file) {
            if(buffered){std::fwrite(pending.data(),1,pending.size(),file);
                std::fprintf(file,"TRACE_BUFFER dropped=%zu\n",dropped);}
            std::fclose(file);
        }
    }

    double milliseconds(c3x_renderer_i64 ticks) const {
        return frequency.QuadPart > 0 ? 1000.0 * ticks / frequency.QuadPart : 0.0;
    }

    void write(char const * stage, char const * detail, bool important = false) {
        if (!level) return;
        std::lock_guard<std::mutex> guard(write_mutex);
        LARGE_INTEGER now = {};
        QueryPerformanceCounter(&now);
        if (level < 2 && !important &&
            now.QuadPart - last_summary.QuadPart < frequency.QuadPart) return;
        last_summary = now;
        char line[1024];
        int count = std::snprintf(line, sizeof(line),
            "[C3X renderer] qpc=%lld ms=%.3f process=%lu thread=%lu sequence=%llu stage=%s %s\n",
            static_cast<long long>(now.QuadPart), milliseconds(now.QuadPart),
            GetCurrentProcessId(),GetCurrentThreadId(), static_cast<unsigned long long>(sequence.load(std::memory_order_relaxed)), stage, detail);
        if (count <= 0) return;
        std::size_t size = std::min(static_cast<std::size_t>(count), sizeof(line) - 1u);
        if(buffered) {
            if(pending.size()+size<=file_limit)pending.append(line,size);
            else ++dropped;
            return;
        }
        OutputDebugStringA(line);
        // Stop at 8 MiB (32 MiB for explicit region diagnostics); debugger output remains available. No
        // unbounded file growth, per-line flush, path disclosure, or rotation I/O.
        if (file && bytes + size <= file_limit) {
            bytes += std::fwrite(line, 1, size, file);
        }
    }

    std::uint64_t usage_view(c3x_renderer_frame_v1 const& frame) {
        if(!level)return 0;
        auto request=++usage_sequence;
        unsigned visible=0,cities=0,roads=0,rails=0,farms=0,mines=0,camps=0,resources=0,tile_units=0;
        c3x_renderer_tile_v1 const* reference=nullptr;
        for(unsigned i=0;i<frame.tile_count;++i) {
            auto const& tile=frame.tiles[i];if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            if(!reference)reference=&tile;
            ++visible;cities+=tile.city_id>=0;roads+=tile.road_mask!=0;rails+=tile.railroad_mask!=0;
            farms+=(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)!=0;
            mines+=(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)!=0;
            camps+=(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)!=0;
            resources+=tile.resource_id>=0;tile_units+=tile.unit_type_id>=0;
        }
        long long origin_x=reference?reference->anchor_x-static_cast<long long>(reference->tile_x)*frame.tile_width/2:0;
        long long origin_y=reference?reference->anchor_y-static_cast<long long>(reference->tile_y)*frame.tile_height/2:0;
        char detail[768];
        std::snprintf(detail,sizeof(detail),
            "request=%llu origin_valid=%u origin_x=%lld origin_y=%lld tile_width=%d tile_height=%d target_width=%d target_height=%d "
            "world_width=%d world_height=%d wrap_x=%d wrap_y=%d world_revision=%lld clock=%lld clock_frequency=%lld "
            "hour=%d season=%d captured=%u visible=%u cities=%u roads=%u railroads=%u farms=%u mines=%u camps=%u resources=%u tile_units=%u",
            static_cast<unsigned long long>(request),unsigned(reference!=nullptr),origin_x,origin_y,
            frame.tile_width,frame.tile_height,frame.target_width,frame.target_height,frame.world_width_tiles,frame.world_height_tiles,
            frame.world_wrap_x,frame.world_wrap_y,static_cast<long long>(frame.world_topology_revision),
            static_cast<long long>(frame.presentation_time_ticks),static_cast<long long>(frame.presentation_frequency),frame.hour,frame.season,
            frame.tile_count,visible,cities,roads,rails,farms,mines,camps,resources,tile_units);
        write("usage-view",detail,true);
        return request;
    }

    void usage_result(std::uint64_t request,int result,c3x_renderer_i64 elapsed,c3x_renderer_output_v1 const& output) {
        if(!request)return;
        bool ok=result==C3X_RENDERER_RESULT_OK;
        char detail[384];
        std::snprintf(detail,sizeof(detail),
            "request=%llu result=%d call_ms=%.3f built=%u reused=%u upload_bytes=%u geometry_ms=%.3f draw_submit_ms=%.3f "
            "readback_wait_ms=%.3f visible_animation=%u recoveries=%u fallback=%u endpoint=dll_return",
            static_cast<unsigned long long>(request),result,milliseconds(elapsed),
            ok?output.geometry_tiles_built:0,ok?output.geometry_tiles_reused:0,ok?output.geometry_upload_bytes:0,
            ok?milliseconds(output.geometry_ticks):0,ok?milliseconds(output.draw_ticks):0,ok?milliseconds(output.readback_ticks):0,
            ok?output.visible_animation_count:0,ok?output.device_recoveries:0,ok?output.fallback_tile_count:0);
        write("usage-result",detail,true);
    }
};
