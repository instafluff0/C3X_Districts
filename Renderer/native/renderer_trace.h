#pragma once

// Diagnostics are opt-in, bounded, and serialized by the existing worker call
// gate and a short trace mutex. No tile loop performs I/O. QPC timestamps match injected capture logs.
#include <cstdio>

struct RendererTrace {
    std::mutex write_mutex;
    int level = 0;
    FILE * file = nullptr;
    std::size_t bytes = 0;
    std::uint64_t sequence = 0;
    LARGE_INTEGER frequency = {};
    LARGE_INTEGER last_summary = {};

    RendererTrace() {
        char value[16] = {};
        if (GetEnvironmentVariableA("C3X_RENDERER_TRACE", value, sizeof(value)))
            level = std::clamp(std::atoi(value), 0, 2);
        QueryPerformanceFrequency(&frequency);
        char path[MAX_PATH] = {};
        DWORD length = GetEnvironmentVariableA("C3X_RENDERER_TRACE_FILE", path, sizeof(path));
        if (level && length > 0 && length < sizeof(path))
            fopen_s(&file, path, "wb");
    }

    ~RendererTrace() { if (file) std::fclose(file); }

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
            "[C3X renderer] qpc=%lld ms=%.3f thread=%lu sequence=%llu stage=%s %s\n",
            static_cast<long long>(now.QuadPart), milliseconds(now.QuadPart),
            GetCurrentThreadId(), static_cast<unsigned long long>(sequence), stage, detail);
        if (count <= 0) return;
        std::size_t size = std::min(static_cast<std::size_t>(count), sizeof(line) - 1u);
        OutputDebugStringA(line);
        // Stop file logging at 8 MiB; debugger output remains available. No
        // unbounded file growth, per-line flush, path disclosure, or rotation I/O.
        if (file && bytes + size <= 8u * 1024u * 1024u) {
            bytes += std::fwrite(line, 1, size, file);
        }
    }
};
