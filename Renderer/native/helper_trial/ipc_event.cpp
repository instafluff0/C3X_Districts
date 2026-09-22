#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
struct Wire {
    volatile LONG sequence;
    volatile LONG acknowledged;
    unsigned char reserved[56];
};
static_assert(sizeof(Wire) == 64, "fixed cross-bitness record");
double now_ms() {
    LARGE_INTEGER time = {}, frequency = {};
    QueryPerformanceCounter(&time);
    QueryPerformanceFrequency(&frequency);
    return 1000.0 * double(time.QuadPart) / double(frequency.QuadPart);
}
std::string name(std::string const& base, char const* part) {
    return base + "_" + part;
}
}

#if defined(_WIN64)
int main(int argc, char** argv) {
    if (argc != 4 || std::string(argv[1]) != "--child") return 2;
    std::string base(argv[2]);
    unsigned samples = unsigned(std::strtoul(argv[3], nullptr, 10));
    HANDLE mapping = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, name(base, "map").c_str());
    HANDLE request = OpenEventA(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, name(base, "request").c_str());
    HANDLE response = OpenEventA(EVENT_MODIFY_STATE | SYNCHRONIZE, FALSE, name(base, "response").c_str());
    if (!mapping || !request || !response) return 3;
    auto* wire = static_cast<Wire*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Wire)));
    if (!wire || !SetEvent(response)) return 4;
    int result = 0;
    for (unsigned i = 1; i <= samples; ++i) {
        if (WaitForSingleObject(request, 5000) != WAIT_OBJECT_0 ||
            InterlockedCompareExchange(&wire->sequence, 0, 0) != LONG(i)) {
            result = 5; break;
        }
        InterlockedExchange(&wire->acknowledged, LONG(i));
        if (!SetEvent(response)) { result = 6; break; }
    }
    UnmapViewOfFile(wire);
    CloseHandle(response); CloseHandle(request); CloseHandle(mapping);
    return result;
}
#else
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    unsigned samples = unsigned(std::strtoul(argv[2], nullptr, 10));
    if (samples < 100 || samples > 10000) return 2;
    std::string base = "Local\\C3XHelperTrialEvent_" + std::to_string(GetCurrentProcessId());
    HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0,
                                        sizeof(Wire), name(base, "map").c_str());
    HANDLE request = CreateEventA(nullptr, FALSE, FALSE, name(base, "request").c_str());
    HANDLE response = CreateEventA(nullptr, FALSE, FALSE, name(base, "response").c_str());
    if (!mapping || !request || !response) return 3;
    auto* wire = static_cast<Wire*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(Wire)));
    if (!wire) return 4;
    std::string executable(argv[0]);
    std::size_t slash = executable.find_last_of("\\/");
    if (slash == std::string::npos) return 4;
    executable = executable.substr(0, slash + 1) + "ipc_x64.exe";
    std::string command = "\"" + executable + "\" --child " + base + " " + std::to_string(samples);
    std::vector<char> command_line(command.begin(), command.end()); command_line.push_back('\0');
    STARTUPINFOA startup = {}; startup.cb = sizeof(startup);
    PROCESS_INFORMATION child = {};
    if (!CreateProcessA(executable.c_str(), command_line.data(), nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &child)) return 5;
    CloseHandle(child.hThread);
    int result = 0;
    std::vector<double> intervals;
    if (WaitForSingleObject(response, 10000) != WAIT_OBJECT_0) result = 6;
    for (unsigned i = 1; result == 0 && i <= samples; ++i) {
        double begin = now_ms();
        InterlockedExchange(&wire->sequence, LONG(i));
        if (!SetEvent(request) || WaitForSingleObject(response, 1000) != WAIT_OBJECT_0 ||
            InterlockedCompareExchange(&wire->acknowledged, 0, 0) != LONG(i)) {
            result = 7; break;
        }
        intervals.push_back(now_ms() - begin);
    }
    if (result != 0) TerminateProcess(child.hProcess, unsigned(result));
    else if (WaitForSingleObject(child.hProcess, 5000) != WAIT_OBJECT_0) {
        TerminateProcess(child.hProcess, 8);
        result = 8;
    }
    DWORD child_exit = 0;
    GetExitCodeProcess(child.hProcess, &child_exit);
    if (result == 0 && child_exit != 0) result = 9;
    CloseHandle(child.hProcess);
    UnmapViewOfFile(wire);
    CloseHandle(response); CloseHandle(request); CloseHandle(mapping);
    std::sort(intervals.begin(), intervals.end());
    double total = 0;
    for (double value : intervals) total += value;
    double mean = intervals.empty() ? 0 : total / intervals.size();
    auto percentile = [&](double rank) {
        std::size_t count = std::max<std::size_t>(1, std::size_t(std::ceil(rank * intervals.size())));
        return intervals.empty() ? 0 : intervals[std::min(intervals.size(), count) - 1];
    };
    std::ofstream report(argv[1], std::ios::binary);
    if (!report) return 10;
    report << "{\"status\":\"" << (result ? "fail" : "pass")
           << "\",\"bytes_per_message\":64,\"samples\":" << intervals.size()
           << ",\"roundtrip_ms_mean\":" << mean
           << ",\"roundtrip_ms_p95\":" << percentile(.95)
           << ",\"roundtrip_ms_p99\":" << percentile(.99)
           << ",\"roundtrip_ms_max\":" << (intervals.empty() ? 0 : intervals.back())
           << ",\"error\":" << result << "}\n";
    std::printf("%s shared-memory/event x86-x64 roundtrip samples=%zu mean=%.3f p95=%.3f ms\n",
                result ? "FAIL" : "PASS", intervals.size(), mean, percentile(.95));
    return result;
}
#endif
