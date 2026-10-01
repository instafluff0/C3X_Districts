#pragma once
#include <cstdint>

namespace c3x_renderer { namespace sandbox {
// Private HWND presentation witness. Status is retained even when statistics
// are unavailable; zero counters never stand in for successful observations.
struct ZoomPresentationProbe {
    int waitable=0, present_status=-1, statistics_status=-1, counter_status=-1;
    unsigned last_present=0, present_count=0, present_refresh=0, sync_refresh=0;
    long long sync_qpc=0, sync_gpu_qpc=0, observation_qpc=0;
    std::uint64_t admissions=0, denials=0, waits=0, message_wakes=0, timeouts=0;
};
}}
