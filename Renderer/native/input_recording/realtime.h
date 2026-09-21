#pragma once
#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>
namespace c3x_inputs {
// Diagnostic playback only. Production cadence and presentation still own work.
// One bounded timestamp log survives worker resets; no per-frame GPU readback.
struct RealtimeReplay {
    std::atomic<bool> enabled{false};
    std::mutex mutex;std::vector<double> times;unsigned offered=0,pending=0,errors=0,overflow=0;
    long long origin=0,frequency=1;
    void begin(){std::lock_guard<std::mutex> lock(mutex);if(enabled)return;
        LARGE_INTEGER q={},f={};QueryPerformanceCounter(&q);QueryPerformanceFrequency(&f);
        origin=q.QuadPart;frequency=f.QuadPart;times.reserve(32768);enabled=true;}
    void offer(int result){if(!enabled)return;LARGE_INTEGER q={};QueryPerformanceCounter(&q);
        std::lock_guard<std::mutex> lock(mutex);++offered;
        if(result==C3X_RENDERER_RESULT_OK){if(times.size()<32768)times.push_back(double(q.QuadPart-origin)/double(frequency));else ++overflow;}
        else if(result==C3X_RENDERER_RESULT_PENDING||result==C3X_RENDERER_RESULT_SUPERSEDED)++pending;else ++errors;}
    unsigned read(unsigned first,unsigned capacity,double* output,unsigned* counts){std::lock_guard<std::mutex> lock(mutex);
        counts[0]=offered;counts[1]=unsigned(times.size());counts[2]=pending;counts[3]=errors;counts[4]=overflow;
        if(first>=times.size())return 0;unsigned n=std::min(capacity,unsigned(times.size())-first);
        std::copy_n(times.data()+first,n,output);return n;}
};
inline RealtimeReplay& realtime_replay(){static RealtimeReplay value;return value;}
}
