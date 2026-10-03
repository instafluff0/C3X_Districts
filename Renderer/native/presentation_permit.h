#pragma once
#include <windows.h>
#include <atomic>
#include <mutex>
#include <stdexcept>

namespace c3x_renderer {
// The DXGI frame-latency signal grants one presentation. Keep that grant
// across a static no-op frame; querying an auto-reset signal again would lose
// it. ready() polls with zero timeout so a busy compositor cannot hold the
// renderer command queue. wait() lets the cadence thread block on the same
// signal *outside* the renderer gate, so frames start at the compositor's
// vsync-aligned opportunity instead of a free-running timer.
class PresentationPermit {
    HANDLE signal=nullptr;
    // Acquired but unused grants. Both the worker (ready) and the cadence
    // thread (wait) may acquire; counting them means no grant can leak.
    std::atomic<int> grants{0};
    std::mutex handle_mutex;
public:
    ~PresentationPermit(){reset();}
    void reset(HANDLE next=nullptr){
        std::lock_guard<std::mutex> lock(handle_mutex);
        if(signal)CloseHandle(signal);
        signal=next;grants=0;
    }
    bool ready(){
        if(grants.load(std::memory_order_acquire)>0)return true;
        HANDLE current=nullptr;
        {std::lock_guard<std::mutex> lock(handle_mutex);current=signal;}
        if(!current)return false;
        DWORD result=WaitForSingleObject(current,0);
        if(result==WAIT_FAILED)throw std::runtime_error("presentation signal failed");
        if(result==WAIT_OBJECT_0)grants.fetch_add(1,std::memory_order_acq_rel);
        return result==WAIT_OBJECT_0;
    }
    // 1: a new presentation opportunity was granted (vsync aligned).
    // 2: an earlier grant is still unused (nothing was presented).
    // 0: no signal, timeout or failure; the caller falls back to its timer.
    int wait(DWORD timeout){
        if(grants.load(std::memory_order_acquire)>0)return 2;
        HANDLE duplicate=nullptr;
        {
            std::lock_guard<std::mutex> lock(handle_mutex);
            if(!signal || !DuplicateHandle(GetCurrentProcess(),signal,GetCurrentProcess(),&duplicate,
                    SYNCHRONIZE,FALSE,0))return 0;
        }
        DWORD result=WaitForSingleObject(duplicate,timeout);
        CloseHandle(duplicate);
        if(result!=WAIT_OBJECT_0)return 0;
        grants.fetch_add(1,std::memory_order_acq_rel);
        return 1;
    }
    void presented(){
        int current=grants.load(std::memory_order_acquire);
        while(current>0 && !grants.compare_exchange_weak(current,current-1,std::memory_order_acq_rel)){}
    }
};
}
