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
// needed: grants to hold before a frame starts. Holding every grant of the
// swap chain's frame latency means no earlier frame is still queued: one
// frame of latency without changing DXGI's limit (a runtime change of that
// limit was accepted but did not shorten the queue on Parallels).
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
    bool ready(int needed=1){
        while(grants.load(std::memory_order_acquire)<needed){
            HANDLE current=nullptr;
            {std::lock_guard<std::mutex> lock(handle_mutex);current=signal;}
            if(!current)return false;
            DWORD result=WaitForSingleObject(current,0);
            if(result==WAIT_FAILED)throw std::runtime_error("presentation signal failed");
            if(result!=WAIT_OBJECT_0)return false;
            grants.fetch_add(1,std::memory_order_acq_rel);
        }
        return true;
    }
    // 1: the needed grants were reached with a new one (vsync aligned).
    // 2: earlier grants already suffice (nothing was presented).
    // 0: no signal, timeout or failure; the caller falls back to its timer.
    // Grants acquired before a timeout are kept, never lost.
    int wait(DWORD timeout,int needed=1){
        if(grants.load(std::memory_order_acquire)>=needed)return 2;
        HANDLE duplicate=nullptr;
        {
            std::lock_guard<std::mutex> lock(handle_mutex);
            if(!signal || !DuplicateHandle(GetCurrentProcess(),signal,GetCurrentProcess(),&duplicate,
                    SYNCHRONIZE,FALSE,0))return 0;
        }
        auto began=GetTickCount64();
        while(grants.load(std::memory_order_acquire)<needed){
            auto elapsed=GetTickCount64()-began;
            if(WaitForSingleObject(duplicate,elapsed>=timeout?0:DWORD(timeout-elapsed))!=WAIT_OBJECT_0)break;
            grants.fetch_add(1,std::memory_order_acq_rel);
        }
        CloseHandle(duplicate);
        return grants.load(std::memory_order_acquire)>=needed?1:0;
    }
    void presented(){
        int current=grants.load(std::memory_order_acquire);
        while(current>0 && !grants.compare_exchange_weak(current,current-1,std::memory_order_acq_rel)){}
    }
};
}
