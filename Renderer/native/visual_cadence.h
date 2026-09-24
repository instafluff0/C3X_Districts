#pragma once
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#ifdef _WIN32
#include <windows.h>
#endif

namespace c3x_renderer {
// One opportunity at a time, with no catch-up queue. The callback owns the
// renderer's existing transaction gate; this scheduler never reads game state.
class VisualCadence {
    std::chrono::steady_clock::duration period;
    std::chrono::steady_clock::duration minimum_pause;
    std::mutex mutex;
    std::condition_variable wake;
    std::thread thread;
    bool enabled=false,stopping=false;
#ifdef _WIN32
    HANDLE timer=nullptr,interrupt=nullptr;
#endif
public:
    explicit VisualCadence(
        std::chrono::steady_clock::duration target=std::chrono::milliseconds(33),
        std::chrono::steady_clock::duration pause=std::chrono::milliseconds(10))
        :period(target),minimum_pause(pause){
#ifdef _WIN32
        timer=CreateWaitableTimerExW(nullptr,nullptr,CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
            TIMER_MODIFY_STATE|SYNCHRONIZE);
        if(!timer)timer=CreateWaitableTimerExW(nullptr,nullptr,0,TIMER_MODIFY_STATE|SYNCHRONIZE);
        interrupt=CreateEventW(nullptr,FALSE,FALSE,nullptr);
#endif
    }
    ~VisualCadence(){stop();
#ifdef _WIN32
        if(timer)CloseHandle(timer);
        if(interrupt)CloseHandle(interrupt);
#endif
    }
    void enable(std::function<void()> callback){
        std::lock_guard<std::mutex> lock(mutex);
        if(!thread.joinable()){
            stopping=false;
            thread=std::thread([this,callback]{
                using Clock=std::chrono::steady_clock;
                std::unique_lock<std::mutex> guard(mutex);
                while(!stopping){
                    wake.wait(guard,[this]{return stopping||enabled;});
                    if(stopping)break;
                    auto begin=Clock::now();
                    guard.unlock();callback();guard.lock();
                    auto now=Clock::now();
                    auto next=begin+period>now+minimum_pause?begin+period:now+minimum_pause;
#ifdef _WIN32
                    if(timer&&interrupt&&enabled){
                        auto nanoseconds=std::chrono::duration_cast<std::chrono::nanoseconds>(next-now).count();
                        auto hundred_ns=(nanoseconds+99)/100;
                        LARGE_INTEGER due={};due.QuadPart=-(hundred_ns>0?hundred_ns:1);
                        ResetEvent(interrupt);
                        if(SetWaitableTimerEx(timer,&due,0,nullptr,nullptr,nullptr,0)){
                            HANDLE handles[2]={timer,interrupt};
                            guard.unlock();auto result=WaitForMultipleObjects(2,handles,FALSE,INFINITE);guard.lock();
                            if(result!=WAIT_OBJECT_0&&result!=WAIT_OBJECT_0+1){CloseHandle(timer);timer=nullptr;}
                            continue;
                        }
                    }
#endif
                    wake.wait_until(guard,next,[this]{return stopping||!enabled;});
                }
            });
        }
        enabled=true;wake.notify_one();
#ifdef _WIN32
        if(interrupt)SetEvent(interrupt);
#endif
    }
    // May be called under the renderer gate. A callback already awake must
    // recheck delivery ownership after acquiring that gate.
    void disable(){std::lock_guard<std::mutex> lock(mutex);enabled=false;wake.notify_one();
#ifdef _WIN32
        if(interrupt)SetEvent(interrupt);
#endif
    }
    // Join outside the renderer gate: a callback may still be leaving it.
    void stop(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;enabled=false;wake.notify_one();}
#ifdef _WIN32
        if(interrupt)SetEvent(interrupt);
#endif
        if(thread.joinable())thread.join();
    }
};
}
