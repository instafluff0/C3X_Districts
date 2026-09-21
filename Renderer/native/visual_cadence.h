#pragma once
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace c3x_renderer {
// One opportunity at a time, with no catch-up queue. The callback owns the
// renderer's existing transaction gate; this scheduler never reads game state.
class VisualCadence {
    std::mutex mutex;
    std::condition_variable wake;
    std::thread thread;
    bool enabled=false,stopping=false;
public:
    ~VisualCadence(){stop();}
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
                    auto next=std::max(begin+std::chrono::milliseconds(33),Clock::now()+std::chrono::milliseconds(10));
                    wake.wait_until(guard,next,[this]{return stopping||!enabled;});
                }
            });
        }
        enabled=true;wake.notify_one();
    }
    // May be called under the renderer gate. A callback already awake must
    // recheck delivery ownership after acquiring that gate.
    void disable(){std::lock_guard<std::mutex> lock(mutex);enabled=false;wake.notify_one();}
    // Join outside the renderer gate: a callback may still be leaving it.
    void stop(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;enabled=false;wake.notify_one();}
        if(thread.joinable())thread.join();
    }
};
}
