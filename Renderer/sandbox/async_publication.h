#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <string>

namespace c3x_async {
// Only this transport thread may wait for the other process. Producers copy
// their input before posting; neither the game nor the GPU loop joins a post.
// The mutex protects queue bookkeeping, never transport or renderer execution.
class Publication {
    struct Entry {std::size_t bytes;std::function<void()> work;unsigned replace_key=0;};
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<Entry> entries;
    std::size_t bytes=0,limit,count_limit;
    std::atomic<bool> fault{false};
    std::atomic<unsigned> submitted{0},consumed{0};
    bool stopping=false;
    std::function<void(char const*)> report;
    std::thread thread;
    void run(){
        for(;;){
            Entry next;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock,[&]{return stopping||!entries.empty();});
                if(stopping&&entries.empty())return;
                next=std::move(entries.front());entries.pop_front();
            }
            try{if(healthy()){next.work();consumed.fetch_add(1,std::memory_order_release);}}
            catch(std::exception const& error){fail(error.what());}
            catch(...){fail("unknown asynchronous renderer failure");}
            {
                std::lock_guard<std::mutex> lock(mutex);bytes-=next.bytes;
            }
        }
    }
public:
    explicit Publication(std::function<void(char const*)> diagnostic={},
        std::size_t byte_limit=128u*1024u*1024u,std::size_t packet_limit=8192):
        limit(byte_limit),count_limit(packet_limit),report(std::move(diagnostic)),thread([this]{run();}){}
    ~Publication(){stop();}
    void stop(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;}
        wake.notify_one();if(thread.joinable())thread.join();
    }
    Publication(Publication const&)=delete;
    Publication& operator=(Publication const&)=delete;
    bool healthy()const{return !fault.load(std::memory_order_acquire);}
    unsigned accepted()const{return submitted.load(std::memory_order_acquire);}
    unsigned completed()const{return consumed.load(std::memory_order_acquire);}
    void fail(char const* reason){
        if(!fault.exchange(true,std::memory_order_acq_rel)&&report)report(reason);
    }
    bool post(std::size_t size,std::function<void()> work,unsigned replace_key=0){
        bool accepted=false;
        std::size_t pending_bytes=0,pending_count=0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            // Replaceable observations move to the end, preserving the order
            // of all reliable commands on either side of the new observation.
            if(replace_key)for(auto at=entries.begin();at!=entries.end();){
                if(at->replace_key==replace_key){bytes-=at->bytes;at=entries.erase(at);
                    consumed.fetch_add(1,std::memory_order_release);}
                else ++at;
            }
            if(!stopping&&healthy()&&size<=limit-bytes&&entries.size()<count_limit){
                entries.push_back({size,std::move(work),replace_key});bytes+=size;submitted.fetch_add(1,std::memory_order_release);accepted=true;
            }
            pending_bytes=bytes;pending_count=entries.size();
        }
        if(accepted)wake.notify_one();
        else fail(("renderer publication queue exhausted; bytes="+std::to_string(pending_bytes)+
            " packets="+std::to_string(pending_count)+" incoming="+std::to_string(size)+
            " accepted="+std::to_string(submitted.load())+" completed="+std::to_string(consumed.load())+
            "; complete scene reconciliation required").c_str());
        return accepted;
    }
    // Configuration, teardown and explicit test witnesses may join transport.
    // This is deliberately separate from post and absent from the frame path.
    template<class Function>auto setup(Function work)->decltype(work()){
        using Result=decltype(work());
        auto task=std::make_shared<std::packaged_task<Result()>>(std::move(work));
        auto done=task->get_future();
        if(!post(sizeof(*task),[task]{(*task)();}))throw std::runtime_error("renderer publication unavailable");
        // The queue must be the sole owner before waiting. If an earlier
        // command faults, discarding this task then wakes the waiter with a
        // broken promise instead of leaving reset/configuration stuck forever.
        task.reset();
        return done.get();
    }
};
}
