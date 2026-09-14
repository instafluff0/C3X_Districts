#pragma once
// A bounded CPU producer for existing scene compilers. The caller grants a read
// lease on resident inputs, pauses before changing them, and validates results
// against current local dependencies before publication. No GPU/game ownership.
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <array>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
namespace c3x_renderer { namespace render_core {
template<class Key,class Input,class Result> class ContentPreparation {
public:
    struct Job {Key key;Input input;};
    using Compile=std::function<std::unique_ptr<Result>(Input const&,std::atomic<bool> const&,unsigned)>;
    struct Statistics {
        std::uint64_t built=0,consumed=0,cancelled=0,rejected=0,evicted=0,invalidated=0;
        std::size_t bytes=0,peak_bytes=0,pending=0;
        unsigned active_peak=0;
        double cpu_ms=0,wait_ms=0;
    };
    static constexpr std::size_t byte_limit=16u*1024u*1024u,job_limit=8192;
private:
    struct Ready {Key key;std::unique_ptr<Result> value;};
    std::mutex mutex;
    std::condition_variable wake,completed;
    std::vector<std::thread> workers;
    unsigned worker_limit=1;
    std::size_t capacity_limit=byte_limit;
    std::atomic<bool> cancel{false};
    bool paused=true,stopping=false,demanded=false;
    Key demand_key{};
    std::array<bool,6> active{};
    std::array<Key,6> active_key{};
    Compile compile;
    std::deque<Job> pending;
    std::deque<Ready> ready;
    Statistics stats;
    void run(unsigned worker) {
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            wake.wait(lock,[&]{return stopping || (!paused && worker<worker_limit && !pending.empty() && ((demanded && pending.front().key==demand_key) || (stats.bytes<capacity_limit/2)));});
            if(stopping)return;
            Job job=std::move(pending.front());pending.pop_front();
            active[worker]=true;active_key[worker]=job.key;
            stats.active_peak=std::max(stats.active_peak,unsigned(std::count(active.begin(),active.end(),true)));
            lock.unlock();auto begin=std::chrono::steady_clock::now();
            std::unique_ptr<Result> value;
            try{value=compile(job.input,cancel,worker);}catch(...){}
            auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
            lock.lock();stats.cpu_ms+=elapsed;
            try {
                if(cancel.load(std::memory_order_relaxed)){
                    ++stats.cancelled;
                    if(!stopping)pending.push_front(std::move(job));
                }else if(value && value->bytes()<=capacity_limit){
                    auto bytes=value->bytes();
                    while(!ready.empty() && (stats.bytes+bytes>capacity_limit || ready.size()>=job_limit)){
                        // Protect the result being joined by the sole consumer.
                        auto victim=ready.begin();
                        if(demanded && victim->key==demand_key)++victim;
                        if(victim==ready.end())break;
                        stats.bytes-=victim->value->bytes();ready.erase(victim);++stats.evicted;
                    }
                    if(stats.bytes+bytes<=capacity_limit){
                        ready.push_back({job.key,std::move(value)});
                        stats.bytes+=bytes;stats.peak_bytes=std::max(stats.peak_bytes,stats.bytes);++stats.built;
                    }else ++stats.rejected;
                }else ++stats.rejected;
            }catch(...){++stats.rejected;}
            active[worker]=false;completed.notify_all();
        }
    }
public:
    ContentPreparation()=default;
    ContentPreparation(ContentPreparation const&)=delete;
    ~ContentPreparation(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;cancel=true;wake.notify_all();}
        for(auto& worker:workers)worker.join();
    }
    // Joins only the active CPU task at its fine-grained cancellation boundary.
    // Completed content survives; the caller may now mutate/reset source owners.
    void pause(){
        std::unique_lock<std::mutex> lock(mutex);paused=true;cancel=true;completed.notify_all();
        completed.wait(lock,[&]{return std::none_of(active.begin(),active.end(),[](bool value){return value;});});
    }
    void clear(){
        pause();std::lock_guard<std::mutex> lock(mutex);
        ready.clear();pending.clear();compile={};stats.bytes=0;
    }
    template<class Valid> bool contains(Key const& key,Valid valid){
        std::lock_guard<std::mutex> lock(mutex);
        for(auto it=ready.begin();it!=ready.end();++it)if(it->key==key){
            if(valid(*it->value))return true;
            stats.bytes-=it->value->bytes();ready.erase(it);++stats.invalidated;return false;
        }
        return false;
    }
    // Must be called while paused. Jobs own scalar capture data; shared assets
    // and world inputs remain borrowed under the caller's explicit read lease.
    void configure(std::deque<Job> jobs,Compile next,unsigned count=1,std::vector<Key> needed={},std::size_t budget=byte_limit){
        std::lock_guard<std::mutex> lock(mutex);
        if(!paused || std::any_of(active.begin(),active.end(),[](bool value){return value;}) || jobs.size()>job_limit || count<1 || count>6 || budget<byte_limit || budget>128u*1024u*1024u)throw std::logic_error("CPU preparation lease/budget");
        if(capacity_limit!=budget){ready.clear();stats.bytes=stats.peak_bytes=0;capacity_limit=budget;}
        std::sort(needed.begin(),needed.end());
        auto required=[&](Key const& key){return std::binary_search(needed.begin(),needed.end(),key);};
        bool immediate=std::any_of(jobs.begin(),jobs.end(),[&](auto const& job){return required(job.key);});
        if(immediate){
            // Old speculative content cannot close the producer gate while a
            // newly selected view needs compilation. Protect ready demand and
            // keep half the refill watermark for useful speculative survivors.
            while(stats.bytes>capacity_limit/4){
                auto victim=std::find_if(ready.begin(),ready.end(),[&](auto const& item){return !required(item.key);});
                if(victim==ready.end())break;
                stats.bytes-=victim->value->bytes();ready.erase(victim);++stats.evicted;
            }
            std::stable_partition(jobs.begin(),jobs.end(),[&](auto const& job){return required(job.key);});
        }
        pending.swap(jobs);compile=std::move(next);worker_limit=count;
    }
    void resume(){
        std::lock_guard<std::mutex> lock(mutex);
        if(pending.empty() || stopping)return;
        while(workers.size()<worker_limit){auto index=unsigned(workers.size());workers.emplace_back([this,index]{run(index);});}
        cancel=false;paused=false;wake.notify_all();
    }
    std::unique_ptr<Result> take(Key const& key,bool caller_compiles_pending=false){
        auto begin=std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex);
        // A demanded missing tile moves ahead of speculative neighbors. The
        // running tile is short, useful CPU work; never spawn duplicate builders.
        auto found=std::find_if(pending.begin(),pending.end(),[&](auto const& j){return j.key==key;});
        bool queued=found!=pending.end();
        if(queued && caller_compiles_pending){pending.erase(found);wake.notify_all();return {};}
        if(queued)std::rotate(pending.begin(),found,std::next(found));
        // One consumer owns GPU adoption. Its demand bypasses speculative
        // backpressure and is protected from eviction while being joined.
        demanded=true;demand_key=key;
        auto running=[&]{for(unsigned i=0;i<active.size();++i)if(active[i] && active_key[i]==key)return true;return false;};
        if(!paused && (queued || running())){
            wake.notify_all();
            completed.wait(lock,[&]{
                if(paused || stopping)return true;
                if(running())return false;
                return std::none_of(pending.begin(),pending.end(),[&](auto const& j){return j.key==key;});
            });
        }
        demanded=false;
        stats.wait_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        for(auto it=ready.begin();it!=ready.end();++it)if(it->key==key){
            stats.bytes-=it->value->bytes();auto value=std::move(it->value);ready.erase(it);++stats.consumed;wake.notify_all();return value;
        }
        return {};
    }
    Statistics statistics(){std::lock_guard<std::mutex> lock(mutex);auto result=stats;result.pending=pending.size();return result;}
};
}}
