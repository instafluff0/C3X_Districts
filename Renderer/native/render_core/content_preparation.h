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
    struct Job {Key key;Input input;bool urgent=false;};
    using Compile=std::function<std::unique_ptr<Result>(Input const&,std::atomic<bool> const&,unsigned)>;
    struct Statistics {
        std::uint64_t built=0,consumed=0,cancelled=0,rejected=0,evicted=0,invalidated=0;
        std::size_t bytes=0,peak_bytes=0,pending=0;
        unsigned active_peak=0;
        double cpu_ms=0,wait_ms=0;
    };
    static constexpr std::size_t byte_limit=16u*1024u*1024u,job_limit=8192;
private:
    struct Ready {Key key;std::unique_ptr<Result> value;bool urgent=false;};
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
    std::array<bool,6> active_urgent{};
    Compile compile;
    // Notification owns no content lease. Unregister joins any callback before
    // its consumer can disappear; callbacks may acquire the consumer's mutex.
    std::mutex notification_mutex;
    std::function<void()> ready_notification;
    std::deque<Job> pending;
    std::deque<Ready> ready;
    Statistics stats;
    void run(unsigned worker) {
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            wake.wait(lock,[&]{return stopping || (!paused && worker<worker_limit && !pending.empty() && ((demanded && pending.front().key==demand_key) || pending.front().urgent || (stats.bytes<capacity_limit/2)));});
            if(stopping)return;
            bool published=false;
            {
            Job job=std::move(pending.front());pending.pop_front();
            active[worker]=true;active_key[worker]=job.key;active_urgent[worker]=job.urgent;
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
                        auto victim=std::find_if(ready.begin(),ready.end(),[&](auto const& item){return !item.urgent && !(demanded && item.key==demand_key);});
                        if(victim==ready.end())victim=ready.begin();
                        if(demanded && victim->key==demand_key)++victim;
                        if(victim==ready.end())break;
                        stats.bytes-=victim->value->bytes();ready.erase(victim);++stats.evicted;
                    }
                    if(stats.bytes+bytes<=capacity_limit){
                        ready.push_back({job.key,std::move(value),active_urgent[worker]});
                        stats.bytes+=bytes;stats.peak_bytes=std::max(stats.peak_bytes,stats.bytes);++stats.built;published=true;
                    }else ++stats.rejected;
                }else ++stats.rejected;
            }catch(...){++stats.rejected;}
            } // Release job input leases before pause/clear can observe completion.
            active[worker]=false;completed.notify_all();
            // Pause may return before notification: source inputs are no longer
            // borrowed. Never call the consumer while holding the content mutex.
            if(published){
                lock.unlock();
                {std::lock_guard<std::mutex> guard(notification_mutex);
                if(ready_notification)ready_notification();}
                lock.lock();
            }
        }
    }
public:
    ContentPreparation()=default;
    void set_ready_notification(std::function<void()> next){
        std::lock_guard<std::mutex> guard(notification_mutex);ready_notification=std::move(next);
    }
    // Advisory only: compilation may finish immediately after this snapshot.
    // The sole consumer may finish other independent content before joining;
    // take() remains the authority for consuming/stealing exactly one result.
    bool compiling(Key const& key){
        std::lock_guard<std::mutex> lock(mutex);
        for(unsigned i=0;i<active.size();++i)if(active[i] && active_key[i]==key)return true;
        return false;
    }
    // Speculative GPU adoption never joins a helper or steals a queued CPU job.
    std::unique_ptr<Result> take_ready(Key const& key){
        std::lock_guard<std::mutex> lock(mutex);
        for(auto it=ready.begin();it!=ready.end();++it)if(it->key==key){
            stats.bytes-=it->value->bytes();auto value=std::move(it->value);ready.erase(it);
            ++stats.consumed;wake.notify_all();return value;
        }
        return {};
    }
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
        // Selection owns urgency for this lease, including surviving results.
        // Queue position alone cannot bypass speculative refill backpressure.
        for(auto& job:jobs)job.urgent=required(job.key);
        for(auto& item:ready)item.urgent=required(item.key);
        bool immediate=std::any_of(jobs.begin(),jobs.end(),[](auto const& job){return job.urgent;});
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
    // Append independently owned immutable inputs without revoking other readers.
    // Borrowed world-input callers continue using pause/configure/resume.
    bool offer(Job job,std::size_t limit,bool urgent=false) {
        std::lock_guard<std::mutex> lock(mutex);
        if(stopping || !compile || limit>job_limit)return false;
        job.urgent=urgent;
        for(auto& item:ready)if(item.key==job.key){item.urgent|=urgent;return true;}
        auto insert_position=[&](){return std::find_if(pending.begin(),pending.end(),[](auto const& item){return !item.urgent;});};
        for(auto it=pending.begin();it!=pending.end();++it)if(it->key==job.key){
            if(urgent && !it->urgent){it->urgent=true;auto position=insert_position();if(position!=pending.end() && position<it)std::rotate(position,it,std::next(it));}
            wake.notify_all();return true;
        }
        for(unsigned i=0;i<active.size();++i)if(active[i] && active_key[i]==job.key){active_urgent[i]|=urgent;return true;}
        if(!limit)return false;
        if(pending.size()>=limit){if(!urgent)return false;pending.pop_back();}
        // Older advancing-unit predictions are due before a newly offered
        // next frame. Keep their FIFO order; speculative fixed-unit work follows.
        if(urgent)pending.insert(insert_position(),std::move(job));
        else pending.push_back(std::move(job));
        while(workers.size()<worker_limit){auto index=unsigned(workers.size());workers.emplace_back([this,index]{run(index);});}
        cancel=false;paused=false;wake.notify_all();return true;
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
