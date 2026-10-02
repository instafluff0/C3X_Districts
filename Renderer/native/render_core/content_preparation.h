#pragma once
// A bounded producer for existing scene compilers. Durable jobs own changing
// inputs; compatibility callers may grant explicit borrowed read leases. Results
// require current dependency validation. Asset/device retirement joins readers.
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
#include <set>
#include <thread>
namespace c3x_renderer { namespace render_core {
template<class Key,class Input,class Result> class ContentPreparation {
public:
    struct Job {Key key;Input input;bool urgent=false;};
    using Compile=std::function<std::unique_ptr<Result>(Input const&,std::atomic<bool> const&,unsigned)>;
    // The callback owns a CPU-only snapshot, never a result that take() can move.
    struct OptionalWork {std::function<void(std::atomic<bool> const&,unsigned)> run;std::size_t bytes=0;};
    struct Statistics {
        std::uint64_t built=0,consumed=0,cancelled=0,rejected=0,evicted=0,invalidated=0;
        std::size_t bytes=0,peak_bytes=0,pending=0;
        unsigned active_peak=0;
        unsigned active=0;
        double cpu_ms=0,wait_ms=0;
        std::size_t required_ready_bytes=0,unneeded_ready_bytes=0,optional_bytes=0;
        std::size_t required_keys=0,expected_consumed_keys=0,consumed_required_keys=0;
        unsigned active_required=0,active_unneeded=0,active_optional=0;
        std::uint64_t retired_pending=0,retired_ready=0,retired_active=0,needed_result_evictions=0;
        std::uint64_t optional_completed=0,optional_cancelled=0,optional_skipped=0;
        double capacity_wait_ms=0;
        std::size_t join_bytes=0,join_peak_bytes=0;
        std::size_t capacity=0,reserved_bytes=0;
        unsigned active_join=0;
        std::uint64_t join_dispatches=0,join_cancelled=0;
    };
    static constexpr std::size_t byte_limit=16u*1024u*1024u,job_limit=8192;
    static constexpr std::size_t optional_byte_limit=byte_limit,optional_job_limit=4;
private:
    struct Ready {Key key;std::unique_ptr<Result> value;bool urgent=false;};
    std::mutex mutex;
    std::condition_variable wake,completed;
    std::vector<std::thread> workers;
    unsigned worker_limit=1;
    std::size_t capacity_limit=byte_limit;
    std::atomic<bool> cancel{false};
    std::atomic<bool> optional_cancel{false};
    bool paused=true,stopping=false,demanded=false;
    bool bounded_results=false;
    Key demand_key{};
    std::array<bool,6> active{};
    std::array<Key,6> active_key{};
    std::array<bool,6> active_urgent{};
    std::array<bool,6> active_join{};
    std::array<bool,6> optional_active{},capacity_waiting{};
    std::array<std::chrono::steady_clock::time_point,6> capacity_since{};
    bool exact_required=false;
    std::vector<Key> required_keys,consumed_keys;
    std::deque<OptionalWork> optional;
    Compile compile;
    // Notification owns no content lease. Unregister joins any callback before
    // its consumer can disappear; callbacks may acquire the consumer's mutex.
    std::mutex notification_mutex;
    std::function<void()> ready_notification;
    std::deque<Job> pending;
    std::deque<Ready> ready;
    Ready joined{Key{},{},false};
    Statistics stats;
    bool required(Key const& key)const{
        return !exact_required || std::binary_search(required_keys.begin(),required_keys.end(),key);
    }
    void select_required(std::vector<Key> keys){
        std::sort(keys.begin(),keys.end());keys.erase(std::unique(keys.begin(),keys.end()),keys.end());
        exact_required=true;required_keys=std::move(keys);consumed_keys.clear();
        if(joined.value && !required(joined.key)){joined.value.reset();stats.join_bytes=0;++stats.retired_ready;}
        for(auto it=ready.begin();it!=ready.end();)if(!required(it->key)){
            stats.bytes-=it->value->bytes();it=ready.erase(it);++stats.retired_ready;
        }else ++it;
    }
    void record_consumed(Key const& key){
        if(exact_required && required(key) && std::find(consumed_keys.begin(),consumed_keys.end(),key)==consumed_keys.end())
            consumed_keys.push_back(key);
    }
    std::unique_ptr<Result> take_joined(Key const& key){
        if(!joined.value || !(joined.key==key))return {};
        auto value=std::move(joined.value);stats.join_bytes=0;++stats.consumed;record_consumed(key);wake.notify_all();return value;
    }
    void prune_jobs(std::deque<Job>& jobs){
        std::set<Key> unique;
        jobs.erase(std::remove_if(jobs.begin(),jobs.end(),[&](auto const& job){
            if(!required(job.key) || !unique.insert(job.key).second)return true;
            if(joined.value && joined.key==job.key)return true;
            for(auto const& item:ready)if(item.key==job.key)return true;
            for(unsigned i=0;i<active.size();++i)if(active[i] && active_key[i]==job.key)return true;
            return false;
        }),jobs.end());
    }
    void capacity_wait(unsigned worker,bool waiting){
        auto now=std::chrono::steady_clock::now();
        if(capacity_waiting[worker] && !waiting)stats.capacity_wait_ms+=
            std::chrono::duration<double,std::milli>(now-capacity_since[worker]).count();
        if(waiting && !capacity_waiting[worker])capacity_since[worker]=now;
        capacity_waiting[worker]=waiting;
    }
    void run(unsigned worker) {
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            wake.wait(lock,[&]{
                if(stopping){capacity_wait(worker,false);return true;}
                if(paused || worker>=worker_limit){capacity_wait(worker,false);return false;}
                if(pending.empty()){
                    capacity_wait(worker,false);
                    return !optional.empty() && std::none_of(optional_active.begin(),optional_active.end(),[](bool v){return v;});
                }
                bool allowed=false;
                if(demanded && pending.front().key==demand_key && !exact_required)allowed=true;
                else if(bounded_results){
                    auto reserved=std::size_t(std::count(active.begin(),active.end(),true))*byte_limit;
                    allowed=stats.bytes+reserved<=capacity_limit-byte_limit;
                    // One synchronous consumer can need the other component of
                    // a partial tile while all ready entries remain required.
                    // Its separate handoff is bounded by the result maximum.
                    if(!allowed && exact_required && demanded && pending.front().key==demand_key &&
                            !joined.value && std::none_of(active_join.begin(),active_join.end(),[](bool v){return v;}))allowed=true;
                }else allowed=pending.front().urgent || stats.bytes<capacity_limit/2;
                capacity_wait(worker,!allowed);return allowed;
            });
            if(stopping)return;
            if(pending.empty()){
                {
                    auto work=std::move(optional.front());optional.pop_front();
                    auto bytes=work.bytes;optional_active[worker]=true;optional_cancel=false;
                    lock.unlock();bool failed=false;
                    try{work.run(optional_cancel,worker);}catch(...){failed=true;}
                    lock.lock();
                    if(failed || optional_cancel.load(std::memory_order_relaxed))++stats.optional_cancelled;
                    else ++stats.optional_completed;
                    stats.optional_bytes-=bytes;
                } // Release the independently owned snapshot before retirement.
                optional_active[worker]=false;completed.notify_all();wake.notify_all();continue;
            }
            bool published=false;
            {
                Job job=std::move(pending.front());pending.pop_front();
                auto reserved=std::size_t(std::count(active.begin(),active.end(),true))*byte_limit;
                active_join[worker]=exact_required && bounded_results && demanded && job.key==demand_key &&
                    stats.bytes+reserved>capacity_limit-byte_limit;
                if(active_join[worker])++stats.join_dispatches;
                active[worker]=true;active_key[worker]=job.key;active_urgent[worker]=job.urgent;
                stats.active_peak=std::max(stats.active_peak,unsigned(std::count(active.begin(),active.end(),true)));
                lock.unlock();auto begin=std::chrono::steady_clock::now();
                std::unique_ptr<Result> value;
                try{value=compile(job.input,cancel,worker);}catch(...){}
                auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                lock.lock();stats.cpu_ms+=elapsed;
                try {
                    if(!required(job.key)){++stats.retired_active;if(active_join[worker])++stats.join_cancelled;}
                    else if(cancel.load(std::memory_order_relaxed)){
                        ++stats.cancelled;if(!stopping)pending.push_front(std::move(job));
                    }else if(value && value->bytes()<=(bounded_results?byte_limit:capacity_limit)){
                        auto bytes=value->bytes();
                        if(active_join[worker]){
                            if(demanded && demand_key==job.key){
                                joined={job.key,std::move(value),active_urgent[worker]};stats.join_bytes=bytes;
                                stats.join_peak_bytes=std::max(stats.join_peak_bytes,bytes);++stats.built;published=true;
                            }else ++stats.join_cancelled;
                        }else{
                        while(!ready.empty() && (stats.bytes+bytes>capacity_limit || ready.size()>=job_limit)){
                            auto victim=std::find_if(ready.begin(),ready.end(),[&](auto const& item){
                                return (exact_required?!required(item.key):!item.urgent) && !(demanded && item.key==demand_key);});
                            if(victim==ready.end() && !exact_required){
                                victim=ready.begin();if(demanded && victim->key==demand_key)++victim;
                            }
                            if(victim==ready.end())break;
                            if(required(victim->key))++stats.needed_result_evictions;
                            stats.bytes-=victim->value->bytes();ready.erase(victim);++stats.evicted;
                        }
                        if(stats.bytes+bytes<=capacity_limit){
                            ready.push_back({job.key,std::move(value),active_urgent[worker]});
                            stats.bytes+=bytes;stats.peak_bytes=std::max(stats.peak_bytes,stats.bytes);++stats.built;published=true;
                        }else ++stats.rejected;
                        }
                    }else ++stats.rejected;
                }catch(...){++stats.rejected;}
            } // Release job input leases before pause/clear can observe completion.
            active[worker]=active_join[worker]=false;completed.notify_all();wake.notify_all();
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
        if(auto value=take_joined(key))return value;
        for(auto it=ready.begin();it!=ready.end();++it)if(it->key==key){
            stats.bytes-=it->value->bytes();auto value=std::move(it->value);ready.erase(it);
            ++stats.consumed;record_consumed(key);wake.notify_all();return value;
        }
        return {};
    }
    ContentPreparation(ContentPreparation const&)=delete;
    ~ContentPreparation(){
        {std::lock_guard<std::mutex> lock(mutex);stopping=true;cancel=true;optional_cancel=true;wake.notify_all();}
        for(auto& worker:workers)worker.join();
    }
    // Joins only the active CPU task at its fine-grained cancellation boundary.
    // Completed content survives; the caller may now mutate/reset source owners.
    void pause(){
        std::unique_lock<std::mutex> lock(mutex);paused=true;cancel=true;optional_cancel=true;completed.notify_all();wake.notify_all();
        completed.wait(lock,[&]{return std::none_of(active.begin(),active.end(),[](bool value){return value;}) &&
            std::none_of(optional_active.begin(),optional_active.end(),[](bool value){return value;});});
    }
    void clear(){
        pause();std::lock_guard<std::mutex> lock(mutex);
        ready.clear();joined.value.reset();pending.clear();optional.clear();compile={};stats.bytes=stats.join_bytes=stats.optional_bytes=0;
        required_keys.clear();consumed_keys.clear();exact_required=false;
    }
    // End borrowed-input ownership without discarding complete, independently
    // owned results or destroying the bounded worker pool. The next lease must
    // validate survivors by stable content key before scheduling or taking them.
    void finish_lease(){
        pause();std::lock_guard<std::mutex> lock(mutex);
        pending.clear();optional.clear();stats.optional_bytes=0;compile={};
    }
    template<class Valid> bool contains(Key const& key,Valid valid){
        std::lock_guard<std::mutex> lock(mutex);
        if(joined.value && joined.key==key){
            if(valid(*joined.value))return true;
            joined.value.reset();stats.join_bytes=0;++stats.invalidated;wake.notify_all();return false;
        }
        for(auto it=ready.begin();it!=ready.end();++it)if(it->key==key){
            if(valid(*it->value))return true;
            stats.bytes-=it->value->bytes();ready.erase(it);++stats.invalidated;return false;
        }
        return false;
    }
    // Must be called while paused. Jobs own scalar capture data; shared assets
    // and world inputs remain borrowed under the caller's explicit read lease.
private:
    void configure_impl(std::deque<Job> jobs,Compile next,unsigned count,std::vector<Key> needed,std::size_t budget,bool bounded,std::vector<Key> const* required_selection){
        std::lock_guard<std::mutex> lock(mutex);
        if(!paused || std::any_of(active.begin(),active.end(),[](bool value){return value;}) ||
                std::any_of(optional_active.begin(),optional_active.end(),[](bool value){return value;}) ||
                jobs.size()>job_limit || count<1 || count>6 || budget<byte_limit || budget>128u*1024u*1024u)throw std::logic_error("CPU preparation lease/budget");
        if(required_selection)select_required(*required_selection);else {exact_required=false;required_keys.clear();consumed_keys.clear();}
        if(capacity_limit!=budget){if(!exact_required){ready.clear();stats.bytes=stats.peak_bytes=0;}capacity_limit=budget;}
        prune_jobs(jobs);
        std::sort(needed.begin(),needed.end());
        auto urgent=[&](Key const& key){return std::binary_search(needed.begin(),needed.end(),key);};
        // Selection owns urgency for this lease, including surviving results.
        // Queue position alone cannot bypass speculative refill backpressure.
        for(auto& job:jobs)job.urgent=urgent(job.key);
        for(auto& item:ready)item.urgent=urgent(item.key);
        bool immediate=std::any_of(jobs.begin(),jobs.end(),[](auto const& job){return job.urgent;});
        if(immediate){
            // Old speculative content cannot close the producer gate while a
            // newly selected view needs compilation. Protect ready demand and
            // keep half the refill watermark for useful speculative survivors.
            while(stats.bytes>capacity_limit/4){
                auto victim=std::find_if(ready.begin(),ready.end(),[&](auto const& item){return exact_required?!this->required(item.key):!urgent(item.key);});
                if(victim==ready.end())break;
                stats.bytes-=victim->value->bytes();ready.erase(victim);++stats.evicted;
            }
            std::stable_partition(jobs.begin(),jobs.end(),[&](auto const& job){return urgent(job.key);});
        }
        pending.swap(jobs);compile=std::move(next);worker_limit=count;bounded_results=bounded;
    }
    // Retarget durable jobs without revoking active readers. A stable compiler
    // consumes owned inputs; old pending jobs are replaced, active/ready work
    // survives and is validated by the consumer before adoption.
    void schedule_impl(std::deque<Job> jobs,Compile next,unsigned count,std::vector<Key> needed,
                  std::size_t budget,bool bounded,std::vector<Key> const* required_selection){
        std::lock_guard<std::mutex> lock(mutex);
        if(stopping || jobs.size()>job_limit || count<1 || count>6 || budget<byte_limit || budget>128u*1024u*1024u)
            throw std::logic_error("durable preparation budget");
        if(required_selection){select_required(*required_selection);
            for(auto const& job:pending)if(!required(job.key))++stats.retired_pending;
        }else {exact_required=false;required_keys.clear();consumed_keys.clear();}
        if(!compile)compile=std::move(next);
        capacity_limit=budget;worker_limit=count;bounded_results=bounded;
        std::sort(needed.begin(),needed.end());
        auto urgent=[&](Key const& key){return std::binary_search(needed.begin(),needed.end(),key);};
        for(auto& item:ready)item.urgent=urgent(item.key);
        for(unsigned i=0;i<active.size();++i)active_urgent[i]=active[i] && urgent(active_key[i]);
        prune_jobs(jobs);
        for(auto& job:jobs)job.urgent=urgent(job.key);
        std::stable_partition(jobs.begin(),jobs.end(),[](auto const& job){return job.urgent;});
        auto target=(!jobs.empty() && jobs.front().urgent)?capacity_limit/4:capacity_limit;
        while(stats.bytes>target){
            auto victim=std::find_if(ready.begin(),ready.end(),[&](auto const& item){return exact_required?!required(item.key):!item.urgent;});
            if(victim==ready.end())break;
            stats.bytes-=victim->value->bytes();ready.erase(victim);++stats.evicted;
        }
        pending.swap(jobs);if(!pending.empty())optional_cancel=true;
        while(!pending.empty() && workers.size()<worker_limit){auto index=unsigned(workers.size());workers.emplace_back([this,index]{run(index);});}
        cancel=false;paused=false;wake.notify_all();
    }
public:
    // Legacy producers retain speculative ready content. World selection uses
    // the exact overload: required keys include ready/active results still due
    // for adoption, while urgency only orders those missing compilations.
    void configure(std::deque<Job> jobs,Compile next,unsigned count=1,std::vector<Key> urgent={},std::size_t budget=byte_limit,bool bounded=false){
        configure_impl(std::move(jobs),std::move(next),count,std::move(urgent),budget,bounded,nullptr);
    }
    void configure(std::deque<Job> jobs,Compile next,unsigned count,std::vector<Key> required,std::vector<Key> urgent,std::size_t budget,bool bounded){
        configure_impl(std::move(jobs),std::move(next),count,std::move(urgent),budget,bounded,&required);
    }
    void schedule(std::deque<Job> jobs,Compile next,unsigned count,std::vector<Key> urgent,std::size_t budget,bool bounded=true){
        schedule_impl(std::move(jobs),std::move(next),count,std::move(urgent),budget,bounded,nullptr);
    }
    void schedule(std::deque<Job> jobs,Compile next,unsigned count,std::vector<Key> required,std::vector<Key> urgent,std::size_t budget,bool bounded){
        schedule_impl(std::move(jobs),std::move(next),count,std::move(urgent),budget,bounded,&required);
    }
    void resume(){
        std::lock_guard<std::mutex> lock(mutex);
        if(pending.empty() || stopping)return;
        while(workers.size()<worker_limit){auto index=unsigned(workers.size());workers.emplace_back([this,index]{run(index);});}
        cancel=false;paused=false;wake.notify_all();
    }
    // Return temporarily reserved lanes without revoking the immutable input
    // lease, cancelling useful jobs or replacing the pending/ready queues.
    void expand_workers(unsigned count){
        std::lock_guard<std::mutex> lock(mutex);
        if(count<1 || count>active.size())throw std::logic_error("CPU preparation worker limit");
        if(count<=worker_limit || stopping)return;
        worker_limit=count;
        if(!paused && !pending.empty())
            while(workers.size()<worker_limit){auto index=unsigned(workers.size());workers.emplace_back([this,index]{run(index);});}
        wake.notify_all();
    }
    // Append independently owned immutable inputs without revoking other readers.
    // Borrowed world-input callers continue using pause/configure/resume.
    bool offer(Job job,std::size_t limit,bool urgent=false) {
        std::lock_guard<std::mutex> lock(mutex);
        if(stopping || !compile || limit>job_limit || !required(job.key))return false;
        if(joined.value && joined.key==job.key){joined.urgent|=urgent;return true;}
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
    // Offered only after publication/adoption detached an independent CPU
    // snapshot. Saturation drops optional backing instead of delaying demand.
    bool offer_optional(OptionalWork work){
        std::lock_guard<std::mutex> lock(mutex);
        if(stopping || paused || !compile || !work.run || !work.bytes ||
                optional.size()>=optional_job_limit || work.bytes>optional_byte_limit-stats.optional_bytes){
            ++stats.optional_skipped;return false;
        }
        stats.optional_bytes+=work.bytes;optional.push_back(std::move(work));
        while(workers.size()<worker_limit){auto index=unsigned(workers.size());workers.emplace_back([this,index]{run(index);});}
        wake.notify_all();return true;
    }
    std::unique_ptr<Result> take(Key const& key,bool caller_compiles_pending=false,std::function<bool()> obsolete={}){
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
        struct DemandRelease {ContentPreparation& queue;
            ~DemandRelease(){
                if(!queue.demanded)return;
                queue.demanded=false;
                if(queue.joined.value && queue.joined.key==queue.demand_key){
                    queue.joined.value.reset();queue.stats.join_bytes=0;++queue.stats.join_cancelled;
                }
                queue.wake.notify_all();
            }
        } demand_release{*this};
        // Cancellation may service a bounded renderer-owner turn. Never call
        // external code under the content mutex: workers must keep publishing
        // immutable results while the consumer yields to native image work.
        auto obsolete_now=[&]{
            if(!obsolete)return false;
            lock.unlock();bool result=false;
            try{result=obsolete();}catch(...){lock.lock();throw;}
            lock.lock();return result;
        };
        auto running=[&]{for(unsigned i=0;i<active.size();++i)if(active[i] && active_key[i]==key)return true;return false;};
        if(!paused && (queued || running())){
            wake.notify_all();
            auto complete=[&]{
                if(paused || stopping || obsolete_now())return true;
                if(running())return false;
                return std::none_of(pending.begin(),pending.end(),[&](auto const& j){return j.key==key;});
            };
            if(obsolete)while(!complete())completed.wait_for(lock,std::chrono::milliseconds(1));
            else completed.wait(lock,complete);
        }
        stats.wait_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        bool cancelled=obsolete_now();demanded=false;
        if(cancelled){
            if(joined.value && joined.key==key){joined.value.reset();stats.join_bytes=0;++stats.join_cancelled;wake.notify_all();}
            return {};
        }
        if(auto value=take_joined(key))return value;
        for(auto it=ready.begin();it!=ready.end();++it)if(it->key==key){
            stats.bytes-=it->value->bytes();auto value=std::move(it->value);ready.erase(it);++stats.consumed;record_consumed(key);wake.notify_all();return value;
        }
        return {};
    }
    Statistics statistics(){
        std::lock_guard<std::mutex> lock(mutex);auto result=stats;result.pending=pending.size();
        result.active=unsigned(std::count(active.begin(),active.end(),true));
        result.capacity=capacity_limit;
        result.required_keys=result.expected_consumed_keys=exact_required?required_keys.size():0;
        result.consumed_required_keys=consumed_keys.size();
        for(auto const& item:ready){if(required(item.key))result.required_ready_bytes+=item.value->bytes();else result.unneeded_ready_bytes+=item.value->bytes();}
        if(joined.value){if(required(joined.key))result.required_ready_bytes+=joined.value->bytes();else result.unneeded_ready_bytes+=joined.value->bytes();}
        auto now=std::chrono::steady_clock::now();
        for(unsigned i=0;i<active.size();++i){
            if(active[i]){if(required(active_key[i]))++result.active_required;else ++result.active_unneeded;}
            if(optional_active[i])++result.active_optional;
            if(active_join[i])++result.active_join;
            if(capacity_waiting[i])result.capacity_wait_ms+=std::chrono::duration<double,std::milli>(now-capacity_since[i]).count();
        }
        result.reserved_bytes=std::size_t(result.active-result.active_join)*byte_limit;
        return result;
    }
};
}}
