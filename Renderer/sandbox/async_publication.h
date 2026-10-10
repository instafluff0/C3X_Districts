#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <string>
#include <chrono>
#include <cstring>
#include <type_traits>

namespace c3x_async {
// Only this transport thread waits on the helper. Receipt, execution and safe
// snapshot supersession have separate counters; reliable work is never replaced.
class Publication {
    using Clock=std::chrono::steady_clock;
    using Observer=std::function<void(char const*,double,double)>;
    struct Entry {
        std::size_t bytes=0,units=0,records=1;std::function<void()> work;
        unsigned replace_key=0,group_key=0;char const* label=nullptr;
        Clock::time_point posted;std::shared_ptr<void> group;bool reconciliation=false;
        std::function<void()> finished;
        bool (*passes)(char const*)=nullptr; // queued tail labels this entry may run ahead of
    };
    mutable std::mutex mutex;std::condition_variable wake;std::deque<Entry> entries;
    std::size_t bytes=0,units=0,records=0,limit,count_limit,work_limit;
    std::size_t peak_bytes=0,peak_records=0,peak_units=0;
    Clock::time_point active_posted;
    std::atomic<bool> fault{false};
    std::atomic<unsigned> submitted{0},consumed{0},superseded{0},abandoned{0},rejected{0};
    bool stopping=false;std::function<void(char const*)> report;Observer observer;std::thread thread;
    std::function<void(char const*)> passing_posted;
    void run(){for(;;){Entry next;Observer observe;
        {std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[&]{return stopping||!entries.empty();});
            if(stopping&&entries.empty())return;
            next=std::move(entries.front());entries.pop_front();observe=observer;active_posted=next.posted;}
        execute(next,observe);
        {std::lock_guard<std::mutex> lock(mutex);active_posted={};}
        wake.notify_all();
        if(next.finished)next.finished();
    }}
    void execute(Entry& next,Observer const& observe){
        auto started=Clock::now();
        try{if(healthy()||next.reconciliation){
            next.work();consumed.fetch_add(unsigned(next.records),std::memory_order_release);
            if(next.reconciliation)fault.store(false,std::memory_order_release);
        }else abandoned.fetch_add(unsigned(next.records),std::memory_order_release);}
        catch(std::exception const& error){abandoned.fetch_add(unsigned(next.records),std::memory_order_release);fail(error.what());}
        catch(...){abandoned.fetch_add(unsigned(next.records),std::memory_order_release);fail("unknown asynchronous renderer failure");}
        if(observe&&next.label)observe(next.label,
            std::chrono::duration<double,std::milli>(started-next.posted).count(),
            std::chrono::duration<double,std::milli>(Clock::now()-started).count());
        // Capacity becomes reusable only after the executed owned payload dies.
        next.work={};next.group.reset();
        {std::lock_guard<std::mutex> lock(mutex);bytes-=next.bytes;units-=next.units;records-=next.records;}
    }
    bool admit(Entry next,std::function<void(std::shared_ptr<void> const&,std::shared_ptr<void> const&)> append={},
               std::size_t join_bytes=0,std::size_t join_units=0,std::size_t join_records=0,
               std::function<void(Entry&)> prepare={},bool wait_capacity=false){
        bool accepted=false;std::string pressure;
        auto size=next.bytes,semantic=next.units;auto label=next.label;bool passing=next.passes!=nullptr;
        {std::unique_lock<std::mutex> lock(mutex);
            if(next.replace_key)for(auto at=entries.begin();at!=entries.end();){
                if(at->replace_key==next.replace_key){bytes-=at->bytes;units-=at->units;records-=at->records;
                    superseded.fetch_add(unsigned(at->records),std::memory_order_release);at=entries.erase(at);}
                else ++at;}
            auto fits=[&]{return next.bytes<=limit-bytes&&next.units<=work_limit-units&&records<count_limit;};
            // Waiting producers leave an eighth of every budget to ordinary
            // state publication, which never waits. Filled to the limits by
            // images, the next unit fact was rejected and faulted the session
            // (a long melee, review 47). An empty queue admits any fitting size.
            auto room=[&]{if(!wait_capacity||(!bytes&&!units&&!records))return fits();
                return next.bytes+limit/8<=limit-bytes&&next.units+work_limit/8<=work_limit-units&&
                    records+count_limit/8<count_limit;};
            if(wait_capacity&&size<=limit&&semantic<=work_limit&&count_limit)
                wake.wait(lock,[&]{return stopping||!healthy()||room();});
            if(!stopping&&(healthy()||next.reconciliation)&&room()){
                // This opt-in factory only copies immutable caller memory. The
                // gate reserves its exact capacity before any payload allocation.
                if(prepare)prepare(next);
                if(healthy()||next.reconciliation){
                    auto at=entries.end();
                    if(next.passes)while(at!=entries.begin()&&!(at-1)->reconciliation&&next.passes((at-1)->label))--at;
                    // A copied fact can join the preceding fact batch across
                    // independent canvas work. Neither fact nor image order is
                    // changed within its own stream; barriers remain barriers.
                    if(append&&at!=entries.begin()&&(at-1)->group_key==next.group_key&&
                       (at-1)->bytes+next.bytes<=join_bytes&&(at-1)->units+next.units<=join_units&&
                       (at-1)->records<join_records){
                        auto& prior=*(at-1);append(prior.group,next.group);prior.bytes+=next.bytes;
                        prior.units+=next.units;++prior.records;
                    }else entries.insert(at,std::move(next));
                    // Every operation consumes budgets even when its container joins.
                    bytes+=size;units+=semantic;++records;
                    peak_bytes=std::max(peak_bytes,bytes);peak_records=std::max(peak_records,records);peak_units=std::max(peak_units,units);
                    submitted.fetch_add(1,std::memory_order_release);accepted=true;
                }
            }
            if(!accepted){
                rejected.fetch_add(1,std::memory_order_release);
                pressure="renderer publication pressure; bytes="+std::to_string(bytes)+" packets="+std::to_string(records)+
                    " work="+std::to_string(units)+" incoming="+std::to_string(next.bytes)+
                    " accepted="+std::to_string(submitted.load())+" executed="+std::to_string(consumed.load())+
                    " superseded="+std::to_string(superseded.load())+"; explicit native/resource/scene reset required";
            }
        }
        if(accepted)wake.notify_all();else fail(pressure.c_str());
        if(accepted&&passing){std::function<void(char const*)> notify;
            {std::lock_guard<std::mutex> lock(mutex);notify=passing_posted;}
            if(notify)notify(label);}
        return accepted;
    }
public:
    struct Status {unsigned accepted,executed,superseded,abandoned,rejected;
        std::size_t bytes,records,units,peak_bytes,peak_records,peak_units;double oldest_ms;};
    // Allow eight semantic operations per packet at the full packet watermark.
    // Work, packet and byte bounds are independent; merging changes none of them.
    static constexpr std::size_t default_work_limit=65536;
    explicit Publication(std::function<void(char const*)> diagnostic={},
        std::size_t byte_limit=128u*1024u*1024u,std::size_t packet_limit=8192,std::size_t semantic_limit=default_work_limit):
        limit(byte_limit),count_limit(packet_limit),work_limit(semantic_limit),report(std::move(diagnostic)),thread([this]{run();}){}
    ~Publication(){stop();}
    void stop(){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}wake.notify_all();if(thread.joinable())thread.join();}
    Publication(Publication const&)=delete;Publication& operator=(Publication const&)=delete;
    bool healthy()const{return !fault.load(std::memory_order_acquire);}
    // Waits for queue progress: each execution, fault and stop wakes it.
    template<class Ready>void wait_progress(Ready ready){
        std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[&]{return stopping||!healthy()||ready();});}
    unsigned accepted()const{return submitted.load(std::memory_order_acquire);}
    unsigned completed()const{return consumed.load(std::memory_order_acquire);}
    Status status()const{std::lock_guard<std::mutex> lock(mutex);
        return {submitted.load(),consumed.load(),superseded.load(),abandoned.load(),rejected.load(),
            bytes,records,units,peak_bytes,peak_records,peak_units,
            active_posted!=Clock::time_point{}?std::chrono::duration<double,std::milli>(Clock::now()-active_posted).count():
                entries.empty()?0:std::chrono::duration<double,std::milli>(Clock::now()-entries.front().posted).count()};}
    void observe(Observer value){std::lock_guard<std::mutex> lock(mutex);observer=std::move(value);}
    // Told the label of each admitted entry that may pass queued work, so a
    // consumer wait inside the work in flight can wake (see run_passing).
    void on_passing_post(std::function<void(char const*)> value){std::lock_guard<std::mutex> lock(mutex);passing_posted=std::move(value);}
    // Called by the work in flight, on this publication thread, while it waits
    // for its consumer. Runs the first queued entry labelled `label` if that
    // entry may pass `in_flight` and every entry queued before it: the same
    // order as if it had been inserted ahead of them. Returns whether it ran.
    bool run_passing(char const* in_flight,char const* label){
        Entry next;Observer observe;
        {std::lock_guard<std::mutex> lock(mutex);
            if(stopping||!label)return false;
            auto at=entries.begin();
            while(at!=entries.end()&&!(at->label&&!std::strcmp(at->label,label)))++at;
            if(at==entries.end()||!at->passes||at->reconciliation||!at->passes(in_flight))return false;
            for(auto before=entries.begin();before!=at;++before)
                if(before->reconciliation||!at->passes(before->label))return false;
            next=std::move(*at);entries.erase(at);observe=observer;
        }
        execute(next,observe);wake.notify_all();
        if(next.finished)next.finished();
        return true;
    }
    void fail(char const* reason){bool first;
        {std::lock_guard<std::mutex> lock(mutex);first=!fault.exchange(true,std::memory_order_acq_rel);}
        wake.notify_all();if(first&&report)report(reason);
    }
    // An entry with `passes` is inserted ahead of the queued tail whose labels
    // it accepts; everything else, and its own later posts, keep their order.
    bool post(std::size_t size,std::function<void()> work,unsigned replace_key=0,char const* label=nullptr,
              std::size_t semantic_work=1,bool (*passes)(char const*)=nullptr){
        Entry entry;entry.bytes=size;entry.units=semantic_work;entry.work=std::move(work);
        entry.replace_key=replace_key;entry.label=label;entry.posted=Clock::now();entry.passes=passes;return admit(std::move(entry));
    }
    template<class Group,class Work,class Append>bool post_group(std::size_t size,std::size_t semantic_work,
        unsigned key,std::shared_ptr<Group> group,Work work,Append append,std::size_t join_bytes,
        std::size_t join_units,std::size_t join_records,char const* label,bool (*passes)(char const*)=nullptr){
        Entry entry;entry.bytes=size;entry.units=semantic_work;entry.group_key=key;entry.group=group;entry.passes=passes;
        entry.label=label;entry.posted=Clock::now();entry.work=[group,work]{work(*group);};
        return admit(std::move(entry),[append](auto const& target,auto const& incoming){append(*std::static_pointer_cast<Group>(target),*std::static_pointer_cast<Group>(incoming));},
            join_bytes,join_units,join_records);
    }
    // Reliable image producers may wait for the execution prefix to free space.
    // No caller pixels are copied, and no identity is published, before that fit.
    // Oversized requests, stop and fault return without invoking the factory.
    template<class Group,class Factory,class Work,class Append>bool post_group_wait(std::size_t size,std::size_t semantic_work,
        unsigned key,Factory factory,Work work,Append append,std::size_t join_bytes,
        std::size_t join_units,std::size_t join_records,char const* label){
        Entry entry;entry.bytes=size;entry.units=semantic_work;entry.group_key=key;
        entry.label=label;entry.posted=Clock::now();
        return admit(std::move(entry),[append](auto const& target,auto const& incoming){
            append(*std::static_pointer_cast<Group>(target),*std::static_pointer_cast<Group>(incoming));
        },join_bytes,join_units,join_records,[factory,work](Entry& value){
            auto group=factory();value.group=group;value.work=[group,work]{work(*group);};
        },true);
    }
    // Configuration/reset joins outside the frame path. Recovery is an explicit
    // generation retirement, with abandoned accepted work still visible above.
    template<class Function>auto reconcile(Function work)->decltype(work()){
        using Result=decltype(work());auto task=std::make_shared<std::promise<Result>>();
        auto committed=std::make_shared<std::promise<void>>();auto committed_future=committed->get_future();
        auto done=task->get_future();Entry entry;entry.bytes=sizeof(*task);entry.units=1;entry.reconciliation=true;
        entry.posted=Clock::now();entry.work=[task,work]{
            try{if constexpr(std::is_void<Result>::value){work();task->set_value();}
                else task->set_value(work());}
            catch(...){task->set_exception(std::current_exception());throw;}
        };
        entry.finished=[committed]{committed->set_value();};
        // Explicit teardown may join the failed prefix; ordinary state
        // publication never waits for capacity. No old reliable suffix crosses reset.
        {std::unique_lock<std::mutex> lock(mutex);wake.wait(lock,[&]{return !records||stopping;});}
        if(!admit(std::move(entry)))throw std::runtime_error("renderer reconciliation unavailable");
        task.reset();committed_future.get();return done.get();
    }
    template<class Function>auto setup(Function work)->decltype(work()){
        using Result=decltype(work());auto task=std::make_shared<std::packaged_task<Result()>>(std::move(work));auto done=task->get_future();
        if(!post(sizeof(*task),[task]{(*task)();}))throw std::runtime_error("renderer publication unavailable");
        task.reset();return done.get();
    }
};
}
