#pragma once
#include "gpu_frame_api.h"
#include "gpu_image_commands.h"
#include "native_hit_scene.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
namespace c3x_gpu_images {
// Caller-thread transport only. No native pointers, leases or D3D objects enter
// packets. Consecutive draws coalesce until a resource/CPU/frame boundary.
class WorkerClient {
    c3x_renderer_gpu_images_fn execute;std::int64_t ticket,session;
    std::vector<c3x_renderer_gpu_command_v1> pending;
    c3x_renderer_gpu_result_v1 result={sizeof(result)};
    bool failed=false;std::uint64_t batches=0,calls=0;
    // The CPU input-coverage model answers rare native form hit tests, yet it
    // must observe every native UI command in order. Building it on the game
    // thread cost about a third of Civ III's UI thread on busy maps (thousands
    // of commands per second, fullscreen transfers splitting into hundreds of
    // regions). One ordered worker applies the identical model; a query waits
    // for every earlier command. A bounded-model refusal skips only that
    // command's coverage, as the inline model did, and is reported once.
    class HitWorker {
        struct Operation {unsigned kind=0;Id id=0;unsigned width=0,height=0;Format format=Format::rgb555;
            std::vector<unsigned> pixels;Command command{};};
        c3x_native_hit::Scene scene;
        std::mutex mutex;std::condition_variable wake,idle,drained;
        std::deque<Operation> queue;bool busy=false,stopping=false;std::uint64_t failures=0;
        // Bounded backlog: a hit query waits for every earlier command, so an
        // unbounded queue let Civ III's end-of-load UI burst stall the first
        // hover/zoom query for ~7 s. Past the bound the producer waits until
        // the backlog halves, which costs what inline processing did.
        // C3X_RENDERER_HIT_BACKLOG overrides the operation bound for measurement.
        std::size_t backlog_operations=512;static constexpr std::size_t backlog_bytes=64u*1024u*1024u;
        // Caller-thread staging: one lock and wake per batch instead of per
        // native command (thousands per second while scrolling).
        std::vector<Operation> staged;static constexpr std::size_t stage_operations=64;
        // Caller thread: canvases Civ III's form hit test never reads. Their
        // draws (most of the busy-map stream) never reach the worker.
        std::unordered_set<Id> exempted;
        std::size_t queued_bytes=0;
        std::thread thread;
        // Diagnostic stream for offline replay (C3X_RENDERER_HIT_TRACE=1 with a
        // trace file): every applied operation and its measured apply time,
        // and every query with its answer.
        std::FILE* dump=nullptr;std::uint64_t dump_bytes=0;
        void record(Operation const& op,std::uint64_t began,std::uint32_t micros){
            if(!dump||dump_bytes>(1536ull<<20))return;
            auto put=[&](auto value){std::fwrite(&value,sizeof(value),1,dump);dump_bytes+=sizeof(value);};
            put(std::uint32_t(op.kind));put(began);put(micros);
            if(op.kind==0){put(std::uint64_t(op.id));put(std::uint32_t(op.width));put(std::uint32_t(op.height));put(std::uint32_t(op.format));}
            else if(op.kind==1||op.kind==5)put(std::uint64_t(op.id));
            else if(op.kind==2){put(std::uint64_t(op.id));put(std::uint32_t(op.pixels.size()));
                std::fwrite(op.pixels.data(),sizeof(unsigned),op.pixels.size(),dump);dump_bytes+=op.pixels.size()*sizeof(unsigned);}
            else{auto const& c=op.command;
                put(std::uint32_t(c.kind));put(std::uint64_t(c.destination));put(std::uint64_t(c.source));
                for(int v:{c.area.left,c.area.top,c.area.right,c.area.bottom,c.clip.left,c.clip.top,c.clip.right,c.clip.bottom,c.source_x,c.source_y})put(std::int32_t(v));
                put(std::uint32_t(c.color));put(std::uint64_t(c.background));put(std::uint64_t(c.detail));put(std::uint64_t(c.background_detail));
                put(std::int32_t(c.source_width));put(std::int32_t(c.source_height));put(std::uint64_t(c.program));}
        }
        void apply(Operation& op){
            if(op.kind==0)scene.create(op.id,op.width,op.height,op.format);
            else if(op.kind==1)scene.destroy(op.id);
            else if(op.kind==2)scene.upload(op.id,op.pixels.data(),op.pixels.size());
            else if(op.kind==5)scene.exempt(op.id);
            else scene.submit(op.command);
        }
        void run(){
            std::unique_lock<std::mutex> lock(mutex);
            for(;;){
                wake.wait(lock,[&]{return stopping||!queue.empty();});
                if(queue.empty()){if(stopping)return;continue;}
                auto op=std::move(queue.front());queue.pop_front();busy=true;
                queued_bytes-=op.pixels.size()*sizeof(unsigned);
                if(queue.size()<=backlog_operations/2&&queued_bytes<=backlog_bytes/2)drained.notify_all();
                lock.unlock();
                std::string error;auto applying=std::chrono::steady_clock::now();
                try{apply(op);}catch(std::exception const& e){error=e.what();}catch(...){error="native input coverage failure";}
                if(dump)record(op,std::uint64_t(applying.time_since_epoch().count()),std::uint32_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-applying).count()));
                if(!error.empty()&&failures<4){char line[320];
                    std::snprintf(line,sizeof(line),"[C3X renderer] stage=native-hit-scene-refused operation=%u detail=%s\n",op.kind,error.c_str());
#ifdef _WIN32
                    OutputDebugStringA(line);
#else
                    std::fputs(line,stderr);
#endif
                }
                lock.lock();busy=false;failures+=!error.empty();
                if(queue.empty())idle.notify_all();
            }
        }
        void push(Operation&& op){staged.push_back(std::move(op));if(staged.size()>=stage_operations)publish();}
    public:
        void publish(){
            if(staged.empty())return;
            std::unique_lock<std::mutex> lock(mutex);
            for(auto& op:staged){queued_bytes+=op.pixels.size()*sizeof(unsigned);queue.push_back(std::move(op));}
            staged.clear();wake.notify_one();
            if(queue.size()>backlog_operations||queued_bytes>backlog_bytes){
                auto began=std::chrono::steady_clock::now();
                drained.wait(lock,[&]{return stopping||(queue.size()<=backlog_operations/2&&queued_bytes<=backlog_bytes/2);});
                profile.add(profile.backlog,began);
            }
        }
        HitWorker(){
#ifdef _WIN32
            char option[4]={},path[MAX_PATH]={};
            char backlog[16]={};
            if(GetEnvironmentVariableA("C3X_RENDERER_HIT_BACKLOG",backlog,sizeof(backlog))>0){
                auto value=std::strtoul(backlog,nullptr,10);if(value>=64&&value<=65536)backlog_operations=value;}
            if(GetEnvironmentVariableA("C3X_RENDERER_HIT_TRACE",option,sizeof(option))==1&&option[0]=='1'){
                auto length=GetEnvironmentVariableA("C3X_RENDERER_TRACE_FILE",path,MAX_PATH-8);
                if(length>0&&length<MAX_PATH-8){strcat_s(path,".hit");if(fopen_s(&dump,path,"wb"))dump=nullptr;}
            }
#endif
            thread=std::thread([this]{run();});
        }
        ~HitWorker(){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}wake.notify_all();drained.notify_all();thread.join();if(dump)std::fclose(dump);}
        HitWorker(HitWorker const&)=delete;HitWorker& operator=(HitWorker const&)=delete;
        void create(Id id,unsigned width,unsigned height,Format format){exempted.erase(id);Operation op;op.kind=0;op.id=id;op.width=width;op.height=height;op.format=format;push(std::move(op));}
        void destroy(Id id){exempted.erase(id);Operation op;op.kind=1;op.id=id;push(std::move(op));}
        void exempt(Id id){if(!id||!exempted.insert(id).second)return;Operation op;op.kind=5;op.id=id;push(std::move(op));}
        void upload(Id id,std::uint32_t const* pixels,std::size_t count){if(exempted.count(id))return;Operation op;op.kind=2;op.id=id;op.pixels.assign(pixels,pixels+count);push(std::move(op));}
        void submit(Command const& command){if(exempted.count(command.destination))return;Operation op;op.kind=3;op.command=command;push(std::move(op));}
        bool pixel(Id id,int x,int y,unsigned& value){
            auto began=std::chrono::steady_clock::now();publish();
            std::unique_lock<std::mutex> lock(mutex);
            idle.wait(lock,[&]{return queue.empty()&&!busy;});
            bool found=scene.pixel(id,x,y,value); // the worker cannot dequeue while this lock is held
            if(dump&&dump_bytes<=(1536ull<<20)){ // query record (kind 4); the worker is idle under this lock
                auto put=[&](auto v){std::fwrite(&v,sizeof(v),1,dump);dump_bytes+=sizeof(v);};
                put(std::uint32_t(4));put(std::uint64_t(began.time_since_epoch().count()));put(std::uint32_t(0));
                put(std::uint64_t(id));put(std::int32_t(x));put(std::int32_t(y));put(std::uint32_t(found));put(std::uint32_t(value));}
            profile.add(profile.query,began);return found;
        }
        // Game-thread waits, reported with the client's call profile.
        struct Wait {double ms=0,max=0;std::uint64_t count=0;};
        struct Profile {Wait backlog,query;
            static void add(Wait& wait,std::chrono::steady_clock::time_point began){
                double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-began).count();
                wait.ms+=ms;wait.max=std::max(wait.max,ms);++wait.count;}
        } profile;
        Profile take(){std::lock_guard<std::mutex> lock(mutex);auto value=profile;profile={};return value;}
    };
    std::unique_ptr<HitWorker> hit_scene;
    // Rate-limited game-thread wait summary, enabled with the input-cost
    // diagnostic switch: helper transport calls and input-coverage waits.
    struct CallProfile {
        bool enabled=false;std::chrono::steady_clock::time_point reported=std::chrono::steady_clock::now();
        HitWorker::Wait execute;
        CallProfile(){
#ifdef _WIN32
            char value[4]={};enabled=GetEnvironmentVariableA("C3X_RENDERER_TRACE_INPUT",value,sizeof(value))==1&&value[0]=='1';
#else
            auto value=std::getenv("C3X_RENDERER_TRACE_INPUT");enabled=value&&value[0]=='1';
#endif
        }
    } call_profile;
    void report_waits(){
        auto now=std::chrono::steady_clock::now();
        if(!call_profile.enabled||now-call_profile.reported<std::chrono::seconds(2))return;
        auto hit=hit_scene?hit_scene->take():HitWorker::Profile{};
        char line[384];std::snprintf(line,sizeof(line),
            "[C3X renderer] stage=native-call-waits calls=%llu execute_ms=%.1f execute_max_ms=%.2f backlog_waits=%llu backlog_ms=%.1f backlog_max_ms=%.2f queries=%llu query_ms=%.1f query_max_ms=%.2f window_ms=%.0f\n",
            (unsigned long long)call_profile.execute.count,call_profile.execute.ms,call_profile.execute.max,
            (unsigned long long)hit.backlog.count,hit.backlog.ms,hit.backlog.max,(unsigned long long)hit.query.count,hit.query.ms,hit.query.max,
            std::chrono::duration<double,std::milli>(now-call_profile.reported).count());
#ifdef _WIN32
        OutputDebugStringA(line);
#else
        std::fputs(line,stderr);
#endif
        call_profile.execute={};call_profile.reported=now;
    }
    c3x_renderer_gpu_images_v1 request(int action,Id image=0)const{
        c3x_renderer_gpu_images_v1 r={};r.struct_size=sizeof(r);r.action=action;r.ticket=ticket;r.image=std::int64_t(image);return r;
    }
    bool run(c3x_renderer_gpu_images_v1 const& r,unsigned* pixels=nullptr,unsigned count=0,bool admission=false){
        if(failed)throw std::runtime_error("GPU image session is no longer usable");
        ++calls;
        auto packet=r;bool prelude=packet.action!=C3X_GPU_SUBMIT && !pending.empty();
        if(prelude){packet.commands=pending.data();packet.command_count=unsigned(pending.size());packet.command_struct_size=sizeof(pending[0]);}
        c3x_renderer_gpu_result_v1 next={sizeof(next)};auto began=std::chrono::steady_clock::now();
        auto code=execute(&packet,&next,pixels,count);
        if(hit_scene)hit_scene->publish(); // with each helper batch boundary
        if(call_profile.enabled){HitWorker::Profile::add(call_profile.execute,began);report_waits();}
        // The worker executes the draw prelude before the resource operation,
        // including a create admission refusal. Never replay those draws.
        if(prelude && (code==C3X_RENDERER_RESULT_OK || (admission && code==C3X_RENDERER_RESULT_BAD_ARGUMENT))){pending.clear();++batches;}
        if(admission&&code==C3X_RENDERER_RESULT_BAD_ARGUMENT)return false;
        if(code!=C3X_RENDERER_RESULT_OK){
            auto kind=r.command_count?r.commands[0].kind:-1;
            failed=true;pending.clear();throw std::runtime_error("GPU image packet failed: action="+std::to_string(r.action)+" result="+std::to_string(code)+" commands="+std::to_string(r.command_count)+" first_kind="+std::to_string(kind)+"; current composition must not be published");
        }
        result=next;return true;
    }
public:
    WorkerClient(c3x_renderer_gpu_images_fn fn,c3x_renderer_gpu_frame_v1 const& frame,bool input_coverage=false):execute(fn),ticket(frame.ticket),session(frame.session){
        if(!fn||ticket<=0||session<=0||frame.struct_size!=sizeof(frame))throw std::runtime_error("missing GPU image session");pending.reserve(2048);
        if(input_coverage)hit_scene=std::make_unique<HitWorker>();
    }
    void hit_exempt(Id id){if(hit_scene)hit_scene->exempt(id);}
    bool hit_pixel(Id id,int x,int y,unsigned& value)const{
        if(!hit_scene||!hit_scene->pixel(id,x,y,value))return false;
        if(value==c3x_native_hit::opaque_map)value=1;return true;
    }
    // The native owner must flush before advancing the map ticket, drain before
    // retiring a session, and never publish after failure. Destruction sends no work.
    bool flushed()const{return pending.empty();}
    void flush(){if(pending.empty())return;auto r=request(C3X_GPU_SUBMIT);r.commands=pending.data();r.command_count=unsigned(pending.size());r.command_struct_size=sizeof(pending[0]);run(r);pending.clear();++batches;}
    void advance(c3x_renderer_gpu_frame_v1 const& next){
        if(failed||!pending.empty()||next.struct_size!=sizeof(next)||next.session!=session||next.ticket<=ticket)throw std::runtime_error("GPU ticket advance requires a flushed live session");
        ticket=next.ticket;
    }
    Id create(unsigned width,unsigned height,Format format){
        if(format!=Format::rgb555&&format!=Format::rgb565&&format!=Format::bgra32)return 0;
        // Invalid dimensions are rejected before a compound packet reaches
        // the ABI validator, which cannot execute its draw prelude. Preserve
        // the old flush-before-admission behavior for that rejected request.
        if(!width || !height || width>2240 || height>1260){flush();return 0;}
        auto r=request(C3X_GPU_CREATE);r.width=int(width);r.height=int(height);
        r.format=format==Format::rgb555?C3X_GPU_RGB555:format==Format::rgb565?C3X_GPU_RGB565:C3X_GPU_BGRA32;
        if(!run(r,nullptr,0,true))return 0;
        auto id=Id(result.image);if(hit_scene)hit_scene->create(id,width,height,format);return id;
    }
    bool destroy(Id id){if(failed)return false;run(request(C3X_GPU_DESTROY,id));if(hit_scene)hit_scene->destroy(id);return true;}
    bool upload(Id id,std::uint64_t revision,std::uint32_t const* pixels,std::size_t count){
        if(count>2240u*1260u)return false;auto r=request(C3X_GPU_UPLOAD,id);r.revision=std::int64_t(revision);r.pixels=pixels;r.pixel_count=unsigned(count);run(r);
        if(hit_scene)hit_scene->upload(id,pixels,count);
        return true;
    }
    bool submit(Command const* commands,std::size_t count){
        if(failed)throw std::runtime_error("GPU image session is no longer usable");
        if(!commands||!count||count>2048)return false;
        if(count+pending.size()>2048)flush();
        for(std::size_t n=0;n<count;++n){auto const& c=commands[n];
            if(hit_scene&&c.kind<Kind::world_begin)hit_scene->submit(c);
            pending.push_back({int(c.kind),std::int64_t(c.destination),std::int64_t(c.source),
            {c.area.left,c.area.top,c.area.right,c.area.bottom},{c.clip.left,c.clip.top,c.clip.right,c.clip.bottom},c.source_x,c.source_y,c.color,std::int64_t(c.background),std::int64_t(c.detail),std::int64_t(c.background_detail),c.source_width,c.source_height,std::int64_t(c.program)});}
        return true;
    }
    bool readback(Id id,std::uint32_t* pixels,std::size_t count){
        if(!pixels||!count||count>2240u*1260u)return false;auto r=request(C3X_GPU_READBACK,id);r.pixel_count=unsigned(count);run(r,pixels,unsigned(count));return result.pixel_count==count;
    }
    c3x_renderer_gpu_result_v1 stats()const{return result;}
    std::uint64_t submitted_batches()const{return batches;}
    std::uint64_t worker_calls()const{return calls;}
};
}
