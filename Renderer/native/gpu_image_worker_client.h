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
#include <cstdio>
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
        static constexpr std::size_t backlog_operations=256,backlog_bytes=64u*1024u*1024u;
        std::size_t queued_bytes=0;
        std::thread thread;
        void apply(Operation& op){
            if(op.kind==0)scene.create(op.id,op.width,op.height,op.format);
            else if(op.kind==1)scene.destroy(op.id);
            else if(op.kind==2)scene.upload(op.id,op.pixels.data(),op.pixels.size());
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
                std::string error;
                try{apply(op);}catch(std::exception const& e){error=e.what();}catch(...){error="native input coverage failure";}
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
        void push(Operation&& op){
            std::unique_lock<std::mutex> lock(mutex);
            queued_bytes+=op.pixels.size()*sizeof(unsigned);
            queue.push_back(std::move(op));wake.notify_one();
            if(queue.size()>backlog_operations||queued_bytes>backlog_bytes)
                drained.wait(lock,[&]{return stopping||(queue.size()<=backlog_operations/2&&queued_bytes<=backlog_bytes/2);});
        }
    public:
        HitWorker():thread([this]{run();}){}
        ~HitWorker(){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}wake.notify_all();drained.notify_all();thread.join();}
        HitWorker(HitWorker const&)=delete;HitWorker& operator=(HitWorker const&)=delete;
        void create(Id id,unsigned width,unsigned height,Format format){Operation op;op.kind=0;op.id=id;op.width=width;op.height=height;op.format=format;push(std::move(op));}
        void destroy(Id id){Operation op;op.kind=1;op.id=id;push(std::move(op));}
        void upload(Id id,std::uint32_t const* pixels,std::size_t count){Operation op;op.kind=2;op.id=id;op.pixels.assign(pixels,pixels+count);push(std::move(op));}
        void submit(Command const& command){Operation op;op.kind=3;op.command=command;push(std::move(op));}
        bool pixel(Id id,int x,int y,unsigned& value){
            std::unique_lock<std::mutex> lock(mutex);
            idle.wait(lock,[&]{return queue.empty()&&!busy;});
            return scene.pixel(id,x,y,value); // the worker cannot dequeue while this lock is held
        }
    };
    std::unique_ptr<HitWorker> hit_scene;
    c3x_renderer_gpu_images_v1 request(int action,Id image=0)const{
        c3x_renderer_gpu_images_v1 r={};r.struct_size=sizeof(r);r.action=action;r.ticket=ticket;r.image=std::int64_t(image);return r;
    }
    bool run(c3x_renderer_gpu_images_v1 const& r,unsigned* pixels=nullptr,unsigned count=0,bool admission=false){
        if(failed)throw std::runtime_error("GPU image session is no longer usable");
        ++calls;
        auto packet=r;bool prelude=packet.action!=C3X_GPU_SUBMIT && !pending.empty();
        if(prelude){packet.commands=pending.data();packet.command_count=unsigned(pending.size());packet.command_struct_size=sizeof(pending[0]);}
        c3x_renderer_gpu_result_v1 next={sizeof(next)};auto code=execute(&packet,&next,pixels,count);
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
