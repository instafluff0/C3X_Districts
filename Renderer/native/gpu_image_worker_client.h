#pragma once
#include "gpu_frame_api.h"
#include "gpu_image_commands.h"
#include <vector>
#include <stdexcept>
#include <algorithm>
namespace c3x_gpu_images {
// Caller-thread transport only. No native pointers, leases or D3D objects enter
// packets. Consecutive draws coalesce until a resource/CPU/frame boundary.
class WorkerClient {
    c3x_renderer_gpu_images_fn execute;std::int64_t ticket,session;
    std::vector<c3x_renderer_gpu_command_v1> pending;
    c3x_renderer_gpu_result_v1 result={sizeof(result)};
    bool failed=false;std::uint64_t batches=0,calls=0;
    c3x_renderer_gpu_images_v1 request(int action,Id image=0)const{
        c3x_renderer_gpu_images_v1 r={};r.struct_size=sizeof(r);r.action=action;r.ticket=ticket;r.image=std::int64_t(image);return r;
    }
    bool run(c3x_renderer_gpu_images_v1 const& r,unsigned* pixels=nullptr,unsigned count=0,bool admission=false){
        if(failed)throw std::runtime_error("GPU image session is no longer usable");
        ++calls;
        c3x_renderer_gpu_result_v1 next={sizeof(next)};auto code=execute(&r,&next,pixels,count);
        if(admission&&code==C3X_RENDERER_RESULT_BAD_ARGUMENT)return false;
        if(code!=C3X_RENDERER_RESULT_OK){
            auto kind=r.command_count?r.commands[0].kind:-1;
            failed=true;pending.clear();throw std::runtime_error("GPU image packet failed: action="+std::to_string(r.action)+" result="+std::to_string(code)+" commands="+std::to_string(r.command_count)+" first_kind="+std::to_string(kind)+"; current composition must not be published");
        }
        result=next;return true;
    }
public:
    WorkerClient(c3x_renderer_gpu_images_fn fn,c3x_renderer_gpu_frame_v1 const& frame):execute(fn),ticket(frame.ticket),session(frame.session){
        if(!fn||ticket<=0||session<=0||frame.struct_size!=sizeof(frame))throw std::runtime_error("missing GPU image session");pending.reserve(2048);
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
        flush();auto r=request(C3X_GPU_CREATE);r.width=int(width);r.height=int(height);
        r.format=format==Format::rgb555?C3X_GPU_RGB555:format==Format::rgb565?C3X_GPU_RGB565:C3X_GPU_BGRA32;return run(r,nullptr,0,true)?Id(result.image):0;
    }
    bool destroy(Id id){if(failed)return false;flush();run(request(C3X_GPU_DESTROY,id));return true;}
    bool upload(Id id,std::uint64_t revision,std::uint32_t const* pixels,std::size_t count){
        if(count>2240u*1192u)return false;flush();auto r=request(C3X_GPU_UPLOAD,id);r.revision=std::int64_t(revision);r.pixels=pixels;r.pixel_count=unsigned(count);run(r);return true;
    }
    bool submit(Command const* commands,std::size_t count){
        if(failed)throw std::runtime_error("GPU image session is no longer usable");
        if(!commands||!count||count>2048)return false;
        if(count+pending.size()>2048)flush();
        for(std::size_t n=0;n<count;++n){auto const& c=commands[n];pending.push_back({int(c.kind),std::int64_t(c.destination),std::int64_t(c.source),
            {c.area.left,c.area.top,c.area.right,c.area.bottom},{c.clip.left,c.clip.top,c.clip.right,c.clip.bottom},c.source_x,c.source_y,c.color,std::int64_t(c.background),std::int64_t(c.detail),std::int64_t(c.background_detail),c.source_width,c.source_height,std::int64_t(c.program)});}
        return true;
    }
    bool readback(Id id,std::uint32_t* pixels,std::size_t count){
        if(!pixels||!count||count>2240u*1192u)return false;flush();auto r=request(C3X_GPU_READBACK,id);r.pixel_count=unsigned(count);run(r,pixels,unsigned(count));return result.pixel_count==count;
    }
    c3x_renderer_gpu_result_v1 stats()const{return result;}
    std::uint64_t submitted_batches()const{return batches;}
    std::uint64_t worker_calls()const{return calls;}
};
}
