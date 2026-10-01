#pragma once
#include "input_recording/codec.h"
#include <map>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace c3x_remote_scene {
// A batch preserves every resource operation and image-version boundary. Only
// consecutive image requests can join; camera, unit and present calls fence it.
struct ImageBatch {
    struct Operation {c3x_inputs::Images image;c3x_renderer_i64 created=0;};
    struct Reply {int code=0;c3x_renderer_gpu_result_v1 value={sizeof(value)};};
    static constexpr unsigned operation_limit=64,work_limit=4096;
    static constexpr std::size_t join_bytes=1024u*1024u,payload_limit=16u*1024u*1024u;
    std::vector<Operation> operations;
    static unsigned work(c3x_renderer_gpu_images_v1 const& image){
        return 1u+image.command_count+(image.pixel_count+65535u)/65536u;
    }
    static void encode(c3x_inputs::Writer& out,std::vector<Operation> const& values){
        c3x_inputs::require(!values.empty()&&values.size()<=operation_limit,"image batch operation bound");
        out.u32(unsigned(values.size()));unsigned units=0;
        for(auto const& value:values){out(value.created);c3x_inputs::images(out,value.image.value);
            units+=work(value.image.value);}
        c3x_inputs::require(units<=work_limit&&out.bytes.size()<=payload_limit,"image batch work/byte bound");
    }
    static std::vector<Operation> decode(c3x_inputs::Reader& in){
        unsigned count=0;in(count);c3x_inputs::require(count&&count<=operation_limit,"image batch count");
        std::vector<Operation> result(count);unsigned units=0;
        for(auto& value:result){in(value.created);c3x_inputs::images(in,value.image);
            units+=work(value.image.value);
            c3x_inputs::require(value.image.value.action!=C3X_GPU_READBACK&&value.created>=0&&
                (!value.created||value.image.value.action==C3X_GPU_CREATE),"image batch reliable operation");}
        c3x_inputs::require(units<=work_limit,"image batch semantic work bound");in.done();return result;
    }
    template<class IO>static void reply_fields(IO& io,Reply& reply){
        io(reply.code);auto& v=reply.value;io(v.image);io(v.pixel_count);
        io(v.resident_bytes);io(v.uploads);io(v.commands);io(v.readbacks);
    }
    // Negative IDs name resources created earlier in this same batch. External
    // resources are resolved by the bridge's executed prefix before admission.
    // A failure returns its exact prefix; it never acknowledges the suffix.
    template<class Execute>static std::vector<Reply> execute(std::vector<Operation>& values,Execute invoke){
        std::map<c3x_renderer_i64,c3x_renderer_i64> aliases;std::vector<Reply> replies;
        auto image=[&](c3x_renderer_i64 id){if(id>=0)return id;
            auto at=aliases.find(-id);c3x_inputs::require(at!=aliases.end(),"image batch alias outside lifetime");return at->second;};
        for(auto& op:values){auto& v=op.image.value;v.image=image(v.image);
            for(auto& draw:op.image.commands){draw.destination=image(draw.destination);draw.source=image(draw.source);
                draw.background=image(draw.background);draw.detail=image(draw.detail);draw.background_detail=image(draw.background_detail);draw.program=image(draw.program);}
            op.image.bind();Reply reply;reply.code=invoke(v,reply.value);replies.push_back(reply);
            if(reply.code!=C3X_RENDERER_RESULT_OK)break;
            if(op.created)c3x_inputs::require(aliases.emplace(op.created,reply.value.image).second,"image batch duplicate create");
            if(v.action==C3X_GPU_DESTROY)for(auto at=aliases.begin();at!=aliases.end();++at)
                if(at->second==v.image){aliases.erase(at);break;}
        }
        return replies;
    }
};
}

// One bounded admitted batch. The receipt thread can answer progress/cancel
// requests while its executor waits for a renderer-owner preparation checkpoint.
// No second context or presenter is created, and the next reliable prefix waits
// for this batch's explicit execution receipt.
namespace c3x_remote_scene {
class ImageBatchService {
    using Clock=std::chrono::steady_clock;
    std::mutex mutex;std::condition_variable wake;
    std::vector<ImageBatch::Operation> pending;std::vector<ImageBatch::Reply> result;
    std::function<std::vector<ImageBatch::Reply>(std::vector<ImageBatch::Operation>&)> execute;
    std::function<void()> notify;
    bool stopping=false,occupied=false,ready=false;
    unsigned sequence=0;std::size_t bytes=0,units=0,operations=0;
    std::string error;double service_ms=0;std::thread worker;
    void run(){std::unique_lock<std::mutex> lock(mutex);
        for(;;){wake.wait(lock,[&]{return stopping||(!ready&&occupied);});
            if(stopping&&(!occupied||ready))return;
            auto work=std::move(pending);lock.unlock();auto begin=Clock::now();
            std::vector<ImageBatch::Reply> replies;std::string failure;
            try{replies=execute(work);}catch(std::exception const& e){failure=e.what();}
            catch(...){failure="unknown image batch execution failure";}
            auto elapsed=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            work.clear();lock.lock();result=std::move(replies);error=std::move(failure);service_ms=elapsed;ready=true;
            lock.unlock();if(notify)notify();lock.lock();
            if(stopping)return;
        }
    }
public:
    struct Status {unsigned sequence;std::size_t bytes,units,operations;bool ready;double service_ms;};
    template<class Execute>explicit ImageBatchService(Execute value,std::function<void()> complete={}):
        execute(std::move(value)),notify(std::move(complete)),worker([this]{run();}){}
    ~ImageBatchService(){{std::lock_guard<std::mutex> lock(mutex);stopping=true;}wake.notify_all();worker.join();}
    bool admit(unsigned id,std::size_t size,std::vector<ImageBatch::Operation> work){
        std::size_t semantic=0;for(auto const& op:work)semantic+=ImageBatch::work(op.image.value);
        c3x_inputs::require(size<=ImageBatch::payload_limit&&work.size()<=ImageBatch::operation_limit&&semantic<=ImageBatch::work_limit,"image admission capacity");
        std::lock_guard<std::mutex> lock(mutex);if(stopping||occupied)return false;
        sequence=id;bytes=size;units=semantic;operations=work.size();pending=std::move(work);result.clear();error.clear();
        occupied=true;ready=false;wake.notify_one();return true;
    }
    bool poll(unsigned id,std::vector<ImageBatch::Reply>& replies,std::string& failure,double& elapsed){
        std::lock_guard<std::mutex> lock(mutex);
        c3x_inputs::require(occupied&&sequence==id,"image execution receipt outside admission");
        if(!ready)return false;
        replies=std::move(result);failure=std::move(error);elapsed=service_ms;
        occupied=false;ready=false;bytes=units=operations=0;return true;
    }
    Status status(){std::lock_guard<std::mutex> lock(mutex);return {sequence,bytes,units,operations,ready,service_ms};}
};
}
