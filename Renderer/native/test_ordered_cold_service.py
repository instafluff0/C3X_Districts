"""Production batching, pressure recovery and cooperative cold service contracts."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class OrderedColdServiceTests(unittest.TestCase):
    def test_helper_cadence_waits_for_the_whole_native_batch_receipt(self):
        source=(ROOT/'Renderer/native/helper_trial/scene_workload.cpp').read_text()
        body='    void start_direct_cadence(){'+source.split('    void start_direct_cadence(){',1)[1].split('    void stop_direct_cadence()',1)[0]
        counters='    struct CadenceCounters {'+source.split('    struct CadenceCounters {',1)[1].split('    using Definitions=',1)[0]
        report='    void report_direct_cadence('+source.split('    void report_direct_cadence(',1)[1].split('    void start_direct_cadence()',1)[0]
        run_cpp(r'''
#include "Renderer/native/ordered_image_batch.h"
#include <cassert>
#include <functional>
#include <cstdint>
#include <cstdio>
using LONG=long;struct LARGE_INTEGER{long long QuadPart=0;};
long long ticks=123;bool QueryPerformanceCounter(LARGE_INTEGER* value){value->QuadPart=ticks;return true;}
bool QueryPerformanceFrequency(LARGE_INTEGER* value){value->QuadPart=1000;return true;}
unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
void InterlockedExchange(volatile LONG* p,LONG v){*p=v;}void InterlockedIncrement(volatile LONG* p){++*p;}
LONG InterlockedCompareExchange(volatile LONG* p,LONG v,LONG match){auto old=*p;if(old==match)*p=v;return old;}
void OutputDebugStringA(char const*){}
struct Owner{
 bool direct_surface_bound=true,direct_display_ready=true;
 long long pressure_present_ticks=0;int (*priority_front_pending)()=nullptr;
 struct{std::function<bool()> offer;bool stopped=false;void disable(){stopped=true;}
  template<class F>void enable_retrying(F f){offer=f;}}direct_cadence;
 struct Batch{std::size_t bytes=0;bool ready=false;
  c3x_remote_scene::ImageBatchService::Status status(){return {1,bytes,0,0,ready,0,{}};}}batch;
 Batch* image_batches=&batch;
 struct{LONG presented_zoom_q16=0,presented_pan=0,visual_frames=0,native_queue_records=0;}values;
 decltype(values)* telemetry=&values;
 std::function<int(long long,long long,unsigned,std::uint64_t*,unsigned*,unsigned*)> visual_shared;
 std::function<int(int,void*,void*,void*,void*,unsigned)> native_image;
'''+counters+report+body+r'''
};
int main(){Owner owner;unsigned samples=0;
 owner.visual_shared=[&](auto...){++samples;return C3X_RENDERER_RESULT_OK;};
 owner.native_image=[](auto...){return 65536;};owner.start_direct_cadence();
 // Individual renderer calls may finish, but the admitted batch still owns
 // its reliable suffix. Thousands of cadence ticks do not enter that suffix.
 owner.batch.bytes=100;for(unsigned n=0;n<1000;++n)assert(owner.direct_cadence.offer());
 assert(!samples&&!owner.values.visual_frames);
 owner.batch.bytes=0;assert(!owner.direct_cadence.offer());
 assert(samples==1&&owner.values.visual_frames==1&&owner.values.presented_zoom_q16==65536);
 owner.values.native_queue_records=512;ticks+=249;
 assert(owner.direct_cadence.offer()&&samples==1); // reliable prefix has priority
 ++ticks;assert(!owner.direct_cadence.offer()&&samples==2); // periodic UI/front refresh
 owner.values.native_queue_records=0;++ticks;
 assert(!owner.direct_cadence.offer()&&samples==3); // normal cadence resumes immediately
 owner.visual_shared=[&](auto...){++samples;return C3X_RENDERER_RESULT_BUSY;};
 assert(owner.direct_cadence.offer()&&samples==4&&owner.values.visual_frames==3);
}
''')

    def test_admission_does_not_wait_for_execution_and_aliases_keep_order(self):
        run_cpp(r'''
#include "Renderer/native/ordered_image_batch.h"
#include <cassert>
#include <future>
using namespace c3x_remote_scene;
using namespace std::chrono_literals;
int main(){
 std::promise<void> entered,release,notified;auto held=release.get_future();
 std::vector<int> order;std::vector<long long> sources;
 ImageBatchService service([&](auto& work){entered.set_value();held.wait();
  return ImageBatch::execute(work,[&](auto const& request,auto& result){
   order.push_back(request.action);if(request.action==C3X_GPU_CREATE)result.image=71;
   if(request.action==C3X_GPU_UPLOAD){assert(request.image==71);assert(request.pixels[0]==123);}
   for(unsigned i=0;i<request.command_count;++i){auto const& draw=request.commands[i];
    assert(draw.destination==71&&draw.source==71);sources.push_back(draw.source);}
   if(request.action==C3X_GPU_DESTROY)assert(request.image==71);
   return C3X_RENDERER_RESULT_OK;
  });
 },[&]{assert(service.status().ready);notified.set_value();});
 std::vector<ImageBatch::Operation> work(4);
 work[0].created=1;work[0].image.value.action=C3X_GPU_CREATE;
 work[1].image.value.action=C3X_GPU_UPLOAD;work[1].image.value.image=-1;
 work[1].image.value.pixel_count=1;work[1].image.pixels={123};
 work[2].image.value.action=C3X_GPU_SUBMIT;work[2].image.commands.resize(1);
 work[2].image.commands[0].destination=work[2].image.commands[0].source=-1;
 work[3].image.value.action=C3X_GPU_DESTROY;work[3].image.value.image=-1;
 for(auto& op:work)op.image.bind();c3x_inputs::Writer wire;ImageBatch::encode(wire,work);
 c3x_inputs::Reader reader{wire.bytes};auto owned=ImageBatch::decode(reader);
 auto begin=std::chrono::steady_clock::now();assert(service.admit(11,wire.bytes.size(),std::move(owned)));
 assert(std::chrono::steady_clock::now()-begin<50ms);entered.get_future().get();
 assert(!service.admit(12,wire.bytes.size(),work));assert(service.status().bytes==wire.bytes.size());
 std::vector<ImageBatch::Reply> replies;std::string error;double ms=0;
 assert(!service.poll(11,replies,error,ms));assert(order.empty());
 std::this_thread::sleep_for(30ms);release.set_value();
 notified.get_future().get();assert(service.poll(11,replies,error,ms));
 assert(error.empty()&&replies.size()==4&&ms>=30);assert(service.status().bytes==0);
 assert(order==std::vector<int>({C3X_GPU_CREATE,C3X_GPU_UPLOAD,C3X_GPU_SUBMIT,C3X_GPU_DESTROY}));
 assert(sources==std::vector<long long>({71}));
 // A failing operation acknowledges the executed prefix, never its suffix.
 auto partial=work;int calls=0;
 auto result=ImageBatch::execute(partial,[&](auto const&,auto& reply){reply.image=71;return ++calls==2?C3X_RENDERER_RESULT_DEVICE_ERROR:C3X_RENDERER_RESULT_OK;});
 assert(calls==2&&result.size()==2&&result.back().code==C3X_RENDERER_RESULT_DEVICE_ERROR);
}
''')

    def test_semantic_budget_survives_batching_and_reset_reports_retired_work(self):
        run_cpp(r'''
#include "Renderer/sandbox/async_publication.h"
#include <cassert>
#include <vector>
using namespace std::chrono_literals;
struct Group {std::vector<int> work;};
int main(){
 std::promise<void> entered,release;auto held=release.get_future();int reports=0,draws=0,generation=1;
 c3x_async::Publication queue([&](char const*){++reports;},1024,8,8);
 assert(queue.post(1,[&]{entered.set_value();held.wait();}));entered.get_future().get();
 auto post=[&](int id){auto group=std::make_shared<Group>();group->work={id};
  return queue.post_group(1,2,2,group,[&](auto const& batch){draws+=int(batch.work.size());},
   [](auto& to,auto& from){to.work.push_back(from.work[0]);},1024,64,64,"images");};
 assert(post(1)&&post(2)&&post(3));auto before=queue.status();
 assert(before.records==4&&before.units==7&&before.peak_units==7);
 auto begin=std::chrono::steady_clock::now();assert(!post(4));
 assert(std::chrono::steady_clock::now()-begin<50ms&&reports==1&&!queue.healthy());
 release.set_value();
 // This models the explicit native/resource/scene retirement boundary. It
 // clears the resource generation; old receipt is never counted as execution.
 assert(queue.reconcile([&]{++generation;return generation;})==2);
 auto reset=queue.status();assert(reset.abandoned==3&&reset.executed==2&&reset.rejected==1);
 assert(reset.accepted==reset.executed+reset.abandoned+reset.superseded);
 assert(draws==0&&queue.healthy());assert(post(5));queue.setup([]{});assert(draws==1);
 // A failed reset leaves the generation unavailable.
 queue.fail("test failure");bool failed=false;
 try{queue.reconcile([]{throw std::runtime_error("reset failed");});}catch(std::exception const&){failed=true;}
 assert(failed&&!queue.healthy());
}
''')

    def test_content_wait_yields_without_joining_under_the_content_mutex(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace std::chrono_literals;
using namespace c3x_renderer::render_core;
struct Result {int id;std::size_t bytes()const{return 1;}};
int main(){
 ContentPreparation<int,int,Result> pool;std::atomic<unsigned> compiled{0};
 pool.configure({{1,1},{2,2},{3,3}},[&](int id,auto const&,unsigned){
  std::this_thread::sleep_for(40ms);++compiled;return std::make_unique<Result>(Result{id});
 },2);pool.resume();int service_turns=0;std::vector<int> chunks;
 auto owner=std::this_thread::get_id();
 for(int id:{1,2,3}){auto result=pool.take(id,false,[&]{
   assert(std::this_thread::get_id()==owner);auto status=pool.statistics();
   (void)status;++service_turns;return false;
  });assert(result&&result->id==id);chunks.push_back(id);}
 assert(service_turns>10&&compiled==3&&chunks==std::vector<int>({1,2,3}));
 assert(pool.statistics().consumed==3&&pool.statistics().cancelled==0);pool.clear();
}
''')

    def test_production_camera_service_keeps_completed_chunks_and_cancels_at_owner(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        body='    void service_camera_preparation() {'+source.split('    void service_camera_preparation() {',1)[1].split('\n    void run() {',1)[0]
        run_cpp(r'''
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>
using c3x_renderer_i64=long long;
struct LARGE_INTEGER {long long QuadPart=0;};void QueryPerformanceCounter(LARGE_INTEGER* v){++v->QuadPart;}
struct c3x_renderer_output_v1 {};
struct Owner {
 enum class Command {none,gpu_images,tactical,reset};
 std::atomic<long long> camera_obsolete_through{0};std::atomic<bool> camera_cancelled{false},foreground_pending{false};
 long long job_camera_ticket=7;std::mutex state_mutex;bool has_job=false;Command job_command=Command::none;
 unsigned long long latest_job_sequence=0,completed_job_sequence=0,camera_service_turns=0;
 int last_job_result=0;std::condition_variable completed,wake;
 struct {unsigned frame_tiles_built=0,frame_tiles_reused=0;struct {double milliseconds(long long){return 0;}void write(char const*,char const*,bool){}}trace;}renderer_state;
 std::vector<int> chunks;int draws=0;
 int execute_command(Command,c3x_renderer_output_v1&){++draws;return 1;}
'''+body+r'''
};
int main(){Owner owner;
 for(int chunk=0;chunk<20;++chunk){owner.chunks.push_back(chunk);owner.renderer_state.frame_tiles_built=unsigned(owner.chunks.size());
  owner.has_job=true;owner.job_command=Owner::Command::gpu_images;owner.foreground_pending=true;++owner.latest_job_sequence;
  owner.service_camera_preparation();assert(!owner.camera_cancelled&&owner.completed_job_sequence==owner.latest_job_sequence);}
 assert(owner.chunks.size()==20&&owner.draws==20&&owner.camera_service_turns==20);
 owner.camera_obsolete_through=6;owner.service_camera_preparation();assert(!owner.camera_cancelled);
 owner.camera_obsolete_through=7;owner.service_camera_preparation();assert(owner.camera_cancelled);
 assert(owner.chunks.size()==20&&owner.draws==20);
 // A real producer offers consecutive operations only after each receipt. A
 // preparation boundary services the already admitted prefix and returns to
 // terrain preparation without waiting for the producer to offer more work.
 Owner burst;std::atomic<bool> entered{false},done{false};
 std::thread producer([&]{for(unsigned i=0;i<20;++i){
  std::unique_lock<std::mutex> lock(burst.state_mutex);burst.job_command=Owner::Command::gpu_images;
  burst.has_job=true;burst.foreground_pending=true;auto sequence=++burst.latest_job_sequence;
  entered=true;burst.wake.notify_one();burst.completed.wait(lock,[&]{return burst.completed_job_sequence==sequence;});
 }done=true;});
 while(!entered)std::this_thread::yield();burst.service_camera_preparation();
 assert(burst.draws>=1&&burst.draws<=8);while(!done){burst.service_camera_preparation();std::this_thread::yield();}
 producer.join();assert(burst.draws==20&&burst.completed_job_sequence==20&&!burst.camera_cancelled);
}
''')


if __name__=='__main__':
    unittest.main()
