"""Host execution of bounded image-prefix timing and helper classification."""
import os
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


@unittest.skipUnless(os.name == "posix", "host-only C++ stubs")
class HelperCadenceCountersTests(unittest.TestCase):
    def test_batch_counters_distinguish_execution_from_receipt_retirement(self):
        run_cpp(r'''
#include "Renderer/native/ordered_image_batch.h"
#include <cassert>
#include <future>
using namespace c3x_remote_scene;
using namespace std::chrono_literals;
int main(){
 std::promise<void> entered,release,ready;auto held=release.get_future();
 ImageBatchService batch([&](auto&){entered.set_value();held.wait();
  return std::vector<ImageBatch::Reply>{{C3X_RENDERER_RESULT_DEVICE_ERROR,{}}};
 },[&]{ready.set_value();});
 std::vector<ImageBatch::Operation> work(1);work[0].image.value.action=C3X_GPU_UPLOAD;
 assert(batch.admit(7,100,work));entered.get_future().get();
 auto active=batch.status();assert(active.bytes==100&&!active.ready);
 assert(active.timing.admitted==1&&active.timing.started==1);
 assert(active.timing.completed==0&&active.timing.retired==0);
 assert(!batch.admit(8,100,work));
 std::vector<ImageBatch::Reply> replies;std::string error;double elapsed=0;
 assert(!batch.poll(7,replies,error,elapsed));
 std::this_thread::sleep_for(2ms);release.set_value();ready.get_future().get();
 auto completed=batch.status();assert(completed.ready&&completed.bytes==100);
 assert(completed.timing.admitted==1&&completed.timing.completed==1&&completed.timing.retired==0);
 assert(completed.timing.execution.total_ns>0&&completed.timing.execution.max_ns==completed.timing.execution.total_ns);
 assert(completed.timing.ready_to_retirement.total_ns==0);
 bool wrong=false;try{batch.poll(8,replies,error,elapsed);}catch(std::exception const&){wrong=true;}
 assert(wrong&&batch.status().timing.retired==0);
 std::this_thread::sleep_for(2ms);assert(batch.poll(7,replies,error,elapsed));
 auto retired=batch.status();assert(!retired.bytes&&!retired.ready);
 assert(retired.timing.retired==1&&retired.timing.ready_to_retirement.total_ns>0);
 assert(retired.timing.ready_to_retirement.max_ns==retired.timing.ready_to_retirement.total_ns);
 assert(replies.size()==1&&replies[0].code==C3X_RENDERER_RESULT_DEVICE_ERROR&&error.empty());
}
''')

    def test_actual_offer_classifies_each_attempt_without_trace_zero_io(self):
        source = (ROOT / "Renderer/native/helper_trial/scene_workload.cpp").read_text()
        counters = "    struct CadenceCounters {" + source.split("    struct CadenceCounters {", 1)[1].split("    using Definitions=", 1)[0]
        report = "    void report_direct_cadence(" + source.split("    void report_direct_cadence(", 1)[1].split("    void start_direct_cadence()", 1)[0]
        body = "    void start_direct_cadence(){" + source.split("    void start_direct_cadence(){", 1)[1].split("    void stop_direct_cadence()", 1)[0]
        run_cpp(r'''
#include "Renderer/native/ordered_image_batch.h"
#include <cassert>
#include <cstdio>
using LONG=long;struct LARGE_INTEGER{long long QuadPart=0;};
long long ticks=1000;unsigned rows=0,errors=0;std::string last;
bool QueryPerformanceCounter(LARGE_INTEGER* v){v->QuadPart=ticks;return true;}
bool QueryPerformanceFrequency(LARGE_INTEGER* v){v->QuadPart=1000;return true;}
unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
void InterlockedExchange(volatile LONG* p,LONG v){*p=v;}
void InterlockedIncrement(volatile LONG* p){++*p;}
LONG InterlockedCompareExchange(volatile LONG* p,LONG v,LONG expected){auto old=*p;if(old==expected)*p=v;return old;}
void OutputDebugStringA(char const* s){last=s;
 if(last.find("stage=helper-cadence-summary")!=std::string::npos)++rows;else ++errors;}
struct Owner{
 bool direct_surface_bound=true,direct_display_ready=true;long long pressure_present_ticks=0;
 struct{std::function<bool()> offer;bool stopped=false;void disable(){stopped=true;}
  template<class F>void enable_retrying(F f){offer=f;}}direct_cadence;
 struct Batch{std::size_t bytes=0;bool ready=false;
  c3x_remote_scene::ImageBatchService::Status status(){return {1,bytes,0,0,ready,0,{}};}}batch;
 Batch* image_batches=&batch;
 struct{LONG presented_zoom_q16=0,visual_frames=0,native_queue_records=0;}values;
 decltype(values)* telemetry=&values;
 int code=C3X_RENDERER_RESULT_OK;
 std::function<int(long long,long long,unsigned,std::uint64_t*,unsigned*,unsigned*)> visual_shared;
 std::function<int(int,void*,void*,void*,void*,unsigned)> native_image;
''' + counters + report + body + r'''
};
int main(){Owner o;o.visual_shared=[&](auto...){return o.code;};
 o.native_image=[](auto...){return 65536;};o.start_direct_cadence();
 o.batch.bytes=1;assert(o.direct_cadence.offer());o.batch.ready=true;assert(o.direct_cadence.offer());
 o.batch.bytes=0;assert(!o.direct_cadence.offer());o.values.native_queue_records=512;++ticks;
 assert(o.direct_cadence.offer());o.values.native_queue_records=0;
 o.code=C3X_RENDERER_RESULT_BUSY;assert(o.direct_cadence.offer());
 o.code=C3X_RENDERER_RESULT_PENDING;assert(!o.direct_cadence.offer());
 o.code=C3X_RENDERER_RESULT_ERROR;assert(!o.direct_cadence.offer());
 auto c=o.cadence_counts;assert(c.attempts==7&&c.prefix_active==1&&c.prefix_ready_unretired==1);
 assert(c.pressure_holds==1&&c.dll_busy==1&&c.pending==1&&c.presented==1&&c.errors==1);
 assert(c.clock_failures==0&&o.values.visual_frames==1&&o.direct_cadence.stopped&&!rows&&errors==1);
 o.cadence_diagnostics=true;o.report_direct_cadence(o.batch.status());assert(rows==1);
 assert(last.find("stage=helper-cadence-summary")!=std::string::npos);
 assert(last.find("attempts=7")!=std::string::npos&&last.find("prefix_ready_unretired=1")!=std::string::npos);
 o.report_direct_cadence(o.batch.status());assert(rows==1);
}
''')


if __name__ == "__main__":
    unittest.main()
