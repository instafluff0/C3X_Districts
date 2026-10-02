"""Host execution of bounded image-prefix timing and helper classification."""
import os
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def _helper_offer_contract():
    source = (ROOT / "Renderer/native/helper_trial/scene_workload.cpp").read_text()
    counters = "    struct CadenceCounters {" + source.split("    struct CadenceCounters {", 1)[1].split("    using Definitions=", 1)[0]
    report = "    void report_direct_cadence(" + source.split("    void report_direct_cadence(", 1)[1].split("    void start_direct_cadence()", 1)[0]
    body = "    void start_direct_cadence(){" + source.split("    void start_direct_cadence(){", 1)[1].split("    void stop_direct_cadence()", 1)[0]
    return r'''
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
 std::function<int()> priority_front_pending;
 int code=C3X_RENDERER_RESULT_OK;
 std::function<int(long long,long long,unsigned,std::uint64_t*,unsigned*,unsigned*)> visual_shared;
 std::function<int(int,void*,void*,void*,void*,unsigned)> native_image;
''' + counters + report + body + r'''
};
'''


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
        run_cpp(_helper_offer_contract() + r'''
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

    def test_new_complete_front_gets_priority_without_bypassing_prefix_or_consuming_failed_present(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        query='    int trial_priority_front_pending()const{' + source.split('    int trial_priority_front_pending()const{',1)[1].split('    int trial_bind_surface(',1)[0]
        commit='if(committed)trial_front_pending.store(' + source.split('if(committed)trial_front_pending.store(',1)[1].split(';',1)[0]+';'
        ack='trial_presented_front_revision=renderer_state.gpu_composition->committed_revision();\n                        trial_front_pending.store(false,std::memory_order_release);'
        assert ack in source
        assert source.count('trial_presented_front_revision=0;')==3 # field plus retire and bind
        assert source.count('trial_front_pending.store(false,std::memory_order_release);')==3
        clear='trial_front_pending.store(false,std::memory_order_release);\n        trial_presented_front_revision=0;'
        assert clear in source
        rebind='trial_front_pending.store(false,std::memory_order_release);\n                trial_presented_front_revision=0;'
        assert rebind in source
        # Execute the actual helper callback and actual DLL priority updates in a host stub.
        prefix=_helper_offer_contract()
        worker=r'''
        struct Front {std::uint64_t revision=1; std::uint64_t committed_revision()const{return revision;}};
        struct Worker{
         std::atomic<bool> trial_front_pending{false}; std::uint64_t trial_presented_front_revision=0;
         Front value;struct {Front* gpu_composition=nullptr;}renderer_state;
         Worker(){renderer_state.gpu_composition=&value;}
         QUERY
         void commit(bool committed=true){auto* session=&value; COMMIT}
         void present_success(){ACK}
         void reset(){CLEAR value.revision=0;}
         void rebind(){REBIND}
        };
        '''.replace('QUERY',query).replace('COMMIT',commit).replace('ACK',ack).replace('CLEAR',clear).replace('REBIND',rebind)
        main=r'''
        int main(){Owner o;Worker worker;unsigned calls=0,queries=0;
         o.priority_front_pending=[&]{++queries;return worker.trial_priority_front_pending();};
         o.visual_shared=[&](auto...){++calls;if(o.code==C3X_RENDERER_RESULT_OK)worker.present_success();return o.code;};
         o.native_image=[](auto...){return 65536;};o.start_direct_cadence();
         worker.commit();assert(worker.trial_priority_front_pending());
         assert(!o.direct_cadence.offer());assert(calls==1&&!worker.trial_priority_front_pending());
         o.values.native_queue_records=512;++ticks;
         worker.commit();assert(!worker.trial_priority_front_pending()); // duplicate completed native front
         assert(o.direct_cadence.offer());assert(calls==1); // ordinary pressure throttle remains
         worker.value.revision=2;worker.commit();assert(worker.trial_priority_front_pending());
         assert(!o.direct_cadence.offer());assert(calls==2&&!worker.trial_priority_front_pending());
         ++ticks;worker.value.revision=3;worker.commit();
         o.batch.bytes=1;o.batch.ready=false;auto q=queries;
         assert(o.direct_cadence.offer());assert(calls==2&&queries==q); // active reliable prefix wins
         o.batch.ready=true;assert(o.direct_cadence.offer());assert(calls==2&&queries==q); // ready but not retired also wins
         o.batch.bytes=0;o.code=C3X_RENDERER_RESULT_BUSY;
         assert(o.direct_cadence.offer());assert(calls==3&&worker.trial_priority_front_pending());
         o.code=C3X_RENDERER_RESULT_PENDING;
         assert(!o.direct_cadence.offer());assert(calls==4&&worker.trial_priority_front_pending());
         o.code=C3X_RENDERER_RESULT_OK;
         assert(!o.direct_cadence.offer());assert(calls==5&&!worker.trial_priority_front_pending());
         // A commit after successful presentation must not be consumed by helper bookkeeping.
         ++ticks;worker.value.revision=4;worker.commit();assert(worker.trial_priority_front_pending());
         assert(!o.direct_cadence.offer());assert(calls==6&&!worker.trial_priority_front_pending());
         // Failed initial admission can be retried unchanged and still needs its first presentation.
         ++ticks;worker.value.revision=5;worker.commit(false);assert(!worker.trial_priority_front_pending());
         worker.commit();assert(worker.trial_priority_front_pending());assert(!o.direct_cadence.offer());assert(calls==7);
         // New device/session revision can repeat an old number without an ABA miss after reset.
         worker.reset();assert(!worker.trial_priority_front_pending());worker.value.revision=1;worker.commit();
         ++ticks;assert(!o.direct_cadence.offer());assert(calls==8&&!worker.trial_priority_front_pending());
         o.priority_front_pending={};++ticks;worker.value.revision=2;worker.commit();
         assert(o.direct_cadence.offer());assert(calls==8); // old DLL export fallback
         ticks+=251;assert(!o.direct_cadence.offer());assert(calls==9);
         assert(o.values.visual_frames==7); // only seven successful presents, no BUSY/PENDING telemetry
         assert(o.cadence_counts.prefix_active==1&&o.cadence_counts.prefix_ready_unretired==1);
         assert(o.cadence_counts.dll_busy==1&&o.cadence_counts.pending==1);
         assert(o.cadence_counts.pressure_holds==2);
         worker.value.revision=3;worker.commit();assert(worker.trial_priority_front_pending());
         worker.rebind();assert(!worker.trial_priority_front_pending());
         worker.commit();assert(worker.trial_priority_front_pending());
        }
        '''
        run_cpp('#include <atomic>\n'+prefix+worker+main)

    def test_actual_native_commit_preserves_identical_full_front_revision(self):
        s=(ROOT/'Renderer/native/retained_composition.h').read_text()
        body='    void commit(Id image,Rect area){'+s.split('    void commit(Id image,Rect area){',1)[1].split('    Texture sample(',1)[0]
        run_cpp(r'''
        #include <cassert>
        #include <algorithm>
        #include <memory>
        #include <map>
        #include <vector>
        #include <cstdint>
        struct Rect{int left=0,top=0,right=0,bottom=0;};using Id=unsigned;
        struct Node{Rect area;};struct Patch{Rect area;std::shared_ptr<Node> node;unsigned output;};
        struct Picture{unsigned width=0,height=0,format=0;std::uint64_t version=0;std::vector<Patch> patches;};
        struct Retained{bool admitted=true;Picture front;std::uint64_t front_revision=0;std::map<Id,Picture> images;
         Rect intersect(Rect a,Rect b){return {std::max(a.left,b.left),std::max(a.top,b.top),std::min(a.right,b.right),std::min(a.bottom,b.bottom)};}
         bool empty(Rect a){return a.left>=a.right||a.top>=a.bottom;}
         Rect extent(Picture const&p){return {0,0,int(p.width),int(p.height)};}
         Picture read(Id id,Rect){return images.at(id);}std::shared_ptr<Node>node(){return std::make_shared<Node>();}
         void write(Picture&,Rect,std::shared_ptr<Node>const&,unsigned){}
        ''' + body + r'''
        };
        int main(){Retained r;Rect full{0,0,2240,1260};r.images[1]={2240,1260,2,7,{}};
         r.commit(1,full);assert(r.front_revision==1);r.commit(1,full);assert(r.front_revision==1);
         r.images[1].version=8;r.commit(1,full);assert(r.front_revision==2);
         r.commit(1,{0,0,10,10});assert(r.front_revision==3);
         r.commit(1,{0,0,0,0});assert(r.front_revision==3);
         r.admitted=false;r.images[1].version=9;r.commit(1,full);assert(r.front_revision==3);
        }
        ''')


if __name__ == "__main__":
    unittest.main()
