"""Exercise the production CPU queue, leases, pressure and shared compiler."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


class ContentPreparationTests(unittest.TestCase):
    def test_exact_required_retargets_near_full_ready_without_duplicate_dispatch(self):
        run_cpp(r"""
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
#include <map>
using namespace c3x_renderer::render_core;
struct Input {int key;std::shared_ptr<int const> lease;};
struct Result {int key;std::size_t bytes()const{return 8u*1024u*1024u;}};
using Queue=ContentPreparation<int,Input,Result>;
int main(){
 Queue queue;std::atomic<unsigned> first{0},retargeted{0},cancelling{0};
 std::atomic<bool> release_first{false},release_retarget{false};std::mutex calls_mutex;std::map<int,unsigned> calls;
 auto lease=std::make_shared<int const>(73);std::weak_ptr<int const> lifetime=lease;
 auto compiler=[&](Input const& input,auto const& stop,unsigned){
  assert(*input.lease==73);{std::lock_guard<std::mutex> guard(calls_mutex);++calls[input.key];}
  if(input.key<100){++first;while(!release_first && !stop)std::this_thread::yield();}
  else if(input.key<200){++retargeted;while(!release_retarget && !stop)std::this_thread::yield();}
  else if(input.key>=300){++cancelling;while(!stop)std::this_thread::yield();}
  return std::make_unique<Result>(Result{input.key});
 };
 auto until=[&](auto predicate){auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(!predicate()){assert(std::chrono::steady_clock::now()<end);std::this_thread::yield();}};
 std::deque<Queue::Job> initial;std::vector<int> required;
 for(int i=1;i<=20;++i){initial.push_back({i,{i,lease}});required.push_back(i);}
 queue.schedule(std::move(initial),compiler,4,required,{},64u*1024u*1024u,true);
 until([&]{return first==4;});release_first=true;
 until([&]{auto stats=queue.statistics();assert(stats.bytes+stats.active*Queue::byte_limit<=64u*1024u*1024u);
  return stats.bytes==56u*1024u*1024u && !stats.active;});
 std::this_thread::sleep_for(std::chrono::milliseconds(5));
 std::deque<Queue::Job> next;
 for(unsigned i=0;i<500;++i)next.push_back({100,{100,lease}}); // shared recipe occurrences
 for(int i:{7,101,102,103,999})next.push_back({i,{i,lease}});
 queue.schedule(std::move(next),compiler,4,{7,100,101,102,103},{},64u*1024u*1024u,true);
 until([&]{return retargeted>=3;});
 // A resident owner now supplies two components. Their active immutable
 // inputs survive, but their eventual unused results must not occupy ready.
 queue.schedule({{100,{100,lease}},{200,{200,lease}},{201,{201,lease}},{202,{202,lease}}},
  compiler,4,{7,100,200,201,202},{200},64u*1024u*1024u,true);
 lease.reset();assert(!lifetime.expired());release_retarget=true;
 for(int key:{7,100,200,201,202}){auto result=queue.take(key);assert(result && result->key==key);}
 queue.finish_lease();auto stats=queue.statistics();
 assert(lifetime.expired() && stats.active_peak==4 && stats.bytes==0 && stats.pending==0);
 assert(stats.unneeded_ready_bytes==0 && stats.retired_ready==6 && stats.retired_active>=2);
 assert(stats.required_keys==5 && stats.expected_consumed_keys==5 && stats.consumed_required_keys==5);
 assert(stats.needed_result_evictions==0 && stats.rejected==0 && stats.capacity_wait_ms>0);
 assert(calls[100]==1 && calls[999]==0 && calls[200]==1 && calls[201]==1 && calls[202]==1);
 for(auto const& row:calls)assert(row.second<=1);
 // Cancellation still joins all four active owners before reset can retire inputs.
 auto last=std::make_shared<int const>(73);std::weak_ptr<int const> cancelled_lifetime=last;
 queue.schedule({{300,{300,last}},{301,{301,last}},{302,{302,last}},{303,{303,last}}},
  compiler,4,{300,301,302,303},{},64u*1024u*1024u,true);
 last.reset();until([&]{return cancelling==4;});queue.finish_lease();
 assert(cancelled_lifetime.expired() && queue.statistics().active==0 && queue.statistics().cancelled==4);
}
""")

    def test_required_ready_pressure_joins_missing_component_and_retires_obsolete_join(self):
        run_cpp(r"""
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
#include <map>
using namespace c3x_renderer::render_core;
struct Input {int key;std::shared_ptr<int const> lease;};
struct Result {int key;std::size_t bytes()const{return key>=100?16u*1024u*1024u:8u*1024u*1024u;}};
using Queue=ContentPreparation<int,Input,Result>;
int main(){
 Queue queue;std::atomic<bool> entered{false},release{false},obsolete{false},abandoned{false};
 std::mutex mutex;std::map<int,unsigned> calls;
 auto compiler=[&](Input const& input,auto const& stop,unsigned){
  {std::lock_guard<std::mutex> lock(mutex);++calls[input.key];}
  if(input.key==100){entered=true;while(!release && !stop)std::this_thread::yield();assert(*input.lease==42);}
  return std::make_unique<Result>(Result{input.key});
 };
 auto until=[&](auto predicate){auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(!predicate()){assert(std::chrono::steady_clock::now()<end);std::this_thread::yield();}};
 std::deque<Queue::Job> jobs;std::vector<int> required;
 for(int key=1;key<=20;++key){jobs.push_back({key,{key,{}}});required.push_back(key);}
 queue.schedule(std::move(jobs),compiler,4,required,{},64u*1024u*1024u,true);
 until([&]{auto stats=queue.statistics();return stats.bytes==56u*1024u*1024u && !stats.active;});
 // All seven ready results remain required, but a partial tile needs key8.
 auto missing=queue.take(8);assert(missing && missing->key==8);
 auto stats=queue.statistics();assert(stats.join_dispatches==1 && stats.join_peak_bytes==8u*1024u*1024u);
 assert(stats.bytes==56u*1024u*1024u && !stats.join_bytes && !stats.active_join);
 auto lease=std::make_shared<int const>(42);std::weak_ptr<int const> lifetime=lease;
 queue.schedule({{100,{100,lease}}},compiler,4,{1,2,3,4,5,6,7,100},{},64u*1024u*1024u,true);
 lease.reset();std::thread consumer([&]{abandoned=!queue.take(100,false,[&]{return obsolete.load();});});
 until([&]{return entered.load();});stats=queue.statistics();
 assert(stats.active_join==1 && stats.bytes+stats.reserved_bytes<=stats.capacity);
 queue.schedule({{101,{101,{}}}},compiler,4,{1,2,3,4,5,6,7,101},{},64u*1024u*1024u,true);
 obsolete=true;consumer.join();assert(abandoned && !lifetime.expired());
 release=true;until([&]{return queue.statistics().active_join==0;});assert(lifetime.expired());
 auto next=queue.take(101);assert(next && next->key==101);stats=queue.statistics();
 assert(stats.join_peak_bytes==Queue::byte_limit && !stats.join_bytes && stats.join_cancelled==1);
 for(int key=1;key<=7;++key){auto result=queue.take(key);assert(result && result->key==key);}
 queue.finish_lease();stats=queue.statistics();
 assert(stats.needed_result_evictions==0 && stats.rejected==0 && stats.bytes==0 && stats.join_bytes==0);
 assert(stats.expected_consumed_keys==8 && stats.consumed_required_keys==8);
 for(auto const& item:calls)assert(item.second==1);
}
""")

    def test_optional_owned_backing_does_not_hold_publication_or_all_lanes(self):
        run_cpp(r"""
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int key;std::shared_ptr<int const> cpu;std::size_t bytes()const{return 16;}};
using Queue=ContentPreparation<int,int,Result>;
int main(){
 Queue queue;std::atomic<bool> entered{false},release{false},written{false};
 auto compiler=[](int key,auto const&,unsigned){return std::make_unique<Result>(Result{key,std::make_shared<int const>(42)});};
 queue.schedule({{1,1}},compiler,4,{1},{},64u*1024u*1024u,true);
 auto adopted=queue.take(1);assert(adopted && adopted->key==1);
 auto snapshot=std::move(adopted->cpu);std::weak_ptr<int const> lifetime=snapshot;adopted.reset();
 assert(queue.offer_optional({[snapshot,&entered,&release,&written](auto const&,unsigned){
  entered=true;while(!release)std::this_thread::yield();assert(*snapshot==42);written=true;
 },Queue::optional_byte_limit}));snapshot.reset();
 auto until=[&](auto predicate){auto end=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(!predicate()){assert(std::chrono::steady_clock::now()<end);std::this_thread::yield();}};
 until([&]{return entered.load();});assert(!lifetime.expired());
 assert(!queue.offer_optional({[](auto const&,unsigned){assert(false);},1}));
 auto stats=queue.statistics();assert(stats.active_optional==1 && stats.optional_bytes==Queue::optional_byte_limit);
 queue.schedule({{2,2}},compiler,4,{2},{},64u*1024u*1024u,true);
 auto demanded=queue.take(2);assert(demanded && demanded->key==2 && !written);
 // Optional snapshot does not borrow either moved result. Retirement joins it.
 release=true;queue.finish_lease();stats=queue.statistics();
 assert(written && lifetime.expired() && stats.active_optional==0 && stats.optional_bytes==0);
 assert(stats.optional_skipped==1 && stats.optional_cancelled==1 && stats.consumed==2);
}
""")

    def test_bounded_world_results_do_not_evict_unconsumed_demand(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t bytes()const{return 8u*1024u*1024u;}};
using Pool=ContentPreparation<int,int,Result>;
int main(){
 Pool pool;std::deque<Pool::Job> jobs;std::vector<int> needed;
 for(int i=0;i<100;++i){jobs.push_back({i,i});needed.push_back(i);}
 pool.configure(jobs,[](int value,auto const&,unsigned){return std::make_unique<Result>(Result{value});},
     4,needed,64u*1024u*1024u,true);pool.resume();
 std::this_thread::sleep_for(std::chrono::milliseconds(50));
 assert(pool.statistics().pending>0 && pool.statistics().evicted==0);
 for(int i=0;i<100;++i){auto result=pool.take(i);assert(result && result->value==i);}
 pool.finish_lease();auto stats=pool.statistics();
 assert(stats.built==100 && stats.consumed==100 && stats.evicted==0 && stats.rejected==0);
 assert(stats.peak_bytes<=64u*1024u*1024u);
}
''')

    def test_finished_lease_keeps_owned_results_and_reuses_workers(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t bytes()const{return 1;}};
using Pool=ContentPreparation<int,int,Result>;
int main(){
 Pool pool;std::thread::id first,second;std::atomic<bool> entered{false},left{false};
 auto lease=std::make_shared<int>(17);std::weak_ptr<int> borrowed=lease;
 pool.configure({{1,1},{2,2}},[&,lease](int const& input,auto const& stop,unsigned){
  first=std::this_thread::get_id();
  if(input==2){entered=true;while(!stop)std::this_thread::yield();left=true;return std::unique_ptr<Result>{};}
  return std::make_unique<Result>(Result{*lease});
 });lease.reset();pool.resume();while(!entered)std::this_thread::yield();
 pool.finish_lease();assert(left && borrowed.expired());
 auto stats=pool.statistics();assert(stats.active==0 && stats.pending==0 && stats.bytes==1);
 assert(pool.contains(1,[](auto const& result){return result.value==17;}));
 pool.configure({{3,3}},[&](int const& value,auto const&,unsigned){
  second=std::this_thread::get_id();return std::make_unique<Result>(Result{value});
 },1,{1,3});pool.resume();assert(pool.take(1)->value==17);assert(pool.take(3)->value==3);
 pool.finish_lease();assert(first==second); // persistent pool, no per-view thread churn
 pool.configure({{4,4}},[](int const& value,auto const&,unsigned){return std::make_unique<Result>(Result{value});});
 pool.resume();while(!pool.statistics().built || pool.statistics().pending || pool.statistics().active)std::this_thread::yield();
 pool.finish_lease();assert(!pool.contains(4,[](auto const&){return false;}));assert(pool.statistics().invalidated==1);
 pool.clear();assert(pool.statistics().bytes==0);
}
''')

    def test_workers_pressure_cancellation_and_failure(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
#include <set>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t size;std::size_t bytes()const{return size;}};
using Pool=ContentPreparation<int,int,Result>;
int main(){
 for(unsigned count:{1u,2u,4u,6u}){
  std::mutex mutex;std::set<unsigned> seen;std::atomic<int> concurrent{0},peak{0};
  Pool pool; // Joins before the compiler's borrowed synchronization owners die.
  auto compiler=[&](int const& input,std::atomic<bool>const& stop,unsigned worker){
   {std::lock_guard<std::mutex> lock(mutex);seen.insert(worker);}
   int active=++concurrent,old=peak;while(active>old && !peak.compare_exchange_weak(old,active)){}
   for(int i=0;i<10 && !stop;++i)std::this_thread::sleep_for(std::chrono::milliseconds(1));
   --concurrent;
   if(input==-1)throw std::runtime_error("optional preparation failed");
   return std::make_unique<Result>(Result{input,input==-2?Pool::byte_limit+1:3u*1024u*1024u});
  };
  std::deque<Pool::Job> jobs;for(int i=0;i<40;++i)jobs.push_back({i,i});
  pool.configure(jobs,compiler,count);pool.resume();
  // Fill speculative backpressure, then demand its furthest queued entry.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  auto last=pool.take(39);assert(last && last->value==39);
  assert(pool.statistics().peak_bytes<=Pool::byte_limit);
  assert(peak<=int(count));if(count>1)assert(peak>1);
  pool.pause();assert(concurrent==0);auto before=pool.statistics().built;
  std::this_thread::sleep_for(std::chrono::milliseconds(15));assert(pool.statistics().built==before);
  pool.clear();pool.configure({{-1,-1},{-2,-2},{50,50}},compiler,count);pool.resume();
  assert(!pool.take(-1));assert(!pool.take(-2));auto good=pool.take(50);assert(good && good->value==50);
  pool.clear();assert(pool.statistics().bytes==0);
  pool.configure(jobs,compiler,count);pool.resume();
  std::this_thread::sleep_for(std::chrono::milliseconds(2));pool.pause();assert(concurrent==0);
  assert(pool.statistics().cancelled>0);pool.resume();assert(pool.take(0));
 }
 // A full speculative reservoir must make room for a newly selected view.
 // Preserve already-ready demand, and reject stale content before scheduling.
 {Pool pool;auto compiler=[](int const& input,auto const&,unsigned){
   return std::make_unique<Result>(Result{input,3u*1024u*1024u});};
  std::deque<Pool::Job> old;for(int i=0;i<10;++i)old.push_back({i,i});
  pool.configure(old,compiler);pool.resume();
  while(pool.statistics().bytes<9u*1024u*1024u)std::this_thread::yield();pool.pause();
  assert(pool.contains(0,[](auto const&){return true;}));
  assert(!pool.contains(1,[](auto const&){return false;}));assert(pool.statistics().invalidated==1);
  auto built=pool.statistics().built;
  pool.configure({{60,60},{50,50}},compiler,2,{0,50});pool.resume();
  auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(pool.statistics().built==built){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
  assert(pool.take(0)->value==0);assert(pool.take(50)->value==50);
  assert(pool.statistics().evicted>0);pool.clear();
 }
 // A larger reservoir still obeys demand and shrinking drops old storage.
 {Pool pool;auto compiler=[](int const& input,auto const&,unsigned){
   return std::make_unique<Result>(Result{input,20u*1024u*1024u});};
  pool.configure({{1,1},{2,2},{3,3}},compiler,6,{},64u*1024u*1024u);pool.resume();
  assert(pool.take(1));assert(pool.take(2));pool.pause();
  assert(pool.statistics().peak_bytes<=64u*1024u*1024u);
  pool.configure({{4,4}},compiler,1);assert(pool.statistics().bytes==0);
  pool.resume();assert(!pool.take(4));pool.clear();
 }
 // The renderer may take a queued task for immediate foreground compilation.
 {Pool pool;std::atomic<bool> entered{false},release{false};std::atomic<int> calls{0};
  pool.configure({{1,1},{2,2}},[&](int const& input,auto const&,unsigned){
   ++calls;entered=true;while(!release)std::this_thread::yield();return std::make_unique<Result>(Result{input,1});
  });pool.resume();while(!entered)std::this_thread::yield();
  assert(pool.compiling(1) && !pool.compiling(2));
  assert(!pool.take(2,true));release=true;assert(pool.take(1));pool.pause();assert(calls==1);
  assert(!pool.compiling(1) && !pool.compiling(2));
 }
 // Destruction joins active compilers before borrowed owners may disappear.
 std::atomic<bool> entered{false},left{false};
 {Pool pool;pool.configure({{1,1}},[&](int const&,std::atomic<bool>const& stop,unsigned){
  entered=true;while(!stop)std::this_thread::yield();left=true;return std::make_unique<Result>(Result{1,1});
 });pool.resume();while(!entered)std::this_thread::yield();}
 assert(left);
}
''')

    def test_selected_view_fills_budget_without_serial_consumer_demand(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t bytes()const{return 3u*1024u*1024u;}};
using Pool=ContentPreparation<int,int,Result>;
int main(){
 Pool pool;auto compile=[](int const& input,auto const&,unsigned){return std::make_unique<Result>(Result{input});};
 // Current dependencies exceed the speculative half-budget watermark but fit
 // the real budget. All must prepare before the GPU owner requests any result.
 pool.configure({{90,90},{1,1},{2,2},{3,3},{4,4},{5,5}},compile,2,{1,2,3,4,5});pool.resume();
 auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
 while(pool.statistics().built<5){assert(std::chrono::steady_clock::now()<deadline);std::this_thread::yield();}
 pool.pause();assert(pool.statistics().pending==1);assert(pool.statistics().evicted==0);
 assert(pool.statistics().bytes==15u*1024u*1024u);
 for(int key=1;key<=5;++key)assert(pool.contains(key,[](auto const&){return true;}));
 // A new selection replaces urgency; useful selected ready content survives,
 // previous-view content can be evicted, and no new speculative work runs full.
 pool.configure({{91,91},{6,6}},compile,1,{5,6});pool.resume();
 assert(pool.take(6)->value==6);pool.pause();assert(pool.contains(5,[](auto const&){return true;}));
 assert(pool.statistics().peak_bytes<=Pool::byte_limit);pool.clear();
}
''')

    def test_ready_adoption_and_notification_lifetime(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {int value;std::size_t bytes()const{return 1;}};
using Pool=ContentPreparation<int,std::shared_ptr<int>,Result>;
int main(){
 Pool pool;std::atomic<bool> entered{false},release{false},notifying{false};
 std::atomic<unsigned> calls{0};std::mutex consumer;std::condition_variable wake;unsigned revision=0;
 std::unique_lock<std::mutex> lock(consumer);
 pool.set_ready_notification([&]{notifying=true;std::lock_guard<std::mutex> guard(consumer);++revision;wake.notify_one();});
 auto input=std::make_shared<int>(1);std::weak_ptr<int> lease=input;
 pool.configure({{1,input},{2,std::make_shared<int>(2)}},[&](auto const& input,auto const&,unsigned){
  ++calls;entered=true;while(!release)std::this_thread::yield();
  return std::make_unique<Result>(Result{*input});
 });
 input.reset();pool.resume();while(!entered)std::this_thread::yield();
 assert(!pool.take_ready(1));assert(!pool.take_ready(2));
 assert(pool.statistics().pending==1 && calls==1); // No wait, steal, or duplicate compiler.
 release=true;while(!notifying)std::this_thread::yield();
 // Notification is waiting for the consumer lock. Pause must still finish:
 // publication ended the CPU input lease before notifying the GPU scheduler.
 pool.pause();assert(lease.expired());assert(pool.take_ready(1)->value==1);assert(!pool.take_ready(1));
 assert(wake.wait_for(lock,std::chrono::seconds(5),[&]{return revision==1;}));
 lock.unlock();pool.set_ready_notification({}); // Joins callback before consumer destruction.
 pool.resume();assert(pool.take(2)->value==2);pool.pause();assert(calls==2 && revision==1);
 pool.clear();
}
''')

    def test_shared_terrain_compiler_parallel_parity(self):
        run_cpp(r'''
#include "Renderer/native/source_fidelity/terrain_compiler.h"
#include "Renderer/lab/shared/natural/patterns.h"
#include <cassert>
#include <cstring>
namespace c3x_renderer {
float stable_random(std::uint32_t value){return patterns::stable_random(value);}
std::uint32_t stable_hash(std::uint32_t value){return patterns::feature_hash(value);}
}
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;
int main(){
 NaturalData natural;natural.fields.resize(1);auto& field=natural.fields[0];
 field.width=field.height=16;field.pixels.resize(256);
 for(unsigned i=0;i<256;++i)field.pixels[i]=std::uint8_t(i*31);field.minimum=0;field.maximum=1;
 // The actual jungle-floor compiler selects recipes25..34 and reads its
 // selected body's hull. A height field alone is not a valid natural source.
 natural.bodies.resize(1);natural.bodies[0].vertices={
  {{-.5f,-.5f,0},{0,0,1},{0,0}},{{.5f,-.5f,0},{0,0,1},{1,0}},{{.5f,.5f,0},{0,0,1},{1,1}}};
 natural.recipes.resize(35);
 natural.recipes[0]={0,1,0,180,0,0,2,1,0};natural.recipes[25]={0,1,0,121,0,0,2,1,0};
 std::array<ReliefFields,14> assets;render_core::World dimensions{16,16,false,false};
 std::vector<std::uint32_t> data(128,2+(2<<8));render_core::WorldCoast world;
 TerrainCompileScratch foreground;std::array<TerrainCompileScratch,4> scratch;
 TerrainPreparation pool;
 auto equal=[](TerrainSurfaces const& result,TerrainSurfaces const& expected){
  assert(result.world==expected.world && result.coast==expected.coast);
  assert(result.rivers.size()==expected.rivers.size());
  for(unsigned i=0;i<result.rivers.size();++i){auto const& a=result.rivers[i];auto const& b=expected.rivers[i];
   assert(a.first==b.first && a.second->values==b.second->values);
   assert(a.second->inputs->values==b.second->inputs->values && a.second->inputs->flow==b.second->inputs->flow);}
  for(unsigned layer=0;layer<3;++layer){auto const& a=result.meshes[layer];auto const& b=expected.meshes[layer];
   assert(a.vertices==b.vertices && a.indices==b.indices && a.bounds==b.bounds);
   assert(a.world_low==b.world_low && a.world_high==b.world_high && a.projected_bounds.extent==b.projected_bounds.extent);
   assert(a.vertex_stride==b.vertex_stride && a.index_stride==b.index_stride && a.shared_grid==b.shared_grid);}
 };
 // Exercise exact foreground/worker geometry and dependency parity at both
 // lattice sizes. Default64 hill admission is checked separately below.
 for(bool bounded:{false,true})for(int real:{2,5,6,8}){
  std::fill(data.begin(),data.end(),2+(real<<8));world.update(dimensions,data.data(),data.size(),real);
  TerrainCompileInput input;input.tile_x=8;input.tile_y=4;input.real_terrain_type=real;input.ground=2;
  input.tile_width=128;input.tile_height=64;input.target_height=480;input.world_revision=real;
  input.detail.mountain=bounded?32:64;input.key[0]=real;input.key[2]=input.detail.identity();
  auto expected=compile_terrain_surfaces(natural,assets,world,input,foreground,[]{return false;},bounded);
  assert(expected && !expected->meshes[real==6?2:0].empty());
  assert(expected->bytes()<TerrainPreparation::byte_limit);
  if(real==6 || real==8)assert(!expected->rivers.empty()); // Includes empty river-bucket proofs.
  std::deque<TerrainPreparation::Job> jobs;
  for(int i=0;i<12;++i){auto job=input;job.key[1]=i;jobs.push_back({job.key,job});}
  pool.configure(jobs,[&](auto const& job,auto const& stop,unsigned worker){
   return compile_terrain_surfaces(natural,assets,world,job,scratch[worker],[&]{return stop.load();},bounded);
  },4);pool.resume();
  for(auto const& job:jobs){auto result=pool.take(job.key);assert(result);
   equal(*result,*expected);
  }
  pool.clear();
 }
 // The same default64 hill used to fail because expanded rock triangles
 // exceeded the raw transient cap. Indexed emission now admits it without
 // changing any packed byte or dependency; unbounded recovery stays expanded.
 std::fill(data.begin(),data.end(),2+(5<<8));world.update(dimensions,data.data(),data.size(),20);
 TerrainCompileInput oversized;oversized.tile_x=8;oversized.tile_y=4;oversized.real_terrain_type=5;
 oversized.ground=2;oversized.tile_width=128;oversized.tile_height=64;oversized.target_height=480;
 oversized.world_revision=20;oversized.key[0]=20;
 auto recovery=compile_terrain_surfaces(natural,assets,world,oversized,foreground,[]{return false;},false);
 assert(recovery && recovery->bytes()<TerrainPreparation::byte_limit);
 auto bounded_hill=compile_terrain_surfaces(natural,assets,world,oversized,foreground,[]{return false;},true);
 assert(bounded_hill);equal(*bounded_hill,*recovery);
 auto before_hill=pool.statistics();
 pool.configure({{oversized.key,oversized}},[&](auto const& job,auto const& stop,unsigned worker){
  return compile_terrain_surfaces(natural,assets,world,job,scratch[worker],[&]{return stop.load();},true);
 });pool.resume();auto prepared_hill=pool.take(oversized.key);assert(prepared_hill);equal(*prepared_hill,*recovery);
 pool.pause();auto after_hill=pool.statistics();
 assert(after_hill.rejected==before_hill.rejected && after_hill.consumed==before_hill.consumed+1);pool.clear();
 // Preserve explicit bounded rejection independently of redundant triangle
 // expansion. No denied table allocation may enter the ready queue.
 std::vector<MapVertex> rejected_vertices;std::vector<unsigned> rejected_indices;
 HillDecalOutput rejected_output(rejected_vertices,rejected_indices);
 assert(!rejected_output.initialize(8u*1024u*1024u,[](std::size_t){return true;}));
 assert(rejected_output.rejected() && rejected_output.scratch_bytes()==0);
 assert(rejected_vertices.empty() && rejected_indices.empty());
 // A supported default64 legacy expanded input still rejects through the
 // actual compiler and queue; unbounded recovery remains available and exact.
 auto excessive=oversized;excessive.indexed=false;excessive.key[0]=21;
 auto large_recovery=compile_terrain_surfaces(natural,assets,world,excessive,foreground,[]{return false;},false);
 assert(large_recovery && large_recovery->bytes()<TerrainPreparation::byte_limit);
 auto before_rejection=pool.statistics();
 pool.configure({{excessive.key,excessive}},[&](auto const& job,auto const& stop,unsigned worker){
  return compile_terrain_surfaces(natural,assets,world,job,scratch[worker],[&]{return stop.load();},true);
 });pool.resume();assert(!pool.take(excessive.key));pool.pause();auto after_rejection=pool.statistics();
 assert(after_rejection.rejected==before_rejection.rejected+1 && after_rejection.consumed==before_rejection.consumed);pool.clear();
 auto large_retry=compile_terrain_surfaces(natural,assets,world,excessive,foreground,[]{return false;},false);
 assert(large_retry);equal(*large_retry,*large_recovery);
 // Cancel after mesh generation has begun; private scratch must be reusable.
 unsigned checks=0;
 assert(!compile_terrain_surfaces(natural,assets,world,oversized,foreground,[&]{return ++checks==20;},false));
 assert(checks==20);
 auto retried=compile_terrain_surfaces(natural,assets,world,oversized,foreground,[]{return false;},false);
 assert(retried);equal(*retried,*recovery);
 auto cancelled=compile_terrain_surfaces(natural,assets,world,TerrainCompileInput{},foreground,[]{return true;});
 assert(!cancelled);
}
''', timeout=60)


if __name__ == '__main__':
    unittest.main()
