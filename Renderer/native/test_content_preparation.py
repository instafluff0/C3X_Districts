"""Exercise the production CPU queue, leases, pressure and shared compiler."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ContentPreparationTests(unittest.TestCase):
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
  Pool pool;std::mutex mutex;std::set<unsigned> seen;std::atomic<int> concurrent{0},peak{0};
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
  assert(!pool.take(2,true));release=true;assert(pool.take(1));pool.pause();assert(calls==1);
 }
 // Destruction joins active compilers before borrowed owners may disappear.
 std::atomic<bool> entered{false},left{false};
 {Pool pool;pool.configure({{1,1}},[&](int const&,std::atomic<bool>const& stop,unsigned){
  entered=true;while(!stop)std::this_thread::yield();left=true;return std::make_unique<Result>(Result{1,1});
 });pool.resume();while(!entered)std::this_thread::yield();}
 assert(left);
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
 std::array<ReliefFields,14> assets;render_core::World dimensions{16,16,false,false};
 std::vector<std::uint32_t> data(128,2+(2<<8));render_core::WorldCoast world;
 TerrainCompileScratch foreground;std::array<TerrainCompileScratch,4> scratch;
 TerrainPreparation pool;
 for(int real:{2,5,6,8}){
  std::fill(data.begin(),data.end(),2+(real<<8));world.update(dimensions,data.data(),data.size(),real);
  TerrainCompileInput input;input.tile_x=8;input.tile_y=4;input.real_terrain_type=real;input.ground=2;
  input.tile_width=128;input.tile_height=64;input.target_height=480;input.world_revision=real;
  input.key[0]=real;
  auto expected=compile_terrain_surfaces(natural,assets,world,input,foreground,[]{return false;},false);
  assert(expected && !expected->layers[real==6?2:0].empty());
  std::deque<TerrainPreparation::Job> jobs;
  for(int i=0;i<12;++i){auto job=input;job.key[1]=i;jobs.push_back({job.key,job});}
  pool.configure(jobs,[&](auto const& job,auto const& stop,unsigned worker){
   return compile_terrain_surfaces(natural,assets,world,job,scratch[worker],[&]{return stop.load();});
  },4);pool.resume();
  for(auto const& job:jobs){auto result=pool.take(job.key);assert(result);
   assert(result->world==expected->world && result->coast==expected->coast);
   assert(result->indices==expected->indices);
   for(unsigned layer=0;layer<3;++layer){auto const& a=result->layers[layer];auto const& b=expected->layers[layer];
    assert(a.size()==b.size());if(!a.empty())assert(!std::memcmp(a.data(),b.data(),a.size()*sizeof(MapVertex)));}
  }
  pool.clear();
 }
 auto cancelled=compile_terrain_surfaces(natural,assets,world,TerrainCompileInput{},foreground,[]{return true;});
 assert(!cancelled);
}
''', timeout=60)


if __name__ == '__main__':
    unittest.main()
