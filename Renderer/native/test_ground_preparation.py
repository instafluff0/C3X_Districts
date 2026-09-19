"""Production ground compilation, bounded worker leases and packed adoption."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


class GroundPreparationTests(unittest.TestCase):
    def test_shared_sampling_detail_and_wrapped_material_coordinates(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/source_fidelity/ground_preparation.h"
#include <cassert>
#include <set>
using namespace c3x_renderer;
int main(){
 c3x_renderer_tile_v1 tile{};tile.tile_x=98;tile.tile_y=50;
 struct Neighbor{int relief=-1;} neighbor;
 std::set<std::pair<int,int>> observed;
 auto topology=[&](int x,int y){observed.emplace(x,y);return &neighbor;};
 bool dune=false;
 auto world=[&](int c,int r){render_core::Tile value;value.real=dune && c==75 && r==25?0:2;return value;};
 for(int width:{128,160,192}){
  assert(fidelity::ground_grid_detail(tile,-1,false,true,width,1000,topology,world)==12);
  neighbor.relief=5;assert(fidelity::ground_grid_detail(tile,-1,false,true,width,1000,topology,world)==24);
  neighbor.relief=-1;dune=true;assert(fidelity::ground_grid_detail(tile,-1,false,true,width,1000,topology,world)==24);dune=false;
 }
 assert((observed==std::set<std::pair<int,int>>{{97,49},{99,49},{99,51},{97,51}}));
 assert(fidelity::ground_grid_detail(tile,-1,false,true,64,3000,topology,world)==8);
 assert(fidelity::ground_grid_detail(tile,5,false,true,64,3000,topology,world)==12);
 assert(fidelity::ground_grid_detail(tile,5,false,false,128,512,topology,world)==24);
 assert(fidelity::ground_grid_detail(tile,5,false,false,128,768,topology,world)==16);
 assert(fidelity::ground_grid_detail(tile,5,false,false,128,769,topology,world)==12);
 assert(fidelity::ground_grid_detail(tile,5,false,false,64,2048,topology,world)==12);
 assert(fidelity::ground_grid_detail(tile,5,false,false,64,2049,topology,world)==8);
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=100;
 frame.world_wrap_x=frame.world_wrap_y=1;
 for(float fraction:{.25f,.5f,.75f}){
  float u=74+fraction,v=24+fraction;
  auto expected=fidelity::ground_surface_uv(tile,frame,u,v,.26f);
  auto wrapped=tile;wrapped.tile_x+=100;
  assert(fidelity::ground_surface_uv(wrapped,frame,u+50,v+50,.26f)==expected);
  wrapped=tile;wrapped.tile_y+=100;
  assert(fidelity::ground_surface_uv(wrapped,frame,u+50,v-50,.26f)==expected);
 }
}
''')

    def test_returning_lanes_keeps_active_inputs_and_ready_results(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {unsigned key=0;std::size_t bytes()const{return sizeof(*this);}};
int main(){
 ContentPreparation<unsigned,unsigned,Result> queue;
 std::atomic<unsigned> entered{0};std::atomic<bool> release{false};
 queue.configure({{1,1},{2,2},{3,3},{4,4}},[&](auto input,auto const& stop,unsigned){
  ++entered;while(!release.load()){assert(!stop.load());std::this_thread::yield();}
  auto r=std::make_unique<Result>();r->key=input;return r;
 },2);
 queue.resume();while(entered.load()!=2)std::this_thread::yield();
 assert(queue.statistics().active==2 && queue.statistics().pending==2);
 queue.expand_workers(4);while(entered.load()!=4)std::this_thread::yield();
 assert(queue.statistics().active==4 && queue.statistics().pending==0);
 release=true;
 for(unsigned i=4;i;--i){auto result=queue.take(i);assert(result && result->key==i);}
 auto stats=queue.statistics();assert(stats.built==4 && stats.consumed==4 && !stats.cancelled && !stats.rejected);
 queue.clear();assert(queue.statistics().active==0 && queue.statistics().bytes==0);
}
''')

    def test_selected_batch_backpressure_demand_and_frame_unwind(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/source_fidelity/ground_preparation.h"
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 fidelity::GroundPreparation queue;
 render_core::CapturedScene scene;
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=32;
 assert(scene.begin(frame));c3x_renderer_tile_v1 tile{};tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 assert(scene.update(tile,2,-1,2,11));scene.finish();auto view=scene.observation_view();
 std::atomic<unsigned> entered{0},finished{0};
 auto owner=std::make_shared<int>(7);std::weak_ptr<int> weak=owner;
 {
  fidelity::GroundPreparationLease lease(queue);
  fidelity::GroundPreparationInput input;input.compile.tile=tile;
  lease.start({{1,input},{2,input}},[&,owner](auto const&,auto&,auto const& stop){
   ++entered;
   while(!stop.load()){
    auto observed=view.current(view.key(0,0));assert(observed && observed->semantic==11);
    assert(*owner==7);std::this_thread::yield();
   }
   ++finished;return std::make_unique<fidelity::PreparedGround>();
  });
  owner.reset();while(entered.load()!=2)std::this_thread::yield();
  for(int i=0;i<1000;++i)scene.attach(tile,{});
  // Early-return/exception cleanup joins while the immutable source is alive.
 }
 assert(finished==2 && weak.expired());
 assert(queue.statistics().pending==0 && queue.statistics().bytes==0);
 assert(scene.begin(frame));assert(scene.update(tile,11,-1,11,12));scene.finish();
 {
  fidelity::GroundPreparationLease lease(queue);
  std::deque<fidelity::GroundPreparation::Job> jobs;
  for(unsigned i=0;i<64;++i){fidelity::GroundPreparationInput input;input.compile.ground=int(i);jobs.push_back({i,input});}
  lease.start(std::move(jobs),[](auto const& input,auto&,auto const&)->std::unique_ptr<fidelity::PreparedGround>{
   if(input.compile.ground==31)throw std::bad_alloc();
   auto r=std::make_unique<fidelity::PreparedGround>();r->grid_hits=unsigned(input.compile.ground);
   r->meshes[0].vertices.resize(1024*1024);return r;
  });
  // Wait for the ordinary refill gate, then demand the last pending tile.
  // A full ready queue cannot starve an out-of-order current-view request.
  while(queue.statistics().bytes<queue.byte_limit/2)std::this_thread::yield();
  for(unsigned i=64;i-->0;){auto result=queue.take(i);
   assert(bool(result)==(i!=31));if(result)assert(result->grid_hits==i);
  }
  auto stats=queue.statistics();assert(stats.peak_bytes<=queue.byte_limit);
  assert(stats.active_peak==2 && stats.evicted==0 && stats.rejected==1);
 }
 assert(queue.statistics().pending==0 && queue.statistics().bytes==0);
}
''', timeout=90)

    def test_cliff_private_queries_borrow_the_loaded_hill_assets(self):
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        height = "auto cliff_natural_height_at=" + source.split("auto cliff_natural_height_at=", 1)[1].split(
            "                c3x_renderer::fidelity::compile_cliff_surfaces", 1)[0]
        run_cpp(r'''
#include "Renderer/native/source_fidelity/surface_query_scratch.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 render_core::WorldCoast world;
 std::vector<std::uint32_t> bits(512,2|(5<<8));
 world.update({32,32,true,true},bits.data(),bits.size(),1);
 fidelity::NaturalWorld natural;natural.fields.resize(1);
 natural.fields[0].width=natural.fields[0].height=2;
 natural.fields[0].pixels={0,64,128,255};
 fidelity::SurfaceQueryScratch cliff_query_scratch;
 cliff_query_scratch.bind(natural,world.world(),1);
 assert(cliff_query_scratch.rivers.fields.empty());
 auto ignore=[](auto,auto){};
 fidelity::SurfaceQueries cliff_queries(world,cliff_query_scratch.shore_samples,16,8,ignore,ignore,true);
 auto cliff_pickup_height_at=[](float,float){return 0.f;};
 bool retain_height_samples=true;
''' + height + r'''
 float support=0;
 float h=cliff_natural_height_at(12.5f,4.5f,&support);
 assert(h>2.5f && support>0);
 assert(h==cliff_natural_height_at(12.5f,4.5f));
 assert(cliff_query_scratch.height_samples.hits==1);
}
''')

    def test_persistent_query_pages_reset_on_dimensions_wrap_and_revision(self):
        run_cpp(r'''
#include "Renderer/native/source_fidelity/surface_query_scratch.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 fidelity::NaturalData assets;render_core::WorldCoast world;
 std::vector<std::uint32_t> bits(512,2|(2<<8));
 world.update({32,32,false,false},bits.data(),bits.size(),1);
 fidelity::SurfaceQueryScratch scratch;scratch.bind(assets,world.world(),1);
 scratch.rivers.river_page_entry(4,4);assert(scratch.rivers.river_pages.size()==1);
 scratch.bind(assets,world.world(),1);assert(scratch.rivers.river_pages.size()==1);
 // A new world may reuse the revision number; dimensions and wrapping still
 // change the meaning of every river-page support/dependency index.
 world.update({16,16,false,false},bits.data(),128,1);
 scratch.bind(assets,world.world(),1);assert(scratch.rivers.river_pages.empty());
 scratch.rivers.river_page_entry(4,4);
 world.update({16,16,true,false},bits.data(),128,1);
 scratch.bind(assets,world.world(),1);assert(scratch.rivers.river_pages.empty());
 scratch.rivers.river_page_entry(4,4);scratch.bind(assets,world.world(),2);
 assert(scratch.rivers.river_pages.empty() && scratch.rivers.borrowed_data==&assets);
}
''')

    def test_scope_joins_cancellation_errors_and_releases_captures(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scoped_preparation.h"
#include <cassert>
using namespace c3x_renderer::render_core;
struct Result {unsigned value=0;std::size_t size=4;std::size_t bytes()const{return size;}};
int main(){
 using Task=ScopedPreparation<Result>;Task::Queue queue;
 std::atomic<bool> entered{false},finished{false};
 int borrowed=7;
 {
  Task flight(queue,[&](auto const& stop){
   entered=true;while(!stop.load()){assert(borrowed==7);std::this_thread::yield();}
   finished=true;return std::make_unique<Result>();
  },true);
  while(!entered.load())std::this_thread::yield();
  // Simulate a failure/early return before take(): destruction cancels and
  // joins while the borrowed input is still alive and immutable.
 }
 assert(finished);borrowed=8;
 assert(queue.statistics().pending==0 && queue.statistics().bytes==0);
 for(int i=0;i<4;++i){
  auto owned=std::make_shared<int>(9);std::weak_ptr<int> weak=owned;
  {
   Task flight(queue,[owned,i](auto const&)->std::unique_ptr<Result>{
    if(i==0)throw std::bad_alloc();
    auto r=std::make_unique<Result>();r->value=*owned;
    if(i==1)r->size=Task::Queue::byte_limit+1;
    return r;
   },true);
   owned.reset();auto result=flight.take();
   assert(bool(result)==(i>=2));if(result)assert(result->value==9);
   assert(weak.expired()); // completed callback/input leases cannot linger
  }
 }
 assert(queue.statistics().active_peak==1 && queue.statistics().bytes==0);
 bool threw=false;
 try{Task flight(queue,[](auto const&)->std::unique_ptr<Result>{throw std::bad_alloc();},false);}
 catch(std::bad_alloc const&){threw=true;}
 assert(threw);
}
''')

    def test_real_ground_parallel_pixels_proofs_wrapping_and_cache_lease(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/source_fidelity/ground_preparation.h"
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
using namespace c3x_renderer;
struct RiverNode {int lattice_x,lattice_y;unsigned degree;bool touches_water;};
void equal(fidelity::PreparedGround const& a,fidelity::PreparedGround const& b){
 assert(a.world==b.world && a.coast==b.coast && a.topology==b.topology);
 assert(a.rivers.size()==b.rivers.size());
 for(auto const& proof:a.rivers)assert(proof.second->values==b.rivers.at(proof.first)->values);
 assert(a.water_coverage==b.water_coverage && a.grid_hits==b.grid_hits);
 for(unsigned i=0;i<6;++i){auto const& x=a.meshes[i];auto const& y=b.meshes[i];
  assert(x.vertices==y.vertices && x.indices==y.indices && x.bounds==y.bounds);
  assert(x.world_low==y.world_low && x.world_high==y.world_high);
  assert(x.vertex_stride==y.vertex_stride && x.index_stride==y.index_stride && x.index_count==y.index_count);
 }
}
int main(){
 render_core::WorldCoast coast;
 std::vector<std::uint32_t> bits(512,2|(2<<8));
 for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2){
  unsigned kind=x<12?11u:2u;bits[y*16+x/2]=kind|(kind<<8);
  if(x==14 && y<24)bits[y*16+x/2]|=34u<<16;
 }
 coast.update({32,32,true,true},bits.data(),bits.size(),1);
 fidelity::NaturalWorld natural;natural.update_rivers(coast.world(),1);
 render_core::CapturedScene topology;
 // Ground uses current observations for water-family material selection.
 c3x_renderer_frame_v1 capture{};capture.world_width_tiles=capture.world_height_tiles=32;
 capture.world_wrap_x=capture.world_wrap_y=1;assert(topology.begin(capture));
 for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2){
  c3x_renderer_tile_v1 tile{};tile.tile_x=x;tile.tile_y=y;
  tile.terrain_type=int(bits[y*16+x/2]&255);tile.real_terrain_type=tile.terrain_type;
  tile.tile_flags=C3X_RENDERER_TILE_RENDER;
  assert(topology.update(tile,tile.terrain_type,tile.real_terrain_type,0,123+unsigned(x+y)));
 }
 topology.finish();
 fidelity::GroundTask::Queue queue;
 fidelity::SurfaceQueryScratch reused_scratch,fresh_scratch;
 auto* scratch=&fresh_scratch;
 std::array<RiverNode,3> node_values={{{14,12,3,false},{12,11,1,false},{14,13,1,true}}};
 std::vector<RiverNode const*> nodes;for(auto const& node:node_values)nodes.push_back(&node);
 auto source=[](int,unsigned,int,float,float){return 0.f;};
 auto dune=[](float,float){return 0.f;};
 auto river=[](auto const&,float,float){return 1000.f;};
 auto relief=[](float,float){return std::array<float,3>{};};
 auto weights=[](float,float){return std::array<float,5>{1,0,0,0,0};};
 auto shore=[](float,float,float,float){return 0.f;};

 auto ndc=[](float x){return x;};
 auto key=[&](int x,int y){return topology.key(x,y);};
 for(int width:{64,128,160,192})for(int x:{12,14,44}){
  c3x_renderer_frame_v1 frame{};frame.tile_width=width;frame.tile_height=width/2;frame.world_topology_revision=1;
  frame.world_width_tiles=frame.world_height_tiles=32;frame.world_wrap_x=frame.world_wrap_y=1;
  fidelity::GroundCompileInput input;input.tile.tile_x=x;input.tile.tile_y=12;
  input.tile.terrain_type=2;input.tile.real_terrain_type=2;input.tile.river_code=34;
  input.pickup_profile=input.fidelity_profile=input.river_assets_ready=input.draw_marsh=true;
  input.world_ground=width>=96;input.ground=2;input.half_w=width*.5f;input.half_h=width*.25f;
  input.relief_projection_scale=width/224.f*.82f;input.flat_grid=8;input.tile_ground_grid=16;input.shadow_grid=8;
  input.retain_ground_grids=true;input.reuse_nested_ground_grids=true;
  float u=(x+12)*.5f+.5f,v=(x-12)*.5f+.5f;
  auto center=coast.sample({u,v},[](auto,auto){},[](auto,auto){});
  std::shared_ptr<std::vector<fidelity::CachedGroundGrid>> grids;
  auto uv=[&](float u,float v,float scale){return fidelity::ground_surface_uv(input.tile,frame,u,v,scale);};
  auto compile=[&](auto const& stop){
   return fidelity::prepare_ground(input,frame,natural,*scratch,coast,topology,nodes,grids,center,2.f,true,true,
    key,source,dune,river,relief,weights,shore,uv,ndc,ndc,[&]{return stop.load();});
  };
  fresh_scratch.rivers.reset_world();scratch=&fresh_scratch;
  fidelity::GroundTask serial(queue,compile,false);auto expected=serial.take();assert(expected);
  scratch=&reused_scratch;
  fidelity::GroundTask parallel(queue,compile,true);
  // The main consumer queries its own river state while ground compiles.
  natural.river_sample({u,v});
  auto actual=parallel.take();assert(actual);equal(*expected,*actual);
  assert(reused_scratch.rivers.river_pages.size()<=2);
  if(input.world_ground){
   fidelity::GroundPreparationInput owned;owned.compile=input;
   owned.tile_width=frame.tile_width;owned.tile_height=frame.tile_height;
   owned.world_width=frame.world_width_tiles;owned.world_height=frame.world_height_tiles;
   owned.wrap_x=frame.world_wrap_x;owned.wrap_y=frame.world_wrap_y;
   owned.topology_revision=frame.world_topology_revision;owned.center=center;
   for(auto node:nodes)owned.nodes.push_back({node->lattice_x,node->lattice_y,node->degree,node->touches_water});
   owned.skip_flat_shore=owned.separate_natural_relief=true;
   std::array<fidelity::ReliefFields,16> assets;
   fidelity::GroundPreparation selected;
   fidelity::GroundPreparationLease lease(selected);
   auto view=topology.observation_view();
   lease.start({{1,owned},{2,owned}},[&](auto const& job,auto& lane,auto const& stop){
    return fidelity::compile_selected_ground(job,natural,coast,view,assets,lane,[&]{return stop.load();});
   });
   // Source capture may disappear; queued jobs own its values. Mutable cache
   // attachments use a separate map from the immutable observation view.
   owned={};
   for(int n=0;n<100;++n)topology.attach(input.tile,{});
   auto first=selected.take(2),second=selected.take(1);
   assert(first && second);equal(*actual,*first);equal(*actual,*second);
   lease.finish();assert(selected.statistics().bytes==0);
  }
  if(!input.world_ground){
   grids=std::make_shared<std::vector<fidelity::CachedGroundGrid>>(std::move(expected->pending_grids));
   auto owner=grids;
   fidelity::GroundTask cached(queue,compile,true);owner.reset();
   auto hit=cached.take();assert(hit && hit->grid_hits>0);
   for(unsigned layer=0;layer<5;++layer)assert(hit->meshes[layer].vertices==actual->meshes[layer].vertices);
  }
  std::atomic<bool> stop{true};assert(!compile(stop));
 }
 assert(queue.statistics().active_peak==1 && queue.statistics().pending==0 && queue.statistics().bytes==0);
}
''', timeout=90)


if __name__ == "__main__":
    unittest.main()
