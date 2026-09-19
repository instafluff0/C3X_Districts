"""Production ground compilation, bounded worker leases and packed adoption."""
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


class GroundPreparationTests(unittest.TestCase):
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
#include "Renderer/native/source_fidelity/prepared_ground.h"
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
 std::vector<RiverNode const*> nodes;
 auto source=[](int,unsigned,int,float,float){return 0.f;};
 auto dune=[](float,float){return 0.f;};
 auto river=[](auto const&,float,float){return 1000.f;};
 auto relief=[](float,float){return std::array<float,3>{};};
 auto weights=[](float,float){return std::array<float,5>{1,0,0,0,0};};
 auto shore=[](float,float,float,float){return 0.f;};
 auto uv=[](float u,float v,float scale){return std::array<float,2>{u*scale,v*scale};};
 auto ndc=[](float x){return x;};
 auto key=[&](int x,int y){return topology.key(x,y);};
 for(int width:{64,128,192})for(int x:{12,14,44}){
  c3x_renderer_frame_v1 frame{};frame.tile_width=width;frame.tile_height=width/2;frame.world_topology_revision=1;
  fidelity::GroundCompileInput input;input.tile.tile_x=x;input.tile.tile_y=12;
  input.tile.terrain_type=2;input.tile.real_terrain_type=2;input.tile.river_code=34;
  input.pickup_profile=input.fidelity_profile=input.river_assets_ready=true;
  input.world_ground=width>=96;input.ground=2;input.half_w=width*.5f;input.half_h=width*.25f;
  input.relief_projection_scale=width/224.f*.82f;input.flat_grid=8;input.tile_ground_grid=16;input.shadow_grid=8;
  input.retain_ground_grids=true;input.reuse_nested_ground_grids=true;
  float u=(x+12)*.5f+.5f,v=(x-12)*.5f+.5f;
  auto center=coast.sample({u,v},[](auto,auto){},[](auto,auto){});
  std::shared_ptr<std::vector<fidelity::CachedGroundGrid>> grids;
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
