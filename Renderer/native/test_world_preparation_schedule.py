"""Background world preparation priority, lifetime and cancellation contracts."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp


class WorldPreparationScheduleTests(unittest.TestCase):
    def test_bounded_readiness_route_keeps_cold_oracles_and_repeated_sweeps(self):
        preview=(ROOT/'Renderer/native/world_readiness_preview.h').read_text()
        route=preview.split('// BEGIN bounded readiness route contract (also exercised on the host).',1)[1].split(
            '// END bounded readiness route contract.',1)[0]
        run_cpp('''
#include <vector>
#include <utility>
#include <cassert>
#include <cstring>
int main(){
'''+route+r'''
 WorldReadinessRoute r;r.origins={{2,4},{12,14},{22,24},{32,34},{42,44},{52,54}};
 for(unsigned samples=12;samples<=16;++samples){
  unsigned oracles=0,sweeps=0,first=0,restore=0,repeat=0;
  for(unsigned n=0;n<samples;++n){
   oracles+=r.oracle(n);sweeps+=r.sweep_end(n);
   auto p=r.destination(n);
   if(n<6){assert(p==r.origins[n]);assert(!std::strcmp(r.phase(n),"first"));++first;}
   else {assert(p==r.origins[(n-6)%3]);
    if(n<9){assert(r.oracle(n) && !std::strcmp(r.phase(n),"restore"));++restore;}
    else {assert(!r.oracle(n) && !std::strcmp(r.phase(n),"repeat"));++repeat;}
   }
  }
  assert(first==6 && restore==3 && repeat>=3 && oracles==6 && sweeps==(samples-6)/3);
 }
}
''')

    def test_canonical_cores_keep_real_quality_identity_and_halo_guards(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
#include <set>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=64;
 f.world_wrap_x=f.world_wrap_y=1;f.tile_width=128;f.tile_height=64;
 f.target_width=2240;f.target_height=1260;
 c3x_renderer_tile_v1 view{};view.tile_x=view.tile_y=60;
 view.tile_flags=C3X_RENDERER_TILE_RENDER;f.tiles=&view;f.tile_count=1;
 WorldPreparationSchedule q;q.prioritize(f);q.configure(f,1,2,3,true,17);
 assert(q.next()==63 && WorldPreparationRegion::count(f,true)==64 && q.regions()==64);
 assert(WorldPreparationRegion::count(f)>64);
 std::set<unsigned> seen;
 while(!q.empty()){assert(seen.insert(q.next()).second);q.finish(true);}
 assert(seen.size()==64 && q.completed==64);
 // Camera dimensions and zoom do not discard canonical compiler output.
 f.target_width=800;f.target_height=600;f.tile_width=64;f.tile_height=32;
 q.configure(f,1,2,3,true,17);assert(q.empty() && q.completed==64);
 // A real compiler detail change, device, topology or asset lifetime does.
 q.configure(f,1,2,3,true,18);assert(!q.empty() && !q.completed);
 q.finish(true);q.configure(f,1,2,4,true,18);assert(!q.completed);
 q.finish(true);q.configure(f,1,3,4,true,18);assert(!q.completed);
 q.finish(true);f.world_wrap_x=0;q.configure(f,1,3,4,true,18);assert(!q.completed);
 f.world_wrap_x=1;q.configure(f,1,3,4,true,18);
 while(!q.empty())q.finish(true);
 q.invalidate(f,0,0);assert(q.completed<64 && q.completed>0);
 std::set<unsigned> invalidated;
 while(!q.empty()){assert(invalidated.insert(q.next()).second);q.finish(true);}
 // Wrapped neighbors at either seam are dependencies of the same canonical cores.
 assert(invalidated.count(0) && invalidated.count(63) && q.completed==64);
 CapturedScene scene;scene.publication_scope(f,{1,1,1,1},1);
 std::vector<std::uint32_t> topology(2048,2|(2<<8));
 f.world_topology=topology.data();f.world_topology_count=unsigned(topology.size());
 bool changed=false;WorldPreparationRegion region;
 assert(!region.build(scene,f,0,true)); // No implicit authority.
 for(int y=0;y<64;++y)for(int x=y&1;x<64;x+=2){
  c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;t.terrain_type=t.real_terrain_type=2;
  t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
  assert(scene.publish(t,changed));
 }
 assert(!region.build(scene,f,0,true)); // Explored minimal facts are not full appearance.
 for(int y=0;y<64;++y)for(int x=y&1;x<64;x+=2){
  c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;t.terrain_type=t.real_terrain_type=2;
  t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_PREFETCH;
  assert(scene.publish(t,changed));
 }
 std::set<std::uint64_t> cores;
 for(unsigned n=0;n<64;++n){assert(region.build(scene,f,n,true));assert(region.selected.size()==32);
  assert(region.tiles.size()==512); // Preserve the twelve-tile halo.
  for(auto i:region.selected){auto const& t=region.tiles[i];
   assert(t.tile_x>=0 && t.tile_x<64 && t.tile_y>=0 && t.tile_y<64);
   assert(!(t.tile_flags&C3X_RENDERER_TILE_RENDER));assert(cores.insert(scene.key(t.tile_x,t.tile_y)).second);
  }
 }
 assert(cores.size()==2048 && !region.build(scene,f,64,true));
 // Unseen halo still needs known visibility and topology, never imported art.
 c3x_renderer_tile_v1 unseen{};unseen.tile_x=unseen.tile_y=0;
 unseen.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN;assert(scene.publish(unseen,changed));
 assert(region.build(scene,f,0,true) && region.selected.size()==31);
 f.world_topology=nullptr;assert(!region.build(scene,f,0,true));
 // Tiny wrapped worlds keep the duplicate-occurrence guard and demand fallback.
 CapturedScene tiny;f.world_width_tiles=f.world_height_tiles=16;
 tiny.publication_scope(f,{1,1,1,1},1);
 for(int y=0;y<16;++y)for(int x=y&1;x<16;x+=2){
  c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_PREFETCH;
  assert(tiny.publish(t,changed));
 }
 assert(!region.build(tiny,f,0,true));
}
''')

    def test_demand_reprioritizes_without_rebuilding_completed_regions(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
#include <set>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=64;
 f.tile_width=128;f.tile_height=64;f.target_width=800;f.target_height=600;
 c3x_renderer_tile_v1 tile{};tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 tile.tile_x=tile.tile_y=4;f.tiles=&tile;f.tile_count=1;
 WorldPreparationSchedule q;q.prioritize(f);q.configure(f,1,1,1);
 assert(q.next()==0);auto cancelled=q.next();q.configure(f,1,1,1);
 assert(q.next()==cancelled && !q.completed);q.finish(true);
 tile.tile_x=tile.tile_y=60;q.prioritize(f);q.configure(f,1,1,1);
 assert(q.next()==63 && q.completed==1);std::set<unsigned> seen{0};
 while(!q.empty()){assert(seen.insert(q.next()).second);q.finish(true);q.configure(f,1,1,1);}
 assert(seen.size()==WorldPreparationRegion::count(f) && q.completed==seen.size());
 q.prioritize(f);q.configure(f,1,1,1);assert(q.empty());
 // A local edit re-arms only the cores whose halo uses that tile.
 q.invalidate(f,30,30);auto affected=seen.size()-q.completed;
 assert(affected>0 && affected<seen.size());
 while(!q.empty()){q.finish(true);q.configure(f,1,1,1);}
 assert(q.completed==seen.size());
 // Projection, device or world lifetime changes re-arm the whole schedule.
 q.configure(f,1,1,2);assert(q.completed==0);q.finish(false);assert(q.unavailable==1);
 q.configure(f,1,1,3);assert(q.completed==0 && q.unavailable==0);
 q.finish(true);f.tile_width=160;q.configure(f,1,1,3);assert(q.completed==0);
 q.finish(true);f.target_width=1024;q.configure(f,1,1,3);assert(q.completed==0);
 q.finish(true);q.configure(f,2,1,3);assert(q.completed==0);
 // Wrapped edge occurrences remain part of coverage, with no duplicate region.
 f.world_wrap_x=f.world_wrap_y=1;q.configure(f,3,1,3);seen.clear();
 while(!q.empty()){assert(seen.insert(q.next()).second);q.finish(true);}
 assert(seen.size()==WorldPreparationRegion::count(f));
}
''')


if __name__ == '__main__':
    unittest.main()
