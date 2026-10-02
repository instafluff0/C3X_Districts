"""Host-only executable contracts for the actual raster dependency producers."""
import os
import unittest
from Renderer.native.native_cpp_test import run_cpp


PRELUDE = r'''
#include "Renderer/native/render_core/raster_dependency_revisions.h"
#include <cassert>
#include <unordered_set>
using namespace c3x_renderer::render_core;
using Domain=RasterDependencyRevisions::Domain;
using Keys=std::unordered_set<RasterDependencyRevisions::Key,RasterDependencyRevisions::Hash>;
bool stable(RasterDependencyRevisions const& revisions,RasterDependencyRevisions::Checkpoint before,Keys const& keys){
 std::uint64_t visits=0;return revisions.unchanged(before,keys,visits);
}
'''


@unittest.skipUnless(os.name == "posix", "Host C++ compiler only; never dispatch Windows/VM tests")
class RasterDependencyProducerTests(unittest.TestCase):
    def test_published_local_facts_absence_and_scope(self):
        run_cpp(PRELUDE+r'''
#include "Renderer/native/render_core/captured_scene.h"
int main(){
 RasterDependencyRevisions revisions;CapturedScene scene;scene.bind_raster_dependencies(&revisions);
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;frame.world_wrap_x=1;
 c3x_renderer_camera_identity_v1 identity{};scene.publication_scope(frame,identity,1);
 c3x_renderer_tile_v1 tile{};tile.tile_x=3;tile.tile_y=5;tile.terrain_type=tile.real_terrain_type=2;
 tile.city_id=17;tile.city_size=2;tile.road_mask=3;tile.resource_id=9;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 auto key=scene.key(3,5),other=scene.key(40,40);bool changed=false;
 auto before=revisions.checkpoint();assert(scene.publish(tile,changed));
 for(auto domain:{Domain::appearance,Domain::semantic,Domain::visibility})assert(!stable(revisions,before,{{domain,key}}));
 assert(stable(revisions,before,{{Domain::appearance,other},{Domain::semantic,other},{Domain::visibility,other}}));
 before=revisions.checkpoint();tile.tile_x-=64;tile.anchor_x=1000;tile.anchor_y=-1300;tile.unit_state=99;tile.city_population=22;
 changed=false;assert(scene.publish(tile,changed));assert(!changed && revisions.checkpoint().sequence==before.sequence);
 // Hidden minimal deltas retain permitted routes/resources; only visibility changes.
 tile.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 tile.road_mask=0;tile.resource_id=-1;before=revisions.checkpoint();assert(scene.publish(tile,changed));
 assert(stable(revisions,before,{{Domain::appearance,key},{Domain::semantic,key}}));
 assert(!stable(revisions,before,{{Domain::visibility,key}}));
 before=revisions.checkpoint();tile.real_terrain_type=5;changed=false;assert(scene.publish(tile,changed));assert(changed);
 assert(!stable(revisions,before,{{Domain::semantic,key}}));
 assert(stable(revisions,before,{{Domain::appearance,key},{Domain::visibility,key},{Domain::semantic,other}}));
 // Explicit partial city absence is a local appearance change, not an all-world barrier.
 tile.tile_flags|=C3X_RENDERER_TILE_CITY_BODY_KNOWN;tile.city_id=-1;tile.city_size=-1;
 before=revisions.checkpoint();assert(scene.publish(tile,changed));
 assert(!stable(revisions,before,{{Domain::appearance,key}}));assert(stable(revisions,before,{{Domain::appearance,other}}));
 before=revisions.checkpoint();assert(scene.publish(tile,changed));assert(revisions.checkpoint().sequence==before.sequence);
 before=revisions.checkpoint();assert(!scene.publication_scope(frame,identity,1));assert(stable(revisions,before,{}));
 ++identity.viewer_epoch;assert(scene.publication_scope(frame,identity,1));assert(!stable(revisions,before,{}));
 // Rebinding a replacement owner invalidates the prior owner's certificates.
 before=revisions.checkpoint();scene={};scene.bind_raster_dependencies(&revisions);assert(!stable(revisions,before,{}));
}
''')

    def test_current_full_compatibility_changes_only_at_completed_capture(self):
        run_cpp(PRELUDE+r'''
#include "Renderer/native/render_core/captured_scene.h"
int main(){
 RasterDependencyRevisions revisions;CapturedScene scene;scene.bind_raster_dependencies(&revisions);
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
 c3x_renderer_camera_identity_v1 identity{};scene.publication_scope(frame,identity,1);
 c3x_renderer_tile_v1 tile{};tile.tile_x=3;tile.tile_y=5;tile.terrain_type=tile.real_terrain_type=2;
 tile.city_id=17;tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;
 bool changed=false;assert(scene.publish(tile,changed));auto key=scene.key(3,5),other=scene.key(5,5);
 auto capture=[&]{assert(scene.begin(frame));if(frame.tile_count)assert(scene.update(tile,2,2,2,CapturedScene::topology(tile)));scene.finish();};
 frame.tiles=&tile;frame.tile_count=1;auto before=revisions.checkpoint();capture();
 assert(stable(revisions,before,{{Domain::appearance,key}}));auto authority=scene.world_appearance_revision(key);assert(authority);
 before=revisions.checkpoint();tile.anchor_x=-444;tile.anchor_y=777;capture();assert(revisions.checkpoint().sequence==before.sequence);
 tile.city_id=18;before=revisions.checkpoint();assert(scene.begin(frame));assert(scene.update(tile,2,2,2,CapturedScene::topology(tile)));
 assert(stable(revisions,before,{{Domain::appearance,key}}));scene.finish();
 assert(!scene.world_appearance_revision(key));assert(!stable(revisions,before,{{Domain::appearance,key}}));
 assert(stable(revisions,before,{{Domain::appearance,other}}));
 before=revisions.checkpoint();capture();assert(revisions.checkpoint().sequence==before.sequence);
 // Departing a mismatched full observation restores authority, while departing
 // a compatible observation preserves the exact same effective revision.
 before=revisions.checkpoint();frame.tile_count=0;capture();assert(scene.world_appearance_revision(key)==authority);
 assert(!stable(revisions,before,{{Domain::appearance,key}}));
 tile.city_id=17;frame.tile_count=1;before=revisions.checkpoint();capture();assert(revisions.checkpoint().sequence==before.sequence);
 frame.tile_count=0;capture();assert(revisions.checkpoint().sequence==before.sequence);
 // A standalone absent key becomes usable only after its completed full capture.
 CapturedScene native;native.bind_raster_dependencies(&revisions);tile.tile_x=9;frame.tile_count=1;
 before=revisions.checkpoint();assert(native.begin(frame));assert(native.update(tile,2,2,2,42));native.finish();
 auto id=native.key(9,5);assert(!stable(revisions,before,{{Domain::appearance,id}}));
 assert(!stable(revisions,before,{{Domain::semantic,id}}));assert(!stable(revisions,before,{{Domain::visibility,id}}));
 before=revisions.checkpoint();frame.tile_count=0;assert(native.begin(frame));native.finish();assert(!stable(revisions,before,{{Domain::appearance,id}}));
}
''')

    def test_coast_node_certificates_include_missing_and_unrelated_subtrees(self):
        run_cpp(PRELUDE+r'''
#include "Renderer/native/render_core/coast_index.h"
int main(){
 RasterDependencyRevisions revisions;CoastIndex coast(0,0,16);coast.bind_raster_dependencies(&revisions);
 std::uint64_t missing=0;coast.cell(1,1,[&](auto id,auto value){assert(value==0);missing=id;});assert(missing);
 auto before=revisions.checkpoint();std::vector<CoastSegment> a={{{1.,1.},{2.,2.},.2}};
 coast.set_cell(1,1,a);assert(!stable(revisions,before,{{Domain::coast,missing}}));
 std::uint64_t leaf=0;coast.cell(1,1,[&](auto id,auto value){assert(value);leaf=id;});
 before=revisions.checkpoint();coast.set_cell(1,1,a);assert(revisions.checkpoint().sequence==before.sequence);
 coast.set_cell(12,12,{{{12.,12.},{13.,13.},.4}});assert(stable(revisions,before,{{Domain::coast,leaf}}));
 before=revisions.checkpoint();coast.set_cell(1,1,{});assert(!stable(revisions,before,{{Domain::coast,leaf}}));
 auto removed=coast.revision(leaf);assert(!removed);before=revisions.checkpoint();coast.set_cell(1,1,{});assert(revisions.checkpoint().sequence==before.sequence);
 before=revisions.checkpoint();coast.clear();assert(!stable(revisions,before,{}));
}
''')

    def test_topology_derived_flow_changes_touch_unchanged_raw_cells(self):
        run_cpp(PRELUDE+r'''
#include "Renderer/native/render_core/world_topology.h"
int main(){
 RasterDependencyRevisions revisions;WorldTopology world;world.bind_raster_dependencies(&revisions);
 World dimensions{12,12,false,false};std::vector<std::uint32_t> values(72,2u|(2u<<8));world.update(dimensions,values.data(),values.size());
 for(int c=2;c<=6;++c)values[world.index(c,2)]|=(32u<<16); // Connected horizontal native-edge chain.
 world.update(dimensions,values.data(),values.size());std::vector<unsigned> old(values.size());
 for(std::size_t i=0;i<old.size();++i)old[i]=world.river_flow(i);
 auto water=world.index(7,2);values[water]=11u|(11u<<8);auto before=revisions.checkpoint();
 world.update(dimensions,values.data(),values.size());assert(!stable(revisions,before,{{Domain::world,water}}));
 unsigned changed=0;
 for(std::size_t i=0;i<old.size();++i)if(old[i]!=world.river_flow(i)){
  ++changed;assert(!stable(revisions,before,{{Domain::flow,i}}));
  if(i!=water)assert(stable(revisions,before,{{Domain::world,i}}));
 }
 assert(changed>=4);assert(stable(revisions,before,{{Domain::world,0},{Domain::flow,0}}));
 before=revisions.checkpoint();assert(world.update(dimensions,values.data(),values.size()).empty());assert(revisions.checkpoint().sequence==before.sequence);
 bool rejected=false;try{world.update(dimensions,values.data(),values.size()-1);}catch(std::invalid_argument const&){rejected=true;}
 assert(rejected && revisions.checkpoint().sequence==before.sequence);
 world.clear();assert(!stable(revisions,before,{}));
}
''')

    def test_world_coast_binding_and_identical_revision_update(self):
        run_cpp(PRELUDE+r'''
#include "Renderer/native/render_core/world_coast.h"
int main(){
 RasterDependencyRevisions revisions;WorldCoast coast;coast.bind_raster_dependencies(&revisions);
 World dimensions{8,8,false,false};std::vector<std::uint32_t> values(32,11u|(11u<<8));
 coast.update(dimensions,values.data(),values.size(),1);auto before=revisions.checkpoint();
 auto result=coast.update(dimensions,values.data(),values.size(),2);
 assert(!result.topology_changes && !result.cells_built && revisions.checkpoint().sequence==before.sequence);
 auto index=coast.world().index(2,0);values[index]=2u|(2u<<8);before=revisions.checkpoint();
 result=coast.update(dimensions,values.data(),values.size(),3);assert(result.topology_changes==1 && result.cells_built);
 assert(!stable(revisions,before,{{Domain::world,index}}));
 assert(!stable(revisions,before,{{Domain::coast,1}})); // Empty root became a real contour tree.
 before=revisions.checkpoint();coast.clear();assert(!stable(revisions,before,{}));
}
''')


if __name__ == "__main__":
    unittest.main()
