"""A border change refreshes the borders, never the terrain built for a tile.

Borders are drawn from the current occurrence (draw records and window
tiles carry the edges and colour) and are never compiled (object_inputs).
Previously they were part of a tile's compiled identity: a city's border
edges and colour, first captured near a moving unit, replaced the appearance
of its 21 work-area tiles, cleared their compiled terrain and changed the
revision in their compile context. The job rebuilt all 21 (missing_binding=21,
about 0.4 s; unit-motion game test on user-3350BC, October 17). The world
load had captured them without borders: Civ III's palette for them is only
known after the first unit animation (is->custom_renderer_unit_images).
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp

PRELUDE = r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
#include <unordered_set>
using namespace c3x_renderer::render_core;
using Domain=RasterDependencyRevisions::Domain;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,
 visible=C3X_RENDERER_TILE_VISIBLE;
c3x_renderer_tile_v1 tile(int x,int y,unsigned flags){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
 t.terrain_type=t.real_terrain_type=2;t.barbarian_tribe_id=-1;
 // As Civ III's capture reports absent cities and resources.
 t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;t.resource_id=t.resource_class=-1;
 t.territory_owner_id=3;t.tile_flags=flags|C3X_RENDERER_TILE_TOPOLOGY_HALO;return t;}
bool touched(RasterDependencyRevisions const& raster,RasterDependencyRevisions::Checkpoint checkpoint,std::uint64_t id){
 std::unordered_set<RasterDependencyRevisions::Key,RasterDependencyRevisions::Hash> dependencies{{Domain::appearance,id}};
 std::uint64_t visits=0;return !raster.unchanged(checkpoint,dependencies,visits);}
'''


class TerritoryIdentityTests(unittest.TestCase):
    def test_border_capture_keeps_compiled_terrain_and_refreshes_borders(self):
        run_cpp(PRELUDE + r'''
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;RasterDependencyRevisions raster;scene.bind_raster_dependencies(&raster);
 scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 // The world load: explored land inside a city's borders, without edges or
 // colour. Loading prepares its terrain.
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)assert(scene.publish(tile(x,y,known|explored|C3X_RENDERER_TILE_PREFETCH),changed));
 auto id=scene.key(6,4);
 scene.attach(scene.world_input(id)->occurrence,{1,44},true);
 assert(scene.retained(id)->compiled_views[0].generation==44);
 auto identity=scene.world_appearance_revision(id);assert(identity);
 // Civ III's capture on screen carries the border edges and colour.
 auto native=tile(6,4,known|explored|visible|C3X_RENDERER_TILE_RENDER);native.anchor_x=320;native.anchor_y=96;
 native.territory_edge_mask=5;native.territory_color_rgb=0x1dd8da;
 auto checkpoint=raster.checkpoint();changed=false;
 assert(scene.publish(native,changed));
 f.tiles=&native;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(native,2,-1,2,CapturedScene::topology(native)));scene.finish();
 auto const* record=scene.retained(id);
 std::printf("identity=%llu->%llu compiled=%llu\n",(unsigned long long)identity,
  (unsigned long long)scene.world_appearance_revision(id),(unsigned long long)record->compiled_views[0].generation);
 // The terrain built for the tile and its compile identity stay.
 assert(scene.world_appearance_revision(id)==identity);
 assert(record->compiled.generation==44 && record->compiled_views[0].generation==44);
 // What draws borders is refreshed: the publication changed the tile, the
 // appearance and world input (window tiles) carry the borders, and pixels
 // retained from the tile's appearance are invalidated.
 assert(changed && touched(raster,checkpoint,id));
 assert(record->appearance.territory_edge_mask==5 && record->appearance.territory_color_rgb==0x1dd8da);
 assert(scene.world_input(id)->occurrence.territory_edge_mask==5 &&
  scene.world_input(id)->occurrence.territory_color_rgb==0x1dd8da);
 // A compile from a frame still holding the old borders keeps its binding.
 auto stale=native;stale.territory_edge_mask=stale.territory_color_rgb=0;scene.attach(stale,{1,45});
 assert(scene.retained(id)->compiled_views[0].generation==45);
 // Borders move again (a neighbour's culture), then vanish: one identity.
 for(unsigned edges:{9u,0u}){native.territory_edge_mask=edges;checkpoint=raster.checkpoint();changed=false;
  assert(scene.publish(native,changed) && changed && touched(raster,checkpoint,id));
  assert(scene.world_appearance_revision(id)==identity && scene.retained(id)->compiled_views[0].generation==45);}
 // A compiled fact (a road) still rebuilds the tile.
 native.road_mask=1;assert(scene.publish(native,changed));
 assert(scene.world_appearance_revision(id)!=identity);
 assert(!scene.retained(id)->compiled.generation && !scene.retained(id)->compiled_views[0].generation);
 std::printf("PASS borders keep compiled terrain\n");
}
''')

    def test_observed_borders_ahead_of_their_publication_keep_identity(self):
        # A frame's capture can carry borders the published record does not
        # have yet; it describes the same terrain, so its compile context and
        # its neighbours' proofs must not see a different (zero) identity.
        run_cpp(PRELUDE + r'''
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)assert(scene.publish(tile(x,y,known|explored|C3X_RENDERER_TILE_PREFETCH),changed));
 auto id=scene.key(6,4);auto identity=scene.world_appearance_revision(id);
 auto native=tile(6,4,known|explored|visible|C3X_RENDERER_TILE_RENDER);
 native.territory_edge_mask=5;native.territory_color_rgb=0x1dd8da;
 f.tiles=&native;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(native,2,-1,2,CapturedScene::topology(native)));scene.finish();
 std::printf("observed identity=%llu (record %llu)\n",(unsigned long long)scene.world_appearance_revision(id),(unsigned long long)identity);
 assert(scene.world_appearance_revision(id)==identity);
 scene.attach(native,{1,46});assert(scene.retained(id)->compiled_views[0].generation==46);
}
''')

    def test_revealing_land_inside_borders_keeps_hidden_ring_terrain(self):
        # Hidden-ring terrain is compiled before the tile is explored, from an
        # input without borders. Revealing a tile inside a neighbour's borders
        # differs from it only in the borders: the reveal reuses that terrain.
        run_cpp(PRELUDE + r'''
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;RasterDependencyRevisions raster;scene.bind_raster_dependencies(&raster);
 scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)
  assert(scene.publish(tile(x,y,x==4&&y==4?known|explored|C3X_RENDERER_TILE_PREFETCH:known),changed));
 auto id=scene.key(6,4);auto recipe=scene.world_input(id)->occurrence;
 assert(recipe.tile_flags&CapturedScene::hidden_ring);
 scene.attach(recipe,{1,47},true);auto identity=scene.world_appearance_revision(id);
 auto revealed=tile(6,4,known|explored|visible|C3X_RENDERER_TILE_RENDER);
 revealed.territory_edge_mask=2;revealed.territory_color_rgb=0x1dd8da;
 auto checkpoint=raster.checkpoint();assert(scene.publish(revealed,changed));
 std::printf("reveal identity=%llu->%llu compiled=%llu\n",(unsigned long long)identity,
  (unsigned long long)scene.world_appearance_revision(id),(unsigned long long)scene.retained(id)->compiled_views[0].generation);
 assert(scene.world_appearance_revision(id)==identity && scene.retained(id)->compiled_views[0].generation==47);
 assert(touched(raster,checkpoint,id) && scene.retained(id)->appearance.territory_edge_mask==2);
}
''')


if __name__ == '__main__':
    unittest.main()
