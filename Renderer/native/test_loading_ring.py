"""The hidden ring prepared during loading survives the first view.

Loading compiles the unexplored tiles within two tiles of explored land (the
hidden ring, CapturedScene) from the loading pages. WorldInputCapture rebuilt
each unexplored page tile from the packed topology alone, dropping the map
seed (variant_seed) and the pine flag the producer supplies for it. Civ III's
first on-screen capture carries both, so every ring tile on screen changed
content: a new identity, its compiled ground cleared, and the first view
rebuilt it (145 tiles, about 0.3 s, user-3350BC scroll, October 17).
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class LoadingRingTests(unittest.TestCase):
    def test_ring_prepared_at_load_survives_civ3_capture(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_input_capture.h"
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
constexpr unsigned seed=0x5eed1234u;
// As read_custom_renderer_world_record and read_custom_renderer_tile seed tiles.
unsigned variant(int x,int y){return seed^(unsigned(x)*73856093u)^(unsigned(y)*19349663u);}
int main(){
 ScenePublication journal;CapturedScene scene;WorldInputCapture capture;WorldPreparationRegion region;
 // Grassland, with a pine forest at (2,0), one tile from the explored (0,0).
 std::vector<unsigned> topology(800,2u|(2u<<8));topology[1]=2u|(7u<<8);
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=40;
 f.tile_width=128;f.tile_height=64;f.target_width=1120;f.target_height=630;
 f.world_topology=topology.data();f.world_topology_count=800;
 assert(journal.capture(f,{1,1,1,1}));bool changed=false;assert(journal.apply(scene,changed));
 auto unexplored_record=[&](int x,int y){
  // The loading producer's record for an unexplored tile: topology, the
  // map seed and terrain features, no hidden objects.
  c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;auto top=topology[(y*40+x)/2];
  t.terrain_type=int(top&255u);t.real_terrain_type=int((top>>8)&255u);t.variant_seed=variant(x,y);
  if(t.real_terrain_type==7)t.feature_flags=C3X_RENDERER_FEATURE_FOREST|C3X_RENDERER_FEATURE_PINE;
  t.resource_id=t.resource_class=t.tile_building_id=-1;
  t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;
  t.unit_type_id=t.unit_owner_id=t.unit_class=-1;t.territory_owner_id=t.barbarian_tribe_id=-1;
  t.tile_flags=known|C3X_RENDERER_TILE_TOPOLOGY_HALO;return t;};
 while(!capture.passes){auto p=capture.page(*journal.state());p.count=std::min(p.capacity,800-p.first);
  for(unsigned i=0;i<p.count;++i){auto n=p.first+i;int y=int(n/20),x=2*int(n%20)+(y&1);
   p.tiles[i]=unexplored_record(x,y);
   if(!n)p.tiles[i].tile_flags|=explored|C3X_RENDERER_TILE_PREFETCH;}
  assert(capture.accept(p,journal)&&journal.apply(scene,changed));
 }
 // Loading prepares the ring tile from its region, as prepare_required_world does.
 auto id=scene.key(2,0);
 assert(region.build(scene,f,0,true));c3x_renderer_tile_v1 recipe{};bool selected=false;
 for(auto index:region.selected)if(region.tiles[index].tile_x==2&&region.tiles[index].tile_y==0){recipe=region.tiles[index];selected=true;}
 assert(selected && (recipe.tile_flags&CapturedScene::hidden_ring));
 std::printf("loading recipe seed=%08x features=%u\n",recipe.variant_seed,recipe.feature_flags);
 scene.attach(recipe,{1,44},true);assert(scene.retained(id)->compiled_views[0].generation==44);
 auto identity=scene.world_appearance_revision(id);
 // Civ III's first on-screen capture of the unexplored tile (full, with
 // its real seed, pine flag and owner) arrives before the first view.
 auto native=unexplored_record(2,0);
 native.tile_flags=known|C3X_RENDERER_TILE_RENDER;native.anchor_x=192;native.anchor_y=32;
 native.route_style=0;native.fog_status=0;
 auto delta=f;delta.tiles=&native;delta.tile_count=1;
 assert(journal.capture(delta,{1,1,1,1})&&journal.apply(scene,changed));
 std::printf("after capture: compiled=%llu identity=%llu->%llu\n",
  (unsigned long long)scene.retained(id)->compiled_views[0].generation,
  (unsigned long long)identity,(unsigned long long)scene.world_appearance_revision(id));
 assert(scene.retained(id)->compiled_views[0].generation==44);
 assert(scene.world_appearance_revision(id)==identity);
 // The recipe loading compiled is the terrain Civ III shows: its seed and pines.
 assert(recipe.variant_seed==variant(2,0) && (recipe.feature_flags&C3X_RENDERER_FEATURE_PINE));
 std::printf("PASS loading ring survives the first capture\n");
}
''')

    def test_unexplored_neighbour_identity_ignores_civ3_capture(self):
        # A ring tile with canopy depends on the identity of tiles up to two
        # tiles away, so ring-2 forests depend on unexplored tiles outside
        # the ring. Their identity was 0 until Civ III first captured them
        # on screen, then that capture's revision: every such ring tile was
        # rebuilt by the first view. They are never drawn and compilers read
        # only their world input, whose content revision is their identity.
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 CapturedScene scene;scene.publication_scope(f,{1,1,1,1},1);bool changed=false;
 auto tile=[](int x,int y,unsigned flags){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.terrain_type=2;t.real_terrain_type=7;t.feature_flags=C3X_RENDERER_FEATURE_FOREST;t.variant_seed=unsigned(x*31+y);
  t.barbarian_tribe_id=-1;t.city_id=t.city_owner_id=t.city_size=t.city_culture_group=t.city_era=-1;
  t.resource_id=t.resource_class=-1;t.tile_flags=flags|C3X_RENDERER_TILE_TOPOLOGY_HALO;return t;};
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2)
  assert(scene.publish(tile(x,y,x==4&&y==4?known|explored|C3X_RENDERER_TILE_PREFETCH:known),changed));
 // (10,4) is three tiles from the explored (4,4): outside the ring, within
 // two tiles of the ring forest (8,4).
 auto id=scene.key(10,4);
 assert(!(scene.world_input(id)->occurrence.tile_flags&CapturedScene::hidden_ring));
 assert(scene.world_input(scene.key(8,4))->occurrence.tile_flags&CapturedScene::hidden_ring);
 auto before=scene.world_appearance_revision(id);
 // Civ III captures it on screen, with a foreign unit and owner.
 auto native=tile(10,4,known|C3X_RENDERER_TILE_RENDER);native.tile_flags&=~C3X_RENDERER_TILE_TOPOLOGY_HALO;
 native.unit_type_id=3;native.unit_owner_id=2;native.territory_owner_id=2;
 assert(scene.publish(native,changed));
 f.tiles=&native;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(native,2,-1,2,CapturedScene::topology(native)));scene.finish();
 auto after=scene.world_appearance_revision(id);
 std::printf("unexplored neighbour identity %llu -> %llu (input %llu)\n",(unsigned long long)before,
  (unsigned long long)after,(unsigned long long)scene.world_input(id)->content_revision);
 assert(before && before==after && after==scene.world_input(id)->content_revision);
 // Unobserved, it keeps the same identity.
 f.tiles=nullptr;f.tile_count=0;assert(scene.begin(f));scene.finish();
 assert(scene.world_appearance_revision(id)==after);
 std::printf("PASS unexplored neighbour identity is its world input\n");
}
''')

    def test_stored_ring_water_moves_like_the_window(self):
        # Camera steps draw from the resident world. Its water records took
        # their visibility from native exploration, so hidden-ring water was
        # fog-frozen there while the window (EXPLORED from the ring input)
        # animated it: dark patches along coasts during a scroll, gone once
        # the view settled. Storing the ring at load made every step show it.
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        visible = method(source, '    bool resident_water_visible(int tile_x,int tile_y)const{')
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstdio>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
struct State {
 bool visibility_pass=true;CapturedScene topology_cache;
''' + visible + r'''
};
int main(){
 State state;c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=20;f.tile_width=128;f.tile_height=64;
 state.topology_cache.publication_scope(f,{1,1,1,1},1);bool changed=false;
 for(int y=0;y<20;++y)for(int x=y&1;x<20;x+=2){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;t.terrain_type=t.real_terrain_type=13;
  t.city_id=t.resource_id=t.resource_class=t.barbarian_tribe_id=-1;
  t.tile_flags=(x==4&&y==4?known|explored|C3X_RENDERER_TILE_PREFETCH:known)|C3X_RENDERER_TILE_TOPOLOGY_HALO;
  assert(state.topology_cache.publish(t,changed));}
 // The window's rule (append_tile_geometry): water moves on tiles drawn explored.
 auto window=[&](int x,int y){auto in=state.topology_cache.world_input(state.topology_cache.key(x,y));
  return (in->occurrence.tile_flags&explored)!=0;};
 for(auto at:{std::pair{4,4},std::pair{6,4},std::pair{8,4},std::pair{10,4}}){
  bool stored=state.resident_water_visible(at.first,at.second);
  std::printf("tile %d,%d window=%d stored=%d\n",at.first,at.second,int(window(at.first,at.second)),int(stored));
  assert(stored==window(at.first,at.second));
 }
 assert(state.resident_water_visible(6,4) && !state.resident_water_visible(10,4));
 state.visibility_pass=false;assert(state.resident_water_visible(10,4));
 std::printf("PASS stored ring water moves like the window\n");
}
''')


if __name__ == '__main__':
    unittest.main()
