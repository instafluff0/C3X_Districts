"""Owned scene content and capture eligibility use the production owner."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class CapturedSceneTests(unittest.TestCase):
    def test_owned_appearance_revisions_and_current_lookup(self):
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <vector>
using c3x_renderer::render_core::CapturedScene;
int main(){
 CapturedScene scene;
 c3x_renderer_tile_v1 full={};full.tile_x=2;full.tile_y=2;
 full.tile_flags=C3X_RENDERER_TILE_RENDER;full.city_id=17;full.city_size=1;
 full.resource_id=3;full.terrain_type=2;full.real_terrain_type=2;
 std::vector<c3x_renderer_tile_v1> input={full};
 c3x_renderer_frame_v1 f={};f.world_width_tiles=f.world_height_tiles=100;
 f.world_wrap_x=f.world_wrap_y=1;
 auto submit=[&]{f.tiles=input.data();f.tile_count=unsigned(input.size());
  assert(scene.begin(f));for(auto const& t:input)if(t.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_TOPOLOGY_HALO))
   assert(scene.update(t,t.terrain_type,t.real_terrain_type,t.real_terrain_type,t.road_mask+100));scene.finish();};
 submit();auto key=scene.key(2,2);assert(scene.appearance_revision(key));auto revision=scene.retained(key)->revision;
 assert(revision && scene.current(key)->occurrence.city_id==17);
 scene.attach(full,{2,10});assert(scene.retained(key)->compiled.generation==10);
 scene.attach(full,{3,11});scene.attach(full,{4,12});scene.attach(full,{2,10});
 assert(scene.retained(key)->compiled_views[0].generation==10);
 assert(scene.retained(key)->compiled_views[1].generation==12);
 assert(scene.retained(key)->compiled_views[2].generation==11);
 // Current lookup owns the entire selected record, including authoritative anchors.
 input[0].city_id=99;assert(scene.current(key)->occurrence.city_id==17);
 input[0]=full;input[0].tile_x-=100;input[0].anchor_x=900;
 input[0].visibility_mask=88;input[0].unit_state=9;
 input[0].city_population=23;input[0].square_parts=44;input[0].terrain_overlays=99;
 input[0].tile_building_id=123;std::strcpy(input[0].city_owner,"native label");
 submit();assert(scene.retained(key)->revision==revision);
 assert(scene.retained(key)->compiled.generation==10);
 assert(scene.current(key)->occurrence.tile_x==-98 && scene.current(key)->occurrence.anchor_x==900);
 // Full appearance survives a following lightweight halo; current lookup still
 // has the full appearance regardless of halo order.
 auto halo=full;halo.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
 halo.city_id=-1;halo.city_size=0;halo.resource_id=-1;halo.anchor_x=-42;
 input.push_back(halo);submit();
 scene.attach(halo,{3,20});assert(scene.retained(key)->compiled.generation==10);
 assert(scene.current(key)->occurrence.city_id==17 && scene.current(key)->occurrence.anchor_x==900);
 assert(scene.retained(key)->appearance.city_id==17 && scene.retained(key)->revision==revision);
 std::reverse(input.begin(),input.end());submit();
 assert(scene.current(key)->occurrence.city_id==17 && scene.retained(key)->revision==revision);
 // Camera departure keeps content but revokes eligibility, not just draw flags.
 input.clear();submit();assert(!scene.current(key) && !scene.appearance_revision(key));assert(scene.retained(key)->revision==revision);
 auto edited=full;edited.city_id=-1;edited.resource_id=-1;edited.road_mask=8;
 edited.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_PREFETCH;
 input={edited};submit();assert(scene.retained(key)->revision>revision);
 auto removed_revision=scene.retained(key)->revision;
 assert(!scene.retained(key)->compiled.generation);
 for(auto handle:scene.retained(key)->compiled_views)assert(!handle.generation);
 assert(scene.retained(key)->appearance.city_id==-1 && scene.current(key)->semantic==108);
 assert(!(scene.current(key)->occurrence.tile_flags&C3X_RENDERER_TILE_RENDER));
 input[0].anchor_y=123;submit();assert(scene.retained(key)->revision==removed_revision);
 // A cancelled/incomplete update exposes no partially rebuilt current lookup.
 assert(scene.begin(f));assert(!scene.current(key));
 assert(scene.update(full,2,2,2,100));assert(!scene.current(key));
 submit();assert(scene.current(key)->occurrence.city_id==-1);
 // New world basis cannot keep prior eligibility/content identities.
 f.world_width_tiles=102;submit();assert(scene.retained(key)->revision>removed_revision);
 scene={};assert(!scene.retained(key) && !scene.current(key));
}
''')

    def test_world_identity_survives_more_than_one_capture_and_mesh_eviction(self):
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <vector>
using c3x_renderer::render_core::CapturedScene;
int main(){
 CapturedScene scene;std::vector<c3x_renderer_tile_v1> input(CapturedScene::occurrence_limit);
 c3x_renderer_frame_v1 f={};f.world_width_tiles=f.world_height_tiles=2048;
 for(std::size_t i=0;i<input.size();++i){auto& t=input[i];t.tile_x=int(i%1024)*2;t.tile_y=int(i/1024)*2;t.tile_flags=C3X_RENDERER_TILE_RENDER;t.resource_id=int(i);}
 auto submit=[&]{f.tiles=input.data();f.tile_count=unsigned(input.size());assert(scene.begin(f));
 for(auto const& t:input)assert(scene.update(t,2,2,2,100));scene.finish();};
 submit();assert(scene.size()==CapturedScene::occurrence_limit);
 assert(scene.bytes()<16u*1024u*1024u);
 auto last=input.back();auto keep=scene.key(last.tile_x,last.tile_y);auto revision=scene.retained(keep)->revision;
 auto evicted=scene.key(0,0);auto old_revision=scene.retained(evicted)->revision;
 auto fresh=last;fresh.tile_y=1000;
 input={fresh,last};submit();assert(scene.size()==CapturedScene::occurrence_limit+1 && scene.retained(evicted));
 assert(!scene.current(evicted) && scene.retained(evicted)->revision==old_revision);
 assert(scene.retained(keep)->revision==revision);
 input[0]={};input[0].tile_flags=C3X_RENDERER_TILE_RENDER;submit();
 assert(scene.retained(evicted)->revision==old_revision);
 assert(scene.bytes()<16u*1024u*1024u);
 f.tile_count=CapturedScene::occurrence_limit+1;assert(!scene.begin(f));assert(!scene.current(keep));
}
''')

    def test_content_handles_expire_on_eviction_slot_reuse_and_reset(self):
        run_cpp(r'''
#include "Renderer/native/render_core/resident_content.h"
#include <cassert>
#include <memory>
using c3x_renderer::render_core::ResidentContent;
int main(){
 struct Buffer {int* count;explicit Buffer(int& n):count(&n){++*count;}~Buffer(){--*count;}};
 int live=0;ResidentContent<Buffer> owner(3);
 auto a=std::make_unique<Buffer>(live),b=std::make_unique<Buffer>(live),c=std::make_unique<Buffer>(live);
 auto first=owner.bind(*a),second=owner.bind(*b),third=owner.bind(*c);
 assert(owner.resolve(first)==a.get() && owner.resolve(second)==b.get());
 assert(!owner.bind(*a).generation && live==3); // capacity cannot allocate another binding
 owner.release(second);b.reset();assert(live==2 && !owner.resolve(second));
 b=std::make_unique<Buffer>(live);auto replacement=owner.bind(*b);
 assert(replacement.slot==second.slot && replacement.generation!=second.generation);
 assert(!owner.resolve(second) && owner.resolve(replacement)==b.get());
 owner.release(second);assert(owner.resolve(replacement)==b.get()); // stale release cannot erase replacement
 owner.clear();assert(live==3 && !owner.resolve(first) && !owner.resolve(replacement) && !owner.resolve(third));
 auto after_reset=owner.bind(*a);assert(!(after_reset==first) && owner.resolve(after_reset)==a.get());
 assert(!owner.resolve(first));owner.release(after_reset);a.reset();assert(live==2);
 for(int i=0;i<10000;++i){auto h=owner.bind(*b);assert(owner.resolve(h)==b.get());owner.release(h);assert(!owner.resolve(h));}
 assert(owner.bytes()<1024); // repeated replacement does not grow the slot table
 owner.clear();assert(owner.bytes()==0 && live==2);
}
''')


if __name__ == "__main__":
    unittest.main()
