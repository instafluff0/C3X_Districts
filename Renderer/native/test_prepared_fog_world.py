"""Permitted fog recipes prepare without acquiring hidden native authority."""
import unittest
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp


class PreparedFogWorldTests(unittest.TestCase):
    def test_permitted_recipes_attachment_reveal_and_truthful_readiness(self):
        renderer = Path(__file__).with_name("c3x_renderer.cpp").read_text()
        start = renderer.index("auto compile_context_for=[&]")
        end = renderer.index("\n        };", start) + len("\n        };")
        context_lambda = renderer[start:end]
        program = r'''
#include "Renderer/native/render_core/world_preparation_region.h"
#include "Renderer/native/render_core/terrain_query.h"
#include <cassert>
#include <cstring>
#include <cstdio>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=64;
 f.tile_width=128;f.tile_height=64;f.target_width=2240;f.target_height=1260;
 std::vector<c3x_renderer_u32> topology(2048,2|(7<<8));
 f.world_topology=topology.data();f.world_topology_count=unsigned(topology.size());
 CapturedScene scene;assert(scene.publication_scope(f,{1,1,1,1},1));
 auto make=[](int x,int y){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;
  t.terrain_type=2;t.real_terrain_type=7;t.variant_seed=1234u^(unsigned(x)*73856093u)^(unsigned(y)*19349663u);
  t.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN|
   C3X_RENDERER_TILE_EXPLORED|CapturedScene::partial_facts_mask;
  t.city_id=17;t.city_owner_id=4;t.city_size=1;t.city_culture_group=2;t.city_era=3;
  t.road_mask=1;t.route_style=3;t.irrigation_mask=5;
  t.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION;
  t.resource_id=99;t.resource_class=2;t.has_effect=8;t.tile_building_id=72;
  t.improvement_flags|=C3X_RENDERER_IMPROVEMENT_TILE_BUILDING;t.unit_type_id=90;return t;};
 bool changed=false;
 for(int y=0;y<64;++y)for(int x=y&1;x<64;x+=2){auto t=make(x,y);assert(scene.publish(t,changed));}
 assert(scene.authoritative_size()==0);
 WorldPreparationRegion region;unsigned ready=0,selected=0;
 for(unsigned n=0;n<WorldPreparationRegion::count(f,true);++n){
  assert(region.build(scene,f,n,true));++ready;selected+=unsigned(region.selected.size());
  for(auto i:region.selected){auto const&t=region.tiles[i];
   assert((t.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH))==0);
   assert(t.resource_id==-1&&t.resource_class==-1&&!t.has_effect);
   assert(!(t.improvement_flags&C3X_RENDERER_IMPROVEMENT_TILE_BUILDING));
   assert(t.feature_flags==C3X_RENDERER_FEATURE_FOREST);
   assert(t.city_id==17&&t.road_mask==1&&t.irrigation_mask==5);
   assert(t.variant_seed==make(t.tile_x,t.tile_y).variant_seed);
  }
 }
 assert(ready==64&&selected==2048);
 assert(region.build(scene,f,0,true));auto recipe=region.tiles[region.selected.front()];
 auto key=scene.key(recipe.tile_x,recipe.tile_y);auto rev=scene.world_appearance_revision(key);
 auto& frame=f;auto& topology_cache=scene;
 auto content_tile_for=[](c3x_renderer_tile_v1 tile){return tile;};
 auto coordinate_key=[&](int x,int y){return scene.key(x,y);};
 bool world_ground=true,pickup_profile=true;int content_view_width=2240,content_view_height=1260;
 unsigned content_revision=1,device_generation=1;std::uint64_t compile_quality[]={1,1};
 /* COMPILE_CONTEXT_FIXTURE */
 auto prepared_context=compile_context_for(recipe,7);assert(prepared_context[17]==rev);
 assert(rev>0&&!scene.retained(key)->authoritative);
 scene.attach(recipe,{1,44});assert(!scene.retained(key)->compiled.generation); // ordinary path cannot promote it
 scene.attach(recipe,{1,44},true);assert(scene.retained(key)->compiled.generation==44);
 auto hostile=recipe;hostile.resource_id=9;scene.attach(hostile,{1,45},true);
 assert(scene.retained(key)->compiled.generation==44);
 auto unchanged=make(recipe.tile_x,recipe.tile_y);assert(scene.publish(unchanged,changed));
 assert(scene.world_appearance_revision(key)==rev&&scene.retained(key)->compiled.generation==44);
 // A normal foreground RENDER occurrence of unchanged explored fog must
 // validate the prepared identity before publication as well as afterward.
 // This is the native record shape when no resource/effect/border fact adds
 // to the permitted recipe; anchors, labels and unit fields are not content.
 auto foreground=recipe;foreground.tile_flags|=C3X_RENDERER_TILE_RENDER;
 foreground.anchor_x=320;foreground.anchor_y=160;foreground.unit_type_id=88;
 f.tiles=&foreground;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(foreground,2,-1,2,CapturedScene::topology(foreground)));scene.finish();
 assert(scene.appearance_revision(key)==rev&&scene.world_appearance_revision(key)==rev);
 assert(compile_context_for(foreground,7)==prepared_context);
 scene.attach(foreground,{1,44});assert(scene.retained(key)->compiled.generation==44);
 assert(!scene.retained(key)->authoritative);
 assert(scene.publish(foreground,changed));
 assert(scene.retained(key)->authoritative&&scene.retained(key)->revision==rev);
 assert(scene.world_appearance_revision(key)==rev&&scene.retained(key)->compiled.generation==44);
 assert(scene.retained(key)->compiled_views[0].generation==44);
 assert(compile_context_for(foreground,7)==prepared_context);
 // A foreground capture can legitimately add fields omitted by world-page
 // fog capture. Exact equality then fails: no stale preparation may validate.
 auto extra=foreground;extra.resource_id=7;extra.resource_class=1;
 f.tiles=&extra;assert(scene.begin(f));
 assert(scene.update(extra,2,-1,2,CapturedScene::topology(extra)));scene.finish();
 assert(scene.world_appearance_revision(key)==0);
 assert(compile_context_for(extra,7)[17]==0);
 // Restore the matching observation before testing later publication edits.
 f.tiles=&foreground;assert(scene.begin(f));
 assert(scene.update(foreground,2,-1,2,CapturedScene::topology(foreground)));scene.finish();
 f.tiles=nullptr;f.tile_count=0;assert(scene.begin(f));scene.finish();
 auto first=scene.world_snapshot();
 // Keep the partial-only invalidation cases on a separate unpublished core.
 unchanged=make(2,0);key=scene.key(2,0);rev=scene.world_appearance_revision(key);
 recipe=scene.world_snapshot()->current(key)->occurrence;
 scene.attach(recipe,{1,44},true);first=scene.world_snapshot();
 // An allowed partial city edit invalidates its prepared attachment and exact
 // world proof. The old immutable lease retains its original recipe.
 unchanged.city_size=2;assert(scene.publish(unchanged,changed));
 assert(scene.world_appearance_revision(key)!=rev&&!scene.retained(key)->compiled.generation);
 assert(first->current(key)->occurrence.city_size==1);
 assert(scene.world_snapshot()->current(key)->occurrence.city_size==2);
 // A terrain-only edit also invalidates a recipe even if record partial facts
 // and Record::revision otherwise remain unchanged.
 assert(region.build(scene,f,0,true));recipe=scene.world_snapshot()->current(key)->occurrence;
 scene.attach(recipe,{2,50},true);rev=scene.world_appearance_revision(key);
 unchanged.real_terrain_type=8;assert(scene.publish(unchanged,changed));
 assert(scene.world_appearance_revision(key)!=rev&&!scene.retained(key)->compiled.generation);
 assert(scene.world_snapshot()->current(key)->occurrence.feature_flags==C3X_RENDERER_FEATURE_JUNGLE);
 // A real full reveal may supply hidden resource facts; this is the existing
 // native-authority path and invalidates the prior permitted recipe.
 assert(region.build(scene,f,0,true));recipe=scene.world_snapshot()->current(key)->occurrence;
 scene.attach(recipe,{3,60},true);rev=scene.world_appearance_revision(key);
 unchanged.tile_flags|=C3X_RENDERER_TILE_VISIBLE|C3X_RENDERER_TILE_PREFETCH;
 unchanged.resource_id=7;unchanged.has_effect=0;assert(scene.publish(unchanged,changed));
 assert(scene.retained(key)->authoritative&&!scene.retained(key)->compiled.generation);
 assert(scene.world_appearance_revision(key)!=rev&&scene.world_snapshot()->current(key)->occurrence.resource_id==7);
 assert(first->current(key)->occurrence.resource_id==-1);
 // Unseen/unknown halo input does not poison a known explored core, and never
 // becomes a selected body. A wholly unknown core remains unavailable.
 CapturedScene sparse;assert(sparse.publication_scope(f,{1,1,1,1},1));
 auto t=make(2,2);assert(sparse.publish(t,changed));
 assert(region.build(sparse,f,0,true)&&region.selected.size()==1);
 assert(!region.build(sparse,f,63,true));
 f.world_topology=nullptr;assert(!region.build(sparse,f,0,true));f.world_topology=topology.data();
 auto unseen=make(2,2);unseen.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 assert(sparse.publish(unseen,changed));assert(!region.build(sparse,f,0,true));
 // Completion means successful preparation; failures are attempts, not ready
 // regions. Local invalidation retires each tally consistently.
 WorldPreparationSchedule q;q.configure(f,1,1,1,true);q.finish(true);q.finish(false);
 assert(q.attempted==2&&q.completed==1&&q.unavailable==1);
 while(!q.empty())q.finish(true);
 assert(q.attempted==64&&q.completed==63&&q.unavailable==1);
 q.invalidate(f,0,0);assert(q.attempted==q.completed+q.unavailable&&q.attempted<64);
 while(!q.empty())q.finish(true);
 assert(q.attempted==64&&q.completed+q.unavailable==64);
 q.configure(f,1,1,2,true);assert(!q.attempted&&!q.completed&&!q.unavailable);
 std::puts("world_recipe_contract: pass (64 regions / 2048 fog recipes, authority and lifecycle checks)");
}
'''
        run_cpp(program.replace("/* COMPILE_CONTEXT_FIXTURE */", context_lambda))


if __name__ == '__main__':
    unittest.main()
