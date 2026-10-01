"""Execute native field authority and immutable world adoption on the host."""
import os
import shutil
import unittest
from unittest.mock import patch

from Renderer.native.native_cpp_test import run_cpp


def host_cpp(program):
    if os.name != 'posix' or not (shutil.which('clang++') or shutil.which('g++')):
        raise unittest.SkipTest('Portable host C++ compiler required')
    assert '#include <windows.h>' not in program
    with patch('Renderer.native.native_cpp_test.native_command_result',
               side_effect=AssertionError('Windows dispatch prohibited')), \
         patch('Renderer.native.native_cpp_test.windows_root',
               side_effect=AssertionError('Windows paths prohibited')):
        run_cpp(program)


PREFIX = r'''
#include "Renderer/native/render_core/scene_publication.h"
#include "Renderer/native/render_core/world_preparation_region.h"
#include <cassert>
using namespace c3x_renderer::render_core;
static_assert(!(CapturedScene::partial_facts_mask&C3X_RENDERER_TILE_VISIBILITY_BITS),"field authority is not visibility");
c3x_renderer_frame_v1 frame(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=64;
 f.tile_width=128;f.tile_height=64;f.target_width=2240;f.target_height=1260;return f;
}
c3x_renderer_tile_v1 partial(int x=2,int y=2){
 c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;t.terrain_type=t.real_terrain_type=2;
 t.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN|
  C3X_RENDERER_TILE_EXPLORED|CapturedScene::partial_facts_mask;
 t.city_id=17;t.city_owner_id=4;t.city_size=1;t.city_culture_group=2;t.city_era=3;
 t.city_population=12;t.city_flags=C3X_RENDERER_CITY_WALLED;
 std::strcpy(t.city_owner,"native owner");std::strcpy(t.city_civilization,"native civilization");
 std::strcpy(t.city_era_name,"native era");
 t.road_mask=1;t.railroad_mask=1;t.route_style=3;t.irrigation_mask=5;
 t.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION|
  C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP;t.barbarian_tribe_id=7;
 return t;
}
void observe(CapturedScene& scene,c3x_renderer_frame_v1 f,c3x_renderer_tile_v1& t){
 f.tiles=&t;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(t,2,2,2,CapturedScene::topology(t)));scene.finish();
}
void submit(ScenePublication& journal,CapturedScene& scene,c3x_renderer_frame_v1 f,
            c3x_renderer_tile_v1& t,bool& changed,std::vector<std::pair<int,int>>* dirty=nullptr){
 f.tiles=&t;f.tile_count=1;assert(journal.capture(f,{}));changed=false;assert(journal.apply(scene,changed,dirty));
}
'''


class PartialWorldFactsTests(unittest.TestCase):
    def test_initial_partial_facts_persist_without_full_art_and_local_edits_keep_old_leases(self):
        host_cpp(PREFIX + r'''
int main(){
 ScenePublication journal;CapturedScene scene;auto f=frame();auto t=partial();bool changed=false;
 // Unrelated live fields are deliberately hostile: a marked city/overlay
 // capture cannot turn them into authoritative art.
 t.resource_id=99;t.resource_class=2;t.has_effect=8;t.tile_building_id=72;
 t.improvement_flags|=C3X_RENDERER_IMPROVEMENT_TILE_BUILDING;t.unit_type_id=90;
 submit(journal,scene,f,t,changed);assert(changed);auto key=scene.key(2,2);
 assert(!scene.retained(key)->authoritative && scene.authoritative_size()==0);
 auto first=scene.world_snapshot();auto a=first->current(key);assert(a);
 assert(a->occurrence.city_id==17 && a->occurrence.city_owner_id==4 && a->occurrence.city_flags==C3X_RENDERER_CITY_WALLED);
 assert(a->occurrence.road_mask==1 && a->occurrence.railroad_mask==1 && a->occurrence.route_style==3);
 assert(a->occurrence.irrigation_mask==5 && a->occurrence.barbarian_tribe_id==7);
 assert(!(a->occurrence.improvement_flags&C3X_RENDERER_IMPROVEMENT_TILE_BUILDING));
 assert(a->occurrence.resource_id==-1 && a->occurrence.resource_class==-1 && !a->occurrence.has_effect);
 assert(!a->occurrence.tile_building_id && !a->occurrence.unit_type_id);
 assert(!(a->occurrence.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)));
 assert((a->occurrence.tile_flags&CapturedScene::partial_facts_mask)==CapturedScene::partial_facts_mask);
 assert(a->occurrence.city_population==0 && !a->occurrence.city_owner[0]);
 WorldPreparationRegion region;assert(!region.build(scene,f,0)); // Missing full art stays closed.
 auto other=partial(18,2);submit(journal,scene,f,other,changed);first=scene.world_snapshot();
 auto unchanged=first->current(first->key(18,2));auto revision=scene.retained(key)->revision;
 auto sequence=scene.world_input_sequence();
 // An unmarked minimal fog delta retains both independently admitted facts.
 auto minimal=t;minimal.tile_flags&=~CapturedScene::partial_facts_mask;
 minimal.city_id=-1;minimal.city_size=-1;minimal.city_flags=0;
 minimal.road_mask=minimal.railroad_mask=minimal.irrigation_mask=minimal.improvement_flags=0;
 minimal.barbarian_tribe_id=-1;
 submit(journal,scene,f,minimal,changed);assert(!changed && scene.retained(key)->revision==revision);
 assert(scene.world_input_sequence()==sequence && scene.world_snapshot()==first);
 // A later marked native recipe updates the same fields and only its local
 // dependency closure. An older worker still reads its immutable inputs.
 t.city_size=2;t.road_mask=0;t.railroad_mask=0;t.irrigation_mask=0;
 t.improvement_flags=C3X_RENDERER_IMPROVEMENT_POLLUTION;t.barbarian_tribe_id=-1;
 std::vector<std::pair<int,int>> dirty;submit(journal,scene,f,t,changed,&dirty);
 assert(changed && dirty.size()==1 && dirty[0]==std::make_pair(2,2));
 auto next=scene.world_snapshot();assert(next!=first && scene.retained(key)->revision>revision);
 assert(next->current(key)->occurrence.city_size==2 && !next->current(key)->occurrence.road_mask);
 assert(next->current(key)->occurrence.improvement_flags==C3X_RENDERER_IMPROVEMENT_POLLUTION);
 assert(first->current(key)->occurrence.city_size==1 && first->current(key)->occurrence.road_mask==1);
 assert(next->current(next->key(18,2))==unchanged); // Unchanged region is shared.
 assert(!scene.retained(key)->authoritative && !(next->current(key)->occurrence.tile_flags&C3X_RENDERER_TILE_PREFETCH));
 // Literal visible halo routes still agree with their semantic dependency,
 // even after another field-level capture established remembered overlays.
 auto visible=t;visible.tile_flags&=~CapturedScene::partial_facts_mask;
 visible.tile_flags|=C3X_RENDERER_TILE_VISIBLE;visible.road_mask=1;
 submit(journal,scene,f,visible,changed);
 auto literal=scene.world_snapshot()->current(key);assert(literal->occurrence.road_mask==1);
 assert(literal->semantic==CapturedScene::topology(visible) && literal->occurrence.city_size==2);
 WorldPreparationSchedule schedule;schedule.configure(f,1,1,1);
 while(!schedule.empty())schedule.finish(true);auto total=schedule.completed;
 for(auto const& tile:dirty)schedule.invalidate(f,tile.first,tile.second);
 assert(total==64 && schedule.completed==60); // Exact existing 8-core/12-halo closure.
 // Unexplored input cannot acquire partial authority even if a caller marks it.
 auto unseen=partial(30,2);unseen.tile_flags&=~C3X_RENDERER_TILE_EXPLORED;
 submit(journal,scene,f,unseen,changed);auto unknown=scene.world_snapshot()->current(scene.key(30,2));
 assert(unknown->occurrence.city_id==-1 && !(unknown->occurrence.tile_flags&CapturedScene::partial_facts_mask));
 scene={};assert(first->current(key)->occurrence.city_size==1 && next->current(key)->occurrence.city_size==2);
}
''')

    def test_marked_city_removal_and_native_overlay_updates_preserve_unrelated_full_art(self):
        host_cpp(PREFIX + r'''
int main(){
 ScenePublication journal;CapturedScene scene;auto f=frame();auto t=partial();bool changed=false;
 t.tile_flags|=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;
 t.resource_id=3;t.resource_class=1;t.has_effect=1;t.improvement_flags|=C3X_RENDERER_IMPROVEMENT_TILE_BUILDING;
 submit(journal,scene,f,t,changed);observe(scene,f,t);auto key=scene.key(2,2);scene.attach(t,{1,9});
 auto revision=scene.retained(key)->revision,sequence=scene.world_input_sequence();auto first=scene.world_snapshot();
 auto p=t;p.tile_flags&=~(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE);
 p.city_population=99;std::strcpy(p.city_owner,"updated native label");std::strcpy(p.city_era_name,"updated era label");
 // Population/labels are copied fields, but do not change the static recipe.
 std::vector<std::pair<int,int>> dirty;submit(journal,scene,f,p,changed,&dirty);
 assert(!changed && dirty.size()==1); // The first fog visibility change has its own local closure.
 sequence=scene.world_input_sequence();first=scene.world_snapshot();dirty.clear();
 p.city_population=100;std::strcpy(p.city_civilization,"another native label");
 submit(journal,scene,f,p,changed,&dirty);
 assert(!changed && dirty.empty() && scene.retained(key)->revision==revision);
 assert(scene.retained(key)->compiled.generation==9 && scene.world_input_sequence()==sequence && scene.world_snapshot()==first);
 // Omitted fields remain remembered when only city authority is supplied.
 p.tile_flags&=~C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN;
 p.city_id=p.city_owner_id=p.city_size=p.city_culture_group=p.city_era=-1;
 p.city_flags=p.city_population=0;std::memset(p.city_owner,0,sizeof(p.city_owner));
 p.resource_id=90;p.has_effect=90;p.improvement_flags=p.road_mask=p.irrigation_mask=0;
 submit(journal,scene,f,p,changed,&dirty);assert(changed && dirty.size()==1);
 auto record=scene.retained(key);assert(record->revision>revision && !record->compiled.generation);
 assert(record->appearance.city_id==-1 && record->appearance.resource_id==3 && record->appearance.has_effect==1);
 assert(record->appearance.road_mask==1 && record->appearance.irrigation_mask==5);
 assert(record->appearance.improvement_flags&C3X_RENDERER_IMPROVEMENT_TILE_BUILDING);
 assert(record->appearance.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE);
 assert(first->current(key)->occurrence.city_id==17);
 // Explicit native overlay absence removes its whitelist, preserving raw
 // tile-building/effect/resource policy. A subsequent full copy remains full.
 p.tile_flags|=C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN;
 p.railroad_mask=0;p.barbarian_tribe_id=-1;p.improvement_flags=0;dirty.clear();
 submit(journal,scene,f,p,changed,&dirty);assert(changed && dirty.size()==1);
 record=scene.retained(key);assert(record->appearance.improvement_flags==C3X_RENDERER_IMPROVEMENT_TILE_BUILDING);
 assert(!record->appearance.road_mask && !record->appearance.railroad_mask && !record->appearance.irrigation_mask);
 assert(record->appearance.resource_id==3 && record->appearance.has_effect==1 && record->authoritative);
 t.resource_id=4;t.has_effect=0;t.improvement_flags=0;
 submit(journal,scene,f,t,changed);assert(changed && scene.retained(key)->appearance.resource_id==4);
 assert(!scene.retained(key)->appearance.has_effect && !scene.retained(key)->appearance.improvement_flags);
}
''')

    def test_coalesced_partial_authority_and_standalone_world_inputs(self):
        host_cpp(PREFIX + r'''
int main(){
 auto f=frame();f.world_wrap_x=1;ScenePublication journal;CapturedScene scene;
 auto t=partial();t.tile_flags|=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;t.resource_id=3;t.has_effect=1;
 f.tiles=&t;f.tile_count=1;assert(journal.capture(f,{}));
 auto p=partial(-62,2);p.city_id=25;p.road_mask=0;p.resource_id=99;
 f.tiles=&p;assert(journal.capture(f,{}));auto minimal=p;minimal.tile_flags&=~CapturedScene::partial_facts_mask;
 minimal.city_id=-1;minimal.road_mask=1;minimal.improvement_flags=0;
 f.tiles=&minimal;assert(journal.capture(f,{}));bool changed=false;assert(journal.apply(scene,changed));
 auto key=scene.key(2,2);auto a=scene.world_snapshot()->current(key);
 assert(a->occurrence.city_id==25 && a->occurrence.road_mask==0);
 assert(a->occurrence.resource_id==3 && a->occurrence.has_effect==1 && scene.retained(key)->authoritative);
 // Standalone native observations also derive canonical per-field inputs,
 // but never mistake a partial revision for a standalone full copy.
 CapturedScene standalone;auto only=partial();observe(standalone,frame(),only);
 auto lease=standalone.world_snapshot();auto b=lease->current(standalone.key(2,2));
 assert(b->occurrence.city_id==17 && b->occurrence.road_mask==1);
 assert(!standalone.retained(standalone.key(2,2))->authoritative);
 assert(!(b->occurrence.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)));
 only.city_id=-1;only.city_flags=0;only.road_mask=0;observe(standalone,frame(),only);
 auto latest=standalone.world_snapshot();assert(latest!=lease);
 assert(latest->current(standalone.key(2,2))->occurrence.city_id==-1);
 assert(lease->current(lease->key(2,2))->occurrence.city_id==17);
 // Native camera data remains independent of a published partial authority.
 p.city_id=999;observe(scene,f,p);assert(scene.current(key)->occurrence.city_id==999);
 assert(scene.world_snapshot()->current(key)->occurrence.city_id==25);
}
''')


if __name__ == '__main__':
    unittest.main()
