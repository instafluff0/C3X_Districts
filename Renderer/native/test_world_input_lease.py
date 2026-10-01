"""Execute immutable world leases with a host compiler, without Windows tools."""
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
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <type_traits>
using namespace c3x_renderer::render_core;
static_assert(!std::is_copy_constructible<CapturedScene>::value,"one mutable publication owner");
c3x_renderer_tile_v1 full(int x,int y,int real=2){
 c3x_renderer_tile_v1 tile{};tile.tile_x=x;tile.tile_y=y;
 tile.terrain_type=2;tile.real_terrain_type=real;tile.city_id=17;tile.resource_id=3;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|
  C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;return tile;
}
c3x_renderer_frame_v1 frame(){
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=64;
 f.world_wrap_x=1;f.tile_width=128;f.tile_height=64;return f;
}
void publish(CapturedScene& scene,c3x_renderer_tile_v1 const& tile){
 bool changed=false;assert(scene.publish(tile,changed));
}
'''


class WorldInputLeaseTests(unittest.TestCase):
    def test_compilation_adapter_preserves_native_anchors_without_camera_copy(self):
        host_cpp(PREFIX + r'''
int main(){
 CapturedScene scene;auto f=frame();c3x_renderer_camera_identity_v1 identity{};
 scene.publication_scope(f,identity,1);auto a=full(2,2),b=full(4,2);
 publish(scene,a);publish(scene,b);auto allowance=scene.world_snapshot_allowance();assert(allowance>0);
 a.anchor_x=312;a.anchor_y=124;b.tile_x-=64;b.anchor_x=361;b.anchor_y=155;
 c3x_renderer_tile_v1 tiles[]={a,b};f.tiles=tiles;f.tile_count=2;
 assert(scene.begin(f));for(auto const& tile:tiles)assert(scene.update(tile,-1,-1,-1,CapturedScene::topology(tile)));scene.finish();
 auto bytes=scene.bytes();auto native=scene.compilation_view(false);
 assert(scene.bytes()==bytes && scene.world_snapshot_allowance()==allowance); // No world or camera snapshot created.
 auto id=native.key(66,2),neighbor=native.key(4,2);
 assert(native.current(id)==scene.current(id));
 assert(native.current(id)->occurrence.anchor_x==312 && native.current(id)->occurrence.anchor_y==124);
 assert(native.current(neighbor)->occurrence.tile_x==-60 && native.current(neighbor)->occurrence.anchor_x==361);
 assert(native.current(neighbor)->occurrence.anchor_x-native.current(id)->occurrence.anchor_x==49);
 assert(native.current(neighbor)->occurrence.anchor_y-native.current(id)->occurrence.anchor_y==31);
 auto world=scene.compilation_view(true);auto source=scene.world_snapshot();auto sequence=scene.world_input_sequence();
 assert(world.key(66,2)==id && world.current(id)==source->current(id));
 assert(world.current(id)->occurrence.anchor_x==0 && world.current(id)->occurrence.anchor_y==0);
 assert(world.current(neighbor)->occurrence.tile_x==4 && world.current(neighbor)->occurrence.anchor_x==0);
 assert(world.current(id)->ground==2 && native.current(id)->ground==-1);
 // The foreground lease has ended before this camera capture. Native mode
 // follows its owner; world mode continues to borrow the same immutable source.
 tiles[0].anchor_x=912;tiles[1].anchor_x=961;assert(scene.begin(f));
 assert(!native.current(id) && world.current(id)==source->current(id));
 for(auto const& tile:tiles)assert(scene.update(tile,-1,-1,-1,CapturedScene::topology(tile)));scene.finish();
 assert(native.current(id)->occurrence.anchor_x==912 && native.current(neighbor)->occurrence.anchor_x==961);
 assert(scene.world_snapshot()==source && scene.world_input_sequence()==sequence);
 f.tile_count=0;assert(scene.begin(f));scene.finish();
 assert(!native.current(id) && world.current(id)==source->current(id));
 source.reset();scene={};assert(world.current(id)->occurrence.tile_x==2 && world.current(id)->occurrence.anchor_x==0);
}
''')

    def test_camera_and_gpu_bindings_reuse_world_and_local_edits_copy_one_region(self):
        host_cpp(PREFIX + r'''
#include <cstdlib>
bool fail_next_allocation=false;
void* operator new(std::size_t size){
 if(fail_next_allocation){fail_next_allocation=false;throw std::bad_alloc();}
 if(auto memory=std::malloc(size?size:1))return memory;throw std::bad_alloc();
}
void operator delete(void* memory)noexcept{std::free(memory);}
int main(){
 CapturedScene scene;auto f=frame();c3x_renderer_camera_identity_v1 identity{};
 assert(scene.publication_scope(f,identity,1));
 auto a=full(2,2,5),same_region=full(4,4),other_region=full(18,2);
 publish(scene,a);publish(scene,same_region);publish(scene,other_region);
 auto first=scene.world_snapshot();auto sequence=scene.world_input_sequence();auto id=first->key(66,2);
 assert(id==scene.key(2,2) && first->current(id));
 auto same=first->current(first->key(4,4)),other=first->current(first->key(18,2));
 assert(first->current(id)->occurrence.anchor_x==0 && first->current(id)->ground==2);
 assert(first->current(id)->relief==5 && first->current(id)->surface==5);
 assert(!(first->current(id)->occurrence.tile_flags&C3X_RENDERER_TILE_RENDER));
 assert(first->current(id)->occurrence.tile_flags&C3X_RENDERER_TILE_PREFETCH);
 assert(scene.world_snapshot_allowance()==0);
 // A stale or differently admitted native occurrence cannot rewrite published
 // world inputs; its native anchors and classification still stay available.
 a.tile_x-=64;a.anchor_x=900;a.anchor_y=-10;
 f.tiles=&a;f.tile_count=1;assert(scene.begin(f));
 assert(scene.update(a,-1,-1,-1,123));scene.finish();
 assert(scene.current(id)->occurrence.anchor_x==900 && scene.current(id)->ground==-1);
 assert(scene.world_snapshot()==first && scene.world_input_sequence()==sequence && first->current(id)->semantic==CapturedScene::topology(a));
 scene.attach(a,{1,12});assert(scene.retained(id)->compiled.generation==12);
 assert(scene.world_snapshot()==first && scene.world_input_sequence()==sequence);
 f.tile_count=0;assert(scene.begin(f));scene.finish();
 assert(!scene.current(id) && scene.world_appearance_revision(id));
 assert(scene.world_snapshot()==first);
 auto edited=full(2,2,5);edited.city_id=99;edited.road_mask=8;publish(scene,edited);
 assert(scene.world_snapshot_allowance()>0 && scene.world_input_sequence()>sequence);auto second=scene.world_snapshot();
 sequence=scene.world_input_sequence();publish(scene,edited);
 assert(scene.world_input_sequence()==sequence && scene.world_snapshot()==second);
 assert(second!=first && second->scope_sequence()==first->scope_sequence());
 assert(second->current(id)->occurrence.city_id==99 && first->current(id)->occurrence.city_id==17);
 assert(second->current(id)->semantic!=first->current(id)->semantic);
 assert(second->current(second->key(4,4))!=same); // One block copied, not one world.
 assert(second->current(second->key(18,2))==other);
 assert(!scene.retained(id)->compiled.generation);
 // A stale full observation may not validate art even with a current world lease.
 f.tiles=&a;f.tile_count=1;assert(scene.begin(f));assert(scene.update(a,2,5,5,123));scene.finish();
 assert(!scene.world_appearance_revision(id) && scene.world_snapshot()==second);
 auto before=scene.bytes();first.reset();assert(scene.bytes()<before);
 auto stable=scene.bytes();for(unsigned i=0;i<100;++i){assert(scene.world_snapshot()==second);}
 assert(scene.bytes()==stable);
 // A failed root-table allocation cannot mutate a published lease or leak its
 // charge. Staged local changes remain available for a bounded retry.
 edited.road_mask=16;publish(scene,edited);auto staged=scene.bytes();
 fail_next_allocation=true;bool failed=false;try{scene.world_snapshot();}catch(std::bad_alloc const&){failed=true;}
 assert(failed && scene.bytes()==staged && second->current(id)->occurrence.road_mask==8);
 auto retry=scene.world_snapshot();assert(retry->current(id)->occurrence.road_mask==16);
 assert(retry->current(retry->key(18,2))==other);
}
''')

    def test_absence_reveal_and_latest_semantic_proof_use_actual_validator(self):
        host_cpp(PREFIX + r'''
#include "Renderer/native/render_core/prepared_world_validity.h"
struct Part {
 std::unordered_map<std::size_t,std::uint32_t> world;
 std::unordered_map<std::uint64_t,std::uint64_t> topology,coast;
 std::vector<int> rivers;
};
struct Result {std::unique_ptr<Part> ground,terrain,objects;};
struct Coast {
 std::uint32_t at(std::size_t)const{return 0;}
 Coast const& world()const{return *this;}
 std::uint64_t node_revision(std::uint64_t)const{return 0;}
};
struct Rivers {using CellProof=std::vector<int>;bool valid(CellProof const&)const{return true;}};
int main(){
 CapturedScene scene;auto f=frame();c3x_renderer_camera_identity_v1 identity{};
 scene.publication_scope(f,identity,1);auto tile=full(2,2);publish(scene,tile);
 auto before=scene.world_snapshot();auto missing=before->key(4,4);assert(!before->current(missing));
 Result result{std::make_unique<Part>(),std::make_unique<Part>(),std::make_unique<Part>()};
 result.ground->topology[missing]=0;result.objects->topology[missing]=0;
 Coast coast;Rivers rivers;assert(prepared_world_valid(result,coast,*before,rivers));
 // A known unseen halo supplies topology, never omitted city/resource art.
 auto halo=full(4,4,6);halo.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 halo.city_id=777;halo.resource_id=888;publish(scene,halo);auto after=scene.world_snapshot();
 auto value=after->current(missing);assert(value && value->relief==6 && value->surface==6);
 assert(value->occurrence.city_id==-1 && value->occurrence.resource_id==-1);
 assert(!(value->occurrence.tile_flags&(C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_PREFETCH)));
 assert(!scene.world_appearance_revision(missing));
 assert(prepared_world_valid(result,coast,*before,rivers)); // Old compiler lease remains coherent.
 assert(!prepared_world_valid(result,coast,*after,rivers)); // Adoption must check newest authority.
 result.ground->topology[missing]=result.objects->topology[missing]=value->semantic;
 assert(prepared_world_valid(result,coast,*after,rivers));
 auto sequence=scene.world_input_sequence();
 halo.tile_flags|=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_EXPLORED;publish(scene,halo);
 assert(scene.world_input_sequence()>sequence);
 auto revealed=scene.world_snapshot();assert(revealed!=after);
 assert(revealed->current(missing)->occurrence.city_id==777 && scene.world_appearance_revision(missing));
 assert(after->current(missing)->occurrence.city_id==-1);
 halo.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 halo.city_id=-1;publish(scene,halo);auto hidden=scene.world_snapshot();
 assert(hidden->current(missing)->occurrence.city_id==777); // Last permitted appearance, not halo omissions.
 auto semantic_sequence=scene.world_input_sequence();bool dirty=false;halo.tile_flags|=C3X_RENDERER_TILE_VISIBLE;
 assert(scene.publish(halo,dirty) && !dirty && scene.world_input_sequence()>semantic_sequence);
 auto appearance=scene.appearance_sequence();halo.road_mask=16;
 assert(scene.publish(halo,dirty) && dirty && scene.appearance_sequence()==appearance);
 auto changed=scene.world_snapshot();
 assert(!prepared_world_valid(result,coast,*changed,rivers));
 // Hidden minimal route fields are omissions, not authoritative removals.
 auto route=full(6,6);route.road_mask=3;route.railroad_mask=4;publish(scene,route);
 auto known=scene.world_snapshot();auto route_id=known->key(6,6);auto route_semantic=known->current(route_id)->semantic;
 auto hidden_route=route;hidden_route.road_mask=hidden_route.railroad_mask=0;
 hidden_route.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 dirty=false;assert(scene.publish(hidden_route,dirty) && !dirty);auto last_known=scene.world_snapshot();
 assert(scene.retained(route_id)->semantic==route_semantic && last_known->current(route_id)->semantic==route_semantic);
 assert(last_known->current(route_id)->occurrence.road_mask==3 && last_known->current(route_id)->occurrence.railroad_mask==4);
 route.road_mask=8;route.railroad_mask=16;dirty=false;assert(scene.publish(route,dirty) && dirty);
 auto full_reveal=scene.world_snapshot();assert(full_reveal->current(route_id)->semantic!=route_semantic);
 assert(full_reveal->current(route_id)->occurrence.road_mask==8 && full_reveal->current(route_id)->occurrence.railroad_mask==16);
 assert(known->current(route_id)->occurrence.road_mask==3);
 hidden_route.tile_flags|=C3X_RENDERER_TILE_VISIBLE;dirty=false;
 assert(scene.publish(hidden_route,dirty) && dirty && scene.world_snapshot()->current(route_id)->occurrence.road_mask==0);
 hidden_route.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;hidden_route.road_mask=32;dirty=false;
 assert(scene.publish(hidden_route,dirty) && dirty && scene.world_snapshot()->current(route_id)->occurrence.road_mask==32);
}
''')

    def test_scope_wrap_reset_and_eviction_are_independent_of_world_lease(self):
        host_cpp(PREFIX + r'''
int main(){
 CapturedScene scene;auto f=frame();c3x_renderer_camera_identity_v1 identity{};
 scene.publication_scope(f,identity,1);auto tile=full(2,2);publish(scene,tile);
 auto first=scene.world_snapshot();auto id=first->key(-62,2);assert(id==scene.key(2,2));
 f.tile_width=64;f.target_width=200;f.target_height=100;
 assert(!scene.publication_scope(f,identity,1) && scene.world_snapshot()==first);
 // Content residency has its own generation proof; CPU inputs cannot revive it.
 int content=1;ResidentContent<int> resident(1);auto handle=resident.bind(content);
 f.tiles=&tile;f.tile_count=1;assert(scene.begin(f));assert(scene.update(tile,2,-1,2,1));scene.finish();
 scene.attach(tile,handle);resident.release(handle);auto replacement=resident.bind(content);
 assert(replacement.slot==handle.slot && replacement.generation!=handle.generation);
 assert(!resident.resolve(scene.retained(id)->compiled) && scene.world_snapshot()==first);
 auto sequence=scene.world_input_sequence();
 ++identity.viewer_epoch;assert(scene.publication_scope(f,identity,1));auto fresh=scene.world_snapshot();
 assert(scene.world_input_sequence()>sequence);
 assert(fresh->scope_sequence()!=first->scope_sequence() && !fresh->current(id));
 assert(first->current(id)->occurrence.city_id==17 && first->key(66,2)==id);
 publish(scene,tile);auto second=scene.world_snapshot();
 ++identity.map_epoch;scene.publication_scope(f,identity,1);assert(!scene.world_snapshot()->current(id));
 assert(second->current(id));scene={};assert(first->current(id) && second->current(id));
 first.reset();second.reset();fresh.reset();
 assert(scene.world_snapshot()->current(scene.key(2,2))==nullptr);
}
''')

    def test_standalone_updates_derive_source_classes_without_native_admission(self):
        host_cpp(PREFIX + r'''
int main(){
 CapturedScene scene;auto f=frame();auto tile=full(2,2,4);f.tiles=&tile;f.tile_count=1;
 auto update=[&](std::uint64_t semantic){assert(scene.begin(f));assert(scene.update(tile,-1,-1,-1,semantic));scene.finish();};
 update(17);auto first=scene.world_snapshot();auto id=first->key(2,2);
 assert(first->current(id)->ground==4 && first->current(id)->relief==-1 && first->current(id)->surface==4);
 assert(first->current(id)->semantic==17 && scene.current(id)->ground==-1);
 tile.anchor_x=700;update(17);assert(scene.world_snapshot()==first);
 tile.real_terrain_type=9;update(18);auto marsh=scene.world_snapshot();
 assert(marsh->current(id)->ground==2 && marsh->current(id)->relief==-1 && marsh->current(id)->surface==9);
 tile.real_terrain_type=12;update(19);auto water=scene.world_snapshot();
 assert(water->current(id)->ground==12 && water->current(id)->relief==-1 && water->current(id)->surface==12);
 auto view=scene.world_view();auto native=scene.observation_view();
 assert(scene.begin(f));assert(!native.current(id));assert(view.current(id)->real==12);
 tile.real_terrain_type=10;assert(scene.update(tile,-1,-1,-1,20));scene.finish();
 auto volcano=scene.world_snapshot();assert(volcano->current(id)->relief==10 && volcano->current(id)->surface==10);
 assert(view.current(id)->real==12 && first->current(id)->real==4);
 auto scope=volcano->scope_sequence();f.world_width_tiles=32;update(20);
 assert(scene.world_snapshot()->scope_sequence()!=scope && volcano->key(66,2)==id);
}
''')


if __name__ == '__main__':
    unittest.main()
