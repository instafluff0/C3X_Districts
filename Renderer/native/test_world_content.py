"""Durable world membership and camera-independent preparation contracts."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=Path(__file__).resolve().parents[2]


class WorldContentTests(unittest.TestCase):
    def test_spatial_membership_survives_views_and_releases_with_owner(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_pass_index.h"
#include <cassert>
#include <limits>
using c3x_renderer::render_core::WorldPassIndex;
int main(){
 WorldPassIndex index;std::vector<WorldPassIndex::Key> out;
 // One object crosses cells and a wrapped occurrence uses another raw anchor.
 assert(index.add(7,100,-.25,-1.25,2.5,.5));
 assert(index.add(8,200,99.75,-1.25,102.5,.5));
 auto bytes=index.bytes();
 for(int zoom:{32,64,128,160,192,224})for(int pan:{-1400,0,3000}){
  double left=pan+zoom*.5,right=pan+zoom;
  assert(index.query((left-pan)/zoom,-1,(right-pan)/zoom,0,out));
  assert(out.size()==1 && out[0]==100);assert(index.bytes()==bytes);
 }
 // Selection is conservative on cell edges, and never misses an intersection.
 for(int i=0;i<200;++i)assert(index.add(10+i,1000+i,i*.375,-i*.125,i*.375+.7,-i*.125+.8));
 for(int i=0;i<200;++i){
  assert(index.query(i*.375+.2,-i*.125+.2,i*.375+.4,-i*.125+.4,out));
  assert(std::find(out.begin(),out.end(),1000+i)!=out.end());
 }
 index.erase(7);assert(!index.contains(100));assert(index.contains(200));
 assert(index.add(300,100,10,20,11,21)); // Address recycled by a new immutable owner.
 assert(index.query(-1,-2,3,1,out));assert(std::find(out.begin(),out.end(),100)==out.end());
 index.erase(7);assert(index.contains(100));index.erase(300);assert(!index.contains(100));
 assert(!index.add(400,300,0,0,1e12,1e12));
 assert(!index.add(400,300,0,0,std::numeric_limits<double>::infinity(),1));
 for(unsigned i=0;i<100000;++i){if(!index.add(10000+i,10000+i,i*3.,0,i*3.+.5,1))break;}
 assert(index.bytes()<=index.budget);
 index.clear();assert(index.bytes()==0);assert(index.query(-100,-100,100,100,out) && out.empty());
}
''')

    def test_prepared_world_projection_identity_preserves_real_detail_and_legacy(self):
        source=(ROOT/'Renderer/native/world_preparation.h').read_text()
        method='inline WorldPreparationKey world_preparation_key('+source.split(
            'inline WorldPreparationKey world_preparation_key(',1)[1].split('\ninline WorldPreparationKey object_world_preparation_key(',1)[0]
        key='struct GroundRecipeKey {'+source.split('struct GroundRecipeKey {',1)[1].split('inline GroundRecipeKey ground_recipe_key',1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <array>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include "Renderer/native/render_core/prepared_world_validity.h"
using c3x_renderer::WorldPreparationKind;
'''+key+method+r'''
int main(){
 std::array<std::uint64_t,20> context{};context[14]=64;context[15]=4;context[17]=91;
 c3x_renderer_frame_v1 frame{};frame.tile_width=128;frame.tile_height=64;frame.target_width=2240;frame.target_height=1260;
 auto canonical=world_preparation_key(context,frame,true),legacy=world_preparation_key(context,frame);
 for(int width:{96,128,160,192,224}){frame.tile_width=width;frame.tile_height=width/2;frame.target_width=1400;
  assert(canonical==world_preparation_key(context,frame,true));assert(legacy!=world_preparation_key(context,frame));}
 // Actual shader/terrain detail and changed authoritative facts remain identity.
 context[14]++;assert(canonical!=world_preparation_key(context,frame,true));context[14]--;
 context[15]++;assert(canonical!=world_preparation_key(context,frame,true));context[15]--;
 context[17]++;assert(canonical!=world_preparation_key(context,frame,true));
}
''')

    def test_preparation_key_preserves_content_and_detail_across_camera_changes(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        method='template<class Lookup> c3x_renderer::fidelity::TerrainCompileInput terrain_compile_input('+source.split(
            'template<class Lookup> c3x_renderer::fidelity::TerrainCompileInput terrain_compile_input(',1)[1].split('\n    bool terrain_result_valid',1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <array>
#include <cstdint>
#include <cassert>
namespace c3x_renderer {namespace fidelity {
struct Detail {unsigned value=5;unsigned identity()const{return value;}};
struct CanopyClearing {std::uint64_t hash()const{return 0x5eed;}};
struct TerrainCompileInput {
 using Key=std::array<std::uint64_t,13>;Key key{};CanopyClearing clearing;
 int tile_x,tile_y,real_terrain_type,ground,tile_width,tile_height,target_height;
 long long world_revision;Detail detail;
 bool river_ready,skip_flat_shore,separate_relief,indexed,retain_height;
};
}}
struct State {
 bool city_profile=true,retained_world=true,share_world_meshes=true,river_assets_ready=true;
 int content_view_height=900;unsigned content_revision=13;
 c3x_renderer::fidelity::Detail patch_detail;
 struct Coast {long long value=0;long long revision()const{return value;}} world_coast;
 template<class Lookup> c3x_renderer::fidelity::CanopyClearing canopy_clearing(c3x_renderer_tile_v1 const&,Lookup)const{return {};}
'''+method+r'''
};
int main(){
 State state;c3x_renderer_tile_v1 tile={};tile.tile_x=2;tile.tile_y=4;tile.real_terrain_type=6;
 c3x_renderer_frame_v1 frame={};frame.tile_width=128;frame.tile_height=64;
 frame.world_width_tiles=frame.world_height_tiles=100;
 auto compile=[&]{return state.terrain_compile_input(tile,frame,2,true,true,true,true,state.share_world_meshes,
  [](int,int){return nullptr;});};
 auto first=compile();assert(first.tile_width==128 && first.target_height==128);
 for(int zoom:{64,128,160,192,224}){
  frame.tile_width=zoom;frame.tile_height=zoom/2;state.content_view_height=1200;
  tile.anchor_x+=37;frame.world_topology_revision++;state.world_coast.value++;
  auto next=compile();assert(first.key==next.key);
  // Local proof validation, not this global revision, decides reuse.
  // The revision follows the viewer-masked world, not the capture (9537f440).
  assert(next.world_revision==state.world_coast.revision());
 }
 state.patch_detail.value++;assert(first.key!=compile().key);state.patch_detail.value--;
 state.content_revision++;assert(first.key!=compile().key);state.content_revision--;
 tile.real_terrain_type=7;assert(first.key!=compile().key);tile.real_terrain_type=6;
 state.share_world_meshes=false;assert(compile().tile_width==224 && first.key!=compile().key);
}
''')

    def test_pass_queries_require_current_occurrence_and_preserve_order(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        method='bool query_region_inputs('+source.split('bool query_region_inputs(',1)[1].split(
            '\n    void prepare_region_contributors',1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/world_pass_index.h"
#include "Renderer/native/render_core/region_contributor_index.h"
#include <cassert>
struct State {
 c3x_renderer::render_core::WorldPassIndex world_pass_index;
 c3x_renderer::render_core::RegionContributorIndex region_contributors;
 std::unordered_map<std::uintptr_t,std::vector<std::pair<unsigned,unsigned>>> world_pass_occurrences;
 bool world_pass_affine=true;int shadow_tile_width=128;
 double world_pass_x=0,world_pass_y=0;float world_pass_reflection=0;
'''+method+r'''
};
int main(){
 State s;using Item=std::pair<unsigned,unsigned>;std::vector<Item> out;
 assert(s.world_pass_index.add(1,100,1,1,2,2));
 assert(s.world_pass_index.add(2,200,1,1,2,2)); // retained but not currently observed
 s.world_pass_occurrences[100]={{5,3},{2,1}};
 s.region_contributors.add(0,{0,0},128,128,256,256);s.region_contributors.ready=true;
 assert(s.query_region_inputs(0,128,128,128,out));
 assert((out==std::vector<Item>{{0,0},{2,1},{5,3}}));
 auto bytes=s.world_pass_index.bytes();
 s.region_contributors.clear();s.region_contributors.ready=true;
 for(int width:{64,128,160,192,224}){
  s.shadow_tile_width=width;s.world_pass_x=300;s.world_pass_y=-100;
  assert(s.query_region_inputs(0,300+width,-100+width,width,out));
  assert((out==std::vector<Item>{{2,1},{5,3}}));assert(s.world_pass_index.bytes()==bytes);
 }
 s.shadow_tile_width=128;s.world_pass_x=s.world_pass_y=0;s.world_pass_reflection=256;
 assert(s.query_region_inputs(1,128,384,128,out));assert(out.size()==2);
 s.world_pass_occurrences.clear();assert(s.query_region_inputs(1,128,384,128,out) && out.empty());
 s.world_pass_affine=false;s.region_contributors.ready=false;assert(!s.query_region_inputs(0,0,0,128,out));
}
''')

class PreparedWorldLifetimeTests(unittest.TestCase):
    def test_current_dependency_validation_and_exact_preparation_key(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        method='bool world_result_valid('+source.split('bool world_result_valid(',1)[1].split('\n    std::unique_ptr<c3x_renderer::fidelity::TerrainSurfaces> compile_terrain',1)[0]
        header=(ROOT/'Renderer/native/world_preparation.h').read_text()
        key='struct GroundRecipeKey {'+header.split('struct GroundRecipeKey {',1)[1].split('inline GroundRecipeKey ground_recipe_key',1)[0]+'inline WorldPreparationKey world_preparation_key('+header.split('inline WorldPreparationKey world_preparation_key(',1)[1].split('\ninline WorldPreparationKey object_world_preparation_key(',1)[0]
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/render_core/prepared_world_validity.h"
#include <array>
#include <algorithm>
#include <memory>
#include <map>
#include <vector>
#include <cassert>
namespace c3x_renderer {namespace fidelity {
struct NaturalWorld {using CellProof=std::vector<std::pair<int,int>>;bool valid(CellProof const& p){for(auto x:p)if(x.second!=7)return false;return true;}};
}
struct Part {
 std::map<int,int> world,coast,topology;
 std::vector<std::pair<int,int>> rivers;
};
struct PreparedWorld {std::unique_ptr<Part> ground,terrain,objects;WorldPreparationKind kind=WorldPreparationKind::combined;};
'''+key+r'''
}
struct State {
 struct World {int at(int)const{return 2;}};
 struct Coast {World world()const{return {};}int node_revision(int)const{return 3;}} world_coast;
 struct Record {int semantic=4,ground=3;} record;
 struct Topology {Record value;Record const* current(int key)const{return key?&value:nullptr;}Topology const& world_view()const{return *this;}} topology_cache;
 c3x_renderer::fidelity::NaturalWorld natural;
 bool terrain_result_valid(c3x_renderer::Part const& p){
  for(auto x:p.world)if(x.second!=2)return false;
  for(auto x:p.coast)if(x.second!=3)return false;
  return natural.valid(p.rivers);
 }
'''+method+r'''
};
int main(){
 State state;c3x_renderer::PreparedWorld result;
 assert(!state.world_result_valid(result));
 result.ground=std::make_unique<c3x_renderer::Part>();result.terrain=std::make_unique<c3x_renderer::Part>();result.objects=std::make_unique<c3x_renderer::Part>();
 for(auto* part:{result.ground.get(),result.terrain.get(),result.objects.get()}){
  part->world[1]=2;part->coast[1]=3;part->rivers={{1,7}};
 }
 result.ground->topology[1]=result.objects->topology[1]=4;
 assert(state.world_result_valid(result));
 for(auto* part:{result.ground.get(),result.terrain.get(),result.objects.get()}){
  part->world[1]=9;assert(!state.world_result_valid(result));part->world[1]=2;
  part->coast[1]=9;assert(!state.world_result_valid(result));part->coast[1]=3;
  part->rivers[0].second=9;assert(!state.world_result_valid(result));part->rivers[0].second=7;
 }
 result.objects->topology[1]=8;assert(!state.world_result_valid(result));result.objects->topology[1]=4;
 result.ground->topology[0]=4;assert(!state.world_result_valid(result));result.ground->topology[0]=0;
 assert(state.world_result_valid(result));
 std::array<std::uint64_t,20> context{};c3x_renderer_frame_v1 frame{};frame.tile_width=128;frame.tile_height=64;frame.target_width=1120;frame.target_height=1192;
 auto first=c3x_renderer::world_preparation_key(context,frame);
 frame.presentation_time_ticks=999;assert(first==c3x_renderer::world_preparation_key(context,frame));
 for(unsigned i=0;i<context.size();++i){++context[i];assert(first!=c3x_renderer::world_preparation_key(context,frame));--context[i];}
 ++frame.tile_width;assert(first!=c3x_renderer::world_preparation_key(context,frame));--frame.tile_width;
 ++frame.target_width;assert(first!=c3x_renderer::world_preparation_key(context,frame));
}
''')


class PersistentTileProofTests(unittest.TestCase):
    def test_production_proof_separates_world_dependencies_native_anchors_and_gpu_residency(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        method='bool tile_content_valid('+source.split('bool tile_content_valid(',1)[1].split('\n    bool restore_viewport_geometry',1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/scene_publication.h"
#include <cassert>
#include <map>
#include <vector>
using namespace c3x_renderer::render_core;
struct CachedTileGeometry {
 bool shared_natural=false,world_ground=true,world_objects=true,validity=false;
 ContentHandle natural_content{};
 std::uint64_t validity_epoch=0,validity_world_sequence=0;
 RasterDependencyRevisions::Checkpoint validity_revision{};
 int validity_anchor_x=0,validity_anchor_y=0,source_tile_width=128;
 std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies;
 std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;
 std::vector<std::pair<std::uint64_t,std::array<int,2>>> anchor_dependencies;
 std::vector<int> river_dependencies;
};
struct State {
 RasterDependencyRevisions raster_dependency_revisions;
 CapturedScene topology_cache;
 struct Resident {bool available=true;struct Mesh{CachedTileGeometry proof_value;CachedTileGeometry const* proof=&proof_value;};
  struct Value{bool shared_natural=true,ground_component=false;std::shared_ptr<Mesh> mesh=std::make_shared<Mesh>();} value;
  Value const* resolve(ContentHandle)const{return available?&value:nullptr;}} resident_content;
 struct World{unsigned at(std::size_t)const{return 7;}};
 struct Coast{World world()const{return {};}std::uint64_t node_revision(std::uint64_t)const{return 8;}} world_coast;
 struct Rivers{bool valid(std::vector<int>const&)const{return true;}} natural;
 unsigned frame_tile_invalid_shared=0,frame_tile_invalid_appearance=0,frame_tile_invalid_semantic=0,
  frame_tile_invalid_coast=0,frame_tile_invalid_world=0,frame_tile_invalid_anchor=0,frame_tile_invalid_river=0;
 int shadow_tile_width=128;
 bool ground_proof_current=true;
 bool raster_content_valid(CachedTileGeometry const&){return ground_proof_current;}
'''+method+r'''
};
int main(){
 State state;ScenePublication journal;c3x_renderer_camera_identity_v1 identity{};identity.map_epoch=identity.viewer_epoch=1;
 // Production binds the scene's producer edits to the raster revision stream.
 state.topology_cache.bind_raster_dependencies(&state.raster_dependency_revisions);
 c3x_renderer_frame_v1 f{};f.world_width_tiles=f.world_height_tiles=16;f.world_wrap_x=1;
 c3x_renderer_tile_v1 tiles[3]{};
 for(int i=0;i<3;++i){tiles[i].tile_x=2+2*i;tiles[i].tile_y=2;tiles[i].terrain_type=tiles[i].real_terrain_type=2;
  tiles[i].city_id=-1;tiles[i].tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_TOPOLOGY_HALO|
   C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
  tiles[i].anchor_x=100+128*i;tiles[i].anchor_y=200;}
 f.tiles=tiles;f.tile_count=3;bool changed=false;assert(journal.capture(f,identity)&&journal.apply(state.topology_cache,changed));
 auto observe=[&]{assert(state.topology_cache.begin(f));for(unsigned i=0;i<f.tile_count;++i)
  assert(state.topology_cache.update(f.tiles[i],2,-1,2,CapturedScene::topology(f.tiles[i])));state.topology_cache.finish();};observe();
 auto proof=[&](int input){CachedTileGeometry value;value.natural_content={1,1};
  auto key=state.topology_cache.key(tiles[input].tile_x,2);value.dependencies={{key,CapturedScene::topology(tiles[input])}};
  value.appearance_dependencies={{key,state.topology_cache.world_appearance_revision(key)}};
  value.coast_dependencies={{1,8}};value.world_dependencies={{1,7}};return value;};
 auto related=proof(1),unrelated=proof(2);auto world=state.topology_cache.world_snapshot();
 assert(state.tile_content_valid(related,tiles[0])&&state.tile_content_valid(unrelated,tiles[0]));
 state.resident_content.value.ground_component=true;state.ground_proof_current=false;
 assert(!state.tile_content_valid(related,tiles[0]));state.ground_proof_current=true;
 assert(state.tile_content_valid(related,tiles[0]));state.frame_tile_invalid_shared=0;
 // World dependencies retain authority when a camera no longer observes them.
 f.tile_count=1;tiles[0].anchor_x+=500;observe();assert(world==state.topology_cache.world_snapshot());
 assert(!state.topology_cache.current(state.topology_cache.key(4,2)));
 assert(state.tile_content_valid(related,tiles[0])&&state.tile_content_valid(unrelated,tiles[0]));
 // Native anchor proofs still fail when their actual occurrence is absent.
 auto native=related;native.anchor_dependencies={{state.topology_cache.key(4,2),{128,0}}};native.validity_epoch=0;
 assert(!state.tile_content_valid(native,tiles[0])&&state.frame_tile_invalid_anchor==1);
 // A copied local mutation invalidates precisely its dependency closure.
 auto edit=tiles[1];edit.road_mask=3;f.tiles=&edit;assert(journal.capture(f,identity)&&journal.apply(state.topology_cache,changed));
 f.tiles=tiles;observe();assert(!state.tile_content_valid(related,tiles[0]));assert(state.tile_content_valid(unrelated,tiles[0]));
 assert(world->current(world->key(4,2))->semantic==CapturedScene::topology(tiles[1]));
 // An explicitly absent input is a dependency: later admission cannot reuse it.
 auto absent=unrelated;auto key=state.topology_cache.key(8,2);absent.dependencies={{key,0}};absent.validity_epoch=0;
 assert(state.tile_content_valid(absent,tiles[0]));edit.tile_x=8;f.tiles=&edit;
 assert(journal.capture(f,identity)&&journal.apply(state.topology_cache,changed));f.tiles=tiles;observe();
 assert(!state.tile_content_valid(absent,tiles[0]));
 // GPU residency is independent from unchanged world input authority.
 state.resident_content.available=false;assert(!state.tile_content_valid(unrelated,tiles[0]));
 assert(state.frame_tile_invalid_shared==1);
}
''')
