"""Exact compiler component identity, proofs, backing and assembly contracts."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp


class WorldComponentTests(unittest.TestCase):
    def test_actual_ground_layer_and_shadow_adoption_preserve_ordinary_caller(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('enum GeometryLayer :')
        enum=source[start:source.index('\n};',start)+3]
        start=source.index('bool ground_geometry_layer(')
        predicate=source[start:source.index('\n}',start)+2]
        start=source.index('                bool world_mesh=prepared_world &&')
        world_mesh=source[start:source.index(';',start)+1]
        start=source.index('                        layer<=geometry_river?&prepared_ground->meshes[layer]:')
        end=source.index(',\n                        world_mesh?',start)
        selected=source[start:end].strip()
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
using namespace c3x_renderer;
'''+enum+predicate+r'''
int main(){
 for(unsigned layer=0;layer<=geometry_river;++layer)assert(ground_geometry_layer(layer));
 for(unsigned layer=geometry_natural_terrain;layer<=geometry_natural_mountain;++layer)assert(ground_geometry_layer(layer));
 for(unsigned layer=geometry_route;layer<geometry_natural_terrain;++layer)assert(!ground_geometry_layer(layer));
 for(unsigned layer=geometry_natural_forest0;layer<geometry_layer_count;++layer)assert(!ground_geometry_layer(layer));
 auto prepared_world=std::make_unique<PreparedWorld>();prepared_world->kind=WorldPreparationKind::ground;
 auto prepared_ground=std::make_unique<fidelity::PreparedGround>();
 auto prepared_terrain=std::make_unique<fidelity::TerrainSurfaces>();
 assert(prepared_ground->meshes[5].empty());
 bool pickup_profile=true;unsigned layer=geometry_shadow;
 for(bool component_preparation:{false,true}){
'''+world_mesh+r'''
 auto prepared='''+selected+r''';
 if(component_preparation){
  assert(!world_mesh && !prepared);
  // Empty prepared terrain-shadow must not replace ordinary caller vertices.
  std::vector<render_core::Vertex> caller(3);caller[1].x=1;caller[2].y=1;
  render_core::PreparedMesh mesh;render_core::MeshFormat format;format.pickup=true;format.projection_kind=2;
  assert(render_core::prepare_mesh(caller,nullptr,format,mesh,[]{return false;}) && !mesh.empty());
 }else assert(world_mesh && prepared==&prepared_ground->meshes[5]);
 }
}
''')

    def test_actual_ordinary_context_renewal_requires_exact_normalized_facts(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('                    auto candidate_context=cached->compile_context;')
        end=source.index('                    if(tile_content_valid(*cached,tile)){',start)
        decision=source[start:end]
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
using c3x_renderer::render_core::CapturedScene;
int main(){
 c3x_renderer_tile_v1 tile{};tile.tile_x=16;tile.tile_y=8;tile.terrain_type=tile.real_terrain_type=2;
 tile.city_id=-1;tile.resource_id=-1;tile.tile_flags=C3X_RENDERER_TILE_RENDER;
 struct Cache {std::array<std::uint64_t,20> compile_context{};c3x_renderer_tile_v1 source_facts{};} value;
 value.source_facts=CapturedScene::content(tile);value.compile_context[17]=7;
 std::array<std::uint64_t,20> expected{};expected[17]=19;
 auto content_tile_for=[](auto tile){return tile;};
 struct Plan {bool context_match=false;} plan;
 auto select=[&]{plan={};for(auto* cached:{&value}){
'''+decision+r'''
 }return plan.context_match;};
 // An authority revision alone may change despite identical normalized facts.
 assert(select());tile.anchor_x=900;tile.anchor_y=700;tile.fog_status=3;
 tile.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_VISIBLE;assert(select());
 tile.resource_id=8;assert(!select());tile.resource_id=-1;assert(select());
 tile.city_id=17;assert(!select());tile.city_id=-1;assert(select());
 tile.has_effect=1;assert(!select());tile.has_effect=0;assert(select());
 ++expected[14];assert(!select()); // Other compiler context never renews.
}
''')

    def test_actual_upfront_planner_dispatches_only_missing_components(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('                    auto add=[&](c3x_renderer::WorldPreparationKey const& key,')
        marker='                    }else add(input.key,c3x_renderer::WorldPreparationKind::combined);'
        end=source.index(marker,start)+len(marker)
        planner=source[start:end]
        start=source.index('    bool ground_content_valid(')
        end=source.index('    bool tile_content_valid(',start)
        ground_valid=source[start:end]
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
#include <map>
#include <set>
using namespace c3x_renderer;
struct Mesh {std::shared_ptr<PreparedWorld> proof;unsigned generation=7;};
struct CachedTileGeometry {
 bool shared_natural=true,ground_component=true;
 GroundRecipeKey ground_recipe;std::shared_ptr<Mesh> mesh;
 std::uint64_t last_used=0;
};
struct State {
 render_core::CapturedScene topology_cache;
 render_core::WorldCoast coast;fidelity::NaturalWorld rivers;
 bool raster_content_valid(PreparedWorld const& proof){return render_core::prepared_world_valid(proof,coast,topology_cache.world_view(),rivers);}
'''+ground_valid+r'''
};
int main(){
 State state;auto& topology_cache=state.topology_cache;
 c3x_renderer_frame_v1 content_source{};content_source.world_width_tiles=content_source.world_height_tiles=32;
 content_source.world_wrap_x=content_source.world_wrap_y=1;
 topology_cache.publication_scope(content_source,{1,1,1,1},1);
 c3x_renderer_tile_v1 tile{};tile.tile_x=16;tile.tile_y=8;tile.terrain_type=tile.real_terrain_type=2;tile.city_id=-1;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 bool changed=false;assert(topology_cache.publish(tile,changed));
 std::vector<std::uint32_t> bits(512,2|(2<<8));state.coast.update({32,32,true,true},bits.data(),bits.size(),1);
 state.rivers.update_rivers(state.coast.world(),1);
 unsigned content_revision=11,device_generation=13,world_ready_reused=0;
 std::uint64_t tile_geometry_epoch=19;
 auto coordinate_key=[&](int x,int y){return topology_cache.key(x,y);};
 std::vector<std::uint64_t> demanded_tiles{coordinate_key(tile.tile_x,tile.tile_y)};
 bool backing_only=false,component_preparation=true,loading_preparation=false,batch_preparing=false;
 WorldPreparationInput input;input.ground.compile.tile=tile;
 input.ground.compile.world_ground=input.ground.compile.pickup_profile=input.ground.compile.fidelity_profile=true;
 input.ground.compile.ground=2;input.ground.compile.flat_grid=input.ground.compile.tile_ground_grid=8;
 input.ground.tile_width=128;input.ground.tile_height=64;input.ground.world_width=input.ground.world_height=32;
 input.ground.wrap_x=input.ground.wrap_y=true;input.terrain.tile_x=16;input.terrain.tile_y=8;input.terrain.real_terrain_type=input.terrain.ground=2;
 input.terrain.tile_width=128;input.terrain.tile_height=64;input.terrain.target_height=128;
 input.terrain.key={16,8,66,128,64,128,11,32,32,3,64,31};
 std::array<std::uint64_t,20> expected{};expected[17]=topology_cache.world_appearance_revision(demanded_tiles[0]);
 std::multimap<std::uint64_t,CachedTileGeometry> tile_geometry_cache;
 auto ground_content_valid=[&](auto const& cached,auto const& recipe){return state.ground_content_valid(cached,recipe);};
 auto world_result_valid=[&](auto const& result){return state.raster_content_valid(result);};
 WorldPreparation world_queue;
 std::deque<WorldPreparation::Job> jobs;
 std::vector<WorldPreparationKey> required,needed,backing_keys;
 std::set<WorldPreparationKey> unique;
 struct Plan {
  CachedTileGeometry* ground=nullptr;std::shared_ptr<Mesh> ground_pin;
  GroundRecipeKey ground_recipe;WorldPreparationKey ground_key,object_key;
  bool ground_missing=true,ground_ready=false,object_ready=false;
 } plan;
 auto plan_missing=[&]{
'''+planner+r'''
 };
 std::atomic<unsigned> ground_calls{0},object_calls{0};
 auto compiler=[&](auto const& job,auto const&,unsigned){
  auto result=std::make_unique<PreparedWorld>();result->kind=job.kind;result->upload_ready=true;
  if(world_preparation_needs_ground(job.kind)){++ground_calls;result->ground=std::make_unique<fidelity::PreparedGround>();result->terrain=std::make_unique<fidelity::TerrainSurfaces>();}
  if(world_preparation_needs_objects(job.kind)){++object_calls;result->objects=std::make_unique<objects::PreparedObjects>();}
  return result;
 };
 auto reset=[&]{world_queue.clear();jobs.clear();required.clear();needed.clear();backing_keys.clear();unique.clear();plan={};ground_calls=object_calls=0;};
 auto dispatch=[&]{world_queue.schedule(std::move(jobs),compiler,1,required,needed,WorldPreparation::byte_limit,true);
  for(auto const& key:required){auto result=world_queue.take(key);assert(result && result->complete() && result->kind==key.kind());}
  world_queue.pause();};
 // Cold selected tile demands both exact pieces in the same bounded queue.
 plan_missing();assert(plan.ground_missing && jobs.size()==2 && required.size()==2);dispatch();
 assert(ground_calls==1 && object_calls==1);
 auto saved_recipe=plan.ground_recipe;
 auto owner=std::make_shared<Mesh>();owner->proof=std::make_shared<PreparedWorld>();owner->proof->kind=WorldPreparationKind::ground;
 owner->proof->ground=std::make_unique<fidelity::PreparedGround>();owner->proof->terrain=std::make_unique<fidelity::TerrainSurfaces>();
 owner->proof->ground->topology.emplace(demanded_tiles[0],render_core::ground_topology_value(topology_cache.world_view().current(demanded_tiles[0])));
 CachedTileGeometry retained;retained.ground_recipe=saved_recipe;retained.mesh=owner;
 tile_geometry_cache.emplace(saved_recipe.lookup_hash(),retained);
 // Resource/city appearance changes the ordinary exact key, not terrain.
 reset();auto old_objects=world_preparation_key(expected,content_source,true,WorldPreparationKind::objects);
 tile.resource_id=9;tile.city_id=17;tile.city_size=2;tile.road_mask=3;
 assert(topology_cache.publish(tile,changed));input.ground.compile.tile=tile;
 expected[17]=topology_cache.world_appearance_revision(demanded_tiles[0]);assert(old_objects!=world_preparation_key(expected,content_source,true,WorldPreparationKind::objects));
 plan_missing();assert(!plan.ground_missing && plan.ground_recipe==saved_recipe && plan.ground_pin==owner);
 assert(plan.ground_pin->generation==7 && jobs.size()==1 && required.size()==1 && required[0].kind()==WorldPreparationKind::objects);
 assert(plan.ground_ready && tile_geometry_cache.begin()->second.last_used==tile_geometry_epoch);
 dispatch();assert(!ground_calls && object_calls==1);
 // A same-hash wrong recipe cannot manufacture a ground hit.
 reset();tile_geometry_cache.begin()->second.ground_recipe.words.back()^=1;
 plan_missing();assert(plan.ground_missing && jobs.size()==2);dispatch();assert(ground_calls==1 && object_calls==1);
 tile_geometry_cache.begin()->second.ground_recipe=saved_recipe;
 // Exact current compiler proof must reject a changed ground family too.
 reset();tile.terrain_type=tile.real_terrain_type=11;assert(topology_cache.publish(tile,changed));
 plan_missing();assert(plan.ground_missing && jobs.size()==2);dispatch();assert(ground_calls==1 && object_calls==1);
 world_queue.clear();
}
''')

    def test_ground_recipe_uses_exact_inputs_and_separates_object_identity(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
#include <map>
using namespace c3x_renderer;
int main(){
 fidelity::GroundPreparationInput g;fidelity::TerrainCompileInput t;
 g.compile.world_ground=g.compile.pickup_profile=g.compile.fidelity_profile=true;
 g.compile.tile.tile_x=16;g.compile.tile.tile_y=8;g.compile.tile.terrain_type=2;g.compile.tile.real_terrain_type=5;
 g.compile.tile.river_code=34;g.compile.ground=2;g.compile.uv_scale=.26f;
 g.compile.flat_grid=8;g.compile.tile_ground_grid=24;g.compile.shadow_grid=16;
 g.tile_width=128;g.tile_height=64;g.world_width=g.world_height=64;g.wrap_x=g.wrap_y=true;
 g.center={.5,.25,.75,.125};g.nodes={{14,12,3,false},{14,13,1,true}};
 t.key={16,8,162,128,64,128,7,64,64,3,64,31};
 t.tile_x=16;t.tile_y=8;t.real_terrain_type=5;t.ground=2;t.tile_width=128;t.tile_height=64;t.target_height=128;
 std::array<std::uint64_t,4> lifetime{9,7,3,11};
 auto first=ground_recipe_key(g,t,lifetime);auto key=ground_world_preparation_key(first);
 auto same=[&]{assert(first==ground_recipe_key(g,t,lifetime));};
 auto different=[&](auto edit){auto changed=g;edit(changed);assert(first!=ground_recipe_key(changed,t,lifetime));};
 // Object arrival, cities, overlays, visibility and occurrences do not own terrain.
 g.compile.tile.resource_id=8;g.compile.tile.resource_class=4;g.compile.tile.city_id=17;g.compile.tile.city_size=2;
 g.compile.tile.variant_seed=71;g.compile.tile.improvement_flags=3;g.compile.tile.feature_flags=31;
 g.compile.tile.road_mask=15;g.compile.tile.railroad_mask=7;g.compile.tile.territory_owner_id=3;
 g.compile.tile.anchor_x=900;g.compile.tile.anchor_y=700;g.compile.tile.tile_flags=~0u;same();
 // Canonical compiler output does not consume these overwritten screen fields.
 g.compile.half_w=112;g.compile.half_h=56;g.compile.left=99;g.compile.top=42;
 g.compile.relief_projection_scale=3;g.compile.key_light[0]=.7f;g.compile.prewarming=true;
 g.topology_revision=999;t.world_revision=999;same();
 different([](auto& x){++x.compile.tile.terrain_type;});
 different([](auto& x){++x.compile.tile.real_terrain_type;});
 different([](auto& x){x.compile.tile.river_code^=8;});
 different([](auto& x){x.compile.tile.has_effect=1;});
 different([](auto& x){++x.compile.ground;});
 different([](auto& x){++x.compile.tile_ground_grid;});
 different([](auto& x){x.compile.uv_scale=.260001f;});
 different([](auto& x){x.skip_flat_shore=false;});
 different([](auto& x){x.center.depth+=.001;});
 different([](auto& x){std::swap(x.nodes[0],x.nodes[1]);});
 different([](auto& x){x.nodes[0].touches_water=true;});
 different([](auto& x){x.nodes.push_back({1,2,1,false});});
 for(unsigned i=0;i<lifetime.size();++i){++lifetime[i];assert(first!=ground_recipe_key(g,t,lifetime));--lifetime[i];}
 for(unsigned i=0;i<t.key.size();++i){++t.key[i];assert(first!=ground_recipe_key(g,t,lifetime));--t.key[i];}
 ++t.detail.mountain;assert(first!=ground_recipe_key(g,t,lifetime));--t.detail.mountain;
 t.retain_height=false;assert(first!=ground_recipe_key(g,t,lifetime));t.retain_height=true;
 // An artificial lookup collision still cannot reuse the wrong exact recipe.
 auto collision=key;collision.recipe.words.back()^=1;assert(collision.identity==key.identity && collision!=key);
 std::map<WorldPreparationKey,int> values;values.emplace(key,1);values.emplace(collision,2);assert(values.size()==2);
 c3x_renderer_frame_v1 frame{};std::array<std::uint64_t,20> context{};
 auto combined=world_preparation_key(context,frame,true),objects=world_preparation_key(context,frame,true,WorldPreparationKind::objects);
 assert(combined!=objects && combined!=key && objects.kind()==WorldPreparationKind::objects);
 g.compile.world_ground=false;auto legacy=ground_recipe_key(g,t,lifetime);++g.compile.left;
 assert(legacy!=ground_recipe_key(g,t,lifetime));
}
''')

    def test_kind_specific_proofs_and_actual_ground_compiler_observation_domain(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
#include <map>
using namespace c3x_renderer;
struct Observations {
 struct Record {int ground=11,relief=-1;std::uint64_t semantic=17;};
 std::map<std::uint64_t,Record> records;
 std::uint64_t key(int x,int y)const{return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);}
 Record const* current(std::uint64_t key)const{auto i=records.find(key);return i==records.end()?nullptr:&i->second;}
};
int main(){
 render_core::WorldCoast coast;std::vector<std::uint32_t> bits(512,11|(11<<8));
 coast.update({32,32,true,true},bits.data(),bits.size(),1);
 fidelity::NaturalData natural;fidelity::SurfaceQueryScratch scratch;
 Observations observations;
 for(int y=0;y<32;++y)for(int x=y&1;x<32;x+=2)observations.records.emplace(observations.key(x,y),Observations::Record{});
 auto ground_view=ground_preparation_observations(observations);
 fidelity::GroundPreparationInput input;input.compile.tile.tile_x=16;input.compile.tile.tile_y=8;
 input.compile.tile.terrain_type=input.compile.tile.real_terrain_type=input.compile.ground=11;
 input.compile.world_ground=input.compile.pickup_profile=input.compile.fidelity_profile=true;
 input.compile.flat_grid=input.compile.tile_ground_grid=input.compile.shadow_grid=8;
 input.compile.uv_scale=.26f;input.tile_width=128;input.tile_height=64;input.world_width=input.world_height=32;
 input.wrap_x=input.wrap_y=true;input.topology_revision=1;
 input.center=coast.sample({12.5,4.5},[](auto,auto){},[](auto,auto){});
 std::array<fidelity::ReliefFields,16> assets;
 PreparedWorld result;result.kind=WorldPreparationKind::ground;
 result.ground=fidelity::compile_selected_ground(input,natural,coast,ground_view,assets,scratch,[]{return false;});
 result.terrain=std::make_unique<fidelity::TerrainSurfaces>();assert(result.ground && !result.ground->topology.empty());
 for(auto const& proof:result.ground->topology)assert(proof.second==12);
 fidelity::NaturalWorld rivers;rivers.update_rivers(coast.world(),1);
 auto valid=[&]{return render_core::prepared_world_valid(result,coast,observations,rivers);};
 assert(result.complete() && valid());
 // City/road semantic edits cannot alter the only topology value read above.
 for(auto& entry:observations.records)entry.second.semantic=999;
 assert(valid());
 auto proof=result.ground->topology.begin();auto id=proof->first;
 observations.records[id].ground=12;assert(!valid());observations.records[id].ground=11;
 auto record=observations.records[id];observations.records.erase(id);assert(!valid());observations.records.emplace(id,record);assert(valid());
 auto world=result.ground->world.begin();assert(world!=result.ground->world.end());
 auto saved=world->second;world->second^=1;assert(!valid());world->second=saved;
 result.terrain->coast.emplace(123,999);assert(!valid());result.terrain->coast.clear();assert(valid());
 result.objects=std::make_unique<objects::PreparedObjects>();result.objects->topology.emplace(id,0);assert(valid());
 result.kind=WorldPreparationKind::objects;assert(!valid());result.objects->topology[id]=999;assert(valid());
 result.objects.reset();assert(!result.complete() && !valid());
 result.kind=WorldPreparationKind::ground;assert(valid());result.terrain.reset();assert(!result.complete() && !valid());
 result.kind=WorldPreparationKind(99);assert(!result.complete() && !valid());
 // Present negative ground and absent are distinct; no hash collision proof.
 Observations::Record missing_family;missing_family.ground=-1;
 assert(render_core::ground_topology_value(&missing_family)==4294967296ull);
 assert(render_core::ground_topology_value<Observations::Record>(nullptr)==0);
}
''')

    def test_component_codec_roundtrip_wrong_kind_old_version_and_truncation(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_backing_codec.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 PreparedWorld ground;ground.kind=WorldPreparationKind::ground;
 assert(WorldBackingCodec::encode(ground).empty());
 ground.ground=std::make_unique<fidelity::PreparedGround>();assert(WorldBackingCodec::encode(ground).empty());
 ground.terrain=std::make_unique<fidelity::TerrainSurfaces>();ground.ground->world={{7,19}};ground.ground->topology={{13,12}};
 ground.terrain->world={{8,21}};ground.terrain->coast={{3,91}};
 ground.ground->meshes[0].vertex_stride=4;ground.ground->meshes[0].index_stride=2;
 ground.ground->meshes[0].vertices={1,2,3,4};
 ground.terrain->meshes[2]=ground.ground->meshes[0];
 auto bytes=WorldBackingCodec::encode(ground);assert(!bytes.empty());
 auto restored=WorldBackingCodec::decode(bytes,WorldPreparationKind::ground);
 assert(restored && restored->complete() && !restored->objects && restored->kind==ground.kind);
 assert(restored->ground->meshes[0].vertices==ground.ground->meshes[0].vertices && restored->terrain->coast==ground.terrain->coast);
 assert(!WorldBackingCodec::decode(bytes) && !WorldBackingCodec::decode(bytes,WorldPreparationKind::objects));
 for(std::size_t size=0;size<bytes.size();++size){auto truncated=bytes;truncated.resize(size);assert(!WorldBackingCodec::decode(truncated,WorldPreparationKind::ground));}
 auto invalid=bytes;std::uint32_t old=3;std::memcpy(invalid.data(),&old,sizeof(old));assert(!WorldBackingCodec::decode(invalid,WorldPreparationKind::ground));
 invalid=bytes;std::uint32_t bad=99;std::memcpy(invalid.data()+4,&bad,sizeof(bad));assert(!WorldBackingCodec::decode(invalid,WorldPreparationKind::ground));
 invalid=bytes;invalid.push_back(0);assert(!WorldBackingCodec::decode(invalid,WorldPreparationKind::ground));
 PreparedWorld objects;objects.kind=WorldPreparationKind::objects;assert(WorldBackingCodec::encode(objects).empty());
 objects.objects=std::make_unique<objects::PreparedObjects>();objects.objects->composition=3;objects.objects->topology={{13,999}};
 objects.objects->layers[1].mesh=ground.ground->meshes[0];
 auto object_bytes=WorldBackingCodec::encode(objects);auto object_copy=WorldBackingCodec::decode(object_bytes,WorldPreparationKind::objects);
 assert(object_copy && object_copy->complete() && !object_copy->ground && !object_copy->terrain && object_copy->objects->composition==3);
 assert(object_copy->objects->layers[1].mesh.vertices==ground.ground->meshes[0].vertices);
 assert(!WorldBackingCodec::decode(object_bytes,WorldPreparationKind::ground));
 for(std::size_t size=0;size<object_bytes.size();size+=7){auto truncated=object_bytes;truncated.resize(size);assert(!WorldBackingCodec::decode(truncated,WorldPreparationKind::objects));}
 ground.kind=WorldPreparationKind::combined;assert(WorldBackingCodec::encode(ground).empty());
 ground.objects=std::move(objects.objects);auto combined=WorldBackingCodec::encode(ground);
 auto copy=WorldBackingCodec::decode(combined);assert(copy && copy->complete() && copy->objects && copy->ground && copy->terrain);
 assert(!WorldBackingCodec::decode(combined,WorldPreparationKind::ground));
 ground.kind=WorldPreparationKind::ground;
 fidelity::NaturalWorld::CellKey key{};ground.ground->rivers.emplace(key,nullptr);
 assert(WorldBackingCodec::encode(ground).empty());
}
''')

    def test_retained_ground_skips_only_authored_terrain_assembly(self):
        source=(ROOT/'Renderer/native/source_fidelity/geometry.h').read_text()
        def statement(start):
            brace=source.index('{',start)
            depth=1
            end=brace+1
            while depth:
                depth+=(source[end]=='{')-(source[end]=='}')
                end+=1
            return source[start:end],brace-start
        emitted,_=statement(source.index('    if(!retained_ground_terrain && !cpu_terrain_enabled)'))
        emitted=emitted.replace('#include "terrain_mesh_body.h"','++terrain_emits;')
        forest,brace=statement(source.index('    if(tile.real_terrain_type==7'))
        forest=forest[:brace+1]+'++forest_builds;} '
        joined,_=statement(source.index('    if(!retained_ground_terrain && cpu_terrain_enabled)'))
        # City and forest assembly must remain directly in fidelity's outer
        # block, outside both retained-ground guards (actual lexical structure).
        city=source.index('#include "../city_fidelity/geometry.h"')
        for token in (city,source.index('    if(tile.real_terrain_type==7')):
            prefix=source[:token]
            self.assertEqual(prefix.count('{')-prefix.count('}'),1)
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 unsigned terrain_emits=0,city_builds=0,forest_builds=0,joins=0,compiles=0,uploads=0;
 for(bool component_preparation:{false,true})for(bool retained_ground_terrain:{false,true})for(bool cpu_terrain_enabled:{false,true}){
  c3x_renderer_tile_v1 tile{};tile.real_terrain_type=7;c3x_renderer_frame_v1 frame{};
  int hill_vegetation=0,raised_vegetation=0,ground=2;
  bool skip_flat_shore=true,separate_natural_relief=true,index_natural_grids=true,retain_height_samples=true,world_objects=true;
  unsigned world_terrain_compiles=0;
  auto terrain_compile_input=[](auto const&,auto const&,auto...){return fidelity::TerrainCompileInput{};};
  struct Queue {unsigned* joins;std::unique_ptr<fidelity::TerrainSurfaces> take(fidelity::TerrainCompileInput::Key const&,bool){++*joins;return {};}} terrain_preparation{&joins};
  std::unique_ptr<PreparedWorld> prepared_world;
  auto terrain_result_valid=[](auto const&){return true;};
  int foreground_terrain_scratch=0;auto cancelled=[]{return false;};
  auto compile_terrain=[&](auto const&,auto&,auto,bool){++compiles;return std::make_unique<fidelity::TerrainSurfaces>();};
  auto attach_terrain_vertex_buffer=[&](auto&){++uploads;return true;};
  std::unordered_map<std::size_t,std::uint32_t> world_dependencies;
  std::unordered_map<std::uint64_t,std::uint64_t> coast_dependencies;
  fidelity::NaturalWorld::CellInputs river_dependencies;
  std::unique_ptr<fidelity::TerrainSurfaces> prepared_terrain;
  auto record_natural_phase=[](unsigned){};
  auto assemble=[&]()->bool{
'''+emitted+r'''
 ++city_builds;
'''+forest+joined+r'''
 return true;};
  auto before_emit=terrain_emits,before_join=joins,before_city=city_builds,before_forest=forest_builds;
  assert(assemble());assert(city_builds==before_city+1 && forest_builds==before_forest+1);
  assert(terrain_emits==before_emit+unsigned(!retained_ground_terrain && !cpu_terrain_enabled));
  assert(joins==before_join+unsigned(!retained_ground_terrain && cpu_terrain_enabled));
  assert(bool(prepared_terrain)==(!retained_ground_terrain && cpu_terrain_enabled));
  assert(world_terrain_compiles==unsigned(component_preparation && !retained_ground_terrain && cpu_terrain_enabled));
 }
 assert(terrain_emits==2 && joins==2 && compiles==2 && uploads==2 && city_builds==8 && forest_builds==8);
}
''')


if __name__=='__main__':
    unittest.main()
