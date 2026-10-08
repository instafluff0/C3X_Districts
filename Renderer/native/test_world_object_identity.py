"""Exact CPU object recipes preserve body authority and separate presentation facts."""
import unittest
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp


class WorldObjectIdentityTests(unittest.TestCase):
    def test_palette_and_edges_preserve_object_recipe_but_invalidate_full_raster(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
#include <unordered_set>
using namespace c3x_renderer;
using namespace c3x_renderer::render_core;
int main(){
 CapturedScene scene;RasterDependencyRevisions raster;scene.bind_raster_dependencies(&raster);
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=130;
 c3x_renderer_camera_identity_v1 identity{};identity.map_epoch=1;identity.viewer_epoch=1;
 scene.publication_scope(frame,identity,7);
 c3x_renderer_tile_v1 tile{};tile.tile_x=35;tile.tile_y=19;tile.city_id=-1;tile.resource_id=-1;
 tile.terrain_type=tile.real_terrain_type=2;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
 bool changed=false;assert(scene.publish(tile,changed));auto id=scene.key(35,19);
 std::array<std::uint64_t,20> context{};context[0]=35;context[1]=19;context[17]=scene.retained(id)->revision;
 auto key=object_world_preparation_key(context,frame,true,tile);
 auto appearance=scene.appearance_sequence();auto checkpoint=raster.checkpoint();
 std::unordered_set<RasterDependencyRevisions::Key,RasterDependencyRevisions::Hash> dependencies;
 dependencies.insert({RasterDependencyRevisions::Domain::appearance,id});
 auto before=CapturedScene::content(tile);tile.territory_color_rgb=0x78dd00;
 changed=false;assert(scene.publish(tile,changed)&&changed);context[17]=scene.retained(id)->revision;
 assert(scene.appearance_sequence()>appearance&&object_world_preparation_key(context,frame,true,tile)==key);
 auto after=CapturedScene::content(tile);assert(std::memcmp(&before,&after,sizeof(before)));
 std::uint64_t visits=0;assert(!raster.unchanged(checkpoint,dependencies,visits));
 appearance=scene.appearance_sequence();tile.territory_edge_mask=13;
 assert(scene.publish(tile,changed));context[17]=scene.retained(id)->revision;
 assert(scene.appearance_sequence()>appearance&&object_world_preparation_key(context,frame,true,tile)==key);
 // A deliberately colliding bucket identity still compares the exact recipe.
 auto collision=key;collision.recipe.words[0]^=1;
 assert(collision.identity==key.identity&&collision!=key);
}
''')

    def test_resources_reuse_cpu_recipe_but_change_full_scene_and_raster_identity(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
#include <unordered_set>
using namespace c3x_renderer;using namespace c3x_renderer::render_core;
int main(){
 CapturedScene scene;RasterDependencyRevisions raster;scene.bind_raster_dependencies(&raster);
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=130;
 c3x_renderer_camera_identity_v1 identity{};identity.map_epoch=1;identity.viewer_epoch=1;
 scene.publication_scope(frame,identity,7);
 c3x_renderer_tile_v1 tile{};tile.tile_x=41;tile.tile_y=35;tile.city_id=-1;
 tile.resource_id=tile.resource_class=-1;tile.terrain_type=tile.real_terrain_type=2;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 bool changed=false;assert(scene.publish(tile,changed));auto id=scene.key(tile.tile_x,tile.tile_y);
 std::array<std::uint64_t,20> context{};context[0]=41;context[1]=35;context[17]=scene.retained(id)->revision;
 auto key=object_world_preparation_key(context,frame,true,tile);
 auto before=CapturedScene::content(tile);auto appearance=scene.appearance_sequence();auto checkpoint=raster.checkpoint();
 std::unordered_set<RasterDependencyRevisions::Key,RasterDependencyRevisions::Hash> dependencies;
 dependencies.insert({RasterDependencyRevisions::Domain::appearance,id});
 tile.resource_id=21;tile.resource_class=0;std::strcpy(tile.resource_name,"wheat");
 changed=false;assert(scene.publish(tile,changed)&&changed);context[17]=scene.retained(id)->revision;
 assert(object_world_preparation_key(context,frame,true,tile)==key);
 assert(scene.appearance_sequence()>appearance);std::uint64_t visits=0;
 assert(!raster.unchanged(checkpoint,dependencies,visits));
 auto after=CapturedScene::content(tile);assert(std::memcmp(&before,&after,sizeof(before)));
 assert(after.resource_id==21&&after.resource_class==0&&!std::strcmp(after.resource_name,"wheat"));
 auto const& retained=scene.retained(id)->appearance;
 assert(retained.resource_id==21&&retained.resource_class==0&&!std::strcmp(retained.resource_name,"wheat"));
 auto cpu=CapturedScene::object_inputs(tile);assert(cpu.resource_id==0&&cpu.resource_class==0&&cpu.resource_name[0]==0);
 auto variant=tile;variant.resource_id=5;assert(object_world_preparation_key(context,frame,true,variant)==key);
 variant=tile;variant.resource_class=2;assert(object_world_preparation_key(context,frame,true,variant)==key);
 variant=tile;std::strcpy(variant.resource_name,"iron");assert(object_world_preparation_key(context,frame,true,variant)==key);
 // CPU reuse never masks a changed actual infrastructure/city body.
 variant=tile;variant.road_mask=1;assert(object_world_preparation_key(context,frame,true,variant)!=key);
 variant=tile;variant.city_id=9;assert(object_world_preparation_key(context,frame,true,variant)!=key);
}
''')

    def test_actual_worker_routes_and_city_meshes_do_not_read_resource_facts(self):
        # The runtime translation unit also contains Windows loaders. Extract
        # only these actual portable leaves so this CPU contract stays on host.
        source = Path(__file__).with_name("terrain_scene_runtime.cpp").read_text()
        signatures = (
            "std::uint32_t feature_hash(std::uint32_t value)",
            "FeatureGroup const * find_feature_group(FeatureBundle const & bundle, char const * name)",
            "float stable_random(std::uint32_t value)",
            "std::uint32_t stable_hash(std::uint32_t value)",
        )
        definitions = []
        for signature in signatures:
            start = source.index(signature)
            end = source.index("{", start) + 1
            depth = 1
            while depth:
                depth += (source[end] == "{") - (source[end] == "}")
                end += 1
            definitions.append(source[start:end])
        leaves = "namespace c3x_renderer { namespace {\n" + definitions[0] + "\n}\n"
        leaves += "\n".join(definitions[1:]) + "\n}\n"
        program = r'''
#define NOMINMAX
#include "Renderer/native/object_preparation.h"
#include "Renderer/native/render_core/captured_scene.h"
#include "Renderer/lab/shared/natural/patterns.h"
#include <cassert>
#include <cstring>
// RUNTIME_LEAF_FUNCTIONS
using namespace c3x_renderer;
void same(render_core::PreparedMesh const& a,render_core::PreparedMesh const& b){
 assert(a.vertices==b.vertices&&a.indices==b.indices&&a.bounds==b.bounds);
 assert(a.world_low==b.world_low&&a.world_high==b.world_high);
 assert(a.vertex_stride==b.vertex_stride&&a.index_stride==b.index_stride);
}
int main(){
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets;
 for(unsigned i=0;i<bundles.size();++i)assets.bundles[i]=&bundles[i];
 city_fidelity::Library library;library.materials.resize(2);library.materials[1].ground=1;
 city_fidelity::Model model;city_fidelity::Part part;city_fidelity::Vertex vertex{};
 vertex.position[0]=.1f;vertex.position[2]=.3f;vertex.normal[2]=1;vertex.tangent[0]=1;vertex.bitangent[1]=1;
 part.vertices={vertex};part.indices={0,0,0};model.parts.push_back(part);part.material=1;model.parts.push_back(part);
 library.models.push_back(model);city_fidelity::Composition composition;composition.clearance[1]=100;
 composition.instances.push_back({});library.compositions.push_back(composition);
 fidelity::NaturalData natural;natural.fields.resize(1);natural.fields[0].width=natural.fields[0].height=2;
 natural.fields[0].pixels={0,64,128,255};std::array<fidelity::ReliefFields,14> terrain;
 render_core::WorldCoast coast;std::vector<std::uint32_t> bits(512,2|(2<<8));
 coast.update({32,32,true,true},bits.data(),bits.size(),1);
 render_core::CapturedScene scene;c3x_renderer_frame_v1 frame{};
 frame.world_width_tiles=frame.world_height_tiles=32;frame.world_wrap_x=frame.world_wrap_y=1;
 assert(scene.begin(frame));c3x_renderer_tile_v1 neighbor{};neighbor.tile_x=16;neighbor.tile_y=8;neighbor.road_mask=1;
 neighbor.tile_flags=C3X_RENDERER_TILE_RENDER;assert(scene.update(neighbor,2,-1,2,11));scene.finish();
 auto observations=scene.observation_view();
 for(int width:{64,128,192})for(int x:{14,46}){
  objects::PreparationInput input;auto& p=input.projection;p.tile=neighbor;p.tile.tile_x=x;p.tile.city_id=1;
  p.tile.resource_id=p.tile.resource_class=-1;
  p.tile_width=width;p.half_w=width*.5f;p.half_h=width*.25f;p.content_view_height=480;
  p.relief_projection_scale=width/224.f*.82f;p.feature_projection_scale=width/224.f;p.pickup_profile=p.world_objects=true;
  input.ground=2;input.world_revision=1;input.composition_ready=input.route_ready=true;
  fidelity::TerrainCompileScratch before_scratch,after_scratch;
  auto before=objects::prepare(input,assets,library,natural,terrain,coast,observations,before_scratch,[]{return false;});
  assert(before&&before->city.size()==2&&before->routes==1&&!before->layers[objects::route_layer].mesh.empty());
  input.projection.tile.resource_id=21;input.projection.tile.resource_class=0;
  std::strcpy(input.projection.tile.resource_name,"wheat");
  auto after=objects::prepare(input,assets,library,natural,terrain,coast,observations,after_scratch,[]{return false;});
  assert(after&&after->city.size()==before->city.size()&&after->routes==before->routes&&after->instances==before->instances);
  assert(after->topology==before->topology&&after->world==before->world&&after->coast==before->coast);
  assert(after->composition==before->composition&&after->rivers==before->rivers);
  for(unsigned i=0;i<objects::layer_count;++i)same(after->layers[i].mesh,before->layers[i].mesh);
  for(unsigned i=0;i<before->city.size();++i){same(after->city[i].mesh,before->city[i].mesh);
   assert(after->city[i].material==before->city[i].material&&after->city[i].terrain_conforming==before->city[i].terrain_conforming);}
 }
}
'''
        run_cpp(program.replace("// RUNTIME_LEAF_FUNCTIONS", leaves))

    def test_active_volcano_carries_a_plume_that_observes_its_activity(self):
        # Same portable leaves as the worker test above.
        source = Path(__file__).with_name("terrain_scene_runtime.cpp").read_text()
        definitions = []
        for signature in ("std::uint32_t feature_hash(std::uint32_t value)",
                          "FeatureGroup const * find_feature_group(FeatureBundle const & bundle, char const * name)",
                          "float stable_random(std::uint32_t value)", "std::uint32_t stable_hash(std::uint32_t value)"):
            start = source.index(signature)
            end = source.index("{", start) + 1
            depth = 1
            while depth:
                depth += (source[end] == "{") - (source[end] == "}")
                end += 1
            definitions.append(source[start:end])
        leaves = "namespace c3x_renderer { namespace {\n" + definitions[0] + "\n}\n" + "\n".join(definitions[1:]) + "\n}\n"
        program = r'''
#define NOMINMAX
#include "Renderer/native/object_preparation.h"
#include "Renderer/native/render_core/captured_scene.h"
#include "Renderer/lab/shared/natural/patterns.h"
#include <cassert>
// RUNTIME_LEAF_FUNCTIONS
using namespace c3x_renderer;
int main(){
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets;
 for(unsigned i=0;i<bundles.size();++i)assets.bundles[i]=&bundles[i];
 city_fidelity::Library library;library.materials.resize(2);library.materials[1].ground=1;library.effect_material=1;
 fidelity::NaturalData natural;natural.fields.resize(1);natural.fields[0].width=natural.fields[0].height=2;
 natural.fields[0].pixels={0,64,128,255};std::array<fidelity::ReliefFields,14> terrain;
 render_core::CapturedScene scene;c3x_renderer_frame_v1 frame{};
 frame.world_width_tiles=frame.world_height_tiles=32;frame.world_wrap_x=frame.world_wrap_y=1;
 assert(scene.begin(frame));scene.finish();auto observations=scene.observation_view();
 unsigned const at=(8*32+16)/2; // raw tile (16,8)
 // Dormant, smoldering (bit 24) and erupting (bits 24 and 27) volcanoes.
 for(unsigned state:{0u,1u<<24,(1u<<24)|(1u<<27)}){
  render_core::WorldCoast coast;std::vector<std::uint32_t> bits(512,2|(2<<8));bits[at]=10|(10<<8)|state;
  coast.update({32,32,true,true},bits.data(),bits.size(),1);
  objects::PreparationInput input;auto& p=input.projection;p.tile.tile_x=16;p.tile.tile_y=8;p.tile.city_id=-1;
  p.tile.real_terrain_type=10;p.tile.resource_id=p.tile.resource_class=-1;p.tile.tile_flags=C3X_RENDERER_TILE_RENDER;
  p.tile_width=128;p.half_w=64;p.half_h=32;p.content_view_height=480;
  p.relief_projection_scale=128/224.f*.82f;p.feature_projection_scale=128/224.f;p.pickup_profile=p.world_objects=true;
  input.ground=2;input.world_revision=1;input.city_ready=true;
  fidelity::TerrainCompileScratch scratch;
  auto result=objects::prepare(input,assets,library,natural,terrain,coast,observations,scratch,[]{return false;});
  assert(result);
  // The tile's own activity is a recorded dependency: a change re-prepares it.
  assert(result->world.count(at) && result->world.at(at)==bits[at]);
  if(!state){assert(result->city.empty());continue;}
  assert(result->city.size()==1 && result->city[0].effect && result->city[0].material==1 &&
         result->city[0].terrain_conforming && result->city[0].lighting && !result->city[0].mesh.empty());
 }
}
'''
        run_cpp(program.replace("// RUNTIME_LEAF_FUNCTIONS", leaves))

    def test_compiler_body_and_lifetime_inputs_are_conservative_exact_words(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 c3x_renderer_frame_v1 frame{};frame.tile_width=128;frame.tile_height=64;frame.target_width=2240;frame.target_height=1260;
 c3x_renderer_tile_v1 tile{};tile.city_id=7;tile.city_owner_id=3;tile.city_size=1;tile.variant_seed=123;
 std::strcpy(tile.resource_name,"abc");std::array<std::uint64_t,20> context{};context[17]=9;
 auto key=object_world_preparation_key(context,frame,true,tile);
 assert(key.recipe.words.size()==sizeof(tile)/8);
 assert(key.recipe.words[offsetof(c3x_renderer_tile_v1,variant_seed)/8]==123);
 assert(key.recipe.words[offsetof(c3x_renderer_tile_v1,resource_name)/8]==0);
 auto differs=[&](auto member){auto changed=tile;++(changed.*member);assert(object_world_preparation_key(context,frame,true,changed)!=key);};
 differs(&c3x_renderer_tile_v1::terrain_type);differs(&c3x_renderer_tile_v1::real_terrain_type);
 differs(&c3x_renderer_tile_v1::variant_seed);
 differs(&c3x_renderer_tile_v1::city_id);differs(&c3x_renderer_tile_v1::city_owner_id);differs(&c3x_renderer_tile_v1::city_size);
 differs(&c3x_renderer_tile_v1::city_culture_group);differs(&c3x_renderer_tile_v1::city_era);differs(&c3x_renderer_tile_v1::city_flags);
 differs(&c3x_renderer_tile_v1::river_code);differs(&c3x_renderer_tile_v1::road_mask);differs(&c3x_renderer_tile_v1::railroad_mask);
 differs(&c3x_renderer_tile_v1::route_style);differs(&c3x_renderer_tile_v1::feature_flags);differs(&c3x_renderer_tile_v1::improvement_flags);
 differs(&c3x_renderer_tile_v1::irrigation_mask);differs(&c3x_renderer_tile_v1::has_effect);differs(&c3x_renderer_tile_v1::barbarian_tribe_id);
 auto changed=tile;changed.resource_name[0]='z';assert(object_world_preparation_key(context,frame,true,changed)==key);
 for(unsigned n=0;n<context.size();++n){auto next=context;++next[n];
  assert((object_world_preparation_key(next,frame,true,tile)==key)==(n==17));}
 auto new_view=frame;new_view.target_height=8;new_view.target_width=8;new_view.tile_width=96;new_view.tile_height=48;
 assert(object_world_preparation_key(context,new_view,true,tile)==key);
 assert(object_world_preparation_key(context,new_view,false,tile)!=object_world_preparation_key(context,frame,false,tile));
}
''')

    def test_recipe_reuse_never_promotes_unknown_or_permitted_body_authority(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include <cassert>
using namespace c3x_renderer;using namespace c3x_renderer::render_core;
int main(){
 CapturedScene scene;c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
 c3x_renderer_camera_identity_v1 identity{};scene.publication_scope(frame,identity,1);
 c3x_renderer_tile_v1 hidden{};hidden.tile_x=3;hidden.tile_y=5;hidden.city_id=-1;hidden.resource_id=-1;
 hidden.terrain_type=hidden.real_terrain_type=2;
 hidden.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 bool changed=false;assert(scene.publish(hidden,changed));auto id=scene.key(3,5);
 std::array<std::uint64_t,20> context{};auto empty=object_world_preparation_key(context,frame,true,hidden);
 scene.attach(hidden,{3,4},true);assert(!scene.retained(id)->authoritative&&!scene.current(id));
 auto permitted=hidden;permitted.tile_flags|=C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_CITY_BODY_KNOWN;
 permitted.city_id=17;permitted.city_owner_id=4;permitted.city_size=1;
 assert(object_world_preparation_key(context,frame,true,permitted)!=empty);
 assert(scene.publish(permitted,changed));assert(!scene.retained(id)->authoritative&&!scene.current(id));
 auto complete=permitted;complete.tile_flags|=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBLE;
 auto recipe=object_world_preparation_key(context,frame,true,permitted);
 assert(object_world_preparation_key(context,frame,true,complete)==recipe);
 // Only the actual full native publication grants authority, never key reuse.
 assert(!scene.retained(id)->authoritative);assert(scene.publish(complete,changed));assert(scene.retained(id)->authoritative);
 auto edited=complete;edited.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 assert(object_world_preparation_key(context,frame,true,edited)!=recipe);
}
''')

    def test_compact_store_charges_exact_object_recipe_storage_and_pressure(self):
        run_cpp(r'''
using UINT=unsigned;
#include "Renderer/native/world_preparation.h"
#include "Renderer/native/render_core/compressed_world_store.h"
#include <cassert>
using namespace c3x_renderer;using namespace c3x_renderer::render_core;
struct Codec {
 bool encode(std::vector<unsigned char> const& raw,WorldStoreBuffer& out){out.resize(raw.size());out.bytes=raw;return true;}
 bool decode(std::vector<unsigned char> const& packed,std::vector<unsigned char>& raw,WorldStoreMemory&){raw=packed;return true;}
};
int main(){
 c3x_renderer_tile_v1 tile{};c3x_renderer_frame_v1 frame{};std::array<std::uint64_t,20> context{};
 auto key=object_world_preparation_key(context,frame,true,tile);auto bare=key;bare.recipe.words.clear();bare.recipe.words.shrink_to_fit();
 using Store=CompressedWorldStore<WorldPreparationKey,Codec>;Store full,small;
 auto owned=[](auto const& value){return value.recipe.words.capacity()*sizeof(std::uint64_t);};
 assert(full.configure(1024*1024,owned)&&small.configure(1024*1024,owned));
 std::vector<unsigned char> bytes(128,17);assert(full.put(key,bytes)&&small.put(bare,bytes));
 auto delta=full.statistics().resident_bytes-small.statistics().resident_bytes;
 assert(delta==owned(key)&&full.contains(key)&&!full.contains(bare));
 Store pressure;auto baseline=pressure.statistics().resident_bytes;
 assert(pressure.configure(small.statistics().resident_bytes,owned));
 assert(!pressure.put(key,bytes)&&!pressure.contains(key));
 assert(pressure.statistics().capacity_refusals>0&&pressure.statistics().records==0);
 assert(pressure.statistics().resident_bytes<=pressure.statistics().limit);
 pressure.clear();assert(pressure.statistics().resident_bytes==baseline);
}
''')


if __name__ == '__main__':
    unittest.main()
