"""CPU object descriptions, terrain dependencies and exact source attributes."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ObjectCompilerTests(unittest.TestCase):
    def test_infrastructure_selection_projection_and_fallback(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
#include <cstring>
#include <cstring>
#include <set>
using namespace c3x_renderer;
int main(){
 std::array<FeatureBundle,objects::family_count> bundles;
 objects::Assets assets{};
 for(unsigned f=0;f<bundles.size();++f){
  assets.bundles[f]=&bundles[f];FeatureAsset asset{};asset.id="fixture:crop:e2";
  asset.vertices={{{0,0,0},{0,0,1},{.25f,.5f}},{{1,0,0},{0,0,1},{.5f,.75f}},{{0,1,1},{0,0,1},{.75f,1}}};
  asset.indices={0,1,2};bundles[f].assets.push_back(asset);
 }
 auto group=[&](objects::Family family,char const* name){FeatureGroup g;g.name=name;FeaturePlacement p{};p.asset_index=0;p.scale=1;g.placements.push_back(p);bundles[family].groups.push_back(g);};
 for(auto n:{"ancient","medieval","industrial","modern"})group(objects::city_family,n);
 for(auto n:{"wall_ancient","wall_medieval","wall_industrial"})group(objects::wall_family,n);
 for(auto n:{"farm_0","farm_1","farm_2"})group(objects::farm_family,n);
 for(auto n:{"mine_0","mine_1","mine_2","mine_3","mine_4","mine_5"})group(objects::mine_family,n);
 for(auto n:{"hut_0","hut_1","hut_2","camp"})group(objects::site_family,n);
 for(auto n:{"bridge_railroad_normal","bridge_modern_normal"})group(objects::bridge_family,n);
 c3x_renderer_tile_v1 tile{};tile.city_id=3;tile.city_flags=C3X_RENDERER_CITY_WALLED|C3X_RENDERER_CITY_CAPITAL;
 tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 tile.tile_x=48;tile.tile_y=32;
 for(int era=0;era<4;++era)for(int size=0;size<3;++size){
  tile.city_era=era;tile.city_size=size;tile.route_style=era;
  objects::Plan legacy,composed;
  assert(objects::select_improvements(tile,assets,2,0,true,true,true,false,legacy));
  assert(objects::select_improvements(tile,assets,2,0,true,true,true,true,composed));
  unsigned city=0,walls=0;for(auto const& i:legacy.instances){city+=i.layer==objects::city_layer;walls+=i.layer==objects::wall_layer;}
  assert(city==unsigned(size==0?4:size==1?7:11) && walls==4);
  assert(legacy.instances.size()==composed.instances.size()+city);
  for(auto const& i:composed.instances)assert(i.layer!=objects::city_layer);
  for(int width:{64,128,160,192}){
   objects::Projection input;input.tile=tile;input.tile_width=width;input.content_view_height=640;
   input.half_w=float(width)/2;input.half_h=float(width)/4;input.relief_projection_scale=float(width)/224*.82f;
   input.feature_projection_scale=float(width)/224;input.pickup_profile=true;input.world_objects=true;
   unsigned reads=0;auto relief=[&](float,float){++reads;return std::array<float,3>{5,0,0};};
   auto height=[](float,float){return 7.5f;};objects::Surfaces out;
   objects::compile(composed,input,assets,relief,height,out);
   assert(reads==composed.instances.size() && out.shadows.empty());
   objects::Surfaces indexed;objects::compile(composed,input,assets,relief,height,indexed,true);
   for(unsigned layer=0;layer<objects::layer_count;++layer){
    assert(indexed.indices[layer].size()==out.layers[layer].size());
    for(unsigned i=0;i<indexed.indices[layer].size();++i)
     assert(std::memcmp(&out.layers[layer][i],&indexed.layers[layer][indexed.indices[layer][i]],sizeof(objects::Vertex))==0);
   }
   assert(out.layers[objects::wall_layer].size()==12);
   auto const& v=out.layers[objects::farm_layer].at(0);
   assert(v.u==.25f && v.v==.5f && v.world_valid==1 && v.world_z==7.5f/112);
   input.pickup_profile=false;input.world_objects=false;input.key_light={.5f,.5f,1};objects::Surfaces compatibility;
   objects::compile(composed,input,assets,relief,height,compatibility);assert(!compatibility.shadows.empty());
  }
 }
 tile.road_mask=tile.railroad_mask=1;tile.river_code=2;tile.route_style=3;
 struct Observation{c3x_renderer_tile_v1 occurrence;} neighbor{tile};
 std::set<std::pair<int,int>> queried;
 auto lookup=[&](int x,int y){queried.emplace(x,y);return &neighbor;};
 objects::Plan routes;objects::select_routes(tile,assets,true,true,lookup,routes);
 assert(routes.routes.size()==4 && routes.instances.size()==1 && queried.size()==4);
 for(auto const&r:routes.routes)assert(r.railroad && r.style==4);
 objects::Plan diagnostic;objects::select_routes(tile,assets,true,false,lookup,diagnostic);
 assert(diagnostic.routes.empty() && diagnostic.instances.size()==1);
 objects::Projection projection;projection.tile=tile;projection.tile_width=128;projection.half_w=64;projection.half_h=32;
 projection.pickup_profile=projection.world_objects=true;projection.relief_projection_scale=128.f/224*.82f;
 auto flat=[](float,float){return std::array<float,3>{0,0,0};};auto sloped=[](float u,float v){return std::array<float,3>{u+v,0,0};};
 objects::Surfaces a,b;objects::compile(routes,projection,assets,flat,[](float,float){return 0.f;},a);
 objects::compile(routes,projection,assets,sloped,[](float,float){return 0.f;},b);
 assert(a.layers[objects::route_layer].size()==384);
 for(unsigned i=0;i<384;++i){auto const& x=a.layers[objects::route_layer][i];auto const& y=b.layers[objects::route_layer][i];assert(x.u==y.u && x.v==y.v && x.x==y.x && x.y!=y.y);}
 objects::Plan sites;assert(objects::select_improvements(tile,assets,2,C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP,false,false,false,false,sites));
 assert(sites.instances.size()==2);bundles[objects::farm_family].groups.clear();objects::Plan failure;
 assert(!objects::select_improvements(tile,assets,2,0,true,true,true,false,failure));
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    def test_prepared_objects_private_queries_owned_capture_and_packed_parity(self):
        run_cpp(r'''
#include "Renderer/native/object_preparation.h"
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer;
void same(render_core::PreparedMesh const& a,render_core::PreparedMesh const& b){
 assert(a.vertices==b.vertices && a.indices==b.indices && a.bounds==b.bounds);
 assert(a.world_low==b.world_low && a.world_high==b.world_high);
 assert(a.vertex_stride==b.vertex_stride && a.index_stride==b.index_stride);
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
 auto observations=scene.observation_view();fidelity::TerrainCompileScratch foreground;
 for(int width:{64,128,160,192})for(int x:{14,46}){
  objects::PreparationInput input;auto& p=input.projection;p.tile=neighbor;p.tile.tile_x=x;p.tile.city_id=1;
  p.tile_width=width;p.half_w=width*.5f;p.half_h=width*.25f;p.content_view_height=480;
  p.relief_projection_scale=width/224.f*.82f;p.feature_projection_scale=width/224.f;p.pickup_profile=p.world_objects=true;
  input.ground=2;input.world_revision=1;input.composition_ready=input.route_ready=true;
  auto expected=objects::prepare(input,assets,library,natural,terrain,coast,observations,foreground,[]{return false;});
  assert(expected && expected->city.size()==2 && expected->topology.size()==4 && !expected->world.empty());
  assert(expected->routes==1 && !expected->layers[objects::route_layer].mesh.empty());
  objects::PreparationLease lease;auto job=input;
  lease.queue.configure({{1,job}},[&](auto const& owned,auto const& stop,unsigned){
   return objects::prepare(owned,assets,library,natural,terrain,coast,observations,lease.scratch,[&]{return stop.load();});
  });
  job.projection.tile.city_id=-1; // queued capture owns its value
  lease.queue.resume();for(int i=0;i<100;++i)scene.attach(neighbor,{});
  auto result=lease.queue.take(1);assert(result && result->composition==0);
  assert(result->topology==expected->topology && result->world==expected->world && result->coast==expected->coast);
  for(unsigned i=0;i<objects::layer_count;++i)same(result->layers[i].mesh,expected->layers[i].mesh);
  for(unsigned i=0;i<result->city.size();++i){same(result->city[i].mesh,expected->city[i].mesh);assert(result->city[i].terrain_conforming==(i==1));}
  assert(result->city[0].lighting==result->city[1].lighting);
  city_fidelity::Surfaces cancelled_city;
  fidelity::GroundProjection projection{0,0,64,32,1,480};
  assert(!city_fidelity::compile(library,library.compositions[0],0,0,[](float,float){return 0.f;},projection,cancelled_city,[]{return true;}));
  assert(cancelled_city.chunks.empty());
  assert(!objects::prepare(input,assets,library,natural,terrain,coast,observations,foreground,[]{return true;}));
 }
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",), timeout=90)

    def test_completed_producers_return_lanes_without_foreground_progress(self):
        run_cpp(r'''
#include "Renderer/native/render_core/content_preparation.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer::render_core;
struct Result {std::size_t bytes()const{return sizeof(*this);}};
using Pool=ContentPreparation<unsigned,unsigned,Result>;
int main(){
 Pool terrain,ground,objects;std::atomic<unsigned> terrain_entered{0},ground_entered{0},object_entered{0};
 std::atomic<bool> finish_ground{false},finish_objects{false},finish_terrain{false};
 auto wait_for=[&](auto predicate){auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
  while(!predicate()){assert(std::chrono::steady_clock::now()<until);std::this_thread::yield();}};
 terrain.configure({{1,1},{2,2},{3,3},{4,4}},[&](auto const&,auto const&,unsigned){
  ++terrain_entered;while(!finish_terrain)std::this_thread::yield();return std::make_unique<Result>();},1);
 ground.configure({{1,1},{2,2}},[&](auto const&,auto const&,unsigned){
  ++ground_entered;while(!finish_ground)std::this_thread::yield();return std::make_unique<Result>();},2);
 objects.configure({{1,1}},[&](auto const&,auto const&,unsigned){
  ++object_entered;while(!finish_objects)std::this_thread::yield();return std::make_unique<Result>();});
 auto return_lanes=[&]{auto g=ground.statistics();auto o=objects.statistics();
  unsigned reserved=(g.pending||g.active?2u:0u)+(o.pending||o.active?1u:0u);
  terrain.expand_workers(4-reserved);};
 ground.set_ready_notification(return_lanes);objects.set_ready_notification(return_lanes);
 terrain.resume();ground.resume();objects.resume();wait_for([&]{return terrain_entered==1 && ground_entered==2 && object_entered==1;});
 // The consumer is blocked in its own work; producer completion must make
 // progress without a take(), a polling call, or another render-loop iteration.
 finish_objects=true;wait_for([&]{return terrain_entered==2;});assert(ground.statistics().active==2);
 finish_ground=true;wait_for([&]{return terrain_entered==4;});
 ground.set_ready_notification({});objects.set_ready_notification({}); // join callback borrowers before destruction
 finish_terrain=true;for(unsigned i=1;i<=4;++i)assert(terrain.take(i));
 ground.clear();objects.clear();terrain.clear();
 assert(terrain.statistics().active_peak<=4 && objects.statistics().active_peak==1 && ground.statistics().active_peak==2);
}
''')

    def test_city_selection_materials_lighting_and_terrain_conformity(self):
        run_cpp(r'''
#include "Renderer/native/city_fidelity/compiler.h"
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
#include <cstring>
#include <set>
using namespace c3x_renderer;
int main(){
 city_fidelity::Library library;library.materials.resize(2);library.materials[0].channels=5;library.materials[1].ground=1;
 city_fidelity::Model model;model.high[2]=1;
 city_fidelity::Part part;part.material=0;city_fidelity::Vertex v{};
 v.position[0]=.1f;v.position[1]=.2f;v.position[2]=.3f;v.uv0[0]=.4f;v.uv0[1]=.5f;
 v.uv1[0]=.6f;v.uv2[0]=.7f;v.normal[2]=1;v.tangent[0]=1;v.bitangent[1]=1;
 part.vertices={v};part.indices={0,0,0};model.parts.push_back(part);part.material=1;model.parts.push_back(part);library.models.push_back(model);
 city_fidelity::Composition composition;composition.clearance[0]=.05f;composition.clearance[1]=2.5f;composition.clearance[3]=5;
 composition.environment=1;city_fidelity::Instance instance;instance.bounds[0]=instance.bounds[1]=-.1f;instance.bounds[2]=instance.bounds[3]=.1f;
 instance.lights.push_back({{.1f,.2f,.3f},1,{1,1,1},1,{0,0,1},0});composition.instances.push_back(instance);
 composition.paving.material=1;composition.paving.period[0]=composition.paving.period[1]=2;
 composition.paving.vertices={{0,0,.5f}};composition.paving.indices={0,0,0};composition.paving.atlas[0]=.8f;
 library.compositions.push_back(composition);c3x_renderer_tile_v1 tile{};tile.city_id=1;tile.city_flags=C3X_RENDERER_CITY_CAPITAL;
 struct Land {int base=2,real=2;};struct Shore {double distance=100;};
 std::set<std::pair<int,int>> queries;Land land;double shore=100,river=100;
 auto world=[&](int x,int y){queries.emplace(x,y);return land;};auto coast=[&](float,float){return Shore{shore};};
 auto water=[&](float,float){return river;};auto height=[](float,float){return 4.f;};
 auto selected=city_fidelity::select(library,tile,10,12,world,coast,water,height);
 assert(selected==&library.compositions[0] && !queries.empty()); // capital uses legal generic fallback
 shore=0;assert(!city_fidelity::select(library,tile,10,12,world,coast,water,height));shore=100;
 river=0;assert(!city_fidelity::select(library,tile,10,12,world,coast,water,height));river=100;
 land.real=6;assert(!city_fidelity::select(library,tile,10,12,world,coast,water,height));land.real=2;
 assert(!city_fidelity::select(library,tile,10,12,world,coast,water,[](float x,float){return x*100;}));
 for(int width:{64,128,160,192}){
  fidelity::GroundProjection project{10,12,float(width)/2,float(width)/4,float(width)/224*.82f,640};
  city_fidelity::Surfaces output;city_fidelity::compile(library,*selected,10,12,height,project,output);
  city_fidelity::Surfaces indexed;city_fidelity::compile(library,*selected,10,12,height,project,indexed,city_fidelity::ContinueCompilation{},true);
  assert(indexed.chunks.size()==output.chunks.size());
  for(unsigned c=0;c<output.chunks.size();++c){
   auto const& a=output.chunks[c];auto const& b=indexed.chunks[c];
   assert(b.indices.size()==a.vertices.size() && b.vertices.size()==1 && b.indices.size()==3);
   assert(a.material==b.material && a.environment==b.environment && a.terrain_conforming==b.terrain_conforming);
   for(unsigned i=0;i<b.indices.size();++i)assert(std::memcmp(&a.vertices[i],&b.vertices[b.indices[i]],sizeof(fidelity::MapVertex))==0);
  }
  assert(output.chunks.size()==3);auto const& paving=output.chunks[0];auto const& rigid=output.chunks[1];auto const& ground=output.chunks[2];
  assert(paving.terrain_conforming && paving.source_model==~0u && paving.atlas[0]==.8f);
  assert(!rigid.terrain_conforming && rigid.source_model==0 && rigid.source_part==0 && rigid.environment);
  assert(ground.terrain_conforming && ground.source_part==1);
  assert(paving.lighting==rigid.lighting && ground.lighting==rigid.lighting && rigid.lighting->lights.size()==1 && rigid.lighting->blockers.size()==1);
  auto const& out=rigid.vertices[0];assert(out.u==.4f && out.v==.5f && out.macro_u==.6f && out.relief_owner_u==.7f);
  assert(out.normal_z==1 && out.material_grass==1 && out.authored_relief_height==1 && out.base_terrain==105);
  assert(ground.vertices[0].base_terrain==60 && paving.vertices[0].base_terrain==62.5f);
  assert(ground.vertices[0].world_z==(4.f+.005f)/112);
 }
}
''')
