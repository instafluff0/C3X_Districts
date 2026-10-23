"""Forest exclusions use immutable permitted city facts for every canopy path."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=Path(__file__).resolve().parents[2]


class CanopyWorldInputTests(unittest.TestCase):
    def test_offscreen_city_exclusions_and_partial_city_changes(self):
        source=(ROOT/'Renderer/native/source_fidelity/geometry.h').read_text()
        start=source.index('        std::vector<BuildingBounds> buildings;')
        end=source.index('        auto emit_forest_instance=',start)
        loop=source[start:end].replace('        std::vector<BuildingBounds> buildings;','')
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cmath>
#include <array>
#include <vector>
using namespace c3x_renderer::render_core;
struct BuildingBounds {float x0,y0,x1,y1;};
struct Instance {std::array<float,2> offset{};std::array<float,4> bounds{};};
struct Composition {std::vector<Instance> instances;};
struct Placement {unsigned asset_index=0;float scale=1;};
struct Group {std::vector<Placement> placements;};
struct Vertex {float position[3]{};};
struct Asset {std::vector<Vertex> vertices;};
struct Bundle {std::vector<Asset> assets;};
namespace c3x_renderer {
Group const* find_feature_group(Bundle const&,char const*){return nullptr;}
float stable_random(unsigned){return 0;}
// These instances carry no site-optional flag, which production always keeps
// at their authored offsets.
namespace city_fidelity {
struct SiteBody {unsigned index;float offset[2];bool moved;};
struct Site {std::vector<SiteBody> layout(Composition const& c)const{
 std::vector<SiteBody> bodies;for(unsigned i=0;i<c.instances.size();++i)bodies.push_back({i,{c.instances[i].offset[0],c.instances[i].offset[1]},false});
 return bodies;}};
template<class... Queries> Site site_filter(Composition const&,int,int,Queries...){return {};}
}
}
int main(){
 CapturedScene scene;auto& topology_cache=scene;
 c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
 scene.publication_scope(frame,{1,1,1,1},1);
 c3x_renderer_tile_v1 tile{};tile.tile_x=16;tile.tile_y=8;tile.real_terrain_type=7;
 tile.city_id=-1;tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 auto city=tile;city.tile_x=18;city.tile_y=8;city.real_terrain_type=2;city.city_id=17;city.city_size=1;
 city.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_CITY_BODY_KNOWN;
 bool changed=false;assert(scene.publish(tile,changed)&&scene.publish(city,changed));
 frame.tiles=&tile;frame.tile_count=1;assert(scene.begin(frame));
 assert(scene.update(tile,2,-1,2,CapturedScene::topology(tile)));scene.finish();
 assert(!scene.current(scene.key(18,8))); // the neighboring city is offscreen
 auto ground_observations=scene.compilation_view(true);
 auto observed_coordinate_key=[&](int x,int y){return scene.key(x,y);};
 int nc=12,nr=4;Bundle city_bundle;Composition composition;
 auto selected_city=[&](auto const& record,int,int){
  composition.instances={{{.1f,.2f},{0,0,float(record.city_size)*.25f,.5f}}};
  return &composition;
 };
 // Site queries for version-five packs; unread by unflagged bodies.
 auto world_lookup=[](int,int){return 0;};
 auto shore_sample_at=[](float,float){struct Sample {float distance=99;};return Sample{};};
 struct Natural {struct Sample {float distance=99;};Sample river_sample(std::array<float,2>)const{return {};}} natural;
 auto height_natural=[](float,float){return 0.f;};
 std::vector<BuildingBounds> buildings;
 auto collect=[&]()->bool{buildings.clear();
'''+loop+r'''
 return true;};
 assert(collect()&&buildings.size()==1);
 // Neighbor (18,8) maps to (13,5); exclusion follows the permitted city recipe.
 assert(std::abs(buildings[0].x1-13.85f)<.0001f);
 assert(!scene.retained(scene.key(18,8))->authoritative);
 auto old=ground_observations;
 city.city_size=2;assert(scene.publish(city,changed));
 ground_observations=scene.compilation_view(true);assert(collect()&&buildings.size()==1);
 assert(std::abs(buildings[0].x1-14.10f)<.0001f);
 ground_observations=old;assert(collect()&&std::abs(buildings[0].x1-13.85f)<.0001f);
 city.city_id=-1;assert(scene.publish(city,changed));
 ground_observations=scene.compilation_view(true);assert(collect()&&buildings.empty());
}
''')

    def test_every_canopy_path_records_neighbor_appearance(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('            bool const canopy_city_dependencies=')
        end=source.index('            float left =',start)
        proof=source[start:end]
        start=source.index('            if(canopy_city_dependencies)for')
        end=source.index('            }\n',start)+len('            }\n')
        key=source[start:end]
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <array>
#include <unordered_map>
using namespace c3x_renderer::render_core;
int main(){
 CapturedScene topology_cache;c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
 topology_cache.publication_scope(frame,{1,1,1,1},1);
 auto coordinate_key=[&](int x,int y){return topology_cache.key(x,y);};
 for(auto state:std::vector<std::array<int,4>>{{7,0,0,1},{8,0,0,1},{5,7,0,1},{5,8,0,1},{6,0,7,1},{10,0,8,1},{2,0,0,0}}){
  c3x_renderer_tile_v1 tile{};tile.tile_x=16;tile.tile_y=8;tile.real_terrain_type=state[0];
  int hill_vegetation=state[1],raised_vegetation=state[2];bool retained_world=true;
  std::unordered_map<std::uint64_t,std::uint64_t> appearance_dependencies;
'''+proof+r'''
  assert(canopy_city_dependencies==bool(state[3]));
  assert(appearance_dependencies.size()==(state[3]?25u:0u));
  if(state[3])assert(appearance_dependencies.count(coordinate_key(18,8)));
  struct Observations {unsigned calls=0;CapturedScene::Observation const* current(std::uint64_t){++calls;return nullptr;}} ground_observations;
  auto tile_content_signature=[](auto const&){return std::uint64_t(7);};
  std::uint64_t natural_key=1469598103934665603ull;
'''+key+r'''
  assert(ground_observations.calls==(state[3]?25u:0u));
 }
}
''')

    def test_legacy_canopy_key_follows_neighbor_routes_resources_and_sites(self):
        # The canopy clears neighbors' routes, resources, mines and sites, so
        # the legacy shared-natural key must change with them.
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('            if(canopy_city_dependencies)for')
        end=source.index('            }\n',start)+len('            }\n')
        key=source[start:end]
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
using namespace c3x_renderer::render_core;
int main(){
 auto coordinate_key=[](int x,int y){return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);};
 c3x_renderer_tile_v1 tile{};tile.tile_x=16;tile.tile_y=8;bool canopy_city_dependencies=true;
 struct Observations {CapturedScene::Observation value{};
  CapturedScene::Observation const* current(std::uint64_t){return &value;}} ground_observations;
 auto tile_content_signature=[](c3x_renderer_tile_v1 const& t){
  return std::uint64_t(t.road_mask*3+t.railroad_mask*5+std::uint64_t(t.resource_id+1)*7+t.improvement_flags*11+
   std::uint64_t(t.city_id+1)*13+t.territory_color_rgb*17);};
 auto run=[&]{std::uint64_t natural_key=1469598103934665603ull;
'''+key+r'''
  return natural_key;};
 auto& n=ground_observations.value.occurrence;
 auto reset=[&]{n=c3x_renderer_tile_v1{};n.city_id=n.resource_id=-1;};
 reset();auto bare=run();
 for(int edit=0;edit<6;++edit){reset();
  if(edit==0)n.road_mask=1;
  if(edit==1)n.railroad_mask=1;
  if(edit==2)n.resource_id=3;
  if(edit==3)n.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;
  if(edit==4)n.improvement_flags=C3X_RENDERER_IMPROVEMENT_GOODY_HUT;
  if(edit==5)n.improvement_flags=C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP;
  assert(run()!=bare);
 }
 // Facts the canopy never reads keep the key.
 reset();n.territory_color_rgb=0x123456;assert(run()==bare);
 reset();n.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;assert(run()==bare);
}
''')

    def test_forest_stands_around_huts_and_camps(self):
        # A village or camp merges its pieces into a few assets whose bounds
        # spanned the whole tile, so the canopy kept no tree there: the forest
        # vanished until the hut was entered (2026-10-23). The canopy now keeps
        # clear of the standing pieces only and rings them with the full stand.
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('    template<class Lookup> c3x_renderer::fidelity::CanopyClearing canopy_clearing(')
        end=source.index('        return clearing;\n    }\n',start)+len('        return clearing;\n    }\n')
        clearing=source[start:end]
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include "Renderer/native/hill_vegetation.h"
#include "Renderer/lab/shared/natural/mesh.h"
#include <cassert>
#include <map>
namespace c3x_renderer {
std::uint32_t stable_hash(std::uint32_t value){return patterns::feature_hash(value);}
float stable_random(std::uint32_t value){return patterns::stable_random(value);}
FeatureGroup const* find_feature_group(FeatureBundle const& bundle,char const* name){
 for(auto const& group:bundle.groups)if(group.name==name)return &group;
 return nullptr;}
}
using namespace c3x_renderer;
using fidelity::CanopyClearing;
struct Observation {c3x_renderer_tile_v1 occurrence;};
struct Renderer {
 FeatureBundle bridge_bundle,site_bundle,mine_bundle,farm_bundle,city_bundle,wall_bundle;
 objects::RoutePatterns road_patterns,rail_patterns;
 bool route_assets_ready=false,site_assets_ready=true,mine_assets_ready=false;
 int ground_type(c3x_renderer_tile_v1 const&)const{return 2;}
 // One stationary plant in the middle of a resource tile.
 template<class Part> void resource_footprint(c3x_renderer_tile_v1 const& t,int c,int r,Part part)const{
  if(t.resource_id>=0)part(float(c)+.45f,float(r)+.45f,float(c)+.55f,float(r)+.55f,false);}
 std::vector<std::array<float,4>> resource_boxes(c3x_renderer_tile_v1 const&,bool=false)const{return {};}
'''+clearing+r'''
};
// A pyramid standing on a square, or a flat square when tall is zero.
void piece(FeatureAsset& asset,float x,float y,float half,float tall){
 auto at=[&](float px,float py,float pz){
  asset.vertices.push_back({{px,py,pz},{0,0,1},{0,0}});return std::uint32_t(asset.vertices.size()-1);};
 std::uint32_t a=at(x-half,y-half,0),b=at(x+half,y-half,0),c=at(x+half,y+half,0),d=at(x-half,y+half,0);
 if(tall<=0){for(auto i:{a,b,c,d})asset.vertices[i].position[2]=.01f;
  asset.indices.insert(asset.indices.end(),{a,b,c,a,c,d});return;}
 std::uint32_t top=at(x,y,tall);
 asset.indices.insert(asset.indices.end(),{a,b,top,b,c,top,c,d,top,d,a,top});
}
using Layers=std::array<std::vector<fidelity::MapVertex>,3+fidelity::max_natural_bodies>;
struct Tree {float x0,y0,x1,y1,height;};
int main(){
 // test_canopy's cone bodies and recipes.
 fidelity::NaturalData natural;
 natural.materials.resize(23);natural.bodies.resize(23);
 for(unsigned i=0;i<23;i++){
  natural.bodies[i].material=i;
  auto& v=natural.bodies[i].vertices;
  for(unsigned k=0;k<6;k++){float a=float(k)*1.0471976f;
   v.push_back({{std::cos(a)*.3f,std::sin(a)*.3f,0},{0,0,1},{0,0}});
   v.push_back({{std::cos(a+.5f)*.3f,std::sin(a+.5f)*.3f,0},{0,0,1},{1,0}});
   v.push_back({{0,0,.6f},{0,0,1},{.5f,1}});}
 }
 for(unsigned i=0;i<25;i++)natural.recipes.push_back({i%22,.6f,.1f,i==24?12u:7u,0,0,3,0,0});
 for(unsigned i=25;i<35;i++)natural.recipes.push_back({i%22,.6f,.1f,i==34?13u:12u,0,0,3,0,0});
 // As the site pack bakes them: a village's pieces merged into one asset,
 // six huts about the centre and six low mats out near the tile edge; a
 // camp's palisade ring of stakes around a tent.
 Renderer renderer;
 FeatureAsset village;village.id="hut_0:e0";
 for(int k=0;k<6;k++){float a=float(k)*1.0471976f;
  piece(village,std::cos(a)*.15f,std::sin(a)*.15f,.03f,.1f);
  piece(village,std::cos(a+.5f)*.27f,std::sin(a+.5f)*.27f,.012f,0);}
 FeatureAsset camp;camp.id="camp:e0";piece(camp,0,0,.05f,.12f);
 for(int k=0;k<40;k++){float a=float(k)*.15707963f;piece(camp,std::cos(a)*.21f,std::sin(a)*.21f,.012f,.1f);}
 renderer.site_bundle.assets={village,camp};
 for(char const* name:{"hut_0","hut_1","hut_2","camp"}){
  FeatureGroup group;group.name=name;FeaturePlacement placement{};
  placement.asset_index=std::string(name)=="camp"?1:0;group.placements.push_back(placement);
  renderer.site_bundle.groups.push_back(group);}
 std::map<std::pair<int,int>,Observation> world;
 auto lookup=[&](int x,int y)->Observation const*{auto it=world.find({x,y});return it==world.end()?nullptr:&it->second;};
 auto make=[&](int x,int y,unsigned flags,int resource){
  c3x_renderer_tile_v1 tile{};tile.tile_x=x;tile.tile_y=y;tile.real_terrain_type=7;tile.terrain_type=2;
  tile.city_id=-1;tile.resource_id=resource;tile.barbarian_tribe_id=3;tile.improvement_flags=flags;
  tile.variant_seed=unsigned(x*73856093)^unsigned(y*19349663);
  world[{x,y}]={tile};return tile;};
 auto trees=[&](c3x_renderer_tile_v1 const& tile){
  int c=(tile.tile_x+tile.tile_y)/2,r=(tile.tile_x-tile.tile_y)/2;
  CanopyClearing clearing=renderer.canopy_clearing(tile,lookup);
  fidelity::Tile owner{tile.tile_x,tile.tile_y,c,r,7};
  fidelity::GroundProjection projection{c,r,64,32,128.f/224*.82f,640};Layers layers;
  struct Shore {float distance,beach_width;};
  bool ok=fidelity::emit_forest(natural,owner,projection,{},[](float,float){return 2.5f;},
   [](float,float){return Shore{100,.2f};},[](float,float){return 1000.f;},
   patterns::feature_hash,patterns::stable_random,[]{return false;},layers,clearing);
  assert(ok);
  std::vector<Tree> out;
  for(unsigned body=0;body<natural.bodies.size();body++){
   auto const& v=layers[3+body];auto n=natural.bodies[body].vertices.size();
   for(std::size_t first=0;first+n<=v.size();first+=n){
    Tree t{1e9f,1e9f,-1e9f,-1e9f,0};float z0=1e9f,z1=-1e9f;
    for(std::size_t i=first;i<first+n;i++){
     t.x0=std::min(t.x0,v[i].world_x);t.x1=std::max(t.x1,v[i].world_x);
     t.y0=std::min(t.y0,v[i].world_y);t.y1=std::max(t.y1,v[i].world_y);
     z0=std::min(z0,v[i].world_z);z1=std::max(z1,v[i].world_z);}
    t.height=(z1-z0)*112/(150.f/(.82f*64.f)*112);out.push_back(t);}
  }
  return out;
 };
 for(int x=16;x<=24;x++)for(int y=6;y<=14;y++)if((x+y)%2==0)make(x,y,0,-1);
 unsigned plain=unsigned(trees(world[{20,10}].occurrence).size());
 unsigned hut_trees=0,camp_trees=0,resource_trees=0,neighbor_loss=0;
 for(unsigned variant=0;variant<4;variant++)for(unsigned kind=0;kind<3;kind++){
  // A goody hut, a camp, or a hut on a resource tile, each turned four ways.
  auto site=make(20,10,kind==1?C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP:C3X_RENDERER_IMPROVEMENT_GOODY_HUT,kind==2?7:-1);
  site.variant_seed=variant*0x2545f491u;world[{20,10}].occurrence=site;
  objects::Assets assets{{&renderer.bridge_bundle,&renderer.site_bundle,&renderer.mine_bundle,&renderer.farm_bundle,&renderer.city_bundle,&renderer.wall_bundle}};
  objects::Plan plan;
  assert(objects::select_improvements(site,assets,2,site.improvement_flags,false,false,plan) && plan.instances.size()==1);
  auto const& instance=plan.instances[0];auto const& asset=renderer.site_bundle.assets[instance.asset];
  float cx=15+instance.u,cy=5+1-instance.v,co=std::cos(instance.rotation),si=std::sin(instance.rotation);
  auto stand=trees(site);
  (kind==0?hut_trees:kind==1?camp_trees:resource_trees)+=unsigned(stand.size());
  for(auto const& t:stand){
   // No crown stands on or hides a standing piece (test_canopy's screen
   // test); none stands inside the palisade.
   for(std::size_t i=0;i+2<asset.indices.size();i+=3){
    float top=0,x0=1e9f,y0=1e9f,x1=-1e9f,y1=-1e9f;
    for(std::size_t k=i;k<i+3;k++){auto const& p=asset.vertices[asset.indices[k]].position;
     float x=cx+(p[0]*co-p[1]*si)*instance.scale,y=cy-(p[0]*si+p[1]*co)*instance.scale;
     top=std::max(top,p[2]);x0=std::min(x0,x);x1=std::max(x1,x);y0=std::min(y0,y);y1=std::max(y1,y);}
    if(top<.05f)continue;
    assert(!(t.x0<x1 && t.x1>x0 && t.y0<y1 && t.y1>y0));
    float tx=(t.x0+t.x1)*.5f,ty=(t.y0+t.y1)*.5f,px=(x0+x1)*.5f,py=(y0+y1)*.5f;
    float across=std::abs((tx+ty)-(px+py)),width=((t.x1-t.x0)+(t.y1-t.y0))*.5f,up=(tx-ty)-(px-py);
    assert(!(across<width*.8f && up>0 && up<CanopyClearing::hide_fraction*t.height*150.f/32.f*.8f));
   }
   if(kind==1)assert(std::hypot((t.x0+t.x1)*.5f-cx,(t.y0+t.y1)*.5f-cy)>.21f*instance.scale);
  }
  // The neighbors keep their forest beside the site.
  for(auto [x,y]:{std::pair{19,9},std::pair{21,11},std::pair{19,11},std::pair{21,9}}){
   auto with=trees(world[{x,y}].occurrence).size();
   make(20,10,0,-1);neighbor_loss+=unsigned(trees(world[{x,y}].occurrence).size()-with);
   world[{20,10}].occurrence=site;
  }
 }
 // The old clearing kept no tree on a hut or camp tile. A lone site keeps
 // the full stand around its standing pieces; a resource keeps its thin one.
 assert(hut_trees>=plain*4/3 && camp_trees>=plain*4/6 && neighbor_loss<=4);
 assert(resource_trees>0 && resource_trees<=12*4 && resource_trees<hut_trees);
}
''')

if __name__=='__main__':unittest.main()
