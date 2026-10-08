"""CPU object descriptions, terrain dependencies and exact source attributes."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ObjectCompilerTests(unittest.TestCase):
    def test_metropolis_wall_uses_compact_radius_until_outer_building_needs_space(self):
        run_cpp(r'''
#include "Renderer/native/city_fidelity/runtime.h"
#include <cassert>
using namespace c3x_renderer::city_fidelity;
int main(){
 Composition city;city.size=2;
 Instance outer;outer.offset[0]=.60f;outer.bounds[0]=-.06f;
 outer.bounds[2]=.06f;outer.bounds[1]=-.03f;outer.bounds[3]=.03f;
 city.instances.push_back(outer);
 float compact=metropolis_wall_radius(city);
 assert(compact>=.70f && compact<.71f);
 city.instances[0].offset[0]=.72f;
 float expanded=metropolis_wall_radius(city);
 assert(expanded>.80f && expanded<.82f);
 city.size=1;
 assert(metropolis_wall_radius(city)==0.0f);
}
''')

    def test_city_wall_follows_varying_hill_ground(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 FeatureBundle wall;FeatureAsset asset;asset.id="city/walls/medieval/segment_01";
 asset.vertices={{{-.2f,0,0},{0,0,1},{0,0}},
                 {{.2f,0,0},{0,0,1},{1,0}},
                 {{.2f,0,.02f},{0,0,1},{1,1}}};
 asset.indices={0,1,2};wall.assets.push_back(asset);
 FeaturePlacement placement{};objects::Projection p;p.tile_width=128;
 p.tile.real_terrain_type=5;
 p.content_view_height=640;p.half_w=64;p.half_h=32;
 p.relief_projection_scale=128.f/224.f*.82f;
 p.feature_projection_scale=128.f/224.f;p.pickup_profile=p.world_objects=true;
 auto relief=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float x,float){return 10*x;};
 std::vector<objects::Vertex> vertices,shadows;
 objects::append_instance(p,wall,placement,.5f,.5f,0,1,21,.01f,false,false,
                          relief,height,vertices,shadows);
 assert(vertices.size()==3);
 assert(std::abs(vertices[0].world_z-3.02f/112.f)<.0001f);
 assert(std::abs(vertices[1].world_z-7.02f/112.f)<.0001f);
 assert(vertices[2].world_z>vertices[1].world_z);
 std::vector<objects::Vertex> next;
 objects::append_instance(p,wall,placement,.8f,.5f,0,1,21,.01f,false,false,
                          relief,height,next,shadows);
 assert(next.size()==3);
 assert(next[0].world_z>vertices[0].world_z);
}
''')

    def test_farm_river_edge_trims_as_one_clean_bank(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
#include <cmath>
using namespace c3x_renderer;
int main(){
 FeatureBundle farm;FeatureAsset field;field.id="farm_0:crop:field:e0";
 for(unsigned y=0;y<=8;++y)for(unsigned x=0;x<=8;++x)
  field.vertices.push_back({{float(x)*.1f-.4f,float(y)*.1f-.4f,.002f},
                            {0,0,1},{float(x)*.125f,float(y)*.125f}});
 for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x){
  unsigned a=y*9+x,b=a+1,c=a+9,d=c+1;
  field.indices.insert(field.indices.end(),{a,b,d,a,d,c});
 }
 farm.assets.push_back(field);FeaturePlacement placement{};
 objects::Projection p;p.tile_width=128;p.content_view_height=640;
 p.half_w=64;p.half_h=32;p.relief_projection_scale=128.f/224.f*.82f;
 p.feature_projection_scale=128.f/224.f;p.pickup_profile=p.world_objects=true;
 auto river=[](float u,float v){
  return std::array<float,3>{0,0,u-(.48f+.02f*std::sin(v*12.f))};};
 auto height=[](float,float){return 2.5f;};
 std::vector<objects::Vertex> vertices,shadows;std::vector<unsigned> indices;
 objects::append_instance(p,farm,placement,.5f,.5f,0,1,21,.01f,false,false,
                          river,height,vertices,shadows,&indices);
 assert(!indices.empty() && vertices.size()==indices.size());
 float left=1.f;
 for(auto const& vertex:vertices){
  left=std::min(left,vertex.world_x);
  assert(river(vertex.world_x,vertex.world_y)[2]>=-1e-4f);
 }
 assert(left>.51f && left<.65f);
}
''')

    def test_lab_city_buildings_use_terraces_on_hills(self):
        run_cpp(r'''
#include "Renderer/native/city_fidelity/compiler.h"
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 city_fidelity::Library library;library.materials.resize(2);
 library.materials[0].channels=5;library.materials[1].channels=3;
 city_fidelity::Model model;model.low[2]=0;model.high[2]=.3f;
 city_fidelity::Part part;city_fidelity::Vertex vertex{};
 vertex.normal[2]=1;vertex.tangent[0]=1;vertex.bitangent[1]=1;
 part.vertices={vertex,vertex,vertex};part.vertices[1].position[0]=.1f;
 part.vertices[2].position[1]=.1f;part.indices={0,1,2};model.parts.push_back(part);
 library.models.push_back(model);
 city_fidelity::Composition composition;composition.culture=0;composition.era=0;
 composition.size=0;composition.authority="lab-fixed-hill";composition.clearance[1]=18;
 composition.foundation_material=1;composition.foundation_uv[0]=.60f;
 composition.foundation_uv[1]=.55f;composition.foundation_uv[2]=.79f;
 composition.foundation_uv[3]=.72f;
 composition.foundation_step[0]=.1f;composition.foundation_step[1]=1.f;
 city_fidelity::Instance instance;instance.scale=1;
 instance.bounds[0]=instance.bounds[1]=-.2f;
 instance.bounds[2]=instance.bounds[3]=.2f;
 composition.instances.push_back(instance);library.compositions.push_back(composition);
 c3x_renderer_tile_v1 tile{};tile.city_id=1;
 struct Land{int base=2,real=2;};struct Shore{double distance=100;};
 auto world=[](int,int){return Land{};};auto shore=[](float,float){return Shore{};};
 auto river=[](float,float){return 100.;};
 auto hill=[](float x,float){return (x-10.5f)*12.f;};
 auto selected=city_fidelity::select(library,tile,10,12,world,shore,river,hill);
 assert(selected==&library.compositions[0]);
 library.compositions[0].clearance[1]=2.5f;
 assert(!city_fidelity::select(library,tile,10,12,world,shore,river,hill));
 library.compositions[0].clearance[1]=18;
 fidelity::GroundProjection projection{10,12,64,32,1,480};
 city_fidelity::Surfaces output;
 assert(city_fidelity::compile(library,*selected,10,12,hill,projection,output));
 assert(output.chunks.size()==2);
 auto const& terrace=output.chunks[0];auto const& building=output.chunks[1];
 assert(terrace.vertices.size()>198 && building.vertices.size()==3);
 assert(terrace.material==1 && building.material==0 && !terrace.terrain_conforming);
 assert(terrace.vertices[0].u==.60f && terrace.vertices[2].v==.72f);
 assert(terrace.vertices[6].u==.60f && terrace.vertices[6].v==.55f);
 assert(terrace.vertices[0].world_z>terrace.vertices[2].world_z);
 assert(terrace.vertices[6].world_z>terrace.vertices[8].world_z);
 assert(building.vertices[0].normal_z==1 && building.vertices[0].world_z>0);
 auto uneven=[](float x,float y){return 9.f-40.f*(std::abs(x-10.55f)+std::abs(y-12.55f));};
 city_fidelity::Surfaces peak_output;
 assert(city_fidelity::compile(library,*selected,10,12,uneven,projection,peak_output));
 assert(peak_output.chunks.size()==2);
 for(auto const& v:peak_output.chunks[1].vertices)assert(v.world_z>9.f/112.f);
}
''')

    def test_farm_fields_fit_inside_dry_quadrants_before_clipping(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 FeatureBundle farm;FeatureGroup group;group.name="farm_0";
 for(unsigned palette=0;palette<3;++palette){
  FeatureAsset asset;asset.id="farm_0:crop:field:e0";asset.texture_index=palette;
  asset.vertices={{{-.12f,-.12f,.002f},{0,0,1},{0,0}},
                  {{ .12f,-.12f,.002f},{0,0,1},{1,0}},
                  {{ .12f, .12f,.002f},{0,0,1},{1,1}},
                  {{-.12f, .12f,.002f},{0,0,1},{0,1}}};
  asset.indices={0,1,2,0,2,3};farm.assets.push_back(asset);
  FeaturePlacement placement{};placement.asset_index=palette;group.placements.push_back(placement);
 }
 farm.groups.push_back(group);FeatureBundle other;
 objects::Assets assets{{&other,&other,&other,&farm,&other,&other}};
 c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 tile.variant_seed=18;
 objects::Plan plan;
 assert(objects::select_improvements(tile,assets,0,0,false,true,plan));
 assert(plan.instances.size()==4);
 auto original=plan.instances;
 auto coast=[](float u,float){return std::array<float,3>{0,0,u-.15f};};
 objects::settle_farm_fields(plan,tile,assets,coast);
 assert(plan.instances.size()==4);
 for(unsigned i=0;i<4;++i)if(original[i].u<.5f)
  assert(plan.instances[i].scale<original[i].scale);
 objects::Projection p;p.tile_width=128;p.content_view_height=640;
 p.half_w=64;p.half_h=32;p.relief_projection_scale=128.f/224.f*.82f;
 p.feature_projection_scale=128.f/224.f;p.pickup_profile=p.world_objects=true;
 auto height=[](float,float){return 2.5f;};objects::Surfaces out;
 objects::compile(plan,p,assets,coast,height,out);
 assert(!out.layers[objects::farm_layer].empty());
 for(auto const& vertex:out.layers[objects::farm_layer])
  assert(vertex.world_x>=.175f-1e-4f);
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    def test_farm_trees_and_building_relocate_to_dry_shore(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 FeatureBundle farm;FeatureGroup group;group.name="farm_0";
 for(auto id:{"farm_0:tree:source:e0","farm_0:building:source:e0"}){
  FeatureAsset asset;asset.id=id;farm.assets.push_back(asset);
  FeaturePlacement placement{};placement.asset_index=unsigned(farm.assets.size()-1);
  group.placements.push_back(placement);
 }
 farm.groups.push_back(group);
 FeatureBundle other;
 objects::Assets assets{{&other,&other,&other,&farm,&other,&other}};
 c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 objects::Plan plan;
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 for(unsigned seed=0;seed<128;++seed){
  tile.variant_seed=seed;plan.instances.clear();
  assert(objects::select_improvements(tile,assets,0,0,false,true,plan));
  objects::settle_farm_props(plan,tile,assets,dry);
  unsigned trees=0,buildings=0;
  for(auto const& instance:plan.instances){
   if(instance.asset==0)++trees;else ++buildings;
  }
  assert(trees>=4 && trees<=6 && buildings==1);
 }
 tile.variant_seed=7;plan.instances.clear();
 assert(objects::select_improvements(tile,assets,0,0,false,true,plan));
 auto coast=[](float u,float v){return std::array<float,3>{0,0,u>.55f&&v<.83f?1.f:-1.f};};
 objects::settle_farm_props(plan,tile,assets,coast);
 unsigned trees=0,buildings=0;
 for(auto const& instance:plan.instances){
  assert(coast(instance.u,1.f-instance.v)[2]>0);
  if(instance.asset==0)++trees;else ++buildings;
 }
 assert(trees>=4 && buildings==1);
 auto river=[](float u,float){return std::array<float,3>{0,0,std::abs(u-.5f)>.16f?1.f:-1.f};};
 objects::settle_farm_props(plan,tile,assets,river);
 for(auto const& instance:plan.instances)
  assert(river(instance.u,1.f-instance.v)[2]>0);
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    def test_four_farm_fields_stay_in_separate_quadrants(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
#include <cmath>
using namespace c3x_renderer;
int main(){
 FeatureBundle bundle;FeatureGroup group;group.name="farm_0";
 for(unsigned palette=0;palette<3;++palette){
  FeatureAsset asset;asset.id="farm_0:crop:field:e0";asset.texture_index=palette;
  asset.vertices={{{-.12f,-.12f,.002f},{0,0,1},{0,0}},
                  {{ .12f,-.12f,.002f},{0,0,1},{1,0}},
                  {{ .12f, .12f,.002f},{0,0,1},{1,1}},
                  {{-.12f, .12f,.002f},{0,0,1},{0,1}}};
  asset.indices={0,1,2,0,2,3};bundle.assets.push_back(asset);
  FeaturePlacement placement{};placement.asset_index=palette;group.placements.push_back(placement);
 }
 bundle.groups.push_back(group);
 objects::Assets assets{{&bundle,&bundle,&bundle,&bundle,&bundle,&bundle}};
 c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 for(unsigned seed=0;seed<128;++seed)for(unsigned mask=0;mask<16;++mask)
  for(int ground=0;ground<5;++ground){
   tile.variant_seed=seed;tile.irrigation_mask=mask;
   objects::Plan plan;
   assert(objects::select_improvements(tile,assets,ground,0,false,true,plan));
   assert(plan.instances.size()==4);
   std::array<std::array<float,4>,4> bounds{};
   bool used_palette[3]={};
   for(unsigned i=0;i<4;++i){
    auto const& instance=plan.instances[i];auto const& asset=bundle.assets[instance.asset];
    used_palette[asset.texture_index]=true;
    float cosine=std::cos(instance.rotation),sine=std::sin(instance.rotation);
    bounds[i]={2,2,-1,-1};
    for(auto const& vertex:asset.vertices){
     float x=instance.u+(vertex.position[0]*cosine-vertex.position[1]*sine)*instance.scale;
     float y=instance.v+(vertex.position[0]*sine+vertex.position[1]*cosine)*instance.scale;
     bounds[i][0]=std::min(bounds[i][0],x);bounds[i][1]=std::min(bounds[i][1],y);
     bounds[i][2]=std::max(bounds[i][2],x);bounds[i][3]=std::max(bounds[i][3],y);
    }
    assert(bounds[i][0]>.02f && bounds[i][1]>.02f &&
           bounds[i][2]<.98f && bounds[i][3]<.98f);
   }
   for(unsigned i=0;i<4;++i)for(unsigned j=i+1;j<4;++j)
    assert(bounds[i][2]+.01f<bounds[j][0] || bounds[j][2]+.01f<bounds[i][0] ||
           bounds[i][3]+.01f<bounds[j][1] || bounds[j][3]+.01f<bounds[i][1]);
   assert(unsigned(used_palette[0])+unsigned(used_palette[1])+unsigned(used_palette[2])==2);
  }
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    def test_farm_decal_follows_relief_and_stops_at_shore(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
#include <cmath>
using namespace c3x_renderer;
int main(){
 FeatureBundle bundle;FeatureAsset asset;asset.id="farm_0:crop:field:e0";
 asset.vertices={{{-.4f,-.3f,.002f},{0,0,1},{0,0}},
                 {{.4f,-.3f,.002f},{0,0,1},{1,0}},
                 {{.4f,.3f,.002f},{0,0,1},{1,1}},
                 {{-.4f,.3f,.002f},{0,0,1},{0,1}}};
 asset.indices={0,1,2,0,2,3};bundle.assets.push_back(asset);
 FeaturePlacement placement{};objects::Projection p;p.tile_width=128;p.content_view_height=640;
 p.half_w=64;p.half_h=32;p.relief_projection_scale=128.f/224.f*.82f;
 p.feature_projection_scale=128.f/224.f;p.pickup_profile=p.world_objects=true;
 auto relief=[](float u,float){return std::array<float,3>{u*4.f,0,u-.5f};};
 auto height=[](float,float){return 2.5f;};
 std::vector<objects::Vertex> vertices,shadows;std::vector<unsigned> indices;
 objects::append_instance(p,bundle,placement,.5f,.5f,0,1,21,.01f,false,false,
                          relief,height,vertices,shadows,&indices);
 assert(indices.size()==9 && vertices.size()==9 && shadows.empty());
 objects::Assets assets{{&bundle,&bundle,&bundle,&bundle,&bundle,&bundle}};
 objects::Plan plan;plan.instances.push_back({objects::farm_family,0,objects::farm_layer,
                                           .5f,.5f,0,1,21,.01f,false});
 objects::Surfaces compiled;std::vector<unsigned> counts;
 objects::compile(plan,p,assets,relief,height,compiled,true,&counts);
 assert(counts.size()==1 && counts[0]==9 &&
        compiled.indices[objects::farm_layer].size()==9);
 for(auto const& v:vertices){
  assert(v.world_x>=.5f-1e-5f);
  assert(std::abs(v.world_z-(v.world_x*4.f+2.5f+.002f*150.f/.82f)/112.f)<1e-5f);
 }
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    def test_farm_decal_leaves_an_interior_river_channel_open(self):
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
#include <cmath>
using namespace c3x_renderer;
int main(){
 FeatureBundle bundle;FeatureAsset asset;asset.id="farm_0:crop:field:e0";
 for(unsigned y=0;y<5;++y)for(unsigned x=0;x<5;++x)
  asset.vertices.push_back({{float(x)*.25f-.5f,float(y)*.25f-.5f,.002f},
                            {0,0,1},{float(x)*.25f,float(y)*.25f}});
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x){
  unsigned a=y*5+x,b=a+1,c=a+5,d=c+1;
  asset.indices.insert(asset.indices.end(),{a,b,d,a,d,c});
 }
 bundle.assets.push_back(asset);FeaturePlacement placement{};
 objects::Projection p;p.tile_width=128;p.content_view_height=640;
 p.half_w=64;p.half_h=32;p.relief_projection_scale=128.f/224.f*.82f;
 p.feature_projection_scale=128.f/224.f;p.pickup_profile=p.world_objects=true;
 auto relief=[](float u,float){return std::array<float,3>{0,0,std::abs(u-.5f)-.12f};};
 auto height=[](float,float){return 2.5f;};
 std::vector<objects::Vertex> vertices,shadows;std::vector<unsigned> indices;
 objects::append_instance(p,bundle,placement,.5f,.5f,0,1,21,.01f,false,false,
                          relief,height,vertices,shadows,&indices);
 assert(!indices.empty() && vertices.size()==indices.size());
 bool left=false,right=false;
 for(auto const& vertex:vertices){
  assert(std::abs(vertex.world_x-.5f)>=.12f-1e-5f);
  left|=vertex.world_x<.5f;right|=vertex.world_x>.5f;
 }
 assert(left && right);
}
''')

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
 for(auto n:{"wall_ancient","wall_medieval","wall_industrial","wall_modern"})group(objects::wall_family,n);
 for(auto n:{"farm_0","farm_1","farm_2"})group(objects::farm_family,n);
 for(auto n:{"mine_0","mine_1","mine_2","mine_3","mine_4","mine_5"})group(objects::mine_family,n);
 for(auto n:{"hut_0","hut_1","hut_2","camp"})group(objects::site_family,n);
 for(auto n:{"bridge_railroad_normal","bridge_modern_normal"})group(objects::bridge_family,n);
 c3x_renderer_tile_v1 tile{};tile.city_id=3;tile.city_flags=C3X_RENDERER_CITY_WALLED|C3X_RENDERER_CITY_CAPITAL;
 tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE|C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 tile.tile_x=48;tile.tile_y=32;
 for(int era=0;era<4;++era)for(int size=0;size<3;++size){
  tile.city_era=era;tile.city_size=size;tile.route_style=era;
  objects::Plan composed;
  assert(objects::select_improvements(tile,assets,2,0,true,true,composed));
  // Cities and their walls belong exclusively to the complete city pack.
  // Even available retired models must never enter this infrastructure pass.
  for(auto const& i:composed.instances)
   assert(i.layer!=objects::city_layer && i.layer!=objects::wall_layer);
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
   assert(out.layers[objects::wall_layer].empty());
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
 assert(routes.routes.size()==8 && routes.instances.size()==1 && queried.size()==8);
 for(auto const&r:routes.routes)assert(r.railroad && r.style==4);
 objects::Plan diagnostic;objects::select_routes(tile,assets,true,false,lookup,diagnostic);
 assert(diagnostic.routes.empty() && diagnostic.instances.size()==1);
 tile.railroad_mask=0;neighbor.occurrence.railroad_mask=0;
 objects::Plan dense_roads;objects::select_routes(tile,assets,true,true,lookup,dense_roads);
 assert(dense_roads.routes.size()>=4 && dense_roads.routes.size()<=8);
 auto missing=[](int,int)->Observation const*{return nullptr;};
 objects::Plan isolated;objects::select_routes(tile,assets,true,true,missing,isolated);
 assert(isolated.routes.size()==1 && !isolated.routes[0].railroad && isolated.routes[0].isolated);
 assert(isolated.routes[0].u0>=.20f && isolated.routes[0].u1<=.80f);
 for(auto offset:std::array<std::array<int,2>,8>{{{1,-1},{2,0},{1,1},{0,2},{-1,1},{-2,0},{-1,-1},{0,-2}}}){
  auto one=[&](int x,int y)->Observation const*{
   return x==tile.tile_x+offset[0] && y==tile.tile_y+offset[1]?&neighbor:nullptr;};
  objects::Plan edge;objects::select_routes(tile,assets,true,true,one,edge);
  assert(edge.routes.size()==1 && !edge.routes[0].railroad);
  assert(edge.routes[0].u0>=.34f && edge.routes[0].u0<=.66f);
  assert(edge.routes[0].v0>=.34f && edge.routes[0].v0<=.66f);
 }
 tile.real_terrain_type=6;
 objects::Plan isolated_mountain;objects::select_routes(tile,assets,true,true,missing,isolated_mountain);
 assert(isolated_mountain.routes.size()==1 && isolated_mountain.routes.front().bypass);
 for(auto offset:std::array<std::array<int,2>,2>{{{2,0},{0,2}}}){
  auto forward=[&](int x,int y)->Observation const*{
   return x==tile.tile_x+offset[0] && y==tile.tile_y+offset[1]?&neighbor:nullptr;};
  objects::Plan mountain;objects::select_routes(tile,assets,true,true,forward,mountain);
  assert(mountain.routes.size()==1);
  for(auto const& route:mountain.routes)assert(route.bypass);
  c3x_renderer_tile_v1 far=tile;far.tile_x+=offset[0];far.tile_y+=offset[1];far.real_terrain_type=2;
  Observation near{tile};auto reverse=[&](int x,int y)->Observation const*{
   return x==tile.tile_x && y==tile.tile_y?&near:nullptr;};
  objects::Plan return_path;objects::select_routes(far,assets,true,true,reverse,return_path);
  assert(return_path.routes.size()==1);
  auto const& outbound=mountain.routes.front();auto const& inbound=return_path.routes.front();
  float outbound_u=float(tile.tile_x+tile.tile_y)*.5f+outbound.u1;
  float outbound_v=float(tile.tile_x-tile.tile_y)*.5f+1.f-outbound.v1;
  float inbound_u=float(far.tile_x+far.tile_y)*.5f+inbound.u1;
  float inbound_v=float(far.tile_x-far.tile_y)*.5f+1.f-inbound.v1;
  assert(std::abs(outbound_u-inbound_u)<.0001f && std::abs(outbound_v-inbound_v)<.0001f);
 }
 tile.real_terrain_type=0;
 tile.railroad_mask=1;neighbor.occurrence.railroad_mask=1;
 objects::Projection projection;projection.tile=tile;projection.tile_width=128;projection.half_w=64;projection.half_h=32;
 projection.pickup_profile=projection.world_objects=true;projection.relief_projection_scale=128.f/224*.82f;
 auto flat=[](float,float){return std::array<float,3>{0,0,0};};auto sloped=[](float u,float v){return std::array<float,3>{u+v,0,0};};
 objects::Surfaces a,b;objects::compile(routes,projection,assets,flat,[](float,float){return 0.f;},a);
 objects::compile(routes,projection,assets,sloped,[](float,float){return 0.f;},b);
 unsigned expected_vertices=0;for(auto const& route:routes.routes)expected_vertices+=route.bridge?336u:192u;
 assert(a.layers[objects::route_layer].size()==expected_vertices);
 for(unsigned i=0;i<expected_vertices;++i){auto const& x=a.layers[objects::route_layer][i];auto const& y=b.layers[objects::route_layer][i];assert(x.u==y.u && x.v==y.v && x.x==y.x && x.y!=y.y);}
 objects::Route deck{.5f,.5f,1.f,.5f,0,false,true,false};std::vector<objects::Vertex> deck_vertices;
 objects::append_route(projection,deck,flat,[](float,float){return 0.f;},deck_vertices);
 assert(deck_vertices.size()==336);
 float deck_top=0;for(auto const& vertex:deck_vertices)deck_top=std::max(deck_top,vertex.world_z);
 assert(deck_top>deck_vertices[0].world_z+.05f);
 objects::Route straight{.5f,.5f,1.f,.5f,0,false,false,false};
 std::vector<objects::Vertex> junction_vertices;
 objects::append_route(projection,straight,flat,[](float,float){return 0.f;},junction_vertices);
 float tile_u=float(tile.tile_x+tile.tile_y)*.5f;
 assert(std::abs(junction_vertices.front().world_x-(tile_u+.5f))<.02f);
 assert(std::abs(junction_vertices.back().world_x-(tile_u+1.f))<.03f);
 std::vector<objects::Vertex> crown_vertices;
 auto exposed_crown=[&](float u,float){return std::abs(u-(tile_u+.75f))<.13f?80.f:0.f;};
 objects::append_route(projection,straight,flat,exposed_crown,crown_vertices);
 assert(!crown_vertices.empty() && crown_vertices.size()<junction_vertices.size());
 objects::Plan wet;wet.routes.push_back({.5f,.5f,1.f,.5f,3,false,false,false});
 objects::promote_river_crossings(tile,[&](float u,float){return std::abs(u-(tile_u+.75f))*100.f;},wet);
 assert(wet.routes[0].bridge && std::abs(wet.routes[0].bridge_t-.5f)<.001f);
 assert(!wet.routes[0].bridge_structural && wet.instances.empty());
 objects::Plan bank;bank.routes.push_back({.5f,.5f,1.f,.5f,3,false,false,false});
 objects::promote_river_crossings(tile,[&](float u,float){return 10.f+(tile_u+1.f-u)*40.f;},bank);
 assert(bank.routes[0].bridge && bank.routes[0].bridge_t>.99f);
 std::vector<objects::Vertex> river_deck;
 auto river_bed=[&](float u,float){return std::abs(u-(tile_u+.75f))<.2f?-60.f:0.f;};
 auto river_relief=[&](float u,float v){return std::array<float,3>{river_bed(u,v),0,0};};
 objects::append_route(projection,wet.routes[0],river_relief,river_bed,river_deck);
 float river_top=-1000;for(auto const& vertex:river_deck)river_top=std::max(river_top,vertex.world_z);
 assert(river_top>.10f);
 objects::Plan sites;assert(objects::select_improvements(tile,assets,2,C3X_RENDERER_IMPROVEMENT_GOODY_HUT|C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP,false,false,sites));
 assert(sites.instances.size()==2);bundles[objects::farm_family].groups.clear();objects::Plan failure;
 assert(!objects::select_improvements(tile,assets,2,0,true,true,failure));
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    def test_road_patterns_follow_connection_masks_and_drape_terrain(self):
        run_cpp(r'''
#include "Renderer/native/rigid_object_instance.h"
#include <cassert>
#include <cstring>
#include <map>
#include <set>
using namespace c3x_renderer;
// Exact shared joins: NE, E, SE, S, SW, W, NW, N edge midpoints and corners.
constexpr float join_u[8]={.5f,1,1,1,.5f,0,0,0},join_v[8]={0,0,.5f,1,1,1,.5f,0};
// masks past 256 are variants of the full mask whose center moves, so each
// variant leaves its joins in a different direction.
std::vector<std::uint8_t> blob(bool corrupt=false,unsigned masks=256){
 std::vector<std::uint32_t> offsets{0};std::vector<std::uint8_t> lines;std::vector<float> points;
 auto line=[&](float a,float b,float c,float d,int start,int end){
  std::uint32_t first=std::uint32_t(points.size()/2);std::uint16_t count=9;
  for(int i=0;i<9;++i){points.push_back(a+(c-a)*i/8.f);points.push_back(b+(d-b)*i/8.f);}
  std::uint8_t record[8];std::memcpy(record,&first,4);std::memcpy(record+4,&count,2);
  record[6]=std::uint8_t(std::int8_t(start));record[7]=std::uint8_t(std::int8_t(end));
  lines.insert(lines.end(),record,record+8);
 };
 for(unsigned index=0;index<masks;++index){
  unsigned mask=std::min(index,255u);
  float cu=.5f,cv=.5f;
  if(index>255){cu+=.04f*float(int((index-255)%3)-1);cv+=.04f*float(int((index-255)/3%3)-1);}
  if(!mask)line(.4f,.5f,.6f,.5f,-1,-1);
  for(int d=0;d<8;++d)if(mask>>d&1)line(cu,cv,join_u[d],join_v[d],-1,d);
  offsets.push_back(std::uint32_t(lines.size()/8));
 }
 std::vector<std::uint8_t> out(8+12);std::memcpy(out.data(),"C3XRPAT1",8);
 std::uint32_t header[3]={masks,offsets.back(),std::uint32_t(points.size()/2)};std::memcpy(out.data()+8,header,12);
 auto append=[&](void const* data,std::size_t size){auto p=static_cast<std::uint8_t const*>(data);out.insert(out.end(),p,p+size);};
 append(offsets.data(),offsets.size()*4);append(lines.data(),lines.size());append(points.data(),points.size()*4);
 if(corrupt)out[20+257*4+6]=9; // impossible join direction
 return out;
}
int main(){
 objects::RoutePatterns patterns;
 auto data=blob();assert(patterns.load(data.data(),data.size()));
 assert(patterns.offsets[255+1]-patterns.offsets[255]==8 && patterns.offsets[1]==1);
 objects::RoutePatterns rejected;auto bad=blob(true);
 assert(!rejected.load(bad.data(),bad.size()) && rejected.lines.empty());
 assert(!rejected.load(data.data(),data.size()-4));
 // A railroad sheet adds 16 variants of the full mask; a fully connected tile
 // picks one by a stable hash of its position, every other mask is itself.
 objects::RoutePatterns rail_set;auto rail_data=blob(false,272);
 assert(rail_set.load(rail_data.data(),rail_data.size()) && rail_set.offsets.size()==273);
 {
  std::set<unsigned> picked;
  for(int y=0;y<40;++y)for(int x=y&1;x<40;x+=2){
   unsigned index=rail_set.index(255,x,y);assert(index>=255 && index<272 && index==rail_set.index(255,x,y));
   picked.insert(index);assert(rail_set.index(17,x,y)==17 && patterns.index(255,x,y)==255);
  }
  assert(picked.size()>12);
 }
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets{};
 for(unsigned f=0;f<bundles.size();++f)assets.bundles[f]=&bundles[f];
 FeatureAsset asset{};asset.id="bridge";asset.vertices={{{0,0,0},{0,0,1},{0,0}}};asset.indices={0,0,0};
 bundles[objects::bridge_family].assets.push_back(asset);
 for(auto n:{"bridge_medieval_normal","bridge_railroad_normal"}){FeatureGroup g;g.name=n;FeaturePlacement p{};p.scale=1;g.placements.push_back(p);bundles[objects::bridge_family].groups.push_back(g);}
 assets.road_patterns=&patterns;
 struct Observation{c3x_renderer_tile_v1 occurrence;};
 c3x_renderer_tile_v1 tile{};tile.tile_x=40;tile.tile_y=20;tile.road_mask=1;tile.city_id=-1;tile.route_style=1;
 constexpr int offsets[8][2]={{1,-1},{2,0},{1,1},{0,2},{-1,1},{-2,0},{-1,-1},{0,-2}};
 std::array<Observation,8> around{};
 auto neighbors=[&](unsigned roads,unsigned rails){
  for(int d=0;d<8;++d){around[d].occurrence=tile;around[d].occurrence.tile_x+=offsets[d][0];around[d].occurrence.tile_y+=offsets[d][1];
   around[d].occurrence.road_mask=roads>>d&1;around[d].occurrence.railroad_mask=rails>>d&1;around[d].occurrence.river_code=0;}
  return [&,roads](int x,int y)->Observation const*{
   for(int d=0;d<8;++d)if(x==tile.tile_x+offsets[d][0] && y==tile.tile_y+offsets[d][1])return &around[d];return nullptr;};
 };
 // Bit k is neighbor k+1 in Civ III order; the tile draws that mask's pattern.
 objects::Plan two;objects::select_routes(tile,assets,true,true,neighbors(3u,0u),two);
 assert(two.routes.empty() && two.patterns.size()==2 && two.instances.empty());
 // Roads keep one dirt look across eras (this tile's era is 1).
 for(auto const& p:two.patterns)assert(p.style==0 && p.bridges==0 && p.line>=patterns.offsets[3] && p.line<patterns.offsets[4]);
 // Both railroad tiles: the railroad owns that link and roads omit it. The
 // railroad follows the same patterns on its own mask, as style 4.
 tile.railroad_mask=1;
 objects::Plan mixed;objects::select_routes(tile,assets,true,true,neighbors(3u,2u),mixed);
 assert(mixed.routes.empty() && mixed.patterns.size()==2);
 assert(mixed.patterns[0].style==0 && patterns.lines[mixed.patterns[0].line].end==0);
 assert(mixed.patterns[1].style==4 && patterns.lines[mixed.patterns[1].line].end==1);
 assert(mixed.patterns[1].line>=patterns.offsets[2] && mixed.patterns[1].line<patterns.offsets[3]);
 objects::Plan rail_only;objects::select_routes(tile,assets,true,true,neighbors(3u,3u),rail_only);
 assert(rail_only.routes.empty() && rail_only.patterns.size()==2);
 for(auto const& p:rail_only.patterns)assert(p.style==4 && p.line>=patterns.offsets[3] && p.line<patterns.offsets[4]);
 // A lone railroad draws the mask-zero mark (and no road) unless a city
 // stands there.
 objects::Plan lone;objects::select_routes(tile,assets,true,true,neighbors(0u,0u),lone);
 assert(lone.patterns.size()==1 && lone.patterns[0].line==0 && lone.patterns[0].style==4);
 tile.city_id=4;objects::Plan rail_city;objects::select_routes(tile,assets,true,true,neighbors(0u,0u),rail_city);
 assert(rail_city.patterns.empty() && rail_city.routes.empty());tile.city_id=-1;
 // With a railroad sheet, railroads draw its patterns (roads keep theirs).
 {
  objects::Assets railed=assets;railed.rail_patterns=&rail_set;
  objects::Plan sheet;objects::select_routes(tile,railed,true,true,neighbors(3u,2u),sheet);
  assert(sheet.patterns.size()==2 && sheet.patterns[1].style==4);
  assert(sheet.patterns[0].line>=patterns.offsets[1] && sheet.patterns[0].line<patterns.offsets[2]);
  assert(sheet.patterns[1].line>=rail_set.offsets[2] && sheet.patterns[1].line<rail_set.offsets[3]);
  assert(railed.patterns_for(4)==&rail_set && railed.patterns_for(0)==&patterns && assets.patterns_for(4)==&patterns);
 }
 // A railroad over a tile-diagonal river edge gets the railroad bridge.
 tile.river_code=2;objects::Plan rail_bridge;objects::select_routes(tile,assets,true,true,neighbors(1u,1u),rail_bridge);
 assert(rail_bridge.patterns.size()==1 && rail_bridge.patterns[0].style==4 && rail_bridge.patterns[0].bridges==2u);
 assert(rail_bridge.instances.size()==1 && rail_bridge.instances[0].u==.5f && rail_bridge.instances[0].v==0.f);
 tile.river_code=0;
 tile.railroad_mask=0;
 // An isolated road keeps the mask-zero mark unless a city occupies the tile.
 objects::Plan isolated;objects::select_routes(tile,assets,true,true,neighbors(0u,0u),isolated);
 assert(isolated.patterns.size()==1 && isolated.patterns[0].line==0 && isolated.routes.empty());
 tile.city_id=4;objects::Plan city;objects::select_routes(tile,assets,true,true,neighbors(0u,0u),city);
 assert(city.patterns.empty() && city.routes.empty());
 objects::Plan city_link;objects::select_routes(tile,assets,true,true,neighbors(1u,0u),city_link);
 assert(city_link.patterns.size()==1);tile.city_id=-1;
 objects::Plan disabled;objects::select_routes(tile,assets,true,false,neighbors(3u,0u),disabled);
 assert(disabled.patterns.empty());
 // A river on the NE edge bridges that join and places one authored arch.
 tile.river_code=2;objects::Plan bridged;objects::select_routes(tile,assets,true,true,neighbors(1u,0u),bridged);
 assert(bridged.patterns.size()==1 && bridged.patterns[0].bridges==2u && bridged.instances.size()==1);
 // The authored bridge sits on the shared edge midpoint and follows the road.
 assert(bridged.instances[0].u==.5f && bridged.instances[0].v==0.f);
 assert(std::abs(bridged.instances[0].rotation+1.5707963f)<.01f);
 tile.river_code=0;
 objects::Projection projection;projection.tile=tile;projection.tile_width=128;projection.half_w=64;projection.half_h=32;
 projection.pickup_profile=projection.world_objects=true;projection.relief_projection_scale=128.f/224*.82f;
 auto flat=[](float,float){return std::array<float,3>{0,0,0};};auto zero=[](float,float){return 0.f;};
 auto sloped=[](float u,float v){return std::array<float,3>{8*u+3*v,0,0};};
 float tile_u=float(tile.tile_x+tile.tile_y)*.5f,tile_v=float(tile.tile_x-tile.tile_y)*.5f;
 objects::Plan all;objects::select_routes(tile,assets,true,true,neighbors(255u,0u),all);assert(all.patterns.size()==8);
 objects::Surfaces a,b;objects::compile(all,projection,assets,flat,zero,a);objects::compile(all,projection,assets,sloped,zero,b);
 auto const& va=a.layers[objects::route_layer];auto const& vb=b.layers[objects::route_layer];
 assert(!va.empty() && va.size()==vb.size() && va.size()%6==0);
 bool slope_lit=false;
 for(std::size_t i=0;i<va.size();++i){
  assert(va[i].surface_kind==11 && va[i].base_terrain==0 && va[i].world_valid==1);
  assert(va[i].u>=.0299f && va[i].u<=.9701f && va[i].macro_u==va[i].u);
  assert(va[i].v>.90606654f && va[i].v<.99021526f);
  assert(va[i].world_z<vb[i].world_z+1e-6f && va[i].y>=vb[i].y);
  // Pattern roads select their own shading; nothing here fords.
  assert(va[i].material_desert==0.f && va[i].material_marsh==1.f);
  slope_lit=slope_lit || vb[i].normal_x<-.01f;
 }
 assert(slope_lit);
 // A slope moves only the stroke's width, never its centerline: quads are
 // {left0,right0,right1,left0,right1,left1}.
 auto center=[](objects::Vertex const& a,objects::Vertex const& b){return std::array<float,2>{(a.world_x+b.world_x)*.5f,(a.world_y+b.world_y)*.5f};};
 for(std::size_t i=0;i<va.size();i+=6){
  auto fa=center(va[i],va[i+1]),fb=center(vb[i],vb[i+1]);
  assert(std::abs(fa[0]-fb[0])<1e-5f && std::abs(fa[1]-fb[1])<1e-5f);
 }
 // A steep slope across a path keeps the flat stroke's screen thickness
 // instead of stretching it down the face.
 {
  auto steep=[](float,float v){return std::array<float,3>{70*v,0,0};};
  objects::Surfaces f,s;objects::compile(all,projection,assets,flat,zero,f);objects::compile(all,projection,assets,steep,zero,s);
  auto const& vf=f.layers[objects::route_layer];auto const& vs=s.layers[objects::route_layer];
  assert(vf.size()==vs.size());
  auto thickness=[](std::vector<objects::Vertex> const& v,std::size_t i){
   float ax=(v[i].x+v[i+1].x)*.5f,ay=(v[i].y+v[i+1].y)*.5f,bx=(v[i+2].x+v[i+5].x)*.5f,by=(v[i+2].y+v[i+5].y)*.5f;
   float tx=bx-ax,ty=by-ay,nx=v[i+1].x-v[i].x,ny=v[i+1].y-v[i].y,length=std::hypot(tx,ty);
   return length<1.f?-1.f:std::abs(nx*ty-ny*tx)/length;
  };
  float worst=1.f;std::size_t measured=0;
  for(std::size_t i=0;i<vf.size();i+=6){
   float a=thickness(vf,i),b=thickness(vs,i);
   if(a<0 || b<0)continue;++measured;
   worst=std::max(worst,std::max(a/b,b/a));
  }
  assert(measured>20 && worst<1.25f);
 }
 for(auto const& v:vb)assert(std::abs(v.world_z*112-9.f-std::max(-2.5f,8*v.world_x+3*v.world_y))<.05f);
 // A railroad strip follows the same line a little wider, with its sleeper
 // coordinate unwrapped (0 at the shared join) rather than mirrored.
 {
  objects::Plan road_line,rail_line;road_line.patterns.push_back(all.patterns[1]);
  rail_line.patterns.push_back(all.patterns[1]);rail_line.patterns[0].style=4;
  objects::Surfaces ro,ra;objects::compile(road_line,projection,assets,flat,zero,ro);objects::compile(rail_line,projection,assets,flat,zero,ra);
  auto const& vr=ro.layers[objects::route_layer];auto const& va2=ra.layers[objects::route_layer];
  assert(!vr.empty() && !va2.empty()); // texture turns split the strips differently
  auto width=[](std::vector<objects::Vertex> const& v){return std::hypot(v[0].world_x-v[1].world_x,v[0].world_y-v[1].world_y);};
  // Only slightly wider than a road (stroke and dirt bed), so dense rail
  // networks do not outweigh roads.
  assert(width(va2)>1.1f*width(vr) && width(va2)<1.35f*width(vr));
  bool joined=false;
  for(auto const& v:va2){assert(v.base_terrain==4.f && v.material_marsh==1.f);joined=joined || std::abs(v.u)<1e-5f;}
  for(auto const& v:vr)assert(v.u>=.0299f);
  assert(joined);
 }
 // A per-point mountain fade dissolves the strip (the shader's fade weight).
 {
  objects::PatternRoute climbing=all.patterns[1];
  auto const& line=patterns.lines[climbing.line];
  climbing.fade.assign(line.count,0.f);climbing.fade.back()=1.f;
  objects::Plan one;one.patterns.push_back(climbing);objects::Surfaces out;
  objects::compile(one,projection,assets,flat,zero,out);auto const& v=out.layers[objects::route_layer];
  assert(v.front().material_desert==0.f && v.back().material_desert>.99f);
 }
 // Near a shared join a path eases onto the join's shared axis, so the two
 // halves meet tangent to each other instead of at a corner.
 {
  objects::PatternRoute bent{patterns.offsets[1],0u,0u,{},{},{},0u};   // mask 1: center to the NE join
  bent.joins[2]=.5f;bent.joins[3]=-.8660254f;                        // shared axis 30 degrees off the line
  objects::Plan one;one.patterns.push_back(bent);objects::Surfaces out;
  objects::compile(one,projection,assets,flat,zero,out);auto const& v=out.layers[objects::route_layer];
  // Centerline over the last quad: {left0,right0,right1,left0,right1,left1}.
  std::size_t i=v.size()-6;
  float ax=(v[i].world_x+v[i+1].world_x)*.5f,ay=(v[i].world_y+v[i+1].world_y)*.5f;
  float bx=(v[i+2].world_x+v[i+5].world_x)*.5f,by=(v[i+2].world_y+v[i+5].world_y)*.5f;
  float dx=bx-ax,dy=by-ay,length=std::hypot(dx,dy);
  float along=(dx*.5f+dy*.8660254f)/length;   // world basis: (u,-v) of the tile-local axis
  assert(along>std::cos(8.f*3.14159265f/180.f));
 }
 // The shader blends the track into its owner's ground material by height:
 // each vertex names that material (0 grass ... 4 mountain, 5 marsh).
 for(auto const& v:va)assert(v.surface_coordinate==2.f); // desert (square type 0)
 {
  objects::Projection rock=projection;rock.tile.real_terrain_type=6;rock.tile.terrain_type=2;
  objects::Projection field=projection;field.tile.real_terrain_type=7;field.tile.terrain_type=1;
  objects::Surfaces r,f;objects::compile(all,rock,assets,flat,zero,r);objects::compile(all,field,assets,flat,zero,f);
  for(auto const& v:r.layers[objects::route_layer])assert(v.surface_coordinate==4.f);
  for(auto const& v:f.layers[objects::route_layer])assert(v.surface_coordinate==1.f); // forest over plains
 }
 // Roads cross mountains on the rock surface, as Civ III draws them over it.
 auto peak=[](float,float){return 150.f;};
 objects::Surfaces over;objects::compile(all,projection,assets,flat,peak,over);
 for(auto const& v:over.layers[objects::route_layer]){
  assert(std::abs(v.world_z*112-9.f-147.5f)<.05f);
  // A joined strip crosses into its neighbor; it must stay inside the
  // shader's per-tile diamond test instead of pinching at the seam.
  assert(v.material_grass==.5f && v.material_plains==.5f);
 }
 // Each end of a joined line lands on the neighbor's shared point; the east
 // tile's west join is the same world position.
 for(auto const& route:all.patterns){
  objects::Plan one;one.patterns.push_back(route);objects::Surfaces out;objects::compile(one,projection,assets,flat,zero,out);
  auto const& v=out.layers[objects::route_layer];auto const& last=v[v.size()-2];auto const& last_other=v[v.size()-1];
  int d=patterns.lines[route.line].end;
  float x=(last.world_x+last_other.world_x)*.5f,y=(last.world_y+last_other.world_y)*.5f;
  // A joined end stops exactly on the shared point.
  assert(std::hypot(x-(tile_u+join_u[d]),y-(tile_v+1-join_v[d]))<1e-4f);
  // Screen stroke: a horizontal run is thinner in world units than a vertical one.
  float width=std::hypot(v[0].world_x-v[1].world_x,v[0].world_y-v[1].world_y);
  if(d==1)assert(width>.09f && width<.12f);   // east corner: runs across the screen
  if(d==3)assert(width>.065f && width<.085f); // south corner: runs down the screen
 }
 // Adjacent tiles share one axis at their join, so both halves end on the
 // same world vertices: no gap and no overlapping double coverage. Railroads
 // (a rail on every tile, so no roads) meet the same way.
 for(bool rails:{false,true}){
  // Interior rail tiles are fully connected, so neighbors pick different
  // variants; both sides of each join still agree on its axis.
  objects::Assets world_assets=assets;if(rails)world_assets.rail_patterns=&rail_set;
  auto const& world_set=rails?rail_set:patterns;
  std::map<std::pair<int,int>,Observation> world;
  for(int y=16;y<=24;++y)for(int x=34;x<=48;++x)if((x+y)%2==0){
   c3x_renderer_tile_v1 occurrence=tile;occurrence.tile_x=x;occurrence.tile_y=y;
   occurrence.road_mask=1;occurrence.railroad_mask=rails;occurrence.river_code=0;occurrence.city_id=-1;
   world[{x,y}]=Observation{occurrence};
  }
  auto map_lookup=[&](int x,int y)->Observation const*{auto f=world.find({x,y});return f==world.end()?nullptr:&f->second;};
  auto ends=[&](int x,int y,int join){
   objects::Plan plan;objects::select_routes(world[{x,y}].occurrence,world_assets,true,true,map_lookup,plan);
   objects::Projection p=projection;p.tile=world[{x,y}].occurrence;
   for(auto const& route:plan.patterns){
    auto const& line=world_set.lines[route.line];
    assert(route.style==(rails?4u:0u));
    if(line.start!=join && line.end!=join)continue;
    objects::Plan one;one.patterns.push_back(route);objects::Surfaces out;
    objects::compile(one,p,world_assets,flat,zero,out);auto const& v=out.layers[objects::route_layer];
    bool at_end=line.end==join;
    // First quad is {left0,right0,...}; last quad ends {...,right1,left1}.
    auto const& a=at_end?v[v.size()-2]:v[0];auto const& b=at_end?v[v.size()-1]:v[1];
    return std::array<float,4>{a.world_x,a.world_y,b.world_x,b.world_y};
   }
   assert(false);return std::array<float,4>{};
  };
  for(int d=0;d<8;++d){
   auto mine=ends(40,20,d),theirs=ends(40+offsets[d][0],20+offsets[d][1],(d+4)&7);
   bool same=std::abs(mine[0]-theirs[0])<1e-5f && std::abs(mine[1]-theirs[1])<1e-5f &&
             std::abs(mine[2]-theirs[2])<1e-5f && std::abs(mine[3]-theirs[3])<1e-5f;
   bool swapped=std::abs(mine[0]-theirs[2])<1e-5f && std::abs(mine[1]-theirs[3])<1e-5f &&
                std::abs(mine[2]-theirs[0])<1e-5f && std::abs(mine[3]-theirs[1])<1e-5f;
   assert(same || swapped);
  }
 }
 // Bridges stand only on tile-diagonal river edges. A river through a shared
 // corner that separates the linked tiles makes both halves ford (fade into
 // each bank); a river that only touches the corner moves the join to the
 // open bank instead.
 {
  std::map<std::pair<int,int>,Observation> world;
  for(int y=16;y<=24;++y)for(int x=34;x<=48;++x)if((x+y)%2==0){
   c3x_renderer_tile_v1 occurrence=tile;occurrence.tile_x=x;occurrence.tile_y=y;occurrence.city_id=-1;
   occurrence.road_mask=0;occurrence.railroad_mask=0;occurrence.river_code=0;world[{x,y}]=Observation{occurrence};
  }
  auto map_lookup=[&](int x,int y)->Observation const*{auto f=world.find({x,y});return f==world.end()?nullptr:&f->second;};
  auto& west=world[{40,20}].occurrence;auto& east=world[{42,20}].occurrence;
  west.road_mask=east.road_mask=1;
  auto plan_for=[&](c3x_renderer_tile_v1 const& t){objects::Plan p;objects::select_routes(t,assets,true,true,map_lookup,p);return p;};
  west.river_code=2|8; // north-south through the east corner of the west tile
  auto a=plan_for(west),b=plan_for(east);
  assert(a.instances.empty() && b.instances.empty());
  assert(a.patterns.size()==1 && a.patterns[0].bridges==0u && a.patterns[0].fords==2u);
  assert(b.patterns.size()==1 && b.patterns[0].bridges==0u && b.patterns[0].fords==2u);
  west.river_code=2;east.river_code=128; // both on the north side: the corner is only touched
  a=plan_for(west);b=plan_for(east);
  assert(a.instances.empty() && b.instances.empty() && a.patterns[0].bridges==0u && b.patterns[0].bridges==0u);
  assert(a.patterns[0].fords==0u && b.patterns[0].fords==0u);
  // Both tiles move their shared join the same way: toward the open south bank.
  auto join_open=[&](objects::Plan const& p,int join){
   for(auto const& r:p.patterns){auto const& l=patterns.lines[r.line];
    if(l.end==join)return std::array<float,2>{r.open[2],r.open[3]};
    if(l.start==join)return std::array<float,2>{r.open[0],r.open[1]};}
   return std::array<float,2>{};};
  auto ow=join_open(a,1),oe=join_open(b,5);
  assert(std::abs(ow[0]-.7071068f)<1e-4f && std::abs(ow[1]-.7071068f)<1e-4f && ow==oe);
  // Both diagonals of one corner cross the river: no bridge for either.
  west.river_code=2|8;east.river_code=0;
  auto& north=world[{41,19}].occurrence;auto& south=world[{41,21}].occurrence;
  north.road_mask=south.road_mask=1;north.river_code=8;
  auto n=plan_for(north);a=plan_for(west);
  auto at_corner=[](objects::Plan const& p,float u,float v){unsigned c=0;
   for(auto const& i:p.instances)c+=i.u==u && i.v==v;return c;};
  assert(at_corner(a,1.f,0.f)==0 && at_corner(n,1.f,1.f)==0);
  for(auto const& route:n.patterns){auto const& line=patterns.lines[route.line];
   if(line.start==3)assert(route.fords&1u);if(line.end==3)assert(route.fords&2u);}
 }
 // A long path mirrors its tiled piece inside the atlas interior: it turns at
 // both ends, never samples the outer columns, and is continuous between quads.
 objects::RoutePatterns long_line=patterns;
 for(auto& point:long_line.points)point[0]=point[0]*6.f-2.5f;
 objects::Plan wrap;wrap.patterns.push_back(all.patterns[1]);objects::Surfaces wrapped;
 objects::compile(wrap,projection,objects::Assets{assets.bundles,&long_line},flat,zero,wrapped);
 auto const& wv=wrapped.layers[objects::route_layer];
 bool low=false,high=false;
 for(auto const& v:wv){assert(v.u>=.0299f && v.u<=.9701f && v.macro_u==v.u);low|=v.u<.04f;high|=v.u>.96f;}
 for(std::size_t q=6;q<wv.size();q+=6)assert(std::abs(wv[q].u-wv[q-1].u)<1e-5f);
 assert(low && high);
 // Away from a bridge, points inside a river channel along a tile edge move
 // toward the tile's center onto dry bank; a shared join off any bend stays.
 auto edge_river=[&](float,float v){return std::abs(tile_v+1.f-v)*64.f;}; // along the NE edge
 objects::Plan wet;wet.patterns.push_back(all.patterns[1]);
 objects::promote_river_crossings(tile,edge_river,wet,&assets);
 auto const& moved=wet.patterns[0].points;auto const& source=patterns.lines[all.patterns[1].line];
 assert(moved.size()==source.count && wet.instances.empty());
 for(unsigned i=0;i+1<moved.size();++i)assert(edge_river(0,tile_v+1.f-moved[i][1])>=10.9f);
 assert(moved.back()==patterns.points[source.first+source.count-1]);
 // The join left in the water fades into the bank; the moved points do not.
 auto const& fade=wet.patterns[0].wet;
 assert(fade.size()==source.count && fade.back() && !fade.front());
 objects::Surfaces nudged;objects::compile(wet,projection,assets,flat,zero,nudged);
 assert(!nudged.layers[objects::route_layer].empty());
 // A bridged river join keeps its exact approach for the deck.
 objects::Plan kept;kept.patterns.push_back(bridged.patterns[0]);
 objects::promote_river_crossings(tile,[&](float,float v){return std::abs(tile_v+1.f-v)*64.f;},kept,&assets);
 assert(kept.patterns[0].points.empty());
 // A bridge rests on its lower bank: the authored deck ends sit at the mesh
 // base, so that end meets its bank and the far end settles into the higher
 // bank; no end floats. The seat is one rigid transform, identical for the
 // shared instance and its bounds.
 {
  FeatureAsset span{};span.id="route/bridge/medieval/normal";
  span.vertices={{{-.25f,0,.02f},{0,0,1},{0,0}},{{.25f,0,.02f},{0,0,1},{0,0}},{{0,0,.05f},{0,0,1},{0,0}}};
  span.indices={0,1,2};
  assert(objects::shared_rigid_mesh(span));
  auto own=bundles;own[objects::bridge_family].assets={span};
  objects::Assets with=assets;for(unsigned f=0;f<own.size();++f)with.bundles[f]=&own[f];
  objects::Plan plan;plan.instances.push_back({objects::bridge_family,0u,objects::feature_layer,.5f,0.f,-1.5707963f,1.f,13.f,0.f,false});
  for(bool high_neighbor:{true,false}){
   // One bank (world v past this tile is the NE neighbor) stands 40 higher.
   auto banks=[&](float,float v){return std::array<float,3>{(v>tile_v+1.f)==high_neighbor?40.f:0.f,0,0};};
   objects::Surfaces out;objects::compile(plan,projection,with,banks,zero,out);
   auto const& v=out.layers[objects::feature_layer];
   assert(v.size()==3 && v[0].world_y<tile_v+1.f && v[1].world_y>tile_v+1.f);
   // Both ends stand on the lower bank (world_z carries +2.5).
   for(int i:{0,1})assert(std::abs(v[i].world_z*112-(2.5f+.02f*150.f/.82f))<.05f);
   auto rigid=objects::prepare_rigid(plan.instances[0],projection,with,banks,zero);
   assert(std::abs(rigid.instance.place[7])<.01f);
  }
  // A railroad bridge (a flat truss deck at its base) rests the same way.
  span.id="route/bridge/railroad/normal";own[objects::bridge_family].assets={span};
  auto high_far=[&](float,float v){return std::array<float,3>{v>tile_v+1.f?40.f:0.f,0,0};};
  assert(std::abs(objects::prepare_rigid(plan.instances[0],projection,with,high_far,zero).instance.place[7])<.01f);
  // A bridge shorter than its river's carved channel (about .17 tile from the
  // water's center) rests on the banks beyond, not on the channel's slopes
  // well below the paths climbing onto it.
  {
   FeatureAsset short_span{};short_span.id="route/bridge/railroad/normal";
   short_span.vertices={{{-.156f,0,0},{0,0,1},{0,0}},{{.156f,0,0},{0,0,1},{0,0}},{{0,0,.05f},{0,0,1},{0,0}}};
   short_span.indices={0,1,2};
   own[objects::bridge_family].assets={short_span};
   auto channel=[&](float,float v){return std::array<float,3>{std::abs(v-(tile_v+1.f))<.17f?-20.f:0.f,0,0};};
   auto channel_height=[&](float u,float v){return channel(u,v)[0]+2.5f;};
   assert(std::abs(objects::prepare_rigid(plan.instances[0],projection,with,channel,channel_height).instance.place[7])<.01f);
   own[objects::bridge_family].assets={span};
  }
  // The bridged road ends under the bridge's end; the deck carries it over.
  objects::Plan road;road.patterns.push_back(bridged.patterns[0]);
  objects::Surfaces strip;objects::compile(road,projection,assets,flat,zero,strip);
  auto const& r=strip.layers[objects::route_layer];assert(!r.empty());
  float nearest=1e9f;
  for(auto const& vertex:r)nearest=std::min(nearest,std::hypot(vertex.world_x-(tile_u+.5f),vertex.world_y-(tile_v+1.f)));
  // Default half length .18, less the .04 overlap.
  assert(nearest>.13f && nearest<.16f);
  // A rendered river meanders off the tile edge: here its water centers 0.1
  // tile past the NE join. The bridge stands on that water and the road
  // half shortens to end under its moved end.
  span.id="route/bridge/medieval/normal";own[objects::bridge_family].assets={span};
  objects::Plan meander=bridged;meander.instances[0].asset=0;
  auto off_edge=[&](float,float v){return std::abs(v-(tile_v+1.1f))*64.f;};
  objects::promote_river_crossings(tile,off_edge,meander,&with);
  assert(std::abs(meander.instances[0].u-.5f)<1e-4f && std::abs(meander.instances[0].v+.1f)<.011f);
  assert(std::abs(meander.patterns[0].crossing[1]-.1f)<.011f);
  objects::Plan moved;moved.patterns=meander.patterns;
  objects::Surfaces shifted;objects::compile(moved,projection,with,flat,zero,shifted);
  // Centerline ends: quads are {left0,right0,right1,left0,right1,left1}.
  auto centerline_to_join=[&](std::vector<objects::Vertex> const& v){
   float best=1e9f;
   for(std::size_t i=0;i+5<v.size();i+=6)for(auto [a,b]:{std::pair<std::size_t,std::size_t>{i,i+1},{i+2,i+5}})
    best=std::min(best,std::hypot((v[a].world_x+v[b].world_x)*.5f-(tile_u+.5f),(v[a].world_y+v[b].world_y)*.5f-(tile_v+1.f)));
   return best;
  };
  nearest=centerline_to_join(shifted.layers[objects::route_layer]);
  assert(nearest>.03f && nearest<.05f);
  // A hill bend pushes the water a quarter tile off the edge: the bridge
  // follows it, and this tile's half runs to the join under the bridge's end.
  objects::Plan bent=bridged;bent.instances[0].asset=0;
  objects::promote_river_crossings(tile,[&](float,float v){return std::abs(v-(tile_v+1.24f))*64.f;},bent,&with);
  assert(std::abs(bent.instances[0].v+.24f)<.011f && std::abs(bent.patterns[0].crossing[1]-.24f)<.011f);
  // Water standing well inside this tile puts the bridge there: the road
  // keeps its piece between the join and the bridge (the neighbor's road
  // meets it at the join), skips the deck, and resumes beyond it.
  {
   objects::Plan inside=bridged;inside.instances[0].asset=0;
   objects::promote_river_crossings(tile,[&](float,float v){return std::abs(v-(tile_v+1.f-.24f))*64.f;},inside,&with);
   assert(std::abs(inside.patterns[0].crossing[1]+.24f)<.011f);
   objects::Plan one;one.patterns.push_back(inside.patterns[0]);
   objects::Surfaces out;objects::compile(one,projection,with,flat,zero,out);
   auto const& v=out.layers[objects::route_layer];
   bool on_deck=false;
   for(std::size_t i=0;i+5<v.size();i+=6)for(auto [a,b]:{std::pair<std::size_t,std::size_t>{i,i+1},{i+2,i+5}}){
    float d=std::hypot((v[a].world_x+v[b].world_x)*.5f-(tile_u+.5f),(v[a].world_y+v[b].world_y)*.5f-(tile_v+1.f));
    on_deck=on_deck || (d>.12f && d<.36f);
   }
   assert(centerline_to_join(v)<.005f && !on_deck);
  }
  // Pattern bridges stand at 70% of the source meshes' calibrated scale.
  {
   FeatureAsset long_span{};long_span.id="route/bridge/medieval/normal";
   long_span.vertices={{{-.25f,0,0},{0,0,1},{0,0}},{{.25f,0,0},{0,0,1},{0,0}},{{0,0,.05f},{0,0,1},{0,0}}};
   long_span.indices={0,1,2};
   auto own=bundles;own[objects::bridge_family].assets={long_span};
   objects::Assets scaled=assets;for(unsigned f=0;f<own.size();++f)scaled.bundles[f]=&own[f];
   tile.river_code=2;objects::Plan sized;objects::select_routes(tile,scaled,true,true,neighbors(1u,0u),sized);tile.river_code=0;
   assert(sized.instances.size()==1 && std::abs(sized.instances[0].scale-.7f)<1e-5f);
   assert(std::abs(sized.patterns[0].bridge_half-.175f)<1e-5f);
  }
  // The bridged approach runs straight along the bridge axis a short way
  // past the deck's end, wherever the river puts the bridge: a path bent
  // away from the axis still leaves the bridge straight, and the line's far
  // end (the tile's junction) stays where the pattern puts it.
  for(float c:{0.f,.12f,-.12f}){
   objects::PatternRoute bent_line{patterns.offsets[1],0u,2u,{},{},{},0u};
   bent_line.joins[2]=0.f;bent_line.joins[3]=-1.f;bent_line.crossing[1]=c;bent_line.bridge_half=.18f;
   auto const& line=patterns.lines[bent_line.line];
   bent_line.points.assign(patterns.points.begin()+line.first,patterns.points.begin()+line.first+line.count);
   for(auto& point:bent_line.points)point[0]+=.6f*(.5f-point[1])*(point[1]);   // bow the path sideways
   bent_line.points.back()={.5f,0.f};
   objects::Plan one;one.patterns.push_back(bent_line);objects::Surfaces out;
   objects::compile(one,projection,assets,flat,zero,out);auto const& v=out.layers[objects::route_layer];
   // Every centerline point short of the straight run's end sits on the axis (u=.5).
   float deck_end=std::max(0.f,.18f-c),run=deck_end+std::clamp(.40f-deck_end,.05f,.12f);
   float worst=0;
   for(std::size_t i=0;i+5<v.size();i+=6)for(auto [a,b]:{std::pair<std::size_t,std::size_t>{i,i+1},{i+2,i+5}}){
    float u=(v[a].world_x+v[b].world_x)*.5f-tile_u,wv=(v[a].world_y+v[b].world_y)*.5f-tile_v;
    float d=1.f-wv;   // local v: distance from the NE join along the axis
    if(d<run-.01f)worst=std::max(worst,std::abs(u-.5f));
   }
   assert(worst<.003f);
   // The far end (the tile centre, u=v=.5) is untouched.
   auto const& first=v[0];auto const& second=v[1];
   float cu=(first.world_x+second.world_x)*.5f-tile_u,cv=1.f-((first.world_y+second.world_y)*.5f-tile_v);
   assert(std::hypot(cu-.5f,cv-.5f)<.045f);   // the unjoined end overhangs ~.039 by design
  }
  // Civ III rail patterns often fork just inside the edge, and a few leave
  // a loose stub at the join short of the line it meets. Next to a bridge the
  // whole network leaves it straight: the fork moves to the straight run's
  // end, nothing is drawn on the deck, and the other tile edges stay put.
  {
   objects::RoutePatterns fork;
   auto add=[&](std::vector<std::array<float,2>> const& p,int start,int end){
    fork.lines.push_back({std::uint32_t(fork.points.size()),std::uint16_t(p.size()),std::int8_t(start),std::int8_t(end)});
    fork.points.insert(fork.points.end(),p.begin(),p.end());
   };
   add({{.5f,0.f},{.51f,.05f},{.53f,.10f}},0,-1);                              // into the bridged join
   add({{.53f,.10f},{.58f,.30f},{.55f,.60f},{.5f,1.f}},-1,4);                   // on to SW
   add({{.53f,.10f},{.70f,.12f},{.85f,.06f},{1.f,0.f}},-1,1);                   // on to the E corner
   add({{.5f,0.f},{.52f,.03f}},0,-1);                                          // a loose stub
   add({{1.f,0.f},{.70f,.09f},{.52f,.07f},{.45f,.30f},{.48f,.65f},{.5f,1.f}},1,4); // passing it
   objects::Assets forked=assets;forked.road_patterns=&fork;
   float const sw[]={0.f,1.f},east[]={.70710678f,-.70710678f};
   auto route=[&](unsigned line){
    objects::PatternRoute r{line,0u,0u,{},{},{},0u};
    auto const& l=fork.lines[line];
    if(l.start==0){r.bridges=1u;r.joins[0]=0.f;r.joins[1]=-1.f;r.bridge_half=.18f;}
    if(l.start==1){r.joins[0]=east[0];r.joins[1]=east[1];}
    if(l.end==1){r.joins[2]=east[0];r.joins[3]=east[1];}
    if(l.end==4){r.joins[2]=sw[0];r.joins[3]=sw[1];}
    return r;
   };
   for(auto lines:{std::vector<unsigned>{0,1,2},std::vector<unsigned>{3,4}}){
    objects::Plan plan;for(unsigned line:lines)plan.patterns.push_back(route(line));
    objects::Surfaces out;objects::compile(plan,projection,forked,flat,zero,out);
    auto const& v=out.layers[objects::route_layer];assert(!v.empty());
    // Just past the deck's end (.14 from the join) only the axis is drawn.
    float worst=0;bool on_deck=false,reach_sw=false,reach_east=false;
    for(std::size_t i=0;i+5<v.size();i+=6)for(auto [a,b]:{std::pair<std::size_t,std::size_t>{i,i+1},{i+2,i+5}}){
     float u=(v[a].world_x+v[b].world_x)*.5f-tile_u,d=1.f-((v[a].world_y+v[b].world_y)*.5f-tile_v);
     if(d>.14f && d<.23f && std::abs(u-.5f)<.10f)worst=std::max(worst,std::abs(u-.5f));
     on_deck=on_deck || (d>.01f && d<.13f && std::abs(u-.5f)<.04f);
     reach_sw=reach_sw || std::hypot(u-.5f,d-1.f)<.012f;
     reach_east=reach_east || std::hypot(u-1.f,d)<.012f;
    }
    assert(worst<.004f && !on_deck && reach_sw && reach_east);
   }
  }
  // The river runs in a valley: the land climbs 15 units from .2 to .6 tile
  // off the water. On the map's oblique view a path descending to a deck on
  // the valley floor would be drawn bent into the deck's side. The deck
  // stands at the top of the lower bank instead, and its path is carried
  // level to it, so on screen the path continues the deck's own line (its
  // rails 2.5 over the deck; route strips draw 6.5 over their height).
  {
   auto valley=[&](float,float v){float d=std::abs(tile_v+1.f-v);return 2.5f+15.f*std::clamp((d-.2f)/.4f,0.f,1.f);};
   auto valley_relief=[&](float u,float v){return std::array<float,3>{valley(u,v)-2.5f,0,0};};
   FeatureAsset truss{};truss.id="route/bridge/railroad/normal";
   truss.vertices={{{-.156f,0,0},{0,0,1},{0,0}},{{.156f,0,0},{0,0,1},{0,0}},{{0,0,.05f},{0,0,1},{0,0}}};
   truss.indices={0,1,2};
   auto own=bundles;own[objects::bridge_family].assets={truss};
   objects::Assets with=assets;for(unsigned f=0;f<own.size();++f)with.bundles[f]=&own[f];
   objects::Instance bridge{objects::bridge_family,0u,objects::feature_layer,.5f,0.f,-1.5707963f,1.f,13.f,0.f,false};
   float level=objects::prepare_rigid(bridge,projection,with,valley_relief,valley).instance.place[7];
   assert(std::abs(level-15.f)<.01f);
   objects::RoutePatterns through;
   through.lines.push_back({0u,5u,std::int8_t(0),std::int8_t(4)});
   through.points={{.5f,0.f},{.5f,.25f},{.5f,.5f},{.5f,.75f},{.5f,1.f}};
   objects::Assets straight_through=with;straight_through.road_patterns=&through;
   objects::PatternRoute line{0u,0u,1u,{},{},{},0u};
   line.joins={0.f,-1.f,0.f,1.f};line.bridge_half=.156f;
   objects::Plan one;one.patterns.push_back(line);objects::Surfaces out;
   objects::compile(one,projection,straight_through,valley_relief,valley,out);auto const& v=out.layers[objects::route_layer];assert(!v.empty());
   auto screen=[](float wu,float wv,float h){return std::array<float,2>{64*(wu+wv),32*(wu-wv+1)-h*(128.f/224*.82f)};};
   auto a=screen(tile_u+.5f,tile_v+1.f,level+2.5f),b=screen(tile_u+.5f,tile_v+1.f-.3f,level+2.5f);
   float dx=b[0]-a[0],dy=b[1]-a[1],length=std::hypot(dx,dy),worst=0;int measured=0;
   for(std::size_t i=0;i+5<v.size();i+=6)for(auto [p,q]:{std::pair<std::size_t,std::size_t>{i,i+1},{i+2,i+5}}){
    float wu=(v[p].world_x+v[q].world_x)*.5f,wv=(v[p].world_y+v[q].world_y)*.5f,wz=(v[p].world_z+v[q].world_z)*.5f;
    float along=1.f-(wv-tile_v);if(along>.45f)continue; // the floor and lower wall, carried level
    auto s=screen(wu,wv,wz*112-2.5f);
    worst=std::max(worst,std::abs((s[0]-a[0])*dy-(s[1]-a[1])*dx)/length);++measured;
   }
   assert(measured>3 && worst<.35f);
  }
  // On the edge itself nothing moves.
  objects::Plan centered=bridged;centered.instances[0].asset=0;
  objects::promote_river_crossings(tile,[&](float,float v){return std::abs(v-(tile_v+1.f))*64.f;},centered,&with);
  assert(std::abs(centered.instances[0].v)<1e-4f && std::abs(centered.patterns[0].crossing[1])<1e-4f);
 }
 // Without a pattern pack the existing segment roads remain unchanged.
 objects::Assets legacy=assets;legacy.road_patterns=nullptr;
 objects::Plan old;objects::select_routes(tile,legacy,true,true,neighbors(3u,0u),old);
 assert(old.patterns.empty() && !old.routes.empty());
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    def test_prepared_objects_private_queries_owned_capture_and_packed_parity(self):
        run_cpp(r'''
#define NOMINMAX
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
  assert(expected && expected->city.size()==2 && expected->topology.size()==8 && !expected->world.empty());
  assert(expected->routes==1 && !expected->layers[objects::route_layer].mesh.empty());
  float route_x=0;std::memcpy(&route_x,expected->layers[objects::route_layer].mesh.vertices.data(),sizeof(route_x));
  assert(route_x>5.f); // world projection keeps source-space XY; no kind-2 normalization
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


ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def route_draws_are_decals(renderer_source, pipeline_source):
    """Both map pipelines draw the route layer with the no-depth-write decal state."""
    import re
    retained = re.search(r"OMSetDepthStencilState\(natural\.decal_depth,0\);\s*bool routes_drawn=draw\(geometry_route\);",
                         renderer_source)
    # Other decal-like layers (rivers) may share the route branch.
    fresh = re.search(r"else if \(\(?[^)]*\blayer==geometry_route\b[^)]*\)? && renderer\.natural\.decal_depth\) \{[^}]*"
                      r"OMSetDepthStencilState\(renderer\.natural\.decal_depth,0\);", pipeline_source)
    return bool(retained and fresh)


class RouteRegressionContractTests(unittest.TestCase):
    def test_pattern_routes_fade_on_mountains(self):
        # Pattern roads once draped over mountain peaks as wide streaks, then
        # vanished from mountain tiles entirely, then were capped at a foothill
        # height so the rock cut them along a ragged contour. Civ III shows
        # them on a mountain's foot but never over its top: routes follow the
        # whole rendered mountain and fade out as it rises.
        import re
        source = (ROOT / "Renderer/native/object_preparation.h").read_text()
        fade = re.search(r"pattern_route_fade_start=([0-9.]+)f,pattern_route_fade_end=([0-9.]+)f", source)
        self.assertIsNotNone(fade)
        self.assertTrue(20.0 <= float(fade.group(1)) < float(fade.group(2)) <= 90.0)
        self.assertNotIn("pattern_route_mountain_cap", source)
        self.assertRegex(source, r"float rise=route_height\(x,y\)-height_natural\(x,y\);")
        self.assertRegex(source, r"if\(input\.projection\.tile\.road_mask \|\| input\.projection\.tile\.railroad_mask\)\n")

    def test_route_strips_never_write_depth(self):
        # Overlapping translucent road fringes once wrote depth and rejected
        # another strip's core, leaving light cracks at joins and junctions.
        self.assertTrue(route_draws_are_decals(
            (ROOT / "Renderer/native/c3x_renderer.cpp").read_text(),
            (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()))

    def test_route_mountain_shape_matches_rendered_mesh(self):
        # Routes and objects sampled a stale, taller/narrower copy of the
        # mountain rule, floating or hiding paths on the rock. The mesh, route
        # surfaces and resource seating now build one shared shape.
        for path in ("Renderer/lab/shared/natural/relief_mesh_body.h", "Renderer/native/object_preparation.h",
                     "Renderer/native/c3x_renderer.cpp"):
            source = (ROOT / path).read_text()
            self.assertRegex(source, r"MountainShape\b[^;(]*\(natural,", path)
            self.assertNotIn("struct MountainPiece", source, path)
            self.assertNotIn("natural.macro[variant]", source, path)
