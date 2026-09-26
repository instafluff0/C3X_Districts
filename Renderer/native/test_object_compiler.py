"""CPU object descriptions, terrain dependencies and exact source attributes."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ObjectCompilerTests(unittest.TestCase):
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
 assert(vertices[0].world_z>7.f/112.f);
 assert(vertices[1].world_z==vertices[0].world_z);
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
 assert(objects::select_improvements(tile,assets,0,0,false,true,false,false,plan));
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
  assert(objects::select_improvements(tile,assets,0,0,false,true,false,false,plan));
  objects::settle_farm_props(plan,tile,assets,dry);
  unsigned trees=0,buildings=0;
  for(auto const& instance:plan.instances){
   if(instance.asset==0)++trees;else ++buildings;
  }
  assert(trees>=4 && trees<=6 && buildings==1);
 }
 tile.variant_seed=7;plan.instances.clear();
 assert(objects::select_improvements(tile,assets,0,0,false,true,false,false,plan));
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
   assert(objects::select_improvements(tile,assets,ground,0,false,true,false,false,plan));
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
''')

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
 assert(routes.routes.size()==8 && routes.instances.size()==1 && queried.size()==8);
 for(auto const&r:routes.routes)assert(r.railroad && r.style==4);
 objects::Plan diagnostic;objects::select_routes(tile,assets,true,false,lookup,diagnostic);
 assert(diagnostic.routes.empty() && diagnostic.instances.size()==1);
 tile.railroad_mask=0;neighbor.occurrence.railroad_mask=0;
 objects::Plan dense_roads;objects::select_routes(tile,assets,true,true,lookup,dense_roads);
 assert(dense_roads.routes.size()>=4 && dense_roads.routes.size()<=8);
 auto missing=[](int,int)->Observation const*{return nullptr;};
 objects::Plan isolated;objects::select_routes(tile,assets,true,true,missing,isolated);
 assert(isolated.routes.size()==1 && !isolated.routes[0].railroad);
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
 assert(deck_top>.14f && deck_vertices[0].world_z<.05f);
 objects::Route straight{.5f,.5f,1.f,.5f,0,false,false,false};
 std::vector<objects::Vertex> junction_vertices;
 objects::append_route(projection,straight,flat,[](float,float){return 0.f;},junction_vertices);
 float tile_u=float(tile.tile_x+tile.tile_y)*.5f;
 assert(std::abs(junction_vertices.front().world_x-(tile_u+.5f))<.02f);
 assert(std::abs(junction_vertices.back().world_x-(tile_u+1.f))<.03f);
 objects::Plan wet;wet.routes.push_back({.5f,.5f,1.f,.5f,3,false,false,false});
 objects::promote_river_crossings(tile,assets,[&](float u,float){return std::abs(u-(tile_u+.75f))*100.f;},wet);
 assert(wet.routes[0].bridge && std::abs(wet.routes[0].bridge_t-.5f)<.001f);
 assert(wet.instances.size()==1 && std::abs(wet.instances[0].u-.75f)<.001f);
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
