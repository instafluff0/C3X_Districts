"""Farm kit: one green patchwork on every terrain that drapes on the rendered
ground, stays inside its tile and keeps its routes and resource open."""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]

# A kit bundle: a flat gridded patchwork on the planted texture (unit square,
# like the baked one), a tree, a farmhouse and the "farm_kit" marker group.
KIT = r'''
#include "Renderer/native/object_compiler.h"
#include "Renderer/lab/shared/natural/patterns.h"
#include <cassert>
#include <cmath>
// The runtime's bundle helpers, without its Windows file loader.
namespace c3x_renderer {
FeatureGroup const* find_feature_group(FeatureBundle const& bundle,char const* name){
 for(auto const& group:bundle.groups)if(group.name==name)return &group;return nullptr;}
std::uint32_t stable_hash(std::uint32_t value){return patterns::feature_hash(value);}
float stable_random(std::uint32_t value){return patterns::stable_random(value);}
}
using namespace c3x_renderer;
FeatureAsset flat(char const* id,unsigned texture){
 constexpr unsigned n=36;
 FeatureAsset asset;asset.id=id;asset.texture_index=texture;
 for(unsigned y=0;y<=n;++y)for(unsigned x=0;x<=n;++x)
  asset.vertices.push_back({{float(x)/n-.5f,float(y)/n-.5f,.002f},{0,0,1},{float(x)/n,float(y)/n}});
 for(unsigned y=0;y<n;++y)for(unsigned x=0;x<n;++x){
  unsigned a=y*(n+1)+x,b=a+1,c=a+n+1,d=c+1;asset.indices.insert(asset.indices.end(),{a,b,d,a,d,c});}
 return asset;
}
FeatureAsset raised(char const* id,unsigned texture){
 FeatureAsset asset;asset.id=id;asset.texture_index=texture;
 asset.vertices={{{-.02f,-.02f,0},{0,0,1},{0,0}},{{.02f,-.02f,0},{0,0,1},{1,0}},{{0,0,.05f},{0,0,1},{.5f,1}}};
 asset.indices={0,1,2};return asset;
}
struct Kit {
 FeatureBundle farm,other;objects::Assets assets;
 explicit Kit(bool kit=true):assets{{&other,&other,&other,&farm,&other,&other}}{
  FeatureGroup group;group.name="farm_0";
  farm.assets.push_back(flat("farm_0:crop:patchwork:e0",0));
  group.placements.push_back({});
  for(auto id:{"farm_0:tree:source:e0","farm_0:building:source:e0"}){
   farm.assets.push_back(raised(id,4));
   FeaturePlacement placement{};placement.asset_index=unsigned(farm.assets.size()-1);group.placements.push_back(placement);
  }
  farm.groups.push_back(group);
  if(kit){FeatureGroup marker;marker.name="farm_kit";marker.placements.push_back({});farm.groups.push_back(marker);}
 }
};
objects::Projection projection(){
 objects::Projection p;p.tile_width=128;p.content_view_height=640;
 p.half_w=64;p.half_h=32;p.relief_projection_scale=128.f/224.f*.82f;
 p.feature_projection_scale=128.f/224.f;p.pickup_profile=p.world_objects=true;
 p.tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;return p;
}
// Tile (0,0): world (x,y) is tile-local (u, 1-v).
float local_v(float world_y){return 1.f-world_y;}
// Fields use the planted texture (slot 0); props use slots 4 and 5.
// Clipping interpolates the combined open-ground distance linearly across a
// patchwork triangle, so a cut edge may reach this far into a verge.
constexpr float cut=.015f;
bool field(objects::Vertex const& vertex){return vertex.base_terrain<21.5f;}
'''


class FarmKitTests(unittest.TestCase):
    def test_kit_covers_every_terrain_with_one_clipped_green_patchwork(self):
        run_cpp(KIT + r'''
int main(){
 Kit kit;auto p=projection();
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 2.5f;};
 float turns[2]={9,-9};
 for(int ground=0;ground<5;++ground)for(unsigned seed=0;seed<64;++seed){
  c3x_renderer_tile_v1 tile=p.tile;tile.variant_seed=seed;objects::Plan plan;
  assert(objects::select_improvements(tile,kit.assets,ground,0,false,true,plan));
  assert(plan.farm_kit);
  unsigned patchworks=0;
  for(auto const& instance:plan.instances)if(instance.asset==0){
   ++patchworks;turns[0]=std::min(turns[0],instance.rotation);turns[1]=std::max(turns[1],instance.rotation);}
  assert(patchworks==1);
  objects::settle_farm_fields(plan,tile,kit.assets,dry);
  objects::Surfaces out;objects::compile(plan,p,kit.assets,dry,height,out);
  // Inside its tile, behind a narrow verge, and reaching all four corners.
  float corner[4]={9,9,9,9};
  for(auto const& vertex:out.layers[objects::farm_layer]){
   if(!field(vertex))continue;
   float u=vertex.world_x,v=local_v(vertex.world_y);
   assert(u>=.02f-cut && u<=.98f+cut && v>=.02f-cut && v<=.98f+cut);
   for(unsigned c=0;c<4;++c)corner[c]=std::min(corner[c],std::hypot(u-float(c&1),v-float(c>>1)));
  }
  // Interpolating the edge distance chamfers each corner slightly.
  for(float distance:corner)assert(distance<.1f);
 }
 // Rows turn to any angle, not quarter turns.
 assert(turns[0]<.5f && turns[1]>5.8f);
}
''')

    def test_kit_fields_drape_on_rendered_ground(self):
        run_cpp(KIT + r'''
int main(){
 Kit kit;auto p=projection();c3x_renderer_tile_v1 tile=p.tile;tile.variant_seed=3;
 objects::Plan plan;assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
 // Low relief: the rendered ground rolls 3 to 9 units above the coarse relief.
 auto relief=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float x,float y){return 2.5f+6.f+3.f*std::sin(x*6.f)*std::sin(y*6.f);};
 objects::Surfaces out;objects::compile(plan,p,kit.assets,relief,height,out);
 unsigned fields=0,props=0;
 for(auto const& vertex:out.layers[objects::farm_layer]){
  float ground=height(vertex.world_x,vertex.world_y)-2.5f;
  // Fields lie on the ground; props stand on it, within their small footprint's rise.
  if(field(vertex)){++fields;assert(vertex.world_z*112.f-2.5f>=ground);}
  else {++props;assert(vertex.world_z*112.f-2.5f>=ground-.5f);}
 }
 assert(fields>0 && props>0);
}
''')

    def test_kit_props_stand_on_rendered_ground_as_shared_rigid_instances(self):
        run_cpp(KIT.replace("object_compiler.h", "rigid_object_instance.h") + r'''
int main(){
 Kit kit;auto p=projection();
 auto relief=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 2.5f+7.f;};
 objects::Instance tree{objects::farm_family,1,objects::farm_layer,.3f,.3f,0,1.5f,21,.01f,true};
 assert(kit.farm.assets[1].id.find(":tree:")!=std::string::npos);
 // Kit props seat on the rendered ground; without the kit they keep the relief.
 assert(std::abs(objects::prepare_rigid(tree,p,kit.assets,relief,height,true).instance.place[7]-7.f)<1e-4f);
 assert(std::abs(objects::prepare_rigid(tree,p,kit.assets,relief,height).instance.place[7])<1e-4f);
}
''')

    def test_kit_fields_and_props_leave_routes_open(self):
        run_cpp(KIT + r'''
int main(){
 Kit kit;auto p=projection();
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 2.5f;};
 // An edge road along the gap (v=.5), a corner road across two quadrants
 // (the u=v diagonal) and a railroad from the centre to the u=1 edge.
 objects::Route edge{0,.5f,1,.5f,0,false,false,false};
 objects::Route corner{0,0,1,1,0,false,false,false};
 objects::Route rail{.5f,.5f,1,.5f,4,true,false,false};
 for(int layout=0;layout<3;++layout)for(unsigned seed=0;seed<48;++seed){
  c3x_renderer_tile_v1 tile=p.tile;tile.variant_seed=seed;
  objects::Plan plan,routes;routes.routes.push_back(layout==0?edge:layout==1?corner:rail);
  assert(objects::select_improvements(tile,kit.assets,seed%5,0,false,true,plan));
  objects::clear_farm(plan,routes,kit.assets,{});
  assert(!plan.farm_clearing.paths.empty());
  objects::settle_farm_fields(plan,tile,kit.assets,dry);
  objects::settle_farm_props(plan,tile,kit.assets,dry);
  auto const& clearing=plan.farm_clearing;
  unsigned buildings=0;
  for(auto const& instance:plan.instances){
   auto const& id=kit.farm.assets[instance.asset].id;
   if(id.find(":crop:")!=std::string::npos)continue;
   buildings+=id.find(":building:")!=std::string::npos;
   assert(clearing.at(instance.u,instance.v)>=.10f);
  }
  assert(buildings==1);
  objects::Surfaces out;objects::compile(plan,p,kit.assets,dry,height,out);
  unsigned kept=0;
  for(auto const& vertex:out.layers[objects::farm_layer]){
   if(!field(vertex))continue;
   ++kept;assert(clearing.at(vertex.world_x,local_v(vertex.world_y))>=-cut);
  }
  assert(kept>=200u); // the patchwork stays on both sides
 }
}
''')

    def test_kit_keeps_its_resource_ground_open(self):
        run_cpp(KIT + r'''
int main(){
 Kit kit;auto p=projection();
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 2.5f;};
 std::vector<std::array<float,4>> resource{{.40f,.38f,.60f,.62f}};
 for(unsigned seed=0;seed<48;++seed){
  c3x_renderer_tile_v1 tile=p.tile;tile.variant_seed=seed;
  objects::Plan plan,routes;
  assert(objects::select_improvements(tile,kit.assets,seed%5,0,false,true,plan));
  objects::clear_farm(plan,routes,kit.assets,resource);
  objects::settle_farm_fields(plan,tile,kit.assets,dry);
  objects::settle_farm_props(plan,tile,kit.assets,dry);
  objects::Surfaces out;objects::compile(plan,p,kit.assets,dry,height,out);
  unsigned kept=0;
  for(auto const& vertex:out.layers[objects::farm_layer]){
   if(!field(vertex))continue;
   float u=vertex.world_x,v=local_v(vertex.world_y);
   // Round yard corners cut a little deeper into the .05 margin.
   ++kept;assert(!(u>.40f-.01f && u<.60f+.01f && v>.38f-.01f && v<.62f+.01f));
  }
  assert(kept>0);
  for(auto const& instance:plan.instances)
   if(kit.farm.assets[instance.asset].id.find(":crop:")==std::string::npos)
    assert(!(instance.u>.40f-.1f && instance.u<.60f+.1f && instance.v>.38f-.1f && instance.v<.62f+.1f));
 }
}
''')

    def test_resource_kits_swap_fields_and_crop_kits_keep_no_yard(self):
        run_cpp(KIT + r'''
#include <cstring>
int main(){
 Kit kit;auto p=projection();
 // Ripe fields: a second patchwork on texture 1, for wheat (yard kept) and
 // sugar (the farm is the crop: no yard).
 kit.farm.assets.push_back(flat("farm_kit:crop:ripe:e0",1));
 FeaturePlacement ripe{};ripe.asset_index=unsigned(kit.farm.assets.size()-1);
 FeatureGroup wheat;wheat.name="farm_kit:wheat";wheat.placements.push_back(ripe);kit.farm.groups.push_back(wheat);
 FeatureGroup sugar;sugar.name="farm_kit:sugar:crop";sugar.placements.push_back(ripe);kit.farm.groups.push_back(sugar);
 std::vector<std::array<float,4>> resource{{.4f,.4f,.6f,.6f}};
 for(char const* name:{"Wheat","Sugar","Cattle"}){
  c3x_renderer_tile_v1 tile=p.tile;tile.variant_seed=5;tile.city_id=-1;tile.resource_id=3;
  std::strcpy(tile.resource_name,name);
  objects::Plan plan,routes;
  assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  unsigned texture=9;
  for(auto const& instance:plan.instances)
   if(kit.farm.assets[instance.asset].id.find(":crop:")!=std::string::npos)texture=kit.farm.assets[instance.asset].texture_index;
  objects::clear_farm(plan,routes,kit.assets,resource);
  bool own=std::strcmp(name,"Cattle")!=0,crop=std::strcmp(name,"Sugar")==0;
  assert(texture==(own?1u:0u));
  assert(plan.farm_clearing.boxes.size()==(crop?0u:1u));
 }
}
''')

    def test_kit_patchwork_keeps_its_authored_size(self):
        run_cpp(KIT + r'''
int main(){
 for(float side:{1.f,2.3f,3.3f}){
  Kit kit;for(auto& vertex:kit.farm.assets[0].vertices){vertex.position[0]*=side;vertex.position[1]*=side;}
  for(unsigned seed=0;seed<32;++seed){
   c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;tile.variant_seed=seed;
   objects::Plan plan;assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
   for(auto const& instance:plan.instances)if(instance.asset==0){
    // At least its authored side (never under 1.5 tiles, to cover the
    // shifted tile), and at most a fifth larger.
    float size=instance.scale*side;
    assert(size>=std::max(1.5f,side)-1e-4f && size<=std::max(1.5f,side*1.2f)+1e-4f);
   }
  }
 }
}
''')

    def test_farm_without_kit_keeps_its_previous_layout(self):
        run_cpp(KIT + r'''
int main(){
 Kit kit(false);c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 objects::Plan plan,routes;routes.routes.push_back({0,.5f,1,.5f,0,false,false,false});
 assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
 assert(!plan.farm_kit);
 objects::clear_farm(plan,routes,kit.assets,{{.4f,.4f,.6f,.6f}});
 assert(plan.farm_clearing.paths.empty() && plan.farm_clearing.boxes.empty());
}
''')

    def test_kit_pieces_sort_with_natural_ground_at_any_elevation(self):
        # Object meshes sort on a feature basis that weights ground height
        # about a third as much as terrain does, so raised low relief and
        # hills hid farm fields and props. Kit pieces (fraction .0135-.0335)
        # take the natural basis, in both the packed and the rigid shader.
        native = ROOT / 'Renderer/native/render_core'
        world = (native / 'world_projection.hlsl').read_text()
        projection = between(world, 'float3 project_world_content(', '\n// Resource bodies')
        natural = world[world.index('bool farm_kit_material('):]
        geometry = (native / 'rigid_instance_geometry.hlsl').read_text()
        point = between(geometry, 'struct RigidPoint', '\n// b1') if '\n// b1' in geometry else \
            geometry[geometry.index('struct RigidPoint'):]
        shader = (native / 'rigid_feature.hlsl').read_text()
        depth = between(shader, ' float3 position=project_world_content(p.position,p.world,i.projection,2);',
                        ' return o;')
        depth = '\n'.join(line for line in depth.splitlines() if 'o.position.xy=' not in line)
        run_cpp(r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#define precise
using std::floor;using std::sqrt;using std::abs;
struct float2 {float x,y;};
struct float3 {float x,y,z;float3(float a=0,float b=0,float c=0):x(a),y(b),z(c){}
 float3(float2 v,float c):x(v.x),y(v.y),z(c){}float2 xy()const{return {x,y};}};
struct float4 {float x,y,z,w;};
float2 operator*(float2 a,float s){return {a.x*s,a.y*s};}
float3 operator*(float3 a,float s){return {a.x*s,a.y*s,a.z*s};}
float3 operator/(float3 a,float s){return {a.x/s,a.y/s,a.z/s};}
template<class T>T clamp(T v,T lo,T hi){return v<lo?lo:v>hi?hi:v;}
inline double max(double a,double b){return a>b?a:b;}
inline float frac(float v){return v-std::floor(v);}
struct RigidInput {float3 source_position,source_normal;float4 place0,place1,projection,placement_view;};
float2 c3x_viewport_reserved{1192,0};
''' + re.sub(r'\.xy\b', '.xy()', projection) + '\n' + natural + '\n' + point + r'''
struct Output {float4 position;};
float rigid_depth(RigidInput i){
 RigidPoint p=rigid_point(i);Output o{};
''' + depth + r'''
 return o.position.z;
}
RigidInput part(float material,float ground){
 RigidInput i{};i.source_position=float3(.05f,-.03f,.04f);i.source_normal=float3(0,0,1);
 i.place0={20,30,.5f,.5f};i.place1={1,0,1,ground};i.projection={20,30,128,1192};
 i.placement_view={0,0,0,material};return i;
}
// Packed field decal depth at ground height g (feature height .37 units),
// placed as append_instance does: ground raises the 128-pixel source y.
float decal_depth(float material,float g,float4 projection){
 float3 world(20.5f,30.5f,(g+2.5f+.37f)/112);
 float3 position(.25f,(16.f-g*(128.f/224*.82f))/128.f,.37f);
 float3 projected=project_world_content(position,world,projection,2);
 return resource_natural_depth(projected,world.z,projection,2,material);
}
float natural(float g,float4 projection){
 return project_world_content(float3(),float3(20.5f,30.5f,(g+2.5f)/112),projection,1).z;
}
int main(){
 float4 projection={20,30,128,1192};
 float terrain=natural(40,projection)-natural(0,projection);
 // Kit fields (21.0135) and props (25/26.0135, with emissive codes 2 and 3).
 for(float material:{21.0135f,25.0135f,26.0235f,26.0335f}){
  assert(farm_kit_material(material));
  float field=decal_depth(material,40,projection)-decal_depth(material,0,projection);
  assert(std::fabs(field-terrain)<1e-3f);
  float prop=rigid_depth(part(material,0))-rigid_depth(part(material,40));
  float ground=std::fabs(rigid_depth(part(18,0))-rigid_depth(part(18,40)));
  assert(std::fabs(prop-ground)<2e-6f);
 }
 // Mines, older farms, resources, tile objects and units keep their basis.
 for(float material:{21.01f,22.02f,23.03f,21.f,21.1f,21.21f,21.4035f,30.0135f})assert(!farm_kit_material(material));
 float mine=rigid_depth(part(21.01f,0))-rigid_depth(part(21.01f,40));
 float ground=rigid_depth(part(18,0))-rigid_depth(part(18,40));
 assert(mine>0 && mine<ground*.5f);
 float old=decal_depth(21.01f,40,projection)-decal_depth(21.01f,0,projection);
 assert(old<terrain*.5f);
 std::puts("PASS farm kit natural depth");
}
''')

    def test_irrigated_tile_objects_depend_on_their_resource(self):
        # Object meshes are keyed by these inputs: a farm that keeps its
        # resource open must rebuild when the resource appears or changes.
        run_cpp(r'''
#include "Renderer/native/render_core/captured_scene.h"
#include <cassert>
#include <cstring>
using c3x_renderer::render_core::CapturedScene;
int main(){
 c3x_renderer_tile_v1 tile{};tile.tile_x=4;tile.tile_y=6;tile.city_id=-1;
 tile.resource_id=21;tile.resource_class=0;std::strcpy(tile.resource_name,"wheat");
 auto plain=CapturedScene::object_inputs(tile);
 assert(plain.resource_id==0 && plain.resource_name[0]==0);
 tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 auto farm=CapturedScene::object_inputs(tile);
 assert(farm.resource_id==21 && !std::strcmp(farm.resource_name,"wheat"));
}
''')


if __name__ == "__main__":
    unittest.main()
