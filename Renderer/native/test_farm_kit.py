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
  // Inside its tile (feathering into its edges) and reaching all four corners.
  float corner[4]={9,9,9,9};
  for(auto const& vertex:out.layers[objects::farm_layer]){
   if(!field(vertex))continue;
   float u=vertex.world_x,v=local_v(vertex.world_y);
   assert(u>=-cut && u<=1+cut && v>=-cut && v<=1+cut);
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

    def test_kit_drops_fields_cut_to_slivers(self):
        run_cpp(KIT + r'''
FeatureAsset square(float half){
 FeatureAsset asset;asset.id="farm_kit:crop:field0:e0";
 for(unsigned y=0;y<=8;++y)for(unsigned x=0;x<=8;++x)
  asset.vertices.push_back({{(float(x)/8-.5f)*2*half,(float(y)/8-.5f)*2*half,.002f},{0,0,1},{float(x)/8,float(y)/8}});
 for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x){
  unsigned a=y*9+x,b=a+1,c=a+9,d=c+1;asset.indices.insert(asset.indices.end(),{a,b,d,a,d,c});}
 return asset;
}
int main(){
 FeatureBundle bundle;bundle.assets.push_back(square(.1f));FeaturePlacement placement{};
 auto p=projection();
 // The tile edge at u=.98: a field centred at .5 stays whole, one at .93
 // keeps three quarters, one at 1.05 keeps a .03 strip and goes.
 auto relief=[](float x,float){return std::array<float,3>{0,0,.98f-x};};
 auto height=[](float,float){return 2.5f;};
 for(float centre:{.5f,.93f,1.05f}){
  std::vector<objects::Vertex> vertices,shadows;std::vector<unsigned> indices;
  objects::append_instance(p,bundle,placement,centre,.5f,0,1,21,.0135f,false,false,
                           relief,height,vertices,shadows,&indices,0.f,0.f,true);
  assert((centre<1)==!vertices.empty());
  assert(vertices.size()==indices.size());
  // Without the kit the strip stays, as before.
  vertices.clear();indices.clear();
  objects::append_instance(p,bundle,placement,centre,.5f,0,1,21,.01f,false,false,
                           relief,height,vertices,shadows,&indices);
  assert(!vertices.empty());
 }
}
''')

    def test_fields_split_by_routes_stay_and_dense_junctions_narrow_their_verges(self):
        # In a late-game map almost every farm has roads to all its neighbours
        # (and railroad loops): wide verges plus sliver dropping left only
        # scraps of farmland. Route cuts leave fields; verges narrow when dense.
        run_cpp(KIT + r'''
FeatureAsset square(float half){
 FeatureAsset asset;asset.id="farm_kit:crop:field0:e0";
 for(unsigned y=0;y<=8;++y)for(unsigned x=0;x<=8;++x)
  asset.vertices.push_back({{(float(x)/8-.5f)*2*half,(float(y)/8-.5f)*2*half,.002f},{0,0,1},{float(x)/8,float(y)/8}});
 for(unsigned y=0;y<8;++y)for(unsigned x=0;x<8;++x){
  unsigned a=y*9+x,b=a+1,c=a+9,d=c+1;asset.indices.insert(asset.indices.end(),{a,b,d,a,d,c});}
 return asset;
}
int main(){
 FeatureBundle bundle;bundle.assets.push_back(square(.15f));FeaturePlacement placement{};
 auto p=projection();auto height=[](float,float){return 2.5f;};
 // A road verge through the middle of a .3 field leaves two thin halves.
 auto road=[](float x,float){return std::array<float,3>{0,0,std::abs(x-.5f)-.12f};};
 std::vector<objects::Vertex> vertices,shadows;std::vector<unsigned> indices;
 objects::append_instance(p,bundle,placement,.5f,.5f,0,1,21,.0135f,false,false,
                          road,height,vertices,shadows,&indices,0.f,0.f,true);
 bool left=false,right=false;
 for(auto const& vertex:vertices){left|=vertex.world_x<.38f;right|=vertex.world_x>.62f;}
 assert(left && right);
 // Verges: a lone road is wide, a dense junction about its stroke.
 Kit kit;c3x_renderer_tile_v1 tile=p.tile;tile.variant_seed=1;
 objects::Plan lone,dense,one,many;
 assert(objects::select_improvements(tile,kit.assets,2,0,false,true,lone));
 assert(objects::select_improvements(tile,kit.assets,2,0,false,true,dense));
 one.routes.push_back({0,.5f,1,.5f,0,false,false,false});
 for(unsigned k=0;k<12;++k){float a=float(k)*.5236f;
  many.routes.push_back({.5f,.5f,.5f+.5f*std::cos(a),.5f+.5f*std::sin(a),k%3?0u:4u,k%3==0,false,false});}
 objects::clear_farm(lone,one,kit.assets,{});objects::clear_farm(dense,many,kit.assets,{});
 assert(std::abs(lone.farm_clearing.paths[0][4]-.095f)<1e-5f);
 for(auto const& path:dense.farm_clearing.paths)assert(path[4]<=.0601f);
}
''')

    def test_dense_route_networks_take_the_gap_free_patchwork(self):
        run_cpp(KIT + r'''
#include <cstring>
int main(){
 Kit kit;
 auto add=[&](char const* group,char const* id,unsigned texture){
  kit.farm.assets.push_back(flat(id,texture));
  FeatureGroup g;g.name=group;FeaturePlacement placement{};placement.asset_index=unsigned(kit.farm.assets.size()-1);
  g.placements.push_back(placement);kit.farm.groups.push_back(g);};
 add("farm_kit:dense","farm_kit:crop:dense0:e0",2);
 add("farm_kit:wheat","farm_kit:crop:ripe0:e0",1);
 add("farm_kit:wheat:dense","farm_kit:crop:ripedense0:e0",3);
 auto texture=[&](char const* resource,unsigned lines){
  c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;tile.variant_seed=4;
  tile.city_id=-1;tile.resource_id=resource?3:-1;if(resource)std::strcpy(tile.resource_name,resource);
  objects::Plan plan;plan.farm_route_lines=lines;
  assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  unsigned found=9;
  for(auto const& instance:plan.instances)
   if(kit.farm.assets[instance.asset].id.find(":crop:")!=std::string::npos)found=kit.farm.assets[instance.asset].texture_index;
  return found;};
 // Up to two route lines keep the patchwork with ground between fields;
 // three or more (a junction, a railroad) fill the ground between routes.
 assert(texture(nullptr,0)==0 && texture(nullptr,2)==0 && texture(nullptr,3)==2 && texture(nullptr,12)==2);
 assert(texture("Wheat",1)==1 && texture("Wheat",5)==3 && texture("Cattle",5)==2);
 // Route lines held by the plan itself count too.
 c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 objects::Plan plan;for(unsigned k=0;k<3;++k)plan.routes.push_back({0,.5f,1,.5f,0,false,false,false});
 assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
 bool dense=false;
 for(auto const& instance:plan.instances)dense|=kit.farm.assets[instance.asset].texture_index==2;
 assert(dense);
}
''')

    def test_kit_pieces_share_one_placement(self):
        run_cpp(KIT + r'''
int main(){
 Kit kit;
 // A second field piece of the same patchwork, in its upper half.
 FeatureAsset upper=kit.farm.assets[0];upper.id="farm_kit:crop:field1:e0";
 for(auto& vertex:upper.vertices)vertex.position[1]=vertex.position[1]*.5f+.25f;
 kit.farm.assets.push_back(upper);
 FeaturePlacement second{};second.asset_index=unsigned(kit.farm.assets.size()-1);
 kit.farm.groups[0].placements.push_back(second);
 for(unsigned seed=0;seed<32;++seed){
  c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;tile.variant_seed=seed;
  objects::Plan plan;assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  std::vector<objects::Instance> pieces;
  for(auto const& instance:plan.instances)
   if(kit.farm.assets[instance.asset].id.find(":crop:")!=std::string::npos)pieces.push_back(instance);
  assert(!pieces.empty() && pieces.size()<=2);
  for(auto const& piece:pieces)assert(piece.u==pieces[0].u && piece.v==pieces[0].v &&
   piece.rotation==pieces[0].rotation && piece.scale==pieces[0].scale);
 }
}
''')

    def test_routes_paint_over_farms_in_both_pipelines(self):
        # Route strips write no depth, so a farm drawn after them painted over
        # the roads it overlapped. Farms now draw before the routes.
        fresh = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        scene = between(fresh, 'bool draw_scene(', 'bool ensure_glow(')
        self.assertLess(scene.index('draw(geometry_farm)'), scene.index('draw(geometry_route)'))
        self.assertIn('(layer!=geometry_farm || mirrored) && !draw(layer)', scene)
        lab = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        self.assertEqual(lab.count('draw(geometry_farm)'), 1)
        self.assertLess(lab.index('draw(geometry_farm)'), lab.index('bool routes_drawn=draw(geometry_route)'))

    def test_kit_fields_feather_at_their_cuts(self):
        # Fields carry their distance to a cut in the material's spare digits;
        # the shader turns it, with the source fringe, into a narrow soft edge.
        run_cpp(KIT + r'''
int main(){
 FeatureBundle bundle;bundle.assets.push_back(flat("farm_kit:crop:field0:e0",0));
 for(auto& vertex:bundle.assets[0].vertices){vertex.position[0]*=.3f;vertex.position[1]*=.3f;}
 FeaturePlacement placement{};auto p=projection();auto height=[](float,float){return 2.5f;};
 auto road=[](float x,float){return std::array<float,3>{0,0,std::abs(x-.5f)-.05f};};
 std::vector<objects::Vertex> vertices,shadows;std::vector<unsigned> indices;
 objects::append_instance(p,bundle,placement,.5f,.5f,0,1,21,.0135f,false,false,
                          road,height,vertices,shadows,&indices,0.f,0.f,true);
 bool cut=false,inside=false;
 for(auto const& vertex:vertices){
  float digits=(vertex.base_terrain-21.f)*100.f-1.f,distance=std::abs(vertex.world_x-.5f)-.05f;
  assert(digits>.309f && digits<.341f);
  if(distance<1e-4f){cut=true;assert(digits<.3105f);}
  if(distance>.06f){inside=true;assert(digits>.3395f);}
 }
 assert(cut && inside);
}
''')
        shader = (ROOT / 'Renderer/native/render_core/terrain_scene.hlsl').read_text()
        weights = between(shader, '    float material_fraction = frac(input.material_index);', '    float mine_slot')
        alpha = between(shader, '    float farm_kit_alpha =', '    ground_alpha = lerp(ground_alpha, farm_kit_alpha')
        run_cpp(r'''
#include <cassert>
#include <cmath>
struct Input {float material_index;};
static bool clipped;
float step(float edge,float x){return x>=edge?1.f:0.f;}
float lerp(float a,float b,float t){return a+(b-a)*t;}
float saturate(float x){return x<0?0:x>1?1:x;}
float frac(float x){return x-std::floor(x);}
float smoothstep(float a,float b,float x){float t=saturate((x-a)/(b-a));return t*t*(3-2*t);}
void clip(float x){clipped=x<0;}
struct Sample {float a;};
float alpha(float material,float source,float& field){
 Input input{material};Sample mine_sample{source};clipped=false;
''' + weights + alpha + r'''
 field=farm_kit_field_weight;return clipped?-1.f:farm_kit_alpha;}
int main(){
 float field=0;
 assert(alpha(21.0134f,1,field)>.99f && field==1);       // inside a field
 assert(alpha(21.0131f,1,field)<0 && field==1);          // at a cut: discarded
 assert(alpha(21.01325f,1,field)>.99f);                  // .03 tile in: opaque
 float half=alpha(21.013175f,1,field);assert(half>.3f && half<.7f); // .015 in
 assert(alpha(21.0134f,.2f,field)<.4f);                  // its soft source fringe
 alpha(25.0135f,1,field);assert(field==0);               // kit props stay opaque
 alpha(21.01f,1,field);assert(field==0);                 // mines and old farms too
 alpha(26.0235f,1,field);assert(field==0);               // emissive kit props
}
''')

    def test_kit_fields_cast_no_shadow(self):
        # Lifted, soft-edged field decals cast a thin dark rim beside every
        # field; like resource ground decals they are not shadow casters.
        caster = (ROOT / 'Renderer/native/render_core/source_caster.hlsl').read_text()
        cutout = between(caster, ' float part=frac(i.material);', ' if(abs(i.material-.48)')
        run_cpp(r'''
#include <cassert>
#include <cmath>
struct Input {float material;};
float frac(float x){return x-std::floor(x);}
bool casts(float material){Input i{material};
#define discard return false
''' + cutout + r'''
#undef discard
 return true;}
int main(){
 assert(!casts(21.0131f) && !casts(21.0134f) && !casts(24.01325f)); // kit fields
 assert(casts(25.0135f) && casts(26.0235f));                        // kit trees, farmhouses
 assert(casts(21.f) && casts(21.01f) && casts(22.02f));              // resources, mines
 assert(!casts(21.355f));                                            // resource ground decals
}
''')

    def test_plots_follow_the_road_and_continue_across_tiles(self):
        run_cpp(KIT + r'''
#include <map>
int main(){
 Kit kit;
 // Two plot shapes: a square field and a 2:1 band.
 auto plot=[&](char const* id,float width){FeatureAsset asset=flat(id,0);
  for(auto& vertex:asset.vertices)vertex.position[0]*=width;
  kit.farm.assets.push_back(asset);return unsigned(kit.farm.assets.size()-1);};
 FeatureGroup plots;plots.name="farm_kit:plots";
 for(auto index:{plot("farm_kit:crop:plot0:e0",1),plot("farm_kit:crop:plot1:e0",2)}){
  FeaturePlacement placement{};placement.asset_index=index;plots.placements.push_back(placement);}
 kit.farm.groups.push_back(plots);
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto lay=[&](int x,int y,objects::Route route){
  c3x_renderer_tile_v1 tile{};tile.tile_x=x;tile.tile_y=y;tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
  objects::Plan plan,routes;routes.routes.push_back(route);
  assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  assert(plan.farm_plots);
  objects::clear_farm(plan,routes,kit.assets,{});
  objects::settle_farm_fields(plan,tile,kit.assets,dry);
  std::vector<objects::Instance> fields;
  for(auto const& instance:plan.instances)
   if(kit.farm.assets[instance.asset].id.find(":crop:")!=std::string::npos){
    assert(kit.farm.assets[instance.asset].id.find(":plot")!=std::string::npos); // no patchwork
    fields.push_back(instance);}
  assert(fields.size()>=4);
  return fields;};
 auto quarter=[](float angle){float t=std::fmod(std::fmod(angle,1.5707963f)+1.5707963f,1.5707963f);
  return std::min(t,1.5707963f-t);};
 // A road along the tile's u axis (an edge road) and a corner (diagonal) road.
 for(auto const& field:lay(10,10,{0,.5f,1,.5f,0,false,false,false}))assert(quarter(field.rotation)<.02f);
 for(auto const& field:lay(10,10,{0,0,1,1,0,false,false,false}))assert(std::abs(quarter(field.rotation)-.7854f)<.02f);
 // The same road continuing into the edge neighbour (u+1): its strips run
 // on at the same distances from the road, and each tile's plots end at its
 // edge instead of being cut there.
 std::map<long,int> strips;
 for(int x:{10,11})for(auto const& field:lay(x,x,{0,.5f,1,.5f,0,false,false,false})){
  strips[std::lround(field.v*1e4f)]|=x==10?1:2;
  auto const& asset=kit.farm.assets[field.asset];
  float width=asset.vertices.back().position[0]-asset.vertices.front().position[0];
  float half=(std::abs(std::cos(field.rotation))*field.stretch*width+std::abs(std::sin(field.rotation)))*field.scale*.5f;
  assert(field.u-half>-1e-3f && field.u+half<1.001f);
 }
 for(auto const& strip:strips)assert(strip.second==3);
}
''')

    def test_plots_fill_each_area_between_routes_and_never_straddle_one(self):
        # One direction for a whole tile let plots cross its routes (clipped
        # to pieces on both sides) and left dense junctions patchy. Each area
        # the routes leave open now lays its own plots along one of its
        # routes, keeps them inside itself and fills it.
        run_cpp(KIT + r'''
int main(){
 Kit kit;
 auto plot=[&](char const* id,float width){FeatureAsset asset=flat(id,0);
  for(auto& vertex:asset.vertices)vertex.position[0]*=width;
  kit.farm.assets.push_back(asset);return unsigned(kit.farm.assets.size()-1);};
 FeatureGroup plots;plots.name="farm_kit:plots";
 for(auto index:{plot("farm_kit:crop:plot0:e0",1),plot("farm_kit:crop:plot1:e0",2)}){
  FeaturePlacement placement{};placement.asset_index=index;plots.placements.push_back(placement);}
 kit.farm.groups.push_back(plots);
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 0.f;};
 constexpr float pi=3.14159265f;
 // A junction linking all eight neighbours splits the tile into eight wedges.
 c3x_renderer_tile_v1 tile{};tile.tile_x=10;tile.tile_y=10;tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 objects::Plan plan,routes;
 for(unsigned k=0;k<8;++k){float a=float(k)*pi/4,reach=k&1u?.5f*std::sqrt(2.f):.5f;
  routes.routes.push_back({.5f,.5f,.5f+reach*std::cos(a),.5f+reach*std::sin(a),0,false,false,false});}
 assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
 objects::clear_farm(plan,routes,kit.assets,{});
 objects::settle_farm_fields(plan,tile,kit.assets,dry);
 auto p=projection();p.tile=tile;
 auto wedge=[&](float u,float v){return unsigned(int(std::floor(std::atan2(v-.5f,u-.5f)/(pi/4)+8))%8);};
 std::array<float,8> covered{},open{};
 for(auto const& instance:plan.instances){
  if(kit.farm.assets[instance.asset].id.find(":plot")==std::string::npos)continue;
  // Along one of its wedge's roads (0 or 45 degrees, modulo a turn).
  float turn=std::fmod(std::fmod(instance.rotation,pi/4)+pi/4,pi/4);
  assert(std::min(turn,pi/4-turn)<.03f);
  objects::Plan one=plan;one.instances={instance};
  objects::Surfaces surfaces;objects::compile(one,p,kit.assets,dry,height,surfaces);
  auto const& vertices=surfaces.layers[objects::farm_layer];
  int first=-1;
  for(std::size_t k=0;k+2<vertices.size();k+=3){
   float u[3],v[3];
   for(unsigned i=0;i<3;++i){u[i]=vertices[k+i].world_x-10.f;v[i]=1.f-vertices[k+i].world_y;}
   unsigned here=wedge((u[0]+u[1]+u[2])/3,(v[0]+v[1]+v[2])/3);
   assert(first<0 || unsigned(first)==here);first=int(here);
   covered[here]+=std::abs((u[1]-u[0])*(v[2]-v[0])-(v[1]-v[0])*(u[2]-u[0]))*.5f;
  }
 }
 for(int j=0;j<200;++j)for(int i=0;i<200;++i){float u=(float(i)+.5f)/200,v=(float(j)+.5f)/200;
  if(plan.farm_clearing.at(u,v)>0)open[wedge(u,v)]+=1.f/40000;}
 for(unsigned k=0;k<8;++k)assert(covered[k]>.8f*open[k]);
}
''')

    def test_neighbouring_farms_join_the_plots_of_the_area_they_share(self):
        # Each tile laid out its own part of an area its routes enclose, so
        # one area showed as separate pieces (often at different angles)
        # meeting at feathered tile edges. Farms in a small area now lay out
        # the same plots and cut them exactly, unfeathered, at the edge they
        # share; each draws its own part.
        run_cpp(KIT + r'''
#include <map>
int main(){
 Kit kit;
 auto plot=[&](char const* id,float width){FeatureAsset asset=flat(id,0);
  for(auto& vertex:asset.vertices)vertex.position[0]*=width;
  kit.farm.assets.push_back(asset);return unsigned(kit.farm.assets.size()-1);};
 FeatureGroup plots;plots.name="farm_kit:plots";
 for(auto index:{plot("farm_kit:crop:plot0:e0",1),plot("farm_kit:crop:plot1:e0",2)}){
  FeaturePlacement placement{};placement.asset_index=index;plots.placements.push_back(placement);}
 kit.farm.groups.push_back(plots);
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 0.f;};
 // Farms with roads on world tiles (c,r), c and r in 0..3: every two
 // neighbours link (Civ III), so the areas are the triangles between them.
 std::map<std::pair<int,int>,c3x_renderer_tile_v1> world;
 for(int c=0;c<4;++c)for(int r=0;r<4;++r){c3x_renderer_tile_v1 tile{};tile.tile_x=c+r;tile.tile_y=c-r;
  tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;tile.city_id=-1;tile.road_mask=1;world[{c,r}]=tile;}
 struct Piece {objects::Instance instance;std::vector<objects::Vertex> vertices;};
 auto lay=[&](int c,int r){
  auto const& tile=world[{c,r}];
  auto neighbour=[&](int dx,int dy)->c3x_renderer_tile_v1 const*{
   int x=tile.tile_x+dx,y=tile.tile_y+dy;auto found=world.find({(x+y)/2,(x-y)/2});
   return (x+y)%2==0 && found!=world.end()?&found->second:nullptr;};
  objects::Plan plan,routes;
  int const links[8][2]={{1,0},{-1,0},{0,-1},{0,1},{1,-1},{1,1},{-1,-1},{-1,1}};
  for(auto const& link:links)if(world.count({c+link[0],r+link[1]}))
   routes.routes.push_back({.5f,.5f,.5f+.5f*float(link[0]),.5f-.5f*float(link[1]),0,false,false,false});
  assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  objects::clear_farm(plan,routes,kit.assets,{});
  objects::settle_farm_fields(plan,tile,kit.assets,dry,neighbour);
  auto p=projection();p.tile=tile;
  std::vector<Piece> pieces;
  for(auto const& instance:plan.instances){
   if(kit.farm.assets[instance.asset].id.find(":plot")==std::string::npos || !(instance.region>>8))continue;
   objects::Plan one=plan;one.instances={instance};
   objects::Surfaces surfaces;objects::compile(one,p,kit.assets,dry,height,surfaces);
   pieces.push_back({instance,surfaces.layers[objects::farm_layer]});
  }
  return pieces;};
 // Tiles (1,1) and (2,1) share the edge at world x=2 (their u=1 and u=0).
 auto left=lay(1,1),right=lay(2,1);
 assert(!left.empty() && !right.empty());
 unsigned joined=0;
 for(auto const& a:left)for(auto const& b:right){
  auto const& x=a.instance;auto const& y=b.instance;
  if(std::abs(x.u-1.f-y.u)>1e-4f || std::abs(x.v-y.v)>1e-4f)continue;
  // The same plot, placed by both farms.
  assert(x.asset==y.asset && std::abs(x.rotation-y.rotation)<1e-5f && std::abs(x.scale-y.scale)<1e-5f &&
   std::abs(x.stretch-y.stretch)<1e-5f);
  // Unfeathered where it meets the shared edge away from routes: the full
  // opacity code (.0134), not the cut's (.0131).
  bool reaches_left=false,reaches_right=false;
  for(auto const& vertex:a.vertices){assert(vertex.world_x<=2.0001f);
   reaches_left=reaches_left || (vertex.world_x>1.9999f && vertex.base_terrain-std::floor(vertex.base_terrain)>.01335f);}
  for(auto const& vertex:b.vertices){assert(vertex.world_x>=1.9999f);
   reaches_right=reaches_right || (vertex.world_x<2.0001f && vertex.base_terrain-std::floor(vertex.base_terrain)>.01335f);}
  joined+=reaches_left && reaches_right;
 }
 assert(joined>=1);
}
''')

    def test_plots_stay_as_light_as_the_patchwork(self):
        # Plots gridded 12x12 (288 triangles each, cells under .04 tile) and
        # 1024 ground queries per farm for its water made the 1498 save scroll
        # and jump visibly slower than the patchwork. Plots now use about the
        # patchwork's .078-tile cells, and the water is sampled on a lattice.
        import sys
        sys.path.insert(0, str(ROOT))
        from Renderer.tools.asset_compiler.build_farm_runtime import PLOTS, plot
        triangles = []
        for box in PLOTS:
            mesh = plot(box)
            ys = sorted({vertex["position"][1] for vertex in mesh["vertices"]})
            # Cell height on a plot's longest layout (.5 tile) stays near .1.
            self.assertLessEqual((ys[1] - ys[0]) * .5, .1)
            triangles.append(len(mesh["topology"]["indices"]) // 3)
        self.assertLessEqual(sum(triangles) / len(triangles), 100)
        run_cpp(KIT + r'''
int main(){
 Kit kit;
 FeatureAsset asset=flat("farm_kit:crop:plot0:e0",0);kit.farm.assets.push_back(asset);
 FeatureGroup plots;plots.name="farm_kit:plots";
 FeaturePlacement placement{};placement.asset_index=unsigned(kit.farm.assets.size()-1);plots.placements.push_back(placement);
 kit.farm.groups.push_back(plots);
 unsigned queries=0;
 auto dry=[&](float,float){++queries;return std::array<float,3>{0,0,1};};
 c3x_renderer_tile_v1 tile{};tile.tile_x=10;tile.tile_y=10;tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
 objects::Plan plan,routes;routes.routes.push_back({0,.5f,1,.5f,0,false,false,false});
 assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
 objects::clear_farm(plan,routes,kit.assets,{});
 objects::settle_farm_fields(plan,tile,kit.assets,dry);
 assert(plan.instances.size()>1 && queries<=100);
}
''')

    def test_farm_ground_spans_verges_joins_farms_and_eases_into_other_land(self):
        # The grass ground under a farm (farm_kit:ground) runs under its route
        # verges, continues unfeathered into a neighbouring farm, and eases
        # into other land over a wide, irregular edge rather than the fields'
        # narrow one.
        run_cpp(KIT + r'''
#include <map>
int main(){
 Kit kit;
 auto add=[&](char const* id,char const* name){kit.farm.assets.push_back(flat(id,0));FeatureGroup group;group.name=name;
  FeaturePlacement placement{};placement.asset_index=unsigned(kit.farm.assets.size()-1);
  group.placements.push_back(placement);kit.farm.groups.push_back(group);};
 add("farm_kit:crop:plot0:e0","farm_kit:plots");add("farm_kit:crop:ground:e0","farm_kit:ground");
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 0.f;};
 // Tile (1,1) farms; its u=1 neighbour (2,1) farms too, its u=0 one (0,1) not.
 std::map<std::pair<int,int>,c3x_renderer_tile_v1> world;
 for(int c=0;c<3;++c){c3x_renderer_tile_v1 t{};t.tile_x=c+1;t.tile_y=c-1;t.city_id=-1;t.terrain_type=2; // grassland
  t.improvement_flags=c?C3X_RENDERER_IMPROVEMENT_IRRIGATION:0;world[{c,1}]=t;}
 auto const& tile=world[{1,1}];
 auto neighbour=[&](int dx,int dy)->c3x_renderer_tile_v1 const*{
  int x=tile.tile_x+dx,y=tile.tile_y+dy;auto found=world.find({(x+y)/2,(x-y)/2});
  return (x+y)%2==0 && found!=world.end()?&found->second:nullptr;};
 objects::Plan plan,routes;routes.routes.push_back({0,.5f,1,.5f,0,false,false,false});
 assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
 objects::clear_farm(plan,routes,kit.assets,{});
 objects::settle_farm_fields(plan,tile,kit.assets,dry,neighbour);
 objects::Instance const* ground=nullptr;bool plot_first=false;
 for(auto const& instance:plan.instances){auto const& id=kit.farm.assets[instance.asset].id;
  if(id.find(":ground")!=std::string::npos)ground=&instance;
  else if(id.find(":plot")!=std::string::npos && !ground)plot_first=true;}
 assert(ground && !plot_first); // drawn before (under) the plots
 objects::Plan one=plan;one.instances={*ground};
 auto p=projection();p.tile=tile;
 objects::Surfaces surfaces;objects::compile(one,p,kit.assets,dry,height,surfaces);
 auto fade=[](objects::Vertex const& vertex){return (vertex.base_terrain-std::floor(vertex.base_terrain)-.0131f)/.0003f;};
 bool on_verge=false,joined=false;
 for(auto const& vertex:surfaces.layers[objects::farm_layer]){
  // Tile (1,1) spans world x 1-2 (u) and y 1-2 (1-v).
  float u=vertex.world_x-1.f,v=2.f-vertex.world_y;
  on_verge=on_verge || std::abs(v-.5f)<.03f;                       // under the road's verge
  // At the shared edge as strong as inside (grassland: its code is capped
  // near .36-.5 by the ground's strength, never feathered toward 0 there).
  if(u>.999f && std::abs(v-.5f)<.2f)joined=joined || fade(vertex)>.3f;
  // Still easing in .1 tile from open land (the fields' own edge is opaque
  // from .03 tile in).
  if(u<.1f)assert(fade(vertex)<.51f);
 }
 assert(on_verge && joined);
}
''')

    def test_farm_ground_strength_follows_terrain_and_meets_halfway(self):
        # One green ground on every farm hid grassland, plains, desert and
        # tundra. The ground's strength now follows the tile's terrain (desert
        # weakest, grassland full) and two farms of different terrains meet
        # halfway at their shared edge, so there is no step between them.
        run_cpp(KIT + r'''
#include <map>
int main(){
 Kit kit;
 auto add=[&](char const* id,char const* name){kit.farm.assets.push_back(flat(id,0));FeatureGroup group;group.name=name;
  FeaturePlacement placement{};placement.asset_index=unsigned(kit.farm.assets.size()-1);
  group.placements.push_back(placement);kit.farm.groups.push_back(group);};
 add("farm_kit:crop:plot0:e0","farm_kit:plots");add("farm_kit:crop:ground:e0","farm_kit:ground");
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 0.f;};
 // A desert farm (1,1) beside a grassland farm (2,1), between open land.
 std::map<std::pair<int,int>,c3x_renderer_tile_v1> world;
 for(int c=0;c<4;++c){c3x_renderer_tile_v1 t{};t.tile_x=c+1;t.tile_y=c-1;t.city_id=-1;t.terrain_type=c==1?0:2;
  t.improvement_flags=c==1 || c==2?C3X_RENDERER_IMPROVEMENT_IRRIGATION:0;world[{c,1}]=t;}
 auto ground=[&](int c){
  auto const& tile=world[{c,1}];
  auto neighbour=[&](int dx,int dy)->c3x_renderer_tile_v1 const*{
   int x=tile.tile_x+dx,y=tile.tile_y+dy;auto found=world.find({(x+y)/2,(x-y)/2});
   return (x+y)%2==0 && found!=world.end()?&found->second:nullptr;};
  objects::Plan plan,routes;
  assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  objects::clear_farm(plan,routes,kit.assets,{});
  objects::settle_farm_fields(plan,tile,kit.assets,dry,neighbour);
  objects::Plan one=plan;one.instances.clear();
  for(auto const& instance:plan.instances)if(kit.farm.assets[instance.asset].id.find(":ground")!=std::string::npos)one.instances.push_back(instance);
  auto p=projection();p.tile=tile;
  objects::Surfaces surfaces;objects::compile(one,p,kit.assets,dry,height,surfaces);
  return surfaces.layers[objects::farm_layer];};
 auto fade=[](objects::Vertex const& vertex){return (vertex.base_terrain-std::floor(vertex.base_terrain)-.0131f)/.0003f;};
 // Mean code in the middle of each tile, and along their shared edge (x=2).
 auto mean=[&](std::vector<objects::Vertex> const& vertices,float x0,float x1){
  float sum=0;int count=0;
  for(auto const& vertex:vertices)if(vertex.world_x>=x0 && vertex.world_x<=x1 && vertex.world_y>1.3f && vertex.world_y<1.7f){
   sum+=fade(vertex);++count;}
  assert(count>0);return sum/float(count);};
 auto desert=ground(1),grassland=ground(2);
 float desert_middle=mean(desert,1.35f,1.65f),grassland_middle=mean(grassland,2.35f,2.65f);
 assert(desert_middle<grassland_middle-.05f);
 float desert_edge=mean(desert,1.99f,2.f),grassland_edge=mean(grassland,2.f,2.01f);
 assert(std::abs(desert_edge-grassland_edge)<.03f);
 // Its tint too (the normal's length, 1.1 + t: desert 1, grassland .25), met
 // halfway; world_valid stays 1 for shadows.
 auto tint=[&](std::vector<objects::Vertex> const& vertices,float x0,float x1){
  float sum=0;int count=0;
  for(auto const& vertex:vertices)if(vertex.world_x>=x0 && vertex.world_x<=x1 && vertex.world_y>1.3f && vertex.world_y<1.7f){
   assert(vertex.world_valid==1.f);
   sum+=std::sqrt(vertex.normal_x*vertex.normal_x+vertex.normal_y*vertex.normal_y+vertex.normal_z*vertex.normal_z)-1.1f;++count;}
  assert(count>0);return sum/float(count);};
 assert(std::abs(tint(desert,1.35f,1.5f)-1.f)<.02f && std::abs(tint(grassland,2.5f,2.65f)-.25f)<.02f);
 assert(std::abs(tint(desert,1.99f,2.f)-.625f)<.03f && std::abs(tint(grassland,2.f,2.01f)-.625f)<.03f);
}
''')

    def test_farm_terrain_tint_reaches_the_feature_shader(self):
        # The tint was first carried in world_valid, which the compact feature
        # vertex (48 bytes) drops, so the shader never saw it. It now rides in
        # the normal's length, which that vertex keeps; the shader decodes it
        # from there and normalizes the normal before any lighting.
        run_cpp(r'''
#include "Renderer/native/render_core/prepared_mesh.h"
#include <cassert>
#include <cmath>
int main(){
 using namespace c3x_renderer::render_core;
 std::vector<Vertex> source(3);
 for(unsigned i=0;i<3;++i){auto& v=source[i];v.x=float(i);v.y=float(i*i);v.base_terrain=24.0134f;
  v.normal_z=1.1f+.65f;v.world_x=1.f+float(i);v.world_y=2.f;v.world_z=.03f;v.world_valid=1.f;}
 PreparedMesh mesh;MeshFormat format;format.pickup=true;format.feature=true;
 assert(prepare_mesh(source,nullptr,format,mesh,[]{return false;}));
 assert(mesh.vertex_stride==48u);
 for(unsigned i=0;i<3;++i){float fields[12];std::memcpy(fields,mesh.vertices.data()+i*48u,48u);
  // x y z u v normal(3) material world(3)
  assert(std::abs(std::sqrt(fields[5]*fields[5]+fields[6]*fields[6]+fields[7]*fields[7])-1.75f)<1e-6f);}
}
''')
        shader = (ROOT / 'Renderer/native/render_core/terrain_scene.hlsl').read_text()
        decode = shader[shader.index('float farm_kit_normal_length'):]
        decode = decode[:decode.index('// Resource models cut out')]
        self.assertIn('length(input.geometry_normal)', decode)
        self.assertNotIn('q6_world', decode)
        # Every use of the feature normal normalizes it first.
        feature = shader[shader.index('float4 q6_raw_feature('):shader.index('float4 sample_road_source(')]
        uses = feature.count('input.geometry_normal')
        self.assertEqual(uses, feature.count('normalize(input.geometry_normal)') + 1)  # + the decode

    def test_ground_kit_farms_keep_their_trees_and_farmhouses(self):
        # The plots' wider fade was applied to every farm relief sample, so
        # trees and farmhouses (which need .11/.14 clearance) failed most of
        # their spots and about 70% of them were dropped. Props keep their
        # clearance tests; only plots fade wider.
        run_cpp(KIT + r'''
int main(){
 Kit kit;
 auto add=[&](char const* id,char const* name){kit.farm.assets.push_back(flat(id,0));FeatureGroup group;group.name=name;
  FeaturePlacement placement{};placement.asset_index=unsigned(kit.farm.assets.size()-1);
  group.placements.push_back(placement);kit.farm.groups.push_back(group);};
 add("farm_kit:crop:plot0:e0","farm_kit:plots");add("farm_kit:crop:ground:e0","farm_kit:ground");
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 unsigned planned=0,kept=0;
 for(unsigned seed=0;seed<200;++seed){
  c3x_renderer_tile_v1 tile{};tile.tile_x=10;tile.tile_y=10;tile.city_id=-1;tile.terrain_type=2;tile.variant_seed=seed*2654435761u;
  tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
  objects::Plan plan,routes;routes.routes.push_back({0,.5f,1,.5f,0,false,false,false});
  assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  auto props=[&]{unsigned count=0;for(auto const& i:plan.instances){auto const& id=kit.farm.assets[i.asset].id;
   count+=id.find(":tree:")!=std::string::npos || id.find(":building:")!=std::string::npos;}return count;};
  planned+=props();
  objects::clear_farm(plan,routes,kit.assets,{});
  objects::settle_farm_fields(plan,tile,kit.assets,dry);
  objects::settle_farm_props(plan,tile,kit.assets,dry);
  kept+=props();
 }
 assert(kept*10>=planned*7); // the patchwork kit keeps about 75% beside one road
}
''')

    def test_farms_stop_at_the_foot_of_mountains_and_neighbouring_hills(self):
        # Farm decals drape on the rendered ground, so a neighbouring mountain
        # or hill rising over the farm tile's edge carried the farm ground and
        # plots up its slope. The farm clearance now ends where a mountain
        # lifts the ground or (for a farm not on a hill) a neighbour's hill
        # footprint begins; a hill farm keeps draping over its own hill.
        run_cpp(KIT + r'''
int main(){
 using objects::farm_slope_clearance;
 assert(farm_slope_clearance(0,0,false)==1.f);                 // open land
 assert(farm_slope_clearance(2,0,false)>.16f);                  // a mountain's foot: still whole
 assert(farm_slope_clearance(10,0,false)<=0.f);                 // up its slope: cut
 assert(farm_slope_clearance(0,.05f,false)>.1f && farm_slope_clearance(0,.2f,false)<0.f); // a neighbour's hill
 assert(farm_slope_clearance(0,.9f,true)==1.f);                 // a hill farm on its own hill
 assert(farm_slope_clearance(12,.9f,true)<0.f);                 // ...but not up a mountain
}
''')
        source = (ROOT / 'Renderer/native/object_preparation.h').read_text()
        relief = source[source.index('auto relief=[&](float u,float v)'):]
        relief = relief[:relief.index('return std::array<float,3>{s.height,s.authored_height,farm_clearance};')]
        self.assertIn('farm_slope_clearance(', relief)
        self.assertIn('route_height(u,v)-flat_ground', relief)

    def test_sparse_farms_keep_about_two_trees_and_most_farmhouses(self):
        # The user's chosen density (2026-10-07): between the earlier kit's 4-6
        # trees and a farmhouse on every farm, and 0-2 trees on half of them.
        run_cpp(KIT + r'''
int main(){
 Kit kit;
 auto add=[&](char const* id,char const* name){kit.farm.assets.push_back(flat(id,0));FeatureGroup group;group.name=name;
  FeaturePlacement placement{};placement.asset_index=unsigned(kit.farm.assets.size()-1);
  group.placements.push_back(placement);kit.farm.groups.push_back(group);};
 add("farm_kit:crop:plot0:e0","farm_kit:plots");add("farm_kit:crop:ground:e0","farm_kit:ground");
 add("farm_kit:crop:x:e0","farm_kit:sparse");
 auto dry=[](float,float){return std::array<float,3>{0,0,1};};
 unsigned trees=0,houses=0,farms=0;
 for(unsigned seed=0;seed<400;++seed){
  c3x_renderer_tile_v1 tile{};tile.tile_x=10;tile.tile_y=10;tile.city_id=-1;tile.terrain_type=2;tile.variant_seed=seed*2654435761u;
  tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION;
  objects::Plan plan,routes;routes.routes.push_back({0,.5f,1,.5f,0,false,false,false});
  assert(objects::select_improvements(tile,kit.assets,2,0,false,true,plan));
  objects::clear_farm(plan,routes,kit.assets,{});
  objects::settle_farm_fields(plan,tile,kit.assets,dry);
  objects::settle_farm_props(plan,tile,kit.assets,dry);
  for(auto const& i:plan.instances){auto const& id=kit.farm.assets[i.asset].id;
   trees+=id.find(":tree:")!=std::string::npos;houses+=id.find(":building:")!=std::string::npos;}
  ++farms;
 }
 double per_tree=double(trees)/farms,per_house=double(houses)/farms;
 assert(per_tree>1.6 && per_tree<2.7 && per_house>.6 && per_house<.9);
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
        # Only the depth basis is under test: screen position and the
        # shadow-receiver world point are left out.
        depth = '\n'.join(line for line in depth.splitlines()
                          if 'o.position.xy=' not in line and 'q6_world' not in line)
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
