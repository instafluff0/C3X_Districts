"""Civ III's tile buildings: the site pack's compositions by era and family, on their ground (water, a mountain's
rock), the barricade's ring along a river's bank, colonies around their resource, and the frozen pack's merge."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]

PRELUDE = r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
#include <cmath>
#include <string>
namespace c3x_renderer {
std::uint32_t stable_hash(std::uint32_t value){return patterns::feature_hash(value);}
float stable_random(std::uint32_t value){return patterns::stable_random(value);}
FeatureGroup const* find_feature_group(FeatureBundle const& bundle,char const* name){
 for(auto const& group:bundle.groups)if(group.name==name)return &group;
 return nullptr;}
FeatureComposition const* find_feature_composition(FeatureBundle const& bundle,char const* name){
 for(auto const& composition:bundle.compositions)if(composition.name==name)return &composition;
 return nullptr;}
}
using namespace c3x_renderer;
// A box of half sizes (hu, hv) standing h tall, centred on its origin.
FeatureAsset box(std::string id,float hu,float hv,float h){
 FeatureAsset asset;asset.id=id;
 for(int k=0;k<8;++k)asset.vertices.push_back({{k&1?hu:-hu,k&2?hv:-hv,k&4?h:0.f},{0,0,1},{0,0}});
 asset.indices={0,1,5,0,5,4,1,3,7,1,7,5,3,2,6,3,6,7,2,0,4,2,4,6,4,5,7,4,7,6};
 return asset;
}
// One composition of one instance, centred on the tile as the bake centres a lone piece.
void composition(FeatureBundle& bundle,std::string name,unsigned asset,float scale=1){
 FeatureComposition c;c.name=name;FeatureComposition::Variant variant;
 FeatureComposition::Instance item;item.asset=asset;item.scale=scale;item.u=item.v=.5f;
 variant.instances.push_back(item);c.variants.push_back(variant);bundle.compositions.push_back(c);
}
// The site pack's buildings for every era: a keep, the barricade's ring pieces
// (walls .2 long; a gate in era 1 only), outposts, airfields, radar, victory.
struct Pack {
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets{};
 FeatureBundle& site=bundles[objects::site_family];
 Pack(){
  for(unsigned f=0;f<bundles.size();++f)assets.bundles[f]=&bundles[f];
  auto add=[&](std::string id,std::string name,float hu,float hv,float h){
   site.assets.push_back(box(id,hu,hv,h));composition(site,name,unsigned(site.assets.size()-1));};
  for(int era=0;era<4;++era){std::string e=std::to_string(era);
   add("district/fortress_"+e,"fortress_"+e,.12f,.12f,.2f);add("district/keep_"+e,"keep_"+e,.1f,.1f,.16f);
   add("walls/wall_"+e,"wall_"+e,.1f,.015f,.06f);
   add("district/tower_"+e,"tower_"+e,.03f,.03f,.1f);
   if(era==1)add("district/gate_"+e,"gate_"+e,.1f,.03f,.08f);
   add("district/outpost_"+e,"outpost_"+e,.04f,.04f,.3f);
  }
  add("district/airfield_0","airfield_0",.2f,.15f,.1f);add("decal/runway/x","airfield_1",.35f,.1f,0.f);
  add("district/radar","radar",.08f,.08f,.25f);add("district/victory","victory",.04f,.04f,.3f);
 }
 std::string id(objects::Instance const& instance)const{
  auto const& id=site.assets[instance.asset].id;return id.substr(id.find('/')+1);}
};
// A piece's footprint (u0,v0,u1,v1), as placed (a wall section stretched along its length).
std::array<float,4> footprint(objects::Assets const& assets,objects::Instance const& instance){
 float u0=1e9f,v0=1e9f,u1=-1e9f,v1=-1e9f,co=std::cos(instance.rotation),si=std::sin(instance.rotation);
 for(auto const& p:assets[objects::site_family].assets[instance.asset].vertices){
  float x=p.position[0]*instance.stretch;
  float u=instance.u+(x*co-p.position[1]*si)*instance.scale,v=instance.v+(x*si+p.position[1]*co)*instance.scale;
  u0=std::min(u0,u);u1=std::max(u1,u);v0=std::min(v0,v);v1=std::max(v1,v);}
 return {u0,v0,u1,v1};
}
auto dry=[](float,float){return false;};
auto flat=[](float,float){return 0.f;};
c3x_renderer_tile_v1 site(unsigned flags,int era,unsigned owner_era){
 c3x_renderer_tile_v1 tile{};tile.tile_x=10;tile.tile_y=4;tile.route_style=era;
 tile.improvement_flags=flags|owner_era<<C3X_RENDERER_IMPROVEMENT_BUILDING_ERA_SHIFT;return tile;
}
'''


class TileBuildingTests(unittest.TestCase):
    def test_each_family_draws_its_composition_by_era(self):
        run_cpp(PRELUDE + r'''
#include <map>
int main(){
 Pack pack;
 auto drawn=[&](unsigned flags,int era,unsigned owner_era){
  objects::Plan plan;std::map<std::string,int> out;
  if(!objects::place_tile_buildings(plan,pack.assets,site(flags,era,owner_era),objects::Plan{},{},dry,flat))return std::map<std::string,int>{{"FAILED",1}};
  for(auto const& instance:plan.instances){
   assert(instance.family==objects::site_family && instance.layer==objects::site_layer);
   assert((pack.site.assets[instance.asset].id.rfind("decal/",0)==0)==(instance.owner==0.f));
   ++out[pack.id(instance)];
  }
  return out;
 };
 using M=std::map<std::string,int>;
 for(int era=0;era<4;++era){std::string e=std::to_string(era);
  // Fortresses follow the viewer's era and stand where their composition put them.
  assert(drawn(C3X_RENDERER_IMPROVEMENT_FORTRESS,era,3)==(M{{"fortress_"+e,1}}));
  // A barricade: its keep in a ring of five walls a side, towers at the four
  // corners and, in era 1, the gate in the middle of a front side.
  auto ring=drawn(C3X_RENDERER_IMPROVEMENT_FORTRESS|C3X_RENDERER_IMPROVEMENT_BARRICADE,era,0);
  assert(ring["keep_"+e]==1 && !ring.count("fortress_"+e) && ring["wall_"+e]==(era==1?19:20) && ring["tower_"+e]==4 &&
         ring["gate_1"]==(era==1?1:0));
 }
 // Civ III's airfield has an early and a modern sprite: (era-1)/2.
 for(int era=0;era<3;++era)assert(drawn(C3X_RENDERER_IMPROVEMENT_AIRFIELD,era,0)==(M{{"airfield_0",1}}));
 assert(drawn(C3X_RENDERER_IMPROVEMENT_AIRFIELD,3,0)==(M{{"runway/x",1}}));
 // Outposts follow their owner's era, not the viewer's.
 for(unsigned owner=0;owner<4;++owner)assert(drawn(C3X_RENDERER_IMPROVEMENT_OUTPOST,3-int(owner),owner)==(M{{"outpost_"+std::to_string(owner),1}}));
 assert(drawn(C3X_RENDERER_IMPROVEMENT_RADAR|C3X_RENDERER_IMPROVEMENT_VICTORY_POINT,1,2)==(M{{"radar",1},{"victory",1}}));
 // A colony without its kit in the pack draws nothing.
 assert(drawn(C3X_RENDERER_IMPROVEMENT_COLONY,1,1).empty());
 // A pack without a building's art fails rather than drop it.
 auto radar=pack.site.compositions.end()-2;pack.site.compositions.erase(radar);
 assert(drawn(C3X_RENDERER_IMPROVEMENT_RADAR,3,3)==(M{{"FAILED",1}}));
 // A site pack from before tile buildings (no compositions) draws none, as before.
 pack.site.compositions.clear();
 assert(drawn(C3X_RENDERER_IMPROVEMENT_FORTRESS|C3X_RENDERER_IMPROVEMENT_RADAR,1,1).empty());
}
''')

    def test_the_ring_lies_on_the_line_the_drafted_ring_used(self):
        run_cpp(PRELUDE + r'''
int main(){
 Pack pack;objects::Plan plan;
 assert(objects::place_tile_buildings(plan,pack.assets,site(C3X_RENDERER_IMPROVEMENT_BARRICADE,1,0),objects::Plan{},{},dry,flat));
 unsigned walls=0,towers=0,gates=0;
 for(auto const& i:plan.instances){
  auto id=pack.id(i);
  bool side_u=std::abs(i.u-.03f)<1e-3f || std::abs(i.u-.97f)<1e-3f,side_v=std::abs(i.v-.03f)<1e-3f || std::abs(i.v-.97f)<1e-3f;
  if(id=="keep_1")assert(i.u==.5f && i.v==.5f && i.scale==1.f && i.ground_fit>0.f);
  if(id=="tower_1"){++towers;assert(side_u && side_v);}
  // Walls along each side, turned along it, with the drafted ring's 1.02 overlap;
  // beside the gate, the two halves of its side.
  if(id=="wall_1"){++walls;assert(side_u!=side_v);
   float along=std::abs(std::sin(i.rotation));assert(side_v?along<1e-3f:std::abs(along-1.f)<1e-3f);
   bool gate_side=std::abs(i.v-.97f)<1e-3f;
   assert(i.scale==1.f && std::abs(i.stretch-(gate_side?(.94f-.2f)*.5f/(2*.2f):.94f/(5*.2f))*1.02f)<1e-4f);}
  // The gate in the middle of the front-left side (v at the far edge).
  if(id=="gate_1"){++gates;assert(std::abs(i.v-.97f)<1e-3f && std::abs(i.u-.5f)<1e-3f && i.scale==1.f);}
 }
 assert(walls==19 && towers==4 && gates==1);
}
''')

    def test_the_barricade_ring_wraps_along_a_river(self):
        run_cpp(PRELUDE + r'''
#include <vector>
int main(){
 Pack pack;
 // A river along the tile's back: water and its bank where v < .3.
 auto river=[](float,float v){return v<.3f;};
 objects::Plan plan;
 assert(objects::place_tile_buildings(plan,pack.assets,site(C3X_RENDERER_IMPROVEMENT_BARRICADE,2,0),objects::Plan{},{},river,flat));
 std::vector<objects::Instance> walls,towers;bool keep=false;
 for(auto const& i:plan.instances){
  auto id=pack.id(i);auto b=footprint(pack.assets,i);
  if(id=="keep_2"){keep=true;assert(b[1]>=.3f);}          // the keep stays on the dry ground
  if(id=="wall_2")walls.push_back(i);
  if(id=="tower_2")towers.push_back(i);
  assert(b[1]>.26f);                                    // nothing reaches into the water
 }
 assert(keep);
 // The back of the ring runs along the bank, turned along it, not along the tile's edge.
 unsigned bank=0;
 for(auto const& w:walls){
  assert(w.v>.29f);
  if(std::abs(w.v-.3f)<.03f){++bank;assert(std::abs(std::sin(w.rotation))<1e-3f);}
 }
 assert(bank>=4);
 // Towers where the bank meets the tile's sides, and at the front corners.
 assert(towers.size()==4);
 for(auto const& t:towers)assert(t.v>.28f && (std::abs(t.u-.03f)<.01f || std::abs(t.u-.97f)<.01f));
 // A diagonal bank: the ring follows it in a straight run, turned along it.
 auto diagonal=[](float u,float v){return v<.7f-.6f*u;};
 objects::Plan slanted;
 assert(objects::place_tile_buildings(slanted,pack.assets,site(C3X_RENDERER_IMPROVEMENT_BARRICADE,2,0),objects::Plan{},{},diagonal,flat));
 unsigned along=0;
 for(auto const& i:slanted.instances){
  auto b=footprint(pack.assets,i);
  assert(b[3]+.02f>=.7f-.6f*b[2]);   // no piece wholly in the water (its driest corner is dry)
  if(pack.id(i)=="wall_2" && std::abs(i.v-(.7f-.6f*i.u))<.04f){
   ++along;float heading=std::atan2(std::sin(i.rotation),std::cos(i.rotation));
   assert(std::abs(std::abs(std::tan(heading))-.6f)<.12f);}
 }
 assert(along>=3);
}
''')

    def test_routes_cross_the_ring_through_gates_or_openings(self):
        run_cpp(PRELUDE + r'''
#include <vector>
int main(){
 Pack pack;
 // A road through the tile, back to front (u = .5).
 objects::Plan road;road.routes.push_back(objects::Route{.5f,0.f,.5f,1.f,0,false,false,false});
 auto blocks=[&](objects::Plan const& plan,float u,float v){
  for(auto const& i:plan.instances){auto id=pack.id(i);
   if(id.rfind("wall_",0)!=0)continue;auto b=footprint(pack.assets,i);
   if(u>b[0]+.01f && u<b[2]-.01f && v>b[1]-.01f && v<b[3]+.01f)return true;}
  return false;};
 // Medieval: the road passes through a gate on each side it crosses; no other gate.
 objects::Plan gated;
 assert(objects::place_tile_buildings(gated,pack.assets,site(C3X_RENDERER_IMPROVEMENT_BARRICADE,1,0),road,{},dry,flat));
 std::vector<objects::Instance> gates;
 for(auto const& i:gated.instances)if(pack.id(i)=="gate_1")gates.push_back(i);
 assert(gates.size()==2);
 for(auto const& g:gates)assert(std::abs(g.u-.5f)<.01f && (std::abs(g.v-.03f)<.01f || std::abs(g.v-.97f)<.01f));
 assert(!blocks(gated,.5f,.03f) && !blocks(gated,.5f,.97f));
 // Industrial: a plain opening where the road crosses; towers stay at the corners only.
 objects::Plan open;
 assert(objects::place_tile_buildings(open,pack.assets,site(C3X_RENDERER_IMPROVEMENT_BARRICADE,2,0),road,{},dry,flat));
 unsigned towers=0,flanking=0;
 for(auto const& i:open.instances)if(pack.id(i)=="tower_2"){++towers;
  if(std::abs(std::abs(i.u-.5f)-.07f)<.01f && (std::abs(i.v-.03f)<.01f || std::abs(i.v-.97f)<.01f))++flanking;}
 assert(towers==4 && flanking==0);
 assert(!blocks(open,.5f,.03f) && !blocks(open,.5f,.97f));
 // A bridge landing on a river's bank: the bank's wall opens where the road comes ashore.
 auto river=[](float,float v){return v<.3f;};
 objects::Plan banked;
 assert(objects::place_tile_buildings(banked,pack.assets,site(C3X_RENDERER_IMPROVEMENT_BARRICADE,2,0),road,{},river,flat));
 assert(!blocks(banked,.5f,.2957f));
 // Without the road the bank's wall is whole.
 objects::Plan whole;
 assert(objects::place_tile_buildings(whole,pack.assets,site(C3X_RENDERER_IMPROVEMENT_BARRICADE,2,0),objects::Plan{},{},river,flat));
 assert(blocks(whole,.5f,.2957f));
}
''')

    def test_buildings_leave_the_water_and_take_a_mountains_foot(self):
        run_cpp(PRELUDE + r'''
int main(){
 Pack pack;
 // A river through the tile's middle: the fortress keeps out of it, whole, on the nearer bank.
 auto middle=[](float,float v){return std::abs(v-.5f)<.1f;};
 objects::Plan plan;
 assert(objects::place_tile_buildings(plan,pack.assets,site(C3X_RENDERER_IMPROVEMENT_FORTRESS,1,0),objects::Plan{},{},middle,flat));
 assert(plan.instances.size()==1);
 auto b=footprint(pack.assets,plan.instances[0]);
 assert(b[3]<=.4f+1e-3f || b[1]>=.6f-1e-3f);
 assert(b[0]>=.03f-1e-3f && b[2]<=.97f+1e-3f && b[1]>=.03f-1e-3f && b[3]<=.97f+1e-3f);
 // Water everywhere: the building keeps its own spot rather than vanish.
 objects::Plan flooded;
 assert(objects::place_tile_buildings(flooded,pack.assets,site(C3X_RENDERER_IMPROVEMENT_RADAR,1,1),objects::Plan{},{},
     [](float,float){return true;},flat));
 assert(flooded.instances.size()==1 && flooded.instances[0].u==.5f && flooded.instances[0].v==.5f);
 // A peak in the tile's middle: the radar takes the rock's foot, toward the camera.
 auto peak=[](float u,float v){return std::max(0.f,.4f-1.2f*std::hypot(u-.5f,v-.5f));};
 objects::Plan mountain;
 assert(objects::place_tile_buildings(mountain,pack.assets,site(C3X_RENDERER_IMPROVEMENT_RADAR,1,1),objects::Plan{},{},dry,peak));
 auto const& radar=mountain.instances.at(0);
 assert(peak(radar.u,radar.v)<.15f && radar.u+radar.v>1.2f);
 // Flat dry ground: everything stays where its composition put it.
 objects::Plan level;
 assert(objects::place_tile_buildings(level,pack.assets,site(C3X_RENDERER_IMPROVEMENT_RADAR|C3X_RENDERER_IMPROVEMENT_AIRFIELD,1,1),objects::Plan{},{},dry,flat));
 for(auto const& i:level.instances)assert(i.scale==1.f);
 // The airfield first, at its own spot; the radar beside it, off its ground.
 assert(level.instances[0].u==.5f && level.instances[0].v==.5f);
 auto a=footprint(pack.assets,level.instances[0]),r=footprint(pack.assets,level.instances[1]);
 assert(r[0]>=a[2] || r[2]<=a[0] || r[1]>=a[3] || r[3]<=a[1]);
}
''')

    def test_site_wet_is_the_city_sites_test_and_skips_far_tiles(self):
        run_cpp(PRELUDE + r'''
int main(){
 c3x_renderer_tile_v1 tile{};tile.tile_x=10;tile.tile_y=4;     // world (7,3) .. (8,4)
 int shores=0,rivers=0;
 auto inland=objects::site_wet(tile,[&](float,float){++shores;return 5.f;},[&](float,float){++rivers;return 500.f;});
 assert(shores==1 && rivers==1);
 for(int k=0;k<10;++k)assert(!inland(.1f*float(k),.3f));
 assert(shores==1 && rivers==1);                                // a tile far from water queries nothing more
 // Near water: the shore nearer than .05 tile, a river nearer than 10 pixels.
 auto banks=objects::site_wet(tile,[](float x,float){return (x-7.f)*.2f;},[](float,float y){return (y-3.f)*20.f;});
 assert(banks(.1f,.9f) && !banks(.3f,.5f) && banks(.9f,.6f) && !banks(.9f,.4f));
}
''')

    def test_every_path_lays_tile_buildings_on_the_same_ground(self):
        # The object jobs, the tile compiler's own object path and the forest's
        # exclusions must place the same buildings, so the trees give way where
        # they stand: one function, the city sites' water test and the
        # mountain's raw shape in each. The canopy only keys on their flags.
        preparation = (ROOT / "Renderer/native/object_preparation.h").read_text()
        renderer = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        geometry = (ROOT / "Renderer/native/source_fidelity/geometry.h").read_text()
        self.assertIn("if(!place_tile_buildings(plan,assets,tile,plan,input.colony_boxes,\n        site_wet(tile,", preparation)
        self.assertIn("mountain_shape.sample(natural,tile_u+u,tile_v+1.f-v).displacement)*\n            (.82f/150.f)", preparation)
        self.assertIn("place_tile_buildings(plan,object_assets,tile,routes,\n                        resource_boxes(tile,false,true),tile_building_wet(tile),tile_building_rock(tile)))return false;", renderer)
        self.assertIn("place_tile_buildings(building_plan,object_assets,building_site,building_routes,", geometry)
        self.assertIn("tile_building_wet(building_site),tile_building_rock(building_site)))continue;", geometry)
        self.assertIn("return c3x_renderer::objects::site_wet(site,", renderer)
        self.assertIn("shape.sample(natural,x0+u,y0+1.f-v).displacement)*(.82f/150.f)", renderer)
        canopy = renderer[renderer.index("canopy_clearing(\n"):renderer.index("// Preparation and prefetch enter")]
        self.assertNotIn("place_tile_buildings", canopy)
        self.assertIn("clearing.buildings=", canopy)
        run_cpp(r'''
#include "Renderer/lab/shared/natural/canopy_clearing.h"
#include <cassert>
int main(){
 c3x_renderer::fidelity::CanopyClearing clearing;assert(clearing.hash()==0);
 clearing.buildings=7;auto seven=clearing.hash();assert(seven!=0);
 clearing.buildings=8;assert(clearing.hash()!=seven);
}
''')

    @unittest.skipUnless((ROOT / "Renderer/packs/TileObjectsNormalized/manifest.json").is_file(),
                         "local site source art absent")
    def test_site_pack_merges_the_frozen_buildings_and_moves_huts_into_their_atlas(self):
        import hashlib, json, struct, tempfile
        from Renderer.tools.asset_compiler import build_site_runtime as sites
        _, textures, _ = sites.plan()
        string = lambda value: struct.pack("<I", len(value.encode())) + value.encode()
        # A frozen pack: one atlas whose cells hold the hut and camp textures, one radar.
        cells = {hashlib.sha256((sites.SOURCE / t).read_bytes()).hexdigest(): [0, .5, .125 * k, .125, 512]
                 for k, t in enumerate(textures)}
        bundle = bytearray(b"C3XVEG1\0" + struct.pack("<IIII", 2, 8, 1, 0))
        for k in range(8):
            bundle += string("textures/atlas_0.dds" if k == 0 else "textures/unused.dds")
        bundle += string("district/radar") + struct.pack("<III", 0, 3, 3)
        bundle += b"".join(struct.pack("<8f", x, y, .1, 0, 0, 1, .1, .1) for x, y in ((0, 0), (.1, 0), (0, .1)))
        bundle += struct.pack("<3I", 0, 1, 2)
        bundle += struct.pack("<I", 1) + string("radar") + struct.pack("<I", 1) + struct.pack("<II", 255, 1)
        bundle += struct.pack("<I6f", 0, .5, .5, 0, 1, 0, 0)
        bundle += struct.pack("<I", 1) + string("radar") + struct.pack("<I", 0)
        with tempfile.TemporaryDirectory() as folder:
            frozen, out = Path(folder) / "frozen", Path(folder) / "out"
            (frozen / "textures").mkdir(parents=True)
            (frozen / "textures/atlas_0.dds").write_bytes(b"atlas")
            (frozen / "tile_buildings.bin").write_bytes(bytes(bundle))
            manifest = {"bundle": "tile_buildings.bin", "atlases": ["textures/atlas_0.dds"], "site_cells": cells}
            (frozen / "manifest.json").write_text(json.dumps(manifest))
            sites.build(out, frozen)
            data = (out / "sites.bin").read_bytes()
            self.assertEqual(b"atlas", (out / "textures/atlas_0.dds").read_bytes())
            self.assertEqual(148 + 8, len((out / "textures/unused.dds").read_bytes()), "one 4x4 BC1 block")
            # A changed hut texture is refused rather than drawn from the wrong cell.
            manifest["site_cells"] = {}
            (frozen / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                sites.build(Path(folder) / "stale", frozen)
        at = 8
        def take(layout):
            nonlocal at
            values = struct.unpack_from(layout, data, at); at += struct.calcsize(layout)
            return values
        def text():
            return take(f"<{take('<I')[0]}s")[0].decode()
        version, slots, count, groups = take("<IIII")
        paths = [text() for _ in range(slots)]
        self.assertEqual(2, version)
        self.assertEqual("textures/atlas_0.dds", paths[0])
        self.assertTrue(paths[1].startswith("textures/ground_") and paths[-1] == "textures/unused.dds")
        ids = []
        for _ in range(count):
            ids.append(text())
            texture, vertices, indices = take("<III")
            uvs = [struct.unpack_from("<2f", data, at + k * 32 + 24) for k in range(vertices)]
            at += vertices * 32 + indices * 4
            if not ids[-1].startswith(("decal/", "district/")):
                # A hut or camp body: inside its cells of atlas 0, never wrapping.
                self.assertEqual(0, texture)
                self.assertTrue(all(.5 <= u <= .625 and 0 <= v <= .125 * len(textures) for u, v in uvs), ids[-1])
        self.assertEqual("district/radar", ids[-1])
        for _ in range(groups):
            text(); placements = take("<I")[0]; at += placements * 36
        self.assertEqual(1, take("<I")[0])
        self.assertEqual("radar", text())
        take("<I"); take("<II")
        self.assertEqual(count - 1, take("<I6f")[0], "the composition's asset follows the site assets")
        self.assertEqual((1,), take("<I"))

    def test_bridge_requires_every_building_once_the_renderer_draws_any(self):
        source = (ROOT / "injected_code.c").read_text()
        start = source.index("bool\nvalidate_custom_renderer_replacement_ownership")
        end = source.index("\n// Compact topology", start)
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstddef>
enum {SQ_Forest=7,SQ_Jungle=8,SQ_Swamp=9,SQ_Volcano=10};
struct State {int custom_renderer_tile_count; c3x_renderer_tile_v1* custom_renderer_tiles;};
State state,*is=&state;
''' + source[start:end] + r'''
int main(){
 // Two drawn tiles with a fortress and a colony, and a halo tile with an outpost.
 c3x_renderer_tile_v1 tiles[3]={};
 tiles[0].tile_flags=tiles[1].tile_flags=C3X_RENDERER_TILE_RENDER;
 tiles[0].improvement_flags=C3X_RENDERER_IMPROVEMENT_FORTRESS;
 tiles[1].improvement_flags=C3X_RENDERER_IMPROVEMENT_COLONY;
 tiles[2].improvement_flags=C3X_RENDERER_IMPROVEMENT_OUTPOST;
 state={3,tiles};
 unsigned const terrain=C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED,
  building=C3X_RENDERER_TILE_CUSTOM_TILE_BUILDING_REPLACED;
 unsigned flags[3]={terrain,terrain,0};
 c3x_renderer_output_v1 out={};out.replacement_tile_count=3;out.replacement_tile_flags=flags;
 // An older renderer or site pack draws none: the frame stands, as before.
 assert(validate_custom_renderer_replacement_ownership(&out));
 // One that draws some must draw them all.
 flags[0]|=building;
 assert(!validate_custom_renderer_replacement_ownership(&out));
 flags[1]|=building;
 assert(validate_custom_renderer_replacement_ownership(&out));
 // Never a building where none was captured, nor on the halo.
 tiles[1].improvement_flags=0;
 assert(!validate_custom_renderer_replacement_ownership(&out));
 tiles[1].improvement_flags=C3X_RENDERER_IMPROVEMENT_COLONY;flags[2]=building;
 assert(!validate_custom_renderer_replacement_ownership(&out));
}
''')

    def test_colony_stands_around_its_resource(self):
        run_cpp(PRELUDE + r'''
int main(){
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets{};
 for(unsigned f=0;f<bundles.size();++f)assets.bundles[f]=&bundles[f];
 auto& site=bundles[objects::site_family];
 // Era 1's kit: two anchors, a building and two props; era 2's must stay unused.
 struct Spec {char const* name;float hu,hv,h;};
 Spec const kit[]={{"colony_1_anchor_0",.14f,.10f,.2f},{"colony_1_anchor_1",.08f,.08f,.15f},
  {"colony_1_building_2",.07f,.05f,.1f},{"colony_1_prop_3",.03f,.03f,.03f},{"colony_1_prop_4",.02f,.04f,.03f},
  {"colony_2_anchor_0",.1f,.1f,.1f}};
 for(auto const& s:kit){site.assets.push_back(box(std::string("district/")+s.name,s.hu,s.hv,s.h));
  composition(site,s.name,unsigned(site.assets.size()-1));}
 c3x_renderer_tile_v1 tile{};tile.tile_x=10;tile.tile_y=4;
 tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_COLONY|1u<<C3X_RENDERER_IMPROVEMENT_BUILDING_ERA_SHIFT;
 auto place=[&](std::vector<std::array<float,4>> const& resource){
  objects::Plan plan;assert(objects::place_tile_buildings(plan,assets,tile,objects::Plan{},resource,dry,flat));return plan;};
 auto footprint=[&](objects::Instance const& instance){
  float u0=1e9f,v0=1e9f,u1=-1e9f,v1=-1e9f,co=std::cos(instance.rotation),si=std::sin(instance.rotation);
  for(auto const& p:site.assets[instance.asset].vertices){
   float u=instance.u+(p.position[0]*co-p.position[1]*si)*instance.scale,v=instance.v+(p.position[0]*si+p.position[1]*co)*instance.scale;
   u0=std::min(u0,u);u1=std::max(u1,u);v0=std::min(v0,v);v1=std::max(v1,v);}
  return std::array<float,4>{u0,v0,u1,v1};
 };
 auto overlaps=[](std::array<float,4> a,std::array<float,4> b,float margin){
  return a[0]<b[2]+margin && a[2]>b[0]-margin && a[1]<b[3]+margin && a[3]>b[1]-margin;};
 // A resource in the middle and front of the tile, as rocks or a herd stand.
 std::vector<std::array<float,4>> resource={{.40f,.40f,.62f,.60f},{.55f,.62f,.70f,.80f},{.30f,.62f,.42f,.74f}};
 auto plan=place(resource);
 assert(plan.instances.size()==5);
 std::vector<std::array<float,4>> boxes;
 for(auto const& instance:plan.instances){
  auto const& id=site.assets[instance.asset].id;
  assert(id.rfind("district/colony_1_",0)==0 && instance.family==objects::site_family && instance.owner==.18f);
  auto box=footprint(instance);
  assert(box[0]>=0 && box[1]>=0 && box[2]<=1 && box[3]<=1);
  for(auto const& r:resource)assert(!overlaps(box,r,.01f));     // around the resource, never on it
  for(auto const& other:boxes)assert(!overlaps(box,other,0.f));  // and not on each other
  boxes.push_back(box);
 }
 // The anchors stand at the back of the tile (small u+v), the resource keeps the front.
 for(int k=0;k<2;++k)assert(plan.instances[k].u+plan.instances[k].v<.8f);
 // Deterministic per tile.
 auto again=place(resource);
 for(std::size_t k=0;k<plan.instances.size();++k)assert(again.instances[k].u==plan.instances[k].u && again.instances[k].v==plan.instances[k].v);
 // A crowding resource shrinks the anchor rather than leave it out.
 std::vector<std::array<float,4>> crowd={{.08f,.30f,.92f,.92f},{.30f,.08f,.92f,.30f}};
 auto tight=place(crowd);
 bool anchor=false;
 for(auto const& instance:tight.instances){
  for(auto const& r:crowd)assert(!overlaps(footprint(instance),r,.01f));
  if(site.assets[instance.asset].id=="district/colony_1_anchor_0"){anchor=true;assert(instance.scale<1.f);}
 }
 assert(anchor);
 // By a river the kit keeps out of its water and bank as well.
 objects::Plan banked;
 assert(objects::place_tile_buildings(banked,assets,tile,objects::Plan{},resource,[](float,float v){return v<.25f;},flat));
 assert(banked.instances.size()==5);
 for(auto const& instance:banked.instances){
  auto b=footprint(instance);assert(b[1]>=.25f-1e-3f);
  for(auto const& r:resource)assert(!overlaps(b,r,.01f));
 }
 // Without the colony flag nothing is placed.
 tile.improvement_flags=0;assert(place(resource).instances.empty());
}
''')


if __name__ == "__main__":
    unittest.main()
