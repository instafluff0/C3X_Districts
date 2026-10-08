"""Ground states (pollution, craters, city ruins) and site sizes: selection, draping and pack.
They follow Civ III's tile state exactly (user, 2026-10-07): no art is hidden for them."""
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class GroundStateTests(unittest.TestCase):
    def test_selection_order_tile_state_and_steep_ground(self):
        run_cpp(r'''
#include "Renderer/native/rigid_object_instance.h"
#include <cassert>
#include <cmath>
#include <string>
using namespace c3x_renderer;
static FeatureAsset flat(char const* id){
 FeatureAsset a;a.id=id;
 for(int j=0;j<3;++j)for(int i=0;i<3;++i)a.vertices.push_back({{(i-1)*.5f,(j-1)*.5f,.003f},{0,0,1},{i*.5f,j*.5f}});
 for(int j=0;j<2;++j)for(int i=0;i<2;++i){unsigned k=unsigned(j*3+i);a.indices.insert(a.indices.end(),{k,k+1,k+4,k,k+4,k+3});}
 return a;
}
int main(){
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets{};
 for(unsigned f=0;f<bundles.size();++f)assets.bundles[f]=&bundles[f];
 auto& sites=bundles[objects::site_family];
 // A pack without ground-state groups draws nothing new and does not fail.
 c3x_renderer_tile_v1 tile{};tile.tile_x=40;tile.tile_y=20;tile.real_terrain_type=2;tile.city_id=-1;
 tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_POLLUTION|C3X_RENDERER_IMPROVEMENT_CRATER|C3X_RENDERER_IMPROVEMENT_RUINS;
 {objects::Plan plan;assert(objects::select_improvements(tile,assets,2,tile.improvement_flags,false,false,plan));
  assert(plan.instances.empty());}
 sites.assets={flat("decal/ground/pollution_0:e0"),flat("decal/ground/crater_0:e0"),flat("decal/ground/ruins_0:e0")};
 auto group=[&](char const* name,std::initializer_list<unsigned> parts){
  FeatureGroup g;g.name=name;for(unsigned p:parts){FeaturePlacement placement{};placement.asset_index=p;placement.scale=1;g.placements.push_back(placement);}
  sites.groups.push_back(g);};
 group("pollution_0",{0});group("crater_0",{1});group("ruins_0",{2});
 for(unsigned seed=0;seed<16;++seed){
  tile.variant_seed=seed*977u;
  objects::Plan plan;assert(objects::select_improvements(tile,assets,2,tile.improvement_flags,false,false,plan));
  // Pollution, then craters over it, then the ruins' rubble.
  assert(plan.instances.size()==3);
  for(unsigned i=0;i<3;++i)assert(plan.instances[i].family==objects::site_family && plan.instances[i].asset==i);
  // Decals light as ground (owner 0 -> resource-decal material).
  for(unsigned i=0;i<3;++i)assert(plan.instances[i].owner==0.f);
  // Crater and rubble relief is baked sunlit: never turned.
  assert(plan.instances[1].rotation==0.f && plan.instances[2].rotation==0.f);
 }
 // The tile's ruins flag alone decides: a city on the tile draws over them, as in Civ III.
 tile.city_id=3;{objects::Plan plan;assert(objects::select_improvements(tile,assets,2,tile.improvement_flags,false,false,plan));
  assert(plan.instances.size()==3);}
 tile.city_id=-1;{objects::Plan plan;assert(objects::select_improvements(tile,assets,11,tile.improvement_flags,false,false,plan));
  assert(plan.instances.empty());}
 // On a peak a resource decal is left out (steep_decal); a ground state stays.
 objects::Projection p;p.tile=tile;p.tile_width=128;p.content_view_height=640;p.half_w=64;p.half_h=32;
 p.relief_projection_scale=128.f/224*.82f;p.feature_projection_scale=128.f/224;p.pickup_profile=true;p.world_objects=true;
 float cu=float(tile.tile_x+tile.tile_y)*.5f+.5f,cv=float(tile.tile_x-tile.tile_y)*.5f+.5f;
 auto relief=[](float,float){return std::array<float,3>{0,0,1};};
 auto peak=[&](float u,float v){float d=std::max(std::abs(u-cu),std::abs(v-cv));return 2.5f+std::max(0.f,200.f-300.f*d);};
 sites.assets.push_back(flat("decal/resource:e0"));
 for(unsigned a:{0u,4u}){
  objects::Plan plan;plan.instances.push_back({objects::site_family,a,objects::site_layer,.5f,.5f,0.f,1.f,21.f,0.f,false});
  objects::Surfaces out;objects::compile(plan,p,assets,relief,peak,out);
  assert(out.layers[objects::site_layer].empty()==(a==4u));
 }
 // An irrigated tile keeps its farm under any ground state: the farm pack is
 // still asked for (the empty one fails the selection), as Civ III keeps irrigation.
 unsigned const states[]={0u,unsigned(C3X_RENDERER_IMPROVEMENT_POLLUTION),unsigned(C3X_RENDERER_IMPROVEMENT_CRATER),
  unsigned(C3X_RENDERER_IMPROVEMENT_RUINS)};
 for(unsigned state:states){
  tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_IRRIGATION|state;
  objects::Plan plan;assert(!objects::select_improvements(tile,assets,2,state,false,true,plan));
 }
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    @unittest.skipUnless((ROOT / "Renderer/packs/GroundStatesNormalized/manifest.json").is_file() and
                         (ROOT / "Renderer/packs/TileObjectsNormalized/manifest.json").is_file(),
                         "local ground-state or site source art absent")
    def test_site_pack_carries_ground_states_and_true_proportions(self):
        from Renderer.tools.asset_compiler.build_site_runtime import HEIGHT_SCALE, build, plan
        groups, _, _ = plan()
        # Huts and camps keep Civ VI's own proportions (no 2.6x vertical stretch).
        for role, parts in groups.items():
            for mesh, _ in parts:
                zs = [v["position"][2] for v in mesh["vertices"]]
                self.assertLess(max(zs), .229 / HEIGHT_SCALE * 2.3, role)
        with tempfile.TemporaryDirectory() as folder:
            build(Path(folder))
            data = (Path(folder) / "sites.bin").read_bytes()
        offset = 8
        _, textures, assets, group_count = struct.unpack_from("<IIII", data, offset); offset += 16
        def string():
            nonlocal offset
            n = struct.unpack_from("<I", data, offset)[0]; offset += 4
            value = data[offset:offset + n].decode(); offset += n
            return value
        paths = [string() for _ in range(textures)]
        self.assertEqual(8, len(paths))
        self.assertEqual(7, len(set(paths)), "site art and three ground textures")
        ids = []
        for _ in range(assets):
            ids.append(string())
            _, vertices, indices = struct.unpack_from("<III", data, offset)
            offset += 12 + vertices * 32 + indices * 4
        names = []
        for _ in range(group_count):
            names.append(string())
            offset += 4 + struct.unpack_from("<I", data, offset)[0] * 36
        for prefix, count in (("pollution_", 4), ("crater_", 4), ("ruins_", 3)):
            self.assertEqual(count, sum(n.startswith(prefix) for n in names), prefix)
        self.assertTrue(all(i.startswith("decal/ground/") for i in ids if i.split(":")[0].split("/")[-1][:4] in ("poll", "crat")))
        self.assertEqual(3, sum(i.startswith("decal/ground/ruins_") for i in ids))
        self.assertFalse(any(i.startswith("ground/") for i in ids), "ruins are the rubble decal alone")


if __name__ == "__main__":
    unittest.main()
