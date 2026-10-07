"""The production mine: one chosen building for every era, seated on the visible ground."""
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "Renderer/packs/ImprovementsNormalized"


class MinePlacementTests(unittest.TestCase):
    def test_mine_stands_on_visible_ground_by_terrain(self):
        # Mines used the coarse relief and the feature depth basis, so a hill's or
        # mountain's rendered surface buried or covered them.
        run_cpp(r'''
#include "Renderer/native/rigid_object_instance.h"
#include <cassert>
#include <cmath>
#include <string>
using namespace c3x_renderer;
int main(){
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets{};
 for(unsigned f=0;f<bundles.size();++f)assets.bundles[f]=&bundles[f];
 FeatureAsset building;building.id="mine_0:e1";
 building.vertices={{{-.1f,0,0},{0,0,1},{0,0}},{{.1f,0,0},{0,0,1},{1,0}},{{0,.1f,.1f},{0,0,1},{0,1}}};
 building.indices={0,1,2};bundles[objects::mine_family].assets.push_back(building);
 for(unsigned g=0;g<6;++g){
  FeatureGroup group;group.name="mine_"+std::to_string(g);
  FeaturePlacement placement{};placement.asset_index=0;placement.scale=2;
  group.placements.push_back(placement);bundles[objects::mine_family].groups.push_back(group);
 }
 c3x_renderer_tile_v1 tile{};tile.improvement_flags=C3X_RENDERER_IMPROVEMENT_MINE;tile.tile_x=40;tile.tile_y=20;
 // Centred on flat land and a hill's crown; a mountain's camera-facing foot.
 struct Case{int real;float anchor,scale;};
 for(auto c:{Case{2,.5f,2.f},Case{5,.5f,1.8f},Case{6,.78f,1.5f},Case{10,.78f,1.5f}}){
  tile.real_terrain_type=c.real;
  for(int era=0;era<4;++era){
   tile.route_style=era;tile.variant_seed=unsigned(era*7+c.real);
   objects::Plan plan;assert(objects::select_improvements(tile,assets,2,0,true,false,plan));
   assert(plan.instances.size()==1);
   auto const& instance=plan.instances[0];
   assert(instance.family==objects::mine_family && instance.asset==0);
   assert(std::abs(instance.u-c.anchor)<1e-6f && std::abs(instance.v-c.anchor)<1e-6f);
   assert(std::abs(instance.scale-c.scale)<1e-5f && std::abs(instance.rotation)<=.24f);
   // Emissive code 1 (.02) plus the natural height-depth marker (.0035, as
   // farm kit props): with the feature basis the hill hid the mine's lower half.
   assert(std::abs(instance.owner-.0235f)<1e-6f && std::abs(instance.material-21.f)<1e-6f);
  }
 }
 tile.real_terrain_type=5;tile.route_style=0;
 objects::Plan plan;assert(objects::select_improvements(tile,assets,2,0,true,false,plan));
 objects::Projection p;p.tile=tile;p.tile_width=128;p.content_view_height=640;p.half_w=64;p.half_h=32;
 p.relief_projection_scale=128.f/224*.82f;p.feature_projection_scale=128.f/224;
 p.pickup_profile=true;p.world_objects=true;
 // The rendered hill rises 9 units above the coarse relief.
 auto relief=[](float,float){return std::array<float,3>{0,0,1};};
 auto height=[](float,float){return 2.5f+9.f;};
 objects::Surfaces out;objects::compile(plan,p,assets,relief,height,out);
 float lowest=1e9f;
 for(auto const& vertex:out.layers[objects::mine_layer])lowest=std::min(lowest,vertex.world_z*112.f-2.5f);
 assert(!out.layers[objects::mine_layer].empty() && std::abs(lowest-9.f)<1e-3f);
 // The shared rigid path (Renderer64) seats it the same way.
 assert(std::abs(objects::prepare_rigid(plan.instances[0],p,assets,relief,height).instance.place[7]-9.f)<1e-4f);
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))

    @unittest.skipUnless((PACK / "manifest.json").is_file(), "local ImprovementsNormalized pack absent")
    def test_runtime_holds_the_chosen_building_for_every_era(self):
        from Renderer.tools.asset_compiler import build_mine_runtime as mine
        from Renderer.tools.asset_compiler.improvement_asset_importer import _asset_id
        strategy = json.loads(mine.STRATEGY.read_text())["mine"]
        choice = strategy["runtime_building"]
        with tempfile.TemporaryDirectory() as directory:
            data = mine.build(PACK, target_name=str(Path(directory) / "mine_runtime.bin")).read_bytes()
        cursor = [len(mine.MAGIC)]

        def take(fmt):
            values = struct.unpack_from(fmt, data, cursor[0])
            cursor[0] += struct.calcsize(fmt)
            return values

        def text():
            (length,) = take("<I")
            cursor[0] += length
            return data[cursor[0] - length:cursor[0]].decode()
        version, texture_count, asset_count, group_count = take("<IIII")
        textures = [text() for _ in range(texture_count)]
        self.assertEqual((1, 8, 6), (version, texture_count, group_count))
        self.assertTrue(all(path == mine.PLACEHOLDER for path in textures[2:6]))
        assets = []
        for _ in range(asset_count):
            name = text()
            _texture, vertex_count, index_count = take("<III")
            vertices = [take("<8f") for _ in range(vertex_count)]
            take(f"<{index_count}I")
            assets.append((name, vertices))
        groups = {}
        for _ in range(group_count):
            name = text()
            (count,) = take("<I")
            groups[name] = [take("<IffIIIIff") for _ in range(count)]
        self.assertEqual(len(data), cursor[0])
        self.assertEqual({f"mine_{index}" for index in range(6)}, set(groups))
        self.assertEqual(1, len({tuple(placements) for placements in groups.values()}))
        self.assertTrue(all(abs(placement[1] - choice["scale"]) < 1e-6 for placement in groups["mine_0"]))
        self.assertEqual(asset_count, len(groups["mine_0"]))
        # The building's own worked draws, turned so its Civ VI front faces the camera.
        manifest = json.loads((PACK / "manifest.json").read_text())
        landmark = json.loads((PACK / manifest["assets"][
            _asset_id(strategy["source_package"], choice["source_entry"])]["landmark"]).read_text())
        source = []
        for binding in landmark["draw_bindings"]:
            if "worked" in binding["states"]:
                mesh = json.loads((PACK / landmark["components"]["geometry"][binding["geometry"]]).read_text())
                source += [vertex["position"] for vertex in mesh["vertices"]]
        self.assertEqual(len(source), sum(len(vertices) for _, vertices in assets))
        turn = math.radians(choice["facing_degrees"])
        expected = [(x * math.cos(turn) - y * math.sin(turn), x * math.sin(turn) + y * math.cos(turn), z)
                    for x, y, z in source]
        actual = [vertex[:3] for _, vertices in assets for vertex in vertices]
        for axis in range(3):
            # Each axis sorted on its own: float32 rounding cannot reorder it.
            pairs = zip(sorted(point[axis] for point in expected), sorted(point[axis] for point in actual))
            self.assertTrue(all(abs(a - b) < 1e-5 for a, b in pairs))


if __name__ == "__main__":
    unittest.main()
