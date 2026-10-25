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

    def test_mine_stands_beside_animals(self):
        # A mine and its tile's resource shared one anchor (a flat tile's centre, a
        # mountain's camera-facing foot), so cattle stood inside the mine.
        # settle_mine moves a mine beside an animal (the
        # caller passes only animals' boxes), in front of it when it can; on a
        # hill it keeps near the crown, on a mountain to the foot. Any other
        # resource keeps the mine at its anchor, among its pieces.
        run_cpp(r'''
#include "Renderer/native/object_compiler.h"
#include <cassert>
#include <cmath>
using namespace c3x_renderer;
int main(){
 std::array<FeatureBundle,objects::family_count> bundles;objects::Assets assets{};
 for(unsigned f=0;f<bundles.size();++f)assets.bundles[f]=&bundles[f];
 FeatureAsset building;building.id="mine_0:e1";
 building.vertices={{{-.1f,0,0},{0,0,1},{0,0}},{{.1f,0,0},{0,0,1},{1,0}},{{0,.1f,.1f},{0,0,1},{0,1}}};
 bundles[objects::mine_family].assets.push_back(building);
 auto mined=[&](float anchor,float scale){objects::Plan plan;
  for(unsigned part=0;part<2;++part)plan.instances.push_back({objects::mine_family,0,objects::mine_layer,anchor,anchor,.1f,scale,21.f,.0235f,part==0});
  return plan;};
 auto gap=[](std::array<float,4> m,std::array<float,4> b){
  return std::hypot(std::max({b[0]-m[0],0.f,m[0]-b[2]}),std::max({b[1]-m[1],0.f,m[1]-b[3]}))-m[2];};
 c3x_renderer_tile_v1 tile{};tile.real_terrain_type=2;
 // No animal: the mine keeps its anchor (any other resource it stands on).
 auto plan=mined(.5f,2.f);auto alone=objects::settle_mine(plan,assets,tile,{});
 assert(alone[0]==.5f && alone[1]==.5f && std::abs(alone[2]-.2f)<1e-6f && std::abs(alone[3]-.2f)<1e-6f);
 // Cattle grazing at the centre: the mine stands clear of them, in front, inside the tile.
 std::array<float,4> cattle{.42f,.42f,.58f,.58f};
 plan=mined(.5f,2.f);auto beside=objects::settle_mine(plan,assets,tile,{cattle});
 assert(gap(beside,cattle)>=.02f && beside[0]+beside[1]>1.f);
 for(float c:{beside[0],beside[1]})assert(c>=beside[2] && c<=1-beside[2]);
 for(auto const& part:plan.instances)assert(part.u==beside[0] && part.v==beside[1]);
 // An animal on a hill: the mine stays near the crown.
 tile.real_terrain_type=5;
 plan=mined(.5f,1.8f);auto crown=objects::settle_mine(plan,assets,tile,{cattle});
 assert(std::hypot(crown[0]-.5f,crown[1]-.5f)<=.22f+1e-4f && crown[0]+crown[1]>1.f);
 // Minerals on a mountain foot: the mine keeps its anchor, among them.
 tile.real_terrain_type=6;
 plan=mined(.78f,1.5f);auto foot=objects::settle_mine(plan,assets,tile,{});
 assert(foot[0]==.78f && foot[1]==.78f);
}
''')

    def test_every_mine_caller_settles_beside_animals_only(self):
        # Tile compile, Renderer64 object preparation and the forest clearing
        # around it must agree on one spot.
        renderer = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        preparation = (ROOT / "Renderer/native/object_preparation.h").read_text()
        self.assertEqual(renderer.count("settle_mine("), 2)
        self.assertEqual(renderer.count("resource_boxes(tile,true)") + renderer.count("resource_boxes(tile, true)") +
                         renderer.count("resource_boxes(near_tile,true)"), 3)
        self.assertIn("input.animal_boxes=resource_boxes(tile,true);", renderer)
        self.assertIn("settle_mine(plan,assets,tile,input.animal_boxes);", preparation)

    def test_mountain_mine_stands_on_the_drawn_rock(self):
        # Mines were seated on the natural ground; at a mountain's camera-facing
        # foot the drawn rock rises over it, so the mine vanished inside the
        # mountain. Both object paths now seat it on the mountain's shape, as
        # resources and routes are: Renderer64's rigid mines on route_height
        # (whose shape is built for mine tiles) and the tile compiler's
        # mountain mine plan on resource_height_at.
        preparation = (ROOT / "Renderer/native/object_preparation.h").read_text()
        renderer = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        self.assertIn("(input.mine_ready && (input.projection.tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)))",
                      preparation)
        # Tile buildings stand on the drawn rock too (their shape is built for them).
        self.assertIn('bool on_rock=asset.id.rfind("route/tunnel/",0)==0 || instance.family==mine_family ||\n'
                      '                    (instance.family==site_family && tile_building_asset(asset));', preparation)
        self.assertIn("(input.projection.tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_TILE_BUILDINGS) ||", preparation)
        self.assertIn("return on_rock?route_height(x,y):height_natural(x,y);", preparation)
        self.assertIn("(site_assets_ready && (tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_TILE_BUILDINGS));", renderer)
        self.assertIn("if(mined_peak)c3x_renderer::objects::compile(plan,object_projection,object_assets,relief_at_world,\n"
                      "                    resource_height_at,object_output);", renderer)

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
            indices = take(f"<{index_count}I")
            assets.append((name, vertices, indices))
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
        # Lit faces are split into their own part (bake_light), so compare the
        # triangles' corners rather than vertex lists.
        source = []
        for binding in landmark["draw_bindings"]:
            if "worked" in binding["states"]:
                mesh = json.loads((PACK / landmark["components"]["geometry"][binding["geometry"]]).read_text())
                source += [mesh["vertices"][index]["position"] for index in mesh["topology"]["indices"]]
        self.assertEqual(len(source), sum(len(indices) for _, _, indices in assets))
        turn = math.radians(choice["facing_degrees"])
        expected = [(x * math.cos(turn) - y * math.sin(turn), x * math.sin(turn) + y * math.cos(turn), z)
                    for x, y, z in source]
        actual = [vertices[index][:3] for _, vertices, indices in assets for index in indices]
        for axis in range(3):
            # Each axis sorted on its own: float32 rounding cannot reorder it.
            pairs = zip(sorted(point[axis] for point in expected), sorted(point[axis] for point in actual))
            self.assertTrue(all(abs(a - b) < 1e-5 for a, b in pairs))

    def test_light_bakes_only_faces_civ6_lights(self):
        # Civ VI lays a building's emissive map out by uv2. Sampled by the base
        # colour's uv0 (an atlas many faces share), its glow blocks landed on the
        # mine's planks and beams, and at night the headframe was plastered with
        # yellow. Two faces share one uv0 patch; only the first is lit by uv2.
        import numpy as np
        from Renderer.tools.asset_compiler import build_mine_runtime as mine
        emissive = np.zeros((64, 64, 3))
        emissive[:16, :16] = (.8, .5, .1)
        patch = [[.5, .5], [.75, .5], [.5, .75]]
        lit_uv2 = [[.02, .02], [.2, .02], [.02, .2]]
        dark_uv2 = [[.6, .6], [.9, .6], [.6, .9]]
        vertices = [{"position": [i, 0.0, 0.0], "normal": [0.0, 0.0, 1.0], "uv0": uv0, "uv2": uv2}
                    for i, (uv0, uv2) in enumerate(zip(patch + patch, lit_uv2 + dark_uv2))]
        mesh = {"vertices": vertices, "topology": {"primitive": "triangles", "indices": [0, 1, 2, 3, 4, 5]}}
        dark, lit, light = mine.bake_light([mesh], emissive)
        self.assertEqual([[0.0, 1.0, 2.0]], [[v["position"][0] for v in part["vertices"]] for part in lit])
        self.assertEqual([[3.0, 4.0, 5.0]], [[v["position"][0] for v in part["vertices"]] for part in dark])
        self.assertEqual([0, 1, 2], lit[0]["topology"]["indices"])
        # The lit face's uv0 patch holds its uv2 light; far from it stays black.
        self.assertTrue(np.allclose(light[34, 34], (.8, .5, .1)))
        self.assertTrue((light[5, 5] == 0).all() and (light[60, 10] == 0).all())
        # Without uv2 the light cannot be placed: the face stays dark.
        bare = {**mesh, "vertices": [{k: v for k, v in vertex.items() if k != "uv2"} for vertex in vertices]}
        dark, lit, light = mine.bake_light([bare], emissive)
        self.assertEqual(([], 1, 6), (lit, len(dark), len(dark[0]["topology"]["indices"])))
        self.assertFalse(light.any())

    @unittest.skipUnless((PACK / "manifest.json").is_file(), "local ImprovementsNormalized pack absent")
    def test_runtime_lights_only_civ6_lit_faces(self):
        # Production: Civ VI lights one doorway and a window on this building, about
        # 1% of its surface; the uv0 sampling lit 17.6% of it, mostly timber.
        import numpy as np
        from Renderer.tools.asset_compiler import build_mine_runtime as mine
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
        _version, texture_count, asset_count, _groups = take("<IIII")
        textures = [text() for _ in range(texture_count)]
        total = glowing = 0.0
        brightest = []
        for _ in range(asset_count):
            code = int(text().rsplit(":e", 1)[1])
            _texture, vertex_count, index_count = take("<III")
            vertices = np.array([take("<8f") for _ in range(vertex_count)])
            triangles = vertices[np.array(take(f"<{index_count}I")).reshape(-1, 3)]
            area = np.linalg.norm(np.cross(triangles[:, 1, :3] - triangles[:, 0, :3],
                                           triangles[:, 2, :3] - triangles[:, 0, :3]), axis=1) / 2
            total += area.sum()
            if code:
                # The runtime samples the emissive slot by uv0 (feature.hlsl).
                light = mine.bc1_rgb(PACK / textures[mine.BASE_SLOTS + code - 1])
                size = light.shape[0]
                uv = triangles[:, :, 6:8].mean(1)
                texels = light[(uv[:, 1] * size).astype(int) % size, (uv[:, 0] * size).astype(int) % size]
                brightest.append(texels.max())
                glowing += area.sum()
        self.assertLess(glowing / total, .02)
        # And the doorway does glow.
        self.assertGreater(min(brightest, default=0), .5)


if __name__ == "__main__":
    unittest.main()
