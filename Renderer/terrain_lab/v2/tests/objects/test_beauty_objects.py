import importlib.util
import json
import struct
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[5]
V2 = ROOT / "Renderer/terrain_lab/v2"


class BeautyObjectContractTests(unittest.TestCase):
    def test_tree_fixture_stays_isolated_and_backend_neutral(self):
        fixture = json.loads(
            (V2 / "fixtures/objects/beauty-trees.fixture.json").read_text()
        )
        self.assertEqual(["vegetation", "material", "lighting"], fixture["isolations"])
        self.assertEqual(["civ6.forest"], fixture["references"])
        self.assertEqual(4, fixture["settings"]["samples"])
        source = (V2 / "systems/objects/beauty_objects.cpp").read_text()
        self.assertNotIn("#import <Metal", source)
        self.assertNotIn("#include <d3d11.h>", source)
        self.assertIn("forest recipe has no authored weight", source)
        self.assertIn("4.5f / 6.2f", source)
        self.assertIn("3.5f / 6.2f", source)

    def test_foliage_shader_uses_source_opacity_and_does_not_guess_lean(self):
        shader = (V2 / "shaders/objects/beauty_objects.hlsl").read_text()
        source = (V2 / "systems/objects/beauty_objects.cpp").read_text()
        self.assertIn("Texture2D Opacity : register(t9);", shader)
        self.assertIn("clip(Opacity.Sample(Wrap, input.uv).r - 0.5);", shader)
        self.assertIn("projected[corner].uv[0] = source.uv[0]", source)
        self.assertIn("shadow_batches[placement.object]", source)
        self.assertIn("object.kind == 1 || object.kind == 3", source)

    def test_warrior_uses_authored_normals_and_source_repeat_addressing(self):
        pack = ROOT / "Renderer/packs/UnitWarriorLab"
        if not (pack / "manifest.json").exists():
            self.skipTest("local licensed-source Warrior pack is not installed")
        manifest = json.loads((pack / "manifest.json").read_text())
        for role in ("body", "head", "armor"):
            component = json.loads((
                pack / manifest["assets"][f"unit/warrior/{role}"]["component"]
            ).read_text())
            mesh = json.loads((pack / component["mesh"]).read_text())
            material = json.loads((pack / component["material"]).read_text())
            self.assertEqual(
                "authored_octahedral_snorm8", mesh["provenance"]["normal_source"]
            )
            self.assertEqual("repeat", material["channels"]["base_color"]["address_u"])
            self.assertEqual("repeat", material["channels"]["base_color"]["address_v"])
        shader = (V2 / "shaders/objects/beauty_objects.hlsl").read_text()
        self.assertIn("foliage || repeat_address", shader)

    def test_local_civ5_pack_covers_full_forest_recipe_when_present(self):
        pack = ROOT / "Renderer/packs/Civ5EnvironmentVegetation"
        if not (pack / "manifest.json").exists():
            self.skipTest("local licensed-source pack is not installed")
        manifest = json.loads((pack / "manifest.json").read_text())
        forest = manifest["features"]["forest"]
        self.assertEqual(22, len(forest["variants"]))
        self.assertEqual(25, len(forest["placements"]))
        self.assertEqual(180, sum(item["count"] for item in forest["placements"]))
        for asset_id in forest["variants"]:
            mesh = json.loads((pack / manifest["assets"][asset_id]["mesh"]).read_text())
            self.assertEqual(
                "authored_octahedral_snorm8", mesh["provenance"]["normal_source"]
            )
            if "leafy" in asset_id:
                material = json.loads(
                    (pack / manifest["assets"][asset_id]["material"]).read_text()
                )
                self.assertEqual("mask", material["alpha_mode"])
                self.assertIn("opacity", material)

    def test_lut_sampler_trilinearly_preserves_identity_cube(self):
        path = V2 / "tools/apply_3d_lut_to_bmp.py"
        spec = importlib.util.spec_from_file_location("apply_3d_lut_to_bmp", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        lut = bytearray()
        for blue in (0, 255):
            for green in (0, 255):
                for red in (0, 255):
                    lut.extend((red, green, blue, 255))
        self.assertEqual((64, 128, 192), module.sample_lut(2, bytes(lut), (64, 128, 192)))

    def test_local_beauty_bundle_keeps_recipe_table_when_present(self):
        path = ROOT / "Renderer/packs/BeautyStudies/beauty_objects.bin"
        if not path.exists():
            self.skipTest("local beauty-study bundle is not built")
        data = path.read_bytes()
        self.assertEqual(b"C3XBTO1\0", data[:8])
        version, materials, objects, recipes = struct.unpack_from("<IIII", data, 8)
        self.assertEqual(3, version)
        self.assertEqual(28, materials)
        self.assertEqual(31, objects)
        self.assertEqual(25, recipes)


if __name__ == "__main__":
    unittest.main()
