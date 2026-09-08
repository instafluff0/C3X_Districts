import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
PICKUP = ROOT / "Renderer/handoffs/candidates/lab_v2_complete_r2"
V2 = ROOT / "Renderer/terrain_lab/v2"


class LabStateOfArtPickupTests(unittest.TestCase):
    def test_manifest_validates_all_isolated_witnesses(self):
        spec = importlib.util.spec_from_file_location(
            "validate_state_of_art", PICKUP / "validate_state_of_art.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(0, module.main())

    def test_source_fidelity_composition_is_selected_without_weakening_city_contract(self):
        state = json.loads((PICKUP / "LAB_STATE_OF_ART.json").read_text())
        composition = state["composition_contract"]
        self.assertTrue(composition["combined_scene_authoritative"])
        self.assertTrue(composition["city_excludes_trees"])
        self.assertFalse(composition["cities_rendered"])
        self.assertFalse(composition["cities_changed"])
        self.assertTrue(composition["one_face_and_cast_light_vector"])
        self.assertTrue(composition["canonical_world_projection_shared"])
        self.assertEqual(
            "Renderer/terrain_lab/v2/fixtures/objects/beauty-scene.fixture.json",
            composition["superseded_fixture"],
        )
        self.assertEqual(
            "Renderer/terrain_lab/v2/fixtures/beauty/source-fidelity-r2/inland/fixture.json",
            composition["selected_fixture"],
        )
        self.assertIn("cities", state["excluded_from_current_update"])
        self.assertNotIn("medieval_city", {row["id"] for row in state["studies"]})

    def test_composed_witness_routes_selected_high_definition_paths(self):
        state = json.loads((PICKUP / "LAB_STATE_OF_ART.json").read_text())
        composed = state["composed_witness"]
        fixture = json.loads((ROOT / composed["fixture"]).read_text())
        self.assertEqual(100, fixture["tile_count"])
        self.assertEqual(4, fixture["settings"]["samples"])
        self.assertEqual(16, fixture["settings"]["anisotropy"])
        self.assertEqual(2, fixture["settings"]["render_scale"])
        terrain = json.loads((ROOT / composed["terrain_module"]).read_text())
        self.assertEqual(
            "Renderer/terrain_lab/v2/systems/relief/beauty_terrain.cpp",
            terrain["source"],
        )
        mountain = json.loads((ROOT / composed["mountain_module"]).read_text())
        self.assertEqual(composed["mountain_source"], mountain["source"])
        forest = json.loads((ROOT / composed["forest_module"]).read_text())
        self.assertEqual(composed["forest_source"], forest["source"])
        self.assertEqual(composed["forest_shader"], forest["shader"])
        hydrology = json.loads((ROOT / composed["hydrology_module"]).read_text())
        self.assertEqual(1, hydrology["suppress_relief"])
        self.assertEqual(1, hydrology["river_corridor"])
        self.assertEqual(
            "Renderer/terrain_lab/v2/fixtures/beauty/source-fidelity-r2/inland-shadow-control/fixture.json",
            composed["shadow_control_fixture"],
        )

    def test_composed_shadow_direction_and_projection_are_single_source(self):
        terrain = (V2 / "shaders/relief/beauty_terrain.hlsl").read_text()
        mountain = (V2 / "shaders/relief/beauty_mountain.hlsl").read_text()
        forest = (V2 / "shaders/objects/beauty_objects.hlsl").read_text()
        for shader in (terrain, mountain, forest):
            self.assertIn("float3 light_direction = ShadowL.xyz;", shader)
            self.assertIn("q6_shadow_visibility", shader)
        mountain_source = (V2 / "systems/relief/beauty_mountain.cpp").read_text()
        self.assertIn(
            "float center_x = 40.0f + (world_x + world_y) * half_width;",
            mountain_source,
        )
        evidence = json.loads(
            (V2 / "audits/beauty/out/source-fidelity-r13/inland/shadow-evidence.json").read_text()
        )
        self.assertEqual("Q6 shadow receive flag only", evidence["control_difference"])
        self.assertEqual("up-right", evidence["noon_screen_cast_direction"])
        self.assertGreater(evidence["changed_pixels_delta_gt_6"], 20000)
        self.assertFalse(evidence["cities_rendered"])
        self.assertFalse(evidence["cities_changed"])

    def test_manifest_exposes_composed_case_and_updated_system_entries(self):
        manifest = json.loads((PICKUP / "manifest.json").read_text())
        cases = {row["id"]: row for row in manifest["cases"]}
        self.assertIn("source-fidelity-inland", cases)
        systems = {row["id"]: row for row in manifest["systems"]}
        self.assertIn("beauty_terrain", systems["terrain"]["selected"])
        self.assertIn("beauty_mountain", systems["terrain"]["selected"])
        self.assertIn("22 source bodies", systems["forest_jungle"]["selected"])
        self.assertIn("accepted 100-tile", systems["combined_scene"]["selected"])
        self.assertIn("unchanged", systems["cities"]["native_state"])

    def test_object_transform_preserves_proportions(self):
        source = (V2 / "systems/objects/beauty_objects.cpp").read_text()
        self.assertIn("local_x = (source.position[0]", source)
        self.assertIn("local_y = (source.position[0]", source)
        self.assertIn("local_z = source.position[2] * placement.scale", source)
        self.assertNotIn("placement.scale_z", source)
        builder = (V2 / "tools/build_beauty_objects.py").read_text()
        self.assertIn('"vertices": expanded_vertices(mesh)', builder)
        self.assertNotIn("height_scale", builder)

    def test_source_and_inference_language_remains_explicit(self):
        state = json.loads((PICKUP / "LAB_STATE_OF_ART.json").read_text())
        joined = " ".join(state["known_inferences_and_pending"]).lower()
        self.assertIn("inferred", joined)
        self.assertIn("not yet imported", joined)
        self.assertIn("does not fabricate", joined)


if __name__ == "__main__":
    unittest.main()
