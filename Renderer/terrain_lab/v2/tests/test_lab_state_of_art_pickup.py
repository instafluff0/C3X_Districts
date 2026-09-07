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

    def test_combined_scene_is_explicitly_superseded(self):
        state = json.loads((PICKUP / "LAB_STATE_OF_ART.json").read_text())
        composition = state["composition_contract"]
        self.assertFalse(composition["combined_scene_authoritative"])
        self.assertTrue(composition["city_excludes_trees"])
        self.assertEqual(
            "Renderer/terrain_lab/v2/fixtures/objects/beauty-scene.fixture.json",
            composition["superseded_fixture"],
        )
        self.assertIn("cities", state["excluded_from_current_update"])
        self.assertNotIn("medieval_city", {row["id"] for row in state["studies"]})

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
