import copy
import json
import unittest
from pathlib import Path

from Renderer.definitions import definition_parser
from Renderer.scenes import scene_contract


FIXTURE = Path("Renderer/samples/scenes/grassland_viewport.scene.json")
CONFIG = Path("Renderer/samples/config/default.custom_rendering.txt")


class VisibleSceneContractTests(unittest.TestCase):
    def setUp(self):
        self.scene = scene_contract.load_scene(FIXTURE)

    def test_recorded_fixture_has_required_scene_families(self):
        self.assertEqual(scene_contract.SCHEMA, self.scene["schema"])
        self.assertEqual(4, len(self.scene["tiles"]))
        self.assertEqual({"resource", "city", "unit"}, {item["category"] for item in self.scene["instances"]})
        self.assertEqual("civ3-isometric-pixel", self.scene["projection"]["type"])
        self.assertEqual({"id", "hour", "season"}, set(self.scene["environment"]))

    def test_canonical_round_trip_is_byte_stable(self):
        first = scene_contract.canonical_json(self.scene)
        reparsed = scene_contract.parse_scene_text(first)
        second = scene_contract.canonical_json(reparsed)
        self.assertEqual(first.encode("utf-8"), second.encode("utf-8"))

    def test_missing_unknown_and_malformed_fields_have_paths(self):
        broken = copy.deepcopy(self.scene)
        del broken["viewport"]["width_px"]
        broken["environment"]["hour"] = 24
        broken["tiles"][0]["mystery"] = True
        with self.assertRaises(scene_contract.SceneValidationError) as raised:
            scene_contract.validate_scene(broken)
        paths = {item.path for item in raised.exception.diagnostics}
        self.assertIn("$.viewport.width_px", paths)
        self.assertIn("$.environment.hour", paths)
        self.assertIn("$.tiles[0].mystery", paths)

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(scene_contract.SceneValidationError) as raised:
            scene_contract.parse_scene_text('{"schema":"a","schema":"b"}')
        self.assertIn("duplicate JSON key", str(raised.exception))

    def test_ids_seeds_coordinates_and_anchors_are_verified(self):
        cases = [
            ("id", lambda scene: scene["tiles"][0].__setitem__("id", "tile:wrong")),
            ("seed", lambda scene: scene["instances"][0].__setitem__("variant_seed", 1)),
            ("coordinates", lambda scene: scene["instances"][0]["resolver_input"].__setitem__("map_x", 99)),
            ("anchor", lambda scene: scene["tiles"][1]["anchor_px"].__setitem__("x", 999)),
        ]
        for label, mutate in cases:
            with self.subTest(label=label):
                broken = copy.deepcopy(self.scene)
                mutate(broken)
                with self.assertRaises(scene_contract.SceneValidationError):
                    scene_contract.validate_scene(broken)

    def test_process_pointers_and_source_specific_formats_are_rejected(self):
        pointer = copy.deepcopy(self.scene)
        pointer["tiles"][0]["terrain"]["resolver_input"]["native_pointer"] = "0x1234"
        with self.assertRaises(scene_contract.SceneValidationError) as raised_pointer:
            scene_contract.validate_scene(pointer)
        self.assertTrue(any("process-specific" in item.message for item in raised_pointer.exception.diagnostics))

        source_asset = copy.deepcopy(self.scene)
        source_asset["tiles"][0]["terrain"]["resolver_input"]["pcx_file"] = "TerrainMaterialSet_Base.blp"
        with self.assertRaises(scene_contract.SceneValidationError) as raised_asset:
            scene_contract.validate_scene(source_asset)
        self.assertTrue(any("source-specific" in item.message for item in raised_asset.exception.diagnostics))

    def test_item_environment_cannot_override_scene_environment(self):
        broken = copy.deepcopy(self.scene)
        broken["tiles"][0]["terrain"]["resolver_input"]["hour"] = 3
        with self.assertRaises(scene_contract.SceneValidationError) as raised:
            scene_contract.validate_scene(broken)
        self.assertTrue(any(item.path.endswith("resolver_input.hour") and item.message == "unknown field" for item in raised.exception.diagnostics))

    def test_offline_inspection_replays_every_item_through_the_resolver(self):
        definitions = definition_parser.parse_definition_file(CONFIG, "default", Path.cwd(), Path.cwd())
        catalog = definition_parser.merge_layers([("default", definitions)])
        inspection = scene_contract.inspect_scene(self.scene, catalog)
        self.assertEqual(scene_contract.INSPECTION_SCHEMA, inspection["schema"])
        self.assertEqual(7, len(inspection["items"]))
        self.assertEqual("terrain.grassland.sheet2.sprite0", inspection["items"][0]["resolution"]["winner"]["rule_id"])
        self.assertEqual("matched", inspection["items"][1]["resolution"]["status"])
        self.assertIsNotNone(inspection["items"][1]["resolution"]["winner"]["variant"])
        object_results = inspection["items"][4:]
        self.assertEqual({"category_owned_by_civ3"}, {item["resolution"]["fallback"]["reason"] for item in object_results})
        self.assertTrue(all(item["resolver_input"]["hour"] == 12 for item in inspection["items"]))
        self.assertTrue(all(item["resolver_input"]["season"] == "summer" for item in inspection["items"]))
        json.loads(scene_contract.canonical_json(inspection))

    def test_config_off_inspection_never_attempts_asset_access(self):
        definitions = definition_parser.parse_definition_file(CONFIG, "default", Path.cwd(), Path.cwd())
        catalog = definition_parser.merge_layers([("default", definitions)])
        inspection = scene_contract.inspect_scene(self.scene, catalog, enabled=False, available_assets=set())
        for item in inspection["items"]:
            resolution = item["resolution"]
            self.assertEqual("config_off", resolution["fallback"]["reason"])
            self.assertEqual(0, resolution["asset_availability_checks"])
            self.assertEqual(0, resolution["asset_payload_loads"])

    def test_scene_identifier_changes_with_replay_significant_state(self):
        original = scene_contract.scene_identifier(self.scene)
        changed = copy.deepcopy(self.scene)
        changed["environment"]["hour"] = 13
        self.assertNotEqual(original, scene_contract.scene_identifier(changed))


if __name__ == "__main__":
    unittest.main()
