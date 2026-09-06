import copy
import tempfile
import unittest
from pathlib import Path

from Renderer.inventory import natural_wonder_mapping_inventory as inventory


class NaturalWonderMappingInventoryTests(unittest.TestCase):
    def test_repository_seed_matches_all_default_c3x_definitions(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        definitions = inventory.parse_natural_wonder_config(inventory.DEFAULT_CONFIG)

        self.assertEqual(inventory.validate_mapping(mapping), [])
        self.assertEqual(inventory.validate_against_default_config(mapping, definitions), [])
        self.assertEqual(len(mapping["mappings"]), 18)
        self.assertEqual(mapping["source_roster"]["maximum_runtime_definitions"], 64)
        self.assertIn("fog and shroud", mapping["policy"]["retained_ownership"])
        self.assertIn("name label", mapping["policy"]["retained_ownership"])

    def test_exact_approximate_and_authored_choices_are_explicit(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        by_key = {item["c3x_key"]: item for item in mapping["mappings"]}

        self.assertEqual(by_key["yosemite"]["civ6_artdef"], "FEATURE_YOSEMITE")
        self.assertEqual(by_key["mount_everest"]["mapping_status"], "exact")
        self.assertEqual(by_key["yellowstone"]["mapping_status"], "approximate")
        self.assertEqual(by_key["geirangerfjord"]["civ6_artdef"], "FEATURE_LYSEFJORDEN")
        self.assertEqual(by_key["savanna"]["mapping_status"], "authored_required")
        self.assertIsNone(by_key["savanna"]["civ6_artdef"])

    def test_config_parser_preserves_animation_count_and_direction(self) -> None:
        definitions = inventory.parse_natural_wonder_config(inventory.DEFAULT_CONFIG)
        by_name = {item["name"]: item for item in definitions}

        self.assertEqual(len(by_name["Angel Falls"]["animations"]), 2)
        self.assertEqual(by_name["Angel Falls"]["adjacency_dir"], "southeast")
        self.assertEqual(len(by_name["Yellowstone"]["animations"]), 2)
        self.assertEqual(by_name["Geirangerfjord"]["adjacency_dir"], "south")

    def test_validator_rejects_partial_fallback_and_fake_authored_target(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        broken = copy.deepcopy(mapping)
        broken["mappings"][0]["fallback"] = "partial"
        broken["mappings"][-1]["civ6_artdef"] = "FEATURE_PANTANAL"

        errors = inventory.validate_mapping(broken)
        self.assertTrue(any("fallback" in error for error in errors))
        self.assertTrue(any("authored_required" in error for error in errors))

    def test_discovery_resolves_feature_definitions_and_separates_authored(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        small = copy.deepcopy(mapping)
        small["source_roster"]["definition_count"] = 2
        small["mappings"] = [mapping["mappings"][1], mapping["mappings"][-1]]
        small["mappings"][0]["c3x_default_index"] = 0
        small["mappings"][1]["c3x_default_index"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artdef = root / "Base/ArtDefs/Features.artdef"
            artdef.parent.mkdir(parents=True)
            artdef.write_text('<Root><m_Name text="FEATURE_YOSEMITE"/></Root>', encoding="utf-8")
            report = inventory.build_resolution_report(small, root)

        self.assertEqual(report["summary"]["resolved_count"], 1)
        self.assertEqual(report["summary"]["authored_required_count"], 1)
        self.assertEqual(report["unavailable"], [])


if __name__ == "__main__":
    unittest.main()
