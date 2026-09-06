import copy
import tempfile
import unittest
from pathlib import Path

from Renderer.inventory import resource_mapping_inventory as inventory


class ResourceMappingInventoryTests(unittest.TestCase):
    def test_repository_seed_matches_all_vanilla_biq_resources(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        semantics = inventory.load_json(inventory.DEFAULT_BIQ_SEMANTICS)

        self.assertEqual(inventory.validate_mapping(mapping), [])
        self.assertEqual(inventory.validate_against_biq(mapping, semantics), [])
        self.assertEqual(len(mapping["mappings"]), 26)
        self.assertIn("map resource body", mapping["policy"]["replacement_ownership"])
        self.assertIn("Civilopedia", mapping["policy"]["retained_ownership"])

    def test_special_semantic_matches_and_rubber_stand_in_are_explicit(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        by_id = {item["civ3_id"]: item for item in mapping["mappings"]}

        self.assertEqual(by_id["GOOD_Saltpeter"]["civ6_artdef"], "RESOURCE_NITER")
        self.assertEqual(by_id["GOOD_Game"]["civ6_artdef"], "RESOURCE_DEER")
        self.assertEqual(by_id["GOOD_Diamonds"]["civ6_artdef"], "RESOURCE_DIAMONDS")
        self.assertEqual(by_id["GOOD_Bananas"]["civ6_artdef"], "RESOURCE_BANANAS")
        self.assertEqual(by_id["GOOD_Oasis"]["civ6_artdef"], "FEATURE_OASIS")
        self.assertEqual(by_id["GOOD_Rubber"]["match"], "stand_in")
        self.assertEqual(by_id["GOOD_Rubber"]["confidence"], "low")

    def test_validator_rejects_non_map_fallback_and_duplicate_source(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        broken = copy.deepcopy(mapping)
        broken["mappings"][0]["fallback"] = "none"
        broken["mappings"][1]["civ3_id"] = broken["mappings"][0]["civ3_id"]

        errors = inventory.validate_mapping(broken)
        self.assertTrue(any("fallback" in error for error in errors))
        self.assertTrue(any("duplicate Civ III ID" in error for error in errors))

    def test_discovery_resolves_resource_and_feature_targets(self) -> None:
        mapping = inventory.load_json(inventory.DEFAULT_MAPPING)
        small = copy.deepcopy(mapping)
        small["source_roster"]["resource_count"] = 2
        small["mappings"] = [
            next(item for item in mapping["mappings"] if item["civ3_id"] == "GOOD_Horses"),
            next(item for item in mapping["mappings"] if item["civ3_id"] == "GOOD_Oasis"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artdef = root / "Base/ArtDefs/Resources.artdef"
            artdef.parent.mkdir(parents=True)
            artdef.write_text(
                '<Root><m_Name text="RESOURCE_HORSES"/><m_Name text="FEATURE_OASIS"/></Root>',
                encoding="utf-8",
            )
            report = inventory.build_resolution_report(small, root)

        self.assertEqual(report["summary"]["resolved_count"], 2)
        self.assertEqual(report["unavailable"], [])


if __name__ == "__main__":
    unittest.main()
