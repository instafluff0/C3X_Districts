import copy
import tempfile
import unittest
from pathlib import Path

from Renderer.inventory import wonder_district_mapping_inventory as inventory


class WonderDistrictMappingInventoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wonders = inventory.load_json(inventory.DEFAULT_WONDER_MAPPING)
        self.districts = inventory.load_json(inventory.DEFAULT_DISTRICT_MAPPING)
        self.wonder_roster = inventory.load_json(inventory.DEFAULT_WONDER_ROSTER)

    def test_wonder_seed_matches_config_and_biq(self) -> None:
        config = inventory.parse_blocks(inventory.DEFAULT_WONDER_CONFIG, "#Wonder")
        semantics = inventory.load_json(inventory.DEFAULT_BIQ_SEMANTICS)
        self.assertEqual(inventory.validate_wonder_mapping(self.wonders), [])
        self.assertEqual(inventory.validate_wonders_against_sources(self.wonders, config, semantics), [])
        self.assertEqual(len(self.wonders["mappings"]), 23)
        self.assertIn("not_map_rendered", self.wonders["policy"]["placement"])

    def test_biq_roster_classifies_all_configured_and_mapless_wonders(self) -> None:
        semantics = inventory.load_json(inventory.DEFAULT_BIQ_SEMANTICS)
        self.assertEqual(
            inventory.validate_wonder_roster(self.wonder_roster, semantics, self.wonders), []
        )
        self.assertEqual(self.wonder_roster["counts"], {
            "total": 40, "great": 29, "small": 11,
            "c3x_configured": 23, "not_map_rendered": 17,
        })
        mapless = {item["civ3_id"]: item for item in self.wonder_roster["wonders"] if item["map_status"] == "not_map_rendered"}
        self.assertEqual(mapless["BLDG_Great_Wall"]["source_seed"]["civ6_artdef"], "IMPROVEMENT_GREAT_WALL")
        self.assertEqual(mapless["BLDG_Manhattan_Project"]["source_seed"]["mapping_status"], "authored_required")

    def test_wonder_classes_directions_and_authored_gap_are_explicit(self) -> None:
        by_key = {item["c3x_key"]: item for item in self.wonders["mappings"]}
        self.assertEqual(by_key["pyramids"]["wonder_class"], "great")
        self.assertEqual(by_key["apollo_program"]["wonder_class"], "small")
        self.assertIsNotNone(by_key["colossus"]["native_art"]["alternate"])
        self.assertTrue(by_key["hoover_dam"]["buildable_on_rivers"])
        self.assertEqual(by_key["pentagon"]["mapping_status"], "authored_required")

    def test_district_seed_matches_config_and_dependent_buildings(self) -> None:
        config = inventory.parse_blocks(inventory.DEFAULT_DISTRICT_CONFIG, "#District")
        self.assertEqual(inventory.validate_district_mapping(self.districts), [])
        self.assertEqual(inventory.validate_districts_against_config(self.districts, config), [])
        self.assertEqual(len(self.districts["mappings"]), 21)

    def test_district_composition_keeps_base_attachments_and_delegation_separate(self) -> None:
        by_key = {item["c3x_key"]: item for item in self.districts["mappings"]}
        self.assertEqual(by_key["campus"]["civ6_artdef"], "DISTRICT_CAMPUS")
        self.assertEqual(
            [item["civ3_name"] for item in by_key["campus"]["attachments"]],
            ["Library", "University"],
        )
        self.assertEqual(by_key["wonder_district"]["special_family"], "wonder_delegate")
        self.assertEqual(by_key["bridge"]["mapping_status"], "authored_required")
        self.assertEqual(by_key["great_wall"]["special_family"], "connection_topology")

    def test_validator_rejects_partial_fallback_and_missing_attachment(self) -> None:
        broken = copy.deepcopy(self.districts)
        broken["mappings"][0]["fallback"] = "partial"
        broken["mappings"][2]["attachments"].pop()
        errors = inventory.validate_district_mapping(broken)
        errors.extend(inventory.validate_districts_against_config(
            broken, inventory.parse_blocks(inventory.DEFAULT_DISTRICT_CONFIG, "#District")
        ))
        self.assertTrue(any("fallback" in error for error in errors))
        self.assertTrue(any("attachment order" in error for error in errors))

    def test_resolution_separates_installed_and_authored_targets(self) -> None:
        wonders = copy.deepcopy(self.wonders)
        districts = copy.deepcopy(self.districts)
        wonders["mappings"] = [wonders["mappings"][0], wonders["mappings"][19]]
        districts["mappings"] = [districts["mappings"][2], districts["mappings"][18]]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            building = root / "Base/ArtDefs/Buildings.artdef"
            district = root / "Base/ArtDefs/Districts.artdef"
            building.parent.mkdir(parents=True)
            building.write_text(
                '<Root><m_Name text="BUILDING_PYRAMIDS"/>'
                '<m_Name text="BUILDING_LIBRARY"/><m_Name text="BUILDING_UNIVERSITY"/></Root>',
                encoding="utf-8",
            )
            district.write_text('<Root><m_Name text="DISTRICT_CAMPUS"/></Root>', encoding="utf-8")
            report = inventory.build_resolution_report(wonders, districts, root)
        self.assertEqual(report["summary"]["resolved_count"], 4)
        self.assertEqual(report["summary"]["authored_required_count"], 2)
        self.assertEqual(report["unavailable"], [])


if __name__ == "__main__":
    unittest.main()
