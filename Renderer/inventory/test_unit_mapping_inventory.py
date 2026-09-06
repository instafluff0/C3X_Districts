import json
from pathlib import Path
import tempfile
import unittest

from Renderer.inventory.unit_mapping_inventory import (
    DEFAULT_MAPPING,
    build_resolution_report,
    load_mapping,
    validate_mapping,
)


class UnitMappingInventoryTests(unittest.TestCase):
    def test_checked_in_mapping_is_complete_and_valid(self):
        document = load_mapping()

        self.assertEqual([], validate_mapping(document))
        self.assertEqual(93, len(document["mappings"]))
        by_id = {entry["civ3_id"]: entry for entry in document["mappings"]}
        self.assertEqual("UNIT_WARRIOR", by_id["PRTO_Warrior"]["civ6_artdef"])
        self.assertEqual("UNIT_KOREAN_HWACHA", by_id["PRTO_Hwacha"]["civ6_artdef"])
        self.assertEqual("vanilla", by_id["PRTO_ICBM"]["fallback"])

    def test_duplicate_civ3_ids_are_rejected(self):
        document = load_mapping()
        document = json.loads(json.dumps(document))
        document["mappings"][1]["civ3_id"] = document["mappings"][0]["civ3_id"]

        self.assertTrue(any("duplicate Civ III ID" in error for error in validate_mapping(document)))

    def test_artdef_resolution_reports_resolved_unavailable_and_deferred(self):
        document = {
            "schema": "c3x.civ3_to_civ6_unit_mapping.v0",
            "mappings": [
                {"civ3_id": "PRTO_Warrior", "civ3_name": "Warrior", "civ6_artdef": "UNIT_WARRIOR"},
                {"civ3_id": "PRTO_Archer", "civ3_name": "Archer", "civ6_artdef": "UNIT_ARCHER"},
                {"civ3_id": "PRTO_ICBM", "civ3_name": "ICBM", "civ6_artdef": None},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Units.artdef").write_text(
                '<AssetObjects..ArtDefSet><m_Name text="UNIT_WARRIOR"/></AssetObjects..ArtDefSet>',
                encoding="utf-8",
            )
            report = build_resolution_report(document, root)

        self.assertEqual(1, report["summary"]["resolved_count"])
        self.assertEqual(1, report["summary"]["unavailable_count"])
        self.assertEqual(1, report["summary"]["deferred_count"])
        self.assertEqual("UNIT_ARCHER", report["unavailable"][0]["civ6_artdef"])


if __name__ == "__main__":
    unittest.main()
