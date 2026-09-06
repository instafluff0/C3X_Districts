from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler.unit_visual_calibration import (
    DEFAULT_COMPOUND_PACK,
    DEFAULT_FAMILY_PACK,
    validate_visual_contract,
)


class UnitVisualCalibrationTests(unittest.TestCase):
    def test_checked_in_contract_covers_eight_facings_two_zooms_and_basic_actions(self) -> None:
        report = validate_visual_contract(family_pack=None, compound_pack=None)
        self.assertEqual(8, report["facing_count"])
        self.assertEqual(2, report["zoom_count"])
        self.assertEqual(128, report["cells_per_basic_complete_unit"])
        self.assertEqual({}, report["family_pose_cache_gaps"])
        self.assertEqual({}, report["pose_cache_gaps"])
        self.assertEqual({}, report["unresolved_basic_action_gaps"])

    @unittest.skipUnless(
        (DEFAULT_FAMILY_PACK / "manifest.json").is_file()
        and (DEFAULT_COMPOUND_PACK / "manifest.json").is_file(),
        "local licensed-source unit packs unavailable",
    )
    def test_local_packs_have_caches_and_only_truthful_catapult_gap(self) -> None:
        report = validate_visual_contract()
        self.assertEqual({"catapult": ["death"]}, report["unresolved_basic_action_gaps"])
        self.assertEqual({}, report["family_pose_cache_gaps"])
        self.assertEqual({}, report["pose_cache_gaps"])
        self.assertEqual(1, report["units"]["horseman"]["semantic_bodies"])


if __name__ == "__main__":
    unittest.main()
