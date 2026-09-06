from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from Renderer.tools import lab_v2


class LabV2CampaignTests(unittest.TestCase):
    def test_repository_campaign_is_valid(self) -> None:
        self.assertEqual([], lab_v2.validate_campaign())

    def test_prompt_combines_common_and_owned_role(self) -> None:
        prompt = lab_v2.render_prompt("Q3-hydrology")
        self.assertIn("persistent owner", prompt)
        self.assertIn("Coast, shallow-water, ocean, and river owner", prompt)
        self.assertIn("Immutable v1 rules", prompt)

    def _copy_campaign(self, temporary: str) -> Path:
        source = lab_v2.DEFAULT_CAMPAIGN.parents[2]
        target = Path(temporary) / "v2"
        shutil.copytree(source, target)
        return target / "campaigns/Q1/campaign.json"

    def test_rejects_overlapping_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign_path = self._copy_campaign(temporary)
            package_path = campaign_path.parent / "work_packages/Q2-terrain.json"
            package = json.loads(package_path.read_text(encoding="utf-8"))
            package["owns_paths"][0] = "Renderer/terrain_lab/v2/systems/sampling/"
            package_path.write_text(json.dumps(package), encoding="utf-8")
            errors = lab_v2.validate_campaign(campaign_path)
        self.assertTrue(any("Owned paths overlap" in error for error in errors))

    def test_rejects_unknown_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign_path = self._copy_campaign(temporary)
            package_path = campaign_path.parent / "work_packages/Q5-networks.json"
            package = json.loads(package_path.read_text(encoding="utf-8"))
            package["references"] = ["missing.reference"]
            package_path.write_text(json.dumps(package), encoding="utf-8")
            errors = lab_v2.validate_campaign(campaign_path)
        self.assertTrue(any("unknown reference" in error for error in errors))

    def test_rejects_incomplete_accepted_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign_path = self._copy_campaign(temporary)
            status_path = campaign_path.parent / "status/Q1-sampling.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status["state"] = "accepted"
            status["blockers"] = []
            status_path.write_text(json.dumps(status), encoding="utf-8")
            errors = lab_v2.validate_campaign(campaign_path)
        self.assertTrue(any("candidate, evidence, and approval" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

