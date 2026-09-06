from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
        campaign, packages, statuses = lab_v2.load_campaign()
        campaign_dir = lab_v2.DEFAULT_CAMPAIGN.parent
        # These are metadata tests, not renderer/asset tests. Never recursively
        # copy v2: runtime caches contain large hard-linked blobs, and copytree
        # expands every link into a separate file in the temporary directory.
        files = {lab_v2.DEFAULT_CAMPAIGN,
                 campaign_dir / campaign["reference_catalog"],
                 campaign_dir / campaign["common_prompt"]}
        for package in packages.values():
            files.add(package["_path"])
            files.add(package["_path"].parent / package["prompt"])
        files.update(status["_path"] for status in statuses.values())
        for field in ("visual_benchmark_policy", "source_art_policy",
                      "placement_clearance_policy"):
            if field in campaign:
                files.add(campaign_dir / campaign[field])
        for path in files:
            destination = target / path.resolve().relative_to(source.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        return target / "campaigns/Q1/campaign.json"

    def test_campaign_copy_is_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(shutil, "copytree", side_effect=AssertionError(
                    "Campaign tests must not copy runtime trees")):
                campaign_path = self._copy_campaign(temporary)
            self.assertEqual([], lab_v2.validate_campaign(campaign_path))
            copied = [p for p in (Path(temporary) / "v2").rglob("*") if p.is_file()]
            self.assertTrue(copied)
            self.assertTrue(all(p.suffix in (".json", ".md") for p in copied))
            self.assertLess(sum(p.stat().st_size for p in copied), 2 * 1024 * 1024)
            self.assertFalse((Path(temporary) / "v2/app").exists())
            self.assertFalse((Path(temporary) / "v2/audits").exists())

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
