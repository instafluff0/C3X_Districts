"""Protect explicit approvals and the dependency-selected visual baseline."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Renderer import renderer


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.lab = self.root / "Renderer/lab"
        for name, value in (("ROOT", self.root), ("LAB", self.lab)):
            mock = patch.object(renderer, name, value)
            mock.start(); self.addCleanup(mock.stop)
        mock = patch.object(renderer, "implementation_identity", return_value="current")
        mock.start(); self.addCleanup(mock.stop)
        renderer.write(self.lab / "catalog.json", {"categories": {"lighting": "lighting", "grassland": "grassland"}})
        for key in ("lighting", "grassland"):
            renderer.write(renderer.standard_path(key), {
                "id": key, "revision": 1, "approved_revision": 1,
                "depends_on": ["lighting"] if key == "grassland" else [],
                "implementation": [], "references": {},
                "recipe": {"cases": ["detail", "gameplay"], "hours": [12], "zooms": [128]},
            })

    def preview(self, key, *, stale=False, partial=False):
        records = []
        for case in ("detail",) if partial else ("detail", "gameplay"):
            image = self.lab / "out" / key / (case + ".bmp")
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"deterministic-preview-" + case.encode())
            records.append({"case": case, "hour": 12, "zoom": 128,
                            "image": renderer.relative(image), "sha256": renderer.checksum(image)})
        renderer.write(self.lab / "out" / key / "render.json", {
            "implementation_identity": "stale" if stale else "current",
            "recipe": renderer.standard(key)["recipe"], "outputs": records,
        })

    def test_shared_change_requires_every_affected_preview(self):
        self.preview("lighting")
        with self.assertRaisesRegex(ValueError, "affected category"):
            renderer.approve("lighting", "User approved this appearance")
        self.assertEqual(renderer.standard("lighting")["approved_revision"], 1)
        self.assertFalse((self.lab / "references").exists())

    def test_stale_or_partial_preview_cannot_be_approved(self):
        for args, error in (({"stale": True}, "stale"), ({"partial": True}, "complete category")):
            self.preview("grassland", **args)
            with self.assertRaisesRegex(ValueError, error):
                renderer.approve("grassland", "User approved")
            self.assertEqual(renderer.standard("grassland")["approved_revision"], 1)

    def test_modified_candidate_cannot_be_approved(self):
        self.preview("grassland")
        (self.lab / "out/grassland/detail.bmp").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "image changed"):
            renderer.approve("grassland", "User approved")

    def test_shared_approval_creates_new_revisions_and_retains_prior_images(self):
        old = self.lab / "references/grassland/r1/detail.bmp"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"original")
        for key in ("lighting", "grassland"):
            self.preview(key)
        renderer.approve("lighting", "The user explicitly accepted both comparisons")
        self.assertEqual(old.read_bytes(), b"original")
        for key in ("lighting", "grassland"):
            value = renderer.standard(key)
            self.assertEqual(value["approved_revision"], 2)
            self.assertEqual(len(value["references"]["d3d11"]), 2)
            self.assertEqual(value["approval"]["affected_categories"], ["grassland", "lighting"])

    def test_paths_cannot_escape_repository(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            renderer.local("../outside")


if __name__ == "__main__":
    unittest.main()
