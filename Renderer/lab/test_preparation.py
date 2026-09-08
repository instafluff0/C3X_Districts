"""Source edits must reach production previews, without overwriting local work."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from Renderer.lab import preparation as prep
from Renderer.lab import asset_preparation as assets


class PreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "input.hlsl"
        self.target = self.root / "output.hlsl"
        self.source.write_text("original")
        self.target.write_text("bound original")
        replacement = patch.object(prep, "input_paths", return_value=["input.hlsl"])
        replacement.start(); self.addCleanup(replacement.stop)
        replacement = patch.object(prep, "generate", side_effect=self.generate)
        self.generate_mock = replacement.start(); self.addCleanup(replacement.stop)

    def generate(self, mirror):
        (mirror / "output.hlsl").write_text("bound " + (mirror / "input.hlsl").read_text())

    def test_bootstrap_proves_existing_output_and_warm_run_skips_generation(self):
        self.assertEqual(prep.prepare(self.root), [])
        prep.require_current(self.root)
        self.assertEqual(prep.prepare(self.root), [])
        self.assertEqual(self.generate_mock.call_count, 1)

    def test_source_change_rejects_stale_review_then_refreshes_output(self):
        prep.prepare(self.root)
        self.source.write_text("new source")
        with self.assertRaisesRegex(ValueError, "stale"):
            prep.require_current(self.root)
        self.assertEqual(prep.prepare(self.root), ["output.hlsl"])
        self.assertEqual(self.target.read_text(), "bound new source")
        prep.require_current(self.root)

    def test_missing_output_is_regenerated(self):
        prep.prepare(self.root)
        self.target.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            prep.require_current(self.root)
        self.assertEqual(prep.prepare(self.root), ["output.hlsl"])
        prep.require_current(self.root)

    def test_generated_edits_are_preserved_even_with_changed_source(self):
        prep.prepare(self.root)
        self.target.write_text("user change")
        self.source.write_text("another source")
        with self.assertRaisesRegex(ValueError, "Preserving edited"):
            prep.prepare(self.root)
        self.assertEqual(self.target.read_text(), "user change")

    def test_cache_loss_does_not_authorize_overwriting_unverified_output(self):
        self.target.write_text("unverified")
        with self.assertRaisesRegex(ValueError, "Preserving unverified"):
            prep.prepare(self.root)
        self.assertEqual(self.target.read_text(), "unverified")
        self.assertFalse((self.root / prep.RECEIPT).exists())

    def test_generator_failure_does_not_publish_partial_results(self):
        def fail(mirror):
            self.generate(mirror)
            raise ValueError("adapter failed")
        self.generate_mock.side_effect = fail
        with self.assertRaisesRegex(ValueError, "adapter failed"):
            prep.prepare(self.root)
        self.assertEqual(self.target.read_text(), "bound original")
        self.assertFalse((self.root / prep.RECEIPT).exists())
        self.assertEqual(list((self.root / "Renderer/lab/.cache").iterdir()), [])

    def test_concurrent_source_change_cannot_publish_a_receipt(self):
        def changed(mirror):
            self.generate(mirror)
            self.source.write_text("concurrent edit")
        self.generate_mock.side_effect = changed
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            prep.prepare(self.root)
        self.assertFalse((self.root / prep.RECEIPT).exists())

    def test_all_outputs_are_validated_before_any_replacement(self):
        prep.prepare(self.root)
        (self.root / "other.hlsl").write_text("unverified user edit")
        self.source.write_text("next")
        def two(mirror):
            self.generate(mirror)
            (mirror / "other.hlsl").write_text("new generated")
        self.generate_mock.side_effect = two
        with self.assertRaisesRegex(ValueError, "Preserving unverified"):
            prep.prepare(self.root)
        self.assertEqual(self.target.read_text(), "bound original")

    def test_path_escape_is_rejected(self):
        with patch.object(prep, "input_paths", return_value=["../outside"]):
            with self.assertRaisesRegex(ValueError, "escapes"):
                prep.prepare(self.root)


class ProductionBindingsTests(unittest.TestCase):
    def test_actual_adapters_reproduce_baseline_and_propagate_an_isolated_edit(self):
        # No licensed pack or VM is required. All source edits and generated
        # files in this test stay in an isolated disposable source checkout.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in prep.input_paths(prep.ROOT):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(prep.ROOT / name, target)
            prep.prepare(root)
            for name, expected in prep.receipt(root)["outputs"].items():
                self.assertEqual(prep.digest(prep.ROOT / name), expected, name)
            source = root / "Renderer/lab/shared/shaders/relief/beauty_terrain.hlsl"
            source.write_text("// isolated preparation test\n" + source.read_text())
            with self.assertRaisesRegex(ValueError, "stale"):
                prep.require_current(root)
            changed = prep.prepare(root)
            for family in ("source_fidelity", "environment_refresh", "city_fidelity"):
                name = f"Renderer/native/{family}/terrain.hlsl"
                self.assertIn(name, changed)
                self.assertIn("// isolated preparation test", (root / name).read_text())
            prep.require_current(root)
            self.assertFalse((root / "Renderer/packs").exists())


class AssetPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "input.dds"
        self.source.write_bytes(b"original")
        self.target = self.root / "runtime/texture.dds"
        self.target.parent.mkdir()
        self.target.write_bytes(b"original")
        for name in ("builder.py", "Renderer/lab/asset_preparation.py"):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("builder")
        self.calls = 0
        self.job = ("fixture", self.sources, self.build, "builder.py")

    def sources(self):
        path = self.root / "runtime/manifest.json"
        return json.loads(path.read_text()) if path.exists() else {"input.dds": prep.digest(self.source)}

    def build(self, stage):
        self.calls += 1
        consumed = {"input.dds": prep.digest(self.source)}
        data = self.source.read_bytes()
        if data == b"additional":
            path = self.root / "extra.dds"
            consumed["extra.dds"] = prep.digest(path)
            data += path.read_bytes()
        (stage / "runtime").mkdir()
        (stage / "runtime/texture.dds").write_bytes(data)
        (stage / "runtime/manifest.json").write_text(json.dumps(consumed))
        return consumed

    def refresh(self, check=False):
        return assets.refresh(self.job, root=self.root, check_only=check)

    def test_warm_build_is_skipped_and_source_edit_requires_preparation(self):
        self.refresh()
        self.refresh()
        self.assertEqual(self.calls, 1)
        self.source.write_bytes(b"edited source")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.refresh(check=True)
        self.refresh()
        self.assertEqual(self.target.read_bytes(), b"edited source")
        self.refresh(check=True)

    def test_hardlinked_runtime_output_is_detached_without_changing_content(self):
        self.target.unlink()
        os.link(self.source, self.target)
        self.refresh()
        self.assertFalse(self.target.samefile(self.source))
        self.source.write_bytes(b"source edit")
        self.assertEqual(self.target.read_bytes(), b"original")

    def test_missing_output_is_rebuilt(self):
        self.refresh()
        self.target.unlink()
        self.refresh()
        self.assertEqual(self.target.read_bytes(), b"original")

    def test_missing_generated_dependency_manifest_uses_saved_read_closure(self):
        self.refresh()
        manifest = self.root / "runtime/manifest.json"
        manifest.unlink()
        self.job = ("fixture", lambda: json.loads(manifest.read_text()), self.build, "builder.py")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.refresh(check=True)
        self.refresh()
        self.refresh(check=True)

    def test_deleted_dependency_can_be_retired_by_a_changed_source_catalog(self):
        self.refresh()
        extra = self.root / "extra.dds"
        extra.write_bytes(b"extra")
        self.source.write_bytes(b"additional")
        self.refresh()
        self.source.write_bytes(b"original")
        extra.unlink()
        self.refresh()
        self.refresh(check=True)
        self.assertNotIn("extra.dds", self.sources())

    def test_deleted_still_required_input_does_not_publish_a_pack(self):
        self.refresh()
        self.source.unlink()
        with self.assertRaises(FileNotFoundError):
            self.refresh()
        self.assertEqual(self.target.read_bytes(), b"original")

    def test_edited_generated_output_is_preserved(self):
        self.refresh()
        self.target.write_bytes(b"user edit")
        for check in (True, False):
            with self.assertRaisesRegex(ValueError, "Preserving edited"):
                self.refresh(check=check)
        self.assertEqual(self.target.read_bytes(), b"user edit")

    def test_cache_miss_does_not_overwrite_unknown_output(self):
        self.target.write_bytes(b"unknown output")
        with self.assertRaisesRegex(ValueError, "Preserving unverified"):
            self.refresh()
        self.assertFalse((self.root / "runtime/manifest.json").exists())

    def test_new_bundle_dependency_is_tracked_after_rebuilding(self):
        self.refresh()
        extra = self.root / "extra.dds"
        extra.write_bytes(b"extra")
        self.source.write_bytes(b"additional")
        self.refresh()
        self.refresh(check=True)
        extra.write_bytes(b"new extra")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.refresh(check=True)
        self.refresh()
        self.assertEqual(self.target.read_bytes(), b"additionalnew extra")

    def test_source_changed_during_build_is_not_published(self):
        def changed(stage):
            consumed = self.build(stage)
            self.source.write_bytes(b"concurrent edit")
            return consumed
        self.job = ("fixture", self.sources, changed, "builder.py")
        with self.assertRaisesRegex(ValueError, "sources changed"):
            self.refresh()
        self.assertEqual(self.target.read_bytes(), b"original")

    def test_builder_change_requires_fresh_preparation(self):
        self.refresh()
        (self.root / "builder.py").write_text("new builder")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.refresh(check=True)

    def test_absent_optional_catalog_is_a_tracked_dependency(self):
        optional = self.root / "optional.json"
        def observed(stage):
            consumed = self.build(stage)
            consumed["optional.json"] = prep.digest(optional) if optional.exists() else None
            (stage / "runtime/manifest.json").write_text(json.dumps(consumed))
            return consumed
        self.job = ("fixture", self.sources, observed, "builder.py")
        self.refresh()
        self.refresh(check=True)
        optional.write_text("{}")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.refresh(check=True)
        self.refresh()
        self.refresh(check=True)

    def test_output_symlink_cannot_overwrite_a_source(self):
        self.target.unlink()
        self.target.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "alias"):
            self.refresh()
        self.assertEqual(self.source.read_bytes(), b"original")


if __name__ == "__main__":
    unittest.main()
