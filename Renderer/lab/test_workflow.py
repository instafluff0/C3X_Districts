"""Protect explicit approvals and the dependency-selected visual baseline."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Renderer import renderer


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        quiet = patch("builtins.print")
        quiet.start(); self.addCleanup(quiet.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.lab = self.root / "Renderer/lab"
        for name, value in (("ROOT", self.root), ("LAB", self.lab)):
            mock = patch.object(renderer, name, value)
            mock.start(); self.addCleanup(mock.stop)
        mock = patch.object(renderer, "implementation_identity", return_value="current")
        mock.start(); self.addCleanup(mock.stop)
        mock = patch.object(renderer, "dirty_categories", return_value=[])
        mock.start(); self.addCleanup(mock.stop)
        mock = patch.object(renderer, "category_signatures", return_value={"lighting": "current", "grassland": "current"})
        mock.start(); self.addCleanup(mock.stop)
        for name in ("require_prepared", "prepare_sources", "ensure_candidate"):
            mock = patch.object(renderer, name)
            mock.start(); self.addCleanup(mock.stop)
        renderer.write(self.lab / "catalog.json", {"categories": {"lighting": "lighting", "grassland": "grassland"}})
        for key in ("lighting", "grassland"):
            renderer.write(renderer.standard_path(key), {
                "id": key, "revision": 1, "approved_revision": 1,
                "depends_on": ["lighting"] if key == "grassland" else [],
                "implementation": [], "references": {},
                "tests": ["Renderer.native.test_scroll_damage"],
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
            "input_signature": "current",
            "recipe": renderer.standard(key)["recipe"], "outputs": records,
        })

    def test_shared_change_requires_every_affected_preview(self):
        self.preview("lighting")
        with self.assertRaisesRegex(ValueError, "affected category"):
            renderer.approve("lighting", "User approved this appearance")
        self.assertEqual(renderer.standard("lighting")["approved_revision"], 1)
        self.assertFalse((self.lab / "references").exists())

    def test_blank_statement_is_not_approval(self):
        self.preview("grassland")
        with self.assertRaisesRegex(ValueError, "explicit approval"):
            renderer.approve("grassland", "  ")
        self.assertEqual(renderer.standard("grassland")["approved_revision"], 1)

    def test_unprepared_source_cannot_be_rendered_compared_or_approved(self):
        self.preview("grassland")
        with patch.object(renderer, "require_prepared", side_effect=ValueError("bindings are stale")), \
             patch.object(renderer, "native_render") as draw:
            for action in (lambda: renderer.render("grassland"),
                           lambda: renderer.compare("grassland"),
                           lambda: renderer.approve("grassland", "User approved"),
                           lambda: renderer.integration_receipt("grassland")):
                with self.assertRaisesRegex(ValueError, "bindings are stale"):
                    action()
            draw.assert_not_called()
        self.assertEqual(renderer.standard("grassland")["approved_revision"], 1)

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

    def test_changed_input_scope_cannot_be_approved_with_old_preview(self):
        self.preview("grassland")
        with patch.object(renderer, "category_signatures", return_value={"grassland": "new"}):
            with self.assertRaisesRegex(ValueError, "input selection is stale"):
                renderer.approve("grassland", "User approved")

    def comparison(self, *, changed=False, partial=False):
        from PIL import Image
        self.preview("grassland", partial=partial)
        value = renderer.standard("grassland")
        result = renderer.read(self.lab / "out/grassland/render.json")
        refs = []
        for case in value["recipe"]["cases"]:
            reference = self.lab / "references/grassland/r1" / (case + ".bmp")
            reference.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (2, 2), "green").save(reference)
            refs.append({"case": case, "hour": 12, "zoom": 128, "backend": "d3d11",
                         "image": renderer.relative(reference), "sha256": renderer.checksum(reference)})
        for entry in result["outputs"]:
            path = renderer.local(entry["image"])
            Image.new("RGB", (2, 2), "red" if changed else "green").save(path)
            entry.update(backend="d3d11", sha256=renderer.checksum(path))
        value["references"] = {"d3d11": refs}
        renderer.write(renderer.standard_path("grassland"), value)
        renderer.write(self.lab / "out/grassland/render.json", result)
        return refs

    def test_exact_complete_comparison_records_inputs_not_a_new_approval(self):
        refs = self.comparison()
        renderer.compare("grassland")
        value = renderer.standard("grassland")
        self.assertEqual(value["approved_revision"], 1)
        self.assertNotIn("approval", value)
        self.assertEqual(value["references"]["d3d11"], refs)
        self.assertEqual(value["reviewed_inputs"], {"revision": 1, "signature": "current",
            "basis": "exact_approved_reference_comparison"})
        self.assertFalse((self.root / "Renderer/integration/status.json").exists())

    def test_different_partial_or_old_signature_comparison_does_not_clear_review(self):
        for args in ({"changed": True}, {"partial": True}, {}):
            self.comparison(**args)
            signature = "current" if args else "changed during comparison"
            with patch.object(renderer, "category_signatures", return_value={"grassland": signature}):
                renderer.compare("grassland")
            self.assertNotIn("reviewed_inputs", renderer.standard("grassland"))

    def test_source_selected_consumer_also_requires_explicit_appearance_review(self):
        self.preview("grassland")
        with patch.object(renderer, "dirty_categories", return_value=["lighting"]):
            with self.assertRaisesRegex(ValueError, "affected category"):
                renderer.approve("grassland", "User only reviewed grassland")
        self.assertEqual(renderer.standard("grassland")["approved_revision"], 1)

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

    def delivery(self, *, staged=True):
        candidate = self.root / "Renderer/native/build/candidate/C3XRenderer.dll"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"verified-dll")
        live = self.root / "Renderer/bin/C3XRenderer.dll"
        live.parent.mkdir(parents=True)
        live.write_bytes(b"verified-dll" if staged else b"older-dll")
        renderer.write(self.root / "Renderer/integration/status.json", {"categories": {
            "grassland": {"integrated_revision": 0}, "lighting": {"integrated_revision": 1}}})
        receipt = {"status": "pass", "implementation_identity": "current",
                   "dll_sha256": renderer.checksum(candidate), "categories": ["grassland"],
                   "approved_revisions": {"grassland": 1}}
        renderer.write(self.lab / "out/integration/grassland.json", receipt)
        return candidate, receipt

    def test_dependency_test_selection_is_deduplicated(self):
        self.assertEqual(renderer.test_modules("lighting"), [
            "Renderer.lab.test_backend_bindings", "Renderer.lab.test_dependencies", "Renderer.lab.test_natural_scene",
            "Renderer.lab.test_platform", "Renderer.lab.test_preparation",
            "Renderer.lab.test_workflow", "Renderer.native.test_scroll_damage"])

    def test_integration_requires_an_actual_game_check_and_staged_candidate(self):
        self.delivery(staged=False)
        for note, error in (("", "actual Civ III"), ("Tested game", "staged DLL")):
            with self.assertRaisesRegex(ValueError, error):
                renderer.record_integration("grassland", note)
        self.assertEqual(renderer.pending(), [{"category": "grassland", "approved": 1, "integrated": 0}])

    def test_modified_dll_invalidates_integration_receipt(self):
        candidate, _ = self.delivery()
        candidate.write_bytes(b"changed-dll")
        with self.assertRaisesRegex(ValueError, "DLL changed"):
            renderer.record_integration("grassland", "Tested in game")

    def test_changed_approval_invalidates_integration_receipt(self):
        self.delivery()
        value = renderer.standard("grassland")
        value["approved_revision"] = 2
        renderer.write(renderer.standard_path("grassland"), value)
        with self.assertRaisesRegex(ValueError, "Approval changed"):
            renderer.record_integration("grassland", "Tested in game")

    def test_recording_delivery_only_updates_verified_categories(self):
        self.delivery()
        renderer.record_integration("grassland", "Scroll, zoom, wrap and config-off checked in Civ III")
        status = renderer.read(self.root / "Renderer/integration/status.json")["categories"]
        self.assertEqual(status["lighting"], {"integrated_revision": 1})
        self.assertEqual(status["grassland"]["integrated_revision"], 1)
        self.assertEqual(renderer.pending(), [])

    def test_failed_new_verification_invalidates_older_success(self):
        self.delivery()
        with patch.object(renderer, "run_tests", side_effect=ValueError("regression")):
            with self.assertRaisesRegex(ValueError, "regression"):
                renderer.verify_integration("grassland")
        self.assertEqual(renderer.read(self.lab / "out/integration/grassland.json")["status"], "fail")
        with self.assertRaisesRegex(ValueError, "fresh integration"):
            renderer.integration_receipt("grassland")


class FixtureTests(unittest.TestCase):
    def test_focused_affected_run_uses_complete_dependent_recipes(self):
        with patch("sys.argv", ["renderer.py", "lab", "resources", "--case", "detail", "--affected"]), \
             patch.object(renderer, "prepare_sources"), \
             patch.object(renderer, "affected", return_value=["animation", "resources"]), \
             patch.object(renderer, "render") as draw:
            self.assertEqual(renderer.main(), 0)
        self.assertEqual([(c.args[0], c.kwargs["selected_case"]) for c in draw.call_args_list],
                         [("animation", None), ("resources", "detail")])

    def capture(self, category, case):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.csv"
            renderer.scene(category, case, path)
            return path.read_text()

    def test_animation_phases_use_identical_surroundings(self):
        for context in ("detail", "gameplay"):
            self.assertEqual(self.capture("animation", context + "-start"),
                             self.capture("animation", context + "-mid"))
        self.assertNotEqual(self.capture("animation", "detail-start"),
                            self.capture("animation", "gameplay-start"))

    def test_water_resource_detail_contains_only_sea_and_ocean(self):
        rows = self.capture("resources", "water-detail").splitlines()[1:]
        self.assertEqual({int(row.split(",")[2]) for row in rows}, {12, 13})
        rows = self.capture("resources", "water-gameplay").splitlines()[1:]
        self.assertTrue({2, 11, 12, 13}.issubset({int(row.split(",")[2]) for row in rows}))

    def test_unknown_fixture_fails_before_a_build_or_render(self):
        with patch.object(renderer, "ensure_candidate") as build:
            with self.assertRaisesRegex(ValueError, "Unknown fixture"):
                renderer.render("grassland", selected_case="typo")
            build.assert_not_called()


class BehaviorWitnessTests(unittest.TestCase):
    def test_unit_check_cannot_pass_without_the_executed_matrix(self):
        with self.assertRaisesRegex(ValueError, "Incomplete native unit"):
            renderer.verify_behavior_output("units", "BIQ viewport: 0 fallback")

    def test_a_failed_replay_does_not_hide_independent_results(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "LAB", Path(directory)), \
             patch.object(renderer, "native_render", side_effect=[ValueError("failed"), {}, {}, {}, {}, {}, {}]) as render, \
             patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "scroll"):
                renderer.integration_replays()
            self.assertEqual(render.call_count, 7)
            results = renderer.read(Path(directory) / "out/integration/replays/results.json")
            self.assertEqual([r["status"] for r in results], ["fail"] + ["pass"] * 6)

    def test_replay_requires_all_jumps_and_original_pixel_budget(self):
        prefix = "PICKUP selection p95_ms=0\n" + "PICKUP jump=0,0\n" * 5
        renderer.verify_behavior_output("replay", prefix + "PICKUP pixel parity: changed=1 error=40 bytes=4000")
        for output in (prefix, prefix.replace("PICKUP jump=", "missing=", 1) +
                       "PICKUP pixel parity: changed=0 error=0 bytes=4000",
                       prefix + "PICKUP pixel parity: changed=2 error=0 bytes=4000",
                       prefix + "PICKUP pixel parity: changed=0 error=41 bytes=4000"):
            with self.assertRaises(ValueError):
                renderer.verify_behavior_output("replay", output)

    def test_edit_requires_authoritative_invalidation_and_pixel_comparison(self):
        output = "PICKUP authoritative edit: pass\nPICKUP edit pixel parity: changed=0 error=0 bytes=4000"
        renderer.verify_behavior_output("edits", output)
        for invalid in (output.replace("edit: pass", "edit: FAIL"), output.splitlines()[1]):
            with self.assertRaises(ValueError):
                renderer.verify_behavior_output("edits", invalid)

    def test_animation_requires_temporal_scroll_and_removal_checks(self):
        output = "ANIMATION temporal: pass changed_frames=5\nANIMATION scroll parity: pass\nANIMATION removal parity: pass"
        renderer.verify_behavior_output("animation", output)
        for invalid in (output.replace("frames=5", "frames=0"), output.splitlines()[0], output + "\nFAIL"):
            with self.assertRaises(ValueError):
                renderer.verify_behavior_output("animation", invalid)


class InputFreshnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.lab = self.root / "Renderer/lab"
        for name, value in (("ROOT", self.root), ("LAB", self.lab)):
            replacement = patch.object(renderer, name, value)
            replacement.start(); self.addCleanup(replacement.stop)
        replacement = patch.object(renderer, "catalog", return_value={})
        replacement.start(); self.addCleanup(replacement.stop)

    def test_texture_edit_invalidates_preview_even_with_same_size_and_mtime(self):
        texture = self.root / "Renderer/packs/example/color.dds"
        texture.parent.mkdir(parents=True)
        texture.write_bytes(b"first")
        old = texture.stat()
        before = renderer.implementation_identity()
        self.assertEqual(before, renderer.implementation_identity())
        texture.write_bytes(b"other")
        os.utime(texture, ns=(old.st_atime_ns, old.st_mtime_ns))
        self.assertNotEqual(before, renderer.implementation_identity())

    def test_nested_native_header_invalidates_preview(self):
        header = self.root / "Renderer/native/materials/shared.h"
        header.parent.mkdir(parents=True)
        header.write_text("old")
        before = renderer.implementation_identity()
        header.write_text("new")
        self.assertNotEqual(before, renderer.implementation_identity())

    def test_unchanged_inventory_does_not_rewrite_hash_cache(self):
        path = self.root / "Renderer/packs/example/color.dds"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"texture")
        before = renderer.implementation_identity()
        with patch.object(renderer, "write") as write:
            self.assertEqual(before, renderer.implementation_identity())
            write.assert_not_called()
        path.unlink()
        self.assertNotEqual(before, renderer.implementation_identity())

    def test_pack_inventory_matches_previous_symlink_and_cache_semantics(self):
        packs = self.root / "Renderer/packs"
        folder = packs / "example"
        folder.mkdir(parents=True)
        (folder / "a.dds").write_bytes(b"a")
        (folder / "b.dds").symlink_to(folder / "a.dds")
        (folder / "loop").symlink_to(packs, target_is_directory=True)
        (folder / "broken.dds").symlink_to(folder / "missing.dds")
        (folder / "__pycache__").mkdir()
        (folder / "__pycache__/ignored.pyc").write_bytes(b"ignored")
        expected = {renderer.relative(p): renderer.checksum(p) for p in packs.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts}
        actual = {key: renderer.checksum(p) for key, p, _ in renderer.pack_files()}
        self.assertEqual(actual, expected)
        # Renaming and adding nested input paths remain observable.
        before = renderer.implementation_identity()
        (folder / "new.dds").write_bytes(b"new")
        self.assertNotEqual(before, renderer.implementation_identity())

    def test_pack_file_symlink_cannot_escape_repository(self):
        with tempfile.TemporaryDirectory() as other:
            external = Path(other) / "art.dds"
            external.write_bytes(b"outside")
            packs = self.root / "Renderer/packs"
            packs.mkdir(parents=True)
            (packs / "escape.dds").symlink_to(external)
            with self.assertRaises(ValueError):
                list(renderer.pack_files())

    def test_concurrent_atomic_writes_use_independent_temporary_files(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        gate = Barrier(2)
        original = Path.replace
        def together(path, destination):
            gate.wait(timeout=5)
            return original(path, destination)
        target = self.lab / "receipt.json"
        with patch.object(Path, "replace", together), ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(renderer.write, target, {"writer": n}) for n in range(2)]
            for job in jobs:
                job.result(timeout=10)
        self.assertIn(renderer.read(target), [{"writer": 0}, {"writer": 1}])
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_witness_rebuilds_after_source_or_executable_changes(self):
        sources = [self.lab / "native_preview.cpp", self.lab / "build_native_preview.bat",
                   self.root / "Renderer/native/biq_preview.cpp", self.root / "Renderer/native/c3x_renderer_api.h"]
        for path in sources:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("source")
        binary = self.lab / ".cache/native_preview.exe"
        binary.parent.mkdir(parents=True)
        def compile_tool(*args):
            binary.write_bytes(b"compiled")
            return {"status": "pass"}
        with patch("Renderer.lab.platform.native_command_result", side_effect=compile_tool) as build:
            renderer.ensure_preview_tool()
            renderer.ensure_preview_tool()
            self.assertEqual(build.call_count, 1)
            sources[0].write_text("changed source")
            renderer.ensure_preview_tool()
            self.assertEqual(build.call_count, 2)
            binary.write_bytes(b"changed binary")
            renderer.ensure_preview_tool()
            self.assertEqual(build.call_count, 3)

    def test_candidate_requires_matching_compiled_inputs(self):
        dll = self.root / "Renderer/native/build/candidate/C3XRenderer.dll"
        dll.parent.mkdir(parents=True)
        dll.write_bytes(b"candidate")
        renderer.write(self.lab / "baseline.json", {"dll_sha256": "different"})
        renderer.write(self.lab / ".cache/native-build.json", {
            "dll_sha256": renderer.checksum(dll), "inputs": {"shared.h": "old"}})
        with patch.object(renderer, "native_inputs", return_value={"shared.h": "old"}):
            renderer.require_current_candidate()
        with patch.object(renderer, "native_inputs", return_value={"shared.h": "new"}):
            with self.assertRaisesRegex(ValueError, "current C\\+\\+ inputs"):
                renderer.require_current_candidate()

    def test_missing_candidate_requests_a_build(self):
        with self.assertRaisesRegex(renderer.CandidateBuildRequired, "missing"):
            renderer.require_current_candidate()

    def test_candidate_build_is_selected_only_when_required(self):
        with patch.object(renderer, "require_current_candidate") as current, \
             patch.object(renderer, "build_candidate") as build, patch("builtins.print"):
            renderer.ensure_candidate()
            build.assert_not_called()
            current.side_effect = [renderer.CandidateBuildRequired("stale"), None]
            renderer.ensure_candidate()
            build.assert_called_once_with()
            self.assertEqual(current.call_count, 3)

    def test_candidate_input_errors_do_not_trigger_a_build(self):
        for error in (FileNotFoundError("missing source"), ValueError("malformed receipt")):
            with patch.object(renderer, "require_current_candidate", side_effect=error), \
                 patch.object(renderer, "build_candidate") as build:
                with self.assertRaises(type(error)):
                    renderer.ensure_candidate()
                build.assert_not_called()

    def test_unsuccessful_or_stale_build_cannot_be_used(self):
        with patch.object(renderer, "require_current_candidate",
                          side_effect=renderer.CandidateBuildRequired("still stale")), \
             patch.object(renderer, "build_candidate") as build, patch("builtins.print"):
            with self.assertRaisesRegex(renderer.CandidateBuildRequired, "still stale"):
                renderer.ensure_candidate()
            build.assert_called_once_with()
        with patch.object(renderer, "require_current_candidate",
                          side_effect=renderer.CandidateBuildRequired("stale")), \
             patch.object(renderer, "build_candidate", side_effect=ValueError("compile failed")), \
             patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "compile failed"):
                renderer.ensure_candidate()

    def test_render_prepares_candidate_before_capturing_input_identity(self):
        value = {"recipe": {"cases": ["detail"], "hours": [12], "zooms": [128]},
                 "revision": 1, "depends_on": []}
        events = []
        with patch.object(renderer, "standard", return_value=value), \
             patch.object(renderer, "standard_path", return_value=self.lab / "standard.json"), \
             patch.object(renderer, "require_prepared", side_effect=lambda: events.append("prepared")), \
             patch.object(renderer, "ensure_candidate", side_effect=lambda: events.append("candidate")), \
             patch.object(renderer, "implementation_identity", side_effect=lambda: events.append("identity") or "current"), \
             patch.object(renderer, "category_signatures", return_value={"grassland": "current"}), \
             patch.object(renderer, "native_render", return_value={}), patch("builtins.print"):
            renderer.render("grassland")
        self.assertEqual(events, ["prepared", "candidate", "identity", "identity"])


if __name__ == "__main__":
    unittest.main()
