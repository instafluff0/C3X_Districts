"""Protect current-code previews, optional references and integration checks."""
import json
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
        mock = patch.object(renderer, "category_signatures", return_value={"lighting": "current", "grassland": "current"})
        mock.start(); self.addCleanup(mock.stop)
        for name in ("require_prepared", "prepare_sources", "ensure_candidate"):
            mock = patch.object(renderer, name)
            mock.start(); self.addCleanup(mock.stop)
        renderer.write(self.lab / "catalog.json", {"categories": {"lighting": "lighting", "grassland": "grassland"}})
        for key in ("lighting", "grassland"):
            renderer.write(renderer.standard_path(key), {
                "id": key,
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
            "input_signature": "stale" if stale else "current",
            "recipe": renderer.standard(key)["recipe"], "outputs": records,
        })

    def test_category_approval_requires_only_the_requested_preview(self):
        self.preview("lighting")
        renderer.approve("lighting", "User approved this appearance")
        self.assertEqual(renderer.standard("lighting")["reference_inputs"], "current")
        self.assertNotIn("reference_inputs", renderer.standard("grassland"))

    def test_blank_statement_is_not_approval(self):
        self.preview("grassland")
        with self.assertRaisesRegex(ValueError, "explicit approval"):
            renderer.approve("grassland", "  ")
        self.assertNotIn("reference_inputs", renderer.standard("grassland"))

    def test_unprepared_source_cannot_be_rendered_compared_or_approved(self):
        self.preview("grassland")
        with patch.object(renderer, "require_prepared", side_effect=ValueError("bindings are stale")), \
             patch.object(renderer, "native_render") as draw:
            for action in (lambda: renderer.render("grassland"),
                           lambda: renderer.compare("grassland"),
                           lambda: renderer.approve("grassland", "User approved")):
                with self.assertRaisesRegex(ValueError, "bindings are stale"):
                    action()
            draw.assert_not_called()
        self.assertNotIn("reference_inputs", renderer.standard("grassland"))

    def test_stale_or_partial_preview_cannot_be_approved(self):
        for args, error in (({"stale": True}, "stale"), ({"partial": True}, "complete category")):
            self.preview("grassland", **args)
            with self.assertRaisesRegex(ValueError, error):
                renderer.approve("grassland", "User approved")
            self.assertNotIn("reference_inputs", renderer.standard("grassland"))

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
            reference = self.lab / "references/grassland/approved" / case / (case + ".bmp")
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

    def test_comparison_is_read_only(self):
        refs = self.comparison()
        renderer.compare("grassland")
        value = renderer.standard("grassland")
        self.assertNotIn("approval", value)
        self.assertEqual(value["references"]["d3d11"], refs)
        self.assertNotIn("reference_inputs", value)

    def test_different_or_partial_comparison_remains_read_only(self):
        for args in ({"changed": True}, {"partial": True}):
            self.comparison(**args)
            renderer.compare("grassland")
            self.assertNotIn("reference_inputs", renderer.standard("grassland"))

    def test_new_time_of_day_diagnostic_needs_no_reference_replacement(self):
        refs = self.comparison()
        path = self.lab / "out/grassland/render.json"
        result = renderer.read(path)
        for entry in result["outputs"]:
            entry["hour"] = 18
        renderer.write(path, result)
        compared = renderer.compare("grassland")
        self.assertTrue(all(row["identical_pixels"] is None and row["reference"] == "unavailable" for row in compared))
        self.assertEqual(renderer.standard("grassland")["references"]["d3d11"], refs)

    def test_old_category_signature_rejects_comparison_before_review(self):
        self.comparison()
        with patch.object(renderer, "category_signatures", return_value={"grassland": "changed"}), \
             self.assertRaisesRegex(ValueError, "stale"):
            renderer.compare("grassland")
        self.assertNotIn("reference_inputs", renderer.standard("grassland"))

    def test_category_reference_update_is_local(self):
        self.preview("grassland")
        renderer.approve("grassland", "User reviewed grassland")
        self.assertEqual(renderer.standard("grassland")["reference_inputs"], "current")
        self.assertNotIn("reference_inputs", renderer.standard("lighting"))

    def test_category_approval_replaces_one_fixed_reference(self):
        old = self.lab / "references/lighting/approved/detail/detail.bmp"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"original")
        self.preview("lighting")
        renderer.approve("lighting", "The user explicitly accepted this comparison")
        lighting = renderer.standard("lighting")
        self.assertEqual(len(lighting["references"]["d3d11"]), 2)
        self.assertTrue(all("/approved/" in row["image"] for row in lighting["references"]["d3d11"]))
        self.assertNotEqual(old.read_bytes(), b"original")

    def test_paths_cannot_escape_repository(self):
        with self.assertRaisesRegex(ValueError, "escapes"):
            renderer.local("../outside")

    def test_dependency_test_selection_is_deduplicated(self):
        self.assertEqual(renderer.test_modules("lighting"), [
            "Renderer.lab.test_backend_bindings", "Renderer.lab.test_dependencies",
            "Renderer.lab.test_platform", "Renderer.lab.test_preparation",
            "Renderer.lab.test_workflow", "Renderer.native.test_scroll_damage"])

    def test_integration_does_not_render_or_compare_references(self):
        candidate = self.root / "Renderer/native/build/candidate/C3XRenderer.dll"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"verified-dll")
        with patch.object(renderer, "run_tests", return_value=[]), \
             patch.object(renderer, "affected", return_value=["grassland"]), \
             patch.object(renderer, "render") as render, \
             patch.object(renderer, "compare") as compare, \
             patch.object(renderer, "integration_replays", return_value=[]), \
             patch("Renderer.lab.platform.changed_injected_sources", return_value=False):
            result = renderer.verify_integration_checks("grassland")
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scope"], "focused")
        self.assertEqual(result["categories"], ["grassland"])
        self.assertEqual(result["input_signatures"], {"grassland": "current"})
        self.assertNotIn("comparisons", result)
        render.assert_not_called()
        compare.assert_not_called()

    def test_renderer_only_integration_preserves_checks_without_compiling_other_work(self):
        candidate = self.root / "Renderer/native/build/candidate/C3XRenderer.dll"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"verified-dll")
        with patch.object(renderer, "run_tests", return_value=["renderer tests"]) as tests, \
             patch.object(renderer, "integration_replays", return_value=["renderer replays"]) as replays, \
             patch("Renderer.lab.platform.changed_injected_sources", return_value=True), \
             patch("Renderer.lab.platform.injected_compile_result") as injected:
            result = renderer.verify_integration_checks("grassland", renderer_only=True)
        tests.assert_called_once()
        replays.assert_called_once()
        injected.assert_not_called()
        self.assertEqual(result["injected_compile"], "not_requested_renderer_only")
        self.assertEqual(result["status"], "pass")

    def test_failed_new_verification_invalidates_older_success(self):
        with patch.object(renderer, "run_tests", side_effect=ValueError("regression")):
            with self.assertRaisesRegex(ValueError, "regression"):
                renderer.verify_integration("grassland")
        self.assertEqual(renderer.read(self.lab / "out/integration/grassland.json")["status"], "fail")

    def test_focused_resource_tests_exclude_unit_proofs(self):
        with patch.object(renderer, "test_modules", return_value=[]), \
             patch.object(renderer.subprocess, "run") as run:
            run.return_value.returncode = 0
            modules = renderer.run_tests("resources", integration=True)
        self.assertIn("Renderer.native.test_animation_runtime", modules)
        self.assertNotIn("Renderer.native.test_unit_animation_runtime", modules)
        self.assertNotIn("Renderer.native.test_unit_bridge", modules)

    def test_full_resource_tests_preserve_unit_proofs(self):
        with patch.object(renderer, "test_modules", return_value=[]), \
             patch.object(renderer.subprocess, "run") as run:
            run.return_value.returncode = 0
            modules = renderer.run_tests("resources", integration=True, full=True)
        self.assertIn("Renderer.native.test_unit_animation_runtime", modules)
        self.assertIn("Renderer.native.test_unit_bridge", modules)


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

    def test_river_fixture_is_a_complete_landscape_and_connected_watershed(self):
        rows = [tuple(map(int, row.split(",")))
                for row in self.capture("rivers", "gameplay").splitlines()[1:]]
        tiles = {(x, y): (base, real, river)
                 for x, y, base, real, _, _, river in rows}
        real_terrains = {real for base, real, river in tiles.values()}
        base_terrains = {base for base, real, river in tiles.values()}
        self.assertTrue({5, 6, 7, 8}.issubset(real_terrains))
        self.assertTrue({0, 1, 2, 11, 12, 13}.issubset(base_terrains))

        edges = set()
        for (x, y), (_, _, mask) in tiles.items():
            c, r = (x + y) // 2, (x - y) // 2
            candidates = ((2, (c, r + 1), (c + 1, r + 1)),
                          (8, (c + 1, r), (c + 1, r + 1)),
                          (32, (c, r), (c + 1, r)),
                          (128, (c, r), (c, r + 1)))
            for bit, a, b in candidates:
                if mask & bit:
                    edges.add(tuple(sorted((a, b))))
        degree = {}
        for a, b in edges:
            degree[a] = degree.get(a, 0) + 1
            degree[b] = degree.get(b, 0) + 1
        endpoints = [node for node, count in degree.items() if count == 1]
        self.assertEqual(len(endpoints), 2)
        self.assertTrue(any(a[0] != b[0] for a, b in edges))
        self.assertTrue(any(a[1] != b[1] for a, b in edges))

        def endpoint_touches_water(node):
            c, r = node
            neighbors = ((c - 1, r - 1), (c - 1, r), (c, r - 1), (c, r))
            return any(tiles.get((i + j, i - j), (2, 2, 0))[0] >= 11
                       for i, j in neighbors)

        self.assertEqual(sorted(endpoint_touches_water(node) for node in endpoints),
                         [False, True])
        self.assertGreaterEqual(sum(real == 6 for _, real, _ in tiles.values()), 12)
        self.assertGreaterEqual(sum(real == 7 for _, real, _ in tiles.values()), 10)
        self.assertGreaterEqual(sum(real == 8 for _, real, _ in tiles.values()), 6)

        def natural_real(c, r):
            return tiles.get((c + r, c - r), (2, 2, 0))[1]

        incident_tiles = set()
        incident_mountain_counts = []
        for a, b in edges:
            c, r = min(a, b)
            incident = (((c, r), (c, r - 1)) if a[1] == b[1]
                        else ((c, r), (c - 1, r)))
            incident_tiles.update(incident)
            incident_mountain_counts.append(sum(natural_real(i, j) == 6
                                                  for i, j in incident))
        self.assertGreaterEqual(sum(natural_real(c, r) == 6
                                    for c, r in incident_tiles), 6)
        self.assertLessEqual(max(incident_mountain_counts), 1)
        forest_rows = {r for (x, y), (_, real, _) in tiles.items() if real == 7
                       for r in ((x - y) // 2,)}
        jungle_rows = {r for (x, y), (_, real, _) in tiles.items() if real == 8
                       for r in ((x - y) // 2,)}
        self.assertLessEqual(min(forest_rows), -5)
        self.assertGreaterEqual(max(forest_rows), -1)
        self.assertLessEqual(min(jungle_rows), -1)
        self.assertGreaterEqual(max(jungle_rows), 3)

    def test_unknown_fixture_fails_before_a_build_or_render(self):
        with patch.object(renderer, "ensure_candidate") as build:
            with self.assertRaisesRegex(ValueError, "Unknown fixture"):
                renderer.render("grassland", selected_case="typo")
            build.assert_not_called()


class BehaviorWitnessTests(unittest.TestCase):
    def test_private_preview_does_not_rebuild_the_shared_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();dll=root/"candidate.dll";preview=root/"private preview.exe"
            dll.write_bytes(b"dll");preview.write_bytes(b"exe");output=root/"output"
            def fixture(path,command,run_id):
                (path/"gameplay-h12-z128.bmp").write_bytes(b"image")
                return {"status":"pass","output_tail":"BIQ viewport: 0 fallback, output=image"}
            with patch.object(renderer,"ROOT",root), \
                 patch.object(renderer,"scene"), \
                 patch.object(renderer,"standard",return_value={"recipe":{"objects":False}}), \
                 patch.object(renderer,"ensure_preview_tool") as build, \
                 patch("Renderer.lab.platform.run_native_fixture",side_effect=fixture), patch("builtins.print"):
                result=renderer.native_render("grassland","gameplay",12,128,output,candidate=dll,preview=preview)
                self.assertEqual(result["backend"],"d3d11")
                build.assert_not_called()
                batch=(output/"render.bat").read_text()
                self.assertIn('"..\\..\\private preview.exe"',batch)
                self.assertNotIn('..\\lab\\.cache\\native_preview.exe',batch)

    def test_fixture_batch_records_own_process_without_killing_other_tasks(self):
        import inspect
        source = inspect.getsource(renderer.native_render)
        self.assertIn('C3X_LAB_PID_FILE', source)
        self.assertIn('echo {run_id}', source)
        self.assertNotIn('taskkill /F /IM', source)

    def test_pending_replay_prevents_overlapping_the_next_case(self):
        from Renderer.lab.platform import NativeFixturePending
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "LAB", Path(directory)), \
             patch.object(renderer, "native_render", side_effect=NativeFixturePending("PID 1234 still running")) as render, \
             patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "terrain-edit"):
                renderer.integration_replays("shadows")
            render.assert_called_once()

    def test_unit_check_cannot_pass_without_the_executed_matrix(self):
        with self.assertRaisesRegex(ValueError, "Incomplete native unit"):
            renderer.verify_behavior_output("units", "BIQ viewport: 0 fallback")

    def test_unit_check_requires_individual_phases_and_expanded_identity_checks(self):
        lines=["UNIT body matrix drawn=288 status=pass", "UNIT cached anchor translation: pass",
               "UNIT repeated native cursor: pass", "UNIT retained terrain unchanged: pass",
               "UNIT post-draw terrain parity: pass", "UNIT config-off preserves canvas: pass",
               "UNIT action interruption and held endpoint: pass draws=582",
               "UNIT independent ambient phases and exact repeat: pass"]
        for zoom in (0,1):
            lines.append(f"UNIT magenta underlay parity zoom={zoom}")
            for mode in ("RGB555","RGB565"):
                lines.extend((f"UNIT {mode} clipped zoom={zoom}",
                              f"UNIT {mode} magenta clipped parity zoom={zoom} status=pass"))
        complete="\n".join(lines)
        renderer.verify_behavior_output("units",complete)
        for invalid in (complete.replace("draws=582","draws=564"),
                        complete.replace("UNIT independent ambient phases and exact repeat: pass", "")):
            with self.assertRaisesRegex(ValueError,"Incomplete native unit"):
                renderer.verify_behavior_output("units",invalid)

    def test_a_failed_replay_does_not_hide_independent_results(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "LAB", Path(directory)), \
             patch.object(renderer, "affected", return_value=["animation"]), \
             patch.object(renderer, "native_render", side_effect=[ValueError("failed"), {}, {}, {}, {}, {}]) as render, \
             patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "scroll"):
                renderer.integration_replays("animation", full=True)
            self.assertEqual(render.call_count, 6)
            results = renderer.read(Path(directory) / "out/integration/replays/animation/results.json")
            self.assertEqual([r["status"] for r in results], ["fail"] + ["pass"] * 5)

    def test_category_replays_do_not_share_output_files(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "LAB", Path(directory)), \
             patch.object(renderer, "native_render", return_value={}) as render, patch("builtins.print"):
            renderer.integration_replays("shadows")
            first = {call.args[4] for call in render.call_args_list}
            render.reset_mock()
            renderer.integration_replays("resources")
            second = {call.args[4] for call in render.call_args_list}
            self.assertFalse(first.intersection(second))

    def test_partial_edit_reuse_has_an_existing_coast(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(renderer, "LAB", Path(directory)), \
             patch.object(renderer, "native_render", return_value={}) as render, patch("builtins.print"):
            renderer.integration_replays("grassland")
            self.assertEqual(render.call_count, 1)
            self.assertEqual(render.call_args.args[:4], ("shorelines", "lowland", 12, 128))
            self.assertEqual(render.call_args.kwargs, {"behavior": "edits", "center": (10, 18)})

    def test_focused_resource_check_runs_only_its_playback_witness(self):
        with patch.object(renderer, "affected", return_value=["animation", "resources"]):
            names = [case[0] for case in renderer.integration_replay_cases("resources")]
        self.assertEqual(names, ["resource-playback"])
        with patch.object(renderer, "affected", return_value=["grassland"]):
            self.assertIn("terrain-edit", [case[0] for case in renderer.integration_replay_cases("grassland")])

    def test_full_resource_check_preserves_the_exhaustive_sweep(self):
        with patch.object(renderer, "affected", return_value=["animation", "resources"]):
            names = [case[0] for case in renderer.integration_replay_cases("resources", full=True)]
        self.assertEqual(names, ["scroll", "reduced-scroll", "world-wrap", "resource-playback",
                                 "unit-actions-day", "unit-actions-night"])

    def test_focused_terrain_integration_skips_generic_scroll_sweep(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(renderer, "LAB", Path(directory)), \
             patch.object(renderer, "affected", return_value=["grassland"]), \
             patch.object(renderer, "native_render", return_value={}) as render, \
             patch("builtins.print"):
            results = renderer.integration_replays("grassland")
        self.assertEqual([result["name"] for result in results], ["terrain-edit"])
        self.assertEqual(render.call_count, 1)

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
    def test_first_build_does_not_require_or_create_a_staged_dll(self):
        candidate = self.root / "Renderer/native/build/candidate/C3XRenderer.dll"
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"tested-candidate")
        with patch.object(renderer, "prepare_sources"), \
             patch.object(renderer, "native_inputs", return_value={}), \
             patch.object(renderer, "ensure_preview_tool"), \
             patch("Renderer.lab.platform.native_command_result", return_value={"status": "pass"}):
            renderer.build_candidate()
        self.assertFalse((self.root / "Renderer/bin/C3XRenderer.dll").exists())
        self.assertEqual(renderer.read(self.lab / ".cache/native-build.json")["dll_sha256"],
                         renderer.checksum(candidate))

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
        replacement = patch.object(renderer, "native_inputs", return_value={})
        replacement.start(); self.addCleanup(replacement.stop)

    def test_preparation_receipts_define_the_bounded_pack_closure(self):
        receipt = self.lab / ".cache/assets/example.json"
        renderer.write(receipt, {"inputs": {"Renderer/packs/current/source.dds": "first"},
                                 "outputs": {"Renderer/packs/runtime/color.dds": "built"}})
        before = renderer.implementation_identity()
        unrelated = self.root / "Renderer/packs/history/unused.dds"
        unrelated.parent.mkdir(parents=True)
        unrelated.write_bytes(b"ignored")
        self.assertEqual(before, renderer.implementation_identity())
        renderer.write(receipt, {"inputs": {"Renderer/packs/current/source.dds": "changed"},
                                 "outputs": {"Renderer/packs/runtime/color.dds": "built"}})
        self.assertNotEqual(before, renderer.implementation_identity())

    def test_injected_contract_invalidates_preview(self):
        header = self.root / "C3X.h"
        header.write_text("old")
        with patch.object(renderer, "native_inputs", return_value={renderer.relative(header): renderer.checksum(header)}):
            before = renderer.implementation_identity()
            header.write_text("new")
            self.assertNotEqual(before, renderer.implementation_identity())

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
                 "depends_on": []}
        events = []
        with patch.object(renderer, "standard", return_value=value), \
             patch.object(renderer, "standard_path", return_value=self.lab / "standard.json"), \
             patch.object(renderer, "require_prepared", side_effect=lambda categories: events.append("prepared")), \
             patch.object(renderer, "ensure_candidate", side_effect=lambda: events.append("candidate")), \
             patch.object(renderer, "category_signatures", side_effect=lambda: events.append("identity") or {"grassland": "current"}), \
             patch.object(renderer, "native_render", return_value={}), patch("builtins.print"):
            renderer.render("grassland")
        self.assertEqual(events, ["prepared", "candidate", "identity", "identity"])


if __name__ == "__main__":
    unittest.main()
