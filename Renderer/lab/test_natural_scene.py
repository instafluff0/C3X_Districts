"""Partial Mac scenes must never certify complete category approval."""
import contextlib
import copy
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from Renderer import renderer as r
from Renderer.lab.backends import natural_scene as scene


class NaturalScene(unittest.TestCase):
    def test_portable_input_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "scene-inputs"
            compiled = subprocess.run(["clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                str(r.LAB / "scene/test_grassland.cpp"),
                str(r.ROOT / "Renderer/native/environment_runtime.cpp"), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PASS grassland pilot fixture and DDS boundaries", result.stdout)

    def test_scope_and_recipe(self):
        scene.request("grassland", "detail")
        for args in (("cities", "detail"), ("grassland", None),
                     ("grassland", "gameplay"), ("grassland", "detail", True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                scene.request(*args)
        value = r.standard("grassland")
        scene.recipe(value)
        for key, changed in (("hours", [0]), ("zooms", [64]), ("objects", True), ("feature", 5)):
            bad = copy.deepcopy(value)
            bad["recipe"][key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                scene.recipe(bad)

    def test_cli_rejects_before_preparing_or_native_work(self):
        for args in (("lab", "grassland", "--backend", "metal"),
                     ("lab", "cities", "--backend", "metal", "--case", "detail")):
            with mock.patch("sys.argv", ["renderer", *args]), mock.patch.object(r, "prepare_sources") as prepare, \
                 mock.patch.object(r, "render") as native, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(r.main(), 1)
                prepare.assert_not_called()
                native.assert_not_called()

    def test_comparison_guards_and_approval_isolation(self):
        from PIL import Image
        value = copy.deepcopy(r.standard("grassland"))
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory).resolve()
            lab = root / "Renderer/lab"
            output = lab / "out/grassland/metal"
            output.mkdir(parents=True)
            for name, result in (("ROOT", root), ("LAB", lab)):
                stack.enter_context(mock.patch.object(r, name, result))
            stack.enter_context(mock.patch.object(r, "standard", return_value=value))
            stack.enter_context(mock.patch.object(r, "implementation_identity", return_value="current"))
            stack.enter_context(mock.patch.object(r, "require_prepared"))
            approved = root / "approved.bmp"
            candidate = output / "detail.bmp"
            Image.new("RGB", (2, 1), (100, 100, 100)).save(approved)
            image = Image.open(approved).copy()
            image.putpixel((0, 0), (101, 100, 100))
            image.save(candidate)
            value["references"] = {"d3d11": [{"case": "detail", "hour": 12, "zoom": 128,
                "image": r.relative(approved), "sha256": r.checksum(approved)}]}
            record = {"backend": "metal", "scope": "grassland-detail-pilot", "implementation_identity": "current",
                "recipe": value["recipe"], "image": r.relative(candidate),
                "sha256": r.checksum(candidate), "wall_seconds": 1}
            record_path = output / "render.json"
            r.write(record_path, record)
            before = copy.deepcopy(value)
            with contextlib.redirect_stdout(io.StringIO()):
                metrics = scene.compare("grassland")
            self.assertEqual(metrics["maximum_channel_difference"], 1)
            self.assertEqual(metrics["identical_pixel_fraction"], .5)
            self.assertEqual(metrics["mean_channel_difference"], [.5, 0, 0])
            self.assertFalse(metrics["complete_category_parity"])
            self.assertFalse(metrics["approval_changed"])
            self.assertEqual(value, before)
            self.assertEqual(r.checksum(approved), value["references"]["d3d11"][0]["sha256"])
            for key, changed in (("implementation_identity", "stale"), ("scope", "complete"),
                                 ("backend", "d3d11"), ("recipe", {}), ("sha256", "altered")):
                r.write(record_path, dict(record, **{key: changed}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    scene.compare("grassland")
            r.write(record_path, record)
            value["references"]["d3d11"][0]["sha256"] = "altered"
            with self.assertRaisesRegex(ValueError, "Approved image"):
                scene.compare("grassland")


if __name__ == "__main__":
    unittest.main()
