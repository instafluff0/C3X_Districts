"""Portable platform gates. GPU witnesses are produced separately by runner.py."""

import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

V2 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(V2 / "app"))
import cache
import runner


class PlatformTests(unittest.TestCase):
    def test_content_cache_integrity_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            c = cache.Cache(folder)
            calls = []

            def build(p):
                calls.append(1)
                p.write_bytes(b"content")

            a = c.artifact("asset", {"source": "one"}, build)
            self.assertEqual(c.artifact("asset", {"source": "one"}, build), a)
            self.assertEqual(calls, [1])
            a.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt asset cache"):
                c.artifact("asset", {"source": "one"}, build)
            self.assertNotEqual(c.artifact("asset", {"source": "two"}, build), a)

    def test_cache_partial_commit_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            c = cache.Cache(folder)
            p = c.artifact("shader", {}, lambda p: p.write_bytes(b"a"))
            (p.parent / "record.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete shader cache"):
                c.artifact("shader", {}, lambda p: p.write_bytes(b"b"))

    def test_fixture_and_contract(self):
        f, m = runner.fixture(V2 / "tests/platform/micro.fixture.json")
        self.assertEqual(f["tile_count"], 16)
        self.assertEqual(m["contract"], 1)
        self.assertEqual(f["settings"]["anisotropy"], 8)
        runner.fixture(V2 / "tests/platform/complete.fixture.json")

    def test_unsupported_settings_fail(self):
        f, _ = runner.fixture(V2 / "tests/platform/micro.fixture.json")
        for key, value in [
            ("samples", 3),
            ("anisotropy", 17),
            ("render_scale", 3),
            ("mip_bias", float("nan")),
            ("postprocess", "unknown"),
            ("camera_offsets", [[float("inf"), 0]]),
        ]:
            with self.subTest(key=key):
                s = copy.deepcopy(f["settings"])
                s[key] = value
                with self.assertRaises(ValueError):
                    runner.validate_settings(s)
        s = copy.deepcopy(f["settings"])
        s["mystery"] = 1
        with self.assertRaises(ValueError):
            runner.validate_settings(s)

    def test_ownership_and_path_escape(self):
        # The user's single-lead authorization spans v2; the historical track
        # namespaces must still reject access to production and frozen inputs.
        runner.owned(V2 / "shared/frozen_scene.cpp", "Q1-sampling")
        with self.assertRaisesRegex(ValueError, "does not own"):
            runner.owned(runner.ROOT / "Renderer/native/renderer.cpp", "Q1-sampling")
        with self.assertRaisesRegex(ValueError, "escapes"):
            runner.local("../not-owned")

    def test_bias_only_changes_implicit_texture_reads(self):
        source = "a.Sample(s,float2(1,2)) + b.SampleLevel(s,u,0) + c.Sample(s,u)"
        self.assertEqual(runner.apply_mip_bias(source, 0), source)
        changed = runner.apply_mip_bias(source, -0.5)
        self.assertIn("a.SampleBias(s,float2(1,2),-0.5)", changed)
        self.assertIn("b.SampleLevel(s,u,0)", changed)
        self.assertIn("c.SampleBias(s,u,-0.5)", changed)

    def test_frozen_v1_guard(self):
        guard = json.loads((V2 / "shared/frozen_guard.json").read_text())
        for name, expected in guard["files"].items():
            self.assertEqual(cache.file_hash(runner.local(name)), expected, name)

    def test_settings_change_identity(self):
        f, _ = runner.fixture(V2 / "tests/platform/micro.fixture.json")
        initial = cache.digest(cache.canonical(f))
        for key, value in [
            ("samples", 4),
            ("anisotropy", 16),
            ("mip_bias", -0.5),
            ("render_scale", 2),
            ("camera_offsets", [[0.5, 0]]),
        ]:
            g = copy.deepcopy(f)
            g["settings"][key] = value
            self.assertNotEqual(cache.digest(cache.canonical(g)), initial)

    def test_missing_module_and_contract_drift(self):
        f, _ = runner.fixture(V2 / "tests/platform/micro.fixture.json")
        with tempfile.TemporaryDirectory(dir=V2 / "tests/platform") as temp:
            p = Path(temp) / "fixture.json"
            f["modules"] = [
                "Renderer/terrain_lab/v2/tests/platform/does-not-exist.json"
            ]
            p.write_text(json.dumps(f))
            with self.assertRaisesRegex(ValueError, "missing module"):
                runner.fixture(p)
            f["schema"] = "unknown"
            p.write_text(json.dumps(f))
            with self.assertRaisesRegex(ValueError, "contract drift"):
                runner.fixture(p)


class IncrementalBuildTests(unittest.TestCase):
    def test_owner_sources_and_shader_changes_have_separate_cache_keys(self):
        import shutil

        if not shutil.which("clang++"):
            self.skipTest("C++ toolchain unavailable")
        with tempfile.TemporaryDirectory(dir=V2 / "tests/platform") as folder:
            root = Path(folder)
            a = root / "a.cpp"
            b = root / "b.cpp"
            a.write_text("int owner_a(){return 1;}\n")
            b.write_text("int owner_b(){return 2;}\n")
            c = cache.Cache(root / "cache")
            one = runner.compile_cpp(c, a)
            two = runner.compile_cpp(c, b)
            self.assertNotEqual(one, two)
            a.write_text("int owner_a(){return 3;}\n")
            self.assertNotEqual(runner.compile_cpp(c, a), one)
            self.assertEqual(runner.compile_cpp(c, b), two)
            self.assertTrue(c.events[-1]["hit"])
            shader = root / "policy.hlsl"
            shader.write_text((V2 / "tests/platform/diagnostic.hlsl").read_text())
            try:
                runner.tools()
            except ValueError:
                self.skipTest("optional shader tools unavailable")
            before = runner.shaders(c, shader, 0)
            shader.write_text(shader.read_text() + "\n// shader-only change\n")
            after = runner.shaders(c, shader, 0)
            self.assertNotEqual(before["PSMain"], after["PSMain"])
            self.assertEqual(runner.compile_cpp(c, b), two)
            self.assertTrue(c.events[-1]["hit"])

    def test_frozen_reduction_matches_cross_owner_witness(self):
        with tempfile.TemporaryDirectory(dir=V2 / "tests/platform") as folder:
            root = Path(folder)
            source = root / "reduce.cpp"
            binary = root / "reduce"
            image = root / "checker.bmp"
            source.write_text(
                '#include "../../../contracts/packet_v1.h"\nint main(int,char**argv){std::vector<uint8_t> p={0,0,0,255,255,255,255,255,0,0,0,255,255,255,255,255};return labv2::write_bmp(argv[1],p,2,2,2)?0:1;}\n'
            )
            runner.run(["clang++", "-std=c++17", source, "-o", binary])
            runner.run([binary, image])
            self.assertEqual(image.read_bytes()[54:58], bytes([127, 127, 127, 255]))

    def test_compact_packet_round_trip_and_corruption(self):
        import shutil
        from packet_store import compact_packet

        if not shutil.which("clang++"):
            self.skipTest("C++ toolchain unavailable")
        with tempfile.TemporaryDirectory(dir=V2 / "tests/platform") as folder:
            root = Path(folder)
            c = cache.Cache(root / "cache")
            obj = runner.compile_cpp(c, V2 / "tests/platform/owner_a/builder.cpp")
            binary = root / "builder"
            runner.run(["clang++", obj, "-o", binary])
            packet = root / "module.packet"
            runner.run([binary, packet, "128", "128", "12", "1", "unused"])
            raw = packet.read_bytes()
            compact_packet(packet, root / "content")
            validate = root / "validate"
            runner.run(
                [
                    "clang++",
                    "-std=c++17",
                    V2 / "shared/validate_packet.cpp",
                    "-o",
                    validate,
                ]
            )
            self.assertIn("PASS packet", runner.run([validate, packet]))
            refs = list(Path(str(packet) + ".blobs").iterdir())
            self.assertTrue(refs)
            refs[0].write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt packet content resource"):
                runner.run([validate, packet])
            packet.write_bytes(raw[:12])
            with self.assertRaisesRegex(ValueError, "truncated"):
                runner.run([validate, packet])


if __name__ == "__main__":
    unittest.main()
