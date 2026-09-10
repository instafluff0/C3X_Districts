"""Evidence must reject stale, partial, changed-quality and changed-camera runs."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from Renderer.native.analyze_navigation_run import compare, digest, distribution, inspect


class NavigationAnalysisTests(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        for name in ("C3XRenderer.dll", "biq_preview.exe"):
            (root / name).write_bytes(name.encode())
        args = dict(scenario="navigation", resident=True, resident_steps=14, width=4, height=4,
                    tile_width=128, waves="0", dependency_control=False)
        receipt = dict(invocation="current", binaries={n: digest(root/n) for n in ("C3XRenderer.dll", "biq_preview.exe")},
                       inputs={"scene": "frozen"}, args=args, environment={"C3X_RENDERER_PREVIEW_SEASON": "0"})
        (root / "inputs.json").write_text(json.dumps(receipt))
        lines = []
        for cycle in range(2):
            for step in range(6):
                lines.append(f"NAV cycle={cycle} step={step} result=1")
                if cycle:
                    lines.append(f"NAV parity step={step} status=pass")
        header = bytearray(54)
        header[:2] = b"BM"
        struct.pack_into("<I", header, 10, 54)
        struct.pack_into("<ii", header, 18, 4, -4)
        struct.pack_into("<H", header, 28, 32)
        for i in range(14):
            lines.append(f"RESIDENT_NAV step={i} x=35 y={41+i*2} pixel_y={(i+1)*64} result=1 built=0 reused=10 upload_bytes=0 ms=10 capture_ms=1 geometry_ms=2 draw_ms=3 readback_ms=4")
            (root / f"zoom.bmp.resident{i}.bmp").write_bytes(header + bytes([i])*64)
        lines.extend(("RESIDENT_END status=pass", "BIQ 100x100 viewport: 0 fallback"))
        (root / "benchmark.log").write_text("\n".join(lines))
        (root / "renderer.log").write_text("")
        (root / "completion.txt").write_text("0\n")
        (root / "evidence.json").write_text(json.dumps(dict(invocation="current", returncode=0,
            inputs_unchanged=True, binaries_unchanged=True, images={p.name: digest(p) for p in root.glob("*.bmp")})))
        return root

    def test_pair_allows_only_cache_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=self.fixture(Path(temporary)/"a");b=self.fixture(Path(temporary)/"b")
            data=json.loads((b/"inputs.json").read_text())
            data["args"]["dependency_control"]=True
            (b/"inputs.json").write_text(json.dumps(data))
            result=compare(a,b)
            self.assertTrue(result["all_images_exact"])
            self.assertFalse(result["candidate"]["timing"]["ms"]["hundred_sample_requirement_met"])
            data["environment"]["C3X_RENDERER_PREVIEW_SEASON"]="1"
            (b/"inputs.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"environment"):compare(a,b)
            data["environment"]["C3X_RENDERER_PREVIEW_SEASON"]="0";data["args"]["waves"]="1"
            (b/"inputs.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"quality"):compare(a,b)

    def test_modified_pixels_binary_camera_and_incomplete_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            image=root/"zoom.bmp.resident0.bmp";original=image.read_bytes()
            image.write_bytes(original[:-1]+b"X")
            with self.assertRaisesRegex(ValueError,"Image changed"):inspect(root)
            image.write_bytes(original)
            log=root/"benchmark.log";original=log.read_text()
            log.write_text(original.replace("x=35", "x=37",1))
            with self.assertRaisesRegex(ValueError,"camera"):inspect(root)
            log.write_text(original.replace("RESIDENT_END status=pass", "RESIDENT_END status=FAIL"))
            with self.assertRaisesRegex(ValueError,"sweep"):inspect(root)
            log.write_text(original)
            (root/"C3XRenderer.dll").write_bytes(b"other")
            with self.assertRaisesRegex(ValueError,"Binary changed"):inspect(root)

    def test_distribution_does_not_promote_small_samples_or_invalid_times(self):
        self.assertEqual(distribution(list(range(1,101)))["p95_ms"],95)
        for values in ([], [float("nan")], [-1]):
            with self.assertRaises(ValueError):distribution(values)


if __name__ == "__main__":
    unittest.main()
