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

    def test_dense_scene_settings_cannot_disappear_in_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            a=self.fixture(Path(temporary)/"a");b=self.fixture(Path(temporary)/"b")
            data=json.loads((b/"inputs.json").read_text())
            data["args"].update(idle_steps=100,idle_units=0,dense_scene=False)
            data["environment"].update(C3X_RENDERER_PREVIEW_IDLE_UNITS="0",C3X_RENDERER_PREVIEW_DENSE_SCENE="")
            (b/"inputs.json").write_text(json.dumps(data))
            self.assertTrue(compare(a,b)["all_images_exact"])
            data["args"]["dense_scene"]=True
            (b/"inputs.json").write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"quality"):compare(a,b)

    def test_camera_call_times_require_verified_coalescing_and_are_not_presentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text());receipt["args"]["camera_view"]=True
            (root/"inputs.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,"camera completion"):inspect(root)
            log=root/"benchmark.log";original=log.read_text()
            calls="\n".join(f"CAMERA ticket={i+1} accepted_ms=1 final_ms=9 poll_max_ms=0.5 repeat_max_ms=0.2 identical_coalesced=1 stale_rejected=1 result=1" for i in range(14))
            log.write_text(calls+"\n"+original)
            report=inspect(root)[1]
            self.assertEqual(report["standalone_queue"]["accepted_ms"]["p95_ms"],1)
            self.assertIsNone(report["native_presented_frames"])
            log.write_text(calls.replace("identical_coalesced=1","identical_coalesced=0",1)+"\n"+original)
            with self.assertRaisesRegex(ValueError,"camera completion"):inspect(root)

    def test_idle_requires_advancing_clocks_retained_geometry_and_changed_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text())
            receipt["args"].update(scenario="idle",idle_steps=14)
            (root/"inputs.json").write_text(json.dumps(receipt))
            for i in range(14):
                (root/f"zoom.bmp.idle{i}.bmp").write_bytes((root/f"zoom.bmp.resident{i}.bmp").read_bytes())
            completion=json.loads((root/"evidence.json").read_text())
            completion["images"]={p.name:digest(p) for p in root.glob("*.bmp")}
            (root/"evidence.json").write_text(json.dumps(completion))
            lines=["IDLE_BEGIN steps=14 warmup=10 pose_hz=15 paced=0 x=75 y=39 tile_width=128 units=0"]
            lines += [f"IDLE_FRAME step={i} ticks={1000000+(i+11)*1000000//15} result=1 visible=3 built=0 reused=10 upload_bytes=0 changed=1 ms=10 recoveries=0" for i in range(14)]
            lines += ["IDLE_END status=pass changed_frames=13", "BIQ viewport: 0 fallback"]
            log=root/"benchmark.log";original="\n".join(lines);log.write_text(original)
            report=inspect(root)[1]
            self.assertEqual(report["changed_frames"],13)
            self.assertEqual(report["timing"]["ms"]["samples"],14)
            self.assertIsNone(report["native_presented_frames"])
            for invalid in (original.replace("ticks=1733333","ticks=1000000"),
                            original.replace("built=0","built=1",1),
                            original.replace("upload_bytes=0","upload_bytes=256",1)):
                log.write_text(invalid)
                with self.assertRaisesRegex(ValueError,"Idle clocks"):inspect(root)
            log.write_text(original.replace("changed_frames=13","changed_frames=0"))
            with self.assertRaisesRegex(ValueError,"pose changes"):inspect(root)

    def test_animation_requires_exact_scroll_and_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary)/"run")
            receipt=json.loads((root/"inputs.json").read_text());receipt["args"]["scenario"]="animation"
            (root/"inputs.json").write_text(json.dumps(receipt))
            for i in range(6):
                (root/f"zoom.bmp.animation-{i}.bmp").write_bytes((root/f"zoom.bmp.resident{i}.bmp").read_bytes())
            (root/"zoom.bmp").write_bytes((root/"zoom.bmp.resident0.bmp").read_bytes())
            completion=json.loads((root/"evidence.json").read_text())
            completion["images"]={p.name:digest(p) for p in root.glob("*.bmp")}
            (root/"evidence.json").write_text(json.dumps(completion))
            lines=["ANIMATION zoom-return parity: pass"]
            lines += [f"ANIMATION temporal frame={i} visible=3 terrain_built=0 terrain_upload=0 ms=10" for i in range(6)]
            lines += ["ANIMATION temporal: pass changed_frames=5", "ANIMATION scroll parity: pass changed=0 error=0", "ANIMATION removal parity: pass changed=0 error=0", "BIQ viewport: 0 fallback"]
            log=root/"benchmark.log";log.write_text("\n".join(lines))
            report=inspect(root)[1]
            self.assertEqual(report["timing"]["ms"]["samples"],6)
            self.assertFalse(report["timing"]["ms"]["hundred_sample_requirement_met"])
            log.write_text(log.read_text().replace("error=0","error=1",1))
            with self.assertRaisesRegex(ValueError,"exact animation"):inspect(root)


if __name__ == "__main__":
    unittest.main()
