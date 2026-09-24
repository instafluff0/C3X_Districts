"""Check clock alignment and incomplete-evidence handling in capture reports."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from Renderer.tools.analyze_renderer64_capture import analyze


class Renderer64CaptureAnalysisTests(unittest.TestCase):
    def test_correlates_input_with_helper_present_without_claiming_display(self):
        with TemporaryDirectory() as temporary:
            session = Path(temporary)
            (session / "inspection").mkdir()
            (session / "window").mkdir()
            (session / "session.json").write_text(json.dumps({
                "renderer_backend": "Renderer64 direct surface", "result": "recording-ended-game-still-running",
                "presentmon_target": "C3XRendererHelper64.exe"}))
            (session / "inspection/report.json").write_text(json.dumps({
                "complete": True, "verified_prefix": True, "frequency": 1000,
                "qpc_origin": 1000, "calls": 2}))
            (session / "inspection/timeline.jsonl").write_text("\n".join(json.dumps(row) for row in (
                {"family": 7, "input_ticks": 10, "result_ticks": 15, "result": 1},
                {"family": 21, "input_ticks": 20, "result_ticks": 24, "result": 1})) + "\n")
            with (session / "frames.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("Application", "QPCTime", "msBetweenPresents"))
                writer.writeheader()
                writer.writerow({"Application": "C3XRendererHelper64.exe", "QPCTime": 1030,
                                 "msBetweenPresents": 16})
                writer.writerow({"Application": "C3XRendererHelper64.exe", "QPCTime": 2030,
                                 "msBetweenPresents": 20})
            (session / "renderer64-memory.jsonl").write_text(json.dumps({"private_bytes": 104857600}) + "\n")
            (session / "window/timeline.jsonl").write_text(
                json.dumps({"event": "process_memory", "private_bytes": 209715200,
                            "free_bytes": 314572800}) + "\n" + json.dumps({"frame": 1}) + "\n")
            (session / "renderer.log").write_text("\n".join((
                "[C3X renderer] qpc=10 ms=900 stage=composite render_wait_ms=17.5",
                "[C3X renderer] stage=native-copy-admission operation=9 reason=bits-lease",
                "[C3X renderer] stage=native-cpu-barrier operation=9 gpu_ms=6.5 total_ms=8.0",
                "[C3X renderer] stage=native-handoff call_ms=27.0")) + "\n")
            (session / "renderer-runtime.log.x64").write_text("\n".join((
                "[C3X renderer] stage=visual-readiness ready=1 active=1",
                "[C3X renderer] stage=direct-visual drawn=1 sample_ms=4.0 present_ms=2.0 total_ms=7.0",
                "[C3X renderer] stage=trial-present-phase display_ms=12.0 target_ms=10.0 present_ms=0.5",
                "[C3X renderer] stage=video-memory local_budget=419430400 local_usage=209715200",
                "[C3X renderer] stage=visual-frame request_ms=9.0")) + "\n")
            result = analyze(session)
            self.assertEqual(result["families"]["camera"]["bridge_call_service"]["median_ms"], 5)
            self.assertEqual(result["families"]["camera"]["next_helper_present_opportunity"]["median_ms"], 20)
            self.assertEqual(result["families"]["unit_move"]["next_helper_present_opportunity"]["median_ms"], 10)
            self.assertEqual(result["presentation"]["camera_following_1s"]["rows"], 1)
            self.assertEqual(result["presentation"]["unit_move_following_1s"]["rows"], 1)
            self.assertEqual(result["presentation"]["outside_camera_or_move"]["rows"], 1)
            self.assertEqual(result["helper_private_peak_mib"], 100)
            self.assertEqual(result["window_samples"], 1)
            self.assertEqual(result["missing"], [])
            pipeline = result["pipeline"]
            self.assertEqual(pipeline["map_render_wait"]["median_ms"], 17.5)
            self.assertEqual(pipeline["native_copy_admission"]["reasons"], {"bits-lease": 1})
            self.assertEqual(pipeline["native_cpu_barriers"]["gpu_readback"]["median_ms"], 6.5)
            self.assertEqual(pipeline["direct_visual"]["changed_frames"], 1)
            self.assertEqual(pipeline["direct_visual"]["present"]["median_ms"], 2)
            self.assertEqual(pipeline["direct_present"]["target_bind"]["median_ms"], 10)
            self.assertEqual(pipeline["gpu_memory"]["local_budget_mib_min"], 400)
            self.assertEqual(pipeline["gpu_memory"]["local_usage_mib_peak"], 200)
            (session / "frames.csv").unlink()
            self.assertIn("helper_presentmon", analyze(session)["missing"])

    def test_rejects_legacy_backend(self):
        with TemporaryDirectory() as temporary:
            session = Path(temporary)
            (session / "session.json").write_text('{"renderer_backend":"x86"}')
            with self.assertRaisesRegex(ValueError, "Renderer64"):
                analyze(session)


if __name__ == "__main__":
    unittest.main()
