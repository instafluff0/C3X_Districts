import unittest

from Renderer.tools.analyze_renderer_trace import analyze


class TraceTests(unittest.TestCase):
    def test_idle_work_cumulative_time_is_not_a_latency_sample(self):
        report = analyze("""[C3X renderer] stage=prewarm cumulative_ms=800
[C3X renderer] stage=worker-complete wait_ms=3 cumulative_prewarm_ms=900
""")
        self.assertEqual(["worker-complete.wait_ms"], list(report["timings"]))
        self.assertEqual(3, report["timings"]["worker-complete.wait_ms"]["p95_ms"])

    def test_mixed_debugger_trace_does_not_double_count_composite_or_maxima(self):
        report = analyze("""
unrelated process output
[C3X renderer] qpc=123 thread=4 sequence=1 stage=frame cache=tiles scene=1 built=2 reused=8 evicted=1 gpu_bytes=4096 upload_bytes=512 render_ms=10.0
[C3X renderer] qpc=124 frame=1 stage=composite built=2 reused=8 upload_bytes=512 capture_ms=2.0 render_wait_ms=11.0
[C3X renderer] qpc=125 frame=1 stage=map-complete total_ms=14.0 max_capture_ms=20.0
[C3X renderer] qpc=126 sequence=2 stage=frame cache=viewport-current scene=0 built=0 reused=0 gpu_bytes=4096 render_ms=0.0
[C3X renderer] qpc=127 sequence=2 stage=worker-complete wait_ms=0.5
[C3X renderer] stage=frame render_ms=nan readback_wait_ms=unfinished
""")
        self.assertEqual(2, report["totals"]["built"])
        self.assertEqual(512, report["totals"]["upload_bytes"])
        self.assertEqual({"scene": 1}, report["invalidations"])
        self.assertEqual(4096, report["peak_gpu_buffer_bytes"])
        self.assertEqual(2, report["timings"]["frame.render_ms"]["samples"])
        self.assertEqual(10, report["timings"]["frame.render_ms"]["p95_ms"])
        self.assertNotIn("map-complete.max_capture_ms", report["timings"])
        self.assertNotIn("frame.readback_wait_ms", report["timings"])


if __name__ == "__main__":
    unittest.main()
