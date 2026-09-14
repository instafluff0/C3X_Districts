import unittest

from Renderer.tools.analyze_renderer_trace import analyze


class TraceTests(unittest.TestCase):
    def test_usage_separates_workloads_and_does_not_treat_utc_as_latency(self):
        lines = ["[C3X renderer] process=8 qpc=1 stage=usage-session qpc_frequency=1000 utc_unix_ms=123456789 schema=1"]
        views = [(128,0,0,1),(128,0,0,1),(128,16,0,1),(160,16,0,1),(160,1000,0,1),(160,1000,0,2)]
        for request,(zoom,x,y,revision) in enumerate(views,1):
            lines.append(f"[C3X renderer] process=8 qpc={request*100} stage=usage-view request={request} origin_valid=1 origin_x={x} origin_y={y} tile_width={zoom} tile_height={zoom//2} target_width=800 target_height=600 world_width=100 world_height=100 wrap_x=1 wrap_y=0 world_revision={revision} visible=20 cities=3 tile_units=4")
            lines.append(f"[C3X renderer] process=8 qpc={request*100+10} stage=usage-result request={request} result=1 call_ms=10")
        lines.append("[C3X renderer] process=8 stage=unit-body id=12 action=3 cache_hit=0 ms=14")
        report = analyze("\n".join(lines))
        usage = report["usage"]
        self.assertEqual(6,usage["completed_calls"])
        self.assertEqual({"initial","stationary","nearby_camera_change","zoom","distant_camera_change","world_change"},set(usage["workloads"]))
        self.assertEqual(3,usage["maximum_observed_tile_counts"]["cities"])
        self.assertNotIn("usage-session.utc_unix_ms",report["timings"])
        self.assertEqual(1,report["unit_activity"]["pose_misses"])
        self.assertEqual(14,report["timings"]["unit-body.call_ms"]["p95_ms"])
        self.assertFalse(usage["workloads"]["zoom"]["hundred_sample_requirement_met"])

    def test_unit_miss_cost_and_mode_evidence(self):
        report=analyze("""[C3X renderer] ms=999999 stage=unit-cache-only-miss reason=cache-miss-color entries=42 bytes=4096
[C3X renderer] ms=999999 stage=unit-body id=1 cache_hit=0 ms=40
[C3X renderer] ms=999999 stage=unit-body id=1 cache_hit=1 ms=0.5
[C3X renderer] ms=999999 stage=unit-body id=2 cache_hit=0
""")
        self.assertEqual(40,report["timings"]["unit-body.miss_call_ms"]["total_ms"])
        self.assertEqual(.5,report["timings"]["unit-body.hit_call_ms"]["total_ms"])
        self.assertEqual({"cache-miss-color":1},report["unit_activity"]["miss_reasons"])
        self.assertEqual(42,report["unit_activity"]["maximum_miss_entries"])
        self.assertFalse(report["native_handoff_observed"])
        self.assertTrue(analyze("[C3X renderer] stage=native-handoff")['native_handoff_observed'])

    def test_usage_rejects_partial_invalid_and_cross_session_pairs(self):
        report=analyze("""[C3X renderer] process=8 qpc=1 stage=usage-session qpc_frequency=1000
[C3X renderer] process=8 qpc=10 stage=usage-view request=1 origin_valid=0
[C3X renderer] process=9 qpc=20 stage=usage-result request=1 result=1 call_ms=10
[C3X renderer] process=8 qpc=20 stage=usage-result request=1 result=1 call_ms=nan
[C3X renderer] process=8 qpc=30 stage=usage-view request=2 origin_valid=0
[C3X renderer] process=8 qpc=40 stage=usage-session qpc_frequency=1000
[C3X renderer] process=8 qpc=50 stage=usage-result request=2 result=1 call_ms=10
[C3X renderer] process=8 qpc=60 stage=usage-view request=3 origin_valid=0
[C3X renderer] process=8 qpc=70 stage=usage-result request=3 result=-1 call_ms=10
""")
        usage=report["usage"]
        self.assertEqual(0,usage["completed_calls"])
        self.assertEqual(1,usage["unmatched_begins"])
        self.assertEqual(2,usage["unmatched_results"])
        self.assertEqual(1,usage["invalid_records"])
        self.assertEqual(1,usage["failed_calls"])

    def test_cadence_configuration_and_speculative_duration_are_distinct(self):
        report=analyze("""[C3X renderer] ms=999999 stage=timer-return interval_ms=33 visual_only=1 total_ms=4
[C3X renderer] ms=999999 stage=timer-return interval_ms=66 visual_only=0 total_ms=8
[C3X renderer] ms=999999 stage=unit-pixels-prepared built=2 ms=45
[C3X renderer] ms=999999 stage=unit-pixels-prepared built=0
""")
        self.assertEqual({"interval_33_visual_1":1,"interval_66_visual_0":1},report["native_timer_opportunities"])
        self.assertNotIn("timer-return.interval_ms",report["timings"])
        self.assertEqual(45,report["timings"]["unit-pixels-prepared.call_ms"]["total_ms"])
        self.assertEqual({"with_output":1,"without_output":1},report["unit_pixel_jobs"])

    def test_native_animation_clock_is_not_render_duration(self):
        report=analyze("[C3X renderer] stage=map-complete total_ms=50 animation_ms=59000")
        self.assertNotIn("map-complete.animation_ms",report["timings"])
        self.assertEqual(50,report["timings"]["map-complete.total_ms"]["p50_ms"])

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
