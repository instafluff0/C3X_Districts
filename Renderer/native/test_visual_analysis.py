import unittest
from Renderer.native.analyze_visual_frames import analyze


class VisualAnalysisTests(unittest.TestCase):
    def test_missing_evidence_is_not_zero_work(self):
        result=analyze('VISUAL_SAMPLE begin_qpc=10 end_qpc=20 request_ms=1 desktop_ms=2','')
        self.assertFalse(any(result['proof'].values()))

    def test_exact_interval_rejects_builds_and_ignores_setup(self):
        log='VISUAL_SAMPLE begin_qpc=10 end_qpc=20 request_ms=1 desktop_ms=2'
        trace='qpc=9 stage=frame built=100 upload_bytes=100\nqpc=15 stage=frame built=0 upload_bytes=0'
        self.assertTrue(analyze(log,trace)['proof']['static_world_builds_zero'])
        self.assertFalse(analyze(log,trace+'\nqpc=19 stage=frame built=1 upload_bytes=0')['proof']['static_world_builds_zero'])
        self.assertFalse(analyze(log,'qpc=15 stage=frame built=0')['proof']['static_world_builds_zero'])

    def test_reversed_interval_fails(self):
        with self.assertRaises(ValueError):
            analyze('VISUAL_SAMPLE begin_qpc=20 end_qpc=10 request_ms=1 desktop_ms=2','')

    def test_missing_later_frame_is_not_proved_by_first_frame(self):
        log='VISUAL_SAMPLE begin_qpc=10 end_qpc=20 request_ms=1 desktop_ms=2\nVISUAL_SAMPLE begin_qpc=30 end_qpc=40 request_ms=1 desktop_ms=2'
        trace='qpc=15 stage=frame built=0 upload_bytes=0\nqpc=16 stage=material-submission reused=1\nqpc=17 stage=shared-scene-surface static_selected=0 readbacks=0'
        self.assertFalse(any(analyze(log,trace)['proof'].values()))
