"""Clock/lifecycle rejection controls for external window evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from Renderer.tools.inspect_window_witness import correlate, inspect


class WindowEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.frequency = 24_000_000
        self.origin = 4_000_000_000_000
        self.utc = 134_344_000_000_000_000
        self.started = {'qpc_frequency': self.frequency,
                        'started_qpc': self.origin + self.frequency,
                        'qpc_after_utc': self.origin + self.frequency + 24,
                        'utc_filetime_100ns': self.utc + 10_000_000}
        self.report = {'verified_prefix': True, 'complete': False, 'qpc_origin': self.origin,
                       'frequency': self.frequency, 'utc_filetime_100ns': self.utc,
                       'qpc_utc_bracket_ticks': 24, 'duration_seconds': 5, 'accepted_presentations': 2}
        self.frame = {'compositor_100ns': (self.origin + self.frequency * 2) * 10_000_000 // self.frequency}
        self.write_inputs()

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self):
        (self.root / 'report.json').write_text(json.dumps(self.report))
        calls = [{'frame': n, 'result_ticks': self.frequency * second, 'sequence': n, 'call': n}
                 for n, second in ((1, 1.5), (2, 2.5))]
        (self.root / 'timeline.jsonl').write_text('\n'.join(json.dumps(row) for row in calls))

    def test_candidate_precedes_compositor_and_prefix_stays_incomplete(self):
        frames = [dict(self.frame)]
        alignment = correlate(frames, self.started, self.root)
        self.assertEqual(frames[0]['preceding_presentation_candidate'], 1)
        self.assertAlmostEqual(frames[0]['candidate_age_ms'], 500, places=3)
        self.assertFalse(alignment['input_complete'])

    def test_different_clock_or_nonoverlapping_recording_rejected(self):
        for change in ({'utc_filetime_100ns': self.utc + 50_000_000},
                       {'qpc_frequency': 10_000_000}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                correlate([dict(self.frame)], self.started | change, self.root)
        with self.assertRaises(ValueError):
            correlate([{'compositor_100ns': self.frame['compositor_100ns'] + 100_000_000}], self.started, self.root)

    def test_unverified_input_rejected(self):
        self.report['verified_prefix'] = False
        self.write_inputs()
        with self.assertRaises(ValueError):
            correlate([dict(self.frame)], self.started, self.root)

    def test_interrupted_window_keeps_verified_images_without_claiming_completion(self):
        (self.root / 'started.json').write_text(json.dumps(self.started))
        row = dict(self.frame, frame=1, previous_compositor_100ns=0,
                   arrival_qpc=self.origin + self.frequency * 2,
                   saved_qpc=self.origin + self.frequency * 2 + 24000,
                   width=320, height=240, bytes=3)
        (self.root / 'timeline.jsonl').write_text(json.dumps(row) + '\n')
        (self.root / 'window-000001.jpg').write_bytes(b'jpg')
        summary, _, _ = inspect(self.root)
        self.assertFalse(summary['complete'])
        self.assertFalse(summary['qualified_for_gameplay'])
        self.assertEqual(summary['frames'], 1)
        with (self.root / 'timeline.jsonl').open('a') as stream:
            stream.write('{"frame":2,"compositor_')
        summary, _, _ = inspect(self.root)
        self.assertFalse(summary['complete'])
        self.assertTrue(summary['truncated_final_timeline_record'])
        self.assertEqual(summary['frames'], 1)
        (self.root / 'window-000001.jpg').write_bytes(b'truncated')
        with self.assertRaises(ValueError):
            inspect(self.root)


if __name__ == '__main__':
    unittest.main()
