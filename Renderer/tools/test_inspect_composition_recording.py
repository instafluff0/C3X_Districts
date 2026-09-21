from pathlib import Path
import io
import json
import struct
import tempfile
import unittest

from Renderer.tools.inspect_composition_recording import inspect


def event(kind, ordinal, payload=b''):
    # Inspector deliberately checks framing only; strict GPU reader checks hash.
    return struct.pack('<IIIIQQQ', 0x31523343, kind, len(payload), 0, ordinal, 0, ordinal * 10) + payload


class InspectRecordingTests(unittest.TestCase):
    def test_ten_minute_timeline_is_not_input_replay_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'ten-minutes.c3xr'
            header = struct.pack('<IIQ', 0x52433343, 3, 1000)
            def timed(kind, ordinal, ticks, payload):
                return struct.pack('<IIIIQQQ', 0x31523343, kind, len(payload), 0, ordinal, 7, ticks) + payload
            display = struct.pack('<QIiiii', 31, 1, 0, 0, 2240, 1260)
            rejected = struct.pack('<QIiiii', 31, 0, 0, 0, 2240, 1260)
            data = header + timed(9, 1, 0, display) + timed(9, 2, 300000, rejected)
            data += timed(9, 3, 600000, display)
            footer = bytearray(event(14, 4, struct.pack('<I', 0)))
            struct.pack_into('<Q', footer, 32, 600000)
            file.write_bytes(data + footer)
            timeline = io.StringIO()
            report = inspect(file, timeline=timeline)
            entries = [json.loads(line) for line in timeline.getvalue().splitlines()]
            self.assertEqual([e['display'] for e in entries], [1, 2])
            self.assertEqual(entries[1]['seconds'], 600)
            self.assertEqual(entries[1]['file_offset'], 16 + 68 * 2)
            self.assertFalse(entries[1]['physical_display_confirmed'])
            self.assertEqual(report['maximum_between_display_gap_seconds'], 600)
            self.assertEqual(report['capacity']['linear_projection_bytes'], len(data + footer) - 16)
            self.assertEqual(report['event_bytes']['display'], 3 * 68)
            self.assertEqual(len(report['seconds']), 3)
            self.assertTrue(report['seconds'][-1]['aggregated_tail'])
            self.assertFalse(report['renderer_input_replay_ready'])
            self.assertIn('retained_ambient_inputs_and_visual_opportunities', report['missing_input_families'])
            # A valid sequence cannot conceal a reversed recorded clock.
            file.write_bytes(header + timed(9, 1, 100, display) + timed(9, 2, 99, display))
            with self.assertRaisesRegex(ValueError, 'clock moved backwards'):
                inspect(file)

    def test_thread_evidence_version_preserves_legacy_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'threads.c3xr'
            header = struct.pack('<IIQ', 0x52433343, 3, 1000)
            payload = struct.pack('<QIQQIiI', 42, 108, 1, 2, 0, 0, 77)
            file.write_bytes(header + event(10, 1, payload) + event(11, 2, struct.pack('<Qi', 42, 1)))
            self.assertEqual(inspect(file)['unfinished_native_calls'], 0)
            file.write_bytes(header + event(10, 1, payload[:-4]))
            with self.assertRaisesRegex(ValueError, 'native begin'):
                inspect(file)

    def test_native_observations_and_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'sample.c3xr'
            header = struct.pack('<IIQ', 0x52433343, 2, 1000)
            begin = event(10, 1, struct.pack('<QIQQIi', 42, 108, 1, 2, 0, 0))
            end = event(11, 2, struct.pack('<Qi', 42, 0))
            visual = event(13, 3, struct.pack('<QQIQQQI', 100, 1000, 1, 4096, 2, 1, 1))
            data = header + begin + end + visual
            file.write_bytes(data + event(14, 4, struct.pack('<I', 0)))
            report = inspect(file)
            self.assertEqual(report['stop_reason'], 'closed')
            self.assertEqual(report['native_zero_or_negative_results'], {108: 1})
            self.assertEqual(report['visual_ready_observations'], 1)
            self.assertFalse(report['payload_integrity_checked'])
            self.assertEqual(report['unfinished_native_calls'], 0)
            file.write_bytes(data + event(14, 4, struct.pack('<I', 0))[:-1])
            report = inspect(file)
            self.assertTrue(report['truncated_tail'])
            self.assertEqual(report['valid_prefix_bytes'], len(data))
            self.assertIsNone(report['stop_reason'])
            file.write_bytes(header + event(11, 1, struct.pack('<Qi', 42, 0)))
            with self.assertRaisesRegex(ValueError, 'Unmatched'):
                inspect(file)
            file.write_bytes(header + event(14, 1, struct.pack('<I', 0)) + begin)
            with self.assertRaisesRegex(ValueError, 'after footer'):
                inspect(file)


if __name__ == '__main__':
    unittest.main()
