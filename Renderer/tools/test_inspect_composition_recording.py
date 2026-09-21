from pathlib import Path
import struct
import tempfile
import unittest

from Renderer.tools.inspect_composition_recording import inspect


def event(kind, ordinal, payload=b''):
    # Inspector deliberately checks framing only; strict GPU reader checks hash.
    return struct.pack('<IIIIQQQ', 0x31523343, kind, len(payload), 0, ordinal, 0, ordinal * 10) + payload


class InspectRecordingTests(unittest.TestCase):
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
