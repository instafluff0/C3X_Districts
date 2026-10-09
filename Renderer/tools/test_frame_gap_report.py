import json
import pathlib
import tempfile
import unittest

from Renderer.tools.frame_gap_report import report


class FrameGapReportTests(unittest.TestCase):
    def test_gaps_are_measured_within_each_segment(self):
        f = 24_000_000
        with tempfile.TemporaryDirectory() as name:
            folder = pathlib.Path(name)
            events = [(10 + n) * f for n in range(25)]
            (folder / 'mouse-events.json').write_text(json.dumps({'events': [{'qpc': q} for q in events]}))
            # 1x idle is the 8 s before event 0: frames every 16 ms with one
            # 50 ms gap; 1x scroll x runs from event 0 to event 1.
            presents = [int(2.5 * f + n * 0.016 * f) for n in range(300)]
            presents.append(presents[-1] + int(0.050 * f))
            presents += [presents[-1] + int((n + 1) * 0.016 * f) for n in range(165)]
            presents += [events[0] + int(n * 0.040 * f) for n in range(20)]
            lines = [f'[C3X renderer] qpc={q} stage=route-presented present_index={i} frequency={f} present_qpc={q} pan_x=0'
                     for i, q in enumerate(presents)]
            (folder / 'renderer-core.log.x64').write_text('\n'.join(lines) + '\n')
            rows = {row['segment']: row for row in report(folder)}
        idle, scroll = rows['1x idle'], rows['1x scroll x']
        self.assertEqual(idle['gap_p50'], 16.0)
        self.assertEqual(idle['gap_max'], 50.0)
        self.assertEqual(idle['over_30ms'], 1)
        self.assertEqual(scroll['gap_p50'], 40.0)
        self.assertEqual(scroll['over_30ms'], 19)


if __name__ == '__main__':
    unittest.main()
