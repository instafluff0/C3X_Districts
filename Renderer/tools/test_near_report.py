import json
import pathlib
import tempfile
import unittest
from Renderer.tools.near_report import report


class NearReportJumpTests(unittest.TestCase):
    # Civ III can move the camera on the minimap press, before the release.
    # Timing from the release skipped that move and reported the next distant
    # handoff (a later jump or scroll) as a 4-10 s jump (October 8).
    def capture(self, folder, moves):
        f = 1000
        events = [n * 1000 for n in range(25)]
        (folder / 'mouse-events.json').write_text(json.dumps({'events': [{'qpc': q} for q in events]}))
        samples = [{'qpc': q, 'frames': q // 16} for q in range(0, 26000, 20)]
        (folder / 'cadence.json').write_text(json.dumps({'qpc_frequency': f, 'samples': samples}))
        lines = [f'1\t0.0\t[1]\t[C3X renderer] qpc=0 stage=native-handoff requested=0,0 displayed=0,0']
        for q, x, y in moves:
            lines.append(f'1\t{q / f}\t[1]\t[C3X renderer] qpc={q} stage=native-handoff requested={x},{y} displayed={x},{y}')
        (folder / 'renderer.log').write_text('\n'.join(lines) + '\n')

    def jumps(self, moves):
        with tempfile.TemporaryDirectory() as name:
            folder = pathlib.Path(name)
            self.capture(folder, moves)
            return {row['segment']: row['jump_ms'] for row in report(folder) if row['segment'].startswith('jump')}

    def test_move_between_press_and_release_counts_from_the_press(self):
        # Presses are events 4 and 6 (4000 and 6000), releases 5 and 7.
        result = self.jumps([(4040, 3488, 0), (6030, 2016, 524), (16000, 2144, 524)])
        self.assertEqual(result, {'jump 1': 40.0, 'jump 2': 30.0})

    def test_ignored_click_reports_no_jump_rather_than_a_later_move(self):
        result = self.jumps([(6030, 2016, 524), (16000, 4144, 524)])
        self.assertIsNone(result['jump 1'])
        self.assertEqual(result['jump 2'], 30.0)


if __name__ == '__main__':
    unittest.main()
