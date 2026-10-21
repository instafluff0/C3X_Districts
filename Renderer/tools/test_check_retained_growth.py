import pathlib
import tempfile
import unittest

from Renderer.tools.check_retained_growth import check

# The user's October 21 diplomacy-then-city-screen capture (review 90), as
# (seconds, nodes, slowest ms): every third direct-visual sample plus its worst
# frame. Software-cursor presents ran in about 25-30 s (diplomacy) and
# 35-39 s (city screen).
USER = [(16.14, 125, 18.6), (16.85, 129, 21.5), (17.63, 131, 6.4), (18.43, 144, 5.3), (19.66, 127, 24.9),
        (20.43, 129, 3.1), (21.23, 129, 3.4), (22.03, 150, 6.1), (22.83, 153, 7.3), (23.64, 168, 16.0),
        (24.50, 198, 20.1), (25.03, 214, 22.1), (25.80, 451, 51.1), (26.55, 262, 13.1), (27.34, 675, 217.8),
        (27.74, 749, 11.7), (28.11, 735, 7.6), (28.67, 512, 36.4), (30.54, 1025, 17.7), (31.34, 1004, 9.8),
        (32.11, 1038, 6.1), (32.91, 1038, 3.7), (33.90, 1055, 48.8), (34.67, 1057, 9.5), (35.07, 1071, 9.4),
        (35.55, 168, 5.6), (36.33, 176, 24.4), (37.12, 200, 17.6), (37.93, 180, 11.6), (38.74, 168, 6.1),
        (39.54, 204, 15.5), (40.87, 319, 27.0), (41.63, 241, 9.1), (42.83, 241, 6.0), (44.03, 242, 5.1)]


def capture(frames, cursor=(), depth=None):
    """A renderer.log in DebugView form; depth(t) adds plan_depth."""
    lines = []
    for t, nodes, ms in frames:
        qpc = int(t * 24e6)
        lines.append((t, f'[C3X renderer] qpc={qpc} stage=direct-visual drawn=1 result=1 prepare_ms=0.000 '
                         f'sample_ms={ms:.3f} present_ms=0.1 total_ms={ms:.3f} ready=1 nodes={nodes} bytes=0 '
                         f'operations=1 assemblies=1 copies=1 copied_pixels=1000 attempts=8'))
        plan = '' if depth is None else f' plan_depth={depth(t)}'
        lines.append((t, f'[C3X renderer] qpc={qpc} stage=retained-execution-counts topology_builds=1 '
                         f'topology_reuses=0 batch_builds=0 batch_reuses=0 source_binds=0 source_reuses=0 '
                         f'plan_nodes={nodes}{plan} spatial_plans=0'))
    for t in cursor:
        lines.append((t, '[C3X renderer] stage=native-ui-present result=1 ticket=2 map_image=1 display_image=1 '
                         'area=100,200,132,233'))
    lines.sort()
    directory = pathlib.Path(tempfile.mkdtemp())
    (directory / 'renderer.log').write_text(''.join(
        f'{n}\t{t:.6f}\t[5600]\t{text}\n' for n, (t, text) in enumerate(lines)))
    return directory


def presents(begin, end, rate=20):
    return [begin + n / rate for n in range(int((end - begin) * rate))]


def city_open():
    """Map view, then a city screen that jumps to ~450 nodes and stays there."""
    frames = [(t / 2, 179, 6.) for t in range(0, 60)]
    frames += [(30.0, 233, 20.), (30.5, 287, 30.), (31.0, 359, 40.), (31.5, 446, 60.)]
    frames += [(32 + t / 2, 446 + 13 * (t % 2), 6.) for t in range(76)]
    return frames


class RetainedGrowthTests(unittest.TestCase):
    def test_user_capture_without_depth_fails_on_the_diplomacy_hover(self):
        ok, lines = check(capture(USER, presents(25.1, 29.9) + presents(35.2, 38.9)))
        self.assertFalse(ok, lines)
        diplomacy = [line for line in lines if line.startswith('cursor hover 25.1')]
        self.assertEqual(len(diplomacy), 1)
        self.assertIn('451 -> 749', diplomacy[0])
        self.assertIn('build-up', '\n'.join(lines))

    def test_screen_opening_is_a_lead_not_a_failure(self):
        ok, lines = check(capture(city_open(), presents(40, 70)))
        self.assertTrue(ok, lines)
        self.assertTrue(any(line.startswith('node rise') and '(+267)' in line for line in lines), lines)

    def test_shallow_plans_pass_whatever_the_node_count(self):
        ok, lines = check(capture(USER, presents(25.1, 29.9), depth=lambda t: 7))
        self.assertTrue(ok, lines)

    def test_deepening_plan_fails_with_its_stretch(self):
        def depth(t):
            return int(6 + 30 * (t - 25)) if 25 <= t <= 30 else 6
        ok, lines = check(capture(city_open(), presents(25, 30), depth=depth))
        self.assertFalse(ok, lines)
        stretch = [line for line in lines if line.startswith('build-up')]
        self.assertEqual(len(stretch), 1, lines)
        self.assertIn('build-up 27.0-30.0s: plan_depth 156 > 64', stretch[0])
        self.assertNotIn('cursor-sized presents 0', stretch[0])

    def test_a_deep_but_steady_screen_passes(self):
        # The city screen steps from 12 to 34 when it opens and stays there.
        ok, lines = check(capture(city_open(), presents(40, 70), depth=lambda t: 34 if t >= 31.5 else 12))
        self.assertTrue(ok, lines)
        self.assertTrue(any(line.startswith('deepest plan 34 at') for line in lines), lines)

    def test_hover_mode_requires_the_software_cursor(self):
        ok, lines = check(capture(city_open(), depth=lambda t: 7), require_hover=True)
        self.assertFalse(ok)
        self.assertIn('software cursor was not active', lines[-1])
        self.assertTrue(check(capture(city_open(), presents(40, 70), depth=lambda t: 7), require_hover=True)[0])

    def test_no_frame_samples_is_not_a_pass(self):
        self.assertFalse(check(capture([], presents(1, 5)))[0])


if __name__ == '__main__':
    unittest.main()
