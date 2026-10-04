import unittest
from Renderer.tools.check_navigation_cadence import check


class NavigationCadenceTests(unittest.TestCase):
    def samples(self):
        return [{'qpc': n * 100, 'frames': n, 'zoom_q16': 65536 if n < 5 else 32768}
                for n in range(501)]

    def test_responsive_zoom_and_continuing_presentations_pass(self):
        self.assertEqual(check(self.samples(), 1000, [(0, 32768)])['status'], 'pass')

    def test_forty_second_stall_fails_even_if_zoom_eventually_finishes(self):
        rows = self.samples()
        for n in range(1, 401):
            rows[n].update(frames=0, zoom_q16=65536)
        result = check(rows, 1000, [(0, 32768)])
        self.assertEqual(result['status'], 'fail')
        self.assertGreater(result['max_observed_no_progress_seconds'], 39)
        self.assertEqual(len(result['failures']), 2)

    def test_missing_observation_is_not_a_successful_freeze_check(self):
        rows = self.samples()
        result = check(rows[:10] + rows[400:], 1000, [(0, 32768)])
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['max_observed_no_progress_seconds'], 0)

    def test_rapid_reversal_is_superseded_but_missing_target_fails(self):
        result = check(self.samples(), 1000, [(0, 57344), (.2, 32768)])
        self.assertEqual(result['status'], 'pass')
        self.assertTrue(result['zoom_completions'][0]['superseded'])
        self.assertEqual(check(self.samples(), 1000, [(0, 57344)])['status'], 'fail')
