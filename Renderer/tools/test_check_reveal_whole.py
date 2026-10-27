import unittest

from Renderer.tools.check_reveal_whole import pieces

SESSION = '[C3X renderer] qpc=1 ms=1 process=1 thread=1 sequence=0 stage=usage-session schema=1 qpc_frequency=24000000'


def shown(ms, cells, age):
    return (f'[C3X renderer] qpc={int(ms * 24000)} ms=1 process=1 thread=1 sequence=1 '
            f'stage=reveal-shown cells={cells} capture_age_ms={age:.1f}')


class RevealWholeCheckTests(unittest.TestCase):
    def test_one_capture_shown_in_two_samples_is_split(self):
        # The pre-fix reveal-scroll Scout step: 4 cells on the drawn view,
        # then 1 cell 72 ms later with the next view, from one capture.
        reveals = pieces([SESSION, shown(1000.0, 4, 93.6), shown(1072.2, 1, 165.8)])
        self.assertEqual(len(reveals), 1)
        self.assertEqual([cells for _, cells in reveals[0]['samples']], [4, 1])

    def test_whole_reveals_from_separate_captures_pass(self):
        reveals = pieces([SESSION, shown(1000.0, 3, 129.9), shown(7000.0, 5, 165.8)])
        self.assertEqual([len(reveal['samples']) for reveal in reveals], [1, 1])


if __name__ == '__main__':
    unittest.main()
