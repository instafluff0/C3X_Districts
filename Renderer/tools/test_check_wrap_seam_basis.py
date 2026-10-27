import unittest

from Renderer.tools.check_wrap_seam_basis import misplaced


def draw(qpc, camera, translation):
    return (f'[C3X renderer] qpc={qpc} ms=1 process=1 thread=1 sequence=1 stage=fresh-scene-phases prepare=1 '
            f'camera={camera},-1632 translation={translation:.1f},0.0 pose_samples=1')


WINDOW = '[C3X renderer] qpc=1 stage=canonical-membership covered=0 incremental=1 required=1 entering=1 leaving=1'
COVERED = '[C3X renderer] qpc=1 stage=canonical-membership covered=1 incremental=1 required=1 entering=0 leaving=0'


class WrapSeamBasisCheckTests(unittest.TestCase):
    # The user's 2950 BC map (100 tiles wide) at the step from x=0 to x=64,
    # as traced: the window kept basis 6336 while the camera re-based by the
    # world width on a covered step (own basis -64).
    lines = [WINDOW, draw(10, -6400, -64), COVERED, draw(20, -6400, -64), COVERED, draw(30, -64, -128)]

    def test_a_kept_basis_across_a_covered_re_base_is_off_screen(self):
        draws, found = misplaced(self.lines + [draw(40, -64, -128)], 6400)
        self.assertEqual(draws, 4)
        self.assertEqual([qpc for qpc, *_ in found], [30, 40])

    def test_a_traced_re_base_keeps_the_layers_on_screen(self):
        fixed = self.lines[:-1] + ['[C3X renderer] qpc=29 stage=window-basis-rebase basis=-64 rebase=-6400 camera=-64',
                                   self.lines[-1]]
        draws, found = misplaced(fixed + [draw(40, -64, -128)], 6400)
        self.assertEqual((draws, found), (4, []))

    def test_a_changed_window_takes_the_draws_own_basis(self):
        draws, found = misplaced(self.lines + [WINDOW, draw(40, -64, -128)], 6400)
        self.assertEqual([qpc for qpc, *_ in found], [30])


if __name__ == '__main__':
    unittest.main()
