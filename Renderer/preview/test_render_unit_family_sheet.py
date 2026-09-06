from __future__ import annotations

import unittest

from Renderer.preview.render_unit_family_sheet import BASIC_ACTIONS, _fit_cell, _projected_bounds
from Renderer.tools.asset_compiler.unit_family_action_validator import SOCKET_PROFILE


class RenderUnitFamilySheetTests(unittest.TestCase):
    def test_projected_bounds_and_fit_are_finite_and_inside_cell(self) -> None:
        mesh = {
            "vertices": [
                {"position": [-1.0, -1.0, 0.0]},
                {"position": [1.0, -1.0, 0.0]},
                {"position": [0.0, 1.0, 2.0]},
            ]
        }
        bounds = _projected_bounds([(mesh, None)], 1.0)
        scale, center = _fit_cell(bounds, 220, 210)
        self.assertGreater(scale, 0.0)
        self.assertTrue(all(isinstance(value, int) for value in center))
        self.assertLess(bounds[0], bounds[1])
        self.assertLess(bounds[2], bounds[3])

    def test_inferred_profile_covers_proof_pack_rigid_points(self) -> None:
        self.assertEqual(
            {"Root", "ArmBand", "Hat", "WeaponPrimary", "WeaponSecondary"},
            set(SOCKET_PROFILE),
        )

    def test_basic_sheet_is_deliberately_limited_to_eight_actions(self) -> None:
        self.assertEqual(
            ("idle", "fidget", "move", "fortify", "attack", "defend", "victory", "death"),
            BASIC_ACTIONS,
        )


if __name__ == "__main__":
    unittest.main()
