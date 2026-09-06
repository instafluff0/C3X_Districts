from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler.unit_action_validator import ACTIONS, CORE_BONES, SOCKET_PROFILE


class UnitActionValidatorTests(unittest.TestCase):
    def test_first_slice_has_four_actions_and_explicit_socket_evidence(self) -> None:
        self.assertEqual(("idle", "move", "attack", "death"), ACTIONS)
        self.assertIn("Head", CORE_BONES)
        self.assertEqual("Head", SOCKET_PROFILE["Hat"]["bone"])
        self.assertEqual("Inven_R_Hand", SOCKET_PROFILE["WeaponPrimary"]["bone"])
        self.assertTrue(all(item["status"] == "inferred_lab_profile" for item in SOCKET_PROFILE.values()))


if __name__ == "__main__":
    unittest.main()
