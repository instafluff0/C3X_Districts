from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler.unit_member_resolver import ASSETS_ROOT, resolve_unit


@unittest.skipUnless(ASSETS_ROOT.is_dir(), "local Civ VI assets are unavailable")
class UnitMemberResolverTests(unittest.TestCase):
    def test_warrior_recipe_preserves_formation_bins_tint_and_candidates(self) -> None:
        recipe = resolve_unit(ASSETS_ROOT, "UNIT_WARRIOR", "Any")
        self.assertEqual("Warrior", recipe["member"]["type"])
        self.assertEqual(4, recipe["member"]["count"])
        self.assertEqual(4, len(recipe["formation"]["combat_offsets"]))
        selected = {item["role"]: item for item in recipe["selected_components"]}
        self.assertEqual("Male_Caucasian_BuffBody_FullA", selected["Body"]["source_entry"])
        self.assertEqual("Male_Cauc_Head_01", selected["Head"]["source_entry"])
        self.assertEqual("Warrior_Weapon_01", selected["Weapon"]["source_entry"])
        self.assertEqual("Warrior_Armor_01", selected["Armor"]["source_entry"])
        self.assertEqual("USE_CIV_COLOR", selected["Armor"]["tint"])
        hair = next(item for item in recipe["member"]["attachments"] if item["role"] == "Hair")
        self.assertEqual(2, len(hair["bins"]))
        self.assertEqual("EmptyUnitAttachment", hair["bins"][0]["selection"]["source_entry"])

    def test_great_general_recipe_honors_explicit_bin_candidates(self) -> None:
        recipe = resolve_unit(ASSETS_ROOT, "UNIT_GREAT_GENERAL", "Any")
        selected = {item["role"]: item for item in recipe["selected_components"]}
        self.assertEqual("GreatGeneral_Classical_Male", recipe["member"]["type"])
        self.assertEqual("GreatGeneral_Classical_SaddleA", selected["Armor"]["source_entry"])
        self.assertEqual(
            [{"role": "Rider", "point": "RiderAttach", "member": "Rider"}],
            recipe["virtual_attachments"],
        )
        armor = next(item for item in recipe["member"]["attachments"] if item["role"] == "Armor")
        self.assertEqual(
            "explicit # candidate in exact culture, otherwise Any",
            armor["bins"][0]["selection_rule"],
        )

    def test_classical_great_general_rider_can_be_resolved_as_compound_part(self) -> None:
        recipe = resolve_unit(ASSETS_ROOT, "UNIT_GREAT_GENERAL", "Any", "Rider")
        self.assertEqual("Rider", recipe["member"]["variation"])
        self.assertTrue(recipe["member"]["is_attachment"])
        self.assertEqual([], recipe["virtual_attachments"])
        selected = {item["role"]: item for item in recipe["selected_components"]}
        self.assertEqual("GreatGeneral_Classical_Male", selected["Armor"]["source_entry"])
        self.assertEqual("GreatGeneral_Classical_Sword", selected["Weapon"]["source_entry"])

    def test_catapult_member_recipes_can_be_resolved_independently(self) -> None:
        vehicle = resolve_unit(ASSETS_ROOT, "UNIT_CATAPULT", "Any", member_index=0)
        defender = resolve_unit(ASSETS_ROOT, "UNIT_CATAPULT", "Any", member_index=1)
        self.assertEqual(2, vehicle["member"]["recipe_count"])
        self.assertEqual(0, vehicle["member"]["recipe_index"])
        self.assertEqual(1, defender["member"]["recipe_index"])
        self.assertEqual("Catapult", vehicle["member"]["type"])
        self.assertEqual("CatapultDefender", defender["member"]["type"])

    def test_virtual_rider_operator_and_gunner_have_fetchable_variations(self) -> None:
        horse = resolve_unit(ASSETS_ROOT, "UNIT_HORSEMAN", "Any")
        horse_rider = resolve_unit(ASSETS_ROOT, "UNIT_HORSEMAN", "Any", "Rider")
        catapult = resolve_unit(ASSETS_ROOT, "UNIT_CATAPULT", "Any", member_index=0)
        operator = resolve_unit(
            ASSETS_ROOT, "UNIT_CATAPULT", "Any", "OperatorA", member_index=0
        )
        tank = resolve_unit(ASSETS_ROOT, "UNIT_TANK", "Any")
        gunner = resolve_unit(ASSETS_ROOT, "UNIT_TANK", "Any", "TankGunner")
        self.assertEqual("Rider", horse["virtual_attachments"][0]["member"])
        self.assertEqual("Rider", horse_rider["member"]["variation"])
        self.assertEqual("OperatorA", catapult["virtual_attachments"][0]["member"])
        self.assertEqual("OperatorA", operator["member"]["variation"])
        self.assertEqual("TankGunner", tank["virtual_attachments"][0]["member"])
        self.assertEqual("TankGunner", gunner["member"]["variation"])


if __name__ == "__main__":
    unittest.main()
