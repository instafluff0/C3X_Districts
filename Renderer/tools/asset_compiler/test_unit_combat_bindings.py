from __future__ import annotations

import json
import struct
import unittest
from pathlib import Path

import numpy as np

from Renderer.tools.asset_compiler import unit_combat_bindings as combat

MAP = json.loads((Path(__file__).with_name("civ6_combat_effect_map.json")).read_text())


def rig_blob(translations) -> bytes:
    """A one-bone payload whose palette moves the bone per frame (identity bind)."""
    frames = len(translations)
    header = b"C3XANM1\0" + struct.pack("<5I", 1, 0, 0, 1, frames) + bytes(4)
    palettes = b"".join(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [*t, 1]], "<f4").tobytes()
                        for t in translations)
    trailer = b"C3XRIG1\0" + struct.pack("<2I", 1, 0) + bytes(32) + struct.pack("<iI", -1, 0) + \
        np.eye(4, dtype="<f4").tobytes()
    return header + palettes + trailer


def timing(state, releases):
    socket = {"names": ["FX_bone_muzzle"], "local": {"matrix": np.eye(4).ravel().tolist()}}
    return {"nodes": [{"attack": {"state": state}, "components": [{"asset": "body", "sockets": [socket]}]}],
            "releases": releases}


def effect(name, normalized):
    return {"kind": "effect", "asset": "body", "effect": name, "normalized": normalized,
            "socket": "FX_bone_muzzle", "payload_bone_index": 0}


class UnitCombatBindingTests(unittest.TestCase):
    def resolve(self, evidence):
        blob = rig_blob([(0, 0, 0), (0.2, 0.1, 0.3)])
        return combat.resolve(evidence, MAP, ["body"], {"part0": {"mesh": "m.bin"}}, lambda path: blob)

    def test_releases_sockets_impact_set_and_bearing_come_from_evidence(self) -> None:
        result = self.resolve(timing("ATTACK_P", [
            effect("FX_Cannonball_Trail", 0.4),                     # no release effect; naval munition
            effect("FX_Caravel_Cannon_MuzzFlash_01", 1.0),
            {"kind": "impact", "asset": "body", "impact": {"collection": "MaterialVFX", "element": "Bullet_Hit"}},
            effect("FX_Caravel_Cannon_MuzzFlash_02", 0.0),
        ]))
        self.assertEqual(["combat/muzzle_naval", "combat/muzzle_naval"], [r["profile"] for r in result["releases"]])
        self.assertEqual([0.0, 1.0], [r["normalized"] for r in result["releases"]])
        # The socket point follows the bone at the release frame.
        np.testing.assert_allclose([0.2, 0.1, 0.3], result["releases"][1]["position"], atol=1e-6)
        # A munition named by a release outranks the generic impact name.
        self.assertEqual("naval", result["impact_set"])
        self.assertEqual(90, result["bearing"])

    def test_small_arms_and_melee_never_fall_back_to_shells(self) -> None:
        rifle = self.resolve(timing("ATTACK", [effect("FX_Rifle_Muzzleflash", 0.3), {
            "kind": "impact", "asset": "body", "impact": {"collection": "MaterialVFX", "element": "Bullet_Hit"}}]))
        self.assertEqual("bullet", rifle["impact_set"])
        sword = self.resolve(timing("ATTACK", [effect("FX_Smack_Premult", 0.5)]))
        self.assertEqual(("melee", []), (sword["impact_set"], sword["releases"]))
        torpedo = self.resolve(timing("ATTACK", [effect("FX_Torpedo_Gato_Class", 0.5)]))
        self.assertEqual(("torpedo", ["combat/torpedo_launch"]),
                         (torpedo["impact_set"], [r["profile"] for r in torpedo["releases"]]))

    def test_publish_writes_bounded_pack_fields_and_class_fallback(self) -> None:
        binding = {"attack": {"part_count": 1}}
        resolved = self.resolve(timing("ATTACK_S", [effect("FX_Artillery_MuzzleFlash", 0.25)]))
        civ3 = {"ATTACK1": {"duration_s": 1.0, "sync_s": 0.1}}
        combat.publish(binding, resolved, civ3, {"unit_class": 0})
        self.assertEqual("shell", binding["impact_set"])
        self.assertEqual(-90, binding["attack_bearing"])
        self.assertEqual((0.1, 0.25), (binding["attack_sync"], binding["attack_first_release"]))
        self.assertEqual(1, binding["attack"]["release_count"])
        self.assertEqual({"phase", "profile", "x", "y", "z"}, set(binding["attack"]["release0"]))
        # No evidence: the Civ III unit class picks the munition, with no releases or sync.
        bare = {"attack": {}}
        combat.publish(bare, None, civ3, {"unit_class": 1})
        self.assertEqual({"impact_set": "naval", "attack": {}}, bare)
        # Civ III abilities name missiles and nuclear weapons whatever their class.
        for role, munition in (({"unit_class": 0, "nuclear_weapon": True}, "nuclear"),
                               ({"unit_class": 0, "cruise_missile": True}, "missile")):
            unit = {"attack": {}}
            combat.publish(unit, None, None, role)
            self.assertEqual(munition, unit["impact_set"])
        bomber = {"attack": {}}
        combat.publish(bomber, None, {"VICTORY": {"duration_s": 1.66, "sync_s": 0.638}}, {"unit_class": 2})
        self.assertEqual(0.638, bomber["bomb_blast_s"])

    def test_far_off_reports_do_not_freeze_or_rush_the_clip(self) -> None:
        resolved = self.resolve(timing("ATTACK", [effect("FX_SAM_Launch_01", 0.0)]))
        late = {"attack": {}}
        combat.publish(late, resolved, {"ATTACK1": {"duration_s": 1.0, "sync_s": 0.9}}, {"unit_class": 2})
        self.assertNotIn("attack_sync", late)
        self.assertTrue(combat.warp_is_gentle(0.188, 0.166))   # battleship: nearly aligned
        self.assertTrue(combat.warp_is_gentle(0.074, 0.247))   # artillery: skips 19% of the wind-up
        self.assertFalse(combat.warp_is_gentle(0.05, 0.6))     # would skip over half the clip
        self.assertFalse(combat.warp_is_gentle(0.08, 0.0))     # cannot hold phase 0 until the report

    def test_clip_warp_lands_first_release_on_native_sync_and_inverts(self) -> None:
        for sync, first in ((0.07, 0.25), (0.19, 0.17), (None, 0.3)):
            if sync is not None:
                self.assertAlmostEqual(first, combat.clip_phase(sync, sync, first))
            self.assertAlmostEqual(1.0, combat.clip_phase(1.0, sync, first))
            for phase in np.linspace(0, 1, 11):
                clip = combat.clip_phase(phase, sync, first)
                self.assertAlmostEqual(phase, combat.native_phase(clip, sync, first))
                self.assertGreaterEqual(clip, 0.0)


if __name__ == "__main__":
    unittest.main()
