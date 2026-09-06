from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler import combat_effect_contract as contract


class CombatEffectContractTests(unittest.TestCase):
    def test_contract_preserves_authority_and_atomic_fallback(self) -> None:
        document = contract.load_contract()
        self.assertEqual("civ3", document["authority"]["gameplay"])
        self.assertEqual("civ3", document["authority"]["audio"])
        self.assertEqual("native_unit_and_native_effect", document["bindings"]["default_when_unmapped"])
        self.assertTrue(document["ownership"]["double_effects_forbidden"])

    def test_native_suppression_preserves_audio_and_nuclear_outcomes_are_explicit(self) -> None:
        document = contract.load_contract()
        bridge = document["native_bridge"]
        self.assertIn("low byte", bridge["pixel_suppression"]["field_access"])
        self.assertIn("0x184", bridge["pixel_suppression"]["field_access"])
        self.assertIn("Unit::do_nuke_tile", bridge["nuclear_policy"]["detonated_boundary"])
        self.assertIn(
            "Unit::get_intercepted_as_nuke",
            bridge["nuclear_policy"]["intercepted_boundary"],
        )
        self.assertNotIn(
            "detonation",
            bridge["nuclear_policy"]["delivery"].lower().replace("never evidence of detonation", ""),
        )

    def test_absolute_trace_sampling_skips_frames_without_extending_event(self) -> None:
        event = {
            "event_id": "bombard/17/impact/0",
            "profile_id": "ballistic_shell",
            "spawn_ms": 1000,
            "release_ms": 1200,
            "impact_ms": 1600,
            "cleanup_ms": 2500,
        }
        self.assertEqual("staged", contract.sample_event(event, 1100)["state"])
        flight = contract.sample_event(event, 1400)
        self.assertEqual("flight", flight["state"])
        self.assertAlmostEqual(0.5, flight["phase"])
        self.assertEqual("impact", contract.sample_event(event, 2400)["state"])
        self.assertEqual("complete", contract.sample_event(event, 9000)["state"])

    def test_interruption_cleans_up_immediately(self) -> None:
        event = {
            "event_id": "bombard/17/impact/0",
            "profile_id": "ballistic_shell",
            "spawn_ms": 1000,
            "release_ms": 1200,
            "impact_ms": 1600,
            "cleanup_ms": 2500,
            "interrupted_ms": 1450,
        }
        self.assertEqual("flight", contract.sample_event(event, 1449)["state"])
        sampled = contract.sample_event(event, 1450)
        self.assertEqual("interrupted", sampled["state"])
        self.assertFalse(sampled["active"])


if __name__ == "__main__":
    unittest.main()
