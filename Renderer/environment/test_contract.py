from __future__ import annotations

import copy
import unittest
from pathlib import Path

from Renderer.environment import contract


FIXTURE = Path(__file__).parents[1] / "samples" / "environment" / "m6_4_environment.fixture.json"


class EnvironmentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = contract.load_fixture(FIXTURE)
        self.flame = self.fixture["ambient_attachments"][0]

    def test_two_sizes_and_four_environment_phases(self) -> None:
        self.assertEqual([[320, 200], [640, 480]], self.fixture["viewports"])
        states = {name: contract.evaluate_environment(hour) for name, hour in self.fixture["hours"].items()}
        self.assertGreater(states["noon"].sun_intensity, states["sunset"].sun_intensity)
        self.assertGreater(states["midnight"].moon_intensity, states["noon"].moon_intensity)
        self.assertGreater(states["midnight"].emissive_scale, states["noon"].emissive_scale)
        self.assertTrue(all(math_value >= 0 for state in states.values() for math_value in (
            state.exposure, state.water_fresnel, state.water_specular)))

    def test_static_emissive_is_active_at_night_but_never_animated(self) -> None:
        state = contract.evaluate_environment(0)
        self.assertGreater(contract.activation(self.fixture["material"]["emissive"]["activation"], state), 0.99)
        static = copy.deepcopy(self.flame)
        static["animated"] = False
        status = contract.attachment_status(static, state, 2_468_000, current_states={"operational"})
        self.assertTrue(status["active"])
        self.assertFalse(status["animated"])

    def test_absolute_phase_survives_frame_skip_pause_and_reset(self) -> None:
        state = contract.evaluate_environment(0)
        kwargs = {"current_states": {"operational"}}
        first = contract.attachment_status(self.flame, state, 2_468_000, **kwargs)
        after_skips = contract.attachment_status(self.flame, state, 2_468_000, **kwargs)
        after_reset = contract.attachment_status(self.flame, state, 2_468_000, **kwargs)
        self.assertTrue(first["animated"])
        self.assertEqual(first["phase_millionths"], after_skips["phase_millionths"])
        self.assertEqual(first, after_reset)
        paused = contract.attachment_status(self.flame, state, 2_468_000, visible=False, **kwargs)
        self.assertFalse(paused["animated"])

    def test_visibility_state_and_missing_resources_degrade_explicitly(self) -> None:
        state = contract.evaluate_environment(0)
        hidden = contract.attachment_status(self.flame, state, 100, visible=False, current_states={"operational"})
        wrong_state = contract.attachment_status(self.flame, state, 100, current_states={"damaged"})
        missing = contract.attachment_status(
            self.flame, state, 100, resources_available=False, current_states={"operational"})
        fallback = contract.attachment_status(
            self.flame, state, 100, owner_replaced=False, current_states={"operational"})
        self.assertEqual("omit-attachment", hidden["degrade"])
        self.assertEqual("omit-attachment", wrong_state["degrade"])
        self.assertEqual("owner-fallback", missing["degrade"])
        self.assertEqual("owner-fallback", fallback["degrade"])
        self.assertTrue(all(not item["animated"] for item in (hidden, wrong_state, missing, fallback)))

    def test_contract_rejects_unresolved_light_and_preserves_retained_layers(self) -> None:
        invalid = copy.deepcopy(self.fixture)
        invalid["ambient_attachments"][0]["light_id"] = "missing"
        with self.assertRaises(contract.EnvironmentContractError):
            contract.validate_fixture(invalid)
        self.assertEqual(
            {"fog", "labels", "selection", "minimap", "hud", "ui"},
            set(self.fixture["retained_civ3_layers"]),
        )


if __name__ == "__main__":
    unittest.main()

