import unittest

from Renderer.tools.arbitrary_scenario_stress import run_stress


class ArbitraryScenarioStressTests(unittest.TestCase):
    def test_arbitrary_ids_palette_override_and_failure_policy(self) -> None:
        result = run_stress()
        self.assertEqual(9, result["summary"]["cases"])
        self.assertEqual(31, result["summary"]["arbitrary_palette_row"])
        self.assertEqual(1, result["summary"]["hard_failures"])
        self.assertEqual(1, result["summary"]["native_disabled"])
        self.assertEqual([["cities", "city_style/scenario_obsidian"]], result["partial_override_changed_rules"])
        self.assertEqual("none", result["runtime_activation"])


if __name__ == "__main__":
    unittest.main()
