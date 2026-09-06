#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_project


class VerificationTests(unittest.TestCase):
    def test_completed_status_only_references_registered_checks(self) -> None:
        status = json.loads(verify_project.DEFAULT_STATUS.read_text(encoding="utf-8"))
        portable, local = verify_project.collect_completed_checks(status)
        self.assertTrue(portable)
        self.assertEqual(set(portable + local) - set(verify_project.CHECKS), set())
        self.assertEqual(len(portable), len(set(portable)))
        self.assertEqual(len(local), len(set(local)))

    def test_synthetic_civbig_gate_passes(self) -> None:
        result = verify_project.check_civbig_synthetic_roundtrip()
        self.assertEqual(result["status"], "pass", result["detail"])

    def test_preview_gate_passes(self) -> None:
        result = verify_project.check_preview_smoke()
        self.assertEqual(result["status"], "pass", result["detail"])

    def test_native_bridge_gate_is_registered(self) -> None:
        self.assertIn("native_civ3_bridge", verify_project.CHECKS)

    def test_m5_2_scene_export_gate_is_registered(self) -> None:
        self.assertIn("m5_2_scene_export", verify_project.CHECKS)

    def test_m6_3_terrain_gates_are_registered(self) -> None:
        self.assertIn("m6_3_definition_terrain", verify_project.CHECKS)
        self.assertIn("m6_3_local_terrain_art", verify_project.CHECKS)

    def test_m6_5_connected_terrain_gate_is_registered(self) -> None:
        self.assertIn("m6_5_connected_terrain", verify_project.CHECKS)

    def test_m6_6_terrain_gates_are_registered(self) -> None:
        self.assertIn("m6_6_vanilla_terrain", verify_project.CHECKS)
        self.assertIn("m6_6_local_biq_terrain", verify_project.CHECKS)

    def test_m6_7_approved_terrain_gates_are_registered(self) -> None:
        self.assertIn("m6_7_approved_terrain_integration", verify_project.CHECKS)
        self.assertIn("m6_7_local_approved_terrain_payload", verify_project.CHECKS)
        self.assertIn("i11_approved_marsh_integration", verify_project.CHECKS)
        self.assertIn("i11_local_approved_marsh_payload", verify_project.CHECKS)
        self.assertIn("i13_approved_river_integration", verify_project.CHECKS)
        self.assertIn("i13a_approved_lighting_integration", verify_project.CHECKS)

    def test_m6_7_portable_contract_gate_passes(self) -> None:
        result = verify_project.check_m6_7_approved_terrain_integration()
        self.assertEqual(result["status"], "pass", result["detail"])

    def test_l11_handoff_gate_uses_filename_before_gate_id(self) -> None:
        with mock.patch.object(verify_project, "check_lab_handoff", return_value={"status": "pass"}) as check:
            verify_project.check_terrain_lab_l11_handoff()
        self.assertEqual("L11_marsh.json", check.call_args.args[0])
        self.assertEqual("L11", check.call_args.args[1])

    def test_l12_handoff_gate_uses_192_tile_contract(self) -> None:
        with mock.patch.object(verify_project, "check_lab_handoff", return_value={"status": "pass"}) as check:
            verify_project.check_terrain_lab_l12_handoff()
        self.assertEqual("L12_volcano.json", check.call_args.args[0])
        self.assertEqual("L12", check.call_args.args[1])
        self.assertEqual(192, check.call_args.args[2])

    def test_l13a_handoff_gate_uses_192_tile_contract(self) -> None:
        with mock.patch.object(verify_project, "check_lab_handoff", return_value={"status": "pass"}) as check:
            verify_project.check_terrain_lab_l13a_handoff()
        self.assertEqual("L13A_lighting.json", check.call_args.args[0])
        self.assertEqual("L13A", check.call_args.args[1])
        self.assertEqual(192, check.call_args.args[2])

    def test_l21_explicit_closure_direction_is_a_valid_approval_basis(self) -> None:
        result = verify_project.check_terrain_lab_l21_handoff()
        self.assertEqual(result["status"], "pass", result["detail"])

    def test_synthetic_civblp_probe_gate_passes(self) -> None:
        result = verify_project.check_civblp_probe_synthetic()
        self.assertEqual(result["status"], "pass", result["detail"])

    def test_expensive_native_commands_are_memoized(self) -> None:
        completed = mock.Mock(returncode=0, stdout="ok", stderr="")
        verify_project.run_native_build.cache_clear()
        verify_project.run_injected_compile.cache_clear()
        with mock.patch.object(verify_project.subprocess, "run", return_value=completed) as run:
            self.assertIs(verify_project.run_native_build(), verify_project.run_native_build())
            self.assertIs(verify_project.run_injected_compile(), verify_project.run_injected_compile())
        self.assertEqual(2, run.call_count)
        verify_project.run_native_build.cache_clear()
        verify_project.run_injected_compile.cache_clear()


if __name__ == "__main__":
    unittest.main()
