#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_project_state


class ProjectStateTests(unittest.TestCase):
    def test_repository_project_state_is_valid(self) -> None:
        self.assertEqual(check_project_state.validate_project_state(), [])

    def test_rejects_multiple_ready_steps(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        source["milestones"][1]["steps"][4]["status"] = "ready"
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "project_status.json"
            status_path.write_text(json.dumps(source), encoding="utf-8")
            with mock.patch.object(check_project_state, "RENDERER_ROOT", check_project_state.RENDERER_ROOT):
                errors = check_project_state.validate_project_state(status_path)
        self.assertTrue(any("Exactly the declared next_step" in error for error in errors))

    def test_rejects_completed_step_without_verification(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        del source["milestones"][1]["steps"][0]["verification"]
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "project_status.json"
            status_path.write_text(json.dumps(source), encoding="utf-8")
            errors = check_project_state.validate_project_state(status_path)
        self.assertTrue(any("no portable verification gates" in error for error in errors))

    def test_rejects_missing_patch_dependency_ledger(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        del source["civ3_patch_dependencies"]
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "project_status.json"
            status_path.write_text(json.dumps(source), encoding="utf-8")
            errors = check_project_state.validate_project_state(status_path)
        self.assertTrue(any("civ3_patch_dependencies" in error for error in errors))

    def test_rejects_incomplete_human_patch_request(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        source["civ3_patch_dependencies"]["required_user_action"] = [{"symbol": "Missing_Function"}]
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "project_status.json"
            status_path.write_text(json.dumps(source), encoding="utf-8")
            errors = check_project_state.validate_project_state(status_path)
        self.assertTrue(any("patch request 0 is missing fields" in error for error in errors))
        self.assertTrue(any("supported-build addresses" in error for error in errors))

    def test_natural_constructed_wonders_and_districts_are_separate_milestones(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        milestones = {milestone["id"]: milestone for milestone in source["milestones"]}

        self.assertIn("docs/natural_wonder_rendering.md", source["required_docs"])
        self.assertIn("docs/wonder_and_district_rendering.md", source["required_docs"])
        self.assertEqual("M11", source["milestones"][-1]["id"])
        self.assertEqual(["M9.1", "M9.2"], [step["id"] for step in milestones["M9"]["steps"]])
        self.assertEqual(
            ["M10.1", "M10.2"],
            [step["id"] for step in milestones["M10"]["steps"]],
        )
        self.assertEqual(
            ["M11.1", "M11.2", "M11.3"],
            [step["id"] for step in milestones["M11"]["steps"]],
        )
        natural_scope = " ".join(
            text
            for step in milestones["M9"]["steps"]
            for field in ("scope", "acceptance", "must_not")
            for text in step[field]
        ).lower()
        self.assertIn("landmark", natural_scope)
        self.assertIn("native natural-wonder name label", natural_scope)
        self.assertIn("exclusive", natural_scope)
        self.assertIn("without native body", natural_scope)
        district_scope = " ".join(
            text
            for step in milestones["M11"]["steps"]
            for field in ("scope", "acceptance", "must_not")
            for text in step[field]
        )
        self.assertIn("by-building", district_scope)
        self.assertIn("by-count", district_scope)
        self.assertIn("exclusive", district_scope)
        self.assertIn("without restoring", district_scope)

    def test_shared_environment_precedes_category_lighting(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        milestones = {milestone["id"]: milestone for milestone in source["milestones"]}
        m6_steps = {step["id"]: step for step in milestones["M6"]["steps"]}

        self.assertIn("docs/environment_lighting_and_ambient_effects.md", source["required_docs"])
        self.assertIn("docs/civ6_lighting_findings.md", source["required_docs"])
        self.assertEqual("complete", m6_steps["M6.4"]["status"])
        self.assertEqual("complete", m6_steps["M6.5"]["status"])
        self.assertEqual("complete", m6_steps["M6.6"]["status"])
        self.assertEqual("complete", m6_steps["M6.7"]["status"])
        self.assertEqual("blocked_by_previous", m6_steps["M6.8"]["status"])
        lab_steps = {step["id"]: step for step in milestones["LAB"]["steps"]}
        self.assertEqual("complete", lab_steps["L11"]["status"])
        self.assertEqual("complete", lab_steps["L12"]["status"])
        integration_steps = {step["id"]: step for step in milestones["INTEGRATION"]["steps"]}
        self.assertEqual("complete", integration_steps["I12"]["status"])
        self.assertEqual("complete", lab_steps["L13"]["status"])
        self.assertEqual("complete", lab_steps["L13A"]["status"])
        self.assertEqual("complete", lab_steps["L14"]["status"])
        self.assertEqual("complete", lab_steps["L15"]["status"])
        self.assertEqual("complete", lab_steps["L16"]["status"])
        self.assertEqual("complete", lab_steps["L17"]["status"])
        self.assertEqual("complete", lab_steps["L18"]["status"])
        self.assertEqual("ready", lab_steps["L19"]["status"])
        self.assertEqual("L19", source["next_step"]["id"])
        environment_scope = " ".join(
            m6_steps["M6.4"][field][index]
            for field in ("scope", "acceptance", "must_not")
            for index in range(len(m6_steps["M6.4"][field]))
        ).lower()
        self.assertIn("moon", environment_scope)
        self.assertIn("emissive", environment_scope)
        self.assertIn("ambient", environment_scope)
        self.assertIn("static scenes request no continuous redraw", environment_scope)

        production_scope = " ".join(
            text
            for step_id in ("M6.5", "M6.6", "M6.7")
            for field in ("scope", "acceptance", "must_not")
            for text in m6_steps[step_id][field]
        ).lower()
        self.assertIn("checkerboard", production_scope)
        self.assertIn("all fourteen", production_scope)
        self.assertIn("cache-relevant", production_scope)
        self.assertIn("zero fallback tiles", production_scope)
        self.assertIn("hard failure without native replay", production_scope)

    def test_map_resource_mapping_preserves_native_non_map_icons(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        milestones = {milestone["id"]: milestone for milestone in source["milestones"]}
        m7_steps = {step["id"]: step for step in milestones["M7"]["steps"]}

        self.assertIn("docs/civ3_to_civ6_resource_mapping.md", source["required_docs"])
        self.assertIn(
            "inventory/vanilla_conquests_to_civ6_resources.json",
            source["required_docs"],
        )
        resource_scope = " ".join(
            text
            for field in ("scope", "acceptance", "must_not")
            for text in m7_steps["M7.2"][field]
        ).lower()
        self.assertIn("map resource bodies", resource_scope)
        self.assertIn("civilopedia", resource_scope)
        self.assertIn("resources.pcx globally", resource_scope)
        self.assertIn("player-specific visibility", resource_scope)

    def test_lab_systems_promote_independently_to_game_integration(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        milestones = {milestone["id"]: milestone for milestone in source["milestones"]}
        lab_steps = milestones["LAB"]["steps"]
        m7 = milestones["M7"]
        m7_steps = {step["id"]: step for step in m7["steps"]}
        prerequisite = m7["promotion_prerequisites"]
        promotion = source["lab_promotion_policy"]

        self.assertIn("terrain_lab/PLAN.md", source["required_docs"])
        self.assertIn("docs/renderer_workstreams.md", source["required_docs"])
        by_system = prerequisite["by_system"]
        self.assertEqual(["L13"], by_system["rivers"])
        self.assertEqual(["L16"], by_system["resources"])
        self.assertEqual(["L17"], by_system["cities"])
        self.assertEqual(["L20"], by_system["units"])
        self.assertEqual(["L21"], by_system["combined_release_scene"])
        self.assertIn("not a global integration prerequisite", prerequisite["rule"])
        self.assertTrue(all(step["promotion_required"] for step in lab_steps))
        self.assertEqual(192, promotion["minimum_contiguous_tiles"])
        self.assertEqual({"L9": 48, "L10": 96, "L11": 96}, promotion["accepted_legacy_promotions"])
        self.assertTrue(promotion["required_user_visual_approval"])
        self.assertIn("next lab gate", promotion["advance_rule"])
        self.assertEqual("L16", m7_steps["M7.2"]["lab_handoffs"][0])
        self.assertEqual(["L17"], m7_steps["M7.3"]["lab_handoffs"])
        self.assertEqual(["L20"], m7_steps["M7.4"]["lab_handoffs"])
        for step_id in ("M7.1", "M7.2", "M7.3", "M7.4", "M7.5"):
            self.assertEqual("blocked_by_previous", m7_steps[step_id]["status"])

    def test_workstreams_and_vm_workflow_are_machine_readable(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        workstreams = source["workstreams"]

        self.assertEqual("L19", workstreams["renderer_lab"]["current_step"])
        self.assertEqual("I13A", workstreams["game_integration"]["current_step"])
        self.assertIn("renderer_dev.py lab", workstreams["renderer_lab"]["iteration_command"])
        self.assertIn("renderer_dev.py integration", workstreams["game_integration"]["iteration_command"])
        self.assertEqual("Windows 11", workstreams["windows_vm"]["name"])
        self.assertTrue(workstreams["windows_vm"]["shared_repository"].startswith("Y:\\"))
        self.assertIn("same-numbered I#", workstreams["promotion_rule"])

    def test_every_lab_gate_has_a_same_numbered_integration_gate(self) -> None:
        source = json.loads(check_project_state.DEFAULT_STATUS.read_text(encoding="utf-8"))
        milestones = {milestone["id"]: milestone for milestone in source["milestones"]}
        lab_steps = {step["id"]: step for step in milestones["LAB"]["steps"]}
        integration_steps = {
            step["lab_gate"]: step for step in milestones["INTEGRATION"]["steps"]
        }
        self.assertEqual(set(lab_steps), set(integration_steps))
        for lab_id, lab_step in lab_steps.items():
            integration = integration_steps[lab_id]
            self.assertEqual("I" + lab_id[1:], integration["id"])
            if integration["status"] == "complete":
                self.assertEqual("complete", lab_step["status"])


if __name__ == "__main__":
    unittest.main()
