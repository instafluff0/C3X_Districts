from __future__ import annotations

import json
import unittest

from Renderer.tools.asset_compiler.worker_builder_action_compiler import (
    DEFAULT_STRATEGY,
    SOURCE_FREE_FORBIDDEN,
    compile_runtime,
    validate_strategy,
)


class WorkerBuilderActionCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = json.loads(DEFAULT_STRATEGY.read_text(encoding="utf-8"))

    def test_all_native_worker_jobs_are_covered_by_job_id(self) -> None:
        validate_strategy(self.strategy)
        runtime = compile_runtime(self.strategy)
        self.assertEqual([str(value) for value in range(13)], list(runtime["jobs"]))
        self.assertEqual("worker_job_id", runtime["authority"]["primary"])
        self.assertEqual("diagnostic_only", runtime["authority"]["animation_slot"])
        self.assertFalse(runtime["applicability"]["unit_name_detection"])
        self.assertEqual(
            "unit_definition_then_optional_default_worker_era_profile",
            runtime["applicability"]["body_resolution"],
        )

    def test_known_many_to_one_native_slots_do_not_collapse_semantics(self) -> None:
        jobs = {job["id"]: job for job in self.strategy["worker_jobs"]}
        self.assertEqual("IRRIGATE", jobs[1]["civ3_slot"])
        self.assertEqual("IRRIGATE", jobs[8]["civ3_slot"])
        self.assertNotEqual(jobs[1]["name"], jobs[8]["name"])
        self.assertEqual("DEFAULT", jobs[9]["civ3_slot"])
        self.assertEqual("DEFAULT", jobs[10]["civ3_slot"])
        self.assertEqual("DEFAULT", jobs[11]["civ3_slot"])

    def test_action_selects_exactly_one_generic_tool(self) -> None:
        runtime = compile_runtime(self.strategy)
        for job in runtime["jobs"].values():
            self.assertIsInstance(job["tool"], str)
            self.assertTrue(job["tool"].startswith("unit/worker/tool/"))
        self.assertEqual("exclusive_attachment_group", runtime["attachments"]["mode"])

    def test_runtime_output_contains_no_source_specific_names(self) -> None:
        encoded = json.dumps(compile_runtime(self.strategy), sort_keys=True)
        self.assertFalse([needle for needle in SOURCE_FREE_FORBIDDEN if needle in encoded])

    def test_capture_variants_are_deterministic_not_random_per_redraw(self) -> None:
        runtime = compile_runtime(self.strategy)
        self.assertEqual("stable_body_variation_modulo_4", runtime["capture"]["selection"])
        self.assertEqual(4, len(runtime["capture"]["actions"]))
        self.assertEqual("stable_body_variation_modulo_4", runtime["optional_repair"]["selection"])
        self.assertEqual(4, len(runtime["optional_repair"]["actions"]))
        self.assertIsNone(runtime["optional_repair"]["ordinary_worker_job"])

    def test_primary_work_loops_but_optional_repair_and_capture_clamp(self) -> None:
        clips = self.strategy["clips"]
        self.assertTrue(all(clips[name]["playback"] == "loop" for name in ("work_ground", "work_heavy", "work_cut")))
        self.assertTrue(all(clips[f"work_repair_{index}"]["playback"] == "clamp" for index in range(1, 5)))
        self.assertTrue(all(clips[f"captured_{index}"]["playback"] == "clamp" for index in range(1, 5)))

    def test_basic_contract_delegates_worker_specialties(self) -> None:
        basic_path = DEFAULT_STRATEGY.with_name("unit_action_conversion.json")
        basic = json.loads(basic_path.read_text(encoding="utf-8"))
        self.assertEqual(
            "worker_builder_action_strategy.json",
            basic["supplemental_action_contracts"]["worker_jobs_capture_build_and_fortress"],
        )
        self.assertNotIn("worker_jobs", basic["deferred_slots"])


if __name__ == "__main__":
    unittest.main()
