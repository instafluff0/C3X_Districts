"""Execute the production readiness parser without any native/VM invocation."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]


def actual_parser():
    source = (ROOT / "Renderer/native/record_gpu_frame.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
    blocks = [node for node in function.body if isinstance(node, ast.If)
              and isinstance(node.test, ast.Attribute)
              and isinstance(node.test.value, ast.Name)
              and node.test.value.id == "args" and node.test.attr == "world_readiness"]
    assert len(blocks) == 1, "Expected the actual readiness receipt/admission block"
    dropped = next(node for node in function.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "dropped"
                           for target in node.targets))
    wrapper = ast.parse("""
def parse(log, trace, args):
    import re
    passed = True
    complete = ['fixture', '0']
    invocation = 'fixture'
    unchanged = True
    receipt = {}
    return passed, receipt
""")
    wrapper.body[0].body[-1:-1] = [copy.deepcopy(dropped), copy.deepcopy(blocks[0])]
    namespace = {}
    exec(compile(ast.fix_missing_locations(wrapper), "<actual readiness parser>", "exec"), namespace)
    return namespace["parse"]


def fixture():
    log = ["WORLD_READINESS total=800 authoritative=800 passes=1 regions=25 attempted=25 unavailable=0 first_request_ms=10 preparation_ms=20 largest_free=67108864"]
    checkpoints = []
    restore_total = 0
    for sample in range(16):
        begin = 1000 + sample * 100
        phase = "first" if sample < 6 else "restore" if sample < 9 else "repeat"
        restore = phase == "restore"
        # Pre/post checkpoints come from world_status, outside timed requests.
        common = {"pid": 42, "scope": 11, "device": 3, "canonical": 1,
                  "compiler_total": 200, "recovery_total": 7}
        checkpoints.append(dict(common, qpc=begin - 10, restore_total=restore_total))
        if restore:
            restore_total += 5
        checkpoints.append(dict(common, qpc=begin + 30, restore_total=restore_total))
        destination = sample if sample < 6 else (sample - 6) % 3
        log.append("WORLD_JUMP sample=%d x=%d y=0 result=1 request_ms=1 present_ms=.5 desktop_wait_ms=.5 desktop_ms=2 built=0 reused=32 uploads=%d readbacks=0 largest_free=67108864 available_va=134217728 geometry_bytes=1048576 begin_qpc=%d end_qpc=%d phase=%s" %
                   (sample, destination * 2, 4096 if restore else 0, begin, begin + 20, phase))
    log.append("WORLD_EVICTION sample=6 result=1 evicted=32 geometry_bytes=1048576 begin_qpc=1550 end_qpc=1560")
    log.extend("WORLD_SWEEP sweep=%d sample=%d geometry_bytes=1048576 available_va=134217728 largest_free=67108864 geometry_growth=0 va_growth=0 plateau=1" %
               (sweep, sample) for sweep, sample in [(1, 8), (2, 11), (3, 14)])
    log.extend("WORLD_ORACLE x=%d y=0 differing_channels=0 max_channel_delta=0 independent=1 clock=1000000" %
               index for index in range(6))
    log.append("PASS world readiness workload: samples=16 live_input=unmeasured desktop=measured oracles=6 fixture=bounded evictions=1 sweeps=3")
    return "\n".join(log), checkpoints


def trace(checkpoints, extra=""):
    return "\n".join("[C3X renderer] stage=world-compiler-checkpoint " +
                     " ".join("%s=%d" % pair for pair in event.items())
                     for event in checkpoints) + extra


def parse(log, checkpoint_trace):
    return actual_parser()(log, checkpoint_trace, SimpleNamespace(
        world_readiness=True, world_readiness_only=True,
        world_readiness_samples=16, x64_helper=True))


class WorldCompilerCheckpointParserTests(unittest.TestCase):
    def test_repeated_zero_is_proved_by_cumulative_endpoints_without_backing_rows(self):
        log, checkpoints = fixture()
        passed, receipt = parse(log, trace(checkpoints))
        self.assertTrue(passed)
        world = receipt["world_readiness"]
        self.assertTrue(world["zero_world_compiler_calls"])
        self.assertEqual(world["restored_contributors"], 15)
        self.assertTrue(world["zero_repeat_adoption_or_upload"])
        self.assertTrue(all(row["compiler_checkpoint_coverage"] for row in world["samples"]))
        self.assertTrue(all(row["world_compiler_calls"] == 0 for row in world["samples"]))
        self.assertEqual([row["world_restore_calls"] for row in world["samples"]],
                         [0] * 6 + [5] * 3 + [0] * 7)

    def test_absent_trace_or_absent_brackets_never_means_zero(self):
        log, _ = fixture()
        traces = ["", "[C3X renderer] qpc=1001 stage=unrelated result=1",
                  "[C3X renderer] qpc=1001 stage=world-backing compiler_calls=0 restore_calls=0"]
        for evidence in traces:
            with self.subTest(evidence=evidence):
                passed, receipt = parse(log, evidence)
                self.assertFalse(passed)
                world = receipt["world_readiness"]
                self.assertFalse(world["zero_world_compiler_calls"])
                self.assertTrue(all(row["world_compiler_calls"] is None and
                                    row["world_restore_calls"] is None and
                                    not row["compiler_checkpoint_coverage"] for row in world["samples"]))

    def test_every_invocation_requires_both_strict_outer_checkpoints(self):
        for sample in [0, 6, 15]:
            for side in [0, 1]:
                log, checkpoints = fixture()
                del checkpoints[2 * sample + side]
                # Adjacent requests can otherwise provide an outer checkpoint.
                # Remove that entire direction to prove missing coverage fails.
                begin = 1000 + sample * 100
                checkpoints = [event for event in checkpoints
                               if event["qpc"] >= begin] if side == 0 else [
                                   event for event in checkpoints if event["qpc"] <= begin + 20]
                with self.subTest(sample=sample, side=side):
                    passed, receipt = parse(log, trace(checkpoints))
                    self.assertFalse(passed)
                    row = receipt["world_readiness"]["samples"][sample]
                    self.assertFalse(row["compiler_checkpoint_coverage"])
                    self.assertIsNone(row["world_compiler_calls"])
        # Checkpoints at begin/end do not certify work before/after the call.
        log, checkpoints = fixture()
        checkpoints[0]["qpc"], checkpoints[1]["qpc"] = 1000, 1020
        checkpoints = checkpoints[:2]
        passed, receipt = parse(log, trace(checkpoints))
        self.assertFalse(passed)
        self.assertFalse(receipt["world_readiness"]["samples"][0]["compiler_checkpoint_coverage"])

    def test_missing_required_identity_or_total_field_rejects_coverage(self):
        for field in ["pid", "scope", "device", "canonical", "compiler_total", "recovery_total", "restore_total"]:
            for side in [0, 1]:
                log, checkpoints = fixture()
                del checkpoints[side][field]
                with self.subTest(field=field, side=side):
                    passed, receipt = parse(log, trace(checkpoints))
                    self.assertFalse(passed)
                    self.assertFalse(receipt["world_readiness"]["samples"][0]["compiler_checkpoint_coverage"])
                    self.assertIsNone(receipt["world_readiness"]["samples"][0]["world_compiler_calls"])

    def test_mismatched_scope_device_pid_and_noncanonical_export_reject(self):
        for field in ["pid", "scope", "device", "canonical"]:
            log, checkpoints = fixture()
            checkpoints[1][field] += 1
            with self.subTest(field=field):
                passed, receipt = parse(log, trace(checkpoints))
                self.assertFalse(passed)
                row = receipt["world_readiness"]["samples"][0]
                self.assertFalse(row["compiler_checkpoint_coverage"])
                self.assertIsNone(row["world_restore_calls"])

    def test_each_cumulative_counter_must_be_monotone(self):
        for field in ["compiler_total", "recovery_total", "restore_total"]:
            log, checkpoints = fixture()
            checkpoints[0][field] = 500
            checkpoints[1][field] = 499
            with self.subTest(field=field):
                passed, receipt = parse(log, trace(checkpoints))
                self.assertFalse(passed)
                self.assertFalse(receipt["world_readiness"]["samples"][0]["compiler_checkpoint_coverage"])
                self.assertIsNone(receipt["world_readiness"]["samples"][0]["world_compiler_calls"])

    def test_known_compilation_and_recovery_are_counted_not_relabelled_restore(self):
        log, checkpoints = fixture()
        checkpoints[1]["compiler_total"] += 3
        checkpoints[1]["recovery_total"] += 2
        checkpoints[1]["restore_total"] += 4
        passed, receipt = parse(log, trace(checkpoints))
        self.assertFalse(passed)
        row = receipt["world_readiness"]["samples"][0]
        self.assertTrue(row["compiler_checkpoint_coverage"])
        self.assertEqual(row["world_compiler_calls"], 5)
        self.assertEqual(row["world_restore_calls"], 4)
        self.assertFalse(receipt["world_readiness"]["zero_world_compiler_calls"])

    def test_dropped_trace_cannot_qualify_even_with_zero_brackets(self):
        log, checkpoints = fixture()
        passed, receipt = parse(log, trace(checkpoints, "\nTRACE_BUFFER dropped=1"))
        self.assertFalse(passed)
        self.assertTrue(receipt["world_readiness"]["zero_world_compiler_calls"])


if __name__ == "__main__":
    unittest.main()
