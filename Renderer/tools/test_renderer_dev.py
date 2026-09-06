#!/usr/bin/env python3
import os
import sys
import unittest
from pathlib import Path, PureWindowsPath
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import renderer_dev


class RendererDevTests(unittest.TestCase):
    def test_default_vm_contract_names_the_documented_host_and_share(self) -> None:
        self.assertEqual("Windows 11", renderer_dev.DEFAULT_VM)
        self.assertEqual(
            PureWindowsPath(r"Y:\fun\Civilization III Complete\Conquests\C3X_Districts"),
            renderer_dev.DEFAULT_WINDOWS_ROOT,
        )
        self.assertEqual("C3X_Shared_Verify", renderer_dev.INJECTED_VERIFY_LINK)
        self.assertEqual("C3X_Districts", renderer_dev.LIVE_MOD_DIR)

    def test_windows_arg_quoting_preserves_spaces(self) -> None:
        self.assertEqual('"two words"', renderer_dev.quote_windows_arg("two words"))
        self.assertEqual("integration", renderer_dev.quote_windows_arg("integration"))

    def test_injected_change_detection_is_fail_safe(self) -> None:
        completed = mock.Mock(returncode=1, stdout="")
        with mock.patch.object(renderer_dev.subprocess, "run", return_value=completed):
            self.assertTrue(renderer_dev.changed_injected_sources())

    def test_windows_root_can_be_overridden(self) -> None:
        with mock.patch.dict(os.environ, {"C3X_RENDERER_WINDOWS_ROOT": r"X:\repo"}):
            self.assertEqual(PureWindowsPath(r"X:\repo"), renderer_dev.windows_root())

    def test_windows_live_target_can_be_overridden(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"C3X_RENDERER_WINDOWS_LIVE_TARGET": r"\\host\share\repo"},
        ):
            self.assertEqual(
                PureWindowsPath(r"\\host\share\repo"),
                renderer_dev.windows_live_target(),
            )

    def test_python_runtime_can_be_overridden(self) -> None:
        with mock.patch.dict(os.environ, {"C3X_RENDERER_PYTHON": sys.executable}):
            self.assertEqual(sys.executable, renderer_dev.python_executable())

    def test_live_checkout_requires_the_actual_root_to_link_to_the_shared_repo(self) -> None:
        result = {"status": "pass", "returncode": 0}
        with mock.patch.object(renderer_dev, "windows_command_result", return_value=result) as run:
            actual = renderer_dev.live_checkout_result()
        command = run.call_args.args[1]
        self.assertIn("rev-parse --show-toplevel", command)
        self.assertIn("find /i", command)
        self.assertIn(renderer_dev.DEFAULT_WINDOWS_LIVE_TARGET.as_posix(), command)
        self.assertIn("Conquests\\C3X_Districts", command)
        self.assertNotIn("copy", command.lower())
        self.assertEqual("live_checkout_link", actual["name"])

    def test_native_build_propagates_approved_payload_smoke_failure(self) -> None:
        build = (renderer_dev.RENDERER_ROOT / "native" / "BUILD.bat").read_text(
            encoding="utf-8"
        )
        approved = build[build.index('build\\native_smoke.exe "build\\candidate\\C3XRenderer.dll" --definitions') :]
        approved = approved[:approved.index(":approved_terrain_done")]
        self.assertIn("if errorlevel 1", approved)
        self.assertNotIn("%errorlevel%", approved)
        workflow = Path(renderer_dev.__file__).read_text(encoding="utf-8")
        self.assertIn('call BUILD.bat portable', workflow)
        self.assertIn('approved_terrain_smoke_result()', workflow)

    def test_native_build_rejects_a_locked_stale_live_dll(self) -> None:
        build = (renderer_dev.RENDERER_ROOT / "native" / "BUILD.bat").read_text(
            encoding="utf-8"
        )
        staging = build[build.index('copy /y "build\\candidate\\C3XRenderer.dll"') :]
        self.assertIn("if errorlevel 1", staging)
        self.assertIn("exit /b 1", staging)
        self.assertIn("stale build", staging)
        self.assertNotIn('if not exist "..\\bin\\C3XRenderer.dll"', staging)

    def test_approved_smoke_retries_one_empty_vm_failure(self) -> None:
        failed = {"status": "fail", "returncode": 255, "output_tail": ""}
        passed = {
            "status": "pass",
            "returncode": 0,
            "output_tail": "PASS approved_terrain_integration",
        }
        with mock.patch.object(Path, "is_file", return_value=True), \
             mock.patch.object(
                 renderer_dev, "native_command_result", side_effect=[failed, passed]
             ) as run:
            result = renderer_dev.approved_terrain_smoke_result()
        self.assertEqual("pass", result["status"])
        self.assertEqual(255, result["retry_after_returncode"])
        self.assertEqual(2, run.call_count)

    def test_lab_workflow_runs_the_authorized_gate_script(self) -> None:
        source = Path(renderer_dev.__file__).read_text(encoding="utf-8")
        self.assertIn('f"RUN_{results[0][\'next_step\']}.bat"', source)
        self.assertIn('results[0]["next_step"].startswith("LQ")', source)
        self.assertIn('str(RENDERER_ROOT / "tools" / "lab_v2.py")', source)
        self.assertIn('"Renderer.terrain_lab.test_canonical_reference_contract"', source)
        self.assertNotIn('call RUN_L12.bat', source)


if __name__ == "__main__":
    unittest.main()
