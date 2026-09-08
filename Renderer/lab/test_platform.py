"""Protect VM dispatch and compile safety without historical milestone gates."""
import os
import tempfile
from pathlib import Path, PureWindowsPath
from unittest import TestCase, mock
from Renderer.lab import platform


class PlatformTests(TestCase):
    def test_transport_error_with_confirmed_completion_does_not_retry(self):
        with mock.patch.object(platform, "native_command_result", return_value={"status": "fail"}) as run, \
             mock.patch.object(platform, "native_completion", return_value={"status": "pass"}), mock.patch("builtins.print"):
            self.assertEqual(platform.run_native_fixture(Path("fixture"), "draw", "id")["status"], "pass")
            run.assert_called_once()

    def test_retry_requires_verified_absence_and_is_bounded(self):
        absent = {"status": "pass", "output_tail": "INFO: No tasks are running which match the specified criteria."}
        with mock.patch.object(platform, "native_command_result", side_effect=[{"status": "fail"}, absent, {"status": "pass"}]) as run, \
             mock.patch.object(platform, "native_completion", side_effect=[ValueError(), ValueError(), {"status": "pass"}]), \
             mock.patch("builtins.print"):
            self.assertEqual(platform.run_native_fixture(Path("fixture"), "draw", "id")["status"], "pass")
            self.assertEqual(run.call_count, 3)
            self.assertIn("tasklist", run.call_args_list[1].args[1])
            self.assertEqual(run.call_args_list[0], run.call_args_list[2])
        with mock.patch.object(platform, "native_command_result", side_effect=[{"status": "fail"}, absent, {"status": "fail"}]) as run, \
             mock.patch.object(platform, "native_completion", side_effect=ValueError()), mock.patch("builtins.print"):
            with self.assertRaises(ValueError):
                platform.run_native_fixture(Path("fixture"), "draw", "id")
            self.assertEqual(run.call_count, 3)

    def test_live_or_uncertain_process_is_not_restarted(self):
        for state in ({"status": "pass", "output_tail": '"native_preview.exe","1234"'},
                      {"status": "fail", "output_tail": "query failed"},
                      {"status": "pass", "output_tail": "unrecognized response"}):
            with mock.patch.object(platform, "native_command_result", side_effect=[{"status": "fail"}, state]) as run, \
                 mock.patch.object(platform, "native_completion", side_effect=ValueError()):
                with self.assertRaises(ValueError):
                    platform.run_native_fixture(Path("fixture"), "draw", "id")
                self.assertEqual(run.call_count, 2)

    def test_late_completion_is_rechecked_before_retry(self):
        with mock.patch.object(platform, "native_command_result", return_value={"status": "fail"}) as run, \
             mock.patch.object(platform, "native_completion", side_effect=[ValueError(), {"status": "pass"}]), \
             mock.patch("builtins.print"):
            self.assertEqual(platform.run_native_fixture(Path("fixture"), "draw", "id")["status"], "pass")
            self.assertEqual(run.call_count, 2)

    def test_native_completion_requires_current_id_log_and_exit_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "not confirmed"):
                platform.native_completion(root, "current")
            (root / "native.log").write_text("old image was rendered\n")
            for text in ("older 0", "current", "current bad", "current 0 extra"):
                (root / "completion.txt").write_text(text)
                with self.assertRaisesRegex(ValueError, "not confirmed"):
                    platform.native_completion(root, "current")
            (root / "completion.txt").write_text("current 0\n")
            self.assertEqual(platform.native_completion(root, "current")["status"], "pass")
            (root / "completion.txt").write_text("current 1\n")
            self.assertEqual(platform.native_completion(root, "current")["status"], "fail")
            (root / "completion.txt").write_text("current 0\n")
            (root / "native.log").write_text("FAIL actual render\n")
            self.assertEqual(platform.native_completion(root, "current")["status"], "fail")
            (root / "native.log").unlink()
            with self.assertRaisesRegex(ValueError, "not confirmed"):
                platform.native_completion(root, "current")

    def test_shared_checkout_can_be_overridden(self):
        with mock.patch.dict(os.environ, {"C3X_RENDERER_WINDOWS_ROOT": r"X:\repo"}):
            self.assertEqual(platform.windows_root(), PureWindowsPath(r"X:\repo"))

    def test_default_share_is_derived_without_a_machine_specific_directory(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(Path, "home", return_value=Path("/home/example")), \
             mock.patch.object(platform, "ROOT", Path("/home/example/projects/mod")):
            self.assertEqual(platform.windows_root(), PureWindowsPath(r"\\Mac\Home\projects\mod"))

    def test_checkout_outside_home_requires_explicit_share(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(Path, "home", return_value=Path("/home/example")), \
             mock.patch.object(platform, "ROOT", Path("/projects/mod")):
            with self.assertRaisesRegex(ValueError, "WINDOWS_ROOT"):
                platform.windows_root()

    def test_reported_test_failure_overrides_zero_transport_exit(self):
        result = mock.Mock(returncode=0, stdout="FAIL renderer: bad pixels\n")
        with mock.patch.object(platform.subprocess, "run", return_value=result), mock.patch("builtins.print"):
            self.assertEqual(platform.native_command_result("Renderer/native", "test")["status"], "fail")

    def test_injected_detection_is_fail_safe_and_limited_to_injected_sources(self):
        result = mock.Mock(returncode=1, stdout="")
        with mock.patch.object(platform.subprocess, "run", return_value=result) as run:
            self.assertTrue(platform.changed_injected_sources())
            self.assertEqual(run.call_args.args[0][-2:], ["C3X.h", "injected_code.c"])
        with mock.patch.object(platform.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="")):
            self.assertFalse(platform.changed_injected_sources())

    def test_approved_injected_script_is_used_not_installer(self):
        with mock.patch.object(platform, "native_command_result", return_value={"status": "pass"}) as run:
            platform.injected_compile_result()
            command = run.call_args.args[1]
            self.assertIn("TEST_INJECTED_CODE_COMPILE.bat", command)
            self.assertNotIn("INSTALL.bat", command)

    def test_native_candidate_build_cannot_stage(self):
        build = (platform.ROOT / "Renderer/native/BUILD.bat").read_text()
        stop = build.index('if /i "%~1"=="candidate-compile" exit /b 0')
        stage = build.index('copy /y "build\\candidate\\C3XRenderer.dll"')
        self.assertLess(stop, stage)
        self.assertIn("if errorlevel 1", build[stage:])
