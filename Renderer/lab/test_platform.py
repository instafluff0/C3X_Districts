"""Protect VM dispatch and compile safety without historical milestone gates."""
import os
import tempfile
from pathlib import Path, PureWindowsPath
from unittest import TestCase, mock
from Renderer.lab import platform


class PlatformTests(TestCase):
    def test_timeout_waits_for_own_live_process_without_restarting(self):
        live = {"status": "pass", "output_tail": '\"native_preview.exe\",\"1234\",\"Services\"'}
        with mock.patch.object(platform, "native_command_result", side_effect=[{"status": "fail"}, live]) as run, \
             mock.patch.object(platform, "fixture_process", return_value=1234), \
             mock.patch.object(platform, "native_completion", side_effect=[ValueError(), ValueError(), ValueError(), {"status": "pass"}]), \
             mock.patch.object(platform.time, "sleep") as sleep, mock.patch("builtins.print"):
            self.assertEqual(platform.run_native_fixture(Path("fixture"), "draw", "id")["status"], "pass")
            self.assertEqual(run.call_count, 2)
            self.assertIn("PID eq 1234", run.call_args.args[1])
            sleep.assert_called_once_with(5)

    def test_process_receipt_is_bound_to_the_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in ("old 1234", "id 0", "id bad", "id 1234 extra"):
                (root / "process.txt").write_text(value)
                self.assertIsNone(platform.fixture_process(root, "id"))
            (root / "process.txt").write_text("id 1234")
            self.assertEqual(platform.fixture_process(root, "id"), 1234)

    def test_transient_observation_failure_keeps_the_same_invocation(self):
        with mock.patch.object(platform, "native_command_result", return_value={"status": "fail", "output_tail": "temporary VM error"}) as run, \
             mock.patch.object(platform, "native_completion", side_effect=[ValueError(), ValueError(), {"status": "pass"}]), \
             mock.patch.object(platform.time, "sleep") as sleep, mock.patch("builtins.print"):
            self.assertEqual(platform.wait_native_fixture(Path("fixture"), "id", 1234)["status"], "pass")
            run.assert_called_once()
            sleep.assert_called_once_with(5)

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

    def test_transport_timeout_is_a_bounded_failure(self):
        error = platform.subprocess.TimeoutExpired(["prlctl"], 120, output="partial")
        with mock.patch.object(platform.subprocess, "run", side_effect=error), mock.patch("builtins.print"):
            result = platform.native_command_result("Renderer/native", "test", timeout_seconds=120)
        self.assertEqual(result["status"], "fail")
        self.assertIsNone(result["returncode"])
        self.assertIn("bounded fixture wait", result["output_tail"])

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
