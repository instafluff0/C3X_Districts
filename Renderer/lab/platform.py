"""Small native dispatcher, independent of retired milestone/campaign state."""
from pathlib import Path, PureWindowsPath
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def python_executable():
    return os.environ.get("C3X_RENDERER_PYTHON", sys.executable)


def command_result(command, cwd=ROOT):
    result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    output = result.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    return {"status": "pass" if result.returncode == 0 else "fail",
            "returncode": result.returncode, "output_tail": output[-4000:]}


def changed_injected_sources():
    result = subprocess.run(["git", "status", "--porcelain", "--", "C3X.h", "injected_code.c"],
                            cwd=ROOT, text=True, capture_output=True, check=False)
    return result.returncode != 0 or bool(result.stdout.strip())


def windows_root():
    configured = os.environ.get("C3X_RENDERER_WINDOWS_ROOT")
    if configured:
        return PureWindowsPath(configured)
    try:
        return PureWindowsPath(r"\\Mac\Home") / ROOT.relative_to(Path.home()).as_posix()
    except ValueError:
        raise ValueError("Set C3X_RENDERER_WINDOWS_ROOT to the VM's shared checkout")


def native_command_result(relative_cwd, command, *, timeout_seconds=None):
    if os.name == "nt":
        # cmd parses its own command tail; list2cmdline would escape embedded
        # quotes with backslashes, breaking quoted batch paths and task filters.
        args = 'cmd /d /s /c "' + command + '"'
        cwd = ROOT / relative_cwd
    else:
        vm = os.environ.get("C3X_RENDERER_VM", "Windows 11")
        directory = windows_root() / PureWindowsPath(relative_cwd)
        args = ["prlctl", "exec", vm, "cmd", "/d", "/s", "/c",
                f'pushd "{directory}" && {command}']
        cwd = ROOT
    try:
        result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, check=False, timeout=timeout_seconds)
        output = result.stdout or ""
        returncode = result.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += "\nVM transport timed out after the bounded fixture wait\n"
        returncode = None
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    failed = returncode != 0 or any(line.startswith("FAIL ") for line in output.splitlines())
    return {"status": "fail" if failed else "pass", "returncode": returncode,
            "output_tail": output[-4000:]}


def native_completion(directory, run_id):
    """A per-invocation child exit record, not the VM transport's observation."""
    receipt = directory / "completion.txt"
    log = directory / "native.log"
    try:
        fields = receipt.read_text().strip().split()
        if len(fields) != 2 or fields[0] != run_id or not log.is_file():
            raise ValueError("Native completion is not confirmed for this invocation")
        code = int(fields[1])
        output = log.read_text(errors="replace")
    except (OSError, ValueError) as error:
        raise ValueError("Native completion is not confirmed; inspect the existing Windows process before retrying") from error
    failed = code != 0 or any(line.startswith("FAIL ") for line in output.splitlines())
    return {"status": "fail" if failed else "pass", "returncode": code, "output_tail": output[-4000:]}


class NativeFixturePending(ValueError):
    """An invocation may still be live; do not start the next fixture."""


def fixture_process(directory, run_id):
    try:
        fields = (directory / "process.txt").read_text().split()
        if len(fields) == 2 and fields[0] == run_id and int(fields[1]) > 0:
            return int(fields[1])
    except (OSError, ValueError):
        pass
    return None


def wait_native_fixture(directory, run_id, process):
    # Poll the exact process published by this invocation, not an image-name
    # guess. Each observation is short; never restart a confirmed live render.
    for attempt in range(120):
        try:
            return native_completion(directory, run_id)
        except ValueError:
            state = native_command_result("Renderer/native",
                f'tasklist /FO CSV /NH /FI "PID eq {process}"', timeout_seconds=30)
            try:
                return native_completion(directory, run_id)
            except ValueError:
                absent = "INFO: No tasks are running which match the specified criteria."
                if state["status"] == "pass" and state.get("output_tail", "").strip() == absent:
                    raise ValueError(f"Preview PID {process} exited without a matching completion receipt")
                if state["status"] != "pass" or f'"native_preview.exe","{process}"' not in state.get("output_tail", ""):
                    print(f"Preview PID {process} observation is uncertain; keeping this invocation pending", flush=True)
        if attempt % 6 == 0:
            print(f"Waiting for current native preview PID {process}; it is still running", flush=True)
        time.sleep(5)
    raise NativeFixturePending(f"Preview PID {process} is still pending; inspect that process before another fixture")


def run_native_fixture(directory, command, run_id):
    # A local timeout terminates cmd and can orphan/kill its render before it
    # writes completion. Allow cold D3D shader compilation on Windows hosts.
    timeout = 600 if os.name == "nt" else 120
    transport = native_command_result("Renderer/native", command, timeout_seconds=timeout)
    try:
        result = native_completion(directory, run_id)
    except ValueError:
        process = fixture_process(directory, run_id)
        if process is not None:
            return wait_native_fixture(directory, run_id, process)
        if transport["status"] == "pass":
            raise
        # A transport error is not a stopped render. Ask Windows explicitly and
        # check again for completion before deciding whether retry is safe.
        state = native_command_result("Renderer/native",
            'tasklist /FO CSV /NH /FI "IMAGENAME eq native_preview.exe"')
        try:
            result = native_completion(directory, run_id)
        except ValueError:
            absent = "INFO: No tasks are running which match the specified criteria."
            if state["status"] != "pass" or state.get("output_tail", "").strip() != absent:
                raise NativeFixturePending("Native completion is unconfirmed; inspect the existing invocation before another fixture")
            print("Windows confirms no preview process; retrying failed dispatch once", flush=True)
            transport = native_command_result("Renderer/native", command, timeout_seconds=timeout)
            result = native_completion(directory, run_id)
    if transport["status"] != "pass" and result["status"] == "pass":
        print("Native process completion verified from the current fixture receipt", flush=True)
    return result


def injected_compile_result():
    if os.name == "nt" and (ROOT.parent / "Civ3Conquests.exe").is_file():
        return native_command_result("", "call TEST_INJECTED_CODE_COMPILE.bat")
    conquests = PureWindowsPath(os.environ.get("C3X_RENDERER_CIV3_CONQUESTS",
        r"C:\Program Files (x86)\GOG Galaxy\Games\Civilization III Complete\Conquests"))
    link = conquests / "C3X_Shared_Verify"
    command = (f'mklink /D "{link}" "{windows_root()}" >nul 2>nul & '
               f'cd /d "{link}" && call TEST_INJECTED_CODE_COMPILE.bat')
    return native_command_result("", command)
