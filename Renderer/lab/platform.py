"""Small native dispatcher, independent of retired milestone/campaign state."""
from pathlib import Path, PureWindowsPath
import os
import subprocess
import sys

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
        args = ["cmd", "/d", "/s", "/c", command]
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


def run_native_fixture(directory, command, run_id):
    transport = native_command_result("Renderer/native", command, timeout_seconds=120)
    try:
        result = native_completion(directory, run_id)
    except ValueError:
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
                raise
            print("Windows confirms no preview process; retrying failed dispatch once", flush=True)
            transport = native_command_result("Renderer/native", command, timeout_seconds=120)
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
