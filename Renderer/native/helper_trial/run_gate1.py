"""Build and run the isolated x64-producer/x86-presenter graphics boundary trial.

This does not stage C3X, run INSTALL.bat, or launch Civ III. Results live under
Renderer/native/build/helper_trial so a VM transport failure cannot be mistaken
for a successful native process. A missing child completion receipt is uncertain,
and must be inspected before retrying the trial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.lab.platform import ROOT, native_command_result, windows_root


TRIAL = ROOT / "Renderer/native/helper_trial"
BUILD = ROOT / "Renderer/native/build/helper_trial"
X86_MACHINE = 0x014C
X64_MACHINE = 0x8664
MANDATORY_CHECKS = (
    "import", "exact_frames", "partial_native", "resize", "helper_restart",
    "blocked_ui_presentation", "native_gdi_restore", "backpressure",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pe_machine(path: Path) -> int:
    with path.open("rb") as binary:
        if binary.read(2) != b"MZ":
            raise ValueError(f"{path.name} is not a PE executable")
        binary.seek(0x3C)
        offset = struct.unpack("<I", binary.read(4))[0]
        binary.seek(offset)
        if binary.read(4) != b"PE\0\0":
            raise ValueError(f"{path.name} has no PE header")
        return struct.unpack("<H", binary.read(2))[0]


def completed_exit(path: Path, invocation: str) -> int | None:
    try:
        fields = path.read_text(errors="replace").split()
        if len(fields) != 2 or fields[0] != invocation:
            return None
        return int(fields[1])
    except (OSError, ValueError):
        return None


def validate_native_report(report: object, *, require_desktop: bool = False) -> list[str]:
    """Require the actual production presenter and native self-test verdict.

    The standalone self-test owns detailed pixel, lifecycle and timing checks.
    This wrapper deliberately does not infer their success from process exit 0.
    """
    errors = []
    if not isinstance(report, dict):
        return ["native report is not a JSON object"]
    if report.get("status") != "pass":
        errors.append("native self-test did not report pass")
    if report.get("presentation_backend") != "NativePresenter-DComp":
        errors.append("x86 presenter did not certify production NativePresenter/DirectComposition")
    if report.get("cross_process_shared_import") is not True:
        errors.append("cross-process shared D3D11 import was not confirmed")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        errors.append("native scenario checks are missing")
    else:
        errors.extend(f"{name} did not pass" for name in MANDATORY_CHECKS
                      if checks.get(name) != "pass")
    if report.get("desktop_witness") not in ("pass", "unavailable"):
        errors.append("desktop pixel witness failed or was not reported")
    elif require_desktop and report["desktop_witness"] != "pass":
        errors.append("visible desktop pixel witness is unavailable; presentation is unconfirmed")
    if not isinstance(report.get("timing"), dict) or not report["timing"]:
        errors.append("native phase timing is missing")
    if not isinstance(report.get("memory"), dict) or not report["memory"]:
        errors.append("native process memory samples are missing")
    luid = report.get("adapter_luid")
    if not isinstance(luid, dict) or luid.get("match") is not True:
        errors.append("producer and consumer D3D adapters were not confirmed equal")
    return errors


def batch_text(invocation: str, output: Path, *, width: int, height: int,
               frames: int, visible: bool) -> str:
    native = windows_root() / "Renderer/native"
    target = windows_root() / output.relative_to(ROOT)
    options = f"--frames {frames} --width {width} --height {height}"
    if visible:
        options += " --visible"
    return ("@echo off\nsetlocal DisableDelayedExpansion\n"
            f'pushd "{native}"\n'
            "if errorlevel 1 exit /b 90\n"
            f'call helper_trial\\build.bat >"{target / "build.log"}" 2>&1\n'
            "if errorlevel 1 goto build_failed\n"
            f'build\\helper_trial\\bin\\consumer.exe --self-test --report "{target / "native-report.json"}" {options} '
            f'>"{target / "native.log"}" 2>&1\n'
            "if errorlevel 1 goto child_done\n"
            f'build\\helper_trial\\bin\\ipc_x86.exe "{target / "ipc-report.json"}" 1000 '
            f'>>"{target / "native.log"}" 2>&1\n'
            ":child_done\n"
            "set \"TRIAL_CODE=%errorlevel%\"\n"
            f'>"{target / "completion.txt"}" echo {invocation} %TRIAL_CODE%\n'
            "popd\nexit /b %TRIAL_CODE%\n"
            ":build_failed\n"
            f'>"{target / "completion.txt"}" echo {invocation} 91\n'
            "popd\nexit /b 91\n")


def run(width: int, height: int, frames: int, visible: bool, timeout: int) -> int:
    source_before = {file.name: sha256(file) for file in sorted(TRIAL.glob("*")) if file.is_file()}
    invocation = uuid.uuid4().hex
    output = BUILD / invocation
    output.mkdir(parents=True, exist_ok=False)
    batch = output / "run.cmd"
    batch.write_text(batch_text(invocation, output, width=width, height=height,
                                frames=frames, visible=visible), encoding="utf-8")
    print(f"GATE1_INVOCATION {output.relative_to(ROOT)}", flush=True)
    target = windows_root() / output.relative_to(ROOT) / batch.name
    try:
        transport = native_command_result("Renderer/native", f'call "{target}"',
                                          timeout_seconds=timeout)
    except UnicodeError:
        # Parallels can fail to decode cmd output after the Windows child has
        # actually completed. The child receipt, not transport, decides this.
        transport = {"returncode": None, "status": "unknown"}
    child_code = completed_exit(output / "completion.txt", invocation)
    if child_code is None:
        (output / "receipt.json").write_text(json.dumps({
            "schema": 1, "status": "unconfirmed", "invocation": invocation,
            "reason": "native child completion receipt missing or mismatched",
            "transport_exit": transport.get("returncode"),
            "action": "inspect the existing VM invocation before any retry",
        }, indent=2) + "\n", encoding="utf-8")
        print("GATE1_UNCONFIRMED Native completion is unknown. Inspect this invocation and its "
              "Windows process before any retry.", file=sys.stderr)
        return 2
    native_report_file = output / "native-report.json"
    try:
        native_report = json.loads(native_report_file.read_text()) if native_report_file.exists() else None
    except (OSError, json.JSONDecodeError):
        native_report = None
    ipc_report_file = output / "ipc-report.json"
    try:
        ipc_report = json.loads(ipc_report_file.read_text()) if ipc_report_file.exists() else None
    except (OSError, json.JSONDecodeError):
        ipc_report = None
    errors = [] if child_code == 0 else [f"native child exit {child_code}"]
    errors.extend(validate_native_report(native_report, require_desktop=visible))
    if not isinstance(ipc_report, dict) or ipc_report.get("status") != "pass" or ipc_report.get("samples") != 1000:
        errors.append("cross-process shared-memory/event control did not complete")
    binaries = {}
    for name, expected in (("producer.exe", X64_MACHINE), ("consumer.exe", X86_MACHINE),
                           ("ipc_x64.exe", X64_MACHINE), ("ipc_x86.exe", X86_MACHINE)):
        binary = BUILD / "bin" / name
        if not binary.is_file():
            errors.append(f"missing {name}")
            continue
        try:
            machine = pe_machine(binary)
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        binaries[name] = {"pe_machine": f"0x{machine:04x}", "sha256": sha256(binary)}
        if machine != expected:
            errors.append(f"{name} PE Machine is 0x{machine:04x}, expected 0x{expected:04x}")
    sources = {file.name: sha256(file) for file in sorted(TRIAL.glob("*")) if file.is_file()}
    if source_before != sources:
        errors.append("trial sources changed during the native run")
    unconfirmed_desktop = (visible and isinstance(native_report, dict)
                           and native_report.get("desktop_witness") == "unavailable"
                           and errors == ["visible desktop pixel witness is unavailable; presentation is unconfirmed"])
    outcome = "unconfirmed" if unconfirmed_desktop else "pass" if not errors else "fail"
    receipt = {
        "schema": 1,
        "status": outcome,
        "invocation": invocation,
        "scope": "isolated x64 producer to x86 production DirectComposition presenter; no Civ III",
        "environment_note": "Windows 11 ARM64 VM; x64 and x86 binaries execute under emulation. Timings are VM-specific, not native-x64 estimates.",
        "width": width, "height": height, "frames": frames, "visible": visible,
        "child_exit": child_code,
        "transport_exit": transport.get("returncode"),
        "source_sha256": sources,
        "sources_unchanged": source_before == sources,
        "binaries": binaries,
        "native_report": native_report,
        "ipc_control": ipc_report,
        "errors": errors,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    for name in ("build.log", "native.log"):
        log = output / name
        if log.exists():
            print(f"{name}:\n{log.read_text(errors='replace')[-5000:]}", flush=True)
    print(json.dumps({"status": receipt["status"], "invocation": str(output.relative_to(ROOT)),
                      "errors": errors}, indent=2), flush=True)
    return 0 if outcome == "pass" else 2 if outcome == "unconfirmed" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=2240)
    parser.add_argument("--height", type=int, default=1260)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--visible", action="store_true",
                        help="Show the trial window for the optional desktop pixel witness")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    if not (320 <= args.width <= 2240 and 240 <= args.height <= 1260):
        parser.error("viewport exceeds the existing NativePresenter bounds")
    if not (8 <= args.frames <= 1000):
        parser.error("frames must be between 8 and 1000")
    if not (30 <= args.timeout <= 900):
        parser.error("timeout must be between 30 and 900 seconds")
    return run(args.width, args.height, args.frames, args.visible, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
