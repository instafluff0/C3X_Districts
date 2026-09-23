"""Run every recorded native call in x86 while a scene sidecar runs in x64.

This is an interleaving diagnostic, not the production split: the x86 control
still renders its own scene. No candidate is staged or game process launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.lab.platform import ROOT, native_command_result, windows_root
from Renderer.native.helper_trial.run_gate1 import pe_machine

NATIVE = ROOT / "Renderer/native"
OUT = NATIVE / "build/helper_trial/interleaved"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def run(capture: Path, reuse_build: bool) -> int:
    capture = capture.resolve(strict=True)
    if not capture.is_relative_to(ROOT) or not (capture / "segment-000000.c3xi").is_file():
        raise ValueError("capture must be a journal directory under the checkout")
    run_id = uuid.uuid4().hex
    output = OUT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": 1, "status": "unconfirmed", "invocation": run_id,
               "qualified_for_gameplay": False, "capture": capture.relative_to(ROOT).as_posix(),
               "scope": "full_x86_native_replay_with_x64_scene_sidecar"}
    try:
        if not reuse_build:
            for command in ("call helper_trial\\build_gate2.bat", "call BUILD.bat candidate-compile",
                            "call BUILD.bat input-replay"):
                result = native_command_result("Renderer/native", command, timeout_seconds=600)
                if result["status"] != "pass":
                    raise RuntimeError("native build failed; see VM output")
        binaries = {
            "replay": NATIVE / "build/input-recording/replay_inputs.exe",
            "helper": NATIVE / "build/helper_trial/gate2/scene_workload_x64.exe",
            "x86": NATIVE / "build/candidate/C3XRenderer.dll",
            "x64": NATIVE / "build/helper_trial/gate2/C3XRenderer_x64.dll",
        }
        for name, path in binaries.items():
            if pe_machine(path) != (0x8664 if name in ("helper", "x64") else 0x014C):
                raise RuntimeError("wrong PE machine for " + name)
        receipt["binary_sha256"] = {name: digest(path) for name, path in binaries.items()}
        receipt["source_sha256"] = {path.relative_to(ROOT).as_posix(): digest(path)
            for path in (NATIVE / "replay_inputs.cpp", NATIVE / "helper_trial/scene_client.h",
                         NATIVE / "helper_trial/shared_frame_reader.h", NATIVE / "BUILD.bat",
                         NATIVE / "input_recording/replay.h",
                         NATIVE / "helper_trial/scene_wire.h", NATIVE / "helper_trial/scene_workload.cpp",
                         NATIVE / "helper_trial/run_interleaved.py", NATIVE / "helper_trial/build_gate2.bat",
                         NATIVE / "c3x_renderer.cpp", NATIVE / "gpu_composition_session.h")}
        receipt["capture_segments_sha256"] = {path.name: digest(path)
            for path in sorted(capture.glob("segment-*.c3xi"))}
        win = windows_root()
        target = win / output.relative_to(ROOT)
        command = (f'"{win / binaries["replay"].relative_to(ROOT)}" --development '
                   f'"{win / binaries["x86"].relative_to(ROOT)}" '
                   f'"{win / capture.relative_to(ROOT)}" --compare-candidate '
                   f'--performance "{target / "x86.jsonl"}" --x64-scene '
                   f'"{win / binaries["helper"].relative_to(ROOT)}" '
                   f'"{win / binaries["x64"].relative_to(ROOT)}" '
                   f'"{target / "x64-scenes.jsonl"}"')
        token = uuid.uuid4().hex
        batch = output / "run.cmd"
        batch.write_text("@echo off\nsetlocal\n" + command + f' >"{target / "run.log"}" 2>&1\n'
                         "set \"RUN_EXIT=%errorlevel%\"\n"
                         f'>"{target / "completion.txt"}" echo {token} %RUN_EXIT%\n'
                         "exit /b %RUN_EXIT%\n")
        result = native_command_result("Renderer/native",
            f'call "{win / batch.relative_to(ROOT)}"', timeout_seconds=1200)
        completion = (output / "completion.txt").read_text().split() if (output / "completion.txt").is_file() else []
        if len(completion) != 2 or completion[0] != token:
            raise RuntimeError("native completion unconfirmed; inspect this invocation before retry")
        if int(completion[1]):
            raise RuntimeError((output / "run.log").read_text(errors="replace")[-1000:])
        if result["status"] != "pass":
            receipt["transport_warning"] = True
        lines = (output / "run.log").read_text(errors="replace").splitlines()
        summary = next((json.loads(line) for line in reversed(lines)
                        if line.startswith('{"status":"development_input_replay"')), None)
        if not summary or not summary["complete"]:
            raise RuntimeError("full native replay did not complete")
        scene = [json.loads(line) for line in (output / "x64-scenes.jsonl").read_text().splitlines()]
        native = [json.loads(line) for line in (output / "x86.jsonl").read_text().splitlines()]
        if not scene or not any(row["kind"] == 3 for row in scene) or len(native) != summary["calls"]:
            raise RuntimeError("interleaved call coverage incomplete")
        gpu_phase = False
        for row in scene:
            if row["kind"] == 19 and row["subtype"] in (6, 8):
                gpu_phase = False
            if row["kind"] == 3 and row["subtype"] == 3 and row["helper_result"] == 1:
                gpu_phase = True
            row["gpu_phase"] = gpu_phase
        receipt["x86"] = summary
        receipt["x64_scene"] = {"operations": len(scene), "scene_calls": sum(row["kind"] == 3 for row in scene),
            "presentations": sum(row["kind"] == 13 for row in scene),
            "visual_offers": sum(row["kind"] == 12 for row in scene),
            "skipped_native_admissions": sum(not row["executed"] for row in scene),
            "final_frames": sum(row["final_image"] for row in scene),
            "final_valid": sum(row["final_valid"] for row in scene),
            "final_matches": sum(row["final_match"] for row in scene),
            "final_mismatches": [row["sequence"] for row in scene if row["final_valid"] and not row["final_match"]],
            "gpu_phase_final_frames": sum(row["final_valid"] and row["gpu_phase"] for row in scene),
            "gpu_phase_final_matches": sum(row["final_match"] and row["gpu_phase"] for row in scene),
            "final_pixel_differences": [{"sequence": row["sequence"], "pixels": row["different_pixels"],
                "max_channel_delta": row["max_channel_delta"], "bounds": row["difference_bounds"]}
                for row in scene if row["final_valid"] and not row["final_match"] and row["diff_available"]],
            "result_mismatches": [row["sequence"] for row in scene if row["executed"] and not row["matches"]],
            "pixel_witnesses": sum(row["pixel_witness"] for row in scene),
            "pixel_mismatches": [row["sequence"] for row in scene if not row["pixels_match"]],
            "unit_bounds_mismatches": [row["sequence"] for row in scene if not row["bounds_match"]],
            "service_ms": round(sum(row["service_ms"] for row in scene), 3),
            "roundtrip_ms": round(sum(row["roundtrip_ms"] for row in scene), 3),
            "peak_private_mib": round(max(row["x64_private_bytes"] for row in scene) / 1048576, 2)}
        receipt["x86_peak_private_mib"] = round(max(row.get("private_bytes", 0) for row in native) / 1048576, 2)
        receipt["status"] = "pass"
    except Exception as error:
        receipt["error"] = str(error)
        receipt["status"] = "unconfirmed" if "unconfirmed" in str(error) else "fail"
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output.relative_to(ROOT), receipt["status"])
    if receipt["status"] != "pass":
        print(receipt.get("error", "unknown failure"), file=sys.stderr)
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--reuse-build", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(args.capture, args.reuse_build))
