"""Run a bounded same-input x86 versus x64-helper scene comparison on the VM.

The trial only reads an existing input journal. It never stages a renderer,
calls INSTALL.bat, or launches Civ III. Output is kept in ignored native/build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.lab.platform import ROOT, native_command_result, windows_root
from Renderer.native.helper_trial.run_gate1 import pe_machine


NATIVE = ROOT / "Renderer/native"
BUILD = NATIVE / "build/helper_trial/gate2"
SOURCES = (
    NATIVE / "c3x_renderer.cpp",
    NATIVE / "gpu_composition_session.h",
    NATIVE / "helper_trial/build_gate2.bat",
    NATIVE / "helper_trial/scene_workload.cpp",
    NATIVE / "helper_trial/run_gate2.py",
)
PARITY = ("sequence", "family", "subtype", "bytes", "result", "width", "height",
          "rendered", "fallback", "hash", "gpu_hash_valid", "gpu_hash")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def statistics(values: list[float]) -> dict:
    values = sorted(values)
    return {"count": len(values), "total_ms": round(sum(values), 3),
            "mean_ms": round(sum(values) / len(values), 3) if values else 0,
            "p95_ms": round(values[max(0, math.ceil(.95 * len(values)) - 1)], 3) if values else 0,
            "max_ms": round(values[-1], 3) if values else 0}


def summarize(rows: list[dict]) -> dict:
    scenes = [item for item in rows if item["family"] == 3]
    return {
        "calls": len(rows), "scene_calls": len(scenes),
        "scene_value_bytes": sum(item["bytes"] for item in scenes),
        "cpu_map_hashes": sum(item["hash"] != "0" * 32 for item in scenes),
        "gpu_map_hashes": sum(item["gpu_hash_valid"] for item in scenes),
        "shared_gpu_maps": sum(item["shared_map"] for item in scenes),
        "scene_service": statistics([item["service_ms"] for item in scenes]),
        "wire_roundtrip_minus_service": statistics([
            max(0, item["roundtrip_ms"] - item["service_ms"]) for item in rows]),
        "x86_gpu_import": statistics([item["import_ms"] for item in scenes if item["shared_map"]]),
        "renderer_peak_private_mib": round(max(item["private_bytes"] for item in rows) / 1048576, 2),
        "x86_driver_peak_private_mib": round(max(item["driver_private_bytes"] for item in rows) / 1048576, 2),
    }


def child_result(directory: Path, token: str) -> int | None:
    try:
        words = (directory / "completion.txt").read_text().split()
        if len(words) == 2 and words[0] == token:
            return int(words[1])
    except (OSError, ValueError):
        pass
    return None


def run(args: argparse.Namespace) -> int:
    capture = args.capture.resolve(strict=True)
    if not capture.is_relative_to(ROOT) or not capture.is_dir():
        raise ValueError("capture must be an existing journal directory under the checkout")
    if not (capture / "segment-000000.c3xi").is_file():
        raise ValueError("capture has no first segment")
    invocation = uuid.uuid4().hex
    output = BUILD / "runs" / invocation
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": 1, "invocation": invocation, "status": "unconfirmed",
               "scope": "recorded_scene_values_and_real_gpu_map_transfer",
               "qualified_for_gameplay": False, "capture": capture.relative_to(ROOT).as_posix(),
               "verify_pixels": args.verify_pixels, "crash_after": args.crash_after,
               "source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in SOURCES}}
    try:
        receipt["capture_segments_sha256"] = {
            segment.name: digest(segment) for segment in sorted(capture.glob("segment-*.c3xi"))}
        if not args.reuse_build:
            for command in ("call helper_trial\\build_gate2.bat", "call BUILD.bat candidate-compile"):
                result = native_command_result("Renderer/native", command, timeout_seconds=600)
                if result["status"] != "pass":
                    raise RuntimeError("candidate build failed; see VM output")
        binaries = {
            "driver": BUILD / "scene_workload_x86.exe",
            "helper": BUILD / "scene_workload_x64.exe",
            "renderer_x86": NATIVE / "build/candidate/C3XRenderer.dll",
            "renderer_x64": BUILD / "C3XRenderer_x64.dll",
        }
        expected = {"driver": 0x014C, "renderer_x86": 0x014C,
                    "helper": 0x8664, "renderer_x64": 0x8664}
        for key, path in binaries.items():
            if pe_machine(path) != expected[key]:
                raise RuntimeError(f"wrong PE machine for {key}")
        receipt["binary_sha256"] = {key: digest(path) for key, path in binaries.items()}
        win = windows_root()
        rows = {}
        for mode in ("x86", "x64"):
            directory = output / mode
            directory.mkdir()
            target = win / directory.relative_to(ROOT)
            driver = win / binaries["driver"].relative_to(ROOT)
            dll = win / binaries[f"renderer_{mode}"].relative_to(ROOT)
            command = (f'"{driver}" --{"remote" if mode == "x64" else "local"} '
                       f'"{dll}" "{win / capture.relative_to(ROOT)}" "{target / "scenes.jsonl"}"')
            if mode == "x64":
                command += f' "{win / binaries["helper"].relative_to(ROOT)}"'
                if args.crash_after:
                    command += f" --crash-after {args.crash_after}"
            if args.verify_pixels:
                command += " --verify-pixels"
            token = uuid.uuid4().hex
            batch = directory / "run.cmd"
            batch.write_text("@echo off\nsetlocal\n" + command + f' >"{target / "run.log"}" 2>&1\n'
                             "set \"RUN_EXIT=%errorlevel%\"\n"
                             f'>"{target / "completion.txt"}" echo {token} %RUN_EXIT%\n'
                             "exit /b %RUN_EXIT%\n")
            result = native_command_result("Renderer/native", f'call "{win / batch.relative_to(ROOT)}"',
                                           timeout_seconds=900)
            code = child_result(directory, token)
            if code is None:
                raise RuntimeError(f"{mode} native completion unconfirmed; inspect this invocation before retry")
            if code:
                raise RuntimeError(f"{mode} scene workload failed: {(directory / 'run.log').read_text(errors='replace')[-600:]}")
            if result["status"] != "pass":
                receipt.setdefault("transport_warnings", []).append(mode)
            rows[mode] = [json.loads(line) for line in (directory / "scenes.jsonl").read_text().splitlines()]
            receipt[mode] = summarize(rows[mode])
        same = len(rows["x86"]) == len(rows["x64"]) and all(
            all(left[field] == right[field] for field in PARITY)
            for left, right in zip(rows["x86"], rows["x64"]))
        if not same:
            raise RuntimeError("x86 and x64 scene results or pixel witnesses differ")
        receipt["parity"] = "pass"
        receipt["status"] = "pass"
        receipt["limitations"] = [
            "Scene-only selection omits native UI, unit, ambient and camera interleaving from the full recorded workload.",
            "GPU copy timings are CPU submission intervals; this run is not a desktop input-to-display measurement.",
            "Diagnostic pixel readback is explicitly opt-in and is excluded from normal performance claims.",
            "A passing capacity reservation is a virtual-address fixture, not a live Civ III memory measurement.",
        ]
    except Exception as error:
        receipt["error"] = str(error)
        receipt["status"] = "fail" if "unconfirmed" not in str(error) else "unconfirmed"
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(output.relative_to(ROOT), receipt["status"])
    if receipt["status"] != "pass":
        print(receipt.get("error", "unknown failure"), file=sys.stderr)
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--reuse-build", action="store_true")
    parser.add_argument("--verify-pixels", action="store_true")
    parser.add_argument("--crash-after", type=int, default=0)
    options = parser.parse_args()
    if options.crash_after and not (1 < options.crash_after < 54):
        parser.error("--crash-after must be between 2 and 53")
    raise SystemExit(run(options))
