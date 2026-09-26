#!/usr/bin/env python3
"""Render repeatable grassland and plains views of unchanged test.biq terrain."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer.lab.platform import ROOT, run_native_fixture

BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
PREVIEW = ROOT / "Renderer/lab/.cache/native_preview.exe"
CASES = (
    ("grassland", (20, 75), 128, 12),
    ("grassland-close", (20, 75), 192, 12),
    ("grassland-detail", (20, 75), 256, 12),
    ("plains", (60, 65), 128, 12),
    ("plains-close", (60, 65), 192, 12),
    ("plains-detail", (60, 65), 256, 12),
    ("grassland-evening", (20, 75), 128, 18),
    ("plains-evening", (60, 65), 128, 18),
)


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(name: str, center: tuple[int, int], zoom: int, hour: int,
            dll: Path, scene: Path, output: Path) -> Path:
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    image = folder / "preview.bmp"
    run_id = uuid.uuid4().hex
    settings = {
        "C3X_RENDERER_VISUAL_PROFILE": "",
        "C3X_RENDERER_TRACE": "2",
        "C3X_RENDERER_TRACE_FILE": windows(folder / "renderer.log"),
        "C3X_RENDERER_PREVIEW_OBJECTS": "",
        "C3X_RENDERER_PREVIEW_DENSE_SCENE": "",
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": windows(ROOT / "Renderer/custom.custom_rendering.txt"),
        "C3X_RENDERER_SHARED_SCENE_SURFACE": "1",
        "C3X_RENDERER_WORLD_REGIONS": "1",
        "C3X_RENDERER_WORLD_WAVES": "1",
        "C3X_RENDERER_REFLECTION_CONTROL": "0",
        "C3X_RENDERER_WATER_MOTION": "1",
        "C3X_RENDERER_WAVES": "1",
        "C3X_LAB_PID_FILE": windows(folder / "process.txt"),
        "C3X_LAB_RUN_ID": run_id,
    }
    command = "\n".join(f'set "{key}={value}"' for key, value in settings.items())
    command += (f'\n{windows(PREVIEW)} "{windows(dll)}" '
                f'..\\.. "..\\..\\Renderer\\default.custom_rendering.txt" '
                f'"{windows(scene)}" "{windows(image)}" '
                f'960 640 {center[0]} {center[1]} {zoom} {hour}')
    log = windows(folder / "native.log")
    receipt = windows(folder / "completion.txt")
    (folder / "render.bat").write_text(
        "@echo off\nsetlocal\n" + command + f' > "{log}" 2>&1\n'
        'set "C3X_LAB_EXIT=%errorlevel%"\n'
        f'> "{receipt}" echo {run_id} %C3X_LAB_EXIT%\n'
        f'type "{log}"\nexit /b %C3X_LAB_EXIT%\n')
    result = run_native_fixture(folder, f'call "{windows(folder / "render.bat")}"', run_id)
    if result["status"] != "pass" or "0 fallback, output=" not in result["output_tail"] or not image.is_file():
        raise RuntimeError(f"Terrain preview failed: {name}: {result['output_tail']}")
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=[case[0] for case in CASES], action="append")
    args = parser.parse_args()
    dll, output = args.dll.resolve(), args.output.resolve()
    if not dll.is_file() or not PREVIEW.is_file():
        raise FileNotFoundError("Candidate DLL or native preview tool is missing")
    output.mkdir(parents=True, exist_ok=True)
    scene = output / "test-biq.csv"
    subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"),
                    str(BIQ), str(scene)], cwd=ROOT, check=True)
    cases = [case for case in CASES if not args.case or case[0] in args.case]
    images = [capture(*case, dll, scene, output) for case in cases]
    receipt = output / "inputs.json"
    previous = json.loads(receipt.read_text()) if receipt.is_file() else {}
    record = {
        "test_biq_sha256": digest(BIQ),
        "scene_sha256": digest(scene),
        "candidate_dll_sha256": digest(dll),
        "preview_exe_sha256": digest(PREVIEW),
        "terrain_shader_sha256": {
            profile: digest(ROOT / f"Renderer/native/{profile}/terrain.hlsl")
            for profile in ("source_fidelity", "environment_refresh", "city_fidelity")
        },
        "views": {case[0]: digest(image) for case, image in zip(cases, images)},
    }
    if all(previous.get(key) == record[key] for key in record if key != "views"):
        record["views"] = {**previous.get("views", {}), **record["views"]}
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
