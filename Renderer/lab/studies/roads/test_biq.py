#!/usr/bin/env python3
"""Render deterministic road layouts over the unchanged test.biq terrain."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.platform import native_command_result, run_native_fixture
from Renderer.renderer import ensure_preview_tool, native_inputs

OUT = ROOT / "Renderer/lab/out/roads"
BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
DLL = ROOT / "Renderer/native/build/roads_candidate/C3XRenderer.dll"
PREVIEW = ROOT / "Renderer/lab/.cache/native_preview.exe"
CASES = (
    ("ancient", (63, 57), 160, 12, 0, "network"),
    ("medieval", (63, 57), 160, 12, 1, "network"),
    ("industrial", (63, 57), 160, 12, 2, "network"),
    ("modern", (63, 57), 160, 12, 3, "network"),
    ("control", (63, 57), 160, 12, None, "network"),
    ("dense-gameplay", (63, 57), 128, 12, 3, "network"),
    ("forest-river", (85, 25), 160, 12, 1, "network"),
    ("coast-hills", (21, 85), 160, 12, 2, "network"),
    ("sunset", (63, 57), 160, 18, 3, "network"),
    ("isolated-hill", (21, 85), 192, 12, 0, "isolated"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def build_candidate() -> None:
    before = native_inputs()
    receipt = OUT / "candidate_inputs.json"
    if DLL.is_file() and receipt.is_file():
        recorded = json.loads(receipt.read_text())
        if recorded.get("inputs") == before and recorded.get("dll_sha256") == digest(DLL):
            return
    build = OUT / "build_candidate.bat"
    build.write_text(r'''@echo off
setlocal
pushd "%~dp0\..\..\..\native"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "C3X_ROAD_VS_RECORD=%TEMP%\c3x-road-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_ROAD_VS_RECORD%"
set /p C3X_ROAD_VS=<"%C3X_ROAD_VS_RECORD%"
del "%C3X_ROAD_VS_RECORD%" >nul 2>nul
if not defined C3X_ROAD_VS exit /b 2
call "%C3X_ROAD_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 2
if not exist "build\roads_obj" mkdir "build\roads_obj"
if not exist "build\roads_candidate" mkdir "build\roads_candidate"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\roads_obj\ /Fe:build\roads_candidate\C3XRenderer.dll /link /DEF:c3x_renderer.def /IMPLIB:build\roads_candidate\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
exit /b %errorlevel%
''')
    result = native_command_result("Renderer/native", f'call "{windows(build)}"', timeout_seconds=600)
    if result["status"] != "pass" or not DLL.is_file() or native_inputs() != before:
        raise RuntimeError("Isolated road candidate build failed or inputs changed: " + result["output_tail"])
    receipt.write_text(json.dumps({"inputs": before, "dll_sha256": digest(DLL)}, indent=2) + "\n")


def capture(name: str, center: tuple[int, int], zoom: int, hour: int,
            era: int | None, layout: str, scene: Path) -> Path:
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    image = target / "preview.bmp"
    run_id = uuid.uuid4().hex
    settings = {
        "C3X_RENDERER_VISUAL_PROFILE": "",
        "C3X_RENDERER_TRACE": "2",
        "C3X_RENDERER_TRACE_FILE": windows(target / "renderer.log"),
        "C3X_RENDERER_PREVIEW_OBJECTS": "",
        "C3X_RENDERER_PREVIEW_DENSE_SCENE": "",
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS": windows(ROOT / "Renderer/custom.custom_rendering.txt"),
        "C3X_RENDERER_SHARED_SCENE_SURFACE": "1",
        "C3X_RENDERER_WORLD_REGIONS": "1",
        "C3X_RENDERER_WORLD_WAVES": "1",
        "C3X_RENDERER_REFLECTION_CONTROL": "0",
        "C3X_RENDERER_WATER_MOTION": "1",
        "C3X_RENDERER_WAVES": "1",
        "C3X_LAB_ROAD_STUDY": "" if era is None else str(era),
        "C3X_LAB_ROAD_LAYOUT": layout,
        "C3X_LAB_PID_FILE": windows(target / "process.txt"),
        "C3X_LAB_RUN_ID": run_id,
    }
    command = "\n".join(f'set "{key}={value}"' for key, value in settings.items())
    command += (f'\n{windows(PREVIEW)} '
                f'"{windows(DLL)}" '
                f'..\\.. "..\\..\\Renderer\\default.custom_rendering.txt" '
                f'"{windows(scene)}" "{windows(image)}" '
                f'960 640 {center[0]} {center[1]} {zoom} {hour}')
    log = windows(target / "native.log")
    receipt = windows(target / "completion.txt")
    (target / "render.bat").write_text(
        "@echo off\nsetlocal\n" + command + f' > "{log}" 2>&1\n'
        "set \"C3X_LAB_EXIT=%errorlevel%\"\n"
        f'> "{receipt}" echo {run_id} %C3X_LAB_EXIT%\n'
        f'type "{log}"\nexit /b %C3X_LAB_EXIT%\n')
    result = run_native_fixture(target, f'call "{windows(target / "render.bat")}"', run_id)
    if result["status"] != "pass" or "0 fallback, output=" not in result["output_tail"] or not image.is_file():
        raise RuntimeError(f"Road preview failed: {name}: {result['output_tail']}")
    from PIL import Image
    png = target / "preview.png"
    Image.open(image).convert("RGB").save(png)
    return png


def sheet(label: str, names: tuple[str, ...], captures: dict[str, Path]) -> Path:
    from PIL import Image, ImageDraw
    columns = 2
    rows = (len(names) + columns - 1) // columns
    canvas = Image.new("RGB", (1920, rows * 675), "#20252b")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(names):
        x, y = (index % columns) * 960, (index // columns) * 675
        canvas.paste(Image.open(captures[name]), (x, y + 30))
        draw.text((x + 14, y + 8), f"{name} | test.biq | {CASES[next(i for i,c in enumerate(CASES) if c[0]==name)][2]}px",
                  fill="white")
    output = OUT / f"test-biq-road-{label}.png"
    canvas.save(output)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    build_candidate()
    ensure_preview_tool()
    scene = OUT / "test-biq.csv"
    subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"),
                    str(BIQ), str(scene)], cwd=ROOT, check=True)
    captures = {case[0]: capture(*case, scene) for case in CASES}
    duplicate = capture("modern-repeat", (63, 57), 160, 12, 3, "network", scene)
    if digest(captures["modern"]) != digest(duplicate):
        raise RuntimeError("Road layout changed between identical captures")
    (OUT / "inputs.json").write_text(json.dumps({
        "test_biq_sha256": digest(BIQ),
        "exported_scene_sha256": digest(scene),
        "candidate_dll_sha256": digest(DLL),
        "preview_exe_sha256": digest(PREVIEW),
        "identical_modern_replay": True,
    }, indent=2) + "\n")
    for label, names in (
        ("eras", ("ancient", "medieval", "industrial", "modern")),
        ("context", ("control", "dense-gameplay", "forest-river", "coast-hills", "sunset", "isolated-hill")),
    ):
        print(sheet(label, names, captures), flush=True)


if __name__ == "__main__":
    main()
