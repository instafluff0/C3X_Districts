#!/usr/bin/env python3
"""Render Lab-owned farms over the unchanged test.biq terrain in Renderer64."""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from pathlib import Path

from Renderer.lab.platform import ROOT, native_command_result, run_native_fixture
from Renderer.renderer import native_inputs

OUT = ROOT / "Renderer/lab/out/farms"
BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
PREVIEW = OUT / "native_preview_farms.exe"
CANDIDATE = ROOT / "Renderer/native/build/farms_candidate/C3XRenderer.dll"
CASES = (
    ("coast-control", (74, 34), 128, 12, False),
    ("coast-noon", (74, 34), 128, 12, True),
    ("coast-sunset", (74, 34), 128, 18, True),
    ("coast-detail", (74, 34), 192, 12, True),
    ("inland-grass", (74, 52), 128, 12, True),
    ("tundra-coast", (18, 82), 128, 12, True),
)


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def ensure_preview() -> None:
    sources = [ROOT / "Renderer/lab/native_preview.cpp",
               ROOT / "Renderer/native/biq_preview.cpp",
               ROOT / "Renderer/native/c3x_renderer_api.h"]
    if PREVIEW.is_file() and PREVIEW.stat().st_mtime >= max(path.stat().st_mtime for path in sources):
        return
    template = (ROOT / "Renderer/lab/build_native_preview.bat").read_text()
    template = template.replace('pushd "%~dp0"', 'pushd "%~dp0\\..\\.."')
    template = template.replace('.cache\\native_preview', 'out\\farms\\native_preview_farms')
    build = OUT / "build_preview.bat"
    build.write_text(template)
    result = native_command_result("Renderer/lab", f'call "{windows(build)}"', timeout_seconds=180)
    if result["status"] != "pass" or not PREVIEW.is_file():
        raise RuntimeError("Farm study preview build failed")


def build_candidate() -> None:
    before = native_inputs()
    receipt = OUT / "candidate_inputs.json"
    if CANDIDATE.is_file() and receipt.is_file():
        recorded = json.loads(receipt.read_text())
        if recorded.get("inputs") == before and recorded.get("dll_sha256") == hashlib.sha256(CANDIDATE.read_bytes()).hexdigest():
            return
    build = OUT / "build_candidate.bat"
    build.write_text(r'''@echo off
setlocal
pushd "%~dp0\..\..\..\native"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "C3X_FARM_VS_RECORD=%TEMP%\c3x-farm-vs-path.txt"
"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath >"%C3X_FARM_VS_RECORD%"
set /p C3X_FARM_VS=<"%C3X_FARM_VS_RECORD%"
del "%C3X_FARM_VS_RECORD%" >nul 2>nul
if not defined C3X_FARM_VS exit /b 2
call "%C3X_FARM_VS%\VC\Auxiliary\Build\vcvars32.bat" >nul
if errorlevel 1 exit /b 2
if not exist "build\farms_obj" mkdir "build\farms_obj"
if not exist "build\farms_candidate" mkdir "build\farms_candidate"
cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:build\farms_obj\ /Fe:build\farms_candidate\C3XRenderer.dll /link /DEF:c3x_renderer.def /IMPLIB:build\farms_candidate\C3XRenderer.lib d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib
exit /b %errorlevel%
''')
    result = native_command_result("Renderer/native", f'call "{windows(build)}"', timeout_seconds=600)
    if result["status"] != "pass" or not CANDIDATE.is_file() or native_inputs() != before:
        raise RuntimeError("Isolated farm candidate build failed or source changed during build")
    receipt.write_text(json.dumps({"inputs":before,
        "dll_sha256":hashlib.sha256(CANDIDATE.read_bytes()).hexdigest()},indent=2)+"\n")


def capture(name: str, center: tuple[int, int], zoom: int, hour: int,
            farms: bool, scene: Path) -> Path:
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
        "C3X_LAB_FARM_STUDY": "1" if farms else "",
        "C3X_LAB_PID_FILE": windows(target / "process.txt"),
        "C3X_LAB_RUN_ID": run_id,
    }
    command = "\n".join(f'set "{key}={value}"' for key, value in settings.items())
    command += (f'\n{windows(PREVIEW)} '
                f'"{windows(CANDIDATE)}" '
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
        raise RuntimeError(f"Farm preview failed: {name}: {result['output_tail']}")
    from PIL import Image
    png = target / "preview.png"
    Image.open(image).convert("RGB").save(png)
    return png


def main() -> None:
    from PIL import Image, ImageDraw
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_preview()
    build_candidate()
    scene = OUT / "test-biq.csv"
    subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"),
                    str(BIQ), str(scene)], cwd=ROOT, check=True)
    dll = CANDIDATE
    farm_pack = ROOT / "Renderer/packs/ImprovementsNormalized/farm_runtime.bin"
    (OUT / "source.txt").write_text(
        f"test_biq_sha256={hashlib.sha256(BIQ.read_bytes()).hexdigest()}\n"
        f"candidate_dll_sha256={hashlib.sha256(dll.read_bytes()).hexdigest()}\n"
        f"farm_pack_sha256={hashlib.sha256(farm_pack.read_bytes()).hexdigest()}\n"
        "Irrigation placement is a deterministic Lab fixture over unchanged BIQ terrain.\n")
    captures = [capture(*case, scene) for case in CASES]
    sheet = Image.new("RGB", (1920, 900), (24, 29, 33))
    draw = ImageDraw.Draw(sheet)
    for index, (case, path) in enumerate(zip(CASES, captures)):
        x = (index % 3) * 640
        y = (index // 3) * 450
        image = Image.open(path)
        image.thumbnail((640, 400))
        sheet.paste(image, (x, y + 28))
        draw.text((x + 8, y + 7), f"{case[0]} | test.biq | {case[2]}px | {case[3]:02}:00", fill="white")
    output = OUT / "test-biq-farm-review.png"
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
