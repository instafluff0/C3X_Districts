#!/usr/bin/env python3
"""Render deterministic road layouts over the unchanged test.biq terrain."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.platform import native_command_result, run_native_fixture
from Renderer.renderer import native_inputs

OUT = ROOT / "Renderer/lab/out/roads"
BIQ = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
DLL = ROOT / "Renderer/native/build/roads_candidate/C3XRenderer.dll"
PREVIEW = OUT / "native_preview_roads.exe"
RUNTIME = OUT / "runtime-root"
CASES = (
    ("ancient", (77, 25), 160, 12, 0, "network"),
    ("medieval", (77, 25), 160, 12, 1, "network"),
    ("industrial", (77, 25), 160, 12, 2, "network"),
    ("modern", (77, 25), 160, 12, 3, "network"),
    ("control", (77, 25), 160, 12, None, "network"),
    ("dense-gameplay", (77, 25), 128, 12, 3, "network"),
    ("mountain-river", (63, 57), 160, 12, 2, "network"),
    ("bridge-close", (78, 26), 256, 12, 2, "network"),
    ("forest-river", (85, 25), 160, 12, 1, "network"),
    ("coast-hills", (21, 85), 160, 12, 2, "network"),
    ("sunset", (77, 25), 160, 18, 3, "network"),
    ("isolated-control", (21, 85), 192, 12, None, "isolated"),
    ("isolated-hill", (21, 85), 192, 12, 0, "isolated"),
    ("isolated-plains-control", (77, 25), 192, 12, None, "isolated-plains"),
    ("isolated-plains", (77, 25), 192, 12, 0, "isolated-plains"),
    ("isolated-mountain-control", (77, 23), 192, 12, None, "isolated-mountain"),
    ("isolated-mountain", (77, 23), 192, 12, 0, "isolated-mountain"),
    ("cardinal-control", (80, 26), 256, 12, None, "orthogonal"),
    ("cardinal-cross", (80, 26), 256, 12, 0, "orthogonal"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def build_candidate() -> None:
    before = json.loads((RUNTIME / "renderer-inputs.json").read_text())["source_inputs"]
    receipt = OUT / "candidate_inputs.json"
    if DLL.is_file() and receipt.is_file():
        recorded = json.loads(receipt.read_text())
        if recorded.get("inputs") == before and recorded.get("dll_sha256") == digest(DLL):
            return
    build = OUT / "build_candidate.bat"
    build.write_text(r'''@echo off
setlocal
pushd "%~dp0\runtime-root\Renderer\native"
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
    snapshot_dll = RUNTIME / "Renderer/native/build/roads_candidate/C3XRenderer.dll"
    if result["status"] != "pass" or not snapshot_dll.is_file():
        raise RuntimeError("Isolated road candidate build failed: " + result["output_tail"])
    DLL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot_dll, DLL)
    receipt.write_text(json.dumps({"inputs": before, "dll_sha256": digest(DLL)}, indent=2) + "\n")


def build_preview() -> None:
    sources = (ROOT / "Renderer/lab/native_preview.cpp",
               ROOT / "Renderer/native/biq_preview.cpp",
               ROOT / "Renderer/native/c3x_renderer_api.h")
    if PREVIEW.is_file() and PREVIEW.stat().st_mtime >= max(path.stat().st_mtime for path in sources):
        return
    template = (ROOT / "Renderer/lab/build_native_preview.bat").read_text()
    template = template.replace('pushd "%~dp0"', 'pushd "%~dp0\\..\\.."')
    template = template.replace('.cache\\native_preview', 'out\\roads\\native_preview_roads')
    build = OUT / "build_preview.bat"
    build.write_text(template)
    result = native_command_result("Renderer/lab", f'call "{windows(build)}"', timeout_seconds=180)
    if result["status"] != "pass" or not PREVIEW.is_file():
        raise RuntimeError("Isolated road preview build failed: " + result["output_tail"])


def runtime_root() -> None:
    marker = RUNTIME / "renderer-inputs.json"
    if RUNTIME.exists() and not marker.is_file():
        raise RuntimeError("Incomplete private road Lab root; inspect before reusing")
    if not RUNTIME.exists():
        (RUNTIME / "Renderer").mkdir(parents=True)
        shutil.copytree(ROOT / "Renderer/native", RUNTIME / "Renderer/native",
                        ignore=shutil.ignore_patterns("build", "*.obj", "*.ilk", "*.pdb"))
        shutil.copytree(ROOT / "Renderer/packs", RUNTIME / "Renderer/packs",
                        copy_function=os.link)
        for name in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
            shutil.copy2(ROOT / "Renderer" / name, RUNTIME / "Renderer" / name)
    # Generated art packs may be replaced in the shared checkout during a Lab
    # run. Keep this candidate's hardlinks on the current source inodes.
    for source in (ROOT / "Renderer/packs").rglob("*"):
        if not source.is_file():
            continue
        destination = RUNTIME / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or source.stat().st_ino != destination.stat().st_ino:
            if destination.exists():
                destination.unlink()
            os.link(source, destination)
    pack_inputs = {name: digest(RUNTIME / name) for name in (
        "Renderer/packs/NaturalFidelityRuntime/natural.bin",
        "Renderer/packs/RouteDoodadsNormalized/bridge_runtime.bin",
        "Renderer/packs/RouteStylesNormalized/manifest.json")}
    source_inputs = native_inputs()
    if marker.is_file():
        previous = json.loads(marker.read_text())
        if previous.get("source_inputs") == source_inputs and \
           previous.get("pack_inputs") == pack_inputs and \
           "private_vegetation_shadow_repair" in previous:
            return
    for relative, expected in source_inputs.items():
        destination = RUNTIME / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        if digest(destination) != expected:
            raise RuntimeError(f"Native source changed while taking Lab snapshot: {relative}")
    shader = RUNTIME / "Renderer/native/source_fidelity/terrain.hlsl"
    source = shader.read_text()
    missing = "    float coast_coverage : TEXCOORD4;\n};\n#ifdef SANDBOX_TERRAIN_MATERIAL"
    repaired = missing in source
    if repaired:
        source = source.replace(missing,
            "    float coast_coverage : TEXCOORD4;\n"
            "    float coast_inland : TEXCOORD5;\n};\n#ifdef SANDBOX_TERRAIN_MATERIAL", 1)
        shader.write_text(source)
    vegetation = RUNTIME / "Renderer/lab/shared/natural/vegetation_floor_mesh_body.h"
    vegetation_source = vegetation.read_text()
    shadowed = "for(unsigned index=0;index<decal.vertex_count;index+=3)" in vegetation_source
    if shadowed:
        vegetation_source = vegetation_source.replace(
            "for(unsigned index=0;index<decal.vertex_count;index+=3)",
            "for(unsigned decal_vertex=0;decal_vertex<decal.vertex_count;decal_vertex+=3)", 1)
        vegetation_source = vegetation_source.replace("decal.first+index", "decal.first+decal_vertex")
        vegetation.write_text(vegetation_source)
    marker.write_text(json.dumps({
        "private_shader_declaration_repair": repaired,
        "private_vegetation_shadow_repair": shadowed,
        "source_inputs": source_inputs,
        "pack_inputs": pack_inputs,
        "source_terrain_shader_sha256": digest(ROOT / "Renderer/native/source_fidelity/terrain.hlsl"),
        "private_terrain_shader_sha256": digest(shader),
        "private_vegetation_source_sha256": digest(vegetation),
        "pack_layout": "hardlinked read-only local art; not redistributed",
    }, indent=2) + "\n")


def capture(name: str, center: tuple[int, int], zoom: int, hour: int,
            era: int | None, layout: str, scene: Path) -> Path:
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    image = target / "preview.bmp"
    png = target / "preview.png"
    capture_receipt = target / "capture.json"
    identity = {"candidate_dll_sha256": digest(DLL),
                "preview_exe_sha256": digest(PREVIEW),
                "scene_sha256": digest(scene),
                "runtime_shader_sha256": digest(RUNTIME / "Renderer/native/source_fidelity/terrain.hlsl"),
                "runtime_pack_inputs": json.loads((RUNTIME / "renderer-inputs.json").read_text())["pack_inputs"],
                "center": list(center), "zoom": zoom, "hour": hour,
                "era": era, "layout": layout}
    if capture_receipt.is_file() and png.is_file():
        previous = json.loads(capture_receipt.read_text())
        if all(previous.get(key) == value for key, value in identity.items()) and \
           previous.get("image_sha256") == digest(png):
            return png
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
    }
    log = windows(target / "native.log")
    receipt = windows(target / "completion.txt")
    for attempt in range(2):
        run_id = uuid.uuid4().hex
        settings["C3X_LAB_RUN_ID"] = run_id
        command = "\n".join(f'set "{key}={value}"' for key, value in settings.items())
        command += (f'\n{windows(PREVIEW)} '
                    f'"{windows(DLL)}" '
                    f'"{windows(RUNTIME)}" "..\\..\\Renderer\\default.custom_rendering.txt" '
                    f'"{windows(scene)}" "{windows(image)}" '
                    f'960 640 {center[0]} {center[1]} {zoom} {hour}')
        (target / "render.bat").write_text(
            "@echo off\nsetlocal\n" + command + f' > "{log}" 2>&1\n'
            "set \"C3X_LAB_EXIT=%errorlevel%\"\n"
            f'> "{receipt}" echo {run_id} %C3X_LAB_EXIT%\n'
            f'type "{log}"\nexit /b %C3X_LAB_EXIT%\n')
        result = run_native_fixture(target, f'call "{windows(target / "render.bat")}"', run_id)
        if result["status"] == "pass" and "0 fallback, output=" in result["output_tail"] and image.is_file():
            break
        if attempt == 1:
            raise RuntimeError(f"Road preview failed: {name}: {result['output_tail']}")
        time.sleep(2)
    from PIL import Image
    Image.open(image).convert("RGB").save(png)
    capture_receipt.write_text(json.dumps({**identity, "image_sha256": digest(png)}, indent=2) + "\n")
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
    runtime_root()
    build_candidate()
    build_preview()
    scene = OUT / "test-biq.csv"
    subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"),
                    str(BIQ), str(scene)], cwd=ROOT, check=True)
    captures = {case[0]: capture(*case, scene) for case in CASES}
    from PIL import Image, ImageChops
    for control,road in (("isolated-control","isolated-hill"),
                         ("isolated-plains-control","isolated-plains"),
                         ("isolated-mountain-control","isolated-mountain")):
        with Image.open(captures[control]) as before, Image.open(captures[road]) as after:
            if ImageChops.difference(before,after).getbbox() is None:
                raise RuntimeError(f"Built road is invisible: {road}")
    duplicate = capture("modern-repeat", (77, 25), 160, 12, 3, "network", scene)
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
        ("context", ("control", "dense-gameplay", "mountain-river", "forest-river", "sunset", "coast-hills", "bridge-close")),
        ("isolated", ("isolated-control", "isolated-hill", "isolated-plains-control", "isolated-plains", "isolated-mountain-control", "isolated-mountain")),
        ("cardinal", ("cardinal-control", "cardinal-cross")),
    ):
        print(sheet(label, names, captures), flush=True)


if __name__ == "__main__":
    main()
