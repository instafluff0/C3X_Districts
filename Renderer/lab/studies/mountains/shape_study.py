#!/usr/bin/env python3
"""Render sandbox-identical mountain controls and Lab-only shape proposals.

The sandbox renderer and native sources are copied into lab/out before building.
Only the copied mountain mesh is edited. Linked packs are read-only inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab import platform

OUT = ROOT / "Renderer/lab/out/mountains/shape-study"
PRIVATE = OUT / "root"
MESH = Path("Renderer/lab/shared/natural/relief_mesh_body.h")
BASELINE = OUT / "baseline-relief_mesh_body.h"

SHAPES = {
    "sandbox": (1.0, 1.0),
    "lower": (0.68, 1.08),
    "squat": (0.55, 1.10),
}
BIQ_VIEWS = {
    "coastal-ridge": (57, 33),
    "wooded-range": (63, 58),
    "jungle-coast": (17, 49),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_sources() -> None:
    if PRIVATE.exists():
        if not BASELINE.is_file() or not (OUT / "baseline-inputs.txt").is_file():
            raise RuntimeError("Partial isolated Lab copy; preserve its inputs before rebuilding")
        return
    (PRIVATE / "Renderer").mkdir(parents=True)
    shutil.copytree(ROOT / "Renderer/native", PRIVATE / "Renderer/native",
                    ignore=shutil.ignore_patterns("build", "*.obj", "*.ilk", "*.pdb"))
    shutil.copytree(ROOT / "Renderer/sandbox", PRIVATE / "Renderer/sandbox",
                    ignore=shutil.ignore_patterns("out"))
    shutil.copytree(ROOT / "Renderer/lab/shared", PRIVATE / "Renderer/lab/shared")
    for name in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
        shutil.copy2(ROOT / "Renderer" / name, PRIVATE / "Renderer" / name)
    # The builders never write through these links. Preserve ignored source art.
    shutil.copytree(ROOT / "Renderer/packs", PRIVATE / "Renderer/packs",
                    copy_function=os.link)
    shutil.copy2(PRIVATE / MESH, BASELINE)
    (OUT / "baseline-inputs.txt").write_text(
        "Sandbox drawing source: Renderer/sandbox/resident_scene.cpp\n"
        "Mountain geometry: Renderer/lab/shared/natural/relief_mesh_body.h\n"
        f"mesh_sha256={digest(ROOT / MESH)}\n"
        f"sandbox_renderer_sha256={digest(ROOT / 'Renderer/sandbox/resident_scene.cpp')}\n"
        f"sandbox_pipeline_sha256={digest(ROOT / 'Renderer/sandbox/fresh_pipeline.h')}\n")


def shaped_mesh(source: str, height: float, width: float) -> str:
    if (height, width) == (1.0, 1.0):
        return source
    replacements = {
        "connected?(turn?2.08f:2.46f):1.85f":
            f"connected?(turn?{2.08*width:.5f}f:{2.46*width:.5f}f):{1.85*width:.5f}f",
        "connected?(turn?1.82f:1.34f):1.55f":
            f"connected?(turn?{1.82*width:.5f}f:{1.34*width:.5f}f):{1.55*width:.5f}f",
        "connected?142.f:165.f":
            f"connected?{142*height:.5f}f:{165*height:.5f}f",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Mountain mesh expression changed: " + old)
        source = source.replace(old, new)
    return source


def scene(path: Path, kind: str) -> list[tuple[int, int, int]]:
    size = 64
    isolated = [(x, y) for y in (26, 30, 34, 38) for x in (26, 30, 34, 38)]
    connected = [(29, 29), (30, 30), (31, 31), (32, 32), (33, 33),
                 (34, 34), (35, 35), (33, 31), (34, 30), (32, 34)]
    points = isolated if kind == "variety" else connected
    rows = [f"{x},{y},2,{6 if (x, y) in points else 2},0,0,0"
            for y in range(size) for x in range(y % 2, size, 2)]
    path.write_text(f"C3X_BIQ_TERRAIN_V3,{size},{size},{len(rows)}\n" +
                    "\n".join(rows) + "\n")
    return [(x, y, ((x * 0x193) ^ (y * 0x217) ^ 0x6b91) % 5)
            for x, y in points]


def build() -> None:
    from Renderer.lab.platform import native_command_result
    result = native_command_result(
        "Renderer/lab/out/mountains/shape-study/root/Renderer/sandbox",
        "call build_reference_x64.bat", timeout_seconds=600)
    if result["status"] != "pass":
        raise RuntimeError("Isolated sandbox renderer build failed")
    dll = PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll"
    if not dll.is_file():
        raise RuntimeError("Isolated sandbox DLL missing after build")


def windows(path: Path) -> str:
    return "..\\..\\" + path.relative_to(ROOT).as_posix().replace("/", "\\")


def render(label: str, kind: str) -> Path:
    output = OUT / label / kind
    output.mkdir(parents=True, exist_ok=True)
    placed = scene(output / "scene.csv", kind)
    (output / "variants.txt").write_text("raw_x raw_y authored_variant\n" +
        "".join(f"{x} {y} {variant + 1}\n" for x, y, variant in placed))
    dll = PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll"
    exe = PRIVATE / "Renderer/sandbox/out/client_x64.exe"
    capture = output / "sandbox-frame.bmp"
    invocation = output / "run.bat"
    invocation.write_text(
        "@echo off\nsetlocal\n"
        "set \"C3X_RENDERER_VISUAL_PROFILE=\"\n"
        "set \"C3X_RENDERER_TRACE=0\"\n"
        "set \"C3X_RENDERER_SHARED_SCENE_SURFACE=1\"\n"
        "set \"C3X_RENDERER_WATER_MOTION=1\"\n"
        "set \"C3X_RENDERER_WAVES=1\"\n"
        "set \"C3X_SANDBOX_SHADOW_PATCHES=1\"\n"
        "set \"C3X_RENDERER_PREVIEW_UNITS=1\"\n"
        "set \"C3X_SANDBOX_UNITS=1\"\n"
        "set \"C3X_SANDBOX_REPLAY_CLIP=1\"\n"
        "set \"C3X_SANDBOX_CLIP_FRAMES=1\"\n"
        f"set \"C3X_SANDBOX_CAPTURE={windows(capture)}\"\n"
        f"set \"C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS={windows(PRIVATE / 'Renderer/custom.custom_rendering.txt')}\"\n"
        f'"{windows(exe)}" "{windows(dll)}" "{windows(PRIVATE)}" '
        f'"{windows(PRIVATE / "Renderer/default.custom_rendering.txt")}" '
        f'"{windows(output / "scene.csv")}" "{windows(output / "unused.bmp")}" '
        "1280 800 32 32 128 12\nexit /b %errorlevel%\n")
    result = platform.native_command_result("Renderer/native",
        f'call "{windows(invocation)}"', timeout_seconds=600)
    if result["status"] != "pass" or not capture.is_file() or capture.stat().st_size < 100000:
        raise RuntimeError(f"Sandbox {label}/{kind} frame did not complete")
    from PIL import Image
    png = output / "preview.png"
    Image.open(capture).convert("RGB").save(png)
    return png


def render_biq_view(label: str, name: str, scene_path: Path) -> Path:
    """Capture one unchanged test.biq camera through the isolated sandbox client."""
    output = OUT / label / "biq" / name
    output.mkdir(parents=True, exist_ok=True)
    dll = OUT / label / "C3XReference_x64.dll"
    exe = PRIVATE / "Renderer/sandbox/out/client_x64.exe"
    capture = output / "sandbox-frame.bmp"
    center_x, center_y = BIQ_VIEWS[name]
    invocation = output / "run.bat"
    invocation.write_text(
        "@echo off\nsetlocal\n"
        "set \"C3X_RENDERER_VISUAL_PROFILE=\"\n"
        "set \"C3X_RENDERER_TRACE=0\"\n"
        "set \"C3X_RENDERER_SHARED_SCENE_SURFACE=1\"\n"
        "set \"C3X_RENDERER_WATER_MOTION=1\"\n"
        "set \"C3X_RENDERER_WAVES=1\"\n"
        "set \"C3X_SANDBOX_SHADOW_PATCHES=1\"\n"
        "set \"C3X_SANDBOX_WHOLE_WORLD=1\"\n"
        "set \"C3X_RENDERER_PREVIEW_UNITS=1\"\n"
        "set \"C3X_SANDBOX_UNITS=1\"\n"
        "set \"C3X_SANDBOX_REPLAY_CLIP=1\"\n"
        "set \"C3X_SANDBOX_CLIP_FRAMES=1\"\n"
        f"set \"C3X_SANDBOX_CAPTURE={windows(capture)}\"\n"
        f"set \"C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS={windows(PRIVATE / 'Renderer/custom.custom_rendering.txt')}\"\n"
        f'"{windows(exe)}" "{windows(dll)}" "{windows(PRIVATE)}" '
        f'"{windows(PRIVATE / "Renderer/default.custom_rendering.txt")}" '
        f'"{windows(scene_path)}" "{windows(output / "unused.bmp")}" '
        f"1600 900 {center_x} {center_y} 128 12\nexit /b %errorlevel%\n")
    result = platform.native_command_result("Renderer/native",
        f'call "{windows(invocation)}"', timeout_seconds=600)
    if (result["status"] != "pass" or "fallback=0" not in result["output_tail"] or
            not capture.is_file() or capture.stat().st_size < 100000):
        raise RuntimeError(f"Sandbox test.biq {label}/{name} frame did not complete")
    from PIL import Image
    png = output / "preview.png"
    Image.open(capture).convert("RGB").save(png)
    (output / "capture.txt").write_text(
        f"biq_sha256={digest(ROOT / 'Renderer/packs/RendererSourceStudies/maps/test.biq')}\n"
        f"scene_sha256={digest(scene_path)}\n"
        f"dll_sha256={digest(dll)}\n"
        f"camera={center_x},{center_y}\ntile_width=128\nhour=12\n")
    return png


def biq_examples() -> None:
    copy_sources()
    for label in ("lower", "sandbox"):
        dll = OUT / label / "C3XReference_x64.dll"
        receipt = OUT / label / "build.txt"
        if not dll.is_file() or not receipt.is_file() or f"dll_sha256={digest(dll)}" not in receipt.read_text():
            raise RuntimeError(f"Saved isolated {label} DLL is missing or changed")
    scene_path = OUT / "lower/biq/test-biq.csv"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"),
                    str(ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"),
                    str(scene_path)], cwd=ROOT, check=True)
    source_hash = digest(ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq")
    scene_hash = digest(scene_path)
    def current(label: str, name: str) -> bool:
        folder = OUT / label / "biq" / name
        record = folder / "capture.txt"
        if not (folder / "preview.png").is_file() or not record.is_file():
            return False
        values = dict(line.split("=", 1) for line in record.read_text().splitlines())
        return (values.get("biq_sha256") == source_hash and
                values.get("scene_sha256") == scene_hash and
                values.get("dll_sha256") == digest(OUT / label / "C3XReference_x64.dll") and
                values.get("camera") == ",".join(map(str, BIQ_VIEWS[name])))
    for name in BIQ_VIEWS:
        if not current("lower", name):
            render_biq_view("lower", name, scene_path)
    control = OUT / "sandbox/biq/coastal-ridge/preview.png"
    if not current("sandbox", "coastal-ridge"):
        render_biq_view("sandbox", "coastal-ridge", scene_path)
    from PIL import Image, ImageDraw
    baseline = Image.open(control).convert("RGB")
    lower = Image.open(OUT / "lower/biq/coastal-ridge/preview.png").convert("RGB")
    canvas = Image.new("RGB", (baseline.width * 2, baseline.height + 45), "#20252d")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(baseline, (0, 45))
    canvas.paste(lower, (baseline.width, 45))
    draw.text((16, 15), "Sandbox baseline | test.biq coastal ridge", fill="white")
    draw.text((baseline.width + 16, 15), "Lower | 68% height, 108% width", fill="white")
    canvas.save(OUT / "lower/biq/coastal-ridge-comparison.png")


def sheet(images: dict[str, dict[str, Path]]) -> None:
    from PIL import Image, ImageDraw
    labels = [label for label in SHAPES if label in images]
    for kind in ("variety", "connected"):
        if kind == "variety":
            frames = [Image.open(OUT / label / "variety-atlas.png").convert("RGB")
                      for label in labels]
        else:
            frames = [Image.open(images[label][kind]).convert("RGB").crop((320, 210, 960, 550))
                      for label in labels]
        width, height = frames[0].size
        canvas = Image.new("RGB", (width * len(frames), height + 54), "#20252d")
        draw = ImageDraw.Draw(canvas)
        for index, (label, frame) in enumerate(zip(labels, frames)):
            canvas.paste(frame, (index * width, 54))
            height_scale, width_scale = SHAPES[label]
            draw.text((index * width + 18, 16),
                      f"{label} | height {height_scale:.2f} | width {width_scale:.2f} | sandbox renderer",
                      fill="white")
        canvas.save(OUT / f"comparison-{kind}.png")


def atlas(label: str, image: Path) -> None:
    """Sprite-style contact sheet cropped from the real sandbox scene."""
    from PIL import Image, ImageDraw
    frame = Image.open(image).convert("RGB")
    canvas = Image.new("RGB", (4 * 240, 4 * 170), "#20252d")
    draw = ImageDraw.Draw(canvas)
    variants = scene(OUT / label / "variety" / "scene.csv", "variety")
    for index, (x, y, variant) in enumerate(variants):
        column, row = index % 4, index // 4
        # Same 1280x800 camera and tile anchors used by sandbox reference_x64.
        center_x = 64 * (x - 32) + 640
        center_y = 32 * (y - 32) + 368
        crop = frame.crop((center_x - 120, center_y - 70,
                           center_x + 120, center_y + 70))
        canvas.paste(crop, (column * 240, row * 170 + 30))
        draw.text((column * 240 + 10, row * 170 + 9),
                  f"Art {variant + 1} | tile {x},{y}", fill="white")
    canvas.save(OUT / label / "variety-atlas.png")


def finish_sheets() -> None:
    images: dict[str, dict[str, Path]] = {}
    for label in SHAPES:
        paths = {kind: OUT / label / kind / "preview.png" for kind in ("variety", "connected")}
        if all(path.is_file() for path in paths.values()):
            images[label] = paths
            atlas(label, paths["variety"])
    if len(images) > 1:
        sheet(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    parser.add_argument("--render-only", action="store_true",
                        help="Render saved isolated DLLs without recompiling")
    parser.add_argument("--sheets-only", action="store_true",
                        help="Rebuild review sheets from existing sandbox captures")
    parser.add_argument("--biq-examples", action="store_true",
                        help="Capture realistic test.biq mountain areas with saved DLLs")
    args = parser.parse_args()
    if args.biq_examples:
        biq_examples()
        return
    if args.sheets_only:
        finish_sheets()
        return
    copy_sources()
    if not BASELINE.is_file():
        raise RuntimeError("The isolated sandbox baseline mesh is missing")
    original = BASELINE.read_text()
    # The first render uses exactly the sandbox sources copied at setup.
    images: dict[str, dict[str, Path]] = {}
    for label in args.shapes:
        dll = PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll"
        destination = OUT / label / "C3XReference_x64.dll"
        if args.render_only:
            if not destination.is_file():
                raise RuntimeError(f"No saved isolated DLL for {label}")
            if not (OUT / label / "build.txt").is_file():
                height, width = SHAPES[label]
                if (PRIVATE / MESH).read_text() != shaped_mesh(original, height, width):
                    raise RuntimeError(f"No verified mesh source for saved {label} DLL")
                (OUT / label / "build.txt").write_text(
                    f"dll_sha256={digest(destination)}\nmesh_sha256={digest(PRIVATE / MESH)}\n"
                    f"height_scale={height}\nwidth_scale={width}\n")
            shutil.copy2(destination, dll)
        else:
            height, width = SHAPES[label]
            (PRIVATE / MESH).write_text(shaped_mesh(original, height, width))
            build()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dll, destination)
            (OUT / label / "build.txt").write_text(
                f"dll_sha256={digest(destination)}\nmesh_sha256={digest(PRIVATE / MESH)}\n"
                f"height_scale={height}\nwidth_scale={width}\n")
        images[label] = {}
        for kind in ("variety", "connected"):
            existing = OUT / label / kind / "preview.png"
            images[label][kind] = existing if args.render_only and existing.is_file() else render(label, kind)
    finish_sheets()
    print(f"Lab comparisons: {OUT / 'comparison-variety.png'} and {OUT / 'comparison-connected.png'}")


if __name__ == "__main__":
    main()
