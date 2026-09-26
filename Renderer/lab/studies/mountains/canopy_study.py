#!/usr/bin/env python3
"""Render Lab-only forest/jungle mountain art with the lower mountain shape.

Civ III picks its mountain forest/jungle PCX from diagonal terrain neighbors.
The isolated flat-grassland sheets additionally show an authored canopy ring
on the mountain tile, encoded only in this study's CSV and private client.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab import platform
import shape_study

OUT = ROOT / "Renderer/lab/out/mountains/canopy-study"
PRIVATE = OUT / "root"
SHAPE_OUT = ROOT / "Renderer/lab/out/mountains/shape-study"
MARKERS = {"forest": 0xF0000001, "jungle": 0xF0000002}
TYPES = {"forest": 7, "jungle": 8}
BIQ_VIEWS = {"forest-west": (85, 25), "forest-range": (68, 54),
             "jungle-east": (84, 50), "jungle-west": (35, 51)}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one Lab source expression, got {count}: {old[:72]}")
    return source.replace(old, new)


def copy_and_patch() -> None:
    if PRIVATE.exists():
        if not (OUT / "inputs.txt").is_file():
            raise RuntimeError("Partial canopy study root; preserve its inputs")
        return
    shape_study.copy_sources()
    (PRIVATE / "Renderer").mkdir(parents=True)
    for directory in ("native", "sandbox", "lab/shared"):
        shutil.copytree(SHAPE_OUT / "root/Renderer" / directory,
                        PRIVATE / "Renderer" / directory,
                        ignore=shutil.ignore_patterns("out", "build", "*.obj", "*.pdb", "*.ilk"))
    for name in ("default.custom_rendering.txt", "custom.custom_rendering.txt"):
        shutil.copy2(SHAPE_OUT / "root/Renderer" / name, PRIVATE / "Renderer" / name)
    shutil.copytree(ROOT / "Renderer/packs", PRIVATE / "Renderer/packs",
                    copy_function=os.link)
    mesh = PRIVATE / shape_study.MESH
    mesh.write_text(shape_study.shaped_mesh(shape_study.BASELINE.read_text(), 0.68, 1.08))

    native = PRIVATE / "Renderer/native/c3x_renderer.cpp"
    original = native.read_text()
    source = replace_once(original,
        "(tile.real_terrain_type == 7 || tile.real_terrain_type == 8);",
        "(tile.feature_flags & (C3X_RENDERER_FEATURE_FOREST | C3X_RENDERER_FEATURE_JUNGLE)) != 0;")
    start = source.index("            if (feature_assets_ready &&\n"
                         "                (tile.real_terrain_type == 7 || tile.real_terrain_type == 8) &&")
    end = source.index("            if (river_rock_group != nullptr", start)
    section = source[start:end]
    section = section.replace("tile.real_terrain_type == 7 || tile.real_terrain_type == 8",
                              "canopy_kind == 7 || canopy_kind == 8")
    section = section.replace("tile.real_terrain_type", "canopy_kind")
    section = replace_once(section,
        "!(fidelity_profile && canopy_kind == 7)",
        "!(fidelity_profile && canopy_kind == 7 && tile.real_terrain_type != 6)")
    section = "            int canopy_kind = (tile.feature_flags & C3X_RENDERER_FEATURE_FOREST) ? 7 :\n" \
              "                (tile.feature_flags & C3X_RENDERER_FEATURE_JUNGLE) ? 8 : -1;\n" + section
    section = replace_once(section,
        "                        unsigned row = instance / grid_side;",
        "                        unsigned row = instance / grid_side;\n"
        "                        // Isolated rocky summit stays exposed; trees occupy the lower rim.\n"
        "                        if (tile.real_terrain_type == 6 &&\n"
        "                            row > 0 && row + 1 < grid_side &&\n"
        "                            column > 0 && column + 1 < grid_side) continue;")
    section = replace_once(section,
        "float scene_feature_scale = canopy_kind == 7 ? 0.42f : 0.40f;",
        "float scene_feature_scale = tile.real_terrain_type == 6 ?\n"
        "                            (canopy_kind == 7 ? 0.34f : 0.32f) :\n"
        "                            (canopy_kind == 7 ? 0.42f : 0.40f);")
    source = source[:start] + section + source[end:]
    native.write_text(source)

    client = PRIVATE / "Renderer/sandbox/reference_x64.cpp"
    original_client = client.read_text()
    client.write_text(replace_once(original_client,
        "        if (source.real == 7) tile.feature_flags = C3X_RENDERER_FEATURE_FOREST;",
        "        if (source.real == 6 && source.overlays == 0xF0000001u) {\n"
        "            tile.feature_flags = C3X_RENDERER_FEATURE_FOREST; tile.terrain_overlays = 0;\n"
        "        }\n"
        "        if (source.real == 6 && source.overlays == 0xF0000002u) {\n"
        "            tile.feature_flags = C3X_RENDERER_FEATURE_JUNGLE; tile.terrain_overlays = 0;\n"
        "        }\n"
        "        if (source.real == 7) tile.feature_flags = C3X_RENDERER_FEATURE_FOREST;"))
    (OUT / "inputs.txt").write_text(
        f"lower_mesh_sha256={digest(mesh)}\n"
        f"native_original_sha256={hashlib.sha256(original.encode()).hexdigest()}\n"
        f"client_original_sha256={hashlib.sha256(original_client.encode()).hexdigest()}\n"
        "Lab markers are authored only in study CSV scenes, not test.biq.\n")


def windows(path: Path) -> str:
    return shape_study.windows(path)


def build() -> None:
    result = platform.native_command_result(
        "Renderer/lab/out/mountains/canopy-study/root/Renderer/sandbox",
        "call build_reference_x64.bat", timeout_seconds=600)
    if result["status"] != "pass":
        raise RuntimeError("Isolated canopy renderer build failed:\n" + result["output_tail"])
    dll = PRIVATE / "Renderer/sandbox/out/C3XReference_x64.dll"
    if not dll.is_file():
        raise RuntimeError("No isolated canopy DLL")
    (OUT / "build.txt").write_text(f"dll_sha256={digest(dll)}\n")


def grassland_scene(kind: str, path: Path) -> list[tuple[int, int]]:
    size = 64
    points = [(x, y) for y in (26, 30, 34, 38) for x in (26, 30, 34, 38)]
    selected = set(points)
    rows = [f"{x},{y},2,{6 if (x, y) in selected else 2},0,"
            f"{MARKERS[kind] if (x, y) in selected else 0},0"
            for y in range(size) for x in range(y % 2, size, 2)]
    path.write_text(f"C3X_BIQ_TERRAIN_V3,{size},{size},{len(rows)}\n" +
                    "\n".join(rows) + "\n")
    return points


def biq_scenes() -> dict[str, Path]:
    source_biq = ROOT / "Renderer/packs/RendererSourceStudies/maps/test.biq"
    source_csv = OUT / "test-biq-original.csv"
    subprocess.run(["node", str(ROOT / "Renderer/sandbox/export_biq.js"),
                    str(source_biq), str(source_csv)], cwd=ROOT, check=True)
    with source_csv.open(newline="") as stream:
        header = next(stream).strip()
        rows = list(csv.reader(stream))
    lookup = {(int(row[0]), int(row[1])): int(row[3]) for row in rows}
    scenes = {}
    counts = {}
    for kind, terrain in TYPES.items():
        derived = [row.copy() for row in rows]
        selected = []
        for row in derived:
            x, y = map(int, row[:2])
            if int(row[3]) != 6:
                continue
            neighbors = [lookup.get((x + dx, y + dy))
                         for dx in (-1, 1) for dy in (-1, 1)]
            if neighbors == [terrain] * 4:
                row[5] = str(MARKERS[kind])
                selected.append((x, y))
        if not selected:
            raise RuntimeError(f"No surrounded mountain in test.biq for {kind}")
        path = OUT / f"test-biq-{kind}-lab.csv"
        path.write_text(header + "\n" + "\n".join(",".join(row) for row in derived) + "\n")
        scenes[kind] = path
        counts[kind] = selected
    (OUT / "biq-derivation.txt").write_text(
        f"source_biq_sha256={digest(source_biq)}\n"
        f"source_csv_sha256={digest(source_csv)}\n"
        "Mountain canopy is Lab-authored on terrain-6 tiles whose four diagonal neighbors are forest or jungle.\n" +
        "".join(f"{kind}_raw_tiles={counts[kind]}\n" for kind in TYPES))
    return scenes


def render(scene: Path, output: Path, center: tuple[int, int], size: tuple[int, int]) -> Path:
    output.mkdir(parents=True, exist_ok=True)
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
        "set \"C3X_SANDBOX_WHOLE_WORLD=1\"\n"
        "set \"C3X_RENDERER_PREVIEW_UNITS=1\"\n"
        "set \"C3X_SANDBOX_UNITS=1\"\n"
        "set \"C3X_SANDBOX_REPLAY_CLIP=1\"\n"
        "set \"C3X_SANDBOX_CLIP_FRAMES=1\"\n"
        f"set \"C3X_SANDBOX_CAPTURE={windows(capture)}\"\n"
        f"set \"C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS={windows(PRIVATE / 'Renderer/custom.custom_rendering.txt')}\"\n"
        f'"{windows(exe)}" "{windows(dll)}" "{windows(PRIVATE)}" '
        f'"{windows(PRIVATE / "Renderer/default.custom_rendering.txt")}" '
        f'"{windows(scene)}" "{windows(output / "unused.bmp")}" '
        f"{size[0]} {size[1]} {center[0]} {center[1]} 128 12\n"
        "exit /b %errorlevel%\n")
    result = platform.native_command_result("Renderer/native",
        f'call "{windows(invocation)}"', timeout_seconds=600)
    if (result["status"] != "pass" or "fallback=0" not in result["output_tail"] or
            not capture.is_file() or capture.stat().st_size < 100000):
        raise RuntimeError(f"Lab capture failed at {center}:\n{result['output_tail']}")
    from PIL import Image
    image = output / "preview.png"
    Image.open(capture).convert("RGB").save(image)
    (output / "capture.txt").write_text(
        f"scene_sha256={digest(scene)}\ndll_sha256={digest(dll)}\n"
        f"camera={center[0]},{center[1]}\nfallback=0\n")
    return image


def atlas(kind: str, frame_path: Path, points: list[tuple[int, int]]) -> Path:
    from PIL import Image, ImageDraw
    frame = Image.open(frame_path).convert("RGB")
    canvas = Image.new("RGB", (960, 680), "#20252d")
    draw = ImageDraw.Draw(canvas)
    for index, (x, y) in enumerate(points):
        column, row = index % 4, index // 4
        center_x = 64 * (x - 32) + 640
        center_y = 32 * (y - 32) + 368
        crop = frame.crop((center_x - 120, center_y - 70,
                           center_x + 120, center_y + 70))
        canvas.paste(crop, (column * 240, row * 170 + 30))
        variant = ((x * 0x193) ^ (y * 0x217) ^ 0x6b91) % 5 + 1
        draw.text((column * 240 + 10, row * 170 + 9),
                  f"Art {variant} | tile {x},{y}", fill="white")
    path = OUT / f"{kind}-grassland-sheet.png"
    canvas.save(path)
    return path


def context_sheet() -> Path:
    from PIL import Image, ImageDraw
    canvas = Image.new("RGB", (1520, 1010), "#20252d")
    draw = ImageDraw.Draw(canvas)
    for index, (name, center) in enumerate(BIQ_VIEWS.items()):
        kind = name.split("-", 1)[0]
        path = OUT / kind / "biq" / name / "preview.png"
        frame = Image.open(path).convert("RGB")
        crop = frame.crop((420, 210, 1180, 690))
        x, y = (index % 2) * 760, (index // 2) * 505
        canvas.paste(crop, (x, y + 25))
        draw.text((x + 12, y + 7),
                  f"{kind} | test.biq tile {center[0]},{center[1]}", fill="white")
    path = OUT / "test-biq-canopy-contexts.png"
    canvas.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("all", "forest", "jungle"), default="all")
    parser.add_argument("--sheets-only", action="store_true",
                        help="Rebuild the context sheet from existing captures")
    args = parser.parse_args()
    if args.sheets_only:
        print(context_sheet())
        return
    copy_and_patch()
    build()
    for kind in TYPES:
        if args.kind not in ("all", kind):
            continue
        scene = OUT / f"{kind}-grassland.csv"
        points = grassland_scene(kind, scene)
        frame = render(scene, OUT / kind / "grassland", (32, 32), (1280, 800))
        print(atlas(kind, frame, points))
    biq = biq_scenes()
    for name, center in BIQ_VIEWS.items():
        kind = name.split("-", 1)[0]
        if args.kind not in ("all", kind):
            continue
        print(render(biq[kind], OUT / kind / "biq" / name, center, (1600, 900)))
    if args.kind == "all":
        print(context_sheet())


if __name__ == "__main__":
    main()
