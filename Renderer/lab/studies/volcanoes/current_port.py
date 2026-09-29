#!/usr/bin/env python3
"""Capture the current native volcano forms on isolated grassland tiles.

The fixture uses the production coordinate selector. It does not force a
particular variant into a tile or change the native preview's terrain rules.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer import renderer

OUT = ROOT / "Renderer/lab/out/volcanoes/current-port"
DLL = ROOT / "Renderer/native/build/candidate/C3XRenderer.dll"
PREVIEW = ROOT / "Renderer/lab/.cache/native_preview.exe"
FROZEN_DLL = OUT / "candidate/C3XRenderer.dll"
FAMILIES = (
    "Current crater cone", "Smooth cone", "Broad crater cone",
    "Offset steep cone", "Broken ridge", "Breached rim",
    "Paired shoulders", "Eroded cone",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hash32(value: int) -> int:
    value &= 0xffffffff
    value ^= value >> 16
    value = value * 0x7feb352d & 0xffffffff
    value ^= value >> 15
    value = value * 0x846ca68b & 0xffffffff
    return value ^ (value >> 16)


def slot(x: int, y: int) -> int:
    return hash32(x * 73856093 ^ y * 19349663) & 15


def tile_for_slot(index: int) -> tuple[int, int]:
    options = ((abs(x - 16) + abs(y - 16), x, y)
               for y in range(8, 25) for x in range(y % 2, 32, 2)
               if slot(x, y) == index)
    _, x, y = min(options)
    return x, y


def scene(destination: Path, center: tuple[int, int], kind: str) -> None:
    x0, y0 = center
    around = {(x0 + dx, y0 + dy) for dx in (-1, 1) for dy in (-1, 1)}
    neighbor = 7 if kind == "forest" else 8 if kind == "jungle" else 2
    rows = []
    for y in range(32):
        for x in range(y % 2, 32, 2):
            real = 10 if (x, y) == center else neighbor if (x, y) in around else 2
            rows.append(f"{x},{y},2,{real},0,0,0")
    destination.write_text(
        f"C3X_BIQ_TERRAIN_V3,32,32,{len(rows)}\n" + "\n".join(rows) + "\n")


def capture(index: int, kind: str, dll_hash: str, preview_hash: str,
            *, case: str = "detail") -> Path:
    center = tile_for_slot(index)
    target = OUT / (f"{index:02d}-{kind}" if case == "detail" else f"{index:02d}-{case}")
    target.mkdir(parents=True, exist_ok=True)
    original = renderer.scene
    try:
        renderer.scene = lambda _category, _case, destination, **_: scene(destination, center, kind)
        result = renderer.native_render("volcanoes", case, 12, 224, target,
                                        center=center, candidate=FROZEN_DLL, preview=PREVIEW)
    finally:
        renderer.scene = original
    bitmap = ROOT / result["image"]
    picture = target / "preview.png"
    Image.open(bitmap).convert("RGB").save(picture, optimize=True)
    bitmap.unlink()
    (target / "capture.json").write_text(json.dumps({
        "slot": index, "family": FAMILIES[index // 2], "kind": kind,
        "case": case,
        "tile": list(center), "dll_sha256": dll_hash,
        "preview_sha256": preview_hash, "scene_sha256": digest(target / "scene.csv"),
        "image_sha256": digest(picture), "fallback": 0,
    }, indent=2) + "\n")
    return picture


def sheet(images: list[Path]) -> Path:
    canvas = Image.new("RGB", (1280, 960), "#20252d")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    for index, picture in enumerate(images):
        frame = Image.open(picture).convert("RGB")
        x, y = index % 4 * 320, index // 4 * 240
        canvas.paste(frame.crop((160, 100, 480, 320)), (x, y + 20))
        draw.text((x + 7, y + 2), f"{index:02d} {FAMILIES[index // 2]}",
                  fill="white", font=font)
    output = OUT / "grassland-16-current.png"
    canvas.save(output, optimize=True)
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before, preview_hash = digest(DLL), digest(PREVIEW)
    FROZEN_DLL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(DLL, FROZEN_DLL)
    if digest(DLL) != before or digest(FROZEN_DLL) != before:
        raise ValueError("Native binary changed while creating the Lab snapshot")
    images = [capture(index, "bare", before, preview_hash) for index in range(16)]
    for kind in ("forest", "jungle"):
        capture(5, kind, before, preview_hash)
    active = capture(5, "bare", before, preview_hash, case="active")
    if Image.open(active).convert("RGB").tobytes() != \
            Image.open(images[5]).convert("RGB").tobytes():
        raise ValueError("Active volcano introduced an unwanted material change")
    if digest(FROZEN_DLL) != before or digest(PREVIEW) != preview_hash:
        raise ValueError("Native Lab binary changed during the captures")
    print(sheet(images))


if __name__ == "__main__":
    main()
