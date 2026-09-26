"""Capture a test.biq city and its matching production ground mesh for Border Lab."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from Renderer.lab.platform import native_command_result
from Renderer.lab.studies.borders.mesh_surface import GroundSurface, RenderDepth
from Renderer.lab.studies.borders.preview import (
    biq_city_territory, compose, draped_paths, read_biq_terrain,
)
from Renderer.lab.studies.cities.build_layouts import ROOT


OUT = ROOT / "Renderer/lab/out/borders"
TERRAIN = ROOT / "Renderer/lab/out/cities/test-biq/terrain.csv"
DLL = ROOT / "Renderer/native/build/city-preview/C3XRenderer.dll"
PACK_ROOT = r"..\lab\out\cities\test-biq\root"
SITE = (20, 64)
PREFIX = OUT / "test-biq-ground"
EXAMPLES = (
    ("mountain-crossing", (1, 1, 2, 1), (196, 35, 50)),
    ("forest-crossing", (1, 1, 1, 2), (32, 110, 194)),
    ("mountain-and-forest", (1, 1, 2, 2), (32, 135, 115)),
)


def example_territory(bounds: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    """An invented, connected ownership rectangle on genuine BIQ land tiles."""
    west, east, south, north = bounds
    selected = {(SITE[0]+dc+dr, SITE[1]+dc-dr)
                for dc in range(-west, east+1) for dr in range(-south, north+1)}
    terrain = read_biq_terrain(TERRAIN)
    if not all(tile in terrain and terrain[tile][0] < 11 for tile in selected):
        raise ValueError("border example crosses a non-land BIQ tile")
    return selected


def capture() -> Path:
    if not DLL.is_file() or not TERRAIN.is_file():
        raise ValueError("build the isolated city Lab renderer and capture test.biq first")
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob(PREFIX.name + ".*_*.bin"):
        old.unlink()
    bitmap = OUT / "test-biq-ground.bmp"
    controls = {
        "C3X_RENDERER_VISUAL_PROFILE": "city-fidelity",
        "C3X_RENDERER_PREVIEW_OBJECTS": "1",
        "C3X_RENDERER_PREVIEW_CITY_ONLY": "1",
        "C3X_RENDERER_PREVIEW_CITY": "4,0,1,1,1",
        "C3X_RENDERER_PREVIEW_CITY_SITE": "20,64",
        "C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS":
            PACK_ROOT + r"\Renderer\custom.custom_rendering.txt",
        "C3X_RENDERER_BORDER_MESH_PREFIX": r"..\lab\out\borders\test-biq-ground",
        "C3X_RENDERER_BORDER_MESH_SITE": "20,64",
        "C3X_RENDERER_SHARED_SCENE_SURFACE": "0",
    }
    command = " && ".join(f'set "{key}={value}"' for key, value in controls.items())
    command += (r' && build\city-preview\biq_preview.exe '
                r'build\city-preview\C3XRenderer.dll '
                f'"{PACK_ROOT}" '
                r'..\default.custom_rendering.txt '
                r'..\lab\out\cities\test-biq\terrain.csv '
                r'..\lab\out\borders\test-biq-ground.bmp '
                '1280 800 20 64 256 12')
    result = native_command_result("Renderer/native", command, timeout_seconds=300)
    if result["status"] != "pass" or "0 fallback" not in result["output_tail"] or not bitmap.is_file():
        raise ValueError("test.biq native city and ground mesh capture failed")
    surface = GroundSurface(PREFIX, 256)
    depth = RenderDepth(PREFIX, (1280, 800))
    with Image.open(bitmap) as background:
        owned = biq_city_territory(SITE, TERRAIN)
        result_image = compose(background, 256, "crimson-brush", owned=owned,
                               center=SITE, surface=surface)
        previews = [("Original hill", result_image)]
        for name, bounds, color in EXAMPLES:
            owned = example_territory(bounds)
            sample_depths = []
            paths = draped_paths(owned, background.size, 256, SITE, surface, sample_depths)
            occluded = sum(depth.is_occluded(x, y, value)
                           for path, values in zip(paths, sample_depths)
                           for (x, y), value in zip(path, values))
            if occluded < 20:
                raise ValueError(f"{name} no longer demonstrates a visible occlusion")
            example = compose(background, 256, "crimson-brush", color,
                              owned, SITE, surface)
            example.save(OUT / f"test-biq-{name}.png")
            if name == "forest-crossing":
                example.crop((335, 405, 785, 620)).resize(
                    (900, 430), Image.Resampling.LANCZOS).save(
                        OUT / "test-biq-inward-detail.png")
            if name == "mountain-and-forest":
                for label, box in (
                    ("clear-north", (545, 175, 690, 260)),
                    ("clear-south", (355, 570, 550, 695)),
                ):
                    example.crop(box).resize(
                        ((box[2]-box[0])*3, (box[3]-box[1])*3),
                        Image.Resampling.LANCZOS).save(
                            OUT / f"test-biq-{label}-detail.png")
            previews.append((name.replace("-", " ").title(), example))
    target = OUT / "test-biq-city.png"
    result_image.save(target)
    sheet = Image.new("RGB", (1280, 864), (32, 35, 35))
    labels = ImageDraw.Draw(sheet)
    for index, (label, preview) in enumerate(previews):
        left, top = (index % 2) * 640, (index // 2) * 432
        labels.text((left+14, top+8), label, fill=(245, 245, 245))
        sheet.paste(preview.resize((640, 400), Image.Resampling.LANCZOS),
                    (left, top+32))
    sheet.save(OUT / "test-biq-occlusion-examples.png")
    return target


if __name__ == "__main__":
    print(capture())
