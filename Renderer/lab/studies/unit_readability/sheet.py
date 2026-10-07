#!/usr/bin/env python3
"""Render unit sheets through the live Renderer64 unit path and compare them.

    python3 Renderer/lab/studies/unit_readability/sheet.py build
    python3 Renderer/lab/studies/unit_readability/sheet.py render LABEL [--pack P] [--env K=V ...] [--owner HEX]
    python3 Renderer/lab/studies/unit_readability/sheet.py compose OUT.png LABEL [LABEL ...] [--civ3 6]

`build` compiles unit_sheet.cpp on the Windows VM (x64, Renderer64 flags).
`render` draws the roster (same grid as civ3_sheet.py) into the transparent
unit layer and stores LABEL.f16. `compose` places that layer over flat grass,
applies the production display transfer (native/scene_display.h, filmic 0.5)
and stacks labelled rows with the Civ III sheet. Outputs: lab/out/unit-readability.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path, PureWindowsPath

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from Renderer.lab.platform import native_command_result, windows_root  # noqa: E402
import civ3_sheet  # noqa: E402
from image_io import write_png  # noqa: E402

OUT = ROOT / "Renderer/lab/out/unit-readability"
BUILD = OUT / "build"
GRASS_DISPLAY = np.array([112, 136, 52], float) / 255
FILMIC = .5
# Civ III direction numbering; 5 faces south-west like the Civ III sheet.
DIRECTION = 5
SOURCES = ("terrain_scene_runtime.cpp", "environment_runtime.cpp", "terrain_definition_runtime.cpp",
           "scene_export.cpp", "frame_scheduler.cpp")


def win(path: Path) -> PureWindowsPath:
    return windows_root() / path.relative_to(ROOT).as_posix()


def vm(command: str, timeout=900):
    result = native_command_result("Renderer/native", command, timeout_seconds=timeout)
    if result.get("returncode") != 0:
        raise SystemExit(result.get("output_tail", "VM command failed"))
    return result.get("output_tail", "")


def build():
    BUILD.mkdir(parents=True, exist_ok=True)
    native = win(ROOT / "Renderer/native")
    sources = " ".join(f'"{native}\\{name}"' for name in SOURCES)
    (BUILD / "build.bat").write_text(r'''@echo off
setlocal
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VS=%%i"
if not defined VS exit /b 2
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 2
pushd "%~dp0"
''' + f'cl /nologo /std:c++17 /EHsc /O2 /bigobj /DC3X_HELPER_TRIAL /DC3X_RENDERER64_FRESH "{win(HERE / "unit_sheet.cpp")}" {sources} '
      '/Fo:.\\ /Fe:unit_sheet.exe /link d3d11.lib d3dcompiler.lib dxgi.lib dcomp.lib gdi32.lib msimg32.lib '
      'user32.lib bcrypt.lib ole32.lib windowscodecs.lib > build.log 2>&1\nexit /b %errorlevel%\n')
    vm(f'call "{win(BUILD / "build.bat")}"', timeout=1200)
    print("built", BUILD / "unit_sheet.exe")


def render(label, pack, env, owner, units, reflect=False):
    (OUT / f"{label}.units").write_text("".join(f"PRTO_{u} {DIRECTION}\n" for u in units))
    sets = " && ".join(f'set "{k}={v}"' for k, v in env.items())
    command = (f'{sets + " && " if sets else ""}"{win(BUILD / "unit_sheet.exe")}" "{win(ROOT / "Renderer/packs" / pack)}" '
               f'"{win(OUT / (label + ".units"))}" "{win(OUT / (label + ".f16"))}" {owner}{" reflect" if reflect else ""}')
    print(vm(command, timeout=600).strip().splitlines()[-1])


def display(rgb):
    """native/scene_display.h scene_display_srgb, vectorised."""
    x = np.maximum(rgb, 0)
    neutral = x / (1 + x.max(axis=-1, keepdims=True))
    f = np.minimum(x * .65, 64)
    f = np.clip((f * (2.51 * f + .03)) / (f * (2.43 * f + .59) + .14), 0, 1)
    v = neutral + (f - neutral) * FILMIC
    return np.where(v <= .0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0), 1 / 2.4) - .055)


def inverse_display(target, exposure, iterations=80):
    """Scene radiance (before exposure) whose display value is target."""
    x = np.full_like(target, .2)
    for _ in range(iterations):
        x = x * np.clip(target / np.maximum(display(x * exposure), 1e-5), .5, 2)
    return x


def load(label):
    data = (OUT / f"{label}.f16").read_bytes()
    magic, w, h = struct.unpack_from("<3I", data, 0)
    exposure, = struct.unpack_from("<f", data, 12)
    layer = np.frombuffer(data, "<f2", w * h * 4, 16).reshape(h, w, 4).astype(float)
    return layer, exposure


def owner_base(canvas, cx, cy, owner_rgb, exposure, strength=.45):
    """Mock-up only: a soft civ-coloured ellipse under a unit, in linear light."""
    rows, cols = canvas.shape[:2]
    ys, xs = np.mgrid[0:rows, 0:cols]
    d = np.sqrt(((xs + .5 - cx) / 30) ** 2 + ((ys + .5 - cy) / 15) ** 2)
    weight = (np.clip((1.15 - d) / .35, 0, 1) * strength)[..., None]
    colour = inverse_display(np.array(owner_rgb, float)[None, None, :] / 255, exposure)
    canvas[:] = canvas * (1 - weight) + colour * weight


def composite(label, base=None):
    layer, exposure = load(label)
    ground = inverse_display(GRASS_DISPLAY[None, None, :].copy(), exposure)
    rows, cols = layer.shape[:2]
    canvas = np.empty((rows, cols, 3))
    canvas[:] = inverse_display(np.array([[[88, 104, 44]]], float) / 255, exposure)
    ys, xs = np.mgrid[0:rows, 0:cols]
    for index in range(36):
        cx = (index % civ3_sheet.COLUMNS) * civ3_sheet.CELL_W + civ3_sheet.CELL_W // 2
        cy = (index // civ3_sheet.COLUMNS) * civ3_sheet.CELL_H + civ3_sheet.CELL_H // 2 + 25
        inside = np.abs(xs + .5 - cx) / 64 + np.abs(ys + .5 - cy) / 32 <= 1
        canvas[inside] = ground[0, 0]
        if base is not None and cy < rows:
            owner_base(canvas, cx, cy, base, exposure)
    rgb = layer[..., :3] + canvas * (1 - layer[..., 3:4])
    return (np.clip(display(rgb * exposure), 0, 1) * 255 + .5).astype(np.uint8), layer


def water(label, reflectance=None, shift=(4, 4)):
    """LABEL's unit layer over sea water, with its reflected pass.

    Mirrors water_natural.hlsl for deep water (coverage 1) seen along the
    shared view (.43,-.43,1) on a flat mean normal: the water radiance W, from
    an in-game ocean patch (OUT/ocean_patch.npy, display RGB), gains
    R*(mirror - sky*mirror_alpha), with R the Fresnel term from the noon
    environment and small ripple offsets. `reflectance` overrides R. The
    reflected pass draws with an 8 px guard band against the main pass's 4,
    hence the default `shift`."""
    import json
    layer, exposure = load(label)
    mirror, _ = load(label + ".reflect")
    env = json.loads((OUT / f"{label}.env.json").read_text())
    rows, cols = layer.shape[:2]
    patch = np.load(OUT / "ocean_patch.npy").astype(float) / 255
    tile = np.concatenate([patch, patch[::-1]], 0)
    tile = np.concatenate([tile, tile[:, ::-1]], 1)
    reps = (rows // tile.shape[0] + 1, cols // tile.shape[1] + 1, 1)
    canvas = inverse_display(np.tile(tile, reps)[:rows, :cols], exposure)
    view = np.array([.43, -.43, 1]) / np.linalg.norm([.43, -.43, 1])
    f0 = env["water_fresnel"]
    fresnel = f0 + (1 - f0) * (1 - view[2]) ** 5
    r = fresnel if reflectance is None else reflectance
    ray_y = view[1] * -1  # reflect(-view, +z).y
    band = np.clip((ray_y - .30) / .60, 0, 1); band = band * band * (3 - 2 * band)
    sky_light = .6 * (np.array(env["ambient"]) + np.array(env["sun"]) * env["sun_intensity"] + np.array(env["moon"]) * env["moon_intensity"])
    sky = sky_light * (np.array([.16, .25, .36]) + (np.array([.42, .55, .68]) - np.array([.16, .25, .36])) * band)
    ys, xs = np.mgrid[0:rows, 0:cols]
    # Ripple normals offset the mirror lookup by up to (3, 1.5) pixels.
    sx = np.clip(np.rint(xs + shift[0] + 1.6 * np.sin(ys * .9 + xs * .07)), 0, cols - 1).astype(int)
    sy = np.clip(np.rint(ys + shift[1] + .8 * np.sin(xs * .31 + ys * .2)), 0, rows - 1).astype(int)
    m = mirror[sy, sx]
    canvas = canvas + r * (m[..., :3] - sky * m[..., 3:4])
    rgb = layer[..., :3] + canvas * (1 - layer[..., 3:4])
    image = (np.clip(display(rgb * exposure), 0, 1) * 255 + .5).astype(np.uint8)
    return image, {"fresnel": float(fresnel), "reflectance": float(r), "sky": sky.tolist()}


def context(output, label, terrain, units, base=None):
    """Units from LABEL's sheet cells placed on a real terrain render."""
    from image_io import read_bmp
    layer, exposure = load(label)
    background = read_bmp(terrain).astype(float) / 255
    canvas = inverse_display(background, exposure)
    unit_layer = np.zeros(canvas.shape[:2] + (4,))
    for name, (x, y) in units.items():
        index = civ3_sheet.ROSTER.index(name)
        cx = (index % civ3_sheet.COLUMNS) * civ3_sheet.CELL_W + civ3_sheet.CELL_W // 2
        cy = (index // civ3_sheet.COLUMNS) * civ3_sheet.CELL_H + civ3_sheet.CELL_H // 2 + 25
        cell = layer[max(0, cy - 110):cy + 40, cx - 80:cx + 80]
        top, left = y - (cy - max(0, cy - 110)), x - 80
        h, w = cell.shape[:2]
        region = unit_layer[top:top + h, left:left + w]
        region[:] = cell[:region.shape[0], :region.shape[1]] + region * (1 - cell[:region.shape[0], :region.shape[1], 3:4])
        if base is not None:
            owner_base(canvas, x, y, base, exposure)
    rgb = unit_layer[..., :3] + canvas * (1 - unit_layer[..., 3:4])
    return (np.clip(display(rgb * exposure), 0, 1) * 255 + .5).astype(np.uint8)


def label_strip(text, width):
    """Tiny 5x7 caps label so sheets need no font library."""
    glyphs = {c: g for c, g in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-.+ ", [
        "01110100011000111111100011000110001", "11110100011111010001100011000111110", "01110100011000010000100001000101110",
        "11110100011000110001100011000111110", "11111100001111010000100001000011111", "11111100001111010000100001000010000",
        "01111100001000010111100011000101111", "10001100011111110001100011000110001", "01110001000010000100001000010001110",
        "00111000100001000010100101001001100", "10001100101010011000101001001010001", "10000100001000010000100001000011111",
        "10001110111010110001100011000110001", "10001110011010110011100011000110001", "01110100011000110001100011000101110",
        "11110100011111010000100001000010000", "01110100011000110001101011001001101", "11110100011111010100100101000110001",
        "01111100000111000001000011000111110", "11111001000010000100001000010000100", "10001100011000110001100011000101110",
        "10001100011000110001010100101000100", "10001100011000110101101011101110001", "10001010100010000100010101000110001",
        "10001010100010000100001000010000100", "11111000010001000100010001000011111", "01110100111010110111001011000101110",
        "00100011000010000100001000010001110", "01110100010000100110010001000011111", "11110000010111000001000011000111110",
        "00010001100101011111000100001000010", "11111100001111000001000011000111110", "01110100001111010001100011000101110",
        "11111000010001000100010000100001000", "01110100011000101110100011000101110", "01110100011000101111000010000101110",
        "00000000000000011111000000000000000", "00000000000000000000000000110001100", "00000001000010011111001000010000000",
        "00000000000000000000000000000000000"])}
    strip = np.zeros((14, width, 3), np.uint8)
    strip[:] = (30, 34, 30)
    x = 6
    for char in text.upper():
        bits = glyphs.get(char, glyphs[" "])
        for i, bit in enumerate(bits):
            if bit == "1":
                strip[3 + i // 5, x + i % 5] = (235, 235, 225)
        x += 6
        if x > width - 8:
            break
    return strip


def compose(output, labels, civ3, sea=False, reflectance=None):
    rows = []
    if civ3 is not None:
        import subprocess
        subprocess.run([sys.executable, str(HERE / "civ3_sheet.py"), "--civ", str(civ3)], check=True,
                       capture_output=True)
        from image_io import read_bmp  # noqa: F401  (kept for terrain backgrounds)
        civ = np.frombuffer(zlib_png_rgb(OUT / f"civ3-ntp{civ3:02d}.png"), np.uint8)
        rows += [label_strip(f"CIV III SPRITES NTP{civ3:02d}", 960), civ.reshape(-1, 960, 3)]
    for label in labels:
        if sea:
            image, info = water(label, reflectance)
            print(label, info)
        else:
            image, _ = composite(label)
        rows += [label_strip(label, image.shape[1]), image]
    sheet = np.concatenate(rows)
    write_png(OUT / output, sheet)
    print(OUT / output)


def zlib_png_rgb(path):
    """Decode the study's own 8-bit RGB PNGs (no filters other than None)."""
    import zlib
    data = path.read_bytes()
    pos, chunks, width = 8, b"", 0
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            width = struct.unpack(">I", body[:4])[0]
        if kind == b"IDAT":
            chunks += body
        pos += 12 + length
    raw = zlib.decompress(chunks)
    stride = width * 3 + 1
    return b"".join(raw[i * stride + 1:(i + 1) * stride] for i in range(len(raw) // stride))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    r = sub.add_parser("render")
    r.add_argument("label")
    r.add_argument("--pack", default="UnitAnimationFidelity")
    r.add_argument("--env", nargs="*", default=[])
    r.add_argument("--owner", default="3964fc")
    r.add_argument("--units", default=",".join(civ3_sheet.ROSTER))
    r.add_argument("--reflect", action="store_true")
    c = sub.add_parser("compose")
    c.add_argument("output")
    c.add_argument("labels", nargs="+")
    c.add_argument("--civ3", type=int)
    c.add_argument("--sea", action="store_true")
    c.add_argument("--reflectance", type=float)
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        build()
    elif args.command == "render":
        env = dict(item.split("=", 1) for item in args.env)
        render(args.label, args.pack, env, args.owner, args.units.split(","), args.reflect)
    else:
        compose(args.output, args.labels, args.civ3, args.sea, args.reflectance)


if __name__ == "__main__":
    main()
