#!/usr/bin/env python3
"""Build the source-independent mine bundle: one chosen building for every era.

The strategy's mine "runtime_building" names the Civ VI component, its scale
and the facing that turns its Civ VI front toward the C3X camera. The runtime
selects group mine_<family*3+variant>; all six groups hold the same building.
The bundle binds six base and two emissive textures; unused slots share a 4x4
placeholder. Its emissive textures are night light baked by uv0 for only the
faces Civ VI lights (bake_light).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Renderer.preview.render_improvement_sheet import (
    IDENTITY,
    _decal_documents,
    _decal_mesh,
    _load_json,
    _matrix_multiply,
    _skeleton_worlds,
    _transform_mesh,
)


MAGIC = b"C3XVEG1\0"


def bundle_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def collect_parts(root: Path, manifest: dict, asset_id: str, transform=None, stack=()):
    if asset_id in stack or len(stack) >= 12:
        raise ValueError("mine component graph cycles or is too deep")
    transform = IDENTITY if transform is None else transform
    landmark = _load_json(root / manifest["assets"][asset_id]["landmark"])
    parts = []
    for binding in landmark["draw_bindings"]:
        if "worked" not in binding["states"]:
            continue
        mesh = _load_json(root / landmark["components"]["geometry"][binding["geometry"]])
        material = _load_json(root / landmark["components"]["materials"][binding["material"]])
        channels = material.get("channels", {})
        base = channels.get("base_color")
        if base:
            parts.append((_transform_mesh(mesh, transform), base["texture"],
                          channels.get("emissive", {}).get("texture")))
    decal_path = landmark["components"].get("decal")
    if decal_path:
        for decal in _decal_documents(root, decal_path):
            base = decal.get("channels", {}).get("base_color")
            if base:
                parts.append((_decal_mesh(decal, transform), base["texture"], None))
    skeleton_worlds = [
        _skeleton_worlds(_load_json(root / path))
        for path in landmark["components"]["skeletons"]
    ]
    for point in landmark["attachment_points"]:
        if point["binding_status"] != "resolved":
            continue
        child_transform = _matrix_multiply(
            skeleton_worlds[point["skeleton"]][point["bone"]], transform)
        parts.extend(collect_parts(root, manifest, point["component_asset"],
                                   child_transform, stack + (asset_id,)))
    return parts


def merged_asset(asset_id: str, texture_index: int, emissive_code: int, meshes: list[dict]) -> bytes:
    vertices = []
    indices = []
    for mesh in meshes:
        first = len(vertices)
        vertices.extend(mesh["vertices"])
        indices.extend(first + index for index in mesh["topology"]["indices"])
    payload = bytearray(bundle_string(f"{asset_id}:e{emissive_code}"))
    payload.extend(struct.pack("<III", texture_index, len(vertices), len(indices)))
    for vertex in vertices:
        payload.extend(struct.pack("<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"])))
    payload.extend(struct.pack(f"<{len(indices)}I", *indices))
    return bytes(payload)


def group_payload(name: str, placements: list[tuple[int, float]], scale: float = 1.18) -> bytes:
    payload = bytearray(bundle_string(name))
    payload.extend(struct.pack("<I", len(placements)))
    for asset_index, radius in placements:
        # Largest merged part sorts first so the renderer uses it for the one
        # compound footprint shadow while discarding redundant child shadows.
        payload.extend(struct.pack("<IffIIIIff", asset_index, scale, 0.04, 1, 1, 5, 0,
                                   radius, 0.0))
    return bytes(payload)


STRATEGY = Path(__file__).with_name("improvement_render_strategy.json")
PLACEHOLDER = "textures/unused_bc1.dds"
BASE_SLOTS, EMISSIVE_SLOTS = 6, 2
# A triangle whose mean light is below this (BC1 noise and the soft halo
# around a glow block) stays dark.
LIGHT_FLOOR = 8 / 255


def bc1_header(size: int, mips: int) -> bytes:
    """A square BC1 sRGB DX10 DDS header."""
    header = struct.pack("<7I44s32s5I", 124, 0xA1007, size, size, max(1, size // 4) ** 2 * 8, 0, mips,
                         bytes(44), struct.pack("<II4s5I", 32, 0x4, b"DX10", 0, 0, 0, 0, 0),
                         0x401008, 0, 0, 0, 0)
    return b"DDS " + header + struct.pack("<5I", 72, 3, 0, 1, 0)


def bc1_rgb(path: Path) -> np.ndarray:
    """The top mip of a BC1 DX10 DDS as stored (sRGB) values 0-1."""
    data = path.read_bytes()
    if data[:4] != b"DDS " or data[84:88] != b"DX10" or struct.unpack_from("<I", data, 128)[0] not in (71, 72):
        raise ValueError(f"BC1 DX10 texture required: {path.name}")
    height, width = struct.unpack_from("<II", data, 12)
    bw, bh = max(1, width // 4), max(1, height // 4)
    blocks = np.frombuffer(data, np.uint8, bw * bh * 8, 148).reshape(bh, bw, 8)
    ends = np.ascontiguousarray(blocks[..., :4]).view("<u2").astype(np.int32)
    rgb = lambda v: np.stack([(v >> 11 & 31) / 31, (v >> 5 & 63) / 63, (v & 31) / 31], -1)
    c0, c1 = rgb(ends[..., 0]), rgb(ends[..., 1])
    four = (ends[..., 0] > ends[..., 1])[..., None]
    palette = np.stack([c0, c1, np.where(four, (2 * c0 + c1) / 3, (c0 + c1) / 2),
                        np.where(four, (c0 + 2 * c1) / 3, 0)], 2)
    bits = np.ascontiguousarray(blocks[..., 4:]).view("<u4")[..., 0]
    index = (bits[..., None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    texels = np.take_along_axis(palette, index[..., None].astype(np.int64), 2)
    return texels.reshape(bh, bw, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)


def bc1_dds(image: np.ndarray) -> bytes:
    """A square RGB image (0-1, power-of-two side) as BC1 sRGB with a box-filtered
    mip chain; each block's endpoints are its brightest and darkest texels."""
    top, levels = image.shape[0], []
    while True:
        side = image.shape[0]
        blocks = image.reshape(side // 4, 4, side // 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 3)
        luma = blocks @ np.array([.299, .587, .114])
        q = lambda c: ((np.rint(c[:, 0] * 31).astype(np.int32) << 11) | (np.rint(c[:, 1] * 63).astype(np.int32) << 5)
                       | np.rint(c[:, 2] * 31).astype(np.int32))
        pick = lambda at: blocks[np.arange(len(blocks)), at]
        c0, c1 = q(pick(luma.argmax(1))), q(pick(luma.argmin(1)))
        c0, c1 = np.maximum(c0, c1), np.minimum(c0, c1)
        rgb = lambda v: np.stack([(v >> 11 & 31) / 31, (v >> 5 & 63) / 63, (v & 31) / 31], -1)
        e0, e1 = rgb(c0), rgb(c1)
        palette = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 1)
        index = ((blocks[:, :, None] - palette[:, None]) ** 2).sum(-1).argmin(-1)
        index = np.where((c0 == c1)[:, None], 0, index).astype(np.uint32)
        out = np.zeros((len(blocks), 8), np.uint8)
        out[:, 0:2] = c0.astype("<u2")[:, None].view(np.uint8)
        out[:, 2:4] = c1.astype("<u2")[:, None].view(np.uint8)
        out[:, 4:8] = (index << (2 * np.arange(16, dtype=np.uint32))).sum(1).astype("<u4")[:, None].view(np.uint8)
        levels.append(out.tobytes())
        if side <= 4:
            break
        image = image.reshape(side // 2, 2, side // 2, 2, 3).mean((1, 3))
    return bc1_header(top, len(levels)) + b"".join(levels)


def bake_light(meshes: list[dict], emissive: np.ndarray) -> tuple[list[dict], list[dict], np.ndarray]:
    """Split meshes into dark and lit triangles and bake the lit ones' light.

    Civ VI lays an emissive map out by uv2, not by the base colour's uv0 (an
    atlas that many faces share), so sampling it by uv0 lit planks and beams
    all over the building. A triangle is lit when its mean light, sampled by
    uv2, reaches LIGHT_FLOOR. The returned image (the emissive map's size) holds
    that light laid out by the lit triangles' uv0, spread a little past their
    edges so mips keep it. A mesh without uv2 stays dark (the material's
    non-emissive missing policy)."""
    size = emissive.shape[0]
    light = np.zeros_like(emissive)
    filled = np.zeros(emissive.shape[:2], bool)
    weights = [(a, b) for a in np.linspace(1 / 12, 11 / 12, 6) for b in np.linspace(1 / 12, 11 / 12, 6) if a + b < 1]

    def texel(uv: np.ndarray) -> np.ndarray:
        x = (np.mod(uv[..., 0], 1) * size).astype(int) % size
        y = (np.mod(uv[..., 1], 1) * size).astype(int) % size
        return emissive[y, x]

    dark, lit = [], []
    for mesh in meshes:
        vertices = mesh["vertices"]
        triangles = np.array(mesh["topology"]["indices"]).reshape(-1, 3)
        keep = {False: [], True: []}
        for triangle in triangles:
            corners = [vertices[i] for i in triangle]
            glowing = all("uv2" in corner for corner in corners)
            if glowing:
                uv2 = np.array([corner["uv2"] for corner in corners])
                samples = np.array([uv2[0] * (1 - a - b) + uv2[1] * a + uv2[2] * b for a, b in weights])
                glowing = texel(samples).max(1).mean() >= LIGHT_FLOOR
            keep[bool(glowing)].append(triangle)
            if not glowing:
                continue
            uv0 = np.array([corner["uv0"] for corner in corners]) * size - .5
            if uv0.min() < -.5 or uv0.max() > size - .5:
                raise ValueError("A lit triangle's uv0 leaves the atlas; it cannot be baked")
            x0, y0 = np.floor(uv0.min(0)).astype(int).clip(0, size - 1)
            x1, y1 = np.ceil(uv0.max(0)).astype(int).clip(0, size - 1)
            ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            (ax, ay), (bx, by), (cx, cy) = uv0
            area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
            if abs(area) < 1e-9:
                continue
            wb = ((xs - ax) * (cy - ay) - (cx - ax) * (ys - ay)) / area
            wc = ((bx - ax) * (ys - ay) - (xs - ax) * (by - ay)) / area
            inside = (wb >= -1e-6) & (wc >= -1e-6) & (wb + wc <= 1 + 1e-6)
            wa = 1 - wb - wc
            uv = wa[..., None] * uv2[0] + wb[..., None] * uv2[1] + wc[..., None] * uv2[2]
            region = light[y0:y1 + 1, x0:x1 + 1]
            region[inside] = np.maximum(region[inside], texel(uv)[inside])
            filled[y0:y1 + 1, x0:x1 + 1] |= inside
        for glowing, chosen in keep.items():
            if not chosen:
                continue
            used = sorted({int(i) for triangle in chosen for i in triangle})
            remap = {old: new for new, old in enumerate(used)}
            part = {**mesh, "vertices": [vertices[i] for i in used],
                    "topology": {**mesh["topology"], "indices": [remap[int(i)] for t in chosen for i in t]}}
            (lit if glowing else dark).append(part)
    # Spread the light 16 texels past the lit faces: their small mips then
    # average light with light, not with the black around it.
    for _ in range(16):
        total = np.zeros_like(light)
        count = np.zeros(filled.shape)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            total += np.roll(light * filled[..., None], (dy, dx), (0, 1))
            count += np.roll(filled, (dy, dx), (0, 1))
        grow = ~filled & (count > 0)
        light[grow] = total[grow] / count[grow][:, None]
        filled |= grow
    return dark, lit, light


def build(pack: Path, strategy_path: Path = STRATEGY, target_name: str = "mine_runtime.bin") -> Path:
    from Renderer.tools.asset_compiler.improvement_asset_importer import _asset_id
    strategy = _load_json(strategy_path)["mine"]
    choice = strategy["runtime_building"]
    manifest = _load_json(pack / "manifest.json")
    asset_id = _asset_id(strategy["source_package"], choice["source_entry"])
    if asset_id not in manifest["assets"]:
        raise ValueError(f"{choice['source_entry']} is not in {pack.name}; run the improvement importer")
    landmark = _load_json(pack / manifest["assets"][asset_id]["landmark"])
    # The building's own worked draws only: no attached props or ground decals.
    facing = math.radians(choice["facing_degrees"])
    turn = [math.cos(facing), math.sin(facing), 0.0, 0.0, -math.sin(facing), math.cos(facing), 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    merged = defaultdict(list)
    for binding in landmark["draw_bindings"]:
        if "worked" not in binding["states"]:
            continue
        mesh = _load_json(pack / landmark["components"]["geometry"][binding["geometry"]])
        channels = _load_json(pack / landmark["components"]["materials"][binding["material"]])["channels"]
        merged[(channels["base_color"]["texture"], channels.get("emissive", {}).get("texture"))].append(
            _transform_mesh(mesh, turn))
    # Each emissive map's light is baked by uv0 for the faces it lights
    # (bake_light); the rest of those parts are dark.
    parts = defaultdict(list)
    for (base, emissive), meshes in sorted(merged.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        if not emissive:
            parts[(base, None)].extend(meshes)
            continue
        dark, lit, light = bake_light(meshes, bc1_rgb(pack / emissive))
        parts[(base, None)].extend(dark)
        if lit:
            data = bc1_dds(light)
            baked = f"textures/mine/light_{hashlib.sha256(data).hexdigest()[:16]}.dds"
            (pack / baked).parent.mkdir(parents=True, exist_ok=True)
            (pack / baked).write_bytes(data)
            parts[(base, baked)].extend(lit)
    bases = sorted({base for base, _ in parts})
    emissives = sorted({emissive for _, emissive in parts if emissive})
    if not parts or len(bases) > BASE_SLOTS or len(emissives) > EMISSIVE_SLOTS:
        raise ValueError("Mine building needs 1-6 base and at most 2 emissive textures")
    placeholder = pack / PLACEHOLDER
    if not placeholder.is_file():
        # A 4x4 BC1 sRGB DX10 DDS: unused slots load it instead of a real texture again.
        placeholder.write_bytes(bc1_header(4, 1) + bytes(8))
    textures = (bases + [PLACEHOLDER] * (BASE_SLOTS - len(bases)) +
                emissives + [PLACEHOLDER] * (EMISSIVE_SLOTS - len(emissives)))
    assets, ranked = [], []
    for (base, emissive), meshes in sorted(parts.items(), key=lambda item: (item[0][0], item[0][1] or "")):
        code = emissives.index(emissive) + 1 if emissive else 0
        radius = max(math.hypot(*vertex["position"][:2]) for mesh in meshes for vertex in mesh["vertices"])
        ranked.append((radius, len(assets)))
        assets.append(merged_asset(f"mine_{bases.index(base)}", bases.index(base), code, meshes))
    placements = [(index, radius) for radius, index in sorted(ranked, reverse=True)]
    groups = [group_payload(f"mine_{index}", placements, choice["scale"]) for index in range(6)]
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / target_name
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=Path("Renderer/packs/ImprovementsNormalized"))
    parser.add_argument("--output", default="mine_runtime.bin", help="file name inside the pack")
    args = parser.parse_args()
    target = build(args.pack.resolve(), target_name=args.output)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
