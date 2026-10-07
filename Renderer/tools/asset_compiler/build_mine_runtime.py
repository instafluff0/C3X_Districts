#!/usr/bin/env python3
"""Build the source-independent mine bundle: one chosen building for every era.

The strategy's mine "runtime_building" names the Civ VI component, its scale
and the facing that turns its Civ VI front toward the C3X camera. The runtime
selects group mine_<family*3+variant>; all six groups hold the same building.
The bundle binds six base and two emissive textures; unused slots share a 4x4
placeholder.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

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
    bases = sorted({base for base, _ in merged})
    emissives = sorted({emissive for _, emissive in merged if emissive})
    if not merged or len(bases) > BASE_SLOTS or len(emissives) > EMISSIVE_SLOTS:
        raise ValueError("Mine building needs 1-6 base and at most 2 emissive textures")
    placeholder = pack / PLACEHOLDER
    if not placeholder.is_file():
        # A 4x4 BC1 sRGB DX10 DDS: unused slots load it instead of a real texture again.
        header = struct.pack("<7I44s32s5I", 124, 0xA1007, 4, 4, 8, 0, 1, bytes(44),
                             struct.pack("<II4s5I", 32, 0x4, b"DX10", 0, 0, 0, 0, 0), 0x401008, 0, 0, 0, 0)
        placeholder.write_bytes(b"DDS " + header + struct.pack("<5I", 72, 3, 0, 1, 0) + bytes(8))
    textures = (bases + [PLACEHOLDER] * (BASE_SLOTS - len(bases)) +
                emissives + [PLACEHOLDER] * (EMISSIVE_SLOTS - len(emissives)))
    assets, ranked = [], []
    for (base, emissive), meshes in sorted(merged.items(), key=lambda item: (item[0][0], item[0][1] or "")):
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
