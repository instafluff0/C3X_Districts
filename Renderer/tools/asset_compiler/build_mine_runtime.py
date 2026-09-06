#!/usr/bin/env python3
"""Build the compact recursive source-independent mine bundle used by L18."""

from __future__ import annotations

import argparse
import json
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


def group_payload(name: str, placements: list[tuple[int, float]]) -> bytes:
    payload = bytearray(bundle_string(name))
    payload.extend(struct.pack("<I", len(placements)))
    for asset_index, radius in placements:
        # Largest merged part sorts first so the renderer uses it for the one
        # compound footprint shadow while discarding redundant child shadows.
        payload.extend(struct.pack("<IffIIIIff", asset_index, 1.18, 0.04, 1, 1, 5, 0,
                                   radius, 0.0))
    return bytes(payload)


def build(pack: Path) -> Path:
    manifest = _load_json(pack / "manifest.json")
    catalog = _load_json(pack / manifest["improvement_catalog"])
    root_ids = [variant for era in catalog["mine"]["eras"] for variant in era["variants"]]
    all_parts = {asset_id: collect_parts(pack, manifest, asset_id) for asset_id in root_ids}
    base_counts = defaultdict(int)
    for parts in all_parts.values():
        for _mesh, base, _emissive in parts:
            base_counts[base] += 1
    # Six dominant materials retain 356/366 source draw parts; the ten tiny
    # rare-prop materials are omitted so two confirmed emissive channels fit
    # the generic eight-texture FeatureBundle ABI.
    base_textures = sorted(base_counts, key=lambda item: (-base_counts[item], item))[:6]
    emissive_textures = sorted({emissive for parts in all_parts.values()
                                for _mesh, _base, emissive in parts if emissive})
    if len(emissive_textures) != 2:
        raise ValueError("L18 compact mine proof expects two confirmed emissive channels")
    textures = base_textures + emissive_textures
    assets = []
    groups = []
    for root_index, asset_id in enumerate(root_ids):
        merged = defaultdict(list)
        for mesh, base, emissive in all_parts[asset_id]:
            if base not in base_textures:
                continue
            emissive_code = 0 if emissive is None else emissive_textures.index(emissive) + 1
            merged[(base_textures.index(base), emissive_code)].append(mesh)
        placements = []
        ranked = []
        for (texture_index, emissive_code), meshes in merged.items():
            radius = max((vertex["position"][0] ** 2 + vertex["position"][1] ** 2) ** 0.5
                         for mesh in meshes for vertex in mesh["vertices"])
            ranked.append((radius, texture_index, emissive_code, meshes))
        for radius, texture_index, emissive_code, meshes in sorted(ranked, reverse=True):
            index = len(assets)
            assets.append(merged_asset(f"mine_{root_index}_{texture_index}", texture_index,
                                       emissive_code, meshes))
            placements.append((index, radius))
        groups.append(group_payload(f"mine_{root_index}", placements))
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / "mine_runtime.bin"
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("Renderer/packs/ImprovementsNormalized"))
    args = parser.parse_args()
    target = build(args.pack.resolve())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
