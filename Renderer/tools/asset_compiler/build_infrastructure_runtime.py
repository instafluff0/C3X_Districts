#!/usr/bin/env python3
"""Build compact source-independent L19B infrastructure bundles."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Renderer.tools.asset_compiler.build_mine_runtime import (
    IDENTITY,
    MAGIC,
    _load_json,
    _matrix_multiply,
    _skeleton_worlds,
    _transform_mesh,
    bundle_string,
    collect_parts,
    group_payload,
    merged_asset,
)


def selected_parts(root: Path, manifest: dict, asset_id: str, selected: set[str],
                   transform=None, stack=()):
    """Collect selected authored children while retaining root attachment transforms."""
    if asset_id in stack or len(stack) >= 12:
        raise ValueError("infrastructure component graph cycles or is too deep")
    transform = IDENTITY if transform is None else transform
    landmark = _load_json(root / manifest["assets"][asset_id]["landmark"])
    parts = []
    if asset_id in selected:
        for binding in landmark["draw_bindings"]:
            if "worked" not in binding["states"]:
                continue
            mesh = _load_json(root / landmark["components"]["geometry"][binding["geometry"]])
            material = _load_json(root / landmark["components"]["materials"][binding["material"]])
            base = material.get("channels", {}).get("base_color")
            if base:
                parts.append((_transform_mesh(mesh, transform), base["texture"], None))
    skeleton_worlds = [
        _skeleton_worlds(_load_json(root / path))
        for path in landmark["components"]["skeletons"]
    ]
    for point in landmark["attachment_points"]:
        if point["binding_status"] != "resolved":
            continue
        child_transform = _matrix_multiply(
            skeleton_worlds[point["skeleton"]][point["bone"]], transform)
        parts.extend(selected_parts(root, manifest, point["component_asset"], selected,
                                    child_transform, stack + (asset_id,)))
    return parts


def write_bundle(target: Path, textures: list[str], groups_and_parts, emissive_texture=None):
    assets: list[bytes] = []
    groups: list[bytes] = []
    base_textures = textures[:-1] if emissive_texture else textures
    for name, parts, silhouette_only in groups_and_parts:
        merged = defaultdict(list)
        for mesh, base, emissive in parts:
            if base not in base_textures:
                continue
            maximum_z = max(vertex["position"][2] for vertex in mesh["vertices"])
            if silhouette_only and maximum_z < 0.004:
                continue
            if maximum_z >= 0.004:
                for vertex in mesh["vertices"]:
                    vertex["position"][2] *= 2.6
            else:
                # Keep authored runway/decal pieces just above the shared
                # terrain surface so they do not z-fight out of the frame.
                for vertex in mesh["vertices"]:
                    vertex["position"][2] += 0.005
            emissive_code = int(emissive_texture is not None and emissive == emissive_texture)
            merged[(base_textures.index(base), emissive_code)].append(mesh)
        placements = []
        ranked = []
        for (texture_index, emissive_code), meshes in merged.items():
            radius = max(
                (vertex["position"][0] ** 2 + vertex["position"][1] ** 2) ** 0.5
                for mesh in meshes for vertex in mesh["vertices"])
            ranked.append((radius, texture_index, emissive_code, meshes))
        for radius, texture_index, emissive_code, meshes in sorted(ranked, reverse=True):
            asset_index = len(assets)
            assets.append(merged_asset(
                f"{name}:base_{texture_index}", texture_index, emissive_code, meshes))
            placements.append((asset_index, radius))
        if not placements:
            raise ValueError(f"empty infrastructure group {name}")
        groups.append(group_payload(name, placements))
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target.write_bytes(output)


def quad(uv_left: float, uv_top: float, uv_right: float, uv_bottom: float,
         z: float = 0.003):
    return {
        "vertices": [
            {"position": [-0.46, -0.46, z], "normal": [0.0, 0.0, 1.0], "uv0": [uv_left, uv_bottom]},
            {"position": [-0.46,  0.46, z], "normal": [0.0, 0.0, 1.0], "uv0": [uv_left, uv_top]},
            {"position": [ 0.46,  0.46, z], "normal": [0.0, 0.0, 1.0], "uv0": [uv_right, uv_top]},
            {"position": [ 0.46, -0.46, z], "normal": [0.0, 0.0, 1.0], "uv0": [uv_right, uv_bottom]},
        ],
        "topology": {"indices": [0, 1, 2, 0, 2, 3]},
    }


def write_ground_bundle(target: Path):
    textures = [
        "../FutureGateCandidates/textures/compound/base_color_c68c12748c938e3e.dds",
        "../AmbientEffectsNormalized/textures/effects/effect_pollution_radiation_atlas.dds",
    ]
    assets = []
    groups = []
    quadrants = [(0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5),
                 (0.0, 0.5, 0.5, 1.0), (0.5, 0.5, 1.0, 1.0)]
    for family, texture_index in (("crater", 0), ("pollution", 1)):
        for index, uv in enumerate(quadrants):
            placements = []
            layer_count = 2 if family == "pollution" else 1
            for layer in range(layer_count):
                layer_uv = quadrants[(index + layer) % len(quadrants)]
                layer_texture_index = 0 if family == "pollution" and layer == 0 else texture_index
                asset_index = len(assets)
                assets.append(merged_asset(
                    f"{family}_{index}_{layer}:base_{layer_texture_index}",
                    layer_texture_index, 0,
                    [quad(*layer_uv, z=0.003 + layer * 0.001)]))
                placements.append((asset_index, 0.65 - layer * 0.01))
            groups.append(group_payload(f"{family}_{index}", placements))
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target.write_bytes(output)


def build(pack: Path) -> list[Path]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    fort_ids = ["infrastructure/fort/medieval", "infrastructure/fort/industrial"]
    fort_parts = {asset_id: collect_parts(pack, manifest, asset_id) for asset_id in fort_ids}
    tower_ids = ["infrastructure/outpost/airstrip_tower",
                 "infrastructure/outpost/industrial_watchtower"]
    tower_parts = {asset_id: collect_parts(pack, manifest, asset_id) for asset_id in tower_ids}
    counts = Counter(base for parts in fort_parts.values() for _mesh, base, _emissive in parts)
    tower_textures = [parts[0][1] for parts in tower_parts.values()]
    fort_textures = [item for item, _count in counts.most_common() if item not in tower_textures]
    textures = fort_textures[:6] + tower_textures
    victory_parts = selected_parts(
        pack, manifest, fort_ids[0],
        {"tile_object/component/afa55ecacbd3f575", "tile_object/component/8b9f26764003fe86"})
    fort_target = pack / "fortification_runtime.bin"
    write_bundle(fort_target, textures, [
        ("fort_0", fort_parts[fort_ids[0]], True),
        ("fort_1", fort_parts[fort_ids[1]], True),
        ("outpost_0", tower_parts[tower_ids[0]], True),
        ("outpost_1", tower_parts[tower_ids[1]], True),
        ("radar", tower_parts[tower_ids[1]], True),
        ("victory", victory_parts, True),
    ])

    airfield_id = "infrastructure/airfield/airstrip"
    airfield_parts = collect_parts(pack, manifest, airfield_id)
    base_counts = Counter(base for _mesh, base, _emissive in airfield_parts)
    emissive_counts = Counter(emissive for _mesh, _base, emissive in airfield_parts if emissive)
    runway_emissive = emissive_counts.most_common(1)[0][0]
    airfield_textures = [item for item, _count in base_counts.most_common(7)] + [runway_emissive]
    airfield_target = pack / "airfield_runtime.bin"
    write_bundle(airfield_target, airfield_textures,
                 [("airfield", airfield_parts, False)], runway_emissive)

    ground_target = pack / "ground_state_runtime.bin"
    write_ground_bundle(ground_target)
    return [fort_target, airfield_target, ground_target]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path,
                        default=Path("Renderer/packs/TileObjectsNormalized"))
    args = parser.parse_args()
    targets = build(args.pack.resolve())
    for target in targets:
        print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
