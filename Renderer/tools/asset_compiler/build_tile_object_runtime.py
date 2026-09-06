#!/usr/bin/env python3
"""Build the compact source-independent goody-hut/colony bundle for L19A."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Renderer.tools.asset_compiler.build_mine_runtime import (
    MAGIC,
    bundle_string,
    collect_parts,
    group_payload,
    merged_asset,
)


def build(pack: Path) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads(
        (pack / manifest["tile_object_catalog"]).read_text(encoding="utf-8")
    )
    roots: list[tuple[str, str]] = []
    roots.extend(
        (f"hut_{index}", asset_id)
        for index, asset_id in enumerate(catalog["goody_hut"]["variants"])
    )
    # Ancient and industrial source stages each provide three variants. Civ III
    # eras map to these two stages in the Lab scene rather than duplicating data.
    for family, era_index in enumerate((0, 2)):
        roots.extend(
            (f"colony_{family * 3 + variant}", asset_id)
            for variant, asset_id in enumerate(
                catalog["colony"]["eras"][era_index]["variants"]
            )
        )
    all_parts = {
        role: collect_parts(pack, manifest, asset_id)
        for role, asset_id in roots
    }
    counts: dict[str, int] = defaultdict(int)
    body_counts: dict[str, int] = defaultdict(int)
    for parts in all_parts.values():
        for mesh, base, _emissive in parts:
            counts[base] += 1
            if max(vertex["position"][2] for vertex in mesh["vertices"]) >= 0.006:
                body_counts[base] += 1
    # Reserve every material used by silhouette-bearing source geometry before
    # padding the proven eight-slot ABI with the most common residual channels.
    # A global frequency sort dropped the hut foliage/stone atlas and retained
    # broad near-flat camp-ground pieces, reducing huts to orange palisades.
    textures = sorted(body_counts, key=lambda item: (-body_counts[item], item))
    textures += sorted(
        (item for item in counts if item not in body_counts),
        key=lambda item: (-counts[item], item),
    )[: 8 - len(textures)]
    if len(textures) != 8:
        raise ValueError("L19A compact tile-object proof expects eight base materials")
    assets: list[bytes] = []
    groups: list[bytes] = []
    for role, _asset_id in roots:
        merged: dict[int, list[dict]] = defaultdict(list)
        for mesh, base, _emissive in all_parts[role]:
            if base not in textures or max(
                vertex["position"][2] for vertex in mesh["vertices"]
            ) < 0.006:
                continue
            # The source coordinates are normalized for Civ VI's hex camera.
            # Preserve XY footprint but correct Z into the established Civ III
            # isometric pixels-per-tile basis so shelters remain legible.
            for vertex in mesh["vertices"]:
                vertex["position"][2] *= 2.6
            merged[textures.index(base)].append(mesh)
        ranked = []
        for texture_index, meshes in merged.items():
            radius = max(
                (vertex["position"][0] ** 2 + vertex["position"][1] ** 2) ** 0.5
                for mesh in meshes
                for vertex in mesh["vertices"]
            )
            ranked.append((radius, texture_index, meshes))
        placements = []
        for radius, texture_index, meshes in sorted(ranked, reverse=True):
            asset_index = len(assets)
            assets.append(
                merged_asset(f"{role}:base_{texture_index}", texture_index, 0, meshes)
            )
            placements.append((asset_index, radius))
        groups.append(group_payload(role, placements))
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / "tile_object_runtime.bin"
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack", type=Path, default=Path("Renderer/packs/TileObjectsNormalized")
    )
    args = parser.parse_args()
    target = build(args.pack.resolve())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
