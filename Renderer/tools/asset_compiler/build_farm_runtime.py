#!/usr/bin/env python3
"""Build the compact recursive source-independent farm bundle used by L19."""

from __future__ import annotations

import argparse
import json
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
        (pack / manifest["improvement_catalog"]).read_text(encoding="utf-8")
    )
    crop = catalog["farm"]["crop_styles"][0]["pieces"][1]
    roots: list[tuple[str, str]] = []
    for era_index, era in enumerate(catalog["farm"]["eras"]):
        roots.extend(
            [
                (f"farm_{era_index}:base", era["tile_bases"][0]),
                (f"farm_{era_index}:building", era["building_pieces"][1]),
                (f"farm_{era_index}:crop", crop),
            ]
        )
    all_parts = {
        role: collect_parts(pack, manifest, asset_id)
        for role, asset_id in roots
    }
    base_counts: dict[str, int] = defaultdict(int)
    crop_counts: dict[str, int] = defaultdict(int)
    for parts in all_parts.values():
        for _mesh, base, _emissive in parts:
            base_counts[base] += 1
    for role, parts in all_parts.items():
        if role.endswith(":crop"):
            for _mesh, base, _emissive in parts:
                crop_counts[base] += 1
    # Crop geometry is the semantic body of irrigation. Preserve all five of
    # its authored materials before selecting one dominant boundary material;
    # a global frequency sort wrongly kept repeated trees and discarded the
    # low-count green/gold field channels.
    if len(crop_counts) != 5:
        raise ValueError("L19 farm crop proof expects five authored base materials")
    base_textures = sorted(crop_counts, key=lambda item: (-crop_counts[item], item))
    base_textures += sorted(
        (item for item in base_counts if item not in crop_counts),
        key=lambda item: (-base_counts[item], item),
    )[:1]
    emissive_textures = sorted(
        {
            emissive
            for parts in all_parts.values()
            for _mesh, _base, emissive in parts
            if emissive
        }
    )
    if len(emissive_textures) != 2:
        raise ValueError("L19 compact farm proof expects two confirmed emissive channels")
    textures = base_textures + emissive_textures
    assets: list[bytes] = []
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for role, _asset_id in roots:
        merged: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for mesh, base, emissive in all_parts[role]:
            if base not in base_textures:
                continue
            emissive_code = 0 if emissive is None else emissive_textures.index(emissive) + 1
            merged[(base_textures.index(base), emissive_code)].append(mesh)
        ranked = []
        for (texture_index, emissive_code), meshes in merged.items():
            radius = max(
                (vertex["position"][0] ** 2 + vertex["position"][1] ** 2) ** 0.5
                for mesh in meshes
                for vertex in mesh["vertices"]
            )
            ranked.append((radius, texture_index, emissive_code, meshes))
        era = int(role.split(":", 1)[0].rsplit("_", 1)[1])
        role_name = role.split(":", 1)[1]
        for radius, texture_index, emissive_code, meshes in sorted(ranked, reverse=True):
            asset_index = len(assets)
            assets.append(
                merged_asset(
                    f"{role}:{role_name}_{texture_index}",
                    texture_index,
                    emissive_code,
                    meshes,
                )
            )
            grouped[era].append((asset_index, radius))
    groups = [group_payload(f"farm_{era}", grouped[era]) for era in range(3)]
    output = bytearray(MAGIC)
    import struct

    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / "farm_runtime.bin"
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack", type=Path, default=Path("Renderer/packs/ImprovementsNormalized")
    )
    args = parser.parse_args()
    target = build(args.pack.resolve())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
