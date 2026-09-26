#!/usr/bin/env python3
"""Build the compact generic farm bundle for Renderer Lab."""

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


def terrain_samples(mesh: dict, steps: int = 12) -> dict:
    """Add barycentric samples to flat source decals without changing their UVs."""
    vertices = mesh["vertices"]
    if not vertices or any(vertex["position"][2] != vertices[0]["position"][2]
                           for vertex in vertices):
        return mesh
    sampled = []
    indices = []
    source = mesh["topology"]["indices"]
    for start in range(0, len(source), 3):
        triangle = [vertices[source[start + corner]] for corner in range(3)]
        lookup = {}
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                weights = (1 - (i + j) / steps, i / steps, j / steps)
                lookup[i, j] = len(sampled)
                sampled.append({key: [sum(weights[corner] * triangle[corner][key][axis]
                                          for corner in range(3))
                                      for axis in range(len(triangle[0][key]))]
                                for key in ("position", "normal", "uv0")})
        for i in range(steps):
            for j in range(steps - i):
                indices.extend((lookup[i, j], lookup[i + 1, j], lookup[i, j + 1]))
                if i + j + 1 < steps:
                    indices.extend((lookup[i + 1, j], lookup[i + 1, j + 1], lookup[i, j + 1]))
    return {**mesh, "vertices": sampled,
            "topology": {**mesh["topology"], "indices": indices}}


def planted_rows(mesh: dict) -> dict:
    """Select one authored planted field from the source crop atlas."""
    # The normalized decal describes the complete atlas. Repeating that atlas
    # over a small field makes its many farms look like a checkerboard. This
    # source region has visible crop rows and a narrow soil border.
    u0, v0, u1, v1 = .248, .515, .465, .748
    return {**mesh, "vertices": [
        {**vertex, "uv0": [u0 + vertex["uv0"][0] * (u1 - u0),
                            v0 + vertex["uv0"][1] * (v1 - v0)]}
        for vertex in mesh["vertices"]]}


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
                (f"farm_{era_index}:building", era["building_pieces"][1]),
                (f"farm_{era_index}:crop", crop),
                (f"farm_{era_index}:tree", era["tile_bases"][0]),
            ]
        )
    all_parts = {
        role: collect_parts(pack, manifest, asset_id)
        for role, asset_id in roots
    }
    crop_counts: dict[str, int] = defaultdict(int)
    for role, parts in all_parts.items():
        if role.endswith(":crop"):
            for _mesh, base, _emissive in parts:
                crop_counts[base] += 1
    if len(crop_counts) != 5:
        raise ValueError("Farm crop source should contain five authored materials")
    ranked_crops = sorted(crop_counts, key=lambda item: (-crop_counts[item], item))
    # Six bound color slots serve farms. Four field palettes leave room for
    # actual source tree canopies and buildings; the fifth crop palette is a
    # soil/path duplicate and cannot justify stripping all raised geometry.
    base_textures = [ranked_crops[i] for i in (0, 2, 3, 4)]
    for role_name in (":tree", ":building"):
        counts: dict[str, int] = defaultdict(int)
        for role, parts in all_parts.items():
            if role.endswith(role_name):
                for mesh, base, _emissive in parts:
                    if max(vertex["position"][2] for vertex in mesh["vertices"]) > 0.02:
                        counts[base] += len(mesh["vertices"])
        choice = max(counts, key=counts.get)
        if choice in base_textures:
            raise ValueError("Farm raised material duplicates a field palette")
        base_textures.append(choice)
    emissive_textures = sorted(
        {
            emissive
            for parts in all_parts.values()
            for _mesh, _base, emissive in parts
            if emissive
        }
    )
    if len(emissive_textures) != 2:
        raise ValueError("Compact farm bundle expects two confirmed emissive channels")
    textures = base_textures + emissive_textures
    assets: list[bytes] = []
    grouped: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for role, _asset_id in roots:
        era = int(role.split(":", 1)[0].rsplit("_", 1)[1])
        if role.endswith(":tree"):
            # A centered source tree can be scattered independently along the
            # plots without importing its large prearranged tile cluster.
            candidates = [mesh for mesh, base, _emissive in all_parts[role]
                          if base == base_textures[4] and
                          max(vertex["position"][2] for vertex in mesh["vertices"]) > .025]
            if not candidates:
                raise ValueError(f"No raised source tree in {role}")
            tree = max(candidates, key=lambda mesh: max(
                vertex["position"][2] for vertex in mesh["vertices"]))
            cx = (min(vertex["position"][0] for vertex in tree["vertices"]) +
                  max(vertex["position"][0] for vertex in tree["vertices"])) * .5
            cy = (min(vertex["position"][1] for vertex in tree["vertices"]) +
                  max(vertex["position"][1] for vertex in tree["vertices"])) * .5
            centered = {**tree, "vertices": [
                {**vertex, "position": [vertex["position"][0] - cx,
                                        vertex["position"][1] - cy,
                                        vertex["position"][2]]}
                for vertex in tree["vertices"]]}
            asset_index = len(assets)
            assets.append(merged_asset(f"{role}:source", 4, 0, [centered]))
            grouped[era].append((asset_index, .04))
            continue
        if role.endswith(":building"):
            candidates = [mesh for mesh, base, _emissive in all_parts[role]
                          if base == base_textures[5] and
                          max(vertex["position"][2] for vertex in mesh["vertices"]) > .03]
            if not candidates:
                raise ValueError(f"No raised source building in {role}")
            building = max(candidates, key=lambda mesh: len(mesh["vertices"]))
            cx = (min(vertex["position"][0] for vertex in building["vertices"]) +
                  max(vertex["position"][0] for vertex in building["vertices"])) * .5
            cy = (min(vertex["position"][1] for vertex in building["vertices"]) +
                  max(vertex["position"][1] for vertex in building["vertices"])) * .5
            centered = {**building, "vertices": [
                {**vertex, "position": [vertex["position"][0] - cx,
                                        vertex["position"][1] - cy,
                                        vertex["position"][2]]}
                for vertex in building["vertices"]]}
            asset_index = len(assets)
            assets.append(merged_asset(f"{role}:source", 5, 0, [centered]))
            grouped[era].append((asset_index, .08))
            continue
        merged: dict[tuple[int, int], list[dict]] = defaultdict(list)
        used_crop_materials: set[tuple[int, int]] = set()
        for mesh, base, emissive in all_parts[role]:
            if base not in base_textures:
                continue
            emissive_code = 0 if emissive is None else emissive_textures.index(emissive) + 1
            key = (base_textures.index(base), emissive_code)
            if role.endswith(":crop"):
                if key in used_crop_materials:
                    continue  # These are alternate decals at the same footprint.
                used_crop_materials.add(key)
                if key[0] < 3:
                    mesh = planted_rows(mesh)
            merged[key].append(
                terrain_samples(mesh))
        ranked = []
        for (texture_index, emissive_code), meshes in merged.items():
            radius = max(
                (vertex["position"][0] ** 2 + vertex["position"][1] ** 2) ** 0.5
                for mesh in meshes
                for vertex in mesh["vertices"]
            )
            ranked.append((radius, texture_index, emissive_code, meshes))
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
