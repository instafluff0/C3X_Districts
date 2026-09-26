"""Build a Lab mine bundle that retains ground-decal identity and source art."""

from __future__ import annotations

import json
import struct
from collections import defaultdict
from pathlib import Path

from Renderer.preview.render_improvement_sheet import (
    IDENTITY, _decal_documents, _decal_mesh, _load_json,
    _matrix_multiply, _skeleton_worlds, _transform_mesh,
)
from Renderer.tools.asset_compiler.build_mine_runtime import bundle_string, merged_asset


def subdivide(mesh: dict, divisions: int = 8) -> dict:
    """Sample an authored decal quad densely enough for relief and shore clipping."""
    corners = mesh["vertices"]
    vertices = []
    for row in range(divisions + 1):
        v = row / divisions
        for column in range(divisions + 1):
            u = column / divisions
            weights = ((1-u)*(1-v), u*(1-v), u*v, (1-u)*v)
            vertices.append({key: [sum(corners[k][key][axis] * weights[k] for k in range(4))
                                   for axis in range(len(corners[0][key]))]
                             for key in ("position", "normal", "uv0")})
    indices = []
    for row in range(divisions):
        for column in range(divisions):
            a = row * (divisions + 1) + column
            b = a + 1
            c = b + divisions + 1
            d = a + divisions + 1
            indices.extend((a, b, c, a, c, d))
    return {"vertices": vertices, "topology": {"indices": indices}}


def collect(root: Path, manifest: dict, asset_id: str, transform=None, stack=()):
    if asset_id in stack or len(stack) >= 12:
        raise ValueError("Mine component graph cycles or is too deep")
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
                          channels.get("emissive", {}).get("texture"), False))
    decal_path = landmark["components"].get("decal")
    if decal_path:
        for decal in _decal_documents(root, decal_path):
            base = decal.get("channels", {}).get("base_color")
            if base:
                parts.append((subdivide(_decal_mesh(decal, transform)),
                              base["texture"], None, True))
    worlds = [_skeleton_worlds(_load_json(root / path))
              for path in landmark["components"]["skeletons"]]
    for point in landmark["attachment_points"]:
        if point["binding_status"] != "resolved":
            continue
        child_transform = _matrix_multiply(
            worlds[point["skeleton"]][point["bone"]], transform)
        parts.extend(collect(root, manifest, point["component_asset"],
                             child_transform, stack + (asset_id,)))
    return parts


def local_part(mesh: dict) -> tuple[dict, float, float]:
    """Keep one authored component rigid, with its own terrain contact origin."""
    vertices = mesh["vertices"]
    z_low = min(vertex["position"][2] for vertex in vertices)
    contact = [vertex for vertex in vertices if vertex["position"][2] <= z_low + .008]
    if len(contact) < 3:
        contact = vertices
    ox = (min(vertex["position"][0] for vertex in contact) +
          max(vertex["position"][0] for vertex in contact)) * .5
    oy = (min(vertex["position"][1] for vertex in contact) +
          max(vertex["position"][1] for vertex in contact)) * .5
    centered = {"vertices": [dict(vertex, position=[vertex["position"][0] - ox,
                                                    vertex["position"][1] - oy,
                                                    vertex["position"][2]])
                             for vertex in vertices], "topology": mesh["topology"]}
    return centered, ox, oy


def build(pack: Path, *, include_ground: bool = True, scale: float = 1.34,
          conform_parts: bool = False) -> dict:
    if conform_parts and include_ground:
        raise ValueError("Individual mine components require the no-decal trial")
    manifest = _load_json(pack / "manifest.json")
    catalog = _load_json(pack / manifest["improvement_catalog"])
    roots = [variant for era in catalog["mine"]["eras"] for variant in era["variants"]]
    all_parts = {asset_id: collect(pack, manifest, asset_id) for asset_id in roots}
    counts = defaultdict(int)
    for parts in all_parts.values():
        for _mesh, base, _emissive, _ground in parts:
            counts[base] += 1
    bases = sorted(counts, key=lambda item: (-counts[item], item))[:6]
    emissives = sorted({emissive for parts in all_parts.values()
                        for _mesh, _base, emissive, _ground in parts if emissive})
    if len(emissives) != 2:
        raise ValueError("The normalized mine proof no longer has two emissive textures")
    textures = bases + emissives
    assets, groups = [], []
    stats = {"source_roots": len(roots), "source_parts": 0,
             "retained_parts": 0, "ground_decals": 0}
    for index, asset_id in enumerate(roots):
        merged = defaultdict(list)
        individual = []
        stats["source_parts"] += len(all_parts[asset_id])
        for mesh, base, emissive, ground in all_parts[asset_id]:
            if base not in bases or (ground and not include_ground):
                continue
            code = 0 if emissive is None else emissives.index(emissive) + 1
            if conform_parts:
                centered, ox, oy = local_part(mesh)
                individual.append((len(mesh["vertices"]), bases.index(base), code,
                                   centered, ox, oy))
            else:
                merged[(bases.index(base), code, ground)].append(mesh)
            stats["retained_parts"] += 1
            stats["ground_decals"] += int(ground)
        placements = []
        if conform_parts:
            for part, (_size, texture, code, mesh, ox, oy) in enumerate(
                    sorted(individual, key=lambda item: -item[0])):
                asset_index = len(assets)
                assets.append(merged_asset(f"mine_{index}:part:{part}:{texture}",
                                           texture, code, [mesh]))
                placements.append((asset_index, ox, oy))
        else:
            ranked = []
            for (texture, code, ground), meshes in merged.items():
                radius = max((vertex["position"][0] ** 2 + vertex["position"][1] ** 2) ** .5
                             for mesh in meshes for vertex in mesh["vertices"])
                ranked.append((ground, -radius, texture, code, meshes))
            for ground, _radius, texture, code, meshes in sorted(
                    ranked, key=lambda value: (value[0], value[1], value[2], value[3])):
                asset_index = len(assets)
                kind = "ground" if ground else "body"
                assets.append(merged_asset(f"mine_{index}:{kind}:{texture}",
                                           texture, code, meshes))
                placements.append((asset_index, .5, 0.0))
        group = bytearray(bundle_string(f"mine_{index}"))
        group.extend(struct.pack("<I", len(placements)))
        for asset_index, ox, oy in placements:
            group.extend(struct.pack("<IffIIIIff", asset_index, scale, .04,
                                     1, 1, 5, 0, ox, oy))
        groups.append(bytes(group))
    output = bytearray(b"C3XVEG1\0")
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / "mine_runtime.bin"
    target.write_bytes(output)
    stats.update({"runtime_assets": len(assets), "bytes": len(output),
                  "include_ground": include_ground, "uniform_scale": scale,
                  "conform_parts": conform_parts})
    return stats
