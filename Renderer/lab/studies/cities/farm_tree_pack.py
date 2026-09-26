#!/usr/bin/env python3
"""Extract one source farm tree for isolated city-composition Lab trials."""

import argparse
import json
import shutil
from pathlib import Path

from Renderer.lab.shared.cities.fingerprint import geometry_digest
from Renderer.tools.asset_compiler.build_mine_runtime import collect_parts


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "Renderer/packs/ImprovementsNormalized"
ASSET = "city/prop/source_farm_tree"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(output):
    output = output.resolve()
    output.relative_to(ROOT / "Renderer/lab")
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    catalog = json.loads((SOURCE / manifest["improvement_catalog"]).read_text())
    tile_base = catalog["farm"]["eras"][0]["tile_bases"][0]
    parts = collect_parts(SOURCE, manifest, tile_base)
    # This is the very same canopy texture and tallest mesh chosen by the
    # preindustrial farm runtime, preserving its source UVs and vertex detail.
    texture = "textures/compound/base_color_7bae9a4a12178e25.dds"
    candidates = [mesh for mesh, base, _ in parts if base == texture and
                  max(vertex["position"][2] for vertex in mesh["vertices"]) > .025]
    if not candidates:
        raise ValueError("farm source tree missing")
    source = max(candidates, key=lambda mesh: max(
        vertex["position"][2] for vertex in mesh["vertices"]))
    center = [(min(v["position"][axis] for v in source["vertices"]) +
               max(v["position"][axis] for v in source["vertices"])) / 2
              for axis in (0, 1)]
    mesh = {**source, "asset_id": ASSET + "/geometry_00",
            "vertices": [{**v, "position": [v["position"][0] - center[0],
                                            v["position"][1] - center[1],
                                            v["position"][2]]}
                         for v in source["vertices"]]}
    write(output / "manifest.json", {"assets": {ASSET: {"landmark": "compound_landmarks/tree.json"}}})
    write(output / "compound_landmarks/tree.json", {
        "components": {"geometry": ["meshes/tree.json"], "materials": ["materials/tree.json"]},
        "draw_bindings": [{"geometry": 0, "material": 0, "states": ["worked"]}],
        "attachment_points": [],
    })
    write(output / "meshes/tree.json", mesh)
    write(output / "materials/tree.json", {
        "name": "source_farm_tree", "alpha_mode": "opaque",
        "channels": {"base_color": {"texture": texture,
                                    "address_u": "clamp", "address_v": "clamp"}},
    })
    target_texture = output / texture
    target_texture.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE / texture, target_texture)
    # This source mesh already carries explicit vertex normals. It has no
    # tangent-space normal map, so fixed orthogonal tangents are sufficient.
    frames = {mesh["asset_id"]: {
        "geometry_digest": geometry_digest(mesh),
        "normals": [v["normal"] for v in mesh["vertices"]],
        "tangents": [[1, 0, 0] for _ in mesh["vertices"]],
        "bitangents": [[0, 1, 0] for _ in mesh["vertices"]],
    }}
    write(output / "frames.json", {"meshes": frames})
    print(f"{ASSET}: {len(mesh['vertices'])} source vertices; {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
