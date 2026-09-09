"""Compile neutral huts and camps into the generic feature-bundle format.

Inputs remain local normalized art. Runtime receives no source-game paths,
gameplay rules, or source-specific loader behavior.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import shutil
import struct

from Renderer.preview.render_improvement_sheet import IDENTITY, _matrix_multiply, _skeleton_worlds, _point
from Renderer.tools.asset_compiler.build_mine_runtime import MAGIC, bundle_string, merged_asset, group_payload

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Renderer/packs/TileObjectsNormalized"
HEIGHT_SCALE = 2.6  # Retained strategic-map calibration; normals follow its inverse transpose.


def transform_mesh(mesh, matrix):
    rows = [matrix[i:i+3] for i in (0, 4, 8)]
    def cross(a, b):
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    cofactors = [cross(rows[1], rows[2]), cross(rows[2], rows[0]), cross(rows[0], rows[1])]
    determinant = sum(a*b for a, b in zip(rows[0], cofactors[0]))
    if abs(determinant) < 1e-10:
        raise ValueError("Singular site component transform")
    vertices = []
    for vertex in mesh["vertices"]:
        position = _point(vertex["position"], matrix, True)
        position[2] *= HEIGHT_SCALE
        normal = [sum(vertex["normal"][i]*cofactors[i][j] for i in range(3))/determinant for j in range(3)]
        normal[2] /= HEIGHT_SCALE
        length = math.sqrt(sum(x*x for x in normal))
        if length < 1e-10 or not all(math.isfinite(x) for x in position + normal):
            raise ValueError("Invalid site vertex")
        vertices.append({**vertex, "position": position, "normal": [x/length for x in normal]})
    return {**mesh, "vertices": vertices}


def plan():
    consumed = {}
    def read(path):
        path = path.resolve()
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes(); consumed[name] = hashlib.sha256(data).hexdigest()
        return data
    def document(path):
        return json.loads(read(path))
    manifest = document(SOURCE / "manifest.json")
    catalog = document(SOURCE / manifest["tile_object_catalog"])
    roots = [(f"hut_{i}", key) for i, key in enumerate(catalog["goody_hut"]["variants"])]
    primitive = next(stage for stage in catalog["barbarian_camp"]["stages"] if stage["default_for_civ3"])
    roots += [("camp", primitive["variants"][0])]
    def collect(key, matrix=IDENTITY, stack=()):
        if key in stack or len(stack) >= 12:
            raise ValueError("Site component cycle")
        landmark = document(SOURCE / manifest["assets"][key]["landmark"])
        parts = []
        for binding in landmark["draw_bindings"]:
            if "worked" not in binding["states"]:
                continue
            mesh = document(SOURCE / landmark["components"]["geometry"][binding["geometry"]])
            material = document(SOURCE / landmark["components"]["materials"][binding["material"]])
            base = material.get("channels", {}).get("base_color")
            if base:
                transformed = transform_mesh(mesh, matrix)
                # Source ground decals do not replace C3X's authoritative ground.
                if max(v["position"][2] for v in transformed["vertices"]) >= .006*HEIGHT_SCALE:
                    parts.append((transformed, base["texture"]))
        skeletons = [_skeleton_worlds(document(SOURCE / name)) for name in landmark["components"]["skeletons"]]
        for point in landmark["attachment_points"]:
            if point["binding_status"] == "resolved":
                child = _matrix_multiply(skeletons[point["skeleton"]][point["bone"]], matrix)
                parts.extend(collect(point["component_asset"], child, stack+(key,)))
        return parts
    groups = {role: collect(key) for role, key in roots}
    textures = sorted({texture for parts in groups.values() for _, texture in parts})
    if not 1 <= len(textures) <= 8 or any(not parts for parts in groups.values()):
        raise ValueError("Site material/body count exceeds the feature binding")
    for texture in textures:
        read(SOURCE / texture)
    for name in ("Renderer/preview/render_improvement_sheet.py", "Renderer/tools/asset_compiler/build_mine_runtime.py", "Renderer/tools/asset_compiler/build_site_runtime.py"):
        read(ROOT / name)
    return groups, textures, consumed


def build(output):
    groups, textures, consumed = plan()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, texture in enumerate(textures):
        target = output / f"textures/base_{i}.dds"; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(SOURCE / texture, target); paths.append(target.relative_to(output).as_posix())
    paths += [paths[0]] * (8-len(paths))
    assets, payloads = [], []
    counts = {}
    for role, parts in groups.items():
        merged = defaultdict(list)
        for mesh, texture in parts:
            merged[textures.index(texture)].append(mesh)
        placements = []
        for texture, meshes in sorted(merged.items()):
            placements.append((len(assets), .5))
            assets.append(merged_asset(role, texture, 0, meshes))
        payloads.append(group_payload(role, placements)); counts[role] = len(parts)
    bundle = bytearray(MAGIC) + struct.pack("<IIII", 1, 8, len(assets), len(payloads))
    for path in paths: bundle.extend(bundle_string(path))
    for part in assets + payloads: bundle.extend(part)
    (output / "sites.bin").write_bytes(bundle)
    (output / "manifest.json").write_text(json.dumps({"schema": "c3x.static_sites.v1", "groups": counts,
        "height_scale": HEIGHT_SCALE, "normals": "inverse_transpose", "neutral": True,
        "source_sha256": consumed}, indent=2)+"\n")
    return consumed
