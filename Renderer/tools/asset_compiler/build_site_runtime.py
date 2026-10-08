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
GROUND = ROOT / "Renderer/packs/GroundStatesNormalized"
LOOKS = Path(__file__).with_name("tile_site_looks.json")
HEIGHT_SCALE = 2.6  # Retained strategic-map calibration; normals follow its inverse transpose.


def transform_mesh(mesh, matrix, height_scale=HEIGHT_SCALE):
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
        position[2] *= height_scale
        normal = [sum(vertex["normal"][i]*cofactors[i][j] for i in range(3))/determinant for j in range(3)]
        normal[2] /= height_scale
        length = math.sqrt(sum(x*x for x in normal))
        if length < 1e-10 or not all(math.isfinite(x) for x in position + normal):
            raise ValueError("Invalid site vertex")
        vertices.append({**vertex, "position": position, "normal": [x/length for x in normal]})
    return {**mesh, "vertices": vertices}


def resize(mesh, body, spread):
    """Grow one piece about its own centre and move it toward the site's centre.
    A uniform scale per piece leaves its normals unchanged."""
    xs = [v["position"][0] for v in mesh["vertices"]]; ys = [v["position"][1] for v in mesh["vertices"]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    return {**mesh, "vertices": [{**v, "position": [(v["position"][0] - cx) * body + cx * spread,
                                                    (v["position"][1] - cy) * body + cy * spread,
                                                    v["position"][2] * body]} for v in mesh["vertices"]]}


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
    looks = document(LOOKS)
    def collect(key, matrix=IDENTITY, stack=(), height_scale=HEIGHT_SCALE):
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
                transformed = transform_mesh(mesh, matrix, height_scale)
                # Source ground decals do not replace C3X's authoritative ground.
                if max(v["position"][2] for v in transformed["vertices"]) >= .006*height_scale:
                    parts.append((transformed, base["texture"]))
        skeletons = [_skeleton_worlds(document(SOURCE / name)) for name in landmark["components"]["skeletons"]]
        for point in landmark["attachment_points"]:
            if point["binding_status"] == "resolved":
                child = _matrix_multiply(skeletons[point["skeleton"]][point["bone"]], matrix)
                parts.extend(collect(point["component_asset"], child, stack+(key,), height_scale))
        return parts
    groups = {}
    for role, key in roots:
        size = looks["sizes"]["camp" if role == "camp" else "hut"]
        groups[role] = [(resize(mesh, size["body"], size["spread"]), texture)
                        for mesh, texture in collect(key, height_scale=size["height_scale"])]
    textures = sorted({texture for parts in groups.values() for _, texture in parts})
    if not 1 <= len(textures) <= 8 or any(not parts for parts in groups.values()):
        raise ValueError("Site material/body count exceeds the feature binding")
    for texture in textures:
        read(SOURCE / texture)
    for name in ("Renderer/preview/render_improvement_sheet.py", "Renderer/tools/asset_compiler/build_mine_runtime.py",
                 "Renderer/tools/asset_compiler/build_site_runtime.py", "Renderer/tools/asset_compiler/ground_state_composer.py"):
        read(ROOT / name)
    ground_sources(read)
    return groups, textures, consumed


def ground_sources(read):
    """The ground-state source art, when the local source pack exists."""
    if not (GROUND / "manifest.json").is_file():
        return None
    manifest = json.loads(read(GROUND / "manifest.json"))
    for decal in manifest["decals"].values():
        for record in decal["decals"]:
            for channel in record["channels"].values():
                read(GROUND / channel["texture"])
    for texture in manifest["textures"].values():
        read(GROUND / texture["texture"])
    return manifest


def flat_grid(span, uv_rect, lift, cells=12):
    """A flat decal over a square of `span` tiles; per-vertex draping follows the ground."""
    u0, v0, u1, v1 = uv_rect
    vertices = [{"position": [(i / cells - .5) * span, (j / cells - .5) * span, lift], "normal": [0.0, 0.0, 1.0],
                 "uv0": [u0 + (u1 - u0) * i / cells, v0 + (v1 - v0) * j / cells]}
                for j in range(cells + 1) for i in range(cells + 1)]
    indices = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            indices += [a, a + 1, a + cells + 2, a, a + cells + 2, a + cells + 1]
    return {"vertices": vertices, "topology": {"indices": indices}}


def ground_groups(first_texture):
    """Compose the ground-state textures and their decal/prop groups.
    Returns ([(name, rgba)], [(role, [(asset_id, texture_slot, [meshes])])]) or None."""
    from Renderer.tools.asset_compiler import ground_state_composer as composer
    manifest = ground_sources(lambda path: path.read_bytes())
    if manifest is None:
        return None
    looks = json.loads(LOOKS.read_text())
    sources = composer.Sources(GROUND, manifest)
    pollution, crater, ruins = looks["pollution"], looks["crater"], looks["ruins"]
    textures = [
        ("pollution", composer.atlas([composer.pollution_cell(sources, pollution, 101 + k) for k in range(pollution["variants"])])),
        ("craters", composer.atlas([composer.crater_cell(sources, crater, 211 + k) for k in range(crater["variants"])])),
        ("rubble", composer.rubble_cell(sources, ruins)),
    ]
    slot = {name: first_texture + index for index, (name, _) in enumerate(textures)}
    quarter = lambda k: (k % 2 * .5, k // 2 * .5, k % 2 * .5 + .5, k // 2 * .5 + .5)
    groups = []
    for k in range(pollution["variants"]):
        groups.append((f"pollution_{k}", [(f"decal/ground/pollution_{k}", slot["pollution"],
                                           [flat_grid(pollution["span"], quarter(k), pollution["lift"])])]))
    for k in range(crater["variants"]):
        groups.append((f"crater_{k}", [(f"decal/ground/crater_{k}", slot["craters"],
                                        [flat_grid(crater["span"], quarter(k), crater["lift"])])]))
    for size, factor in enumerate(ruins["sizes"]):
        groups.append((f"ruins_{size}", [(f"decal/ground/ruins_{size}", slot["rubble"],
                                          [flat_grid(ruins["span"] * factor, (0, 0, 1, 1), ruins["lift"])])]))
    return textures, groups


def build(output):
    groups, textures, consumed = plan()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, texture in enumerate(textures):
        target = output / f"textures/base_{i}.dds"; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(SOURCE / texture, target); paths.append(target.relative_to(output).as_posix())
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
    # Ground states share the bundle's eight texture slots after the site art.
    ground = ground_groups(len(paths)) if len(paths) + 3 <= 8 else None
    if ground is not None:
        from Renderer.tools.asset_compiler.ground_state_composer import encode_bc3
        composed, ground_roles = ground
        for name, rgba in composed:
            target = output / f"textures/ground_{name}.dds"
            target.write_bytes(encode_bc3(rgba)); paths.append(target.relative_to(output).as_posix())
        for role, parts in ground_roles:
            placements = []
            for asset_id, slot, meshes in parts:
                placements.append((len(assets), .5))
                assets.append(merged_asset(asset_id, slot, 0, meshes))
            payloads.append(group_payload(role, placements, 1.0)); counts[role] = len(parts)
    paths += [paths[0]] * (8-len(paths))
    bundle = bytearray(MAGIC) + struct.pack("<IIII", 1, 8, len(assets), len(payloads))
    for path in paths: bundle.extend(bundle_string(path))
    for part in assets + payloads: bundle.extend(part)
    (output / "sites.bin").write_bytes(bundle)
    (output / "manifest.json").write_text(json.dumps({"schema": "c3x.static_sites.v1", "groups": counts,
        "height_scale": HEIGHT_SCALE, "sizes": json.loads(LOOKS.read_text())["sizes"],
        "ground_states": ground is not None, "normals": "inverse_transpose", "neutral": True,
        "source_sha256": consumed}, indent=2)+"\n")
    return consumed
