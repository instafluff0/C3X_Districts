"""Compile neutral huts and camps into the generic feature-bundle format.

Inputs remain local normalized art. Runtime receives no source-game paths,
gameplay rules, or source-specific loader behavior. Night light follows the
tile buildings': faces lit by their emissive map's uv2 layout are their own
asset named for the glow slot, and torches, campfires and lamps ride in a
"lamps/<asset>" companion.
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
from Renderer.tools.asset_compiler import tile_buildings_runtime as tile_buildings

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


# A Civ VI night light or fire by its bone's name: the site lamp mode (0 a
# flame; 1-4 steady warm, white, red, green lamps; 5-8 the same blinking,
# q8_city_effect) and strength. The airstrip's lanterns burn FX_Glow_Lantern,
# the aerodrome's runway FX_Lights_Runway_White; its mast's
# FX_Light_Airport_Tower is taken for red obstruction lights (its colour is not
# decoded from VFX.blp). Torches burn smaller than campfires and braziers.
LAMP_LOOKS = (("fx_runwaylight", 2, 1.2), ("fx_light0", 7, 1.6))


def light_look(semantic: str, bone: str) -> tuple[int, float]:
    n = bone.lower()
    if semantic == "flame":
        return 0, .8 if "torch" in n else 1.3
    for prefix, mode, strength in LAMP_LOOKS:
        if n.startswith(prefix):
            return mode, strength
    band = 2 if "red" in n else 3 if "green" in n else 1 if "white" in n else 0
    return 1 + band + (4 if "blink" in n else 0), 1.4


def plan():
    """The huts' and camp's parts, their textures, the bytes read, and their night
    light: each part is (mesh, texture, lit), lit faces being those Civ VI's
    emissive map lights by uv2 (build_mine_runtime.bake_light); "lights" holds each
    texture's baked light laid out by uv0, "fires" each role's Civ VI torches,
    campfires and lamps (x, y, z, mode, strength) in its parts' space."""
    from Renderer.tools.asset_compiler.build_mine_runtime import bake_light, bc1_rgb
    import numpy as np
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
            emissive = material.get("channels", {}).get("emissive")
            if base:
                transformed = transform_mesh(mesh, matrix, height_scale)
                # Source ground decals do not replace C3X's authoritative ground.
                if max(v["position"][2] for v in transformed["vertices"]) >= .006*height_scale:
                    parts.append([transformed, base["texture"],
                                  emissive["texture"] if isinstance(emissive, dict) else None, []])
        skeletons = [_skeleton_worlds(document(SOURCE / name)) for name in landmark["components"]["skeletons"]]
        bones = [document(SOURCE / name)["bones"] for name in landmark["components"]["skeletons"]]
        # The component's torches, campfires and lamps go with its first part.
        for point in landmark["attachment_points"]:
            if parts and point.get("semantic") in ("flame", "night_light") and \
                    point.get("state_hint") == "operational" and point.get("skeleton") is not None:
                world = skeletons[point["skeleton"]][point["bone"]]
                x, y, z = _point([world[12], world[13], world[14]], matrix, True)
                parts[0][3].append((x, y, z * height_scale,
                                    *light_look(point["semantic"], bones[point["skeleton"]][point["bone"]].get("name", ""))))
        for point in landmark["attachment_points"]:
            if point["binding_status"] == "resolved":
                child = _matrix_multiply(skeletons[point["skeleton"]][point["bone"]], matrix)
                parts.extend(collect(point["component_asset"], child, stack+(key,), height_scale))
        return parts
    groups, lights, fires, maps = {}, {}, {}, {}
    for role, key in roots:
        size = looks["sizes"]["camp" if role == "camp" else "hut"]
        body, spread = size["body"], size["spread"]
        groups[role], fires[role] = [], []
        for mesh, texture, emissive, points in collect(key, height_scale=size["height_scale"]):
            xs = [v["position"][0] for v in mesh["vertices"]]; ys = [v["position"][1] for v in mesh["vertices"]]
            cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
            fires[role] += [((x - cx) * body + cx * spread, (y - cy) * body + cy * spread, z * body, mode, strength)
                            for x, y, z, mode, strength in points]
            mesh = resize(mesh, body, spread)
            if emissive is None or any("uv2" not in v for v in mesh["vertices"]):
                groups[role].append((mesh, texture, False))
                continue
            if emissive not in maps:
                maps[emissive] = bc1_rgb(SOURCE / emissive)
                read(SOURCE / emissive)
            dark, lit, light = bake_light([mesh], maps[emissive])
            groups[role] += [(part, texture, False) for part in dark] + [(part, texture, True) for part in lit]
            if lit:
                held = lights.get(texture)
                if held is not None and held.shape != light.shape:
                    raise ValueError(f"{texture} has emissive maps of two sizes")
                lights[texture] = light if held is None else np.maximum(held, light)
    textures = sorted({texture for parts in groups.values() for _, texture, _ in parts})
    if not 1 <= len(textures) <= 8 or any(not parts for parts in groups.values()):
        raise ValueError("Site material/body count exceeds the feature binding")
    for texture in textures:
        read(SOURCE / texture)
    for name in ("Renderer/preview/render_improvement_sheet.py", "Renderer/tools/asset_compiler/build_mine_runtime.py",
                 "Renderer/tools/asset_compiler/build_site_runtime.py", "Renderer/tools/asset_compiler/ground_state_composer.py",
                 "Renderer/tools/asset_compiler/tile_buildings_runtime.py"):
        read(ROOT / name)
    ground_sources(read)
    tile_buildings.load(read)
    return groups, textures, consumed, {"lights": lights, "fires": fires}


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


def lamp_asset(name: str, texture: int, lamps: list) -> bytes:
    """A companion asset "lamps/<mesh>": one degenerate triangle per night light
    or fire at its point, uv (mode, strength) (object_compiler.h tile_building_lamps)."""
    count = len(lamps) * 3
    payload = bytearray(bundle_string("lamps/" + name) + struct.pack("<III", texture, count, count))
    for x, y, z, mode, strength in lamps:
        payload += struct.pack("<8f", x, y, z, 0.0, 0.0, 1.0, float(mode), strength) * 3
    return bytes(payload + struct.pack(f"<{count}I", *range(count)))


def build(output, frozen=tile_buildings.FROZEN):
    groups, textures, consumed, night = plan()
    output.mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir(exist_ok=True)
    paths = []
    # With Civ III's tile buildings, their atlases take the first slots and
    # hold the hut and camp textures (tile_buildings_runtime).
    buildings = tile_buildings.load(lambda path: path.read_bytes(), frozen)
    if buildings:
        for k, atlas in enumerate(buildings["manifest"]["atlases"]):
            target = output / f"textures/atlas_{k}.dds"
            shutil.copyfile(frozen / atlas, target); paths.append(target.relative_to(output).as_posix())
        cells = {texture: tile_buildings.cell(buildings, (SOURCE / texture).read_bytes()) for texture in textures}
    else:
        for i, texture in enumerate(textures):
            target = output / f"textures/base_{i}.dds"
            shutil.copyfile(SOURCE / texture, target); paths.append(target.relative_to(output).as_posix())
    assets, payloads = [], []
    counts = {}
    for role, parts in groups.items():
        # Faces Civ VI lights at night form their own asset, named for its atlas's
        # glow slot (":glow"), where the tile buildings bring that atlas a glow map.
        merged = defaultdict(list)
        glows = buildings["manifest"].get("glows", {}) if buildings else {}
        for mesh, texture, lit in parts:
            if buildings:
                merged[cells[texture][0], lit and str(cells[texture][0]) in glows].append(
                    tile_buildings.atlas_mesh(mesh, cells[texture]))
            else:
                merged[textures.index(texture), False].append(mesh)
        placements = []
        for (texture, lit), meshes in sorted(merged.items()):
            placements.append((len(assets), .5))
            assets.append(merged_asset(role + "_lit:glow" if lit else role, texture, 0, meshes))
        if night["fires"].get(role):
            owner = sorted(merged)[0]
            assets.append(lamp_asset(f"{role}:e0", owner[0], night["fires"][role]))
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
    if buildings:
        # Unused slots share one 4x4 placeholder: each slot is its own GPU
        # texture, so repeating an atlas would load it again for nothing.
        from Renderer.tools.asset_compiler.build_resource_compositions import dds_header
        (output / "textures/unused.dds").write_bytes(dds_header(4, 4, 1, 72, 8) + bytes(8))
        # Glow maps (night light) take the slots after the ground states.
        slots = {}
        for k, glow in sorted(buildings["manifest"].get("glows", {}).items()):
            target = output / f"textures/glow_{k}.dds"
            shutil.copyfile(frozen / glow, target); paths.append(target.relative_to(output).as_posix())
            slots[int(k)] = len(paths) - 1
        if len(paths) > 8:
            raise ValueError(f"{len(paths)} textures exceed the site pack's eight slots")
        paths += ["textures/unused.dds"] * (8-len(paths))
        offset = len(assets)
        assets = [tile_buildings.lit(asset, slots) for asset in assets + buildings["assets"]]
    else:
        paths += [paths[0]] * (8-len(paths))
    bundle = bytearray(MAGIC) + struct.pack("<IIII", 2 if buildings else 1, 8, len(assets), len(payloads))
    for path in paths: bundle.extend(bundle_string(path))
    for part in assets + payloads: bundle.extend(part)
    if buildings:
        bundle.extend(tile_buildings.tail(buildings, offset))
    (output / "sites.bin").write_bytes(bundle)
    (output / "manifest.json").write_text(json.dumps({"schema": "c3x.static_sites.v2" if buildings else "c3x.static_sites.v1",
        "groups": counts, "height_scale": HEIGHT_SCALE, "sizes": json.loads(LOOKS.read_text())["sizes"],
        "ground_states": ground is not None, "normals": "inverse_transpose", "neutral": True,
        "tile_buildings": [name for name, _ in buildings["compositions"]] if buildings else [],
        "source_sha256": consumed}, indent=2)+"\n")
    return consumed
