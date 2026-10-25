"""Freeze the user's tile-building picks for the production site pack.

Bakes plan.picks (fortress, the barricade ring's pieces, airfield, radar,
outpost, victory point and colony pieces by era) through the district pipeline into
Renderer/packs/TileBuildingsFrozen: a version-2 feature bundle of compositions
over atlases sized to what they hold. The hut and camp textures of the
production site pack join the first BC1 atlas, so the site pack keeps its eight
texture slots (atlases, then the three ground-state decals).
Renderer/tools/asset_compiler/build_site_runtime.py merges the frozen pack into
TileSitesRuntime (the "tile-sites" asset job), as the city pack recomposes
CityCompositionFrozen.

    python3 Renderer/lab/studies/tile_buildings/sitepack.py [--lab-pack TileSitesLab]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.tile_buildings import study  # noqa: E402,F401  (points the borrowed pipeline here)
from Renderer.lab.studies.tile_buildings import plan  # noqa: E402
from Renderer.lab.studies.districts import kits  # noqa: E402
from Renderer.tools.asset_compiler import build_site_runtime as sites  # noqa: E402
from Renderer.tools.asset_compiler.tile_buildings_runtime import FROZEN  # noqa: E402


def whiten(compiler, record: dict) -> None:
    """Civ III's victory marker is white: the obelisk's colour becomes a pale,
    near-white stone keeping its shading (BC1, new file; the material points to it)."""
    import numpy as np
    for _component, landmark, _transform in kits.components(compiler.manifest, compiler.root(record), kits.IDENTITY, []):
        for path in landmark["components"]["materials"]:
            material_path = kits.SOURCES / path
            material = json.loads(material_path.read_text())
            base = material.get("channels", {}).get("base_color")
            if not isinstance(base, dict) or "/white_" in base["texture"]:
                continue
            source = kits.SOURCES / base["texture"]
            rgb = kits._bc1_rgb(source).astype(float)
            luma = (rgb * np.array([.299, .587, .114])).sum(-1, keepdims=True)
            shade = .80 + .20 * (luma - luma.min()) / max(1.0, float(luma.max() - luma.min()))
            image = np.repeat(shade * 245.0, 3, -1) * np.array([1.0, .99, .96])
            side = 1 << (min(image.shape[:2]).bit_length() - 1)
            image = image[:side, :side]
            target = source.with_name("white_" + source.name)
            kits._write_bc1(np.clip(image, 0, 255), target)
            base["texture"] = str(target.relative_to(kits.SOURCES))
            material_path.write_text(json.dumps(material, indent=1, sort_keys=True) + "\n")



def pillage_fires(entries: set) -> dict:
    """Per Civ VI entry, the bones (FNV-1a of their names) whose effect triggers are
    all pillage effects ("PIL"): the aerodrome airport's fire burns only when
    pillaged though its bone is plainly named. Entries are found in the installed
    landmark packages; one that is not keeps all its fires."""
    import re
    from collections import defaultdict
    from Renderer.tools.asset_compiler import clutter_blp_extractor as cbe
    from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
    from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
    from Renderer.tools.asset_compiler.resource_pack_builder import fnv1a32
    names = {}
    for path in (DEFAULT_ASSETS_ROOT / "Base").rglob("*.blp"):
        if any(word in path.name.lower() for word in ("vfx", "light", "fx")):
            for name in re.findall(rb"[A-Za-z0-9_\-. ]{4,80}", path.read_bytes()):
                names[fnv1a32(name.decode())] = name.decode()
    packages = sorted((path for path in DEFAULT_ASSETS_ROOT.parent.rglob("*.blp") if "landmark" in str(path).lower()),
                      key=lambda path: ("/Base/" not in str(path), str(path)))
    found, result = {}, {}
    for path in packages:
        data = path.read_bytes()
        for entry in entries - set(found):
            if entry.encode() in data:
                found[entry] = path
    for entry, path in found.items():
        try:
            package = IndexedStaticPackage(path, entry)
            _landmark, _user, base = cbe.landmark_base_model(package)
        except ValueError:
            continue
        effects = defaultdict(set)
        for _offset, pointer in package.pointer_fields(base, "TimelineSet::Trigger"):
            raw = package.bytes_for(pointer)
            for index in range(len(raw) // 28):
                kind, _a, _start, _attachment, bone, name, _c = struct.unpack_from("<IIfIIII", raw, index * 28)
                if kind == 1:
                    effects[bone].add("pil" in names.get(name, "").lower())
        result[entry] = {bone for bone, pillage in effects.items() if all(pillage)}
    return result


def light_sources(compiler, cells: list) -> dict:
    """Civ VI's night light for the picks, written into the Lab sources before
    baking. Civ VI lays a building's emissive map out by uv2, not by its colour's
    uv0 (an atlas many faces share, by which kits squares the emissive map), so
    each emissive draw splits into its dark faces and the faces uv2 lights
    (build_mine_runtime.bake_light, a "_lit" geometry); one light texture per
    colour texture, laid out by uv0, holds them all. The import's own draws are
    kept and the split is redone on every freeze. Returns the night lights and
    fires (semantic night_light and flame points): mesh key, as kits.bake names
    it, to points in that mesh's space, each (x, y, z, mode, strength)."""
    import numpy as np
    from Renderer.tools.asset_compiler import build_mine_runtime as mine
    manifest = compiler.manifest
    seen, occurrences = {}, []
    for cell in cells:
        for part in cell["parts"]:
            try:
                root = compiler.root(part["record"])
            except ValueError:
                continue    # kits.bake reports it missing
            for asset_id, landmark, transform in kits.components(manifest, root, kits.IDENTITY, []):
                path = kits.SOURCES / manifest["assets"][asset_id]["landmark"]
                seen.setdefault(path, landmark)
                occurrences.append((cell, part, path, transform))
    documents = {}
    for path, document in seen.items():
        source = document.setdefault("source", {key: document[key] if key == "draw_bindings" else
                                                document["components"][key]
                                                for key in ("draw_bindings", "geometry", "materials")})
        document["draw_bindings"] = list(source["draw_bindings"])
        document["components"]["geometry"] = list(source["geometry"])
        document["components"]["materials"] = list(source["materials"])
        documents[path] = document
    # Each emissive draw's faces, dark and lit, and its colour texture's light.
    split, lights = {}, {}
    for path, document in documents.items():
        for binding in document["draw_bindings"]:
            geometry = document["components"]["geometry"][binding["geometry"]]
            material_path = document["components"]["materials"][binding["material"]]
            material = kits.load(kits.SOURCES / material_path)
            channels = material.get("channels", {})
            emissive, base = channels.get("emissive"), channels.get("base_color")
            if not isinstance(emissive, dict) or not isinstance(base, dict) or (geometry, material_path) in split:
                continue
            mesh = kits.load(kits.SOURCES / geometry)
            if any("uv2" not in vertex for vertex in mesh["vertices"]):
                raise ValueError(f"{geometry} has no uv2: re-import (study.py compile into an empty "
                                 f"{kits.SOURCES.name})")
            dark, lit, light = mine.bake_light([mesh], mine.bc1_rgb(kits.SOURCES / emissive["texture"]))
            split[geometry, material_path] = (dark, lit)
            if lit:
                held = lights.get(base["texture"])
                if held is not None and held.shape != light.shape:
                    raise ValueError(f"{base['texture']} has emissive maps of two sizes")
                lights[base["texture"]] = light if held is None else np.maximum(held, light)
    baked = {}
    for texture, light in lights.items():
        data = mine.bc1_dds(light)
        name = f"textures/night/light_{hashlib.sha256(data).hexdigest()[:16]}.dds"
        (kits.SOURCES / name).parent.mkdir(parents=True, exist_ok=True)
        (kits.SOURCES / name).write_bytes(data)
        baked[texture] = name
    for path, document in documents.items():
        components = document["components"]
        bindings = []
        for binding in document["draw_bindings"]:
            geometry = components["geometry"][binding["geometry"]]
            material_path = components["materials"][binding["material"]]
            if (geometry, material_path) not in split:
                bindings.append(binding)
                continue
            dark, lit = split[geometry, material_path]
            material = kits.load(kits.SOURCES / material_path)
            stem = material_path.removesuffix(".json")
            unlit = {**material, "channels": {k: v for k, v in material["channels"].items() if k != "emissive"}}
            unlit.pop("emissive", None)
            lit_material = {**material, "channels": {**material["channels"], "emissive": {
                **material["channels"]["emissive"], "texture": baked.get(material["channels"]["base_color"]["texture"], "")}}}
            for faces, suffix, body in ((dark, "_dark", unlit), (lit, "_lit", lit_material)):
                if not faces:
                    continue
                target = geometry.removesuffix(".json") + suffix + ".json"
                (kits.SOURCES / target).write_text(json.dumps(faces[0]) + "\n")
                (kits.SOURCES / f"{stem}{suffix}.json").write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
                components["geometry"].append(target)
                components["materials"].append(f"{stem}{suffix}.json")
                bindings.append({**binding, "geometry": len(components["geometry"]) - 1,
                                 "material": len(components["materials"]) - 1})
        document["draw_bindings"] = bindings
        path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    # Night lights and fires in the space of the first mesh their component
    # draws; fires Civ VI lights only when pillaged are left out.
    from Renderer.tools.asset_compiler.resource_pack_builder import fnv1a32
    entries = {path: manifest["assets"][asset_id].get("source_entry") for asset_id in manifest["assets"]
               for path in [kits.SOURCES / manifest["assets"][asset_id]["landmark"]] if path in documents}
    burning = {entries[path] for path, document in documents.items() if entries.get(path) and
               any(p.get("semantic") == "flame" for p in document["attachment_points"])}
    fires = pillage_fires(burning)
    lamps = {}
    for cell, part, path, transform in occurrences:
        document = documents[path]
        if part.get("merge") or part.get("ramp"):
            continue
        skeletons = [kits.load(kits.SOURCES / p) for p in document["components"]["skeletons"]]
        points = [(kits._skeleton_worlds(skeletons[p["skeleton"]])[p["bone"]], p["semantic"],
                   skeletons[p["skeleton"]]["bones"][p["bone"]].get("name", ""))
                  for p in document["attachment_points"] if p.get("semantic") in ("night_light", "flame")
                  and p.get("state_hint") == "operational" and p.get("skeleton") is not None]
        pillaged = fires.get(entries.get(path), set())
        points = [point for point in points if fnv1a32(point[2]) not in pillaged]
        owner = next((b for b in document["draw_bindings"] if cell.get("state", "worked") in b["states"] and
                      isinstance(kits.load(kits.SOURCES / document["components"]["materials"][b["material"]])
                                 .get("channels", {}).get("base_color"), dict)), None)
        if not points or owner is None:
            continue
        # The key and mesh space kits.bake gives this draw.
        geometry = document["components"]["geometry"][owner["geometry"]]
        material = kits.load(kits.SOURCES / document["components"]["materials"][owner["material"]])
        fitted = kits.yaw_scale(transform)
        linear = kits.IDENTITY if fitted else transform[:12] + [0.0, 0.0, 0.0, 1.0]
        key = f"district/{geometry.rsplit('/', 1)[-1].removesuffix('.json')}"
        if not fitted:
            key += "/" + format(abs(hash(tuple(round(v, 4) for v in linear))) % 16 ** 8, "08x")
        clip = part.get("clip_z")
        if clip is not None:
            key += f"/z{clip}"
        if part.get("squash"):
            key += f"/s{part['squash']}"
        key += kits._tint_key(part, material)
        if part.get("conform"):
            key = "city/walls/" + key
        found = []
        for world, semantic, bone in points:
            x, y, z = world[12], world[13], world[14]
            x, y, z = (x * linear[0] + y * linear[4] + z * linear[8], x * linear[1] + y * linear[5] + z * linear[9],
                       (x * linear[2] + y * linear[6] + z * linear[10]) * part.get("squash", 1.0))
            if clip is None or z <= clip:
                found.append((x, y, z, *sites.light_look(semantic, bone)))
        if found:
            lamps[key] = found
    return lamps


def night_assets(bundle: dict, lamps: dict) -> None:
    """Marks the bundle's lit meshes (":glow", which build_site_runtime turns into
    their glow atlas's slot, ":g<slot>") and gives each mesh with night lights a
    companion asset "lamps/<its name>": one degenerate triangle per lamp at its
    point in the mesh's space, uv (lamp mode, strength). Renderer64 places the
    lamps wherever the building's instance stands (object_compiler.h
    tile_building_lamps)."""
    import re
    names = {asset["name"] for asset in bundle["assets"]}
    missing = sorted(set(lamps) - names)
    if missing:
        raise ValueError(f"night lights for meshes kits.bake did not make (key drift?): {missing[:3]}")
    companions = []
    for asset in bundle["assets"]:
        key = asset["name"]
        if re.search(r"_lit(/|$)", key):
            asset["name"] += ":glow"
        if key in lamps:
            payload = b"".join(struct.pack("<8f", x, y, z, 0.0, 0.0, 1.0, float(mode), strength)
                               for x, y, z, mode, strength in lamps[key] for _ in range(3))
            count = len(lamps[key]) * 3
            payload += struct.pack(f"<{count}I", *range(count))
            companions.append({"name": "lamps/" + asset["name"], "texture": asset["texture"], "vertices": count,
                               "indices": count, "payload": payload,
                               "points": [point[:3] for point in lamps[key] for _ in range(3)]})
    bundle["assets"] += companions


def sized(kind: tuple, count: int) -> tuple:
    """The smallest square atlas of the kind that holds `count` 512 cells."""
    size = kind[0]
    while size > kits.rc.CELL and ((size // 2) // kits.rc.CELL) ** 2 >= count:
        size //= 2
    return (size, min(kind[1], (size // 4).bit_length()), *kind[2:])


def freeze(out: Path = FROZEN) -> dict:
    # The picks' night light is split here (light_sources), so kits.bake keeps
    # its source glow: no uv2 split or merged pieces of its own.
    catalog = {"id": "tb-picks", "title": "Tile buildings (picks)", "cells": plan.picks(), "night": "sources"}
    compiler = kits.Compiler()
    for cell in catalog["cells"]:
        if cell.get("whiten"):
            for part in cell["parts"]:
                whiten(compiler, part["record"])
    lamps = light_sources(compiler, catalog["cells"])
    _, textures, _, site_night = sites.plan()
    extra = [sites.SOURCE / texture for texture in textures]
    # A hut or camp texture's glow cell holds its baked night light (build_site_runtime.plan).
    def site_glow(path):
        size = kits.rc.dds(path)[0]
        light = site_night["lights"].get(path.relative_to(sites.SOURCE).as_posix())
        if light is None:
            return kits._dark_square(size, kits.OUT / "texture_cache")
        step = light.shape[0] // size
        image = light.reshape(size, step, size, step, 3).mean((1, 3)) if step > 1 else \
            light.repeat(size // light.shape[0], 0).repeat(size // light.shape[0], 1)
        target = kits.OUT / "texture_cache" / f"glow_site_{hashlib.sha256(image.tobytes()).hexdigest()[:16]}.dds"
        return target if target.is_file() else kits._write_bc1(image * 255, target)
    # The hut and camp textures take the next cells of the first BC1 atlas;
    # every atlas shrinks to the cells it holds. kits.bake then lays out each
    # colour atlas's glow maps (its cells in order): those repeat the colour
    # atlas's size and cells, hut and camp cells dark.
    original, placed, colour = kits.rc.pack_atlases, {}, []

    def pack(textures, kind, first=0):
        if any(Path(path).name.startswith("glow") for path in textures):
            size, levels, with_extra = colour.pop(0)
            return original(list(textures) + ([site_glow(path) for path in extra] if with_extra else []),
                            (size, levels, *kits.rc.MODEL_ATLAS[2:]), first)
        with_extra = kind == kits.rc.MODEL_ATLAS
        textures = list(textures) + (extra if with_extra else [])
        if len(textures) <= (kind[0] // kits.rc.CELL) ** 2:
            kind = sized(kind, len(textures))
        colour.append((kind[0], kind[1], with_extra))
        atlases, place = original(textures, kind, first)
        if not placed:   # the model atlases
            placed.update({path: place[path] for path in extra})
        return atlases, place
    kits.rc.pack_atlases = pack
    try:
        summary = kits.bake(catalog, compiler, kits.SCALE, False)
    finally:
        kits.rc.pack_atlases = original
    compiler.save()
    atlases = [f"textures/atlas_{k}.dds" for k in range(summary["atlases"])]
    if len(atlases) + 3 > 8:
        raise ValueError(f"{len(atlases)} atlases leave no slots for the three ground-state decals")
    if out.exists():
        shutil.rmtree(out)
    (out / "textures").mkdir(parents=True)
    for atlas in atlases:
        shutil.copy2(kits.CATALOG / atlas, out / atlas)
    # The ring's wall sections take the terrain under every vertex, as the
    # city walls do (objects::terrain_wall_asset): their assets are "walls/".
    # The runtime stretches a section along its mesh's x, so a mesh longer in
    # y turns a quarter (its compositions turn back, keeping their look).
    from Renderer.lab.studies.tile_buildings import colonies
    bundle = colonies.read_pack(kits.CATALOG / "resource_runtime.bin")
    walls = {item[0] for name, variants in bundle["compositions"] if name.startswith("wall_")
             for _, items in variants for item in items}
    turned = set()
    for index in walls:
        asset = bundle["assets"][index]
        asset["name"] = "walls/" + asset["name"].split("/", 1)[-1]
        xs = [point[0] for point in asset["points"]]
        ys = [point[1] for point in asset["points"]]
        if max(ys) - min(ys) > max(xs) - min(xs):
            payload = bytearray(asset["payload"])
            for k in range(asset["vertices"]):
                x, y, z, nx, ny, nz, u, v = struct.unpack_from("<8f", payload, k * 32)
                struct.pack_into("<8f", payload, k * 32, y, -x, z, ny, -nx, nz, u, v)
            asset["payload"] = bytes(payload)
            turned.add(index)
    for name, variants in bundle["compositions"]:
        for _, items in variants:
            for item in items:
                if item[0] in turned:
                    item[3] += math.pi / 2
    night_assets(bundle, lamps)
    colonies.write_pack(bundle, out / "tile_buildings.bin")
    # Night light: emissive_<k>.dds repeats atlas_<k>.dds's cells (kits.bake writes
    # only atlases with a lit cell) at a quarter of its resolution, its top two
    # mips dropped: lit windows and lamps need no more at map scale.
    glows = {}
    for k in range(summary["atlases"]):
        source = kits.CATALOG / f"textures/emissive_{k}.dds"
        if source.is_file():
            width, height, mips, dxgi, payload = kits.rc.dds(source)
            offset, wide, high = kits.rc.mip_offsets(width, height, mips, 8)[2]
            (out / f"textures/emissive_{k}.dds").write_bytes(
                kits.rc.dds_header(width >> 2, height >> 2, mips - 2, dxgi, wide * high * 8) + payload[offset:])
            glows[str(k)] = f"textures/emissive_{k}.dds"
    cells = {hashlib.sha256(path.read_bytes()).hexdigest(): [*placed[path], kits.rc.dds(path)[0]] for path in extra}
    report = {"schema": "c3x.tile_buildings_frozen.v1", "bundle": "tile_buildings.bin", "atlases": atlases, "glows": glows,
              "site_cells": cells, "source": "Renderer/lab/studies/tile_buildings plan.picks",
              "compositions": list(summary["cells"]), "cells": summary["cells"]}
    (out / "manifest.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lab-pack", help="also build this Lab site pack from the frozen picks (for C3X_RENDERER_SITE_PACK)")
    parser.add_argument("--frozen", help="freeze into this pack instead of production's TileBuildingsFrozen")
    args = parser.parse_args()
    frozen = ROOT / "Renderer/packs" / args.frozen if args.frozen else FROZEN
    report = freeze(frozen)
    print(json.dumps({k: v for k, v in report.items() if k not in ("cells", "site_cells")}, indent=1))
    if args.lab_pack:
        sites.build(ROOT / "Renderer/packs" / args.lab_pack, frozen)
        print("built", args.lab_pack)
