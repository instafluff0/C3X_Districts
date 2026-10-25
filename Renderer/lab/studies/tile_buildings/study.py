#!/usr/bin/env python3
"""Civ III tile buildings as Civ VI kits on the Renderer64 map (Lab only).

Fortresses, barricades, airfields, outposts, radar towers and victory points
have no custom-renderer art yet: the custom renderer skips Civ III's own tile
plane, so they do not show at all. This study proposes Civ VI stand-ins. It
drives the district survey's compile/bake/render pipeline
(`Renderer/lab/studies/districts/kits.py`, unchanged) with its own Lab packs
(`TileBuildingKitSourcesLab`, `TileBuildingKitCatalogLab`) and output folder
(`Renderer/lab/out/tile_buildings/`). Production packs are unchanged.

    python3 Renderer/lab/studies/tile_buildings/study.py refs      # Civ III's sprites
    python3 Renderer/lab/studies/tile_buildings/study.py compile
    python3 Renderer/lab/studies/tile_buildings/study.py render [--catalog tb-fortress ...]
    python3 Renderer/lab/studies/tile_buildings/study.py crops
"""
from __future__ import annotations

import argparse
import functools
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.districts import kits  # noqa: E402

OUT = ROOT / "Renderer/lab/out/tile_buildings"
# Point the borrowed pipeline at this study's packs and folder before any use.
kits.SOURCES = ROOT / "Renderer/packs/TileBuildingKitSourcesLab"
kits.CATALOG = ROOT / "Renderer/packs/TileBuildingKitCatalogLab"
kits.OUT = OUT
OUT.mkdir(parents=True, exist_ok=True)
# A matching DLL and preview pair, copied once from the shared candidate, so a
# peer's API change to the shared preview cannot break later frames.
PINNED = OUT / "pinned"
kits.night_renderer = lambda: (PINNED / "C3XRenderer.dll", PINNED / "native_preview.exe", {}) \
    if (PINNED / "native_preview.exe").is_file() else None
_bake = kits.bake


def bake(catalog, compiler, *args, **kwargs):
    """A catalog of colonies on resources also carries the production resources."""
    summary = _bake(catalog, compiler, *args, **kwargs)
    if catalog.get("resources"):
        from Renderer.lab.studies.tile_buildings import colonies
        summary["merge"] = colonies.merge(kits.CATALOG, catalog["cells"])
    return summary


kits.bake = bake
# Imports keep Civ VI's uv1/uv2: its emissive maps are laid out by uv2, which
# sitepack.light_sources needs to place the buildings' night light.
_lab_compiler = kits._lab_compiler
kits._lab_compiler = lambda: functools.partial(_lab_compiler(), auxiliary_uvs=True)
census = ROOT / "Renderer/lab/out/districts/census.json"
if census.is_file() and not (OUT / "census.json").is_file():
    shutil.copy2(census, OUT / "census.json")

CIV3 = ROOT.parent                      # the Conquests folder
SHEETS = {
    "terrain": CIV3 / "Art/Terrain/TerrainBuildings.PCX",
    "air": CIV3.parent / "civ3PTW/Art/Terrain/x_airfields and detect.PCX",
    "air_shadow": CIV3.parent / "civ3PTW/Art/Terrain/X_AIRfields and detect_shadow.PCX",
    "victory": CIV3 / "Art/Terrain/x_victory.pcx",
}
# (name, sheet, x, y, width, height): Map_Renderer::load_images slices.
SPRITES = [(f"fortress_{era}", "terrain", 0, 64 * era, 128, 64) for era in range(4)] + \
          [(f"colony_{era}", "terrain", 128, 64 * era, 128, 64) for era in range(4)] + \
          [(f"barricade_{era}", "terrain", 384, 64 * era, 128, 64) for era in range(4)] + \
          [(f"airfield_{k}", "air", 128 * k, 0, 128, 64) for k in range(2)] + \
          [(f"outpost_{k}", "air", 128 * k, 64, 128, 128) for k in range(3)] + \
          [("radar", "air", 0, 192, 128, 128), ("victory", "victory", 0, 0, 128, 64)]
GROUND = (112, 128, 70)


def refs() -> None:
    """Civ III's own sprites on a plain grass colour, at 2x (nearest)."""
    from Renderer.inventory.civ3_art_inventory import decode_pcx_rgb
    from Renderer.lab.studies.resources.audit import write_png
    folder = OUT / "civ3"
    folder.mkdir(parents=True, exist_ok=True)
    decoded = {key: decode_pcx_rgb(path) for key, path in SHEETS.items() if path.is_file()}
    for name, sheet, x0, y0, width, height in SPRITES:
        if sheet not in decoded:
            continue
        w, _, rgb = decoded[sheet]
        rows = []
        for y in range(y0, y0 + height):
            row = bytearray()
            for x in range(x0, x0 + width):
                pixel = tuple(rgb[(y * w + x) * 3:(y * w + x) * 3 + 3])
                # Magenta and green are Civ III's transparent keys (sheets vary a little).
                r, g, b = pixel
                key = (r > 240 and g < 16 and b > 240) or (r < 16 and g > 240 and b < 16)
                colour = GROUND if key else pixel
                row += bytes(colour) * 2
            rows += [bytes(row), bytes(row)]
        write_png(folder / f"{name}.png", width * 2, height * 2, rows)
    print(folder)


def adopt_children(compiler) -> int:
    """A DLC landmark (the Alcázar) names children held in its own package, while
    the borrowed compiler looks for them in landmarks/tilebases: retry each such
    child in its root's package and attach it."""
    import json
    adopted = 0
    for key, root in list(compiler.manifest["roots"].items()):
        content, package, _entry = key.split("|")
        stack = [root]
        while stack:
            asset = compiler.manifest["assets"].get(stack.pop())
            if not asset:
                continue
            path = kits.SOURCES / asset["landmark"]
            document = json.loads(path.read_text())
            changed = False
            for point in document["attachment_points"]:
                failure = point.get("compile_failure", "")
                if point.get("binding_status") == "component_compile_unresolved" and \
                        failure.startswith("unresolved child "):
                    try:
                        point["component_asset"] = compiler.root(kits.raw(content, package,
                                                                          failure.removeprefix("unresolved child ")))
                    except ValueError:
                        continue
                    point["binding_status"] = "resolved"
                    point.pop("compile_failure")
                    changed, adopted = True, adopted + 1
                if point.get("binding_status") == "resolved" and point.get("component_asset"):
                    stack.append(point["component_asset"])
            if changed:
                path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    return adopted


SAVE = ROOT / "Renderer/.cache/frame-delivery-step/input-1498AD.SAV"
SCENE_CENTRE = (89, 69)   # the Japanese barricade line, save tiles (83,71)..(93,67)
# Lab-only overlay bits read by native_preview: colony, victory point, era.
COLONY_BIT, VICTORY_BIT = 0x800, 0x1000
FORTRESS, BARRICADE, AIRFIELD, RADAR, OUTPOST = 0x10, 0x10000000, 0x20000000, 0x40000000, 0x80000000
# (family bits, era, preferred terrains, resource, on a river) placed on free land
# near the line; the river and mountain cases check the buildings' ground.
PLACED = [(FORTRESS, 0, (2, 1, 0), None, False), (FORTRESS, 1, (7,), None, False),
          (FORTRESS, 2, (0, 1, 2), None, False), (FORTRESS, 3, (0, 1, 2), None, False),
          (AIRFIELD, 2, (0, 1, 2), None, False), (AIRFIELD, 3, (0, 1, 2), None, False),
          (OUTPOST, 1, (2, 1, 0), None, False), (OUTPOST, 3, (0, 1, 2), None, False),
          (RADAR, 3, (0, 1, 2), None, False), (VICTORY_BIT, 1, (2, 1, 0), None, False),
          (COLONY_BIT, 0, (2, 1), "Horses", False), (COLONY_BIT, 1, (0, 1, 2), "Iron", False),
          (COLONY_BIT, 3, (0,), "Oil", False), (COLONY_BIT, 2, (2, 1, 0), "Cattle", True),
          (FORTRESS, 1, (6,), None, False), (OUTPOST, 2, (5,), None, False), (RADAR, 2, (2, 1, 0), None, True)]


def scene(folder: Path):
    """The 1498 save around its barricade line (era 1, as Japan's medieval
    line) with every other family placed on free land nearby. Returns the
    scene CSV, its cities and the Lab resource list."""
    import json
    import subprocess
    folder.mkdir(parents=True, exist_ok=True)
    csv, cities = folder / "scene.csv", folder / "cities.json"
    subprocess.run(["node", "Renderer/lab/studies/roads/save_window.js", str(SAVE), str(csv), str(cities),
                    str(SCENE_CENTRE[0]), str(SCENE_CENTRE[1])], cwd=ROOT, check=True, capture_output=True)
    rows = csv.read_text().splitlines()
    tiles = {}
    for row in rows[1:]:
        x, y, base, real, bonus, overlays, river = (int(v) for v in row.split(","))
        tiles[(x, y)] = [base, real, bonus, overlays, river]
    towns = {(c["x"], c["y"]) for c in json.loads(cities.read_text())}
    taken = set(towns) | {xy for xy, t in tiles.items() if t[3] & (FORTRESS | BARRICADE)}
    for xy, t in tiles.items():
        if t[3] & BARRICADE:
            t[3] |= 1 << 24              # the line in the Medieval era
    resources = []
    for bits, era, terrains, resource, river in PLACED:
        # The nearest free land tile of a preferred terrain, away from cities and the line.
        # River and hill tiles are scarcer: these cases look a little wider and closer.
        wide = river or terrains == (5,)
        free = [xy for xy, t in tiles.items() if t[1] in terrains and bool(t[4]) == river and
                (4 if wide else 6) <= xy[0] <= (28 if wide else 26) and (6 if wide else 9) <= xy[1] <= (27 if wide else 25)
                and all(abs(xy[0] - o[0]) + abs(xy[1] - o[1]) > (1 if wide else 2) for o in taken)]
        if not free:
            continue
        xy = min(free, key=lambda p: (abs(p[0] - 16) + abs(p[1] - 18), p))
        tiles[xy][3] = (tiles[xy][3] & ~0x1f) | bits | era << 24
        taken.add(xy)
        if resource:
            resources.append(f"{xy[0] - 16},{xy[1] - 16},{resource}")
    csv.write_text(rows[0] + "\n" + "\n".join(f"{x},{y},{','.join(str(v) for v in t)}" for (x, y), t in tiles.items()) + "\n")
    return csv, json.loads(cities.read_text()), ";".join(resources)


def render_scene(pack: str = "", views=("128:16:17", "256:15:16", "256:19:13", "256:12:23", "256:17:22",
                                       "256:25:23")) -> list:
    """Lab frames of the scene through the production site path (the current
    candidate DLL; `pack` is a C3X_RENDERER_SITE_PACK, empty for production)."""
    import shutil
    from Renderer import renderer
    csv, cities, resources = scene(OUT / "scene")
    renderer.scene = lambda category, case, destination, *, world_size=32: shutil.copyfile(csv, destination)
    frames = []
    for view in views:
        zoom, cx, cy = (int(v) for v in view.split(":"))
        listed = ";".join(f"{int(r.split(',')[0]) + 16 - cx},{int(r.split(',')[1]) + 16 - cy},{r.split(',', 2)[2]}"
                          for r in resources.split(";") if r)
        env = {"C3X_LAB_TILE_OVERLAYS": "1", "C3X_RENDERER_SITE_PACK": pack, "C3X_LAB_TILE_RESOURCES": listed,
               "C3X_LAB_TILE_CITIES": ";".join(f"{c['x'] - cx},{c['y'] - cy},0,2,{0 if (c.get('pop') or 0) < 7 else 1},0,0"
                                               for c in cities)}
        target = OUT / "scene" / f"z{zoom}_{cx}_{cy}_{pack or 'production'}"
        renderer.native_render("infrastructure", "save-window", 12, zoom, target, center=(cx, cy), extra_env=env)
        frames.append(target)
    return frames


def main() -> int:
    from Renderer.lab.studies.tile_buildings import plan
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("refs", "compile", "render", "crops", "scene"))
    parser.add_argument("--pack", default="", help="scene: a Lab site pack (C3X_RENDERER_SITE_PACK)")
    parser.add_argument("--catalog", action="append")
    parser.add_argument("--zoom", action="append", type=int)
    args = parser.parse_args()
    if args.action == "refs":
        refs()
        return 0
    if args.action == "scene":
        for frame in render_scene(args.pack):
            print(frame, flush=True)
        return 0
    catalogs = plan.catalogs()
    if args.catalog:
        unknown = set(args.catalog) - set(catalogs)
        if unknown:
            parser.error(f"unknown catalogs {sorted(unknown)}")
    zooms = tuple(args.zoom or (128, 256))
    if args.action == "compile":
        compiler = kits.Compiler()
        for key, spec in catalogs.items():
            if args.catalog and key not in args.catalog:
                continue
            for cell in spec["cells"]:
                for part in cell["parts"]:
                    try:
                        compiler.root(part["record"])
                    except ValueError as exc:
                        print(f"{key} {cell['name']}: {exc}", file=sys.stderr)
            compiler.save()
            print(key, "compiled", flush=True)
        print("children adopted from their roots' packages:", adopt_children(compiler))
        compiler.save()
    elif args.action == "render":
        kits.render(catalogs, zooms, args.catalog)
    else:
        kits.crops(catalogs, zooms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
