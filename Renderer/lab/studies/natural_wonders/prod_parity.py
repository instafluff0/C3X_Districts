#!/usr/bin/env python3
"""Render natural wonders through the production port (prod_tree.py) for Lab parity.

The scene is fed the way the game will feed it: the wonder tile keeps its C3X host
terrain, BIQ bonus 0x40000000 marks it and C3X_LAB_NATURAL_WONDER names it in its
tile record, and Renderer64 resolves the kit, its relief slot and its bodies from
Renderer/packs/NaturalWonderRuntime. Crops use the study's cell and context framing,
beside the approved study frames (crops/{key}-cell-stamp[-season]-z256.png).

    python3 Renderer/lab/studies/natural_wonders/prod_parity.py [--wonder KEY ...] [--season N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mountains import private_tree
from Renderer.lab.studies.natural_wonders import kits, prod_tree

OUT = kits.OUT / "parity"
MARKER = 0x40000000
DEFAULT = ("yellowstone", "angel_falls", "moraine_lake", "ha_long_bay", "mount_fuji")
LAB_HOST = False                                            # --lab-host: the study's scene (stamped tile real 6)
SAVE_SCENES: dict = {}                                      # --save: {key: scene tiles} from a game's save
PACK_SHADERS = False                                        # --pack-shaders: the game's runtime shader pack


def save_scenes(path: Path) -> dict:
    """{kit key: {(dx, dy): (base, real, river, bonus)}} for every natural wonder a C3X save
    placed: the wonder's real neighbourhood (terrain and river edges) in the study's window.
    Reads an uncompressed Conquests save (an autosave); the map is taken as 100 wide and
    wrapping east-west, as the scripted new game makes it."""
    import struct
    from Renderer.inventory.natural_wonder_mapping_inventory import DEFAULT_CONFIG, parse_natural_wonder_config
    data, width = path.read_bytes(), 100
    names = [item["name"] for item in parse_natural_wonder_config(DEFAULT_CONFIG)]
    keys = {item["name"]: key for key, item in kits.definitions().items()}
    terrain, rivers, at = [], [], 0
    while (at := data.find(b"TILE", at)) >= 0:
        size = struct.unpack_from("<i", data, at + 4)[0]
        if size == 36:
            rivers.append(data[at + 8])
        elif size == 12:
            terrain.append(data[at + 13])
        at += 4
    if len(terrain) != len(rivers) or not terrain:
        raise ValueError(f"{path.name}: unexpected tile sections")
    at = data.index(b"natural_wonder_districts") + len("natural_wonder_districts") + 4
    count = struct.unpack_from("<i", data, at)[0]
    scenes = {}
    for n in range(count):
        x, y, wonder = struct.unpack_from("<3i", data, at + 4 + 12 * n)
        key = keys.get(names[wonder]) if 0 <= wonder < len(names) else None
        if key is None:
            continue
        index = (x + y * width) // 2                        # Civ III resolves an odd x + y here too
        y = 2 * index // width
        x = 2 * index - y * width + (y & 1)
        tiles = {}
        for dy in range(-6, 7):
            for dx in range(-8, 9):
                ty = y + dy
                if (dx + dy) % 2 or not 0 <= ty < len(terrain) * 2 // width:
                    continue
                i = ((x + dx) % width + ty * width) // 2
                tiles[(dx, dy)] = (terrain[i] & 15, terrain[i] >> 4, rivers[i], 0)
        scenes[key] = tiles
    return scenes


class ProductionCases(kits.Cases):
    """The study's scenes with production facts: host terrain, a marked and named tile, no Lab bodies."""

    def terrain(self, category, case, dx, dy, base, real):
        key = self.key(case)
        tile = (SAVE_SCENES.get(key) or kits.scene_tiles(key)).get((dx, dy))
        if tile is None:
            return 2, 2, 0, 0, 0
        base, real, river, bonus = tile
        if (dx, dy) == (0, 0):
            bonus |= MARKER
            if LAB_HOST and kits.stamped(key):
                real = 6                                    # the study's host: a mountain over the ground
        return base, real, river, bonus, 0

    def resources(self, category, case):
        return ""


def render(keys, season: int, zoom: int = 256) -> dict:
    from Renderer import renderer
    from Renderer.lab.studies.resources import audit
    private_tree.OUT = kits.OUT
    tree = private_tree.tree(prod_tree.NAME)
    dll = tree / "Renderer/native/build/candidate/C3XRenderer.dll"
    preview_source = tree / "Renderer/lab/.cache/native_preview.exe"
    if not dll.is_file() or not preview_source.is_file():
        raise SystemExit("build the production tree first: prod_tree.py prepare && prod_tree.py build")
    preview = OUT / "prod_preview.exe"
    OUT.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(preview_source.read_bytes())
    shaders = ROOT / "Renderer/packs/Renderer64ResidentRuntime" if PACK_SHADERS else tree if season == 0 else season_root(tree, season)
    cases = kits.install(list(keys))
    production = ProductionCases(list(keys))
    original = renderer.lab_tile_cases
    renderer.lab_tile_cases = lambda category, case: production if production.applies(category, case) else original(category, case)
    report = {}
    tag = (f"-{kits.SEASON_NAMES[season]}" if season else "") + ("-labhost" if LAB_HOST else "") + ("-save" if SAVE_SCENES else "") + ("-pack" if PACK_SHADERS else "")
    for key in keys:
        target = OUT / f"{key}{tag}-z{zoom}"
        env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": "..\\..\\" + renderer.relative(shaders).replace("/", "\\"),
               "C3X_LAB_NATURAL_WONDER": kits.definitions()[key]["name"],
               "C3X_RENDERER_PREVIEW_SEASON": str(season), "C3X_LAB_EFFECT_TIME": str(kits.EFFECT_TIME)}
        for attempt in range(2):
            kits._vm_idle()
            try:
                renderer.native_render("resources", f"native-nw-{key}", 12, zoom, target,
                                       candidate=dll, preview=preview, extra_env=env)
                break
            except ValueError as exc:
                report[key] = f"render failed: {exc}"
        else:
            print("FAILED", key, report[key], flush=True)
            continue
        bmp = target / f"native-nw-{key}-h12-z{zoom}.bmp"
        image = audit.read_bmp(bmp)
        cx, cy = image[0] // 2, image[1] // 2
        cell = audit.crop(image, cx - zoom // 2, cy - zoom * 7 // 16, zoom, zoom * 11 // 16)
        audit.write_png(OUT / f"{key}-cell-prod{tag}-z{zoom}.png", *cell)
        context = audit.crop(image, cx - zoom * 3 // 2, cy - zoom, zoom * 3, zoom * 15 // 8)
        audit.write_png(OUT / f"{key}-context-prod{tag}-z{zoom}.png", *context)
        log = (target / "native.log").read_text(errors="replace")
        report[key] = {"rendered": True, "wonder_bodies": "wonder" in log}
        print(key, json.dumps(report[key]), flush=True)
    renderer.lab_tile_cases = original
    (OUT / f"report{tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def season_root(tree: Path, season: int) -> Path:
    """The tree's runtime shaders with the tree's (patched) season overlay pinned to one season,
    as the study pins it."""
    import importlib.util
    path = tree / "Renderer/lab/shared/seasons/seasons.py"
    spec = importlib.util.spec_from_file_location("prod_tree_seasons", path)
    seasons = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = seasons
    spec.loader.exec_module(seasons)
    variant = f"prod-s{season}"
    private_tree.variant_root(prod_tree.NAME, variant, [])
    root = kits.OUT / "ranges" / f"root-{variant}"
    for relative in seasons.TARGETS:
        path = root / relative
        if not path.is_file():
            continue
        text = seasons.apply(relative, path.read_text())
        for index in ("int(C3XSeason.x + 0.5)", "int(game_resource_look.z + 0.5)"):
            text = text.replace(f"#define C3X_SEASON_INDEX {index}", f"#define C3X_SEASON_INDEX {season}")
        path.write_text(text)
    return root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wonder", action="append", choices=sorted(kits.PICKS))
    parser.add_argument("--season", type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument("--lab-host", action="store_true", help="host stamped wonders on a mountain tile, as the study did")
    parser.add_argument("--save", type=Path, help="render the wonders this save placed in their real neighbourhoods")
    parser.add_argument("--pack-shaders", action="store_true",
                        help="compile the game's runtime shader pack (it picks the season from C3X_RENDERER_PREVIEW_SEASON)")
    parser.add_argument("--zoom", type=int, default=256, choices=(128, 256), help="128 is Civ III's 1x map zoom")
    args = parser.parse_args()
    global LAB_HOST, SAVE_SCENES, PACK_SHADERS
    LAB_HOST, PACK_SHADERS = args.lab_host, args.pack_shaders
    if args.save:
        SAVE_SCENES = save_scenes(args.save)
        print("save wonders:", ", ".join(sorted(SAVE_SCENES)), flush=True)
    render(args.wonder or (sorted(SAVE_SCENES) if SAVE_SCENES else DEFAULT), args.season, args.zoom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
