#!/usr/bin/env python3
"""Lab parity of the production wonder port (a private build of the checkout, prod_tree.py)
against the approved study.

Each approved catalog renders again through the production path: the WonderRuntime
pack (build_wonder_pack.py) loaded by the private tree, selected by wonder facts.
The Lab resource pack keeps only a marker per cell that names its tile, and
C3X_LAB_TILE_WONDERS gives each cell's facts (state, progress, facing, era, alias)
as frame ABI 24 carries them in game (the preview fills the tile fields). Frames land in out/wonders/parity/ beside the
study's own renders for side-by-side review.

    python3 Renderer/lab/studies/wonders/prod_parity.py [--catalog ID ...] [--zoom 256]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.wonders import kits, plan, prod_tree

PARITY = kits.OUT / "parity"
FACTS = kits.CATALOG / "wonder_facts.txt"
BUILDING, COMPLETE, UNUSED = 2, 3, 1
ALTERNATE_FLAG, ON_RIVER_FLAG = 1, 2   # c3x_renderer_wonder_flags
ALTERNATE = [0]   # --alternate sets C3X's alternate-direction fact on every cell


def river_wonders() -> set[str]:
    """Plan keys of wonders whose C3X config needs a river (buildable_on_rivers), as
    injected_code.c reports them (C3X_RENDERER_WONDER_ON_RIVER)."""
    from Renderer.tools.asset_compiler import build_wonder_pack as wp
    configs = wp.config_wonders()
    return {entry["c3x_key"] for entry in json.loads(plan.LEDGER.read_text())["mappings"]
            if configs.get(entry["c3x_name"], {}).get("buildable_on_rivers") == "1"}


def facts(catalog: dict) -> list[str]:
    """One fact line per cell: its resource name, state, progress, flags and era, wonder name."""
    ledger = {entry["c3x_key"]: entry["c3x_name"]
              for entry in json.loads(plan.LEDGER.read_text())["mappings"]}
    rivers = river_wonders()
    lines = []
    for cell in catalog["cells"]:
        if cell.get("ghost"):
            continue
        if cell.get("wonder") == "wonder_district":
            era = plan.WD_ERAS.index(cell["label"].split(" · ")[-1])
            lines.append(f"{cell['name']}\t{UNUSED} 0 0 {era}\t-")
            continue
        built = min((part.get("built", 1.0) for part in cell["parts"]), default=1.0)
        state, progress = (COMPLETE, 255) if built >= 1.0 else (BUILDING, int(round(built * 255)))
        flags = (ALTERNATE_FLAG if cell.get("alternate", ALTERNATE[0]) else 0) | \
            (ON_RIVER_FLAG if cell["wonder"] in rivers else 0)
        lines.append(f"{cell['name']}\t{state} {progress} {flags} 0\t{ledger[cell['wonder']]}")
    return lines


def bake(catalog: dict, _compiler=None, _scale: float = 1.0, _ground: bool = True) -> dict:
    """The parity resource pack (a marker per cell) and the cells' facts."""
    cache = kits.OUT / "texture_cache"
    meshes = {"clearance/wonder": kits.marker_mesh(cache)}
    marker = {"model": "clearance/wonder", "u": .5, "v": .5, "rotation": 0.0, "scale": 1.0, "lift": 0.0, "ground_fit": 0.0}
    compositions = [(cell["name"], [(kits.dk.ALL_TERRAINS, [marker])]) for cell in catalog["cells"]]
    kits.write_pack(catalog["id"], meshes, compositions)
    FACTS.write_text("\n".join(facts(catalog)) + "\n")
    report = {cell["name"]: {"label": cell.get("label", ""), "parts": [], "instances": 1, "missing": [], "sources": []}
              for cell in catalog["cells"]}
    summary = {"catalog": catalog["id"], "meshes": 1, "cells": report}
    (kits.CATALOG / "composition_report.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def rivers(catalogs: dict) -> None:
    """Civ III river edge bits for the Lab tiles: a catalog's own "rivers" ({(dx, dy):
    bits} at raw offsets), else a short river along the south-west edge (bit 32; its
    neighbour across holds bit 2) of every river wonder's tile, as roster.mountain-water
    lays one."""
    from Renderer.lab.studies.resources import roster
    spans = river_wonders()
    marked = {}
    for key, spec in catalogs.items():
        case = f"native-district-{key}"
        if "rivers" in spec:
            marked[case] = {((dx + dy + 32) // 2, (dx - dy) // 2): bits for (dx, dy), bits in spec["rivers"].items()}
            continue
        for x, y, cell in kits.dk.layout(spec):
            if cell.get("wonder") in spans:
                c, r = (x + y + 32) // 2, (x - y) // 2
                for dc in range(-3, 4):
                    marked.setdefault(case, {})[(c + dc, r)] = 32
                    marked[case][(c + dc, r - 1)] = 2
    original = roster.river

    def river(case, dx, dy, bits):
        c, r = (dx + dy + 32) // 2, (dx - dy) // 2
        return original(case, dx, dy, bits) | marked.get(case, {}).get((c, r), 0)
    roster.river = river


def render(catalogs: dict, zooms) -> None:
    from Renderer import renderer
    dll, built_preview = prod_tree.binaries()
    preview = kits.OUT / "prod_preview.exe"
    shutil.copy2(built_preview, preview)
    tree = prod_tree.private_tree.tree(prod_tree.NAME)
    native = renderer.native_render
    out, frame = kits.dk.OUT, kits.dk.frame
    kits.dk.OUT = PARITY
    kits.dk.bake = bake
    try:
        seasons_used = sorted({shot.get("season", 0) for spec in catalogs.values() for shot in kits.dk.shots(spec, zooms)})
        for season in seasons_used:
            env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": "..\\..\\" + renderer.relative(tree).replace("/", "\\"),
                   "C3X_RENDERER_PREVIEW_SEASON": str(season),
                   "C3X_LAB_TILE_WONDERS": "..\\..\\" + renderer.relative(FACTS).replace("/", "\\")}

            def production(*args, extra_env=None, **kwargs):
                return native(*args, extra_env={**(extra_env or {}), **env}, **{**kwargs, "candidate": dll, "preview": preview})
            renderer.native_render = production
            subset = {key: {**spec, "shots": [shot for shot in kits.dk.shots(spec, zooms) if shot.get("season", 0) == season]}
                      for key, spec in catalogs.items()}
            kits.dk.render({key: spec for key, spec in subset.items() if spec["shots"]}, zooms)
    finally:
        renderer.native_render = native
        kits.dk.OUT = out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", action="append")
    parser.add_argument("--zoom", action="append", type=int)
    parser.add_argument("--alternate", action="store_true")
    args = parser.parse_args()
    ALTERNATE[0] = int(args.alternate)
    if args.alternate:
        global PARITY
        PARITY = PARITY / "alternate"
    specs = plan.everything()
    # Noon, night and autumn foliage: the parity shots.
    shots = ({"zoom": 256, "hour": 12}, {"zoom": 256, "hour": 22}, {"zoom": 256, "hour": 12, "season": 1, "tag": "-autumn"})
    chosen = {key: {**spec, "shots": shots} for key, spec in specs.items()
              if key in (args.catalog or ["wonder-stages-ancient"])}
    rivers(chosen)
    # The study's frame cache keys on the scene, not on the pack or binaries under test:
    # parity frames always render fresh.
    for key in chosen:
        for stale in (PARITY / "renders").glob(f"{key}-*"):
            shutil.rmtree(stale)
    render(chosen, tuple(args.zoom or (256,)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
