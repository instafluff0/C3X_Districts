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



def sized(kind: tuple, count: int) -> tuple:
    """The smallest square atlas of the kind that holds `count` 512 cells."""
    size = kind[0]
    while size > kits.rc.CELL and ((size // 2) // kits.rc.CELL) ** 2 >= count:
        size //= 2
    return (size, min(kind[1], (size // 4).bit_length()), *kind[2:])


def freeze(out: Path = FROZEN) -> dict:
    catalog = {"id": "tb-picks", "title": "Tile buildings (picks)", "cells": plan.picks()}
    compiler = kits.Compiler()
    for cell in catalog["cells"]:
        if cell.get("whiten"):
            for part in cell["parts"]:
                whiten(compiler, part["record"])
    _, textures, _ = sites.plan()
    extra = [sites.SOURCE / texture for texture in textures]
    # The hut and camp textures take the next cells of the first BC1 atlas;
    # every atlas shrinks to the cells it holds.
    original, placed = kits.rc.pack_atlases, {}

    def pack(textures, kind, first=0):
        textures = list(textures) + (extra if kind == kits.rc.MODEL_ATLAS else [])
        if len(textures) <= (kind[0] // kits.rc.CELL) ** 2:
            kind = sized(kind, len(textures))
        atlases, place = original(textures, kind, first)
        if not placed:   # the model atlases; later calls lay out glow maps alike
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
    colonies.write_pack(bundle, out / "tile_buildings.bin")
    cells = {hashlib.sha256(path.read_bytes()).hexdigest(): [*placed[path], kits.rc.dds(path)[0]] for path in extra}
    report = {"schema": "c3x.tile_buildings_frozen.v1", "bundle": "tile_buildings.bin", "atlases": atlases,
              "site_cells": cells, "source": "Renderer/lab/studies/tile_buildings plan.picks",
              "compositions": list(summary["cells"]), "cells": summary["cells"]}
    (out / "manifest.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lab-pack", help="also build this Lab site pack from the frozen picks (for C3X_RENDERER_SITE_PACK)")
    args = parser.parse_args()
    report = freeze()
    print(json.dumps({k: v for k, v in report.items() if k not in ("cells", "site_cells")}, indent=1))
    if args.lab_pack:
        sites.build(ROOT / "Renderer/packs" / args.lab_pack)
        print("built", args.lab_pack)
