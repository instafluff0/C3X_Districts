#!/usr/bin/env python3
"""Large central-building mine trial, using only source variants one and two."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import shutil
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.mines import closeups, study, terrain
from Renderer.lab.studies.mines.build_pack import collect, local_part
from Renderer.preview.render_improvement_sheet import (
    IDENTITY, _load_json, _matrix_multiply, _skeleton_worlds,
)
from Renderer.tools.asset_compiler.build_mine_runtime import bundle_string, merged_asset


LABEL = "central-building"
SCALE = 3.0


def main_building(pack: Path, manifest: dict, root_id: str) -> tuple[dict, str, str, dict]:
    root = _load_json(pack / manifest["assets"][root_id]["landmark"])
    worlds = [_skeleton_worlds(_load_json(pack / path))
              for path in root["components"]["skeletons"]]
    candidates = []
    for point in root["attachment_points"]:
        if point["binding_status"] != "resolved":
            continue
        transform = _matrix_multiply(worlds[point["skeleton"]][point["bone"]], IDENTITY)
        body = [(mesh, base, emissive) for mesh, base, emissive, ground in
                collect(pack, manifest, point["component_asset"], transform) if not ground]
        if body:
            candidates.append((sum(len(mesh["vertices"]) for mesh, _, _ in body),
                               point["component_asset"], body))
    if not candidates:
        raise ValueError("Mine source root contains no attached building")
    count, child_id, body = max(candidates, key=lambda item: item[0])
    if len(body) != 1:
        raise ValueError("The selected central building changed from one draw part")
    mesh, base, emissive = body[0]
    if not emissive:
        raise ValueError("The selected source building lost its emissive material")
    centered, _ox, _oy = local_part(mesh)
    low = min(vertex["position"][2] for vertex in centered["vertices"])
    for vertex in centered["vertices"]:
        vertex["position"][2] -= low
    return centered, base, emissive, {"source_root": root_id,
                                      "selected_child": child_id,
                                      "vertices": count}


def build(pack: Path, scale: float = SCALE) -> dict:
    manifest = _load_json(pack / "manifest.json")
    catalog = _load_json(pack / manifest["improvement_catalog"])
    roots = [variant for family in catalog["mine"]["eras"]
             for variant in family["variants"][:2]]
    selected = [main_building(pack, manifest, root_id) for root_id in roots]
    counts = defaultdict(int)
    emissive_set = set()
    for family in catalog["mine"]["eras"]:
        for root_id in family["variants"]:
            for _mesh, base, emissive, _ground in collect(pack, manifest, root_id):
                counts[base] += 1
                if emissive:
                    emissive_set.add(emissive)
    bases = sorted(counts, key=lambda item: (-counts[item], item))[:6]
    emissives = sorted(emissive_set)
    if len(bases) != 6 or len(emissives) != 2 or any(
            base not in bases or emissive not in emissives
            for _mesh, base, emissive, _info in selected):
        raise ValueError("The source texture set no longer fits the eight-slot mine ABI")
    assets = [merged_asset(f"mine_central_{index}", bases.index(base),
                           emissives.index(emissive) + 1, [mesh])
              for index, (mesh, base, emissive, _info) in enumerate(selected)]
    # The unchanged Lab renderer has three group slots per family. Its third
    # slot repeats source variant one; no source variant three mesh is included.
    groups = []
    for slot, asset_index in enumerate((0, 1, 0, 2, 3, 2)):
        group = bytearray(bundle_string(f"mine_{slot}"))
        group.extend(struct.pack("<I", 1))
        group.extend(struct.pack("<IffIIIIff", asset_index, scale, .04,
                                 1, 1, 5, 0, 0.0, 0.0))
        groups.append(bytes(group))
    payload = bytearray(b"C3XVEG1\0")
    payload.extend(struct.pack("<IIII", 1, len(bases) + len(emissives),
                               len(assets), len(groups)))
    for texture in bases + emissives:
        payload.extend(bundle_string(texture))
    for asset in assets:
        payload.extend(asset)
    for group in groups:
        payload.extend(group)
    (pack / "mine_runtime.bin").write_bytes(payload)
    return {"source_variants": 4, "runtime_assets": len(assets),
            "runtime_groups": len(groups), "scale": scale, "bytes": len(payload),
            "main_buildings": [item[3] for item in selected]}


def render_missing(name: str, scene: Path, width: int) -> None:
    output = study.OUT / LABEL / name
    capture = output / "capture.json"
    dll = study.OUT / "part-ground.dll"
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    if capture.is_file() and (output / "preview.png").is_file():
        prior = json.loads(capture.read_text())
        if (prior["scene_sha256"] == study.digest(scene) and
                prior["dll_sha256"] == study.digest(dll) and
                prior["pack_sha256"] == study.digest(pack) and
                prior["tile_width"] == width):
            return
    for attempt in range(3):
        try:
            print(study.render(LABEL, name, scene, dll, width), flush=True)
            return
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(3)


def grass_sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1200, 4 * 690), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=23)
    for era, label in enumerate(closeups.ERAS):
        y = era * 690
        draw.text((16, y + 8), f"{label} | main building only | source variants 1 and 2 | {SCALE:.1f}x",
                  fill="white", font=font)
        for variant in range(2):
            name = f"era-{era}-variant-{variant}"
            source = Image.open(study.OUT / LABEL / name / "preview.png").convert("RGB")
            crop = source.crop((610, 200, 990, 590)).resize((580, 600), Image.Resampling.LANCZOS)
            x = variant * 600 + 10
            canvas.paste(crop, (x, y + 55))
            draw.text((x + 10, y + 62), f"Variant {variant + 1}", fill="white",
                      font=font, stroke_width=2, stroke_fill="#20272b")
    path = study.OUT / "test-biq-mine-central-grassland.png"
    canvas.save(path)
    return path


def terrain_sheet() -> Path:
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (1208, 2668), "#20272b")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=19)
    draw.text((16, 10), f"test.biq | complete 2.3x mine vs central building {SCALE:.1f}x",
              fill="white", font=font)
    for row, name in enumerate(terrain.TERRAIN_SITES):
        for col, label in enumerate(("part-ground", LABEL)):
            source = Image.open(study.OUT / label / name / "preview.png").convert("RGB")
            crop = source.crop((650, 300, 950, 620)).resize((600, 640), Image.Resampling.BICUBIC)
            x, y = col * 604, row * 648 + 40
            canvas.paste(crop, (x, y + 32))
            draw.text((x + 10, y + 5), f"{name} | {'full' if col == 0 else 'central'}",
                      fill="white", font=font)
    path = study.OUT / "test-biq-mine-central-terrain.png"
    canvas.save(path)
    return path


def main() -> None:
    bare = study.OUT / "bare-pack.bin"
    dll = study.OUT / "part-ground.dll"
    if not bare.is_file() or not dll.is_file():
        raise RuntimeError("Run Renderer/lab/studies/mines/terrain.py first")
    pack = study.PRIVATE / "Renderer/packs/ImprovementsNormalized/mine_runtime.bin"
    shutil.copy2(bare, pack)
    candidate = study.OUT / "central-pack.bin"
    stats = build(pack.parent)
    shutil.copy2(pack, candidate)
    (study.OUT / "central-pack.json").write_text(json.dumps(stats, indent=2) + "\n")
    try:
        grass = closeups.prepare()
        for name, scene in list(grass.items())[:1 if "--single" in sys.argv[1:] else None]:
            if not name.endswith("variant-2"):
                render_missing(name, scene, 256)
        if "--single" not in sys.argv[1:]:
            study.SITES = dict(terrain.TERRAIN_SITES)
            scenes = study.scenes()
            for name in terrain.TERRAIN_SITES:
                render_missing(name, scenes[name], 128)
    finally:
        shutil.copy2(bare, pack)
    if "--single" not in sys.argv[1:]:
        print(grass_sheet(), flush=True)
        print(terrain_sheet(), flush=True)


if __name__ == "__main__":
    main()
