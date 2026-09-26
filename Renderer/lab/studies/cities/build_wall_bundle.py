#!/usr/bin/env python3
"""Build an isolated wall bundle with the rounded Lab city wall pieces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from Renderer.tools.asset_compiler.build_city_runtime import (
    WALLS, asset_payload, group_payload, serialize,
)


ROOT = Path(__file__).resolve().parents[4]
PACK = ROOT / "Renderer/packs/CityAdjunctsNormalized"


def source_asset(pack: Path, manifest: dict, asset_id: str) -> tuple[list[dict], str]:
    landmark_path = pack / manifest["assets"][asset_id]["landmark"]
    landmark = json.loads(landmark_path.read_text(encoding="utf-8"))
    parts = [json.loads((pack / path).read_text(encoding="utf-8"))
             for path in landmark["components"]["geometry"]]
    materials = [json.loads((pack / path).read_text(encoding="utf-8"))
                 for path in landmark["components"]["materials"]]
    textures = {material["channels"]["base_color"]["texture"]
                for material in materials}
    if len(textures) != 1:
        raise ValueError(f"{asset_id}: Lab wall parts require one atlas")
    return parts, textures.pop()


def merged_centered(parts: list[dict]) -> dict:
    vertices: list[dict] = []
    indices: list[int] = []
    for mesh in parts:
        base = len(vertices)
        vertices.extend(mesh["vertices"])
        indices.extend(base + index for index in mesh["topology"]["indices"])
    center_x = (min(v["position"][0] for v in vertices) +
                max(v["position"][0] for v in vertices)) / 2
    center_y = (min(v["position"][1] for v in vertices) +
                max(v["position"][1] for v in vertices)) / 2
    centered = [{**vertex, "position": [vertex["position"][0] - center_x,
                                         vertex["position"][1] - center_y,
                                         vertex["position"][2]]}
                for vertex in vertices]
    return {"vertices": centered, "topology": {"indices": indices}}


def build(output: Path, pack: Path = PACK) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((pack / "city_adjunct_catalog.json").read_text(encoding="utf-8"))
    textures: list[str] = []
    assets: list[bytes] = []
    groups: list[bytes] = []

    # Keep the legacy groups so an isolated replay with a non-Lab city retains
    # its current behavior. This reproduces the existing wall bundle recipe.
    for group_name, asset_id in WALLS:
        parts, texture = source_asset(pack, manifest, asset_id)
        if texture not in textures:
            textures.append(texture)
        assets.append(asset_payload(asset_id, textures.index(texture), parts[0]))
        groups.append(group_payload(group_name, [(len(assets) - 1, 7.2)]))

    for era in ("ancient", "medieval", "industrial"):
        kit = catalog["walls"]["kits"][era]
        for role in ("segment", "gate", "tower"):
            asset_id = (next((asset for asset in kit[role]
                              if asset.endswith("tower_small")), kit[role][0])
                        if role == "tower" else kit[role][0])
            parts, texture = source_asset(pack, manifest, asset_id)
            if texture not in textures:
                textures.append(texture)
            assets.append(asset_payload(asset_id, textures.index(texture),
                                        merged_centered(parts)))
            scale = 2.0 if role == "tower" else 2.3
            groups.append(group_payload(f"wall_lab_{era}_{role}",
                                        [(len(assets) - 1, scale)]))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialize(textures, assets, groups))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "Renderer/lab/out/cities/wall-runtime/wall_runtime.bin")
    args = parser.parse_args()
    target = build(args.output)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
