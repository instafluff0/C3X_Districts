#!/usr/bin/env python3
"""Build compact source-independent city and wall bundles for L17."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


MAGIC = b"C3XVEG1\0"
POOLS = (
    ("ancient", "city/pool/american/ancient", 4.8),
    ("medieval", "city/pool/asian/medieval", 4.8),
    ("industrial", "city/pool/european/industrial", 3.1),
    ("modern", "city/pool/american/modern", 3.1),
)
WALLS = (
    ("wall_ancient", "city/walls/ancient/half_01"),
    ("wall_medieval", "city/walls/medieval/half_01"),
    ("wall_industrial", "city/walls/industrial/half_01"),
)


def bundle_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def asset_payload(asset_id: str, texture_index: int, mesh: dict) -> bytes:
    vertices = mesh["vertices"]
    indices = mesh["topology"]["indices"]
    payload = bytearray(bundle_string(asset_id))
    payload.extend(struct.pack("<III", texture_index, len(vertices), len(indices)))
    for vertex in vertices:
        payload.extend(struct.pack("<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"])))
    payload.extend(struct.pack(f"<{len(indices)}I", *indices))
    return bytes(payload)


def group_payload(name: str, placements: list[tuple[int, float]]) -> bytes:
    payload = bytearray(bundle_string(name))
    payload.extend(struct.pack("<I", len(placements)))
    for asset_index, scale in placements:
        payload.extend(struct.pack("<IffIIIIff", asset_index, scale, 0.04, 1, 1, 5, 0, 0.0, 0.0))
    return bytes(payload)


def serialize(textures: list[str], assets: list[bytes], groups: list[bytes]) -> bytes:
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    return bytes(output)


def build_city(pack: Path) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((pack / manifest["city_catalog"]).read_text(encoding="utf-8"))
    base_textures: list[str] = []
    emissive_textures: list[str] = []
    assets: list[bytes] = []
    groups: list[bytes] = []
    for era_index, (era, pool_id, scale) in enumerate(POOLS):
        placements = []
        component_ids = catalog["pools"][pool_id]["components"]
        # Modern's special first tower uses a separate material family; retain
        # the three coherent shared-material bodies in this compact proof pack.
        if era == "modern":
            component_ids = component_ids[1:]
        first_base = first_emissive = None
        for asset_id in component_ids:
            landmark = json.loads((pack / manifest["assets"][asset_id]["landmark"]).read_text(encoding="utf-8"))
            mesh = json.loads((pack / landmark["components"]["geometry"][0]).read_text(encoding="utf-8"))
            material = json.loads((pack / landmark["components"]["materials"][0]).read_text(encoding="utf-8"))
            base = material["channels"]["base_color"]["texture"]
            emissive = material["channels"].get("emissive", {}).get("texture", base)
            first_base = first_base or base
            first_emissive = first_emissive or emissive
            if base != first_base or emissive != first_emissive:
                raise ValueError(f"{pool_id} proof components do not share one material family")
            placements.append((len(assets), scale))
            assets.append(asset_payload(asset_id, era_index, mesh))
        base_textures.append(first_base)
        emissive_textures.append(first_emissive)
        groups.append(group_payload(era, placements))
    target = pack / "city_runtime.bin"
    target.write_bytes(serialize(base_textures + emissive_textures, assets, groups))
    return target


def build_walls(pack: Path) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    textures: list[str] = []
    assets: list[bytes] = []
    groups: list[bytes] = []
    for group_name, asset_id in WALLS:
        landmark = json.loads((pack / manifest["assets"][asset_id]["landmark"]).read_text(encoding="utf-8"))
        mesh = json.loads((pack / landmark["components"]["geometry"][0]).read_text(encoding="utf-8"))
        material = json.loads((pack / landmark["components"]["materials"][0]).read_text(encoding="utf-8"))
        texture = material["channels"]["base_color"]["texture"]
        if not textures:
            textures.append(texture)
        elif texture != textures[0]:
            raise ValueError("selected wall eras do not share their source material")
        assets.append(asset_payload(asset_id, 0, mesh))
        groups.append(group_payload(group_name, [(len(assets) - 1, 7.2)]))
    target = pack / "wall_runtime.bin"
    target.write_bytes(serialize(textures, assets, groups))
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", type=Path, default=Path("Renderer/packs/CityComponentsNormalized"))
    parser.add_argument("--walls", type=Path, default=Path("Renderer/packs/CityAdjunctsNormalized"))
    args = parser.parse_args()
    for target in (build_city(args.cities.resolve()), build_walls(args.walls.resolve())):
        print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
