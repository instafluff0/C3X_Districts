#!/usr/bin/env python3
"""Build source-independent runtime bundles for normalized shore features."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


MAGIC = b"C3XVEG1\0"


def bundle_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > 4096:
        raise ValueError("runtime-bundle string has an invalid length")
    return struct.pack("<I", len(encoded)) + encoded


def build(pack: Path) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    river_ids = manifest["feature_sets"]["river_rock"]["variants"]
    textures: list[str] = []
    assets: list[bytes] = []
    for texture_index, asset_id in enumerate(river_ids):
        record = manifest["assets"][asset_id]
        mesh = json.loads((pack / record["mesh"]).read_text(encoding="utf-8"))
        material = json.loads((pack / record["material"]).read_text(encoding="utf-8"))
        texture = material["base_color"]["texture"]
        vertices = mesh["vertices"]
        indices = mesh["topology"]["indices"]
        textures.append(texture)
        payload = bytearray(bundle_string(asset_id))
        payload.extend(struct.pack("<III", texture_index, len(vertices), len(indices)))
        for vertex in vertices:
            payload.extend(struct.pack(
                "<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"])
            ))
        payload.extend(struct.pack(f"<{len(indices)}I", *indices))
        assets.append(bytes(payload))

    group = bytearray(bundle_string("river_rock"))
    group.extend(struct.pack("<I", len(river_ids)))
    for asset_index in range(len(river_ids)):
        group.extend(struct.pack(
            "<IffIIIIff", asset_index, 1.0, 0.12, 1, 0, 1, 1, 0.0, 0.0
        ))

    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), 1))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    output.extend(group)
    target = pack / "shore_runtime.bin"
    target.write_bytes(output)
    return target


def build_cliffs(pack: Path) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    group_names = ("cliff_large", "cliff_small")
    asset_ids = [
        asset_id
        for group_name in group_names
        for asset_id in manifest["feature_sets"][group_name]["variants"]
    ]
    asset_indices = {asset_id: index for index, asset_id in enumerate(asset_ids)}
    textures: list[str] = []
    assets: list[bytes] = []
    for asset_id in asset_ids:
        record = manifest["assets"][asset_id]
        mesh = json.loads((pack / record["mesh"]).read_text(encoding="utf-8"))
        material = json.loads((pack / record["material"]).read_text(encoding="utf-8"))
        texture_index = len(textures)
        textures.extend(
            (
                material["base_color"]["texture"],
                material["lean_normal"]["texture_0"],
                material["lean_normal"]["texture_1"],
                material["gloss"]["texture"],
            )
        )
        vertices = mesh["vertices"]
        indices = mesh["topology"]["indices"]
        payload = bytearray(bundle_string(asset_id))
        payload.extend(struct.pack("<III", texture_index, len(vertices), len(indices)))
        for vertex in vertices:
            payload.extend(
                struct.pack("<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"]))
            )
        payload.extend(struct.pack(f"<{len(indices)}I", *indices))
        assets.append(bytes(payload))

    groups: list[bytes] = []
    for group_name in group_names:
        group = manifest["feature_sets"][group_name]
        placements = group.get("placements")
        if placements is None:
            variation = 0.10 if group_name == "cliff_large" else 0.15
            count = 16 if group_name == "cliff_large" else 12
            placements = [
                {
                    "asset": asset_id,
                    "scale": 1.0,
                    "scale_variation": variation,
                    "count": count,
                    "min_count": 0,
                    "priority": 3,
                    "allow_overlap": True,
                    "show_decal": True,
                    "is_center_model": False,
                    "width": 0.0,
                    "low_end_reduction": 0.0,
                }
                for asset_id in group["variants"]
            ]
        payload = bytearray(bundle_string(group_name))
        payload.extend(struct.pack("<I", len(placements)))
        for placement in placements:
            flags = (
                (1 if placement["allow_overlap"] else 0)
                | (2 if placement["show_decal"] else 0)
                | (4 if placement["is_center_model"] else 0)
            )
            payload.extend(
                struct.pack(
                    "<IffIIIIff",
                    asset_indices[placement["asset"]],
                    placement["scale"],
                    placement["scale_variation"],
                    placement["count"],
                    placement["min_count"],
                    placement["priority"],
                    flags,
                    placement["width"],
                    placement["low_end_reduction"],
                )
            )
        groups.append(bytes(payload))

    if len(textures) > 32:
        raise ValueError("Cliff runtime bundle exceeds the 32-texture feature limit")
    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / "cliff_runtime.bin"
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("Renderer/packs/ShoreNormalized"),
    )
    args = parser.parse_args()
    pack = args.pack.resolve()
    river_target = build(pack)
    cliff_target = build_cliffs(pack)
    print(f"wrote {river_target} ({river_target.stat().st_size} bytes)")
    print(f"wrote {cliff_target} ({cliff_target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
