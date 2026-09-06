#!/usr/bin/env python3
"""Build the source-independent L16 Terrain Lab resource-body bundle."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


MAGIC = b"C3XVEG1\0"
SELECTIONS = (
    ("horses", "resource/assets/horse_001", 0.35, 1),
    ("iron", "resource/assets/res_iron_rock01", 0.78, 3),
    ("uranium", "resource/assets/res_uranium_rock01", 0.90, 3),
    ("gold", "resource/assets/res_gold_rock01", 0.78, 3),
    ("dye", "resource/assets/res_dyes_01", 0.48, 3),
    ("wheat", "resource/assets/res_wheat_tuft01", 1.05, 5),
    ("cattle", "resource/assets/res_cattle_cow01", 0.80, 1),
    ("fish", "resource/fish/landmark", 1.75, 1),
)


def bundle_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    if not encoded or len(encoded) > 4096:
        raise ValueError("runtime-bundle string has an invalid length")
    return struct.pack("<I", len(encoded)) + encoded


def material_texture(material: dict) -> str:
    channel = material.get("base_color", material.get("channels", {}).get("base_color"))
    if not isinstance(channel, dict) or not channel.get("texture"):
        raise ValueError("resource material lacks a base-color texture")
    return channel["texture"]


def build(pack: Path) -> Path:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    textures: list[str] = []
    assets: list[bytes] = []
    groups: list[bytes] = []
    for name, asset_id, scale, count in SELECTIONS:
        record = manifest["assets"][asset_id]
        mesh = json.loads((pack / record["mesh"]).read_text(encoding="utf-8"))
        material = json.loads((pack / record["material"]).read_text(encoding="utf-8"))
        texture_index = len(textures)
        textures.append(material_texture(material))
        vertices = mesh["vertices"]
        indices = mesh["topology"]["indices"]
        payload = bytearray(bundle_string(asset_id))
        payload.extend(struct.pack("<III", texture_index, len(vertices), len(indices)))
        for vertex in vertices:
            position = list(vertex["position"])
            # The skinned fish bind pose is authored just below the water plane.
            # Normalize it to a readable surface pose without changing its body.
            if name == "fish":
                position[2] += 0.060
            payload.extend(struct.pack("<8f", *(position + vertex["normal"] + vertex["uv0"])))
        payload.extend(struct.pack(f"<{len(indices)}I", *indices))
        assets.append(bytes(payload))
        group = bytearray(bundle_string(name))
        group.extend(struct.pack("<I", 1))
        group.extend(
            struct.pack(
                "<IffIIIIff", texture_index, scale, 0.10 if count > 1 else 0.03,
                count, 1, 5, 0, 0.0, 0.0,
            )
        )
        groups.append(bytes(group))

    output = bytearray(MAGIC)
    output.extend(struct.pack("<IIII", 1, len(textures), len(assets), len(groups)))
    for texture in textures:
        output.extend(bundle_string(texture))
    for asset in assets:
        output.extend(asset)
    for group in groups:
        output.extend(group)
    target = pack / "resource_runtime.bin"
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=Path("Renderer/packs/ResourceNormalized"))
    args = parser.parse_args()
    target = build(args.pack.resolve())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
