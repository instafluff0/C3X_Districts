#!/usr/bin/env python3
"""Build the source-independent Terrain Lab bundle for normalized river rocks."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=Path("Renderer/packs/ShoreNormalized"),
    )
    args = parser.parse_args()
    target = build(args.pack.resolve())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
