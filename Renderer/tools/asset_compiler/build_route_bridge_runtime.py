#!/usr/bin/env python3
"""Build the source-independent Terrain Lab runtime bundle for normalized road bridges."""

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
    textures: list[str] = []
    assets: list[bytes] = []
    groups: list[bytes] = []
    # The transition length includes its terrain-contour approach. The rigid
    # bridge body occupies the central span; these calibrated scales preserve
    # the authored body proportions without making the body fill that approach.
    scales = {"medieval": 4.20, "industrial": 3.70, "modern": 4.25, "railroad": 3.85}
    for style in ("medieval", "industrial", "modern", "railroad"):
        for state_index, state in enumerate(("normal", "pillaged")):
            stem = f"route_bridge_{style}_{state_index:02d}"
            mesh = json.loads((pack / "meshes" / "compound" / f"{stem}.json").read_text())
            material = json.loads(
                (pack / "materials" / "compound" / f"{stem}.json").read_text()
            )
            texture = material["channels"]["base_color"]["texture"]
            texture_index = len(textures)
            textures.append(texture)
            vertices = mesh["vertices"]
            indices = mesh["topology"]["indices"]
            asset_id = f"route/bridge/{style}/{state}"
            payload = bytearray(bundle_string(asset_id))
            payload.extend(struct.pack("<III", texture_index, len(vertices), len(indices)))
            for vertex in vertices:
                payload.extend(
                    struct.pack(
                        "<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"])
                    )
                )
            payload.extend(struct.pack(f"<{len(indices)}I", *indices))
            assets.append(bytes(payload))
            group = bytearray(bundle_string(f"bridge_{style}_{state}"))
            group.extend(struct.pack("<I", 1))
            group.extend(
                struct.pack(
                    "<IffIIIIff",
                    len(assets) - 1,
                    scales[style],
                    0.0,
                    1,
                    1,
                    1,
                    0,
                    0.0,
                    0.0,
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
    target = pack / "bridge_runtime.bin"
    target.write_bytes(output)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack", type=Path, default=Path("Renderer/packs/RouteDoodadsNormalized")
    )
    args = parser.parse_args()
    target = build(args.pack.resolve())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
