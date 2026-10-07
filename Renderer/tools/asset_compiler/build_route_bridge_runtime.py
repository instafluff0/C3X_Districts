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


def _asset_payload(asset_id: str, texture_index: int, mesh: dict) -> bytes:
    vertices = mesh["vertices"]
    indices = mesh["topology"]["indices"]
    payload = bytearray(bundle_string(asset_id))
    payload.extend(struct.pack("<III", texture_index, len(vertices), len(indices)))
    for vertex in vertices:
        payload.extend(
            struct.pack("<8f", *(vertex["position"] + vertex["normal"] + vertex["uv0"]))
        )
    payload.extend(struct.pack(f"<{len(indices)}I", *indices))
    return bytes(payload)


def _group(name: str, placements: list[tuple[int, float]]) -> bytes:
    group = bytearray(bundle_string(name))
    group.extend(struct.pack("<I", len(placements)))
    for asset_index, scale in placements:
        group.extend(struct.pack("<IffIIIIff", asset_index, scale, 0.0, 1, 1, 1, 0, 0.0, 0.0))
    return bytes(group)


DEFAULT_TUNNEL_PACK = Path("Renderer/packs/RouteTunnelsNormalized")


def tunnel_entrance(mesh: dict) -> dict:
    """The portal mesh's entrance alone: its facade, wing walls, floor and the
    bore behind the arch. Civ VI buries the block behind it (and the rock cap
    on top) under a terrain edit; a C3X tunnel stands at a mountain's foot
    where the rock cannot hide them, so they are dropped (the block reaches
    past the bore, the rubble lies behind the facade above the arch)."""
    vertices = mesh["vertices"]
    indices = mesh["topology"]["indices"]
    kept: list[int] = []
    for start in range(0, len(indices), 3):
        triangle = indices[start:start + 3]
        y = [vertices[i]["position"][1] for i in triangle]
        z = [vertices[i]["position"][2] for i in triangle]
        if max(y) > 0.034 or (min(y) > -0.02 and min(z) > 0.065):
            continue
        kept.extend(triangle)
    used = sorted(set(kept))
    remap = {old: new for new, old in enumerate(used)}
    return {**mesh, "vertices": [vertices[i] for i in used],
            "topology": {**mesh["topology"], "indices": [remap[i] for i in kept]}}


def build(pack: Path, tunnel_pack: Path | None = None) -> Path:
    """Write bridge_runtime.bin. The feature shader has eight bridge texture
    slots (materials 13-20), one per texture here. Bridges keep slots 0, 2, 4
    and 6 (medieval, industrial, modern, railroad) and 5 and 7 (the modern and
    railroad pillaged bridges). Civ III never shows the medieval and industrial
    pillaged bridges, so with the railroad tunnel's normalized pack (see
    route_tunnel_sets.json) slot 1 carries its portal's entrance; without it
    that slot keeps the medieval pillaged bridge."""
    if tunnel_pack is None:
        tunnel_pack = DEFAULT_TUNNEL_PACK
    tunnel = tunnel_pack if (tunnel_pack / "meshes" / "compound" / "route_tunnel_railroad_00.json").is_file() else None
    textures: list[str | None] = [None] * 8
    assets: list[bytes] = []
    groups: list[bytes] = []
    # The transition length includes its terrain-contour approach. The rigid
    # bridge body occupies the central span; these calibrated scales preserve
    # the authored body proportions without making the body fill that approach.
    scales = {"medieval": 4.20, "industrial": 3.70, "modern": 4.25, "railroad": 3.85}
    for style_index, style in enumerate(("medieval", "industrial", "modern", "railroad")):
        for state_index, state in enumerate(("normal", "pillaged")):
            if tunnel and state == "pillaged" and style == "medieval":
                continue
            stem = f"route_bridge_{style}_{state_index:02d}"
            mesh = json.loads((pack / "meshes" / "compound" / f"{stem}.json").read_text())
            material = json.loads(
                (pack / "materials" / "compound" / f"{stem}.json").read_text()
            )
            texture_index = style_index * 2 + state_index
            textures[texture_index] = material["channels"]["base_color"]["texture"]
            assets.append(_asset_payload(f"route/bridge/{style}/{state}", texture_index, mesh))
            groups.append(_group(f"bridge_{style}_{state}", [(len(assets) - 1, scales[style])]))
    if tunnel:
        # The portal's entrance, its texture copied into this pack. It shares
        # the bridges' Civ VI source units, so it takes the railroad bridge's
        # calibrated scale.
        mesh = tunnel_entrance(json.loads((tunnel / "meshes" / "compound" / "route_tunnel_railroad_00.json").read_text()))
        material = json.loads((tunnel / "materials" / "compound" / "route_tunnel_railroad_00.json").read_text())
        source = tunnel / material["channels"]["base_color"]["texture"]
        target = Path("textures") / "tunnel" / source.name
        (pack / target).parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        if not (pack / target).is_file() or (pack / target).read_bytes() != data:
            (pack / target).write_bytes(data)
        textures[1] = target.as_posix()
        assets.append(_asset_payload("route/tunnel/railroad/portal", 1, mesh))
        groups.append(_group("tunnel_railroad", [(len(assets) - 1, scales["railroad"])]))
    if any(texture is None for texture in textures):
        raise ValueError("bridge runtime texture slots are incomplete")

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
    parser.add_argument("--tunnel-pack", type=Path, default=DEFAULT_TUNNEL_PACK)
    args = parser.parse_args()
    target = build(args.pack.resolve(), args.tunnel_pack.resolve())
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
