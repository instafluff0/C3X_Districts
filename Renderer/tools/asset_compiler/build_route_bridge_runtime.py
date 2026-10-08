#!/usr/bin/env python3
"""Build the source-independent Terrain Lab runtime bundle for normalized road bridges."""

from __future__ import annotations

import argparse
import colorsys
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


# Railroad tunnels are out of the game (the user's call, 2026-10-07, after
# portals crowded the 1498 save's ranges): the bundle carries them only when
# --tunnel-pack names this pack explicitly, for Lab study.
TUNNEL_PACK = Path("Renderer/packs/RouteTunnelsNormalized")
# A tunnel portal's block reaches back into its mountain: the runtime picks
# the shortest of these lengths (behind the facade) that the rock covers.
TUNNEL_LENGTHS = (1.0, 1.5, 2.0, 3.0, 4.0)
TUNNEL_FACADE_Y = -0.025
TUNNEL_AXIS_X = -0.00705


def grey_rgb(r: float, g: float, b: float) -> tuple[float, float, float]:
    """The tunnel's brown rock as the mountains' grey rock: its luminance
    (slightly lifted), without its colour. Strong colours (the red signal
    stripes) keep theirs."""
    _, saturation, _ = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    if saturation > .6:
        return r, g, b
    grey = min(255.0, (.2126 * r + .7152 * g + .0722 * b) * 1.12)
    return grey * .98, grey, grey


def _rgb565(rgb: tuple[float, float, float]) -> int:
    r, g, b = (min(255, max(0, round(c))) for c in rgb)
    return (r * 31 + 127) // 255 << 11 | (g * 63 + 127) // 255 << 5 | (b * 31 + 127) // 255


def _unpack565(c: int) -> tuple[int, int, int]:
    return (c >> 11) * 255 // 31, (c >> 5 & 63) * 255 // 63, (c & 31) * 255 // 31


def grey_texture(data: bytes) -> bytes:
    """A BC1 texture (DX10 header, every mip) recoloured by grey_rgb. Only
    block endpoints change; where their order flips, they swap and the
    indices follow, so each block keeps its colour mode."""
    if data[:4] != b"DDS " or data[84:88] != b"DX10" or struct.unpack_from("<I", data, 128)[0] not in (71, 72):
        raise ValueError("tunnel texture is not a DX10 BC1 DDS")
    out = bytearray(data)
    for at in range(148, len(out) - 7, 8):
        c0, c1, indices = struct.unpack_from("<HHI", out, at)
        n0, n1 = (_rgb565(grey_rgb(*_unpack565(c))) for c in (c0, c1))
        if (c0 > c1) != (n0 > n1) and n0 != n1:
            n0, n1 = n1, n0
            if c0 > c1:  # four colours: 0<->1, 2<->3
                indices ^= 0x55555555
            else:  # three colours: 0<->1, 2 and 3 stay
                indices = sum((((indices >> 2 * k & 3) ^ (1 if (indices >> 2 * k & 3) < 2 else 0)) << 2 * k)
                              for k in range(16))
        elif (c0 > c1) and n0 == n1:
            indices = 0  # a flat block
        struct.pack_into("<HHI", out, at, n0, n1, indices)
    return bytes(out)


def stretch_block(mesh: dict, length: float) -> dict:
    """The portal with its block and bore behind the facade `length` times as
    deep; the facade, its wing walls and cutting stay as they are."""
    vertices = []
    for vertex in mesh["vertices"]:
        x, y, z = vertex["position"]
        if y > TUNNEL_FACADE_Y:
            y = TUNNEL_FACADE_Y + (y - TUNNEL_FACADE_Y) * length
        vertices.append({**vertex, "position": [x, y, z]})
    return {**mesh, "vertices": vertices}


def centre_on_cutting(mesh: dict) -> dict:
    """The tunnel's cutting between its wing walls (inner faces at x -0.0393
    and 0.0252) and its arch opening centre on TUNNEL_AXIS_X, not on the mesh
    axis; moved onto it, the rail runs down their middle."""
    return {**mesh, "vertices": [{**vertex, "position": [vertex["position"][0] - TUNNEL_AXIS_X,
                                                         *vertex["position"][1:]]}
                                 for vertex in mesh["vertices"]]}

def build(pack: Path, tunnel_pack: Path | None = None) -> Path:
    """Write bridge_runtime.bin. The feature shader has eight bridge texture
    slots (materials 13-20), one per texture here. Bridges keep slots 0, 2, 4
    and 6 (medieval, industrial, modern, railroad) and 5 and 7 (the modern and
    railroad pillaged bridges). Civ III never shows the medieval and industrial
    pillaged bridges, so with the railroad tunnel's normalized pack (see
    route_tunnel_sets.json) slots 1 and 3 carry its portal and rock cap, their
    rock greyed to the mountains'; without it they keep those pillaged
    bridges. The tunnel group's placements are the cap, then the portal at
    each of TUNNEL_LENGTHS. Without a tunnel pack (the default) the bundle
    has no tunnel group, and the runtime draws no tunnels."""
    tunnel = tunnel_pack if tunnel_pack is not None and \
        (tunnel_pack / "meshes" / "compound" / "route_tunnel_railroad_00.json").is_file() else None
    textures: list[str | None] = [None] * 8
    assets: list[bytes] = []
    groups: list[bytes] = []
    # The transition length includes its terrain-contour approach. The rigid
    # bridge body occupies the central span; these calibrated scales preserve
    # the authored body proportions without making the body fill that approach.
    scales = {"medieval": 4.20, "industrial": 3.70, "modern": 4.25, "railroad": 3.85}
    for style_index, style in enumerate(("medieval", "industrial", "modern", "railroad")):
        for state_index, state in enumerate(("normal", "pillaged")):
            if tunnel and state == "pillaged" and style in ("medieval", "industrial"):
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
        # The portal (with its cutting and the block behind it) and its rock
        # cap, each with its own greyed texture in this pack. They share the
        # bridges' Civ VI source units, so they take the railroad bridge's
        # calibrated scale.
        placements = []
        for part, (mesh_index, material_index, slot) in {"cap": (1, 2, 3), "portal": (0, 0, 1)}.items():
            mesh = centre_on_cutting(json.loads(
                (tunnel / "meshes" / "compound" / f"route_tunnel_railroad_{mesh_index:02d}.json").read_text()))
            material = json.loads((tunnel / "materials" / "compound" / f"route_tunnel_railroad_{material_index:02d}.json").read_text())
            source = tunnel / material["channels"]["base_color"]["texture"]
            target = Path("textures") / "tunnel" / ("grey_" + source.name)
            (pack / target).parent.mkdir(parents=True, exist_ok=True)
            data = grey_texture(source.read_bytes())
            if not (pack / target).is_file() or (pack / target).read_bytes() != data:
                (pack / target).write_bytes(data)
            textures[slot] = target.as_posix()
            for length in (TUNNEL_LENGTHS if part == "portal" else (None,)):
                name = part if length in (None, 1.0) else f"{part}_x{length:g}"
                assets.append(_asset_payload(f"route/tunnel/railroad/{name}", slot,
                                             stretch_block(mesh, length) if length else mesh))
                placements.append((len(assets) - 1, scales["railroad"]))
        groups.append(_group("tunnel_railroad", placements))
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
    parser.add_argument("--tunnel-pack", type=Path, default=None,
                        help=f"add railroad tunnels from this pack (Lab study only; e.g. {TUNNEL_PACK})")
    args = parser.parse_args()
    target = build(args.pack.resolve(), args.tunnel_pack.resolve() if args.tunnel_pack else None)
    print(f"wrote {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
