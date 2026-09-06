#!/usr/bin/env python3
"""Render a lab-only eight-direction action sheet for a compiled unit formation."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.preview.render_feature_asset import draw_mesh, rgb565
from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import write_png
from Renderer.tools.asset_compiler import normalized_animation, normalized_pose_cache, normalized_skin
from Renderer.tools.asset_compiler.civ3_owner_palette_compiler import load_owner_color_table


BACKGROUND = (25, 31, 34)
CIV_COLOR_STRENGTH = 0.82
CIV_COLOR_NEUTRAL_FLOOR = 0.42
CIV_COLOR_GAIN = 0.95
DIRECTIONS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
ACTIONS = ("idle", "move", "attack", "death")
ACTION_PHASE = {"idle": 0.28, "move": 0.34, "attack": 0.58, "death": 0.92}
FONT = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01111 10000 10000 10000 10000 10000 01111", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01111 10000 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "11111 00100 00100 00100 00100 00100 11111", "J": "00111 00010 00010 00010 10010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 11001 10101 10011 10001 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11110 00001 00001 01110 00001 00001 11110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 10000 11110 00001 00001 11110",
    "6": "01110 10000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00001 01110",
    "-": "00000 00000 00000 11111 00000 00000 00000", "/": "00001 00010 00010 00100 01000 01000 10000",
    ":": "00000 00100 00100 00000 00100 00100 00000", ".": "00000 00000 00000 00000 00000 00110 00110",
}


def _srgb_to_linear(value: float) -> float:
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * 12.92 if value <= 0.0031308 else 1.055 * value ** (1.0 / 2.4) - 0.055


def apply_owner_tint(
    rgb: Sequence[int],
    alpha: int,
    owner_color: tuple[int, int, int] | None = None,
    owner_ramp: Sequence[Sequence[int]] | None = None,
    tint_strength: float = CIV_COLOR_STRENGTH,
    representative_palette_index: int = 6,
) -> tuple[int, int, int]:
    """Conservatively modulate source RGB through the inferred Civ VI tint mask."""
    if owner_color is None and owner_ramp is None:
        return tuple(rgb)
    if owner_color is not None and owner_ramp is not None:
        raise ValueError("owner RGB and owner ramp are mutually exclusive")
    if not 0.0 <= tint_strength <= 1.0:
        raise ValueError("owner tint strength must be between zero and one")
    base_linear = [_srgb_to_linear(channel / 255.0) for channel in rgb]
    if owner_ramp is not None:
        if len(owner_ramp) not in {16, 64} or any(len(color) != 3 for color in owner_ramp):
            raise ValueError("Civ III owner-color input must contain 16 or 64 RGB colors")
        if not 0 <= representative_palette_index < len(owner_ramp):
            raise ValueError("representative palette index is outside the supplied color row")
        # The source texture already contains material value and shading. Civ
        # III palette index 6 is the default representative civilization color,
        # but normalized pack metadata may select any preserved runtime slot.
        tint = owner_ramp[representative_palette_index]
    else:
        tint = owner_color or ()
    tint_linear = [_srgb_to_linear(channel / 255.0) for channel in tint]
    tint_weight = (1.0 - alpha / 255.0) * tint_strength
    modulated = [
        base_linear[channel]
        * (CIV_COLOR_NEUTRAL_FLOOR + CIV_COLOR_GAIN * tint_linear[channel])
        for channel in range(3)
    ]
    return tuple(
        int(
            round(
                _linear_to_srgb(
                    base_linear[channel] * (1.0 - tint_weight)
                    + modulated[channel] * tint_weight
                )
                * 255.0
            )
        )
        for channel in range(3)
    )


def draw_text(canvas: Canvas, x: int, y: int, value: str, color: tuple[int, int, int], scale: int = 1) -> None:
    cursor = x
    for character in value.upper():
        if character == " ":
            cursor += 4 * scale
            continue
        glyph = FONT.get(character)
        if glyph is None:
            cursor += 6 * scale
            continue
        for row, bits in enumerate(glyph.split()):
            for column, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            canvas.set_pixel(cursor + column * scale + dx, y + row * scale + dy, color)
        cursor += 6 * scale


def _color_palette(color0: int, color1: int) -> tuple[tuple[int, int, int], ...]:
    first = rgb565(color0)
    second = rgb565(color1)
    return (
        first,
        second,
        tuple((2 * first[channel] + second[channel]) // 3 for channel in range(3)),
        tuple((first[channel] + 2 * second[channel]) // 3 for channel in range(3)),
    )


def decode_bc3(block: bytes, pixel_x: int, pixel_y: int) -> tuple[int, int, int, int]:
    if len(block) != 16:
        raise ValueError("BC3 block must contain exactly 16 bytes")
    alpha0, alpha1 = block[0], block[1]
    alpha_bits = int.from_bytes(block[2:8], "little")
    if alpha0 > alpha1:
        alpha = [alpha0, alpha1] + [
            ((7 - index) * alpha0 + index * alpha1) // 7 for index in range(1, 7)
        ]
    else:
        alpha = [alpha0, alpha1] + [
            ((5 - index) * alpha0 + index * alpha1) // 5 for index in range(1, 5)
        ] + [0, 255]
    color0, color1, selectors = struct.unpack_from("<HHI", block, 8)
    palette = _color_palette(color0, color1)
    pixel = pixel_y * 4 + pixel_x
    color = palette[(selectors >> (pixel * 2)) & 3]
    return (*color, alpha[(alpha_bits >> (pixel * 3)) & 7])


class DdsUnitTexture:
    HEADER_BYTES = 148

    def __init__(
        self,
        data: bytes,
        address_u: str,
        address_v: str,
        tint_color: tuple[int, int, int] | None = None,
        tint_ramp: Sequence[Sequence[int]] | None = None,
        tint_strength: float = CIV_COLOR_STRENGTH,
        tint_mask_source: str = "base_color_alpha_inverse",
        representative_palette_index: int = 6,
    ) -> None:
        if len(data) < self.HEADER_BYTES or data[:4] != b"DDS " or data[84:88] != b"DX10":
            raise ValueError("unit texture is not a DX10 DDS")
        self.height = struct.unpack_from("<I", data, 12)[0]
        self.width = struct.unpack_from("<I", data, 16)[0]
        self.format = struct.unpack_from("<I", data, 128)[0]
        if self.format not in (71, 72, 77, 78):
            raise ValueError(f"unit preview supports BC1/BC3 base color, found DXGI {self.format}")
        if address_u not in ("clamp", "repeat") or address_v not in ("clamp", "repeat"):
            raise ValueError("unit texture has unsupported addressing")
        self.address_u = address_u
        self.address_v = address_v
        self.tint_color = tint_color
        self.tint_ramp = tint_ramp
        self.tint_strength = tint_strength
        if tint_mask_source not in {"base_color_alpha_inverse", "constant_one"}:
            raise ValueError("unit texture has an unsupported owner tint mask source")
        self.tint_mask_source = tint_mask_source
        if not isinstance(representative_palette_index, int) or not 0 <= representative_palette_index < 64:
            raise ValueError("unit texture has an invalid representative palette index")
        self.representative_palette_index = representative_palette_index
        self.block_bytes = 8 if self.format in (71, 72) else 16
        self.blocks_wide = max(1, (self.width + 3) // 4)
        required = self.blocks_wide * max(1, (self.height + 3) // 4) * self.block_bytes
        if self.HEADER_BYTES + required > len(data):
            raise ValueError("unit texture top mip extends past the DDS")
        self.data = data

    @classmethod
    def from_file(
        cls,
        path: Path,
        address_u: str,
        address_v: str,
        tint_color: tuple[int, int, int] | None = None,
        tint_ramp: Sequence[Sequence[int]] | None = None,
        tint_strength: float = CIV_COLOR_STRENGTH,
        tint_mask_source: str = "base_color_alpha_inverse",
        representative_palette_index: int = 6,
    ) -> "DdsUnitTexture":
        return cls(
            path.read_bytes(),
            address_u,
            address_v,
            tint_color,
            tint_ramp,
            tint_strength,
            tint_mask_source,
            representative_palette_index,
        )

    def sample(self, u: float, v: float) -> tuple[int, int, int, int]:
        u = u % 1.0 if self.address_u == "repeat" else max(0.0, min(1.0, u))
        v = v % 1.0 if self.address_v == "repeat" else max(0.0, min(1.0, v))
        x = min(self.width - 1, int(u * self.width))
        y = min(self.height - 1, int(v * self.height))
        offset = self.HEADER_BYTES + ((y // 4) * self.blocks_wide + x // 4) * self.block_bytes
        block = self.data[offset : offset + self.block_bytes]
        if self.block_bytes == 16:
            color = decode_bc3(block, x & 3, y & 3)
        else:
            color0, color1, selectors = struct.unpack("<HHI", block)
            rgb = _color_palette(color0, color1)[(selectors >> (2 * ((y & 3) * 4 + (x & 3)))) & 3]
            color = (*rgb, 255)
        # Unit diffuse alpha participates in tint semantics, not surface opacity.
        tint_alpha = 0 if self.tint_mask_source == "constant_one" else color[3]
        rgb = apply_owner_tint(
            color[:3],
            tint_alpha,
            self.tint_color,
            self.tint_ramp,
            self.tint_strength,
            self.representative_palette_index,
        )
        return (*rgb, 255)


def _multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        sum(a[row * 4 + inner] * b[inner * 4 + column] for inner in range(4))
        for row in range(4)
        for column in range(4)
    )


def _transform_point(value: Sequence[float], matrix: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = value
    return (
        x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12],
        x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13],
        x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14],
    )


def _transform_vector(value: Sequence[float], matrix: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = value
    result = (
        x * matrix[0] + y * matrix[4] + z * matrix[8],
        x * matrix[1] + y * matrix[5] + z * matrix[9],
        x * matrix[2] + y * matrix[6] + z * matrix[10],
    )
    length = math.sqrt(sum(component * component for component in result))
    return tuple(component / length for component in result) if length > 1.0e-12 else (0.0, 0.0, 1.0)


def _skinned_mesh(
    mesh: dict[str, Any], skeleton: dict[str, Any], worlds: Sequence[Sequence[float]]
) -> dict[str, Any]:
    positions = normalized_skin.skin_positions(mesh, skeleton, worlds)
    skin_matrices = [
        _multiply(bone["inverse_bind_matrix"], worlds[index])
        for index, bone in enumerate(skeleton["bones"])
    ]
    vertices = []
    for vertex, position in zip(mesh["vertices"], positions):
        normal = [0.0, 0.0, 0.0]
        for joint, weight in zip(vertex["joints"], vertex["weights"]):
            if weight:
                transformed = _transform_vector(vertex["normal"], skin_matrices[joint])
                for axis in range(3):
                    normal[axis] += transformed[axis] * weight
        length = math.sqrt(sum(component * component for component in normal))
        vertices.append(
            {
                "position": list(position),
                "normal": [component / length for component in normal] if length > 1.0e-12 else vertex["normal"],
                "uv0": vertex["uv0"],
            }
        )
    return {"topology": mesh["topology"], "vertices": vertices}


def _rigid_mesh(mesh: dict[str, Any], socket: Sequence[float], model_scale: float) -> dict[str, Any]:
    return {
        "topology": mesh["topology"],
        "vertices": [
            {
                "position": list(_transform_point([value * model_scale for value in vertex["position"]], socket)),
                "normal": list(_transform_vector(vertex["normal"], socket)),
                "uv0": vertex["uv0"],
            }
            for vertex in mesh["vertices"]
        ],
    }


def _translated(mesh: dict[str, Any], offset: tuple[float, float]) -> dict[str, Any]:
    return {
        "topology": mesh["topology"],
        "vertices": [
            {**vertex, "position": [vertex["position"][0] + offset[0], vertex["position"][1] + offset[1], vertex["position"][2]]}
            for vertex in mesh["vertices"]
        ],
    }


def _paste(destination: Canvas, source: Canvas, x: int, y: int) -> None:
    for row in range(source.height):
        start = row * source.width
        target = (y + row) * destination.width + x
        destination.pixels[target : target + source.width] = source.pixels[start : start + source.width]


def _load_components(
    pack: Path,
    manifest: dict[str, Any],
    owner_color: tuple[int, int, int] | None,
    owner_ramp: Sequence[Sequence[int]] | None,
) -> dict[str, dict[str, Any]]:
    result = {}
    for asset_id, asset in manifest["assets"].items():
        document = json.loads((pack / asset["component"]).read_text(encoding="utf-8"))
        skeleton = normalized_skin.load_skeleton(pack / document["skeleton"])
        mesh_path = pack / document["mesh"]
        mesh = (
            normalized_skin.load_mesh(mesh_path, len(skeleton["bones"]))
            if document["binding_mode"] == "vertex_skin"
            else json.loads(mesh_path.read_text(encoding="utf-8"))
        )
        material = json.loads((pack / document["material"]).read_text(encoding="utf-8"))
        base = material["channels"]["base_color"]
        owner_material = document.get("owner_color")
        if owner_material is None:
            owner_material = {
                "mode": "source_mask" if document.get("tint") == "USE_CIV_COLOR" else "none",
                "mask_source": "base_color_alpha_inverse",
                "strength": CIV_COLOR_STRENGTH,
            }
        use_owner_tint = owner_material["mode"] != "none"
        result[asset_id] = {
            "document": document,
            "skeleton": skeleton,
            "mesh": mesh,
            "texture": DdsUnitTexture.from_file(
                pack / base["texture"],
                base["address_u"],
                base["address_v"],
                owner_color if use_owner_tint else None,
                owner_ramp if use_owner_tint else None,
                owner_material["strength"],
                owner_material["mask_source"],
                owner_material.get("representative_palette_index", 6),
            ),
        }
    return result


def render_sheet(
    pack: Path,
    output: Path,
    formation: bool = False,
    owner_color: tuple[int, int, int] | None = None,
    owner_ramp: Sequence[Sequence[int]] | None = None,
    color_table_id: int | None = None,
) -> dict[str, Any]:
    manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "c3x.unit_pack.v0" or manifest.get("unit_binding", {}).get("status") != "validated_lab_cpu_skin":
        raise ValueError("unit pack has not passed action binding validation")
    recipe = json.loads((pack / manifest["units"]["unit/warrior"]["recipe"]).read_text(encoding="utf-8"))
    if (owner_color is None) == (owner_ramp is None) and owner_color is not None:
        raise ValueError("owner RGB and Civ III owner ramp are mutually exclusive")
    components = _load_components(pack, manifest, owner_color, owner_ramp)
    body = components["unit/warrior/body"]
    body_bone_names = [bone["name"] for bone in body["skeleton"]["bones"]]
    sockets = manifest["unit_binding"]["sockets"]
    cell_width, cell_height = 218, 238
    left, top = 74, 54
    width, height = left + cell_width * len(DIRECTIONS), top + cell_height * len(ACTIONS) + 34
    canvas = Canvas(width, height, BACKGROUND)
    member_scale = recipe["member"]["member_scale"] * recipe["member"]["variation_scale"]
    spacing_x = recipe["formation"]["spacing_x"] / 200.0
    spacing_y = recipe["formation"]["spacing_y"] / 200.0
    formation_offsets = (
        [(spacing_x, spacing_y), (spacing_x, -spacing_y), (-spacing_x, spacing_y), (-spacing_x, -spacing_y)]
        if formation
        else [(0.0, 0.0)]
    )
    render_scale = 470.0 if formation else 620.0
    draw_order = ("body", "head", "armor", "hair", "weapon")
    action_metrics = {}
    for row, action in enumerate(ACTIONS):
        action_info = manifest["animations"][f"animation/unit/warrior/{action}"]
        clip = normalized_animation.load_clip(pack / action_info["clip"])
        sample_time = clip.duration * ACTION_PHASE[action]
        worlds = {}
        for asset_id, component in components.items():
            if component["document"]["binding_mode"] != "vertex_skin":
                continue
            pose_info = action_info["pose_caches"][asset_id]
            pose_cache = normalized_pose_cache.load_pose_cache(pack / pose_info["path"])
            normalized_pose_cache.validate_skeleton_binding(pose_cache, component["skeleton"])
            worlds[asset_id] = pose_cache.sample(
                sample_time, action_info["loop"]
            )
        body_worlds = worlds["unit/warrior/body"]
        rendered_meshes = {}
        for role in draw_order:
            asset_id = f"unit/warrior/{role}"
            component = components[asset_id]
            document = component["document"]
            if document["binding_mode"] == "vertex_skin":
                rendered_meshes[role] = _skinned_mesh(component["mesh"], component["skeleton"], worlds[asset_id])
            else:
                socket_profile = sockets[document["attachment_point"]]
                socket_index = body_bone_names.index(socket_profile["bone"])
                rendered_meshes[role] = _rigid_mesh(
                    component["mesh"], body_worlds[socket_index], document["model_scale"]
                )
        for column, _direction in enumerate(DIRECTIONS):
            cell = Canvas(cell_width, cell_height, BACKGROUND)
            cell_depth = [-math.inf] * (cell_width * cell_height)
            center = (cell_width // 2, 151)
            cell.fill_polygon(
                [(center[0], center[1] - 27), (center[0] + 66, center[1]), (center[0], center[1] + 27), (center[0] - 66, center[1])],
                (47, 61, 53),
            )
            rotation = math.radians(column * 45.0)
            for member_offset in formation_offsets:
                for role in draw_order:
                    component = components[f"unit/warrior/{role}"]
                    draw_mesh(
                        cell,
                        cell_depth,
                        _translated(rendered_meshes[role], member_offset),
                        component["texture"],
                        center,
                        render_scale,
                        rotation,
                        model_scale=member_scale,
                    )
            _paste(canvas, cell, left + column * cell_width, top + row * cell_height)
        action_metrics[action] = {"sample_time": sample_time, "duration": clip.duration, "frame_count": clip.frame_count}
    title = (
        "UNIT WARRIOR - 4 MEMBER SOURCE FORMATION / OWNER TINT DIAGNOSTIC"
        if formation and (owner_color is not None or owner_ramp is not None)
        else (
            "UNIT WARRIOR - 4 MEMBER SOURCE FORMATION / NEUTRAL TINT DIAGNOSTIC"
            if formation
            else (
                f"UNIT WARRIOR - SINGLE MEMBER / CIV III COLOR TABLE {color_table_id:02d}"
                if owner_ramp is not None and color_table_id is not None
                else (
                    "UNIT WARRIOR - SINGLE MEMBER ACTION READABILITY / OWNER TINT DIAGNOSTIC"
                    if owner_color is not None
                    else "UNIT WARRIOR - SINGLE MEMBER ACTION READABILITY / NEUTRAL TINT DIAGNOSTIC"
                )
            )
        )
    )
    draw_text(canvas, 14, 10, title, (232, 224, 194), 2)
    for column, direction in enumerate(DIRECTIONS):
        draw_text(canvas, left + column * cell_width + cell_width // 2 - 8, 35, direction, (164, 203, 188), 1)
    for row, action in enumerate(ACTIONS):
        draw_text(canvas, 8, top + row * cell_height + 118, action, (164, 203, 188), 1)
    footer = (
        "LAB ONLY - EXACT CIV III RAMP / CIV VI OWNER MASK MAPPING INFERRED / SOCKETS INFERRED"
        if owner_ramp is not None
        else (
            "LAB ONLY - RGB DIAGNOSTIC / OWNER ALPHA MASK INFERRED / SOCKETS INFERRED"
            if owner_color is not None
            else "LAB ONLY - REAL SOURCE MESHES AND CLIPS / SOCKETS INFERRED / OWNER TINT NOT RENDERED"
        )
    )
    draw_text(canvas, 14, height - 18, footer, (190, 190, 180), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_png(canvas, output)
    return {
        "output": str(output),
        "width": width,
        "height": height,
        "directions": list(DIRECTIONS),
        "actions": action_metrics,
        "members_per_cell": len(formation_offsets),
        "formation_layout": (
            "four-corner block derived from source SpacingX/SpacingY"
            if formation
            else "single member action readability fixture"
        ),
        "non_background_pixels": canvas.non_background_pixels(BACKGROUND),
        "owner_tint": (
            {
                "status": "exact_civ3_palette_with_inferred_civ6_mask_mapping",
                "color_table_id": color_table_id,
                "primary_ramp": [list(color) for color in owner_ramp],
            }
            if owner_ramp is not None
            else (
                {"status": "inferred_alpha_mask_visualized", "color": list(owner_color)}
                if owner_color is not None
                else {"status": "preserved_unrendered"}
            )
        ),
        "socket_status": "inferred_lab_profile",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--formation", action="store_true")
    parser.add_argument("--owner-color", help="diagnostic owner RGB as r,g,b")
    parser.add_argument("--owner-palette-pack", type=Path, help="compiled Civ III owner-color pack")
    parser.add_argument("--color-table-id", type=int, help="Civ III Leader.Color_Table_ID (0..31)")
    args = parser.parse_args(argv)
    try:
        owner_color = None
        if args.owner_color:
            values = tuple(int(value) for value in args.owner_color.split(","))
            if len(values) != 3 or any(value < 0 or value > 255 for value in values):
                raise ValueError("--owner-color must be three bytes formatted r,g,b")
            owner_color = values
        owner_ramp = None
        color_table_id = None
        if args.owner_palette_pack is not None or args.color_table_id is not None:
            if args.owner_palette_pack is None or args.color_table_id is None:
                raise ValueError("--owner-palette-pack and --color-table-id must be supplied together")
            if owner_color is not None:
                raise ValueError("--owner-color cannot be combined with --owner-palette-pack")
            color_table_id = args.color_table_id
            owner_ramp = load_owner_color_table(args.owner_palette_pack, color_table_id)["primary_ramp"]
        report = render_sheet(
            args.pack,
            args.output,
            args.formation,
            owner_color,
            owner_ramp,
            color_table_id,
        )
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.output} ({report['non_background_pixels']} rendered pixels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
