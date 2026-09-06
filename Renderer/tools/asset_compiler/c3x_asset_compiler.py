#!/usr/bin/env python3
"""Small source-agnostic C3X renderer asset compiler spike.

Source-specific discovery and conversion live here; emitted runtime packs use
ordinary DDS resources and generic C3X metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_CIV6_BASE = Path(
    r"Z:\Library\Application Support\Steam\steamapps\common\Sid Meier's Civilization VI"
    r"\Civ6.app\Contents\Assets\Base"
)

DEFAULT_RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = DEFAULT_RENDERER_ROOT / "docs" / "civ6_asset_report.json"
DEFAULT_PACK = DEFAULT_RENDERER_ROOT / "packs" / "Civ6Prototype"
DEFAULT_LOOSE_SOURCE = DEFAULT_RENDERER_ROOT / "samples" / "ManualLooseSource" / "source_manifest.json"
DEFAULT_LOOSE_PACK = DEFAULT_RENDERER_ROOT / "packs" / "ManualLoosePrototype"
DEFAULT_GRASSLAND_POC_PACK = DEFAULT_RENDERER_ROOT / "packs" / "Civ6GrasslandPOC"
DEFAULT_TEXCONV = DEFAULT_RENDERER_ROOT / "third_party" / "CivNexus6" / "bin" / "Release" / "texconv.exe"

CIVBIG_HEADER_SIZE = 48
DDS_DX10_HEADER_SIZE = 148
BC_BLOCK_BYTES = {
    71: 8,   # DXGI_FORMAT_BC1_UNORM
    72: 8,   # DXGI_FORMAT_BC1_UNORM_SRGB
    74: 16,  # DXGI_FORMAT_BC2_UNORM
    75: 16,  # DXGI_FORMAT_BC2_UNORM_SRGB
    77: 16,  # DXGI_FORMAT_BC3_UNORM
    78: 16,  # DXGI_FORMAT_BC3_UNORM_SRGB
    80: 8,   # DXGI_FORMAT_BC4_UNORM
    81: 8,   # DXGI_FORMAT_BC4_SNORM
    83: 16,  # DXGI_FORMAT_BC5_UNORM
    84: 16,  # DXGI_FORMAT_BC5_SNORM
    95: 16,  # DXGI_FORMAT_BC6H_UF16
    96: 16,  # DXGI_FORMAT_BC6H_SF16
    98: 16,  # DXGI_FORMAT_BC7_UNORM
    99: 16,  # DXGI_FORMAT_BC7_UNORM_SRGB
}
LINEAR_BYTES_PER_PIXEL = {
    10: 8,  # DXGI_FORMAT_R16G16B16A16_FLOAT
    11: 8,  # DXGI_FORMAT_R16G16B16A16_UNORM
    35: 4,  # DXGI_FORMAT_R16G16_UNORM
}
DXGI_FORMAT_NAMES = {
    10: "R16G16B16A16_FLOAT",
    11: "R16G16B16A16_UNORM",
    35: "R16G16_UNORM",
    71: "BC1_UNORM",
    72: "BC1_UNORM_SRGB",
    74: "BC2_UNORM",
    75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM",
    78: "BC3_UNORM_SRGB",
    80: "BC4_UNORM",
    81: "BC4_SNORM",
    83: "BC5_UNORM",
    84: "BC5_SNORM",
    95: "BC6H_UF16",
    96: "BC6H_SF16",
    98: "BC7_UNORM",
    99: "BC7_UNORM_SRGB",
}

TERRAIN_KEYS = ("grass", "plains", "desert", "tundra")
TOKEN_RE = re.compile(r"[A-Za-z0-9_./\\-]+")
CANDIDATE_RE = re.compile(
    r"(terrain|grass|plains|desert|tundra|snow|hill|hills|mountain|mountains|cliff|rock|landmark)",
    re.IGNORECASE,
)
ARTDEF_BLP_ENTRY_RE = re.compile(
    r'<m_EntryName text="([^"]+)"/>.*?<m_XLPClass text="([^"]*)"/>.*?'
    r'<m_XLPPath text="([^"]*)"/>.*?<m_BLPPackage text="([^"]*)"/>.*?'
    r'<m_LibraryName text="([^"]*)"/>.*?<m_ParamName text="([^"]*)"/>',
    re.DOTALL,
)
ASCII_STRING_RE = re.compile(rb"[\x20-\x7E]{4,}")

CIV3_SQUARE_TYPES = {
    "desert": 0,
    "plains": 1,
    "grassland": 2,
    "tundra": 3,
    "floodplain": 4,
    "hills": 5,
    "mountains": 6,
    "forest": 7,
    "jungle": 8,
    "swamp": 9,
    "volcano": 10,
    "coast": 11,
    "sea": 12,
    "ocean": 13,
}


def rel_or_abs(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def read_text_lossy(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes()
    if limit is not None:
        data = data[:limit]
    return data.decode("utf-8", errors="ignore")


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def classify_blp_asset(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()
    if suffix == ".blp":
        return "cooked_blp_package"
    if name.startswith("texture_"):
        return "cooked_texture"
    if name.startswith("animation_"):
        return "cooked_animation"
    if name.startswith("geometry_") or name.startswith("model_"):
        return "cooked_geometry"
    return "cooked_or_unknown"


def scan_artdef(path: Path, root: Path) -> dict[str, Any]:
    text = read_text_lossy(path)
    tokens = TOKEN_RE.findall(text)
    candidates: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if CANDIDATE_RE.search(token) and token not in seen:
            seen.add(token)
            candidates.append(token)
        if len(candidates) >= 200:
            break
    return {
        "path": rel_or_abs(path, root),
        "bytes": path.stat().st_size,
        "candidate_tokens": candidates,
    }


def parse_artdef_blp_entries(path: Path, civ6_base: Path) -> list[dict[str, Any]]:
    text = read_text_lossy(path)
    entries = []
    for match in ARTDEF_BLP_ENTRY_RE.finditer(text):
        entry_name, xlp_class, xlp_path, blp_package, library_name, param_name = match.groups()
        entries.append(
            {
                "artdef": rel_or_abs(path, civ6_base),
                "entry_name": entry_name,
                "xlp_class": xlp_class,
                "xlp_path": xlp_path,
                "blp_package": blp_package,
                "library_name": library_name,
                "param_name": param_name,
            }
        )
    return entries


def extract_ascii_strings(path: Path, include: re.Pattern[str] | None = None, limit: int = 300) -> list[str]:
    data = path.read_bytes()
    strings: list[str] = []
    seen: set[str] = set()
    for match in ASCII_STRING_RE.finditer(data):
        text = match.group(0).decode("ascii", errors="ignore").strip()
        if include is not None and not include.search(text):
            continue
        if text not in seen:
            seen.add(text)
            strings.append(text)
        if len(strings) >= limit:
            break
    return strings


def inspect_blp_package(path: Path, civ6_base: Path, include: re.Pattern[str] | None = None) -> dict[str, Any]:
    exists = path.exists()
    header = ""
    strings: list[str] = []
    if exists:
        with path.open("rb") as f:
            header = f.read(16).hex(" ")
        strings = extract_ascii_strings(path, include=include)
    return {
        "path": rel_or_abs(path, civ6_base),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
        "header_hex": header,
        "strings": strings,
    }


def texture_level_layout(width: int, height: int, dxgi_format: int) -> tuple[int, int]:
    """Return row pitch and byte size for one 2D texture level."""
    if dxgi_format in BC_BLOCK_BYTES:
        blocks_wide = max(1, (width + 3) // 4)
        blocks_high = max(1, (height + 3) // 4)
        pitch = blocks_wide * BC_BLOCK_BYTES[dxgi_format]
        return pitch, pitch * blocks_high
    if dxgi_format in LINEAR_BYTES_PER_PIXEL:
        pitch = width * LINEAR_BYTES_PER_PIXEL[dxgi_format]
        return pitch, pitch * height
    raise ValueError(f"Unsupported CIVBIG DXGI format {dxgi_format}")


def expected_texture_bytes(width: int, height: int, mip_count: int, dxgi_format: int) -> int:
    total = 0
    for _ in range(mip_count):
        total += texture_level_layout(width, height, dxgi_format)[1]
        width = max(1, width // 2)
        height = max(1, height // 2)
    return total


def parse_civbig_header(data: bytes) -> dict[str, int]:
    if len(data) < CIVBIG_HEADER_SIZE:
        raise ValueError(f"CIVBIG file is shorter than its {CIVBIG_HEADER_SIZE}-byte header")
    if data[:6] != b"CIVBIG":
        raise ValueError("File does not start with the CIVBIG signature")

    payload_bytes = struct.unpack_from("<I", data, 8)[0]
    resource_count, mip_count, dxgi_format, width, height, array_size = struct.unpack_from("<6H", data, 32)
    if not width or not height or not mip_count:
        raise ValueError("CIVBIG header has zero width, height, or mip count")
    if dxgi_format not in BC_BLOCK_BYTES and dxgi_format not in LINEAR_BYTES_PER_PIXEL:
        raise ValueError(f"Unsupported CIVBIG DXGI format {dxgi_format}")
    if resource_count != 1 or array_size != 1:
        raise ValueError(
            f"Unsupported CIVBIG resource layout: resource_count={resource_count}, array_size={array_size}"
        )

    expected_payload_bytes = expected_texture_bytes(width, height, mip_count, dxgi_format)
    if payload_bytes != expected_payload_bytes:
        raise ValueError(
            f"CIVBIG payload size {payload_bytes} does not match the expected mip chain size "
            f"{expected_payload_bytes}"
        )
    if CIVBIG_HEADER_SIZE + payload_bytes > len(data):
        raise ValueError("CIVBIG payload extends past the end of the file")

    return {
        "width": width,
        "height": height,
        "mip_count": mip_count,
        "dxgi_format": dxgi_format,
        "format_name": DXGI_FORMAT_NAMES[dxgi_format],
        "payload_bytes": payload_bytes,
        "trailing_padding_bytes": len(data) - CIVBIG_HEADER_SIZE - payload_bytes,
    }


def make_dds_dx10_header(info: dict[str, int]) -> bytes:
    width = info["width"]
    height = info["height"]
    mip_count = info["mip_count"]
    top_level_pitch, top_level_bytes = texture_level_layout(width, height, info["dxgi_format"])
    pitch_or_linear_size = top_level_bytes if info["dxgi_format"] in BC_BLOCK_BYTES else top_level_pitch

    flags = 0x1 | 0x2 | 0x4 | 0x1000
    flags |= 0x80000 if info["dxgi_format"] in BC_BLOCK_BYTES else 0x8
    if mip_count > 1:
        flags |= 0x20000
    dds_caps = 0x1000
    if mip_count > 1:
        dds_caps |= 0x8 | 0x400000

    header_values = [
        124,
        flags,
        height,
        width,
        pitch_or_linear_size,
        0,
        mip_count,
        *([0] * 11),
        32,
        0x4,
        int.from_bytes(b"DX10", "little"),
        0,
        0,
        0,
        0,
        0,
        dds_caps,
        0,
        0,
        0,
        0,
    ]
    dds_header = b"DDS " + struct.pack("<31I", *header_values)
    dx10_header = struct.pack("<5I", info["dxgi_format"], 3, 0, 1, 0)
    return dds_header + dx10_header


def extract_civbig_to_dds(source: Path, output: Path) -> dict[str, Any]:
    data = source.read_bytes()
    info = parse_civbig_header(data)
    payload_end = CIVBIG_HEADER_SIZE + info["payload_bytes"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(make_dds_dx10_header(info) + data[CIVBIG_HEADER_SIZE:payload_end])
    return {
        "schema": "c3x.civbig_extract.v0",
        "source": str(source),
        "output": str(output),
        "dds_bytes": output.stat().st_size,
        **info,
    }


def convert_dds_to_png(dds_path: Path, texconv_path: Path = DEFAULT_TEXCONV) -> tuple[Path | None, str | None]:
    if not texconv_path.exists():
        return None, f"DirectXTex converter not found: {texconv_path}"
    result = subprocess.run(
        [
            str(texconv_path),
            "-nologo",
            "-y",
            "-m",
            "1",
            "-ft",
            "png",
            "-o",
            str(dds_path.parent),
            str(dds_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    png_path = dds_path.with_suffix(".png")
    generated_png = next(
        (
            path
            for path in dds_path.parent.iterdir()
            if path.stem.lower() == dds_path.stem.lower() and path.suffix.lower() == ".png"
        ),
        None,
    )
    if result.returncode != 0 or generated_png is None:
        detail = (result.stderr or result.stdout).strip()
        return None, f"DirectXTex conversion failed ({result.returncode}): {detail}"
    if generated_png.name != png_path.name:
        temporary_name = generated_png.with_name(f"{generated_png.name}.case-normalize")
        generated_png.replace(temporary_name)
        temporary_name.replace(png_path)
    return png_path, None


def discover(civ6_base: Path) -> dict[str, Any]:
    artdefs_dir = civ6_base / "ArtDefs"
    blps_dir = civ6_base / "Platforms" / "Windows" / "BLPs"
    dep_path = civ6_base / "Civ6.dep"

    artdefs = [scan_artdef(p, civ6_base) for p in sorted(artdefs_dir.glob("*.artdef"))]
    blp_files = iter_files(blps_dir)
    blp_counts = Counter(classify_blp_asset(p) for p in blp_files)

    blp_candidates = []
    for path in sorted(blp_files):
        if CANDIDATE_RE.search(path.name):
            blp_candidates.append(
                {
                    "path": rel_or_abs(path, civ6_base),
                    "bytes": path.stat().st_size,
                    "classification": classify_blp_asset(path),
                }
            )
        if len(blp_candidates) >= 250:
            break

    dep_summary: dict[str, Any] = {"exists": dep_path.exists()}
    if dep_path.exists():
        dep_text = read_text_lossy(dep_path)
        dep_summary.update(
            {
                "path": rel_or_abs(dep_path, civ6_base),
                "bytes": dep_path.stat().st_size,
                "mentions": {
                    "Terrains.artdef": dep_text.count("Terrains.artdef"),
                    "TerrainStyle.artdef": dep_text.count("TerrainStyle.artdef"),
                    "Landmarks.artdef": dep_text.count("Landmarks.artdef"),
                    "Clutter.artdef": dep_text.count("Clutter.artdef"),
                },
            }
        )

    loose_sources = []
    for pattern in ("*.fgx", "*.cn6", "*.glb", "*.gltf", "*.dds", "*.mtl", "*.tex", "*.geo"):
        loose_sources.extend(civ6_base.rglob(pattern))
    loose_sources = sorted(set(loose_sources))

    return {
        "schema": "c3x.civ6.discovery.v0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "civ6_base": str(civ6_base),
        "artdefs": artdefs,
        "civ6_dep": dep_summary,
        "blp_tree": {
            "path": rel_or_abs(blps_dir, civ6_base),
            "exists": blps_dir.exists(),
            "file_count": len(blp_files),
            "classification_counts": dict(sorted(blp_counts.items())),
            "candidate_assets": blp_candidates,
        },
        "loose_source_candidates": [
            {
                "path": rel_or_abs(p, civ6_base),
                "bytes": p.stat().st_size,
            }
            for p in loose_sources[:250]
        ],
        "diagnostics": build_diagnostics(civ6_base, loose_sources, blp_files),
    }


def build_diagnostics(civ6_base: Path, loose_sources: list[Path], blp_files: list[Path]) -> list[str]:
    diagnostics = []
    if not civ6_base.exists():
        diagnostics.append(f"Civ VI base path does not exist: {civ6_base}")
    if blp_files:
        diagnostics.append("Found cooked BLP/platform assets; these are indexed but not copied into C3X packs.")
    if not loose_sources:
        diagnostics.append("No loose .fgx/.cn6/.glb/.dds/.mtl/.tex/.geo sources found under the Civ VI base path.")
        diagnostics.append("Real model conversion likely needs Civ VI SDK/Pantry loose assets or permissioned mod source files.")
    blender_found = any(
        os.access(str(Path(folder) / exe), os.X_OK)
        for folder in os.environ.get("PATH", "").split(os.pathsep)
        for exe in ("blender.exe", "blender")
    )
    if not blender_found:
        diagnostics.append("Blender was not found on PATH; headless .cn6 -> .glb conversion is not available yet.")
    return diagnostics


def prototype_manifest(report: dict[str, Any] | None) -> dict[str, Any]:
    source_candidates: dict[str, list[str]] = {key: [] for key in TERRAIN_KEYS}
    mountain_candidates: list[str] = []
    if report:
        for item in report.get("blp_tree", {}).get("candidate_assets", []):
            path = item["path"]
            lower = path.lower()
            for key in TERRAIN_KEYS:
                if key in lower:
                    source_candidates[key].append(path)
            if "mountain" in lower or "rock" in lower or "cliff" in lower:
                mountain_candidates.append(path)

    terrains = {
        "grassland": {
            "material": "materials/grassland.json",
            "preview_color": [83, 143, 79],
            "civ6_source_candidates": source_candidates["grass"][:12],
        },
        "plains": {
            "material": "materials/plains.json",
            "preview_color": [178, 157, 88],
            "civ6_source_candidates": source_candidates["plains"][:12],
        },
        "desert": {
            "material": "materials/desert.json",
            "preview_color": [202, 176, 103],
            "civ6_source_candidates": source_candidates["desert"][:12],
        },
        "tundra": {
            "material": "materials/tundra.json",
            "preview_color": [132, 151, 139],
            "civ6_source_candidates": source_candidates["tundra"][:12],
        },
    }

    mountains = []
    for index in range(5):
        mountains.append(
            {
                "id": f"mountain_{index + 1:02d}",
                "model": None,
                "placeholder": f"mountains/mountain_{index + 1:02d}.json",
                "weight": 1,
                "preview_height": 0.78 + index * 0.06,
                "preview_color": [105 + index * 7, 108 + index * 5, 100 + index * 4],
            }
        )

    return {
        "schema": "c3x.asset_pack.v0",
        "name": "Civ6Prototype",
        "display_name": "Civ VI Prototype Discovery Pack",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Local prototype only. Do not redistribute Firaxis assets.",
        "projection": {
            "tile_width_px": 128,
            "tile_height_px": 64,
            "height_scale_px": 54,
            "basis": {
                "x": [64, 32],
                "y": [-64, 32],
                "z": [0, -54],
            },
        },
        "terrains": terrains,
        "relief": {
            "mountains": {
                "selection": "deterministic_hash(tile_x,tile_y,world_seed)",
                "civ6_source_candidates": mountain_candidates[:24],
                "variants": mountains,
            }
        },
        "diagnostics": (report or {}).get("diagnostics", []),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_material(path: Path, name: str, color: list[int], candidates: list[str]) -> None:
    write_json(
        path,
        {
            "schema": "c3x.material.v0",
            "name": name,
            "albedo": None,
            "normal": None,
            "roughness": None,
            "height": None,
            "preview_color": color,
            "civ6_source_candidates": candidates,
            "status": "procedural_preview_placeholder",
        },
    )


def write_civ3_tile_art_map(path: Path, rules: list[dict[str, Any]]) -> None:
    write_json(
        path,
        {
            "schema": "c3x.civ3_tile_art_map.v0",
            "description": "Rules map Civ III tile renderer metadata to C3X terrain/material assets. sheet_index and sprite_index are nullable wildcards from Tile.SquareParts.",
            "fields": {
                "square_type": "Civ III SquareTypes enum integer from Tile::m50_Get_Square_BaseType",
                "real_type": "Optional Civ III real/underlying terrain byte from Tile::m49_Get_Square_RealType",
                "sheet_index": "Optional (Tile.SquareParts >> 8) & 0xFF",
                "sprite_index": "Optional Tile.SquareParts & 0xFF",
            },
            "rules": rules,
        },
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_color(value: Any, fallback: list[int]) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        return fallback
    color = []
    for item in value:
        if not isinstance(item, int):
            return fallback
        color.append(max(0, min(255, item)))
    return color


def maybe_copy_asset(source_root: Path, pack_dir: Path, rel_path: str | None, diagnostics: list[str], copy_assets: bool) -> str | None:
    if not rel_path:
        return None
    source_path = source_root / rel_path
    if not source_path.exists():
        diagnostics.append(f"Missing loose asset referenced by source manifest: {rel_path}")
        return rel_path.replace("\\", "/")
    if not copy_assets:
        return rel_path.replace("\\", "/")
    dest = pack_dir / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest)
    return rel_path.replace("\\", "/")


def loose_material_to_pack(
    source_root: Path,
    pack_dir: Path,
    material: dict[str, Any],
    terrain_name: str,
    diagnostics: list[str],
    copy_assets: bool,
) -> dict[str, Any]:
    preview_defaults = {
        "grassland": [83, 143, 79],
        "plains": [178, 157, 88],
        "desert": [202, 176, 103],
        "tundra": [132, 151, 139],
    }
    fallback = preview_defaults.get(terrain_name, [128, 128, 128])
    return {
        "schema": "c3x.material.v0",
        "name": terrain_name,
        "albedo": maybe_copy_asset(source_root, pack_dir, material.get("albedo"), diagnostics, copy_assets),
        "normal": maybe_copy_asset(source_root, pack_dir, material.get("normal"), diagnostics, copy_assets),
        "roughness": maybe_copy_asset(source_root, pack_dir, material.get("roughness"), diagnostics, copy_assets),
        "height": maybe_copy_asset(source_root, pack_dir, material.get("height"), diagnostics, copy_assets),
        "preview_color": normalize_color(material.get("preview_color"), fallback),
        "status": "loose_source_import",
    }


def build_from_loose(source_manifest_path: Path, pack_dir: Path, copy_assets: bool = True) -> dict[str, Any]:
    source_manifest_path = source_manifest_path.resolve()
    source_root = source_manifest_path.parent
    source = load_json(source_manifest_path)
    diagnostics: list[str] = []

    if source.get("schema") != "c3x.loose_source.v0":
        diagnostics.append("Source manifest schema is not c3x.loose_source.v0; attempting best-effort import.")

    terrains: dict[str, Any] = {}
    source_terrains = source.get("terrains", {})
    if not isinstance(source_terrains, dict):
        source_terrains = {}
        diagnostics.append("Source manifest terrains field is missing or not an object.")

    for terrain_name, terrain_source in source_terrains.items():
        if not isinstance(terrain_source, dict):
            diagnostics.append(f"Skipping terrain {terrain_name}: expected object.")
            continue
        material_path = f"materials/{terrain_name}.json"
        material = loose_material_to_pack(source_root, pack_dir, terrain_source, terrain_name, diagnostics, copy_assets)
        write_json(pack_dir / material_path, material)
        terrains[terrain_name] = {
            "material": material_path,
            "preview_color": material["preview_color"],
        }

    source_mountains = source.get("relief", {}).get("mountains", {}).get("variants", [])
    if not isinstance(source_mountains, list):
        source_mountains = []
        diagnostics.append("Source manifest relief.mountains.variants field is missing or not a list.")

    mountain_variants = []
    for index, variant in enumerate(source_mountains):
        if not isinstance(variant, dict):
            diagnostics.append(f"Skipping mountain variant {index}: expected object.")
            continue
        variant_id = str(variant.get("id", f"mountain_{index + 1:02d}"))
        model = maybe_copy_asset(source_root, pack_dir, variant.get("model"), diagnostics, copy_assets)
        mountain_variants.append(
            {
                "id": variant_id,
                "model": model,
                "weight": int(variant.get("weight", 1)),
                "preview_height": float(variant.get("preview_height", 0.9)),
                "preview_color": normalize_color(variant.get("preview_color"), [118, 118, 108]),
            }
        )

    if not terrains:
        diagnostics.append("No terrains were imported from loose source manifest.")
    if not mountain_variants:
        diagnostics.append("No mountain variants were imported from loose source manifest.")

    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": str(source.get("name", pack_dir.name)),
        "display_name": str(source.get("display_name", source.get("name", pack_dir.name))),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": str(source.get("source_policy", "Source-agnostic loose asset import.")),
        "source_manifest": str(source_manifest_path),
        "projection": source.get(
            "projection",
            {
                "tile_width_px": 128,
                "tile_height_px": 64,
                "height_scale_px": 54,
                "basis": {"x": [64, 32], "y": [-64, 32], "z": [0, -54]},
            },
        ),
        "terrains": terrains,
        "relief": {
            "mountains": {
                "selection": "deterministic_hash(tile_x,tile_y,world_seed)",
                "variants": mountain_variants,
            }
        },
        "diagnostics": diagnostics,
    }
    write_json(pack_dir / "manifest.json", manifest)
    write_json(
        pack_dir / "source_report_summary.json",
        {
            "schema": "c3x.loose_source_report_summary.v0",
            "source_manifest": str(source_manifest_path),
            "diagnostics": diagnostics,
            "terrain_count": len(terrains),
            "mountain_variant_count": len(mountain_variants),
            "copied_assets": copy_assets,
        },
    )
    return manifest


def build_grassland_poc(civ6_base: Path, pack_dir: Path) -> dict[str, Any]:
    artdef_path = civ6_base / "ArtDefs" / "TerrainStyle.artdef"
    all_entries = parse_artdef_blp_entries(artdef_path, civ6_base) if artdef_path.exists() else []
    grassland_entries = [
        entry
        for entry in all_entries
        if "GRASSLAND" in entry["entry_name"].upper() or "GRASSLAND" in entry["param_name"].upper()
    ]
    terrain_material_package = civ6_base / "Platforms" / "Windows" / "BLPs" / "terrain" / "TerrainMaterialSet_Base.blp"
    shared_data_dir = civ6_base / "Platforms" / "Windows" / "BLPs" / "SHARED_DATA"
    related_cooked_assets = [
        {
            "path": rel_or_abs(path, civ6_base),
            "bytes": path.stat().st_size,
            "classification": classify_blp_asset(path),
        }
        for path in sorted(iter_files(shared_data_dir))
        if "GRASS" in path.name.upper() or "TERRAIN_GRASS" in path.name.upper()
    ][:80]

    package_index = inspect_blp_package(
        terrain_material_package,
        civ6_base,
        include=re.compile(r"GRASS|TerrainMaterial|TextureEntry|ART_DEF", re.IGNORECASE),
    )
    diagnostics = []
    if not artdef_path.exists():
        diagnostics.append(f"Missing Civ VI TerrainStyle artdef: {artdef_path}")
    if not grassland_entries:
        diagnostics.append("No grassland BLP entries found in TerrainStyle.artdef.")
    if not package_index["exists"]:
        diagnostics.append(f"Missing terrain material package: {terrain_material_package}")
    grassland_texture_source = shared_data_dir / "TEXTURE_TER_Grass_Decal_B"
    grassland_dds = pack_dir / "textures" / "grassland_decal_b.dds"
    grassland_png: Path | None = None
    texture_extract: dict[str, Any] | None = None
    if grassland_texture_source.exists():
        try:
            texture_extract = extract_civbig_to_dds(grassland_texture_source, grassland_dds)
            grassland_png, conversion_error = convert_dds_to_png(grassland_dds)
            if conversion_error:
                diagnostics.append(conversion_error)
        except (OSError, ValueError) as exc:
            diagnostics.append(f"Could not extract grassland CIVBIG texture: {exc}")
    else:
        diagnostics.append(f"Missing grassland CIVBIG texture: {grassland_texture_source}")

    write_json(
        pack_dir / "civ6_grassland_sources.json",
        {
            "schema": "c3x.civ6_grassland_sources.v0",
            "civ6_base": str(civ6_base),
            "artdef_entries": grassland_entries,
            "terrain_material_package": package_index,
            "related_cooked_assets": related_cooked_assets,
            "texture_extract": texture_extract,
        },
    )
    write_json(
        pack_dir / "materials" / "grassland.json",
        {
            "schema": "c3x.material.v0",
            "name": "grassland",
            "albedo": "textures/grassland_decal_b.png" if grassland_png else (
                "textures/grassland_decal_b.dds" if texture_extract else None
            ),
            "normal": None,
            "roughness": None,
            "height": None,
            "preview_color": [74, 139, 78],
            "civ6_artdef_entries": grassland_entries,
            "civ6_related_cooked_assets": related_cooked_assets[:20],
            "status": "civ6_cooked_texture_extracted" if texture_extract else "civ6_cooked_reference_poc",
        },
    )
    write_civ3_tile_art_map(
        pack_dir / "civ3_tile_art_map.json",
        [
            {
                "id": "grassland_all_civ3_variants",
                "match": {
                    "square_type": CIV3_SQUARE_TYPES["grassland"],
                    "square_type_name": "SQ_Grassland",
                    "real_type": None,
                    "sheet_index": None,
                    "sprite_index": None,
                },
                "terrain": "grassland",
                "material": "materials/grassland.json",
                "civ6_reference": "ART_DEF_TERRAIN_MATERIAL_GRASSLAND",
                "priority": 100,
            },
            {
                "id": "grassland_sheet_sprite_example",
                "match": {
                    "square_type": CIV3_SQUARE_TYPES["grassland"],
                    "square_type_name": "SQ_Grassland",
                    "real_type": None,
                    "sheet_index": 2,
                    "sprite_index": 0,
                },
                "terrain": "grassland",
                "material": "materials/grassland.json",
                "civ6_reference": "ART_DEF_TERRAIN_MATERIAL_GRASSLAND",
                "priority": 200,
                "note": "Example of the more specific mapping shape needed once captured Civ III SquareParts variants are cataloged.",
            },
        ],
    )

    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "Civ6GrasslandPOC",
        "display_name": "Civ VI Grassland POC",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_policy": "Local prototype only. Extracted Civ VI art must not be redistributed.",
        "projection": {
            "tile_width_px": 128,
            "tile_height_px": 64,
            "height_scale_px": 54,
            "basis": {"x": [64, 32], "y": [-64, 32], "z": [0, -54]},
        },
        "terrains": {
            "grassland": {
                "material": "materials/grassland.json",
                "preview_color": [74, 139, 78],
                "civ6_reference": "ART_DEF_TERRAIN_MATERIAL_GRASSLAND",
            }
        },
        "civ3_tile_art_map": "civ3_tile_art_map.json",
        "relief": {
            "mountains": {
                "selection": "none",
                "variants": [],
            }
        },
        "diagnostics": diagnostics,
    }
    write_json(pack_dir / "manifest.json", manifest)
    write_json(
        pack_dir / "source_report_summary.json",
        {
            "schema": "c3x.grassland_poc_report_summary.v0",
            "diagnostics": diagnostics,
            "artdef_entry_count": len(grassland_entries),
            "related_cooked_asset_count": len(related_cooked_assets),
            "package_string_count": len(package_index["strings"]),
            "texture_extract": texture_extract,
        },
    )
    return manifest


def build_prototype(civ6_base: Path, report_path: Path, pack_dir: Path) -> dict[str, Any]:
    report = discover(civ6_base)
    write_json(report_path, report)

    manifest = prototype_manifest(report)
    write_json(pack_dir / "manifest.json", manifest)

    for terrain_name, terrain in manifest["terrains"].items():
        write_material(
            pack_dir / terrain["material"],
            terrain_name,
            terrain["preview_color"],
            terrain["civ6_source_candidates"],
        )

    for mountain in manifest["relief"]["mountains"]["variants"]:
        write_json(
            pack_dir / mountain["placeholder"],
            {
                "schema": "c3x.placeholder_mountain.v0",
                "id": mountain["id"],
                "model": None,
                "preview_height": mountain["preview_height"],
                "preview_color": mountain["preview_color"],
                "status": "awaiting_normalized_glb_source",
            },
        )

    write_json(
        pack_dir / "source_report_summary.json",
        {
            "schema": "c3x.source_report_summary.v0",
            "report": str(report_path),
            "diagnostics": report["diagnostics"],
            "candidate_asset_count": len(report["blp_tree"]["candidate_assets"]),
            "loose_source_candidate_count": len(report["loose_source_candidates"]),
        },
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C3X renderer asset compiler spike")
    parser.add_argument("--civ6-base", type=Path, default=DEFAULT_CIV6_BASE)
    sub = parser.add_subparsers(dest="command", required=True)

    discover_cmd = sub.add_parser("discover", help="inventory Civ VI art metadata")
    discover_cmd.add_argument("--output", type=Path, default=DEFAULT_REPORT)

    build_cmd = sub.add_parser("build-prototype", help="create the local prototype pack")
    build_cmd.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    build_cmd.add_argument("--pack", type=Path, default=DEFAULT_PACK)

    loose_cmd = sub.add_parser("import-loose", help="build a C3X pack from a loose source manifest")
    loose_cmd.add_argument("--source", type=Path, default=DEFAULT_LOOSE_SOURCE)
    loose_cmd.add_argument("--pack", type=Path, default=DEFAULT_LOOSE_PACK)
    loose_cmd.add_argument("--no-copy-assets", action="store_true")

    grass_cmd = sub.add_parser("build-grassland-poc", help="build a Civ VI grassland metadata/mapping proof of concept")
    grass_cmd.add_argument("--pack", type=Path, default=DEFAULT_GRASSLAND_POC_PACK)

    civbig_cmd = sub.add_parser("extract-civbig", help="extract a standalone cooked CIVBIG texture to DDS")
    civbig_cmd.add_argument("source", type=Path)
    civbig_cmd.add_argument("output", type=Path)
    civbig_cmd.add_argument("--png", action="store_true", help="also convert DDS to PNG with DirectXTex")
    civbig_cmd.add_argument("--texconv", type=Path, default=DEFAULT_TEXCONV)

    args = parser.parse_args(argv)
    if args.command == "discover":
        report = discover(args.civ6_base)
        write_json(args.output, report)
        print(f"Wrote {args.output}")
        for diagnostic in report["diagnostics"]:
            print(f"diagnostic: {diagnostic}", file=sys.stderr)
        return 0

    if args.command == "build-prototype":
        manifest = build_prototype(args.civ6_base, args.report, args.pack)
        print(f"Wrote {args.pack / 'manifest.json'}")
        for diagnostic in manifest["diagnostics"]:
            print(f"diagnostic: {diagnostic}", file=sys.stderr)
        return 0

    if args.command == "import-loose":
        manifest = build_from_loose(args.source, args.pack, copy_assets=not args.no_copy_assets)
        print(f"Wrote {args.pack / 'manifest.json'}")
        for diagnostic in manifest["diagnostics"]:
            print(f"diagnostic: {diagnostic}", file=sys.stderr)
        return 0

    if args.command == "build-grassland-poc":
        manifest = build_grassland_poc(args.civ6_base, args.pack)
        print(f"Wrote {args.pack / 'manifest.json'}")
        for diagnostic in manifest["diagnostics"]:
            print(f"diagnostic: {diagnostic}", file=sys.stderr)
        return 0

    if args.command == "extract-civbig":
        try:
            extract = extract_civbig_to_dds(args.source, args.output)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {args.output} ({extract['width']}x{extract['height']}, {extract['mip_count']} mips)")
        if args.png:
            png_path, conversion_error = convert_dds_to_png(args.output, args.texconv)
            if conversion_error:
                print(f"error: {conversion_error}", file=sys.stderr)
                return 1
            print(f"Wrote {png_path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
