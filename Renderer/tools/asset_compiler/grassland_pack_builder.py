#!/usr/bin/env python3
"""Build the normalized local grassland pack and textured previews."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Renderer.preview import render_textured_patch
from Renderer.tools.asset_compiler import c3x_asset_compiler
from Renderer.tools.asset_compiler import civblp_material_resolver
from Renderer.tools.asset_compiler import civblp_probe
from Renderer.tools.asset_compiler import terrain_geometry_resolver


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACK = RENDERER_ROOT / "packs" / "GrasslandNormalized"
DEFAULT_OUTPUT_DIR = RENDERER_ROOT / "preview" / "out"
DEFAULT_BUILD_REPORT = DEFAULT_OUTPUT_DIR / "grassland_normalized_build.json"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolved_role(binding: dict[str, Any], role_name: str) -> dict[str, Any]:
    matches = [
        item
        for item in binding.get("roles", [])
        if item.get("role") == role_name and item.get("status") == "resolved"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one resolved {role_name} binding, found {len(matches)}")
    return matches[0]


def base_color_role(binding: dict[str, Any]) -> dict[str, Any]:
    return resolved_role(binding, "base_color")


def extract_embedded_texture_role(
    package_path: Path, binding: dict[str, Any], role_name: str, output_dds: Path
) -> dict[str, Any]:
    role = resolved_role(binding, role_name)
    storage = role["storage"]
    metadata = role["metadata"]
    if storage.get("mode") != "embedded_blp_big_data" or not storage.get("bounds_valid"):
        raise ValueError(f"{role_name} is not a validated embedded CIVBLP resource")
    if metadata["format"]["dxgi"] not in c3x_asset_compiler.BC_BLOCK_BYTES:
        raise ValueError(f"{role_name} DXGI format is not supported by the DDS writer")

    actual_size = package_path.stat().st_size
    with package_path.open("rb") as source:
        header_bytes = source.read(civblp_probe.FILE_HEADER_SIZE)
        header = civblp_probe.parse_file_header(header_bytes, actual_size)
        expected_absolute = header["big_data"]["offset"] + storage["relative_offset"]
        if storage["absolute_file_offset"] != expected_absolute:
            raise ValueError("Binding absolute offset does not match the package big-data base")
        byte_count = storage["bytes"]
        expected_bytes = civblp_material_resolver.expected_bc_bytes(
            metadata["width"],
            metadata["height"],
            metadata["mip_count"],
            metadata["format"]["block_bytes"],
        )
        if byte_count != expected_bytes or expected_absolute + byte_count > actual_size:
            raise ValueError(f"Embedded {role_name} range failed mip-size or file-bounds validation")
        source.seek(expected_absolute)
        payload = source.read(byte_count)
    if len(payload) != byte_count:
        raise ValueError(f"Could not read the complete embedded {role_name} payload")

    dds_info = {
        "width": metadata["width"],
        "height": metadata["height"],
        "mip_count": metadata["mip_count"],
        "dxgi_format": metadata["format"]["dxgi"],
        "payload_bytes": byte_count,
    }
    output_dds.parent.mkdir(parents=True, exist_ok=True)
    output_dds.write_bytes(c3x_asset_compiler.make_dds_dx10_header(dds_info) + payload)
    return {
        **dds_info,
        "format_name": metadata["format"]["name"],
        "color_space": metadata["format"]["color_space"],
        "logical_name": role["logical_name"],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "dds_sha256": sha256_file(output_dds),
    }


def extract_embedded_base_color(
    package_path: Path, binding: dict[str, Any], output_dds: Path
) -> dict[str, Any]:
    return extract_embedded_texture_role(package_path, binding, "base_color", output_dds)


def validate_runtime_independence(pack_dir: Path) -> list[str]:
    errors = []
    banned = ("civ6", "civblp", ".blp", "firaxis")
    for path in sorted(pack_dir.rglob("*.json")):
        # Provenance is deliberately source-aware and is never loaded by the
        # runtime.  The manifest, assets, materials, and other runtime records
        # below the pack root remain subject to the strict source-independence
        # scan.
        if "provenance" in path.relative_to(pack_dir).parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        stack = [data]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                lowered = value.lower()
                if any(token in lowered for token in banned):
                    errors.append(f"{path.name} contains source-specific runtime text: {value}")
                if (
                    Path(value).is_absolute()
                    or PureWindowsPath(value).is_absolute()
                    or PurePosixPath(value).is_absolute()
                ):
                    errors.append(f"{path.name} contains an absolute runtime path: {value}")
    return errors


def build_normalized_pack(
    mesh_source: Path,
    dds_source: Path,
    texture_info: dict[str, Any],
    pack_dir: Path,
) -> dict[str, Any]:
    mesh = json.loads(mesh_source.read_text(encoding="utf-8"))
    mesh_errors = terrain_geometry_resolver.validate_normalized_mesh(mesh)
    if mesh_errors:
        raise ValueError("Normalized mesh failed validation: " + "; ".join(mesh_errors))
    texture = render_textured_patch.DdsBc3Texture.from_file(dds_source)
    if (
        texture.width != texture_info["width"]
        or texture.height != texture_info["height"]
        or texture.mip_count != texture_info["mip_count"]
        or texture.dxgi_format != texture_info["dxgi_format"]
    ):
        raise ValueError("DDS metadata does not match the normalized texture information")

    mesh_target = pack_dir / "meshes" / "flat_terrain_patch.json"
    texture_target = pack_dir / "textures" / "grassland_base_color.dds"
    mesh_target.parent.mkdir(parents=True, exist_ok=True)
    texture_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mesh_source, mesh_target)
    shutil.copy2(dds_source, texture_target)
    write_json(
        pack_dir / "materials" / "grassland.json",
        {
            "schema": "c3x.material.v0",
            "name": "grassland",
            "base_color": {
                "texture": "textures/grassland_base_color.dds",
                "format": texture_info["format_name"],
                "color_space": texture_info["color_space"],
                "uv_channel": "uv0",
                "address_u": "repeat",
                "address_v": "repeat",
            },
            "status": "normalized_local_import",
        },
    )
    manifest = {
        "schema": "c3x.asset_pack.v0",
        "name": "GrasslandNormalized",
        "display_name": "Normalized Grassland",
        "source_policy": "Local licensed-source import; derived texture is not redistributable.",
        "projection": {
            "tile_width_px": 128,
            "tile_height_px": 64,
            "height_scale_px": 54,
            "basis": {"x": [64, 32], "y": [-64, 32], "z": [0, -54]},
        },
        "assets": {
            "terrain/grassland/base": {
                "type": "terrain",
                "mesh": "meshes/flat_terrain_patch.json",
                "material": "materials/grassland.json",
            }
        },
        "terrains": {
            "grassland": {
                "mesh": "meshes/flat_terrain_patch.json",
                "material": "materials/grassland.json",
            }
        },
        "relief": {"mountains": {"selection": "none", "variants": []}},
    }
    write_json(pack_dir / "manifest.json", manifest)
    errors = validate_runtime_independence(pack_dir)
    if errors:
        raise ValueError("Runtime pack is not source-independent: " + "; ".join(errors))
    return manifest


def build_local_grassland(
    package_path: Path,
    binding_path: Path,
    mesh_path: Path,
    pack_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    staging_dds = output_dir / "grassland_base_color.dds"
    texture_info = extract_embedded_base_color(package_path, binding, staging_dds)
    manifest = build_normalized_pack(mesh_path, staging_dds, texture_info, pack_dir)
    previews = []
    for width, height in ((640, 480), (1024, 768)):
        canvas = render_textured_patch.render_pack(pack_dir / "manifest.json", width, height, 8)
        output = output_dir / f"grassland_normalized_{width}x{height}.png"
        render_textured_patch.write_png(canvas, output)
        previews.append(
            {
                "width": width,
                "height": height,
                "path": str(output),
                "non_background_pixels": canvas.non_background_pixels(),
                "unique_colors": len(set(canvas.pixels)),
                "sha256": sha256_file(output),
            }
        )
    return {
        "schema": "c3x.grassland_local_build.v0",
        "package": str(package_path),
        "binding": str(binding_path),
        "mesh": str(mesh_path),
        "pack": str(pack_dir),
        "manifest": manifest,
        "texture": texture_info,
        "previews": previews,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and preview the normalized grassland pack")
    parser.add_argument("--package", type=Path, default=civblp_probe.DEFAULT_PACKAGE)
    parser.add_argument("--binding", type=Path, default=civblp_material_resolver.DEFAULT_REPORT)
    parser.add_argument("--mesh", type=Path, default=terrain_geometry_resolver.DEFAULT_MESH)
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_BUILD_REPORT)
    args = parser.parse_args(argv)
    try:
        report = build_local_grassland(
            args.package, args.binding, args.mesh, args.pack, args.output_dir
        )
        write_json(args.report, report)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.pack / 'manifest.json'}")
    for preview in report["previews"]:
        print(
            f"Wrote {preview['path']} "
            f"({preview['non_background_pixels']} drawn pixels, {preview['unique_colors']} colors)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
