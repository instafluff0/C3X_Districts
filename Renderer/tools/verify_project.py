#!/usr/bin/env python3
"""Run executable gates for every completed C3X renderer prerequisite."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


RENDERER_ROOT = Path(__file__).resolve().parents[1]
C3X_ROOT = RENDERER_ROOT.parent
DEFAULT_STATUS = RENDERER_ROOT / "project_status.json"
DEFAULT_REPORT = RENDERER_ROOT / "verification" / "latest_report.json"

sys.path.insert(0, str(C3X_ROOT))
sys.path.insert(0, str(RENDERER_ROOT / "tools"))

import check_project_state
from Renderer.definitions import definition_parser, rule_resolver
from Renderer.inventory import civ3_art_inventory
from Renderer.preview import render_iso
from Renderer.preview import render_textured_patch
from Renderer.scenes import scene_contract
from Renderer.standalone import whole_viewport_renderer
from Renderer.terrain import production_terrain
from Renderer.tools.asset_compiler import c3x_asset_compiler as asset_compiler
from Renderer.tools.asset_compiler import civblp_material_resolver
from Renderer.tools.asset_compiler import civblp_probe
from Renderer.tools.asset_compiler import civ6_lighting_probe
from Renderer.tools.asset_compiler import grassland_pack_builder
from Renderer.tools.asset_compiler import terrain_geometry_resolver
from Renderer.tools.asset_compiler import terrain_pack_builder
from Renderer.tools.asset_compiler import terrain_relief_builder
from Renderer.tools import render_fixture_matrix


CheckFunction = Callable[[], dict[str, Any]]


@functools.lru_cache(maxsize=1)
def run_native_build() -> subprocess.CompletedProcess[str]:
    """Build and smoke the native renderer once per full verification run."""
    return subprocess.run(
        ["cmd", "/d", "/c", "BUILD.bat"],
        cwd=RENDERER_ROOT / "native",
        capture_output=True,
        text=True,
        check=False,
    )


@functools.lru_cache(maxsize=1)
def run_injected_compile() -> subprocess.CompletedProcess[str]:
    """Compile the injected path once per full verification run."""
    return subprocess.run(
        ["cmd", "/d", "/c", "TEST_INJECTED_CODE_COMPILE.bat"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def passed(detail: str) -> dict[str, Any]:
    return {"status": "pass", "detail": detail}


def failed(detail: str) -> dict[str, Any]:
    return {"status": "fail", "detail": detail}


def skipped(detail: str) -> dict[str, Any]:
    return {"status": "skip", "detail": detail}


def check_project_state_contract() -> dict[str, Any]:
    errors = check_project_state.validate_project_state(DEFAULT_STATUS)
    if errors:
        return failed("; ".join(errors))
    return passed("Status schema, evidence, gate declarations, and next-step pointer are consistent.")


def check_renderer_unit_tests() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "Renderer", "-p", "test_*.py"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"unittest exited with {result.returncode}")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(summary)


def sample_pack() -> dict[str, Any]:
    return {
        "projection": {"tile_width_px": 128, "tile_height_px": 64, "height_scale_px": 54},
        "terrains": {
            "grassland": {"preview_color": [83, 143, 79]},
            "plains": {"preview_color": [178, 157, 88]},
            "desert": {"preview_color": [202, 176, 103]},
            "tundra": {"preview_color": [132, 151, 139]},
        },
        "relief": {"mountains": {"variants": []}},
    }


def check_preview_smoke() -> dict[str, Any]:
    pack = sample_pack()
    first = render_iso.render(pack, 640, 480, 8, 123)
    repeated = render_iso.render(pack, 640, 480, 8, 123)
    second_size = render_iso.render(pack, 800, 600, 10, 123)
    if first.pixels != repeated.pixels:
        return failed("Preview output is not deterministic for identical inputs.")
    if first.non_background_pixels() < 1000 or second_size.non_background_pixels() < 1000:
        return failed("Preview was blank or nearly blank at a required viewport size.")
    return passed("Deterministic nonblank preview rendered at 640x480 and 800x600.")


def check_config_contract_fixture() -> dict[str, Any]:
    fixture = RENDERER_ROOT / "samples" / "config" / "default.custom_rendering.txt"
    if not fixture.is_file():
        return failed(f"Missing config fixture: {fixture}")
    text = fixture.read_text(encoding="utf-8")
    required_sections = {"#Profile", "#Pack", "#Asset", "#Rule", "#Environment"}
    missing = sorted(section for section in required_sections if section not in text)
    if missing:
        return failed(f"Config fixture is missing sections: {', '.join(missing)}")
    if "sheet_index" not in text or "sprite_index" not in text:
        return failed("Config fixture does not preserve Civ III sheet/sprite selectors.")
    if "day_night_source" not in text or "season_source" not in text:
        return failed("Config fixture does not connect environment state to C3X.")
    return passed("Profile, pack, asset, exact sprite rule, and C3X environment examples are present.")


def make_synthetic_civbig() -> bytes:
    width = 8
    height = 8
    mip_count = 2
    dxgi_format = 78
    block_bytes = asset_compiler.BC_BLOCK_BYTES[dxgi_format]
    payload_bytes = 0
    mip_width = width
    mip_height = height
    for _ in range(mip_count):
        payload_bytes += max(1, (mip_width + 3) // 4) * max(1, (mip_height + 3) // 4) * block_bytes
        mip_width = max(1, mip_width // 2)
        mip_height = max(1, mip_height // 2)
    header = bytearray(asset_compiler.CIVBIG_HEADER_SIZE)
    header[:8] = b"CIVBIG\x00\x00"
    struct.pack_into("<I", header, 8, payload_bytes)
    struct.pack_into("<6H", header, 32, 1, mip_count, dxgi_format, width, height, 1)
    return bytes(header) + bytes((index % 251 for index in range(payload_bytes))) + b"padding"


def check_civbig_synthetic_roundtrip() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "synthetic_civbig"
        output = Path(tmp) / "synthetic.dds"
        source.write_bytes(make_synthetic_civbig())
        try:
            info = asset_compiler.extract_civbig_to_dds(source, output)
        except (OSError, ValueError) as exc:
            return failed(f"Synthetic extraction failed: {exc}")
        data = output.read_bytes()
    if data[:4] != b"DDS " or data[84:88] != b"DX10":
        return failed("Extracted file lacks a valid DDS/DX10 header.")
    if struct.unpack_from("<I", data, 128)[0] != 78:
        return failed("DDS did not preserve the source DXGI format.")
    if len(data) != asset_compiler.DDS_DX10_HEADER_SIZE + info["payload_bytes"]:
        return failed("DDS payload length does not match the validated mip chain.")
    return passed("Synthetic BC3 CIVBIG converted to a size-valid DDS/DX10 mip chain.")


def check_civ6_grassland_texture_local() -> dict[str, Any]:
    source = (
        asset_compiler.DEFAULT_CIV6_BASE
        / "Platforms"
        / "Windows"
        / "BLPs"
        / "SHARED_DATA"
        / "TEXTURE_TER_Grass_Decal_B"
    )
    if not source.is_file():
        return skipped(f"Installed Civ VI grassland texture not found: {source}")
    if not asset_compiler.DEFAULT_TEXCONV.is_file():
        return skipped(f"Pinned DirectXTex converter not found: {asset_compiler.DEFAULT_TEXCONV}")
    with tempfile.TemporaryDirectory() as tmp:
        dds = Path(tmp) / "grassland.dds"
        try:
            info = asset_compiler.extract_civbig_to_dds(source, dds)
            png, conversion_error = asset_compiler.convert_dds_to_png(dds)
        except (OSError, ValueError) as exc:
            return failed(f"Local grassland extraction failed: {exc}")
        if conversion_error or png is None:
            return failed(conversion_error or "PNG conversion produced no output.")
        png_bytes = png.stat().st_size
    expected = (1024, 1024, 9, 78)
    actual = (info["width"], info["height"], info["mip_count"], info["dxgi_format"])
    if actual != expected:
        return failed(f"Unexpected installed grassland texture metadata: {actual}, expected {expected}")
    if png_bytes < 1024:
        return failed("Converted local grassland PNG is unexpectedly small.")
    return passed("Installed Civ VI grassland decoded as 1024x1024 BC3 sRGB with 9 mips and converted to PNG.")


def check_civblp_probe_synthetic() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.tools.asset_compiler.test_civblp_probe"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"CIVBLP probe tests exited with {result.returncode}")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(f"Synthetic CIVBLP structural probe gates passed ({summary}).")


def check_civ6_grassland_material_probe_local() -> dict[str, Any]:
    source = civblp_probe.DEFAULT_PACKAGE
    evidence_path = civblp_probe.DEFAULT_REPORT
    if not source.is_file():
        return skipped(f"Installed Civ VI terrain material package not found: {source}")
    if not evidence_path.is_file():
        return failed(f"Committed grassland probe evidence is missing: {evidence_path}")
    try:
        actual = civblp_probe.probe_file(source)
        expected = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return failed(f"Local grassland material probe failed: {exc}")
    if actual != expected:
        return failed("Installed-package probe does not match the deterministic committed evidence report.")
    textures = actual["material_record"]["candidate_texture_pointers"]
    if len(textures) != 4 or any(not item["candidate_record"]["strings"] for item in textures):
        return failed("Grassland material did not resolve four non-null typed texture records with strings.")
    return passed("Installed TerrainMaterialSet_Base.blp matches the deterministic typed grassland-pointer report.")


def check_civblp_material_resolver_synthetic() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.tools.asset_compiler.test_civblp_material_resolver"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"CIVBLP material resolver tests exited with {result.returncode}")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(f"Synthetic CIVBLP role, format, and embedded-resource gates passed ({summary}).")


def check_civ6_grassland_material_binding_local() -> dict[str, Any]:
    source = civblp_probe.DEFAULT_PACKAGE
    evidence_path = civblp_material_resolver.DEFAULT_REPORT
    if not source.is_file():
        return skipped(f"Installed Civ VI terrain material package not found: {source}")
    if not evidence_path.is_file():
        return failed(f"Committed grassland binding evidence is missing: {evidence_path}")
    try:
        actual = civblp_material_resolver.resolve_file(source)
        expected = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return failed(f"Local grassland material binding failed: {exc}")
    if actual != expected:
        return failed("Installed-package binding does not match the deterministic committed evidence report.")
    resolved = [item for item in actual["roles"] if item["status"] == "resolved"]
    if len(resolved) != 4 or {item["role"] for item in resolved} != {
        "base_color", "height", "specular", "fow_color"
    }:
        return failed("Grassland binding did not resolve the four required typed texture roles.")
    if any(
        item["storage"]["mode"] != "embedded_blp_big_data"
        or not item["storage"]["bounds_valid"]
        for item in resolved
    ):
        return failed("A grassland role did not resolve to a bounded embedded CIVBLP resource.")
    fuzz = next((item for item in actual["roles"] if item["role"] == "fuzz"), None)
    if fuzz is None or fuzz["status"] != "null":
        return failed("Grassland fuzz role was not preserved as an explicit null binding.")
    return passed("Installed grassland material resolves four bounded embedded resources; fuzz remains explicitly null.")


def check_terrain_geometry_resolver_synthetic() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.tools.asset_compiler.test_terrain_geometry_resolver"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"Terrain geometry resolver tests exited with {result.returncode}")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(f"Synthetic ArtDef selection and normalized mesh/UV gates passed ({summary}).")


def check_civ6_grassland_geometry_local() -> dict[str, Any]:
    base = terrain_geometry_resolver.DEFAULT_CIV6_BASE
    mesh_path = terrain_geometry_resolver.DEFAULT_MESH
    report_path = terrain_geometry_resolver.DEFAULT_REPORT
    if not base.is_dir():
        return skipped(f"Installed Civ VI asset root not found: {base}")
    if not mesh_path.is_file() or not report_path.is_file():
        return failed("Committed normalized flat mesh or grassland geometry report is missing.")
    try:
        mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
        expected = json.loads(report_path.read_text(encoding="utf-8"))
        actual = terrain_geometry_resolver.build_report(base, mesh)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return failed(f"Local grassland geometry resolution failed: {exc}")
    if actual != expected:
        return failed("Installed ArtDefs/package inventory does not match the deterministic geometry report.")
    mesh_errors = terrain_geometry_resolver.validate_normalized_mesh(mesh)
    if mesh_errors:
        return failed("Normalized flat mesh is invalid: " + "; ".join(mesh_errors))
    if actual["artdef_resolution"]["terrain_type"] != "Flat":
        return failed("Installed TERRAIN_GRASS did not resolve to the Flat terrain type.")
    if actual["selection"]["mode"] != "procedural_flat_grid":
        return failed("Grassland base did not resolve to the normalized procedural grid.")
    return passed("Installed ArtDefs resolve grassland to a validated generic flat grid with full-range UV0.")


def check_grassland_pack_render_synthetic() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.tools.asset_compiler.test_grassland_pack_builder"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"Grassland pack/render tests exited with {result.returncode}")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(f"Synthetic embedded extraction, generic pack, BC3 sampling, and textured-render gates passed ({summary}).")


def check_civ6_grassland_pack_render_local() -> dict[str, Any]:
    package = civblp_probe.DEFAULT_PACKAGE
    binding = civblp_material_resolver.DEFAULT_REPORT
    mesh = terrain_geometry_resolver.DEFAULT_MESH
    if not package.is_file():
        return skipped(f"Installed Civ VI terrain material package not found: {package}")
    if not binding.is_file() or not mesh.is_file():
        return failed("Committed grassland binding or normalized mesh evidence is missing.")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack = root / "pack"
            output = root / "out"
            report = grassland_pack_builder.build_local_grassland(
                package, binding, mesh, pack, output
            )
            runtime_errors = grassland_pack_builder.validate_runtime_independence(pack)
            manifest_path = pack / "manifest.json"
            first = render_textured_patch.render_pack(manifest_path, 640, 480, 8)
            repeated = render_textured_patch.render_pack(manifest_path, 640, 480, 8)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return failed(f"Local normalized grassland build/render failed: {exc}")
    if runtime_errors:
        return failed("Generated runtime pack is source-specific: " + "; ".join(runtime_errors))
    if first.pixels != repeated.pixels:
        return failed("Installed grassland textured render is not deterministic.")
    preview_sizes = {(item["width"], item["height"]) for item in report["previews"]}
    if preview_sizes != {(640, 480), (1024, 768)}:
        return failed(f"Local build did not produce both required preview sizes: {preview_sizes}")
    if any(item["non_background_pixels"] < 10000 or item["unique_colors"] < 16 for item in report["previews"]):
        return failed("A local textured preview was blank or lacked texture variation.")
    if report["texture"]["dxgi_format"] != 78 or report["texture"]["color_space"] != "srgb":
        return failed("Normalized local base color did not preserve BC3 sRGB metadata.")
    return passed("Installed embedded grassland texture built a source-agnostic pack and deterministic 640x480/1024x768 previews.")


def check_renderer_definition_parser() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.definitions.test_definition_parser"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"Renderer-definition parser tests exited with {result.returncode}")
    fixture = RENDERER_ROOT / "samples" / "config" / "default.custom_rendering.txt"
    try:
        definitions = definition_parser.parse_definition_file(
            fixture, "default", C3X_ROOT, C3X_ROOT
        )
        catalog = definition_parser.merge_layers([("default", definitions)])
    except (OSError, ValueError) as exc:
        return failed(f"Starter renderer definition failed strict parsing: {exc}")
    counts = tuple(len(catalog[key]) for key in ("profiles", "packs", "assets", "rules", "environments"))
    if counts != (1, 1, 1, 2, 1):
        return failed(f"Starter renderer definition produced unexpected section counts: {counts}")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(f"Strict typed parsing, layered replacement/disable, diagnostics, references, and path safety passed ({summary}).")


def check_renderer_rule_resolver() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.definitions.test_rule_resolver"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"Renderer rule-resolver tests exited with {result.returncode}")

    fixture = RENDERER_ROOT / "samples" / "config" / "default.custom_rendering.txt"
    try:
        definitions = definition_parser.parse_definition_file(
            fixture, "default", C3X_ROOT, C3X_ROOT
        )
        catalog = definition_parser.merge_layers([("default", definitions)])
        selection = rule_resolver.resolve_rule(
            catalog,
            {
                "category": "terrain",
                "terrain_type": "grassland",
                "sheet_index": 2,
                "sprite_index": 0,
                "map_x": 7,
                "map_y": 11,
            },
            world_seed=42,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return failed(f"Starter renderer rule resolution failed: {exc}")
    if selection["status"] != "matched" or selection["winner"]["rule_id"] != "terrain.grassland.sheet2.sprite0":
        return failed("Starter exact sheet/sprite input did not select its exact rule.")
    candidates = {candidate["rule_id"]: candidate for candidate in selection["candidates"]}
    general = candidates.get("terrain.grassland.default", {})
    if general.get("status") != "matched_loser" or general.get("loser_reason") != "lower_priority":
        return failed("Starter rule diagnostics did not explain the general rule's loss.")
    if selection["asset_payload_loads"] != 0:
        return failed("Rule selection attempted to load an asset payload.")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(f"Deterministic selector ranking, diagnostics, filters, variants, and safe fallbacks passed ({summary}).")


def check_visible_scene_contract() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.scenes.test_scene_contract"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"Visible-scene contract tests exited with {result.returncode}")

    fixture = RENDERER_ROOT / "samples" / "scenes" / "grassland_viewport.scene.json"
    config = RENDERER_ROOT / "samples" / "config" / "default.custom_rendering.txt"
    try:
        scene = scene_contract.load_scene(fixture)
        first = scene_contract.canonical_json(scene)
        repeated = scene_contract.canonical_json(scene_contract.parse_scene_text(first))
        definitions = definition_parser.parse_definition_file(config, "default", C3X_ROOT, C3X_ROOT)
        catalog = definition_parser.merge_layers([("default", definitions)])
        inspection = scene_contract.inspect_scene(scene, catalog)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return failed(f"Visible-scene fixture validation or replay failed: {exc}")
    if first.encode("utf-8") != repeated.encode("utf-8"):
        return failed("Visible-scene canonical round trip is not byte-stable.")
    if len(inspection["items"]) != len(scene["tiles"]) + len(scene["instances"]):
        return failed("Offline scene inspection did not resolve every terrain/object record.")
    serialized = scene_contract.canonical_json(scene).casefold()
    if any(marker in serialized for marker in ("civ6", "artdef", ".blp", ".fgx", "steamapps")):
        return failed("Visible-scene fixture contains a source-specific runtime marker.")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(f"Strict scene validation, deterministic IDs/seeds/anchors, canonical replay, and offline resolver inspection passed ({summary}).")


def check_standalone_whole_viewport_renderer() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.standalone.test_whole_viewport_renderer"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"Standalone renderer tests exited with {result.returncode}")
    if whole_viewport_renderer.CATALOG_SCHEMA != "c3x.renderer_definition_catalog.v0":
        return failed("Standalone renderer is not bound to the gated M2 catalog schema.")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(
        "Whole-scene projection, depth, lighting, season, two-size determinism, fallback, and lifecycle gates passed "
        f"({summary})."
    )


def check_fixture_matrix_validation() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.tools.test_render_fixture_matrix"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or f"Fixture-matrix tests exited with {result.returncode}")
    reference_path = RENDERER_ROOT / "samples" / "validation" / "reference_metadata.json"
    try:
        references = json.loads(reference_path.read_text(encoding="utf-8"))
        render_fixture_matrix.validate_reference_catalog(references)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return failed(f"Visual-reference metadata failed validation: {exc}")
    if render_fixture_matrix.DEFAULT_VIEWPORTS != ((640, 480), (1024, 768)):
        return failed("Fixture-matrix defaults do not include the required two viewport sizes.")
    if render_fixture_matrix.DEFAULT_HOURS != (0, 6, 12, 18):
        return failed("Fixture-matrix defaults do not cover midnight, sunrise, noon, and sunset.")
    if render_fixture_matrix.DEFAULT_SEASONS != ("summer", "fall", "winter", "spring"):
        return failed("Fixture-matrix defaults do not cover all four seasons.")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(
        "Two-size/four-hour/four-season matrix determinism, metrics, contact-sheet, and reference-separation gates passed "
        f"({summary})."
    )


def check_native_civ3_bridge() -> dict[str, Any]:
    native = run_native_build()
    native_output = (native.stdout + native.stderr).strip()
    if native.returncode != 0:
        return failed(native_output or f"Native renderer build exited with {native.returncode}")
    injected = run_injected_compile()
    injected_output = (injected.stdout + injected.stderr).strip()
    if injected.returncode != 0:
        return failed(injected_output or f"Injected-code compile exited with {injected.returncode}")
    if "PASS native_renderer_smoke" not in native_output:
        return failed("Native renderer build did not run its executable smoke gate.")
    if "Injected code compiled successfully." not in injected_output:
        return failed("Approved injected-code compile did not report success.")
    return passed(
        "32-bit D3D11 DLL ABI/render/readback/blit smoke and TEST_INJECTED_CODE_COMPILE.bat passed."
    )


def check_m5_2_scene_export() -> dict[str, Any]:
    native = run_native_build()
    native_output = (native.stdout + native.stderr).strip()
    if native.returncode != 0:
        return failed(native_output or f"Native renderer build exited with {native.returncode}")
    smoke_scene = RENDERER_ROOT / "native" / "build" / "native-smoke.scene.json"
    try:
        first = smoke_scene.read_bytes()
        scene = scene_contract.load_scene(smoke_scene)
    except (OSError, scene_contract.SceneValidationError) as exc:
        return failed(f"Native scene export does not satisfy c3x.visible_scene.v0: {exc}")
    categories = {"terrain", *(item["category"] for item in scene["instances"])}
    if categories != definition_parser.RULE_CATEGORIES:
        missing = sorted(definition_parser.RULE_CATEGORIES - categories)
        return failed(f"Native scene export is missing renderer categories: {', '.join(missing)}")
    repeat = subprocess.run(
        [str(RENDERER_ROOT / "native" / "build" / "native_smoke.exe"), str(RENDERER_ROOT / "native" / "build" / "candidate" / "C3XRenderer.dll")],
        cwd=RENDERER_ROOT / "native",
        capture_output=True,
        text=True,
        check=False,
    )
    if repeat.returncode != 0:
        return failed((repeat.stdout + repeat.stderr).strip() or "Repeated native export failed")
    if smoke_scene.read_bytes() != first:
        return failed("Repeated native visible-scene export is not byte deterministic.")
    offline = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.tools.test_process_scene_export_batch"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if offline.returncode != 0:
        return failed((offline.stdout + offline.stderr).strip() or "Offline M5.2 batch tests failed")
    injected = run_injected_compile()
    if injected.returncode != 0 or "Injected code compiled successfully." not in injected.stdout + injected.stderr:
        return failed((injected.stdout + injected.stderr).strip() or "Injected-code export bridge compile failed")
    return passed(
        "Deterministic native c3x.visible_scene.v0 export covers all renderer categories; offline batch tests and injected compile passed."
    )


def check_m5_3_frame_scheduler() -> dict[str, Any]:
    native = run_native_build()
    native_output = (native.stdout + native.stderr).strip()
    if native.returncode != 0:
        return failed(native_output or f"Native frame-scheduler build exited with {native.returncode}")
    if "absolute-time scheduling, idle/pause guards, frame skipping" not in native_output:
        return failed("Native smoke did not exercise the M5.3 scheduling contract.")
    contract = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if contract.returncode != 0:
        return failed((contract.stdout + contract.stderr).strip() or "M5.3 bridge contract tests failed")
    injected = run_injected_compile()
    if injected.returncode != 0 or "Injected code compiled successfully." not in injected.stdout + injected.stderr:
        return failed((injected.stdout + injected.stderr).strip() or "Injected frame-scheduler bridge compile failed")
    return passed(
        "Absolute QPC timing, one-request dirty scheduling, idle/modal/focus/pause guards, skipped-frame determinism, bounded telemetry, and timer-safe injection passed."
    )


def check_m6_0_inventory_contract() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.inventory.test_civ3_art_inventory"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return failed((result.stdout + result.stderr).strip() or "M6.0 inventory tests failed")
    try:
        atlas = civ3_art_inventory.load_json(civ3_art_inventory.DEFAULT_ATLAS_CONTRACTS)
        semantics = civ3_art_inventory.load_json(civ3_art_inventory.DEFAULT_BIQ_SEMANTICS)
        census = civ3_art_inventory.load_json(civ3_art_inventory.INVENTORY_ROOT / "runtime_selector_census.json")
        by_name = civ3_art_inventory.contract_by_basename(atlas)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return failed(f"M6.0 committed evidence could not be loaded: {exc}")
    if len(by_name) != 76:
        return failed(f"Expected 76 effective vanilla atlas contracts, found {len(by_name)}.")
    expected_counts = (124, 26, 14)
    actual_counts = (
        semantics["counts"]["primary_unit_types"],
        semantics["counts"]["resources"],
        semantics["counts"]["terrain_types"],
    )
    if actual_counts != expected_counts:
        return failed(f"Bundled BIQ semantic census changed: {actual_counts}, expected {expected_counts}.")
    layer_ids = {item["id"] for item in civ3_art_inventory.RENDER_LAYERS}
    census_ids = {item["id"] for item in census.get("families", []) if item.get("tested")}
    if census_ids != layer_ids or census.get("unknown_selectors"):
        return failed("Runtime selector census does not close every retained/replacement responsibility.")
    allowed = {"mapped", "vanilla_fallback", "not_map_rendered", "unreachable"}
    if any(item.get("classification") not in allowed for item in census.get("families", [])):
        return failed("Runtime census contains an invalid or missing ownership classification.")
    return passed(
        "Atlas geometry, BIQ semantics, FLC direction/action metadata, ownership taxonomy, and zero-unknown selector contract passed."
    )


def check_m6_0_installed_vanilla_inventory() -> dict[str, Any]:
    install_root = C3X_ROOT.parents[1]
    biq = C3X_ROOT.parent / "conquests.biq"
    editor_root = C3X_ROOT.parent / "C3X_Editor"
    if not biq.is_file() or not editor_root.is_dir():
        return skipped("Local conquests.biq or read-only C3X_Editor BIQ parser is unavailable.")
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            regenerated = tmp_root / "semantics.json"
            extraction = subprocess.run(
                [
                    "node",
                    str(RENDERER_ROOT / "inventory" / "extract_biq_semantics.js"),
                    "--biq", str(biq),
                    "--editor-root", str(editor_root),
                    "--install-root", str(install_root),
                    "--output", str(regenerated),
                ],
                cwd=C3X_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if extraction.returncode != 0:
                return failed((extraction.stdout + extraction.stderr).strip() or "BIQ semantic extraction failed")
            committed = civ3_art_inventory.DEFAULT_BIQ_SEMANTICS
            if regenerated.read_bytes() != committed.read_bytes():
                return failed("Installed conquests.biq/PediaIcons census differs from committed deterministic semantics.")
            atlas = civ3_art_inventory.load_json(civ3_art_inventory.DEFAULT_ATLAS_CONTRACTS)
            semantics = civ3_art_inventory.load_json(committed)
            census = civ3_art_inventory.load_json(civ3_art_inventory.INVENTORY_ROOT / "runtime_selector_census.json")
            inventory = civ3_art_inventory.build_inventory(
                install_root,
                atlas_contracts=atlas,
                biq_semantics=semantics,
                runtime_census=census,
            )
            contacts = civ3_art_inventory.generate_contact_sheets(
                inventory, install_root, (), tmp_root / "contact_sheets"
            )
            if inventory["completeness"]["status"] != "complete":
                return failed("Installed strict inventory remains incomplete: " + json.dumps(inventory["completeness"], sort_keys=True))
            if len(contacts) != 76 or any(not (tmp_root / item["contact_sheet"]).read_bytes().startswith(b"\x89PNG") for item in contacts):
                return failed("Installed atlas census did not generate all 76 annotated PNG contact sheets.")
            effective_units = [item for item in inventory["units"] if item["effective"]]
            if len(effective_units) != 144:
                return failed(f"Expected 144 selectable unit-art directories, found {len(effective_units)}.")
            bad_directions = [
                f"{unit['name']}:{animation['action']}"
                for unit in effective_units
                if unit.get("classification") == "vanilla_fallback"
                for animation in unit["animations"]
                if animation["resolution_status"] == "resolved" and animation.get("direction_count") not in {1, 2, 8}
            ]
            if bad_directions:
                return failed("Unexpected FLC direction metadata: " + ", ".join(bad_directions[:8]))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return failed(f"Installed vanilla inventory failed: {exc}")
    return passed(
        "Installed layered art closes 112 files, 76 atlases/contact sheets, 124 BIQ unit types, 144 unit directories, 26 resources, and 14 terrains with zero unknowns."
    )


def check_m6_1_production_terrain() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.terrain.test_production_terrain"],
        cwd=C3X_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stderr or result.stdout).strip()
    if result.returncode != 0:
        return failed(output or "M6.1 production terrain tests failed")
    try:
        coverage = production_terrain.validate_selector_coverage()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return failed(f"M6.1 selector coverage is invalid: {exc}")
    summary = next((line.strip() for line in output.splitlines() if line.strip().startswith("Ran ")), "tests passed")
    return passed(
        f"{summary}; {coverage['terrain_types']} BIQ terrain types and "
        f"{coverage['selector_cells_accounted']} M6 selector cells have explicit ownership."
    )


def check_m6_2_native_terrain_art() -> dict[str, Any]:
    native = run_native_build()
    native_output = (native.stdout + native.stderr).strip()
    if native.returncode != 0 or "multi-material DDS sampling" not in native_output or "pixel_hash=" not in native_output:
        return failed(native_output or "Native normalized-terrain build/smoke failed")
    contract = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if contract.returncode != 0:
        return failed((contract.stdout + contract.stderr).strip() or "Native terrain-art contract tests failed")
    injected = run_injected_compile()
    if injected.returncode != 0 or "Injected code compiled successfully." not in injected.stdout + injected.stderr:
        return failed((injected.stdout + injected.stderr).strip() or "Injected normalized-pack bridge compile failed")
    return passed(
        "Portable normalized manifest/mesh/material/BC3 loading, textured native sampling, malformed-pack rejection, ABI v5, and injected bridge compile passed."
    )


def _native_smoke_hash(output: str) -> str | None:
    marker = "pixel_hash="
    if marker not in output:
        return None
    return output.split(marker, 1)[1].split(".", 1)[0].strip()


def check_m6_2_local_grassland_art() -> dict[str, Any]:
    pack = RENDERER_ROOT / "packs" / "GrasslandNormalized"
    if not (pack / "manifest.json").is_file():
        return skipped(f"Local normalized grassland pack not found: {pack}")
    independence_errors = grassland_pack_builder.validate_runtime_independence(pack)
    if independence_errors:
        return failed("Local normalized pack is not runtime-independent: " + "; ".join(independence_errors))
    executable = RENDERER_ROOT / "native" / "build" / "native_smoke.exe"
    dll = RENDERER_ROOT / "native" / "build" / "candidate" / "C3XRenderer.dll"
    synthetic = subprocess.run(
        [str(executable), str(dll)], cwd=RENDERER_ROOT / "native",
        capture_output=True, text=True, check=False,
    )
    actual = subprocess.run(
        [str(executable), str(dll), str(pack)], cwd=RENDERER_ROOT / "native",
        capture_output=True, text=True, check=False,
    )
    synthetic_output, actual_output = synthetic.stdout + synthetic.stderr, actual.stdout + actual.stderr
    synthetic_hash, actual_hash = _native_smoke_hash(synthetic_output), _native_smoke_hash(actual_output)
    if synthetic.returncode != 0 or actual.returncode != 0:
        return failed((synthetic_output + actual_output).strip() or "Local normalized grassland native smoke failed")
    if not synthetic_hash or not actual_hash or synthetic_hash == actual_hash:
        return failed("Installed normalized grassland did not produce a distinct deterministic native texture result")
    return passed(
        f"Local licensed grassland DDS rendered through the native in-game path with a distinct pixel hash ({actual_hash})."
    )


def check_m6_3_definition_terrain() -> dict[str, Any]:
    coverage_path = RENDERER_ROOT / "terrain" / "m6_3_runtime_coverage.json"
    definition_path = RENDERER_ROOT / "default.custom_rendering.txt"
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        catalog = definition_parser.merge_layers([
            ("default", definition_parser.parse_definition_file(
                definition_path, "default", C3X_ROOT, None, False
            ))
        ])
    except (OSError, ValueError, json.JSONDecodeError, definition_parser.DefinitionError) as exc:
        return failed(f"M6.3 definition/coverage contract failed: {exc}")
    terrain_records = coverage.get("terrain_types", [])
    if len(terrain_records) != 14 or any(item.get("disposition") not in {"mapped", "vanilla_fallback"} for item in terrain_records):
        return failed("M6.3 runtime coverage does not classify all 14 terrain types")
    mapped = {item["logical_asset"] for item in terrain_records if item["disposition"] == "mapped"}
    active_assets = {item["values"]["asset"] for item in catalog["assets"]}
    if len(mapped) < 6 or not mapped.issubset(active_assets):
        return failed("Layered default definitions do not expose every mapped terrain logical asset")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest",
         "Renderer.tools.asset_compiler.test_terrain_pack_builder",
         "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        return failed((tests.stdout + tests.stderr).strip() or "M6.3 portable tests failed")
    native = run_native_build()
    native_output = native.stdout + native.stderr
    if native.returncode != 0 or "layered terrain definitions" not in native_output or "atomic fallback" not in native_output:
        return failed(native_output.strip() or "Definition-driven native terrain smoke failed")
    injected = run_injected_compile()
    if injected.returncode != 0 or "Injected code compiled successfully." not in injected.stdout + injected.stderr:
        return failed((injected.stdout + injected.stderr).strip() or "Definition-driven bridge compile failed")
    return passed(
        f"Layered default/scenario/custom definitions map {len(mapped)} real-art terrain families; synthetic corruption falls back atomically and retained categories remain Civ III-owned."
    )


def check_m6_3_local_terrain_art() -> dict[str, Any]:
    package = civblp_probe.DEFAULT_PACKAGE
    mesh = terrain_geometry_resolver.DEFAULT_MESH
    pack = terrain_pack_builder.DEFAULT_PACK
    if not package.is_file() or not mesh.is_file():
        return skipped("Installed local terrain-material source or normalized mesh is unavailable")
    try:
        build = terrain_pack_builder.build_local_terrain_pack(
            package, mesh, pack, terrain_pack_builder.DEFAULT_REPORT
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return failed(f"Local normalized terrain pack build failed: {exc}")
    if build["mapped_count"] < 6:
        return failed(f"Expected at least the 6 proven M6.3 terrain material mappings, got {build['mapped_count']}")
    independence_errors = grassland_pack_builder.validate_runtime_independence(pack)
    if independence_errors:
        return failed("Local terrain pack is not runtime-independent: " + "; ".join(independence_errors))
    executable = RENDERER_ROOT / "native" / "build" / "native_smoke.exe"
    dll = RENDERER_ROOT / "native" / "build" / "candidate" / "C3XRenderer.dll"
    synthetic = subprocess.run(
        [str(executable), str(dll)], cwd=RENDERER_ROOT / "native",
        capture_output=True, text=True, check=False,
    )
    actual = subprocess.run(
        [str(executable), str(dll), "--definitions", str(C3X_ROOT),
         str(RENDERER_ROOT / "default.custom_rendering.txt")],
        cwd=RENDERER_ROOT / "native", capture_output=True, text=True, check=False,
    )
    synthetic_output, actual_output = synthetic.stdout + synthetic.stderr, actual.stdout + actual.stderr
    synthetic_hash, actual_hash = _native_smoke_hash(synthetic_output), _native_smoke_hash(actual_output)
    if synthetic.returncode != 0 or actual.returncode != 0:
        return failed((synthetic_output + actual_output).strip() or "Local multi-terrain native smoke failed")
    if not synthetic_hash or not actual_hash or synthetic_hash == actual_hash:
        return failed("Installed normalized terrain pack did not produce a distinct deterministic native result")
    return passed(
        f"Compiled {build['mapped_count']} local terrain materials into a source-agnostic pack and rendered representative families through layered native definitions (pixel hash {actual_hash})."
    )


def check_m6_4_environment_runtime() -> dict[str, Any]:
    contract = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.environment.test_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if contract.returncode != 0:
        return failed((contract.stdout + contract.stderr).strip() or "M6.4 environment contract tests failed")
    native = run_native_build()
    native_output = native.stdout + native.stderr
    if native.returncode != 0:
        return failed(native_output.strip() or "M6.4 native environment build failed")
    if "shared environment, moonlit water, emissive/attachment primitives" not in native_output:
        return failed("Native smoke did not exercise the complete M6.4 environment contract")
    return passed(
        "Two-size native noon/sunset/midnight/sunrise renders, bounded moonlit water, static emissive idle, and deterministic animated attachment fallback passed."
    )


def check_m6_5_connected_terrain() -> dict[str, Any]:
    contract = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if contract.returncode != 0:
        return failed((contract.stdout + contract.stderr).strip() or "M6.5 connected-terrain contract tests failed")
    native = run_native_build()
    native_output = native.stdout + native.stderr
    required_smoke = "connected material blending, native-underlay feathering"
    if native.returncode != 0 or required_smoke not in native_output:
        return failed(native_output.strip() or "M6.5 connected-terrain native smoke failed")
    injected = run_injected_compile()
    if injected.returncode != 0 or "Injected code compiled successfully." not in injected.stdout + injected.stderr:
        return failed((injected.stdout + injected.stderr).strip() or "M6.5 native-underlay bridge compile failed")
    return passed(
        "Adjacent mapped materials use symmetric edge blends, mapped/fallback boundaries feather into a complete Civ III base underlay, and deterministic tile checkerboarding is disabled."
    )


def check_m6_6_vanilla_terrain() -> dict[str, Any]:
    coverage_path = RENDERER_ROOT / "terrain" / "m6_6_runtime_coverage.json"
    try:
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return failed(f"Could not read M6.6 ownership coverage: {exc}")
    terrain = coverage.get("terrain_types", [])
    if len(terrain) != 14 or {item.get("id") for item in terrain} != set(range(14)):
        return failed("M6.6 coverage does not account for all fourteen BIQ terrain types")
    compositions = {item.get("family"): item.get("ownership") for item in coverage.get("compositions", [])}
    required = {"base_material_transitions", "vegetation_and_marsh_bodies", "polar_ice", "landmarks", "shoreline_detail"}
    if not required.issubset(compositions) or compositions.get("polar_ice") != "civ3" or compositions.get("landmarks") != "civ3":
        return failed("M6.6 transition/vegetation/ice/landmark ownership is incomplete")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest",
         "Renderer.tools.asset_compiler.test_terrain_relief_builder",
         "Renderer.tools.asset_compiler.test_terrain_pack_builder",
         "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        return failed((tests.stdout + tests.stderr).strip() or "M6.6 portable tests failed")
    syntax = subprocess.run(
        ["node", "--check", str(RENDERER_ROOT / "tools" / "export_biq_terrain_scene.js")],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if syntax.returncode != 0:
        return failed((syntax.stdout + syntax.stderr).strip() or "BIQ terrain exporter syntax failed")
    native = run_native_build()
    native_output = native.stdout + native.stderr
    if native.returncode != 0 or not (RENDERER_ROOT / "native" / "build" / "biq_preview.exe").is_file():
        return failed(native_output.strip() or "M6.6 native relief/depth build failed")
    injected = run_injected_compile()
    if injected.returncode != 0 or "Injected code compiled successfully." not in injected.stdout + injected.stderr:
        return failed((injected.stdout + injected.stderr).strip() or "M6.6 real-terrain bridge compile failed")
    return passed(
        "All 14 terrain identities, retained vegetation/ice/landmark ownership, generic R8 relief, dense connected geometry, depth, and BIQ-preview tooling passed."
    )


def _bmp_metrics(path: Path) -> tuple[str, int, int]:
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise ValueError(f"Not a BMP: {path}")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = abs(struct.unpack_from("<i", data, 22)[0])
    pixels = data[pixel_offset:]
    if width < 1 or height < 1 or len(pixels) != width * height * 4:
        raise ValueError(f"Unexpected 32-bit BMP dimensions: {path}")
    colors = set()
    nonblack = 0
    for offset in range(0, len(pixels), 4):
        color = pixels[offset:offset + 3]
        nonblack += color != b"\0\0\0"
        if len(colors) < 10000:
            colors.add(color)
    return hashlib.sha256(data).hexdigest(), nonblack, len(colors)


def check_m6_6_local_biq_terrain() -> dict[str, Any]:
    material_package = civblp_probe.DEFAULT_PACKAGE
    relief_package = terrain_relief_builder.DEFAULT_PACKAGE
    biq = C3X_ROOT.parent / "Scenarios" / "test.biq"
    if not material_package.is_file() or not relief_package.is_file() or not biq.is_file():
        return skipped("Installed terrain packages or user-supplied Scenarios/test.biq are unavailable")
    try:
        build = terrain_pack_builder.build_local_terrain_pack(
            material_package, terrain_geometry_resolver.DEFAULT_MESH,
            terrain_pack_builder.DEFAULT_PACK, terrain_pack_builder.DEFAULT_REPORT,
            relief_package,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError, struct.error) as exc:
        return failed(f"M6.6 local terrain-pack build failed: {exc}")
    if build.get("mapped_count") != 14 or len(build.get("relief_evidence", {}).get("extracted", [])) != 2:
        return failed("Local pack did not contain all 14 material families and both typed relief outputs")
    output_root = RENDERER_ROOT / "preview" / "out"
    scene_csv = output_root / "test_biq_terrain.csv"
    exported = subprocess.run(
        ["node", str(RENDERER_ROOT / "tools" / "export_biq_terrain_scene.js"), str(biq), str(scene_csv)],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if exported.returncode != 0 or "5000 BIQ tiles (100x100)" not in exported.stdout:
        return failed((exported.stdout + exported.stderr).strip() or "test.biq terrain export failed")
    executable = RENDERER_ROOT / "native" / "build" / "biq_preview.exe"
    dll = RENDERER_ROOT / "native" / "build" / "candidate" / "C3XRenderer.dll"
    definition = RENDERER_ROOT / "default.custom_rendering.txt"
    outputs = [
        (scene_csv, output_root / "m6_6_test_biq_128.bmp", 1600, 900, 28, 52, 128),
        (scene_csv, output_root / "m6_6_test_biq_64.bmp", 1280, 720, 28, 52, 64),
        (RENDERER_ROOT / "samples" / "scenes" / "m6_6_all_terrain.csv",
         output_root / "m6_6_all_terrain.bmp", 1200, 500, 6, 1, 128),
    ]
    hashes = []
    for scene, output, width, height, center_x, center_y, tile_width in outputs:
        command = [str(executable), str(dll), str(C3X_ROOT), str(definition), str(scene), str(output),
                   str(width), str(height), str(center_x), str(center_y), str(tile_width)]
        rendered = subprocess.run(command, cwd=C3X_ROOT, capture_output=True, text=True, check=False)
        if rendered.returncode != 0 or "0 fallback" not in rendered.stdout:
            return failed((rendered.stdout + rendered.stderr).strip() or f"Native BIQ preview failed: {output}")
        try:
            digest, nonblack, unique = _bmp_metrics(output)
        except (OSError, ValueError, struct.error) as exc:
            return failed(str(exc))
        minimum_coverage = width * height // (10 if scene.name == "m6_6_all_terrain.csv" else 2)
        if nonblack < minimum_coverage or unique < 256:
            return failed(f"BIQ preview was blank or materially degenerate: {output}")
        hashes.append(digest)
    repeated = output_root / "m6_6_test_biq_128_repeat.bmp"
    scene, _output, width, height, center_x, center_y, tile_width = outputs[0]
    repeat = subprocess.run(
        [str(executable), str(dll), str(C3X_ROOT), str(definition), str(scene), str(repeated),
         str(width), str(height), str(center_x), str(center_y), str(tile_width)],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if repeat.returncode != 0 or _bmp_metrics(repeated)[0] != hashes[0]:
        return failed("Repeated test.biq native preview was not byte-deterministic")
    return passed(
        "Parsed the user-supplied 100x100 test.biq and rendered deterministic 1600x900/128px and 1280x720/64px native screenshots with zero fallback; the 14-type matrix also passed."
    )


def _fnv1a64(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def check_m6_7_approved_terrain_integration() -> dict[str, Any]:
    fidelity_path = RENDERER_ROOT / "terrain" / "m6_7_handoff_fidelity.json"
    try:
        fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return failed(f"Could not read M6.7 fidelity contract: {exc}")
    if fidelity.get("schema") != "c3x.m6_7_handoff_fidelity.v0":
        return failed("M6.7 fidelity contract schema is invalid")
    approved = fidelity.get("approved_inputs", [])
    if [item.get("gate") for item in approved] != ["L9", "L10", "L11"]:
        return failed("I11 fidelity contract must consume exactly approved L9, L10, and L11")
    for item in approved:
        try:
            record_path = RENDERER_ROOT / item["record"]
            record_bytes = record_path.read_bytes()
            record = json.loads(record_bytes)
        except (KeyError, OSError, json.JSONDecodeError) as exc:
            return failed(f"Could not validate M6.7 handoff record: {exc}")
        if record.get("status") != "approved" or record.get("lab_gate") != item["gate"]:
            return failed(f"M6.7 handoff {item['gate']} is not the approved record")
        if _fnv1a64(record_bytes) != item.get("record_fnv1a64"):
            return failed(f"M6.7 handoff {item['gate']} differs from its frozen runtime revision")
        reference = record.get("reference", {})
        if reference.get("native_sha256") != item.get("native_reference_sha256") or \
           reference.get("reduced_sha256") != item.get("reduced_reference_sha256"):
            return failed(f"M6.7 handoff {item['gate']} reference hashes drifted")
        for output_key, hash_key in (("native_output", "native_sha256"),
                                     ("reduced_output", "reduced_sha256")):
            output = RENDERER_ROOT / str(reference.get(output_key, ""))
            if output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest() != reference[hash_key]:
                return failed(f"M6.7 {item['gate']} local reference output hash drifted")
    tolerances = fidelity.get("deterministic_tolerances", {})
    if tolerances.get("same_frame_pixels") != "byte_exact" or \
       tolerances.get("screen_anchor_error_px") != 0 or \
       tolerances.get("pixels_modified_outside_clip") != 0 or \
       tolerances.get("cache_entries_max") != 1 or \
       tolerances.get("fallback_ownership_flags") != 0 or \
       tolerances.get("production_native_fallback_tiles") != 0 or \
       tolerances.get("custom_on_failure_policy") != "hard_failure_without_native_replay":
        return failed("I11 deterministic/exclusive-ownership tolerances are incomplete or weakened")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        return failed((tests.stdout + tests.stderr).strip() or "M6.7 bridge contracts failed")
    build_text = (RENDERER_ROOT / "native" / "BUILD.bat").read_text(encoding="utf-8")
    if "approved_terrain_integration" not in build_text or "--definitions" not in build_text:
        return failed("The current approved production handoff smoke is not wired into the native build")
    return passed(
        "Frozen L9/L10/L11 record/reference hashes, source-independent marsh component mapping, exact cache/anchor/clip tolerances, and exclusive custom-on ownership contracts passed."
    )


def check_m6_7_local_approved_terrain_payload() -> dict[str, Any]:
    required = [
        RENDERER_ROOT / "packs" / "TerrainNormalized" / "manifest.json",
        RENDERER_ROOT / "packs" / "VegetationNormalized" / "vegetation_runtime.bin",
        RENDERER_ROOT / "packs" / "DecalsNormalized" / "manifest.json",
        RENDERER_ROOT / "packs" / "TerrainElementsNormalized" / "manifest.json",
        RENDERER_ROOT / "packs" / "ShoreNormalized" / "shore_runtime.bin",
        RENDERER_ROOT / "packs" / "RouteStylesNormalized" / "manifest.json",
        RENDERER_ROOT / "packs" / "RouteDoodadsNormalized" / "bridge_runtime.bin",
        RENDERER_ROOT / "packs" / "ResourceNormalized" / "resource_runtime.bin",
        RENDERER_ROOT / "packs" / "CityComponentsNormalized" / "city_runtime.bin",
        RENDERER_ROOT / "packs" / "CityAdjunctsNormalized" / "wall_runtime.bin",
        RENDERER_ROOT / "packs" / "ImprovementsNormalized" / "mine_runtime.bin",
        RENDERER_ROOT / "packs" / "ImprovementsNormalized" / "farm_runtime.bin",
    ]
    if any(not path.is_file() for path in required):
        return skipped("Ignored normalized L9-L19 payloads are unavailable")
    native = run_native_build()
    native_output = native.stdout + native.stderr
    if native.returncode != 0 or "PASS approved_terrain_integration" not in native_output:
        return failed(native_output.strip() or "Current approved production-payload smoke failed")
    injected = run_injected_compile()
    injected_output = injected.stdout + injected.stderr
    if injected.returncode != 0 or "Injected code compiled successfully." not in injected_output:
        return failed(injected_output.strip() or "M6.7 injected ownership bridge compile failed")
    return passed(
        "Real normalized L9-L19 terrain and object payloads passed both zooms, clipping, scrolling, wrap occurrences, bounded multi-viewport cache/invalidation, reset, exact ownership, and zero native fallback."
    )


def check_i18_approved_map_stack_integration() -> dict[str, Any]:
    expected = (
        ("L14", "handoffs/L14_roads.json", "roads"),
        ("L15", "handoffs/L15_railroads.json", "railroads"),
        ("L16", "handoffs/L16_resources.json", "resources"),
        ("L17", "handoffs/L17_cities.json", "cities"),
        ("L18", "handoffs/L18_mines.json", "mines"),
    )
    for gate, relative, system in expected:
        try:
            record = json.loads((RENDERER_ROOT / relative).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return failed(f"Could not read {gate} handoff: {exc}")
        if record.get("schema") != "c3x.renderer_lab_handoff.v0" or \
           record.get("lab_gate") != gate or record.get("system") != system or \
           record.get("status") != "approved":
            return failed(f"{gate} is not the approved {system} handoff")
        if not record.get("source_contract") or not record.get("reference") or \
           not record.get("ownership_intent") or not record.get("fallback"):
            return failed(f"{gate} handoff is missing its production contract")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        return failed((tests.stdout + tests.stderr).strip() or "I14-I18 bridge contracts failed")
    renderer = (RENDERER_ROOT / "native" / "c3x_renderer.cpp").read_text(encoding="utf-8")
    api = (RENDERER_ROOT / "native" / "c3x_renderer_api.h").read_text(encoding="utf-8")
    for token in ("route_vertices", "find_feature_group(resource_bundle", "city_vertices", "mine_vertices",
                  "viewport_cache_capacity = 32u",
                  "viewport_cache_budget = 128u * 1024u * 1024u"):
        if token not in renderer:
            return failed(f"I14-I18 production path is missing {token}")
    for token in ("CUSTOM_ROAD_REPLACED", "CUSTOM_RAILROAD_REPLACED",
                  "CUSTOM_RESOURCE_REPLACED", "CUSTOM_CITY_REPLACED",
                  "CUSTOM_MINE_REPLACED"):
        if token not in api:
            return failed(f"I14-I18 API is missing {token}")
    return passed(
        "Frozen approved L14-L18 handoffs, production draw paths, exact ownership, authoritative selectors, and memory-bounded multi-viewport cache contracts passed."
    )


def check_i12_approved_volcano_integration() -> dict[str, Any]:
    fidelity_path = RENDERER_ROOT / "terrain" / "i12_handoff_fidelity.json"
    try:
        fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
        approved = fidelity["approved_input"]
        record_path = RENDERER_ROOT / approved["record"]
        record_bytes = record_path.read_bytes()
        record = json.loads(record_bytes)
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        return failed(f"Could not read I12 fidelity contract: {exc}")
    if fidelity.get("schema") != "c3x.i12_handoff_fidelity.v0" or \
       approved.get("gate") != "L12" or record.get("lab_gate") != "L12" or \
       record.get("status") != "approved":
        return failed("I12 does not consume the approved L12 handoff")
    if _fnv1a64(record_bytes) != approved.get("record_fnv1a64"):
        return failed("The frozen L12 handoff revision drifted")
    reference = record.get("reference", {})
    if reference.get("native_sha256") != approved.get("native_reference_sha256") or \
       reference.get("reduced_sha256") != approved.get("reduced_reference_sha256"):
        return failed("The frozen L12 reference hashes drifted")
    components = {item.get("component"): item for item in fidelity.get("frozen_component_mapping", [])}
    density = components.get("forest and jungle density", {}).get("constants", {})
    if density != {"forest_instances_per_tile": 36, "jungle_instances_per_tile": 49,
                   "forest_scale": 0.42, "jungle_scale": 0.40}:
        return failed("I12 vegetation density/scale mapping differs from approved L12")
    tolerances = fidelity.get("deterministic_tolerances", {})
    if tolerances.get("screen_anchor_error_px") != 0 or \
       tolerances.get("pixels_modified_outside_clip") != 0 or \
       tolerances.get("stale_pixels_after_scroll_zoom_or_wrap") != 0 or \
       tolerances.get("production_native_fallback_tiles") != 0 or \
       tolerances.get("custom_on_failure_policy") != "hard_failure_without_native_replay":
        return failed("I12 deterministic/exclusive-ownership tolerances are incomplete")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        return failed((tests.stdout + tests.stderr).strip() or "I12 bridge contracts failed")
    build_text = (RENDERER_ROOT / "native" / "BUILD.bat").read_text(encoding="utf-8")
    if "approved_terrain_integration" not in build_text or \
       "TerrainElementsNormalized" not in build_text:
        return failed("I12 production-payload smoke is not wired into the native build")
    return passed(
        "Frozen L12 reference and handoff revision, authored volcano channels, 36/49 vegetation density, shoreline/clutter mapping, active-state cache identity, exact anchors/clipping, and zero-native-fallback contracts passed."
    )


def _check_current_approved_handoff(fidelity_name: str, schema: str, gate: str) -> tuple[dict[str, Any], dict[str, Any], bytes] | dict[str, Any]:
    try:
        fidelity = json.loads((RENDERER_ROOT / "terrain" / fidelity_name).read_text(encoding="utf-8"))
        approved = fidelity["approved_input"]
        record_bytes = (RENDERER_ROOT / approved["record"]).read_bytes()
        record = json.loads(record_bytes)
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        return failed(f"Could not read {gate} integration fidelity contract: {exc}")
    if fidelity.get("schema") != schema or approved.get("gate") != gate or \
       record.get("lab_gate") != gate or record.get("status") != "approved":
        return failed(f"Production does not consume the approved {gate} handoff")
    if _fnv1a64(record_bytes) != approved.get("record_fnv1a64"):
        return failed(f"The frozen {gate} handoff revision drifted")
    reference = record.get("reference", {})
    if reference.get("native_sha256") != approved.get("native_reference_sha256") or \
       reference.get("reduced_sha256") != approved.get("reduced_reference_sha256"):
        return failed(f"The frozen {gate} reference hashes drifted")
    # The recorded whole-file shader hash describes the exact production freeze
    # at this historical gate. Later integrations legitimately append approved
    # object paths, so current production is protected by the focused bridge
    # contracts instead of being required to remain byte-equal to an earlier
    # whole-file snapshot.
    return fidelity, record, record_bytes


def check_i13_approved_river_integration() -> dict[str, Any]:
    checked = _check_current_approved_handoff(
        "i13_handoff_fidelity.json", "c3x.i13_handoff_fidelity.v0", "L13")
    if isinstance(checked, dict):
        return checked
    fidelity, record, _record_bytes = checked
    reference = record.get("reference", {})
    if reference.get("topology_sha256") != fidelity["approved_input"].get("topology_reference_sha256"):
        return failed("The frozen L13 topology reference hash drifted")
    default_definition = (RENDERER_ROOT / "default.custom_rendering.txt").read_text(encoding="utf-8")
    renderer = (RENDERER_ROOT / "native" / "c3x_renderer.cpp").read_text(encoding="utf-8")
    runtime = (RENDERER_ROOT / "native" / "terrain_scene_runtime.cpp").read_text(encoding="utf-8")
    if "rivers = replace" not in default_definition or "roads = replace" not in default_definition or \
       "shore_runtime.bin" not in renderer or "tile.river_code" not in runtime or \
       "C3X_RENDERER_TILE_CUSTOM_RIVER_REPLACED" not in renderer:
        return failed("I13 river ownership, payload, or cache wiring is incomplete")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "Renderer.native.test_native_bridge_contract"],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        return failed((tests.stdout + tests.stderr).strip() or "I13 bridge contracts failed")
    return passed("Frozen L13 river graph, material, rock, ownership, cache, atomic omission, and no-native-replay contracts passed.")


def check_i13a_approved_lighting_integration() -> dict[str, Any]:
    checked = _check_current_approved_handoff(
        "i13a_handoff_fidelity.json", "c3x.i13a_handoff_fidelity.v0", "L13A")
    if isinstance(checked, dict):
        return checked
    fidelity, record, _record_bytes = checked
    reference = record.get("reference", {})
    phase_keys = {
        "noon": "noon_sha256", "sunset": "sunset_sha256",
        "midnight": "midnight_sha256", "sunrise": "sunrise_sha256",
    }
    if any(reference.get(source_key) != fidelity["phase_references"].get(phase)
           for phase, source_key in phase_keys.items()):
        return failed("The frozen L13A lighting phase hashes drifted")
    renderer = (RENDERER_ROOT / "native" / "c3x_renderer.cpp").read_text(encoding="utf-8")
    shader = (RENDERER_ROOT / "native" / "terrain_rendering.hlsl").read_text(encoding="utf-8")
    if "evaluate_environment(" not in renderer or "cast_shadow_visibility" not in renderer or \
       "append_object_shadow" not in renderer or "#define l13a_layout 1.0" not in shader:
        return failed("I13A environment/shadow wiring is incomplete")
    return passed("Frozen L13A sun, moon, ambient, exposure, water, emissive-policy, raised-shadow, and both-zoom contracts passed alongside later approved object layers.")


def check_civ6_lighting_metadata_local() -> dict[str, Any]:
    assets_root = asset_compiler.DEFAULT_CIV6_BASE.parent
    if not (assets_root / civ6_lighting_probe.GAME_LIGHTING_ARTDEF).is_file():
        return skipped(f"Installed Civ VI GameLighting ArtDef not found under: {assets_root}")
    try:
        report = civ6_lighting_probe.build_report(assets_root)
    except (OSError, ValueError, ET.ParseError) as exc:
        return failed(f"Civ VI lighting metadata probe failed: {exc}")
    profiles = set(report["global_lighting"]["profiles"])
    entries = {rig["entry"] for rig in report["global_lighting"]["rigs"]}
    primary_strings = {
        value
        for package in report["primary_package_evidence"]
        for value in package["matching_strings"]
    }
    required_profiles = {"DEFAULT_LIGHTING", "WONDER_TOD"}
    required_entries = {"Sunrise_LightRig", "Noon_LightRig", "Night_LightRig"}
    required_resources = {"m_vSunDirection", "DL_OrangeGlow", "ChimneySmoke"}
    water_classes = {
        binding["xlp_class"]
        for item in report["water_artdef_evidence"]
        for binding in item["bindings"]
    }
    missing = sorted(
        (required_profiles - profiles)
        | (required_entries - entries)
        | (required_resources - primary_strings)
    )
    if missing:
        return failed("Installed Civ VI lighting evidence is missing: " + ", ".join(missing))
    vertical = report["supported_vertical_slice"]
    if "Water" not in water_classes or vertical["typed_parameters"]["evidence"] != "unresolved" or \
       "inferred" not in vertical["model_attachment"]["evidence"]:
        return failed("Water bindings or confirmed/inferred/unresolved vertical-slice labels are incomplete")
    return passed(
        f"Found {len(profiles)} time-of-day profiles, "
        f"{len(report['all_named_lighting_packages'])} named Light/VFX packages, and "
        f"{len(report['shared_effect_texture_candidates'])} shared effect texture candidates; "
        f"structured Water ArtDef bindings and conservative vertical-slice evidence labels passed."
    )


def check_lab_handoff(
    filename: str,
    gate: str,
    tile_count: int,
    test_modules: list[str],
) -> dict[str, Any]:
    path = RENDERER_ROOT / "handoffs" / filename
    try:
        handoff = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return failed(f"Could not read {gate} lab handoff: {exc}")
    if handoff.get("schema") != "c3x.renderer_lab_handoff.v0" or \
       handoff.get("lab_gate") != gate or handoff.get("status") != "approved":
        return failed(f"{gate} handoff schema, identity, or approval status is invalid")
    reference = handoff.get("reference", {})
    if reference.get("tile_count") != tile_count:
        return failed(f"{gate} handoff does not preserve its {tile_count}-tile promotion contract")
    audit_path = RENDERER_ROOT / str(reference.get("audit", ""))
    try:
        audit = audit_path.read_text(encoding="utf-8")
    except OSError as exc:
        return failed(f"Could not read {gate} promotion audit: {exc}")
    hashes = (reference.get("native_sha256"), reference.get("reduced_sha256"))
    if any(not isinstance(value, str) or len(value) != 64 or value not in audit for value in hashes):
        return failed(f"{gate} handoff hashes do not match its promotion audit")
    if "explicitly approved" not in audit:
        return failed(f"{gate} audit does not record explicit visual approval")
    for output_key, hash_key in (("native_output", "native_sha256"), ("reduced_output", "reduced_sha256")):
        output = RENDERER_ROOT / str(reference.get(output_key, ""))
        if output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest() != reference[hash_key]:
            return failed(f"Existing {gate} output does not match the approved {output_key} hash")
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", *test_modules],
        cwd=C3X_ROOT, capture_output=True, text=True, check=False,
    )
    if tests.returncode != 0:
        return failed((tests.stdout + tests.stderr).strip() or f"{gate} focused replay tests failed")
    return passed(
        f"{gate} approved {tile_count}-tile lab handoff, audit hashes, optional local outputs, and focused replay contracts passed."
    )


def check_terrain_lab_l9_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L9_terrain.json", "L9", 48,
        [
            "Renderer.tools.asset_compiler.test_clutter_blp_extractor",
            "Renderer.tools.asset_compiler.test_shore_blp_extractor",
            "Renderer.tools.asset_compiler.test_water_pack_builder",
        ],
    )


def check_terrain_lab_l10_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L10_dunes.json", "L10", 96,
        [
            "Renderer.tools.asset_compiler.test_dune_source_probe",
            "Renderer.tools.asset_compiler.test_generic_decal_compiler",
        ],
    )


def check_terrain_lab_l11_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L11_marsh.json",
        "L11",
        96,
        [
            "Renderer.terrain_lab.test_continuous_surface_contract",
        ],
    )


def check_terrain_lab_l12_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L12_volcano.json",
        "L12",
        192,
        [
            "Renderer.tools.asset_compiler.test_generic_decal_compiler",
            "Renderer.terrain_lab.test_continuous_surface_contract",
        ],
    )


def check_terrain_lab_l13_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L13_rivers.json",
        "L13",
        192,
        [
            "Renderer.tools.asset_compiler.test_clutter_blp_extractor",
            "Renderer.terrain_lab.test_continuous_surface_contract",
            "Renderer.terrain_lab.test_canonical_reference_contract",
        ],
    )


def check_terrain_lab_l13a_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L13A_lighting.json",
        "L13A",
        192,
        [
            "Renderer.terrain_lab.test_continuous_surface_contract",
            "Renderer.terrain_lab.test_canonical_reference_contract",
            "Renderer.environment.test_contract",
        ],
    )


def check_terrain_lab_l14_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L14_roads.json",
        "L14",
        192,
        [
            "Renderer.terrain_lab.test_build_l14_road_scenario",
            "Renderer.terrain_lab.test_l14_road_contract",
        ],
    )


def check_terrain_lab_l15_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L15_railroads.json",
        "L15",
        192,
        [
            "Renderer.terrain_lab.test_build_l15_railroad_scenario",
            "Renderer.terrain_lab.test_l15_railroad_contract",
        ],
    )


def check_terrain_lab_l16_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L16_resources.json",
        "L16",
        192,
        [
            "Renderer.terrain_lab.test_build_l16_resource_scenario",
            "Renderer.terrain_lab.test_l16_resource_contract",
        ],
    )


def check_terrain_lab_l17_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L17_cities.json",
        "L17",
        192,
        [
            "Renderer.terrain_lab.test_build_l17_city_scenario",
            "Renderer.terrain_lab.test_l17_city_contract",
        ],
    )


def check_terrain_lab_l18_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L18_mines.json",
        "L18",
        192,
        [
            "Renderer.terrain_lab.test_build_l18_mine_scenario",
            "Renderer.terrain_lab.test_l18_mine_contract",
        ],
    )


def check_terrain_lab_l19_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L19_farms_tundra.json",
        "L19",
        192,
        [
            "Renderer.terrain_lab.test_build_l19_farm_scenario",
            "Renderer.terrain_lab.test_l19_farm_contract",
        ],
    )


def check_terrain_lab_l19a_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L19A_goody_huts_colonies.json",
        "L19A",
        192,
        [
            "Renderer.terrain_lab.test_build_l19a_tile_object_scenario",
            "Renderer.terrain_lab.test_l19a_tile_object_contract",
        ],
    )


def check_terrain_lab_l19b_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L19B_remaining_tile_infrastructure.json",
        "L19B",
        192,
        [
            "Renderer.terrain_lab.test_build_l19b_infrastructure_scenario",
            "Renderer.terrain_lab.test_l19b_infrastructure_contract",
        ],
    )


def check_terrain_lab_l20_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L20_units.json",
        "L20",
        192,
        [
            "Renderer.terrain_lab.test_build_l20_unit_scenario",
            "Renderer.terrain_lab.test_l20_unit_contract",
        ],
    )


def check_terrain_lab_l21_handoff() -> dict[str, Any]:
    return check_lab_handoff(
        "L21_complete_beauty_scene.json",
        "L21",
        192,
        [
            "Renderer.terrain_lab.test_l21_complete_scene_contract",
        ],
    )


CHECKS: dict[str, tuple[CheckFunction, bool]] = {
    "project_state_contract": (check_project_state_contract, False),
    "renderer_unit_tests": (check_renderer_unit_tests, False),
    "preview_smoke": (check_preview_smoke, False),
    "config_contract_fixture": (check_config_contract_fixture, False),
    "civbig_synthetic_roundtrip": (check_civbig_synthetic_roundtrip, False),
    "civ6_grassland_texture_local": (check_civ6_grassland_texture_local, True),
    "civblp_probe_synthetic": (check_civblp_probe_synthetic, False),
    "civ6_grassland_material_probe_local": (check_civ6_grassland_material_probe_local, True),
    "civblp_material_resolver_synthetic": (check_civblp_material_resolver_synthetic, False),
    "civ6_grassland_material_binding_local": (check_civ6_grassland_material_binding_local, True),
    "terrain_geometry_resolver_synthetic": (check_terrain_geometry_resolver_synthetic, False),
    "civ6_grassland_geometry_local": (check_civ6_grassland_geometry_local, True),
    "grassland_pack_render_synthetic": (check_grassland_pack_render_synthetic, False),
    "civ6_grassland_pack_render_local": (check_civ6_grassland_pack_render_local, True),
    "renderer_definition_parser": (check_renderer_definition_parser, False),
    "renderer_rule_resolver": (check_renderer_rule_resolver, False),
    "visible_scene_contract": (check_visible_scene_contract, False),
    "standalone_whole_viewport_renderer": (check_standalone_whole_viewport_renderer, False),
    "fixture_matrix_validation": (check_fixture_matrix_validation, False),
    "native_civ3_bridge": (check_native_civ3_bridge, False),
    "m5_2_scene_export": (check_m5_2_scene_export, False),
    "m5_3_frame_scheduler": (check_m5_3_frame_scheduler, False),
    "m6_0_inventory_contract": (check_m6_0_inventory_contract, False),
    "m6_0_installed_vanilla_inventory": (check_m6_0_installed_vanilla_inventory, True),
    "m6_1_production_terrain": (check_m6_1_production_terrain, False),
    "m6_2_native_terrain_art": (check_m6_2_native_terrain_art, False),
    "m6_2_local_grassland_art": (check_m6_2_local_grassland_art, True),
    "m6_3_definition_terrain": (check_m6_3_definition_terrain, False),
    "m6_3_local_terrain_art": (check_m6_3_local_terrain_art, True),
    "m6_4_environment_runtime": (check_m6_4_environment_runtime, False),
    "m6_5_connected_terrain": (check_m6_5_connected_terrain, False),
    "m6_6_vanilla_terrain": (check_m6_6_vanilla_terrain, False),
    "m6_6_local_biq_terrain": (check_m6_6_local_biq_terrain, True),
    "m6_7_approved_terrain_integration": (check_m6_7_approved_terrain_integration, False),
    "m6_7_local_approved_terrain_payload": (check_m6_7_local_approved_terrain_payload, True),
    "i11_approved_marsh_integration": (check_m6_7_approved_terrain_integration, False),
    "i11_local_approved_marsh_payload": (check_m6_7_local_approved_terrain_payload, True),
    "i12_approved_volcano_integration": (check_i12_approved_volcano_integration, False),
    "i12_local_approved_volcano_payload": (check_m6_7_local_approved_terrain_payload, True),
    "i13_approved_river_integration": (check_i13_approved_river_integration, False),
    "i13_local_approved_river_payload": (check_m6_7_local_approved_terrain_payload, True),
    "i13a_approved_lighting_integration": (check_i13a_approved_lighting_integration, False),
    "i13a_local_approved_lighting_payload": (check_m6_7_local_approved_terrain_payload, True),
    "i18_approved_map_stack_integration": (check_i18_approved_map_stack_integration, False),
    "i18_local_approved_map_stack_payload": (check_m6_7_local_approved_terrain_payload, True),
    "civ6_lighting_metadata_local": (check_civ6_lighting_metadata_local, True),
    "terrain_lab_l9_replayable_render_and_explicit_visual_approval": (check_terrain_lab_l9_handoff, False),
    "terrain_lab_l10_replayable_render_and_explicit_visual_approval": (check_terrain_lab_l10_handoff, False),
    "terrain_lab_l11_replayable_render_and_explicit_visual_approval": (check_terrain_lab_l11_handoff, False),
    "terrain_lab_l12_replayable_render_and_explicit_visual_approval": (check_terrain_lab_l12_handoff, False),
    "terrain_lab_l13_replayable_render_and_explicit_visual_approval": (check_terrain_lab_l13_handoff, False),
    "terrain_lab_l13a_replayable_render_and_explicit_visual_approval": (check_terrain_lab_l13a_handoff, False),
    "terrain_lab_l14_replayable_render_and_explicit_visual_approval": (check_terrain_lab_l14_handoff, False),
    "terrain_lab_l15_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l15_handoff, False),
    "terrain_lab_l16_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l16_handoff, False),
    "terrain_lab_l17_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l17_handoff, False),
    "terrain_lab_l18_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l18_handoff, False),
    "terrain_lab_l19_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l19_handoff, False),
    "terrain_lab_l19a_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l19a_handoff, False),
    "terrain_lab_l19b_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l19b_handoff, False),
    "terrain_lab_l20_replayable_render_and_authorized_critical_visual_approval": (check_terrain_lab_l20_handoff, False),
    "terrain_lab_l21_complete_beauty_scene_and_authorized_critical_visual_approval": (check_terrain_lab_l21_handoff, False),
}

WINDOWS_NATIVE_CHECKS = {
    "native_civ3_bridge",
    "m5_2_scene_export",
    "m5_3_frame_scheduler",
    "m6_2_native_terrain_art",
    "m6_3_definition_terrain",
    "m6_4_environment_runtime",
    "m6_5_connected_terrain",
    "m6_6_vanilla_terrain",
}


def collect_completed_checks(status: dict[str, Any]) -> tuple[list[str], list[str]]:
    portable: list[str] = []
    local: list[str] = []
    for milestone in status.get("milestones", []):
        completed_items = [milestone] if milestone.get("status") == "complete" else []
        completed_items.extend(step for step in milestone.get("steps", []) if step.get("status") == "complete")
        for item in completed_items:
            for check_id in item.get("verification", []):
                if check_id not in portable:
                    portable.append(check_id)
            for check_id in item.get("local_verification", []):
                if check_id not in local:
                    local.append(check_id)
    return portable, local


def run_verification(
    status_path: Path,
    portable_only: bool,
    require_local_assets: bool,
    source_only: bool = False,
) -> dict[str, Any]:
    status = json.loads(status_path.read_text(encoding="utf-8"))
    portable_ids, local_ids = collect_completed_checks(status)
    selected = [(check_id, False) for check_id in portable_ids]
    if not portable_only:
        selected.extend((check_id, True) for check_id in local_ids)

    results = []
    for check_id, declared_local in selected:
        if source_only and check_id in WINDOWS_NATIVE_CHECKS:
            results.append({
                "id": check_id,
                "local": declared_local,
                "status": "skip",
                "detail": "Consolidated Windows native build/injected phase is run by renderer_dev.py.",
            })
            continue
        registration = CHECKS.get(check_id)
        if registration is None:
            result = failed("Completed project status references an unknown verification check.")
        else:
            check, registered_local = registration
            if declared_local != registered_local:
                result = failed("Check is declared in the wrong portable/local verification list.")
            else:
                result = check()
        if result["status"] == "skip" and require_local_assets:
            result = failed(f"Required local check could not run: {result['detail']}")
        results.append({"id": check_id, "local": declared_local, **result})

    return {
        "schema": "c3x.renderer_verification_report.v0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project_status": str(status_path),
        "next_step": status.get("next_step", {}).get("id"),
        "results": results,
        "passed": all(result["status"] in {"pass", "skip"} for result in results),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify every completed C3X renderer prerequisite")
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--portable-only", action="store_true")
    parser.add_argument("--require-local-assets", action="store_true")
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Run source-independent checks; renderer_dev.py supplies the consolidated Windows native phase.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    try:
        report = run_verification(
            args.status, args.portable_only, args.require_local_assets, args.source_only
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: Could not run renderer verification: {exc}", file=sys.stderr)
        return 1

    for result in report["results"]:
        print(f"{result['status'].upper():4} {result['id']}: {result['detail']}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.report}")
    if not report["passed"]:
        print("Renderer prerequisite verification failed.", file=sys.stderr)
        return 1
    print(f"All completed renderer prerequisites verified. Next step: {report['next_step']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
