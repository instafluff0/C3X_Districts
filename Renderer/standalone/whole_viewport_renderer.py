#!/usr/bin/env python3
"""Deterministic whole-viewport renderer for C3X visible-scene fixtures.

This is the source-independent M4 reference renderer.  It deliberately owns no
window or presentation surface: validated scenes, merged definitions, and
normalized packs are rasterized into an in-memory map-sized color/depth target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from Renderer.definitions import definition_parser
from Renderer.preview.render_iso import Canvas, clamp_channel
from Renderer.preview.render_textured_patch import (
    BACKGROUND,
    DdsBc3Texture,
    edge,
    safe_pack_path,
    write_png,
)
from Renderer.scenes import scene_contract


CATALOG_SCHEMA = "c3x.renderer_definition_catalog.v0"
PACK_SCHEMA = "c3x.asset_pack.v0"
MESH_SCHEMA = "c3x.normalized_mesh.v0"
MATERIAL_SCHEMA = "c3x.material.v0"
SEASON_TINTS = {
    "summer": (1.04, 1.00, 0.90),
    "fall": (1.10, 0.88, 0.68),
    "winter": (0.80, 0.90, 1.10),
    "spring": (0.92, 1.07, 0.94),
}
RENDERER_ID = "c3x.standalone.whole_viewport"
RENDERER_VERSION = "1"


@dataclass(frozen=True)
class RuntimeAsset:
    definition_id: str
    logical_id: str
    mesh: Mapping[str, Any]
    material: Mapping[str, Any]
    texture: DdsBc3Texture
    scale: float
    offset_x_px: int
    offset_y_px: int


@dataclass(frozen=True)
class RenderFrame:
    canvas: Canvas
    depth_buffer: tuple[float, ...]
    owner_buffer: tuple[str | None, ...]
    primitive_buffer: tuple[int | None, ...]
    inspection: Mapping[str, Any]
    stats: Mapping[str, Any]


def _inside(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label} must contain {length} numbers")
    result = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{label} must contain {length} numbers")
        converted = float(component)
        if not math.isfinite(converted):
            raise ValueError(f"{label} contains a non-finite number")
        result.append(converted)
    return result


def validate_runtime_mesh(mesh: Mapping[str, Any]) -> None:
    if mesh.get("schema") != MESH_SCHEMA:
        raise ValueError("Unsupported normalized-mesh schema")
    if mesh.get("topology", {}).get("primitive") != "triangles":
        raise ValueError("Normalized mesh must use triangle topology")
    vertices = mesh.get("vertices")
    indices = mesh.get("topology", {}).get("indices")
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError("Normalized mesh must contain at least three vertices")
    if not isinstance(indices, list) or not indices or len(indices) % 3:
        raise ValueError("Normalized mesh indices must contain complete triangles")
    for index, vertex in enumerate(vertices):
        if not isinstance(vertex, Mapping):
            raise ValueError(f"Mesh vertex {index} is not an object")
        _finite_vector(vertex.get("position"), 3, f"vertex {index} position")
        _finite_vector(vertex.get("normal"), 3, f"vertex {index} normal")
        _finite_vector(vertex.get("uv0"), 2, f"vertex {index} uv0")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= len(vertices) for index in indices):
        raise ValueError("Normalized mesh contains an out-of-range triangle index")


class PackAssetLoader:
    """Resolve catalog asset IDs to normalized pack payloads with root safety."""

    def __init__(
        self,
        catalog: Mapping[str, Any],
        *,
        mod_root: Path,
        scenario_root: Path | None = None,
    ) -> None:
        if catalog.get("schema") != CATALOG_SCHEMA:
            raise ValueError("Unsupported renderer-definition catalog schema")
        self.catalog = catalog
        self.mod_root = mod_root.resolve()
        self.scenario_root = scenario_root.resolve() if scenario_root else None
        self.pack_definitions = {entry["id"]: entry for entry in catalog.get("packs", [])}
        self.asset_definitions = {entry["id"]: entry for entry in catalog.get("assets", [])}
        self._manifest_cache: dict[str, tuple[Path, Mapping[str, Any]]] = {}
        self._asset_cache: dict[str, RuntimeAsset] = {}
        self._input_records: dict[str, dict[str, Any]] = {}
        self.availability_errors: dict[str, str] = {}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _pack_root(self, pack_id: str) -> Path:
        definition = self.pack_definitions.get(pack_id)
        if definition is None:
            raise ValueError(f"Unknown pack definition {pack_id!r}")
        record = definition["values"]["path"]
        kind = record["root"]
        if kind == "file":
            return Path(record["path"]).resolve()
        root = self.mod_root if kind == "mod" else self.scenario_root
        if root is None:
            raise ValueError(f"Pack {pack_id!r} requires a configured {kind} root")
        candidate = (root / Path(record["path"])).resolve()
        if not _inside(root, candidate):
            raise ValueError(f"Pack {pack_id!r} escapes its configured {kind} root")
        return candidate

    def _manifest(self, pack_id: str) -> tuple[Path, Mapping[str, Any]]:
        cached = self._manifest_cache.get(pack_id)
        if cached is not None:
            return cached
        root = self._pack_root(pack_id)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != PACK_SCHEMA:
            raise ValueError(f"Pack {pack_id!r} has an unsupported manifest schema")
        cached = (root, manifest)
        self._manifest_cache[pack_id] = cached
        return cached

    @staticmethod
    def _logical_entry(manifest: Mapping[str, Any], logical_id: str) -> Mapping[str, Any]:
        assets = manifest.get("assets", {})
        if isinstance(assets, Mapping) and isinstance(assets.get(logical_id), Mapping):
            return assets[logical_id]
        # Compatibility with the M1 terrain-only manifest while normalized packs
        # migrate to the general logical-ID table.
        parts = logical_id.split("/")
        if len(parts) == 3 and parts[0] == "terrain" and parts[2] == "base":
            terrain = manifest.get("terrains", {}).get(parts[1])
            if isinstance(terrain, Mapping):
                return terrain
        raise ValueError(f"Pack has no logical asset {logical_id!r}")

    def _locate(self, asset_id: str) -> tuple[Mapping[str, Any], Path, Mapping[str, Any]]:
        definition = self.asset_definitions.get(asset_id)
        if definition is None:
            raise ValueError(f"Unknown catalog asset {asset_id!r}")
        values = definition["values"]
        root, manifest = self._manifest(values["pack"])
        entry = self._logical_entry(manifest, values["asset"])
        if not isinstance(entry.get("mesh"), str) or not isinstance(entry.get("material"), str):
            raise ValueError(f"Logical asset {values['asset']!r} has no normalized mesh/material")
        return definition, root, entry

    def available_asset_ids(self) -> set[str]:
        available = set()
        self.availability_errors.clear()
        for asset_id in self.asset_definitions:
            try:
                _definition, root, entry = self._locate(asset_id)
                mesh = safe_pack_path(root, entry["mesh"])
                material = safe_pack_path(root, entry["material"])
                if not mesh.is_file() or not material.is_file():
                    raise ValueError("normalized mesh or material file is missing")
                available.add(asset_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self.availability_errors[asset_id] = str(exc)
        return available

    def load(self, asset_id: str) -> RuntimeAsset:
        cached = self._asset_cache.get(asset_id)
        if cached is not None:
            return cached
        definition, root, entry = self._locate(asset_id)
        mesh_path = safe_pack_path(root, entry["mesh"])
        material_path = safe_pack_path(root, entry["material"])
        mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
        material = json.loads(material_path.read_text(encoding="utf-8"))
        validate_runtime_mesh(mesh)
        if material.get("schema") != MATERIAL_SCHEMA:
            raise ValueError("Unsupported normalized-material schema")
        base_color = material.get("base_color")
        if not isinstance(base_color, Mapping) or not isinstance(base_color.get("texture"), str):
            raise ValueError("Normalized material has no base-color texture")
        texture_path = safe_pack_path(root, base_color["texture"])
        texture = DdsBc3Texture.from_file(texture_path)
        values = definition["values"]
        asset = RuntimeAsset(
            definition_id=asset_id,
            logical_id=values["asset"],
            mesh=mesh,
            material=material,
            texture=texture,
            scale=float(values.get("scale", 1.0)),
            offset_x_px=int(values.get("offset_x_px", 0)),
            offset_y_px=int(values.get("offset_y_px", 0)),
        )
        self._asset_cache[asset_id] = asset
        manifest_path = root / "manifest.json"
        self._input_records[asset_id] = {
            "asset_definition_id": asset_id,
            "logical_asset_id": values["asset"],
            "pack_id": values["pack"],
            "files": [
                {"role": "manifest", "path": "manifest.json", "sha256": self._sha256(manifest_path)},
                {"role": "mesh", "path": mesh_path.relative_to(root).as_posix(), "sha256": self._sha256(mesh_path)},
                {"role": "material", "path": material_path.relative_to(root).as_posix(), "sha256": self._sha256(material_path)},
                {"role": "base_color", "path": texture_path.relative_to(root).as_posix(), "sha256": self._sha256(texture_path)},
            ],
        }
        return asset

    def loaded_input_records(self) -> list[dict[str, Any]]:
        return [self._input_records[key] for key in sorted(self._input_records)]


def _normalize(vector: Iterable[float]) -> tuple[float, float, float]:
    x, y, z = vector
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return (x / length, y / length, z / length)


def lighting_state(environment: Mapping[str, Any], hour: int, season: str) -> dict[str, Any]:
    day = max(0.0, math.cos((hour - 12) * math.pi / 12.0))
    sun_elevation = math.radians(62.0 * day)
    azimuth = math.radians(float(environment.get("sun_azimuth_degrees", 135.0)))
    sun_direction = _normalize(
        (
            math.cos(azimuth) * math.cos(sun_elevation),
            math.sin(azimuth) * math.cos(sun_elevation),
            math.sin(sun_elevation),
        )
    )
    midnight = environment.get("midnight_ambient_color", [22, 30, 52])
    night_exposure = float(environment.get("night_exposure", 0.35))
    night_ambient = tuple(float(channel) / 255.0 * night_exposure for channel in midnight)
    day_ambient = (0.36, 0.38, 0.42)
    ambient = tuple(night_ambient[index] * (1.0 - day) + day_ambient[index] * day for index in range(3))
    sun_color = tuple(float(channel) / 255.0 for channel in environment.get("noon_sun_color", [255, 244, 220]))
    seasonal = bool(environment.get("seasonal_materials", False))
    season_tint = SEASON_TINTS[season] if seasonal else (1.0, 1.0, 1.0)
    return {
        "day_factor": day,
        "sun_direction": sun_direction,
        "sun_color": sun_color,
        "ambient": ambient,
        "direct_strength": 0.74 * day,
        "season_tint": season_tint,
    }


def shade_sample(
    color: tuple[int, int, int], normal: tuple[float, float, float], lighting: Mapping[str, Any]
) -> tuple[int, int, int]:
    nx, ny, nz = _normalize(normal)
    sx, sy, sz = lighting["sun_direction"]
    lambert = max(0.0, nx * sx + ny * sy + nz * sz)
    return tuple(
        clamp_channel(
            color[index]
            * lighting["season_tint"][index]
            * (
                lighting["ambient"][index]
                + lighting["direct_strength"] * lambert * lighting["sun_color"][index]
            )
        )
        for index in range(3)
    )


class WholeViewportRenderer:
    """In-memory orthographic renderer with explicit creation/resize/teardown."""

    def __init__(self, width: int, height: int, background: tuple[int, int, int] = BACKGROUND) -> None:
        self.background = background
        self.state = "new"
        self.generation = 0
        self.width = 0
        self.height = 0
        self.canvas: Canvas | None = None
        self.depth_buffer: list[float] = []
        self.owner_buffer: list[str | None] = []
        self.primitive_buffer: list[int | None] = []
        self.resize(width, height)

    def resize(self, width: int, height: int) -> bool:
        if self.state == "closed":
            raise RuntimeError("Renderer is closed")
        if width < 1 or height < 1:
            raise ValueError("Renderer dimensions must be positive")
        if self.state == "ready" and (width, height) == (self.width, self.height):
            return False
        self.width = width
        self.height = height
        self.canvas = Canvas(width, height, self.background)
        self.depth_buffer = [-math.inf] * (width * height)
        self.owner_buffer = [None] * (width * height)
        self.primitive_buffer = [None] * (width * height)
        self.generation += 1
        self.state = "ready"
        return True

    def close(self) -> None:
        self.canvas = None
        self.depth_buffer = []
        self.owner_buffer = []
        self.primitive_buffer = []
        self.state = "closed"

    def __enter__(self) -> "WholeViewportRenderer":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

    def _clear(self) -> None:
        if self.state != "ready":
            raise RuntimeError("Renderer is closed")
        self.canvas = Canvas(self.width, self.height, self.background)
        self.depth_buffer = [-math.inf] * (self.width * self.height)
        self.owner_buffer = [None] * (self.width * self.height)
        self.primitive_buffer = [None] * (self.width * self.height)

    def _draw_asset(
        self,
        instance_id: str,
        map_x: int,
        map_y: int,
        anchor: Mapping[str, int],
        asset: RuntimeAsset,
        projection: Mapping[str, Any],
        map_rect: Mapping[str, int],
        lighting: Mapping[str, Any],
        stats: dict[str, int],
    ) -> None:
        assert self.canvas is not None
        basis_x = projection["tile_x_basis_px"]
        basis_y = projection["tile_y_basis_px"]
        basis_z = projection["elevation_basis_px"]
        vertices = asset.mesh["vertices"]
        transformed = []
        for vertex in vertices:
            px, py, pz = (float(value) * asset.scale for value in vertex["position"])
            screen_x = anchor["x"] + asset.offset_x_px + px * basis_x["x"] + py * basis_y["x"] + pz * basis_z["x"]
            screen_y = anchor["y"] + asset.offset_y_px + px * basis_x["y"] + py * basis_y["y"] + pz * basis_z["y"]
            # The third orthographic basis points toward the viewer.  Larger
            # map X/Y and positive elevation are therefore nearer at equal pixels.
            depth = map_x + px + map_y + py + 2.0 * pz
            transformed.append((screen_x, screen_y, depth))

        indices = asset.mesh["topology"]["indices"]
        clip_left = max(0, map_rect["x"])
        clip_top = max(0, map_rect["y"])
        clip_right = min(self.width - 1, map_rect["x"] + map_rect["width"] - 1)
        clip_bottom = min(self.height - 1, map_rect["y"] + map_rect["height"] - 1)
        for primitive in range(len(indices) // 3):
            vertex_indices = indices[primitive * 3 : primitive * 3 + 3]
            points = [(transformed[index][0], transformed[index][1]) for index in vertex_indices]
            area = edge(points[0], points[1], points[2])
            if abs(area) <= 1e-12:
                raise ValueError(f"Asset {asset.definition_id!r} contains a degenerate projected triangle")
            min_x = max(clip_left, int(math.floor(min(point[0] for point in points))))
            max_x = min(clip_right, int(math.ceil(max(point[0] for point in points))))
            min_y = max(clip_top, int(math.floor(min(point[1] for point in points))))
            max_y = min(clip_bottom, int(math.ceil(max(point[1] for point in points))))
            if min_x > max_x or min_y > max_y:
                continue
            stats["triangles_submitted"] += 1
            triangle_vertices = [vertices[index] for index in vertex_indices]
            for pixel_y in range(min_y, max_y + 1):
                for pixel_x in range(min_x, max_x + 1):
                    sample = (pixel_x + 0.5, pixel_y + 0.5)
                    w0 = edge(points[1], points[2], sample) / area
                    w1 = edge(points[2], points[0], sample) / area
                    w2 = 1.0 - w0 - w1
                    if w0 < -1e-9 or w1 < -1e-9 or w2 < -1e-9:
                        continue
                    depth = sum(
                        weight * transformed[index][2]
                        for weight, index in zip((w0, w1, w2), vertex_indices)
                    )
                    buffer_index = pixel_y * self.width + pixel_x
                    if depth <= self.depth_buffer[buffer_index] + 1e-9:
                        stats["pixels_depth_rejected"] += 1
                        continue
                    u = sum(weight * triangle_vertices[index]["uv0"][0] for index, weight in enumerate((w0, w1, w2)))
                    v = sum(weight * triangle_vertices[index]["uv0"][1] for index, weight in enumerate((w0, w1, w2)))
                    normal = tuple(
                        sum(weight * triangle_vertices[index]["normal"][component] for index, weight in enumerate((w0, w1, w2)))
                        for component in range(3)
                    )
                    color = shade_sample(asset.texture.sample(u, v), normal, lighting)
                    self.canvas.set_pixel(pixel_x, pixel_y, color)
                    self.depth_buffer[buffer_index] = depth
                    self.owner_buffer[buffer_index] = instance_id
                    self.primitive_buffer[buffer_index] = primitive
                    stats["pixels_depth_passed"] += 1

    def render(
        self,
        scene: Mapping[str, Any],
        catalog: Mapping[str, Any],
        assets: PackAssetLoader,
    ) -> RenderFrame:
        if self.state == "closed":
            raise RuntimeError("Renderer is closed")
        validated = scene_contract.validate_scene(scene)
        viewport = validated["viewport"]
        self.resize(viewport["width_px"], viewport["height_px"])
        self._clear()

        environments = {item["id"]: item["values"] for item in catalog.get("environments", [])}
        environment = environments.get(validated["environment"]["id"])
        if environment is None:
            raise ValueError(f"Scene environment {validated['environment']['id']!r} is not defined")
        lighting = lighting_state(
            environment,
            validated["environment"]["hour"],
            validated["environment"]["season"],
        )
        available = assets.available_asset_ids()
        inspection = scene_contract.inspect_scene(validated, catalog, available_assets=available)
        stats = {
            "triangles_submitted": 0,
            "pixels_depth_passed": 0,
            "pixels_depth_rejected": 0,
            "rendered_instances": 0,
            "fallback_instances": 0,
        }
        rendered_ids = []
        fallback_ids = []
        for item in inspection["items"]:
            resolution = item["resolution"]
            if resolution["status"] != "matched":
                fallback_ids.append(item["instance_id"])
                stats["fallback_instances"] += 1
                continue
            asset_id = resolution["winner"]["asset_id"]
            try:
                asset = assets.load(asset_id)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                assets.availability_errors[asset_id] = str(exc)
                fallback_ids.append(item["instance_id"])
                stats["fallback_instances"] += 1
                continue
            metadata = item["resolver_input"]
            self._draw_asset(
                item["instance_id"],
                metadata["map_x"],
                metadata["map_y"],
                item["anchor_px"],
                asset,
                validated["projection"],
                viewport["map_rect_px"],
                lighting,
                stats,
            )
            rendered_ids.append(item["instance_id"])
            stats["rendered_instances"] += 1

        assert self.canvas is not None
        anchor_owners = {}
        for item in inspection["items"]:
            anchor = item["anchor_px"]
            x, y = anchor["x"], anchor["y"]
            anchor_owners[item["instance_id"]] = (
                self.owner_buffer[y * self.width + x] if 0 <= x < self.width and 0 <= y < self.height else None
            )
        frame_stats = {
            **stats,
            "rendered_ids": rendered_ids,
            "fallback_ids": fallback_ids,
            "anchor_owners": anchor_owners,
            "non_background_pixels": self.canvas.non_background_pixels(self.background),
            "renderer_generation": self.generation,
            "lighting": lighting,
            "asset_availability_errors": dict(assets.availability_errors),
        }
        return RenderFrame(
            canvas=self.canvas,
            depth_buffer=tuple(self.depth_buffer),
            owner_buffer=tuple(self.owner_buffer),
            primitive_buffer=tuple(self.primitive_buffer),
            inspection=inspection,
            stats=frame_stats,
        )


def load_catalog(
    default_path: Path,
    *,
    mod_root: Path,
    scenario_path: Path | None = None,
    custom_path: Path | None = None,
    scenario_root: Path | None = None,
) -> dict[str, Any]:
    layers = [("default", default_path), ("scenario", scenario_path), ("custom", custom_path)]
    parsed = []
    for layer, path in layers:
        if path is not None:
            parsed.append(
                (
                    layer,
                    definition_parser.parse_definition_file(
                        path, layer, mod_root, scenario_root or mod_root
                    ),
                )
            )
    return definition_parser.merge_layers(parsed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a replayable C3X visible scene")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--default", type=Path, required=True, help="Default renderer-definition file")
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--custom", type=Path)
    parser.add_argument("--mod-root", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        scene = scene_contract.load_scene(args.scene)
        catalog = load_catalog(
            args.default,
            mod_root=args.mod_root,
            scenario_path=args.scenario,
            custom_path=args.custom,
            scenario_root=args.scenario_root,
        )
        loader = PackAssetLoader(catalog, mod_root=args.mod_root, scenario_root=args.scenario_root)
        viewport = scene["viewport"]
        with WholeViewportRenderer(viewport["width_px"], viewport["height_px"]) as renderer:
            frame = renderer.render(scene, catalog, loader)
            write_png(frame.canvas, args.output)
        print(json.dumps({"output": str(args.output), **frame.stats}, sort_keys=True, default=list))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
