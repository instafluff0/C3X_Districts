#!/usr/bin/env python3
"""Validation, canonical serialization, and offline inspection for visible scenes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from Renderer.definitions.definition_parser import (
    BOOL_RULE_KEYS,
    INT_RULE_KEYS,
    RULE_CATEGORIES,
    STRING_RULE_KEYS,
)
from Renderer.definitions.rule_resolver import resolve_rule


SCHEMA = "c3x.visible_scene.v0"
INSPECTION_SCHEMA = "c3x.visible_scene_resolution.v0"
SEASONS = {"summer", "fall", "winter", "spring"}
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
FORBIDDEN_SOURCE_MARKERS = ("civ6", "artdef", ".blp", ".fgx", "steamapps")

METADATA_BOOL_KEYS = set(BOOL_RULE_KEYS)
METADATA_INT_KEYS = set(INT_RULE_KEYS) - {"priority"}
METADATA_STRING_KEYS = set(STRING_RULE_KEYS) - {"asset", "animation", "adjacent_to"}
METADATA_KEYS = (
    {"category", "adjacent_to"}
    | METADATA_BOOL_KEYS
    | METADATA_INT_KEYS
    | METADATA_STRING_KEYS
)


@dataclass(frozen=True)
class SceneDiagnostic:
    path: str
    message: str
    expected: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "message": self.message, "expected": self.expected}

    def __str__(self) -> str:
        suffix = f"; expected {self.expected}" if self.expected else ""
        return f"{self.path}: {self.message}{suffix}"


class SceneValidationError(ValueError):
    def __init__(self, diagnostics: Iterable[SceneDiagnostic]):
        self.diagnostics = list(diagnostics)
        super().__init__("\n".join(str(item) for item in self.diagnostics))


def tile_identifier(map_x: int, map_y: int) -> str:
    return f"tile:{map_x}:{map_y}"


def instance_identifier(category: str, map_x: int, map_y: int, ordinal: int) -> str:
    return f"{category}:{map_x}:{map_y}:{ordinal}"


def scene_identifier(scene: Mapping[str, Any]) -> str:
    world = scene["world"]
    viewport = scene["viewport"]
    environment = scene["environment"]
    scroll = viewport["scroll_px"]
    return (
        f"scene:{world['seed']}:{scroll['x']}:{scroll['y']}:"
        f"{viewport['width_px']}x{viewport['height_px']}:"
        f"{environment['hour']}:{environment['season']}"
    )


def scene_variant_seed(world_seed: int, item_id: str, map_x: int, map_y: int) -> int:
    material = f"{world_seed}\0{item_id}\0{map_x}\0{map_y}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_scene_text(text: str) -> dict[str, Any]:
    try:
        scene = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SceneValidationError([SceneDiagnostic("$", f"invalid JSON: {exc}")]) from exc
    validate_scene(scene)
    return scene


def load_scene(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SceneValidationError([SceneDiagnostic("$", f"cannot read scene: {exc}")]) from exc
    return parse_scene_text(text)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _object(
    value: Any,
    path: str,
    required: set[str],
    optional: set[str],
    diagnostics: list[SceneDiagnostic],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        diagnostics.append(SceneDiagnostic(path, "value is not an object", "object"))
        return None
    for key in sorted(required - set(value)):
        diagnostics.append(SceneDiagnostic(f"{path}.{key}", "missing required field"))
    for key in sorted(set(value) - required - optional):
        diagnostics.append(SceneDiagnostic(f"{path}.{key}", "unknown field"))
    return value


def _integer(
    value: Any,
    path: str,
    diagnostics: list[SceneDiagnostic],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> bool:
    if not _is_int(value):
        diagnostics.append(SceneDiagnostic(path, "value is not an integer", "integer"))
        return False
    if minimum is not None and value < minimum:
        diagnostics.append(SceneDiagnostic(path, "value is below its minimum", f">= {minimum}"))
        return False
    if maximum is not None and value > maximum:
        diagnostics.append(SceneDiagnostic(path, "value is above its maximum", f"<= {maximum}"))
        return False
    return True


def _string(value: Any, path: str, diagnostics: list[SceneDiagnostic], *, identifier: bool = False) -> bool:
    if not isinstance(value, str) or not value:
        diagnostics.append(SceneDiagnostic(path, "value is not a nonempty string", "nonempty string"))
        return False
    if identifier and not ID_RE.fullmatch(value):
        diagnostics.append(SceneDiagnostic(path, "value is not a stable identifier", "letters, digits, dot, underscore, colon, or hyphen"))
        return False
    return True


def _point(value: Any, path: str, diagnostics: list[SceneDiagnostic]) -> None:
    point = _object(value, path, {"x", "y"}, set(), diagnostics)
    if point is None:
        return
    if "x" in point:
        _integer(point["x"], f"{path}.x", diagnostics)
    if "y" in point:
        _integer(point["y"], f"{path}.y", diagnostics)


def _rect(value: Any, path: str, diagnostics: list[SceneDiagnostic]) -> None:
    rect = _object(value, path, {"x", "y", "width", "height"}, set(), diagnostics)
    if rect is None:
        return
    for key in ("x", "y"):
        if key in rect:
            _integer(rect[key], f"{path}.{key}", diagnostics)
    for key in ("width", "height"):
        if key in rect:
            _integer(rect[key], f"{path}.{key}", diagnostics, minimum=1)


def _metadata(value: Any, path: str, diagnostics: list[SceneDiagnostic]) -> Mapping[str, Any] | None:
    metadata = _object(value, path, {"category", "map_x", "map_y"}, METADATA_KEYS, diagnostics)
    if metadata is None:
        return None
    category = metadata.get("category")
    if _string(category, f"{path}.category", diagnostics) and category not in RULE_CATEGORIES:
        diagnostics.append(SceneDiagnostic(f"{path}.category", "unknown render category", ", ".join(sorted(RULE_CATEGORIES))))
    for key in sorted(METADATA_BOOL_KEYS & set(metadata)):
        if not isinstance(metadata[key], bool):
            diagnostics.append(SceneDiagnostic(f"{path}.{key}", "value is not boolean", "true or false"))
    for key in sorted(METADATA_INT_KEYS & set(metadata)):
        _integer(metadata[key], f"{path}.{key}", diagnostics)
    for key in sorted(METADATA_STRING_KEYS & set(metadata)):
        _string(metadata[key], f"{path}.{key}", diagnostics)
    if "adjacent_to" in metadata:
        adjacent = metadata["adjacent_to"]
        if isinstance(adjacent, str):
            _string(adjacent, f"{path}.adjacent_to", diagnostics)
        elif isinstance(adjacent, list) and adjacent:
            for index, item in enumerate(adjacent):
                _string(item, f"{path}.adjacent_to[{index}]", diagnostics)
        else:
            diagnostics.append(SceneDiagnostic(f"{path}.adjacent_to", "value is not an adjacency string or nonempty list", "string or nonempty string array"))
    return metadata


def _scan_for_source_markers(value: Any, path: str, diagnostics: list[SceneDiagnostic]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = key.casefold()
            if any(marker in lowered for marker in ("pointer", "address", "hwnd", "handle")):
                diagnostics.append(SceneDiagnostic(f"{path}.{key}", "process-specific field is forbidden"))
            _scan_for_source_markers(child, f"{path}.{key}", diagnostics)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_for_source_markers(child, f"{path}[{index}]", diagnostics)
    elif isinstance(value, str):
        lowered = value.casefold()
        marker = next((item for item in FORBIDDEN_SOURCE_MARKERS if item in lowered), None)
        if marker:
            diagnostics.append(SceneDiagnostic(path, f"source-specific marker {marker!r} is forbidden"))


def validate_scene(scene: Any) -> dict[str, Any]:
    diagnostics: list[SceneDiagnostic] = []
    root = _object(
        scene,
        "$",
        {"schema", "scene_id", "profile_id", "world", "viewport", "projection", "environment", "tiles", "instances"},
        set(),
        diagnostics,
    )
    if root is None:
        raise SceneValidationError(diagnostics)

    if root.get("schema") != SCHEMA:
        diagnostics.append(SceneDiagnostic("$.schema", "unsupported scene schema", SCHEMA))
    _string(root.get("scene_id"), "$.scene_id", diagnostics, identifier=True)
    _string(root.get("profile_id"), "$.profile_id", diagnostics, identifier=True)

    world = _object(root.get("world"), "$.world", {"seed", "width_tiles", "height_tiles", "wrap_x", "wrap_y"}, set(), diagnostics)
    if world is not None:
        if "seed" in world:
            _integer(world["seed"], "$.world.seed", diagnostics)
        for key in ("width_tiles", "height_tiles"):
            if key in world:
                _integer(world[key], f"$.world.{key}", diagnostics, minimum=1)
        for key in ("wrap_x", "wrap_y"):
            if key in world and not isinstance(world[key], bool):
                diagnostics.append(SceneDiagnostic(f"$.world.{key}", "value is not boolean", "true or false"))

    viewport = _object(root.get("viewport"), "$.viewport", {"width_px", "height_px", "map_rect_px", "scroll_px"}, set(), diagnostics)
    if viewport is not None:
        for key in ("width_px", "height_px"):
            if key in viewport:
                _integer(viewport[key], f"$.viewport.{key}", diagnostics, minimum=1)
        if "map_rect_px" in viewport:
            _rect(viewport["map_rect_px"], "$.viewport.map_rect_px", diagnostics)
        if "scroll_px" in viewport:
            _point(viewport["scroll_px"], "$.viewport.scroll_px", diagnostics)
        rect = viewport.get("map_rect_px")
        if isinstance(rect, Mapping) and all(_is_int(rect.get(key)) for key in ("x", "y", "width", "height")):
            if rect["x"] < 0 or rect["y"] < 0 or rect["x"] + rect["width"] > viewport.get("width_px", -1) or rect["y"] + rect["height"] > viewport.get("height_px", -1):
                diagnostics.append(SceneDiagnostic("$.viewport.map_rect_px", "map rectangle is outside the viewport"))

    projection = _object(
        root.get("projection"),
        "$.projection",
        {"type", "origin_tile", "origin_px", "tile_x_basis_px", "tile_y_basis_px", "elevation_basis_px"},
        set(),
        diagnostics,
    )
    if projection is not None:
        if projection.get("type") != "civ3-isometric-pixel":
            diagnostics.append(SceneDiagnostic("$.projection.type", "unsupported projection", "civ3-isometric-pixel"))
        for key in ("origin_tile", "origin_px", "tile_x_basis_px", "tile_y_basis_px", "elevation_basis_px"):
            if key in projection:
                _point(projection[key], f"$.projection.{key}", diagnostics)
        x_basis = projection.get("tile_x_basis_px")
        y_basis = projection.get("tile_y_basis_px")
        z_basis = projection.get("elevation_basis_px")
        if isinstance(x_basis, Mapping) and _is_int(x_basis.get("x")) and _is_int(x_basis.get("y")) and not (x_basis["x"] > 0 and x_basis["y"] > 0):
            diagnostics.append(SceneDiagnostic("$.projection.tile_x_basis_px", "basis does not map tile X down/right", "x > 0 and y > 0"))
        if isinstance(y_basis, Mapping) and _is_int(y_basis.get("x")) and _is_int(y_basis.get("y")) and not (y_basis["x"] < 0 and y_basis["y"] > 0):
            diagnostics.append(SceneDiagnostic("$.projection.tile_y_basis_px", "basis does not map tile Y down/left", "x < 0 and y > 0"))
        if isinstance(z_basis, Mapping) and _is_int(z_basis.get("x")) and _is_int(z_basis.get("y")) and not (z_basis["x"] == 0 and z_basis["y"] < 0):
            diagnostics.append(SceneDiagnostic("$.projection.elevation_basis_px", "basis does not map elevation upward", "x = 0 and y < 0"))

    environment = _object(root.get("environment"), "$.environment", {"id", "hour", "season"}, set(), diagnostics)
    if environment is not None:
        _string(environment.get("id"), "$.environment.id", diagnostics, identifier=True)
        if "hour" in environment:
            _integer(environment["hour"], "$.environment.hour", diagnostics, minimum=0, maximum=23)
        season = environment.get("season")
        if _string(season, "$.environment.season", diagnostics) and season not in SEASONS:
            diagnostics.append(SceneDiagnostic("$.environment.season", "unknown season", ", ".join(sorted(SEASONS))))

    tiles = root.get("tiles")
    tile_ids: set[str] = set()
    tile_coordinates: dict[str, tuple[int, int]] = {}
    if not isinstance(tiles, list) or not tiles:
        diagnostics.append(SceneDiagnostic("$.tiles", "value is not a nonempty array", "nonempty tile array"))
        tiles = []
    for index, value in enumerate(tiles):
        path = f"$.tiles[{index}]"
        tile = _object(value, path, {"id", "map_x", "map_y", "anchor_px", "terrain"}, set(), diagnostics)
        if tile is None:
            continue
        identifier = tile.get("id")
        x = tile.get("map_x")
        y = tile.get("map_y")
        _string(identifier, f"{path}.id", diagnostics, identifier=True)
        x_ok = _integer(x, f"{path}.map_x", diagnostics, minimum=0)
        y_ok = _integer(y, f"{path}.map_y", diagnostics, minimum=0)
        if isinstance(identifier, str):
            if identifier in tile_ids:
                diagnostics.append(SceneDiagnostic(f"{path}.id", "duplicate tile ID"))
            tile_ids.add(identifier)
        if isinstance(identifier, str) and x_ok and y_ok:
            if identifier != tile_identifier(x, y):
                diagnostics.append(SceneDiagnostic(f"{path}.id", "tile ID is not deterministic", tile_identifier(x, y)))
            tile_coordinates[identifier] = (x, y)
            if isinstance(world, Mapping):
                if _is_int(world.get("width_tiles")) and x >= world["width_tiles"]:
                    diagnostics.append(SceneDiagnostic(f"{path}.map_x", "coordinate is outside world width"))
                if _is_int(world.get("height_tiles")) and y >= world["height_tiles"]:
                    diagnostics.append(SceneDiagnostic(f"{path}.map_y", "coordinate is outside world height"))
        if "anchor_px" in tile:
            _point(tile["anchor_px"], f"{path}.anchor_px", diagnostics)
        if x_ok and y_ok and isinstance(projection, Mapping) and isinstance(tile.get("anchor_px"), Mapping):
            origin_tile = projection.get("origin_tile")
            origin_px = projection.get("origin_px")
            x_basis = projection.get("tile_x_basis_px")
            y_basis = projection.get("tile_y_basis_px")
            parts = (origin_tile, origin_px, x_basis, y_basis)
            if all(isinstance(part, Mapping) and _is_int(part.get("x")) and _is_int(part.get("y")) for part in parts):
                delta_x = x - origin_tile["x"]
                delta_y = y - origin_tile["y"]
                expected_anchor = {
                    "x": origin_px["x"] + delta_x * x_basis["x"] + delta_y * y_basis["x"],
                    "y": origin_px["y"] + delta_x * x_basis["y"] + delta_y * y_basis["y"],
                }
                if tile["anchor_px"] != expected_anchor:
                    diagnostics.append(SceneDiagnostic(f"{path}.anchor_px", "anchor does not match the captured pixel basis", str(expected_anchor)))
        terrain = _object(tile.get("terrain"), f"{path}.terrain", {"id", "variant_seed", "resolver_input"}, set(), diagnostics)
        if terrain is not None:
            terrain_id = terrain.get("id")
            _string(terrain_id, f"{path}.terrain.id", diagnostics, identifier=True)
            _integer(terrain.get("variant_seed"), f"{path}.terrain.variant_seed", diagnostics, minimum=0, maximum=(1 << 64) - 1)
            metadata = _metadata(terrain.get("resolver_input"), f"{path}.terrain.resolver_input", diagnostics)
            if metadata is not None and x_ok and y_ok:
                if metadata.get("category") != "terrain":
                    diagnostics.append(SceneDiagnostic(f"{path}.terrain.resolver_input.category", "tile terrain category must be terrain"))
                if metadata.get("map_x") != x or metadata.get("map_y") != y:
                    diagnostics.append(SceneDiagnostic(f"{path}.terrain.resolver_input", "resolver coordinates do not match the tile"))
            if isinstance(terrain_id, str) and x_ok and y_ok:
                expected_id = instance_identifier("terrain", x, y, 0)
                if terrain_id != expected_id:
                    diagnostics.append(SceneDiagnostic(f"{path}.terrain.id", "terrain ID is not deterministic", expected_id))
                if isinstance(world, Mapping) and _is_int(world.get("seed")) and _is_int(terrain.get("variant_seed")):
                    expected_seed = scene_variant_seed(world["seed"], terrain_id, x, y)
                    if terrain["variant_seed"] != expected_seed:
                        diagnostics.append(SceneDiagnostic(f"{path}.terrain.variant_seed", "variant seed is not deterministic", str(expected_seed)))

    instances = root.get("instances")
    instance_ids: set[str] = set()
    ordinals: dict[tuple[str, int, int], list[int]] = {}
    if not isinstance(instances, list):
        diagnostics.append(SceneDiagnostic("$.instances", "value is not an array", "object-instance array"))
        instances = []
    for index, value in enumerate(instances):
        path = f"$.instances[{index}]"
        instance = _object(value, path, {"id", "category", "tile_id", "ordinal", "anchor_px", "variant_seed", "resolver_input"}, set(), diagnostics)
        if instance is None:
            continue
        identifier = instance.get("id")
        category = instance.get("category")
        tile_id = instance.get("tile_id")
        ordinal = instance.get("ordinal")
        _string(identifier, f"{path}.id", diagnostics, identifier=True)
        if _string(category, f"{path}.category", diagnostics) and category not in RULE_CATEGORIES - {"terrain"}:
            diagnostics.append(SceneDiagnostic(f"{path}.category", "invalid object-instance category", "non-terrain render category"))
        _string(tile_id, f"{path}.tile_id", diagnostics, identifier=True)
        ordinal_ok = _integer(ordinal, f"{path}.ordinal", diagnostics, minimum=0)
        if "anchor_px" in instance:
            _point(instance["anchor_px"], f"{path}.anchor_px", diagnostics)
        _integer(instance.get("variant_seed"), f"{path}.variant_seed", diagnostics, minimum=0, maximum=(1 << 64) - 1)
        metadata = _metadata(instance.get("resolver_input"), f"{path}.resolver_input", diagnostics)
        if isinstance(identifier, str):
            if identifier in instance_ids:
                diagnostics.append(SceneDiagnostic(f"{path}.id", "duplicate instance ID"))
            instance_ids.add(identifier)
        coordinate = tile_coordinates.get(tile_id) if isinstance(tile_id, str) else None
        if isinstance(tile_id, str) and tile_id not in tile_ids:
            diagnostics.append(SceneDiagnostic(f"{path}.tile_id", "tile reference does not resolve"))
        if coordinate and isinstance(category, str) and ordinal_ok:
            x, y = coordinate
            expected_id = instance_identifier(category, x, y, ordinal)
            if identifier != expected_id:
                diagnostics.append(SceneDiagnostic(f"{path}.id", "instance ID is not deterministic", expected_id))
            ordinals.setdefault((category, x, y), []).append(ordinal)
            if metadata is not None:
                if metadata.get("category") != category:
                    diagnostics.append(SceneDiagnostic(f"{path}.resolver_input.category", "resolver category does not match instance category"))
                if metadata.get("map_x") != x or metadata.get("map_y") != y:
                    diagnostics.append(SceneDiagnostic(f"{path}.resolver_input", "resolver coordinates do not match the referenced tile"))
            if isinstance(world, Mapping) and _is_int(world.get("seed")) and _is_int(instance.get("variant_seed")) and isinstance(identifier, str):
                expected_seed = scene_variant_seed(world["seed"], identifier, x, y)
                if instance["variant_seed"] != expected_seed:
                    diagnostics.append(SceneDiagnostic(f"{path}.variant_seed", "variant seed is not deterministic", str(expected_seed)))
    for (category, x, y), values in ordinals.items():
        expected = list(range(len(values)))
        if sorted(values) != expected:
            diagnostics.append(SceneDiagnostic("$.instances", f"{category} ordinals at ({x}, {y}) are not contiguous", str(expected)))

    if isinstance(root.get("scene_id"), str) and isinstance(world, Mapping) and isinstance(viewport, Mapping) and isinstance(environment, Mapping):
        try:
            expected_scene_id = scene_identifier(root)
        except KeyError:
            expected_scene_id = None
        if expected_scene_id and root["scene_id"] != expected_scene_id:
            diagnostics.append(SceneDiagnostic("$.scene_id", "scene ID is not deterministic", expected_scene_id))

    _scan_for_source_markers(root, "$", diagnostics)
    if diagnostics:
        raise SceneValidationError(diagnostics)
    return root  # type: ignore[return-value]


def _resolvable_items(scene: Mapping[str, Any]):
    for index, tile in enumerate(scene["tiles"]):
        yield f"tiles[{index}].terrain", tile["terrain"], tile["anchor_px"]
    for index, instance in enumerate(scene["instances"]):
        yield f"instances[{index}]", instance, instance["anchor_px"]


def inspect_scene(
    scene: Mapping[str, Any],
    catalog: Mapping[str, Any],
    *,
    enabled: bool = True,
    available_assets: Iterable[str] | None = None,
) -> dict[str, Any]:
    validate_scene(scene)
    environment = scene["environment"]
    items = []
    for source, item, anchor in _resolvable_items(scene):
        metadata = dict(item["resolver_input"])
        metadata["hour"] = environment["hour"]
        metadata["season"] = environment["season"]
        resolution = resolve_rule(
            catalog,
            metadata,
            profile_id=scene["profile_id"],
            enabled=enabled,
            world_seed=scene["world"]["seed"],
            available_assets=available_assets,
        )
        items.append(
            {
                "source": source,
                "instance_id": item["id"],
                "anchor_px": anchor,
                "scene_variant_seed": item["variant_seed"],
                "resolver_input": metadata,
                "resolution": resolution,
            }
        )
    return {
        "schema": INSPECTION_SCHEMA,
        "scene_schema": scene["schema"],
        "scene_id": scene["scene_id"],
        "items": items,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate, canonicalize, or inspect a C3X visible scene")
    parser.add_argument("scene", type=Path)
    parser.add_argument("--catalog", type=Path, help="Merged renderer-definition catalog JSON for offline inspection")
    parser.add_argument("--config-off", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        scene = load_scene(args.scene)
        result: Any = scene
        if args.catalog:
            catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
            result = inspect_scene(scene, catalog, enabled=not args.config_off)
        output = canonical_json(result)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
    except (OSError, ValueError, TypeError, KeyError, SceneValidationError, json.JSONDecodeError) as exc:
        print(f"scene operation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
