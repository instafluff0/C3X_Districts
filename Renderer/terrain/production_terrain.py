#!/usr/bin/env python3
"""M6.1 continuous, source-independent terrain reference renderer.

The production contract is deliberately expressed in Civ III semantic names and
generic C3X logical asset IDs.  It does not read PCX files or any source-game
format.  The CPU rasterizer is an executable reference for the native renderer:
one shared height lattice is built for the complete visible scene, then clipped
to Civ III's authoritative map rectangle.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from Renderer.preview.render_iso import Canvas
from Renderer.preview.render_textured_patch import BACKGROUND, edge
from Renderer.scenes import scene_contract
from Renderer.standalone.whole_viewport_renderer import lighting_state, shade_sample


SCHEMA = "c3x.m6_terrain_selector_coverage.v0"
FIXTURE_SCHEMA = "c3x.m6_terrain_fixture.v0"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE = Path(__file__).with_name("m6_1_selector_coverage.json")
DEFAULT_ATLAS = ROOT / "inventory" / "vanilla_atlas_layouts.json"
DEFAULT_SEMANTICS = ROOT / "inventory" / "vanilla_conquests_biq_semantics.json"

TERRAIN_TYPES = (
    "desert", "plains", "grassland", "tundra", "flood_plain", "hills",
    "mountains", "forest", "jungle", "marsh", "volcano", "coast", "sea", "ocean",
)
DISPLAY_TO_ID = {name.replace("_", " "): name for name in TERRAIN_TYPES}
DISPLAY_TO_ID["mountain"] = "mountains"
LOGICAL_ASSETS = frozenset({f"terrain/{name}/base" for name in TERRAIN_TYPES} | {
    "transition/land/blend", "transition/water/shore", "feature/polar_ice",
})
WATER = frozenset({"coast", "sea", "ocean"})
HEIGHTS = {
    "desert": 0.00, "plains": 0.01, "grassland": 0.00, "tundra": 0.01,
    "flood_plain": -0.01, "hills": 0.28, "mountains": 0.76,
    "forest": 0.13, "jungle": 0.18, "marsh": 0.03, "volcano": 0.88,
    "coast": -0.05, "sea": -0.10, "ocean": -0.16,
}
COLORS = {
    "desert": (205, 177, 104), "plains": (170, 151, 83), "grassland": (76, 142, 72),
    "tundra": (132, 151, 139), "flood_plain": (151, 135, 72), "hills": (124, 127, 68),
    "mountains": (116, 106, 88), "forest": (49, 103, 52), "jungle": (40, 91, 48),
    "marsh": (72, 112, 82), "volcano": (108, 75, 59), "coast": (52, 151, 174),
    "sea": (36, 111, 158), "ocean": (24, 72, 130),
}
TARGET_CONTRACTS = frozenset({
    "terrain_transition_81", "polar_ice_32", "hills_16", "mountain_or_volcano_16",
    "tile_mask_16", "waterfalls_4", "marsh_18", "grassland_forest_and_jungle_50",
    "forest_30", "landmark_forest_32", "landmark_terrain_7",
})


@dataclass(frozen=True)
class TerrainFrame:
    canvas: Canvas
    depth_buffer: tuple[float, ...]
    owner_buffer: tuple[str | None, ...]
    stats: Mapping[str, Any]


class BoundedMaterialCache:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("material cache capacity must be positive")
        self.capacity = capacity
        self.values: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, logical_id: str, color: tuple[int, int, int]) -> tuple[int, int, int]:
        if logical_id in self.values:
            self.hits += 1
            self.values.move_to_end(logical_id)
            return self.values[logical_id]
        self.misses += 1
        self.values[logical_id] = color
        if len(self.values) > self.capacity:
            self.values.popitem(last=False)
            self.evictions += 1
        return color

    def clear(self) -> None:
        self.values.clear()

    def diagnostics(self) -> dict[str, int]:
        return {
            "capacity": self.capacity, "resident": len(self.values), "hits": self.hits,
            "misses": self.misses, "evictions": self.evictions,
        }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def terrain_id(name: str) -> str:
    normalized = name.casefold().replace("-", " ").replace("_", " ").strip()
    result = DISPLAY_TO_ID.get(normalized)
    if result is None:
        raise ValueError(f"unsupported terrain type {name!r}")
    return result


def validate_selector_coverage(
    coverage_path: Path = DEFAULT_COVERAGE,
    atlas_path: Path = DEFAULT_ATLAS,
    semantics_path: Path = DEFAULT_SEMANTICS,
) -> dict[str, Any]:
    coverage, atlas, semantics = _load(coverage_path), _load(atlas_path), _load(semantics_path)
    errors: list[str] = []
    if coverage.get("schema") != SCHEMA:
        errors.append("unsupported coverage schema")
    semantic_names = {terrain_id(item["name"]) for item in semantics.get("terrain_types", [])}
    if semantic_names != set(TERRAIN_TYPES):
        errors.append("logical terrain types do not exactly match the closed BIQ terrain ledger")
    atlas_contracts = {item["id"]: item for item in atlas.get("contracts", [])}
    records = {item.get("contract_id"): item for item in coverage.get("contracts", [])}
    if set(records) != set(TARGET_CONTRACTS):
        errors.append("coverage contract IDs do not exactly match the M6.1 selector boundary")
    dispositions: dict[str, int] = {"mapped": 0, "m7_vanilla_fallback": 0, "retained_civ3": 0}
    selector_cells = 0
    for contract_id in sorted(TARGET_CONTRACTS):
        expected = atlas_contracts.get(contract_id)
        record = records.get(contract_id)
        if expected is None or record is None:
            errors.append(f"missing contract {contract_id}")
            continue
        expected_names = {name.casefold() for name in expected["basenames"]}
        observed: list[str] = []
        for mapping in record.get("mappings", []):
            disposition = mapping.get("disposition")
            if disposition not in dispositions:
                errors.append(f"{contract_id} has unknown disposition {disposition!r}")
                continue
            dispositions[disposition] += len(mapping.get("basenames", []))
            observed.extend(name.casefold() for name in mapping.get("basenames", []))
            logical = mapping.get("logical_assets", [])
            if disposition == "mapped" and (not logical or not set(logical) <= LOGICAL_ASSETS):
                errors.append(f"{contract_id} mapped selector has unknown logical assets")
        if len(observed) != len(set(observed)) or set(observed) != expected_names:
            errors.append(f"{contract_id} basenames are missing, duplicated, or outside the M6.0 ledger")
        selector_cells += int(expected["authored_capacity"]) * len(expected_names)
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "terrain_types": len(semantic_names), "contracts": len(records),
        "atlas_basenames": sum(dispositions.values()), "selector_cells_accounted": selector_cells,
        "dispositions": dispositions,
    }


def load_fixture(path: Path, viewport_id: str, environment_id: str) -> dict[str, Any]:
    fixture = _load(path)
    if fixture.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("unsupported M6 terrain fixture schema")
    viewport = next(item for item in fixture["viewports"] if item["id"] == viewport_id)
    environment = next(item for item in fixture["environments"] if item["id"] == environment_id)
    origin_x, origin_y = fixture["grid_origin"]
    basis_x, basis_y = (64, 32), (-64, 32)
    scroll_x, scroll_y = viewport["scroll_px"]
    anchor_origin = viewport["origin_px"]
    landmark_set = {tuple(item) for item in fixture.get("landmarks", [])}
    polar_ice_set = {tuple(item) for item in fixture.get("polar_ice", [])}
    tiles = []
    for row, terrain_row in enumerate(fixture["terrain_rows"]):
        for column, name in enumerate(terrain_row):
            map_x, map_y = origin_x + column, origin_y + row
            item_id = scene_contract.instance_identifier("terrain", map_x, map_y, 0)
            anchor = {
                "x": anchor_origin[0] + column * basis_x[0] + row * basis_y[0] - scroll_x,
                "y": anchor_origin[1] + column * basis_x[1] + row * basis_y[1] - scroll_y,
            }
            tiles.append({
                "id": scene_contract.tile_identifier(map_x, map_y), "map_x": map_x, "map_y": map_y,
                "anchor_px": anchor,
                "terrain": {
                    "id": item_id,
                    "variant_seed": scene_contract.scene_variant_seed(fixture["world"]["seed"], item_id, map_x, map_y),
                    "resolver_input": {
                        "category": "terrain", "map_x": map_x, "map_y": map_y,
                        "terrain_type": name, "real_terrain_type": name,
                        "landmark": (column, row) in landmark_set, "pcx_file": "semantic-ledger",
                        "sheet_index": row, "sprite_index": column,
                        **({"adjacent_to": ["polar_ice"]} if (column, row) in polar_ice_set else {}),
                    },
                },
            })
    width, height = viewport["size_px"]
    map_height = height - viewport["hud_height_px"]
    scene = {
        "schema": scene_contract.SCHEMA, "scene_id": "pending", "profile_id": "default",
        "world": fixture["world"],
        "viewport": {"width_px": width, "height_px": height, "map_rect_px": {"x": 0, "y": 0, "width": width, "height": map_height}, "scroll_px": {"x": scroll_x, "y": scroll_y}},
        "projection": {
            "type": "civ3-isometric-pixel", "origin_tile": {"x": origin_x, "y": origin_y},
            "origin_px": {"x": anchor_origin[0] - scroll_x, "y": anchor_origin[1] - scroll_y},
            "tile_x_basis_px": {"x": 64, "y": 32}, "tile_y_basis_px": {"x": -64, "y": 32},
            "elevation_basis_px": {"x": 0, "y": -32},
        },
        "environment": {"id": "earthlike", "hour": environment["hour"], "season": environment["season"]},
        "tiles": tiles, "instances": [],
    }
    first_tile = tiles[0]
    first_x, first_y = first_tile["map_x"], first_tile["map_y"]
    for category in fixture.get("retained_categories", []):
        item_id = scene_contract.instance_identifier(category, first_x, first_y, 0)
        scene["instances"].append({
            "id": item_id, "category": category, "tile_id": first_tile["id"], "ordinal": 0,
            "anchor_px": dict(first_tile["anchor_px"]),
            "variant_seed": scene_contract.scene_variant_seed(fixture["world"]["seed"], item_id, first_x, first_y),
            "resolver_input": {"category": category, "map_x": first_x, "map_y": first_y},
        })
    scene["scene_id"] = scene_contract.scene_identifier(scene)
    return scene_contract.validate_scene(scene)


class ProductionTerrainRenderer:
    """Rasterize all visible terrain as one shared lattice with bounded state."""

    def __init__(self, *, material_cache_capacity: int = 16, max_visible_tiles: int = 4096) -> None:
        if max_visible_tiles < 1:
            raise ValueError("visible tile budget must be positive")
        self.cache = BoundedMaterialCache(material_cache_capacity)
        self.max_visible_tiles = max_visible_tiles
        self.generation = 1

    def reset(self) -> None:
        self.cache.clear()
        self.generation += 1

    @staticmethod
    def _canonical(scene: Mapping[str, Any], x: int, y: int) -> tuple[int, int]:
        world = scene["world"]
        if world["wrap_x"]:
            x %= world["width_tiles"]
        if world["wrap_y"]:
            y %= world["height_tiles"]
        return x, y

    def render(
        self,
        scene: Mapping[str, Any],
        *,
        available_assets: set[str] | None = None,
        corrupt_assets: set[str] | None = None,
    ) -> TerrainFrame:
        validated = scene_contract.validate_scene(scene)
        if len(validated["tiles"]) > self.max_visible_tiles:
            raise ValueError(f"visible terrain tile budget exceeded ({len(validated['tiles'])}>{self.max_visible_tiles})")
        available = LOGICAL_ASSETS if available_assets is None else frozenset(available_assets)
        corrupt = frozenset(corrupt_assets or ())
        width, height = validated["viewport"]["width_px"], validated["viewport"]["height_px"]
        canvas = Canvas(width, height, BACKGROUND)
        depth = [-math.inf] * (width * height)
        owners: list[str | None] = [None] * (width * height)
        tile_records: list[dict[str, Any]] = []
        lookup: dict[tuple[int, int], dict[str, Any]] = {}
        fallback: list[str] = []
        for tile in validated["tiles"]:
            metadata = tile["terrain"]["resolver_input"]
            kind = terrain_id(metadata["terrain_type"])
            logical = f"terrain/{kind}/base"
            adjacent = metadata.get("adjacent_to", [])
            if isinstance(adjacent, str):
                adjacent = [adjacent]
            record = {"tile": tile, "kind": kind, "logical": logical, "polar_ice": "polar_ice" in adjacent}
            tile_records.append(record)
            lookup[self._canonical(validated, tile["map_x"], tile["map_y"])] = record
        logical_dependencies: dict[str, list[str]] = {}
        for record in tile_records:
            tile, kind = record["tile"], record["kind"]
            x, y = tile["map_x"], tile["map_y"]
            dependencies = [record["logical"]]
            neighbors = [lookup.get(self._canonical(validated, x + dx, y + dy)) for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))]
            if any(neighbor is not None and ((neighbor["kind"] in WATER) != (kind in WATER)) for neighbor in neighbors):
                dependencies.append("transition/water/shore")
            elif any(neighbor is not None and neighbor["kind"] != kind for neighbor in neighbors):
                dependencies.append("transition/land/blend")
            if record["polar_ice"]:
                dependencies.append("feature/polar_ice")
            logical_dependencies[tile["terrain"]["id"]] = dependencies
            if any(logical not in available or logical in corrupt for logical in dependencies):
                fallback.append(tile["terrain"]["id"])

        def cell_height(x: int, y: int) -> float | None:
            item = lookup.get(self._canonical(validated, x, y))
            return None if item is None or item["tile"]["terrain"]["id"] in fallback else HEIGHTS[item["kind"]]

        vertex_heights: dict[tuple[int, int], float] = {}
        for record in tile_records:
            tile = record["tile"]
            if tile["terrain"]["id"] in fallback:
                continue
            x, y = tile["map_x"], tile["map_y"]
            for vx, vy in ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)):
                adjacent = [cell_height(vx - dx, vy - dy) for dx in (0, 1) for dy in (0, 1)]
                present = [value for value in adjacent if value is not None]
                vertex_heights[(vx, vy)] = sum(present) / len(present) if present else 0.0

        lighting = lighting_state({
            "sun_azimuth_degrees": 135, "midnight_ambient_color": [22, 30, 52],
            "night_exposure": 0.35, "noon_sun_color": [255, 244, 220], "seasonal_materials": True,
        }, validated["environment"]["hour"], validated["environment"]["season"])
        topology: dict[str, dict[str, int]] = {}
        triangles = 0
        clip = validated["viewport"]["map_rect_px"]
        clip_right, clip_bottom = clip["x"] + clip["width"] - 1, clip["y"] + clip["height"] - 1
        bx, by, bz = validated["projection"]["tile_x_basis_px"], validated["projection"]["tile_y_basis_px"], validated["projection"]["elevation_basis_px"]

        for record in tile_records:
            tile, kind, logical = record["tile"], record["kind"], record["logical"]
            instance_id = tile["terrain"]["id"]
            if instance_id in fallback:
                continue
            x, y, anchor = tile["map_x"], tile["map_y"], tile["anchor_px"]
            neighbor_bits = 0
            for bit, (dx, dy) in enumerate(((0, -1), (1, 0), (0, 1), (-1, 0))):
                neighbor = lookup.get(self._canonical(validated, x + dx, y + dy))
                if neighbor is not None and ((neighbor["kind"] in WATER) == (kind in WATER)):
                    neighbor_bits |= 1 << bit
            variant = tile["terrain"]["variant_seed"] % 8
            topology[instance_id] = {"adjacency_mask": neighbor_bits, "variant": variant}
            base = self.cache.get(logical, COLORS[kind])
            delta = int(variant) - 3
            if tile["terrain"]["resolver_input"].get("landmark"):
                delta += 13
            color = tuple(max(0, min(255, channel + delta)) for channel in base)
            if record["polar_ice"]:
                color = tuple((channel + 220) // 2 for channel in color)

            local = ((0, 0), (1, 0), (1, 1), (0, 1))
            points: list[tuple[float, float, float]] = []
            for dx, dy in local:
                z = vertex_heights[(x + dx, y + dy)]
                points.append((anchor["x"] + dx * bx["x"] + dy * by["x"] + z * bz["x"], anchor["y"] + dx * bx["y"] + dy * by["y"] + z * bz["y"], x + dx + y + dy + 2 * z))
            for tri in ((0, 1, 2), (0, 2, 3)):
                projected = [(points[index][0], points[index][1]) for index in tri]
                area = edge(projected[0], projected[1], projected[2])
                if abs(area) <= 1e-12:
                    continue
                min_x = max(clip["x"], 0, math.floor(min(point[0] for point in projected)))
                max_x = min(clip_right, width - 1, math.ceil(max(point[0] for point in projected)))
                min_y = max(clip["y"], 0, math.floor(min(point[1] for point in projected)))
                max_y = min(clip_bottom, height - 1, math.ceil(max(point[1] for point in projected)))
                if min_x > max_x or min_y > max_y:
                    continue
                triangles += 1
                normal = (0.0, 0.0, 1.0)
                shaded = shade_sample(color, normal, lighting)
                for py in range(min_y, max_y + 1):
                    for px in range(min_x, max_x + 1):
                        sample = (px + 0.5, py + 0.5)
                        weights = (edge(projected[1], projected[2], sample) / area, edge(projected[2], projected[0], sample) / area)
                        weights = (weights[0], weights[1], 1.0 - weights[0] - weights[1])
                        if min(weights) < -1e-9:
                            continue
                        zdepth = sum(weights[i] * points[tri[i]][2] for i in range(3))
                        index = py * width + px
                        if zdepth <= depth[index] + 1e-9:
                            continue
                        canvas.set_pixel(px, py, shaded)
                        depth[index], owners[index] = zdepth, instance_id

        rendered = [record["tile"]["terrain"]["id"] for record in tile_records if record["tile"]["terrain"]["id"] not in fallback]
        retained = [item["id"] for item in validated["instances"]]
        return TerrainFrame(canvas, tuple(depth), tuple(owners), {
            "generation": self.generation, "visible_tile_budget": self.max_visible_tiles,
            "visible_tiles": len(tile_records), "shared_vertices": len(vertex_heights),
            "triangles_submitted": triangles, "rendered_ids": rendered, "fallback_ids": fallback,
            "retained_civ3_instance_ids": retained, "retained_civ3_passes": 1,
            "topology": topology, "authoritative_anchors": {record["tile"]["terrain"]["id"]: record["tile"]["anchor_px"] for record in tile_records},
            "logical_dependencies": logical_dependencies,
            "cache": self.cache.diagnostics(), "lighting": lighting,
        })
