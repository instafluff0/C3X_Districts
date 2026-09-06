#!/usr/bin/env python3
"""Parser and layered merge for C3X custom-rendering definitions v0."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable


SECTION_TYPES = ("Profile", "Pack", "Asset", "Rule", "Environment")
LAYER_ORDER = ("default", "scenario", "custom")
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

PROFILE_CATEGORIES = (
    "terrain",
    "features",
    "roads",
    "rivers",
    "improvements",
    "resources",
    "cities",
    "units",
    "effects",
)
OWNERSHIP = {"civ3", "replace", "augment", "capture-only"}
RULE_CATEGORIES = {
    "terrain", "feature", "road", "river", "improvement",
    "resource", "city", "unit", "effect",
}
BOOL_RULE_KEYS = {
    "landmark", "has_forest", "has_jungle", "has_marsh", "has_pollution",
    "has_crater", "has_walls", "is_capital", "fortified",
}
INT_RULE_KEYS = {
    "priority", "map_x", "map_y", "sheet_index", "sprite_index", "pcx_index",
    "river_mask", "road_mask", "railroad_mask", "neighbor_mask", "resource_id",
    "city_style_index", "unit_id", "visibility_mask", "fog_status",
    "territory_owner_id", "city_id", "city_owner_id", "city_population",
    "tile_building_id", "unit_owner_id", "unit_state", "unit_damage", "unit_direction",
}
STRING_RULE_KEYS = {
    "terrain_type", "real_terrain_type", "pcx_file", "improvement",
    "terrain_building", "coast_shape", "resource_name", "resource_class",
    "culture_group", "era", "city_size", "unit_type", "unit_class",
    "direction", "action", "hit_point_band", "owner", "civilization",
    "adjacent_to", "asset", "animation",
}


@dataclass(frozen=True)
class Diagnostic:
    file: str
    line: int
    section_type: str | None
    section_id: str | None
    key: str | None
    message: str
    expected: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "section_type": self.section_type,
            "section_id": self.section_id,
            "key": self.key,
            "message": self.message,
            "expected": self.expected,
        }

    def __str__(self) -> str:
        location = f"{self.file}:{self.line}"
        section = f" #{self.section_type}" if self.section_type else ""
        identity = f" id={self.section_id}" if self.section_id else ""
        key = f" key={self.key}" if self.key else ""
        expected = f"; expected {self.expected}" if self.expected else ""
        return f"{location}:{section}{identity}{key}: {self.message}{expected}"


class DefinitionError(ValueError):
    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = diagnostics
        super().__init__("\n".join(str(item) for item in diagnostics))


@dataclass(frozen=True)
class Definition:
    section_type: str
    identifier: str
    values: dict[str, Any]
    file: str
    line: int
    layer: str
    layer_index: int
    declaration_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "values": {key: value for key, value in self.values.items() if key != "id"},
            "source": {
                "file": self.file,
                "line": self.line,
                "layer": self.layer,
                "layer_index": self.layer_index,
                "declaration_index": self.declaration_index,
            },
        }


def parse_bool(raw: str) -> bool:
    lowered = raw.lower()
    if lowered not in {"true", "false"}:
        raise ValueError("true or false")
    return lowered == "true"


def parse_int(raw: str) -> int:
    try:
        return int(raw, 10)
    except ValueError as exc:
        raise ValueError("a base-10 integer") from exc


def parse_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("a finite number") from exc
    if value != value or value in {float("inf"), float("-inf")}:
        raise ValueError("a finite number")
    return value


def parse_string(raw: str) -> str:
    if not raw:
        raise ValueError("a nonempty string")
    return raw


def enum_parser(values: set[str]) -> Callable[[str], str]:
    def parse(raw: str) -> str:
        lowered = raw.lower()
        if lowered not in values:
            raise ValueError("one of " + ", ".join(sorted(values)))
        return lowered

    return parse


def bounded_int(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        value = parse_int(raw)
        if not minimum <= value <= maximum:
            raise ValueError(f"an integer from {minimum} through {maximum}")
        return value

    return parse


def parse_color(raw: str) -> list[int]:
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("three comma-separated integers from 0 through 255")
    values = [parse_int(part) for part in parts]
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("three comma-separated integers from 0 through 255")
    return values


def parse_seasons(raw: str) -> list[str]:
    aliases = {"autumn": "fall"}
    values = [aliases.get(item.strip().lower(), item.strip().lower()) for item in raw.split(",")]
    if not values or any(value not in {"summer", "fall", "winter", "spring"} for value in values):
        raise ValueError("a comma-separated subset of summer, fall/autumn, winter, spring")
    if len(set(values)) != len(values):
        raise ValueError("seasons without duplicates")
    return values


def parse_hours(raw: str) -> list[int | dict[str, int]]:
    values: list[int | dict[str, int]] = []
    for part in (item.strip() for item in raw.split(",")):
        if not part:
            raise ValueError("comma-separated hours or inclusive ranges in 0..23")
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise ValueError("comma-separated hours or inclusive ranges in 0..23")
            start, end = (bounded_int(0, 23)(piece.strip()) for piece in pieces)
            values.append({"start": start, "end": end})
        else:
            values.append(bounded_int(0, 23)(part))
    return values


def path_is_absolute(value: str) -> bool:
    return (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
    )


def normalized_relative(raw: str) -> Path:
    converted = raw.replace("\\", os.sep).replace("/", os.sep)
    return Path(converted)


def within_root(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def resolve_pack_path(
    raw: str,
    declaring_file: Path,
    mod_root: Path,
    scenario_root: Path | None,
    allow_file: bool = False,
) -> dict[str, str]:
    if ":" not in raw:
        raise ValueError("a path prefixed with mod:, scenario:, or file:")
    prefix, remainder = raw.split(":", 1)
    prefix = prefix.lower()
    if prefix not in {"mod", "scenario", "file"} or not remainder:
        raise ValueError("a path prefixed with mod:, scenario:, or file:")
    relative = normalized_relative(remainder)
    if prefix in {"mod", "scenario"}:
        root = mod_root if prefix == "mod" else scenario_root
        if root is None:
            raise ValueError(f"{prefix}: path requires a configured {prefix} root")
        if path_is_absolute(remainder):
            raise ValueError(f"{prefix}: path must be relative")
        candidate = root / relative
        if not within_root(root, candidate):
            raise ValueError(f"{prefix}: path escapes its configured root")
        normalized = candidate.resolve().relative_to(root.resolve()).as_posix()
        return {"raw": raw, "root": prefix, "path": normalized}
    if not allow_file:
        raise ValueError("file: paths are disabled unless local-development mode is enabled")
    candidate = relative if path_is_absolute(remainder) else declaring_file.parent / relative
    return {"raw": raw, "root": "file", "path": str(candidate.resolve())}


def resolve_pack_asset_path(pack_root: Path, raw: str) -> str:
    if path_is_absolute(raw):
        raise ValueError("pack asset path must be relative")
    candidate = pack_root / normalized_relative(raw)
    if not within_root(pack_root, candidate):
        raise ValueError("pack asset path escapes its pack root")
    return candidate.resolve().relative_to(pack_root.resolve()).as_posix()


def section_schema(section_type: str) -> dict[str, Callable[[str], Any]]:
    common: dict[str, Callable[[str], Any]] = {"id": parse_string, "disabled": parse_bool}
    if section_type == "Profile":
        return {
            **common,
            **{key: enum_parser(OWNERSHIP) for key in PROFILE_CATEGORIES},
            "missing_asset": enum_parser({"fallback", "warn", "error"}),
            "world_seed": parse_string,
            "environment": parse_string,
        }
    if section_type == "Pack":
        return {**common, "path": parse_string}
    if section_type == "Asset":
        return {
            **common,
            "pack": parse_string,
            "asset": parse_string,
            "anchor_x": parse_float,
            "anchor_y": parse_float,
            "scale": parse_float,
            "offset_x_px": parse_int,
            "offset_y_px": parse_int,
            "fit_width_px": parse_int,
            "fit_height_px": parse_int,
            "casts_shadow": parse_bool,
            "receives_shadow": parse_bool,
        }
    if section_type == "Rule":
        schema = {
            **common,
            "category": enum_parser(RULE_CATEGORIES),
            "variant_selection": enum_parser({"coordinate-hash"}),
            "replacement": enum_parser(OWNERSHIP - {"civ3", "capture-only"}),
            "show_in_day_night_hours": parse_hours,
            "show_in_seasons": parse_seasons,
        }
        schema.update({key: parse_bool for key in BOOL_RULE_KEYS})
        schema.update({key: parse_int for key in INT_RULE_KEYS})
        schema.update({key: parse_string for key in STRING_RULE_KEYS})
        return schema
    if section_type == "Environment":
        return {
            **common,
            "day_night_source": enum_parser({"c3x"}),
            "season_source": enum_parser({"c3x"}),
            "sunrise_hour": bounded_int(0, 23),
            "sunset_hour": bounded_int(0, 23),
            "sun_azimuth_degrees": parse_float,
            "noon_sun_color": parse_color,
            "midnight_ambient_color": parse_color,
            "night_exposure": parse_float,
            "shadow_quality": enum_parser({"low", "medium", "high"}),
            "seasonal_materials": parse_bool,
        }
    raise ValueError(f"Unknown section type {section_type}")


REQUIRED_KEYS = {
    "Profile": {"id", "missing_asset"},
    "Pack": {"id", "path"},
    "Asset": {"id", "pack", "asset"},
    "Rule": {"id", "category", "asset"},
    "Environment": {"id"},
}


def parse_definitions(
    text: str,
    source_file: Path,
    layer: str,
    mod_root: Path,
    scenario_root: Path | None = None,
    allow_file: bool = False,
) -> list[Definition]:
    if layer not in LAYER_ORDER:
        raise ValueError(f"Unknown definition layer {layer}")
    source_label = str(source_file)
    diagnostics: list[Diagnostic] = []
    definitions: list[Definition] = []
    current_type: str | None = None
    current_line = 0
    current_values: dict[str, Any] = {}
    current_key_lines: dict[str, int] = {}
    declaration_index = 0

    def diagnostic(line: int, key: str | None, message: str, expected: str | None = None) -> None:
        diagnostics.append(
            Diagnostic(
                source_label,
                line,
                current_type,
                current_values.get("id"),
                key,
                message,
                expected,
            )
        )

    def finish_section() -> None:
        nonlocal declaration_index
        if current_type is None:
            return
        identifier = current_values.get("id")
        if not isinstance(identifier, str) or not ID_RE.fullmatch(identifier):
            diagnostic(current_key_lines.get("id", current_line), "id", "missing or invalid stable ID", "letters, digits, dot, underscore, or hyphen")
            return
        if not current_values.get("disabled", False):
            missing = sorted(REQUIRED_KEYS[current_type] - set(current_values))
            for key in missing:
                diagnostic(current_line, key, "missing required key")
            if missing:
                return
        values = dict(current_values)
        if current_type == "Pack" and "path" in values:
            try:
                values["path"] = resolve_pack_path(
                    values["path"], source_file, mod_root, scenario_root, allow_file
                )
            except ValueError as exc:
                diagnostic(current_key_lines["path"], "path", str(exc))
                return
        definitions.append(
            Definition(
                current_type,
                identifier,
                values,
                source_label,
                current_line,
                layer,
                LAYER_ORDER.index(layer),
                declaration_index,
            )
        )
        declaration_index += 1

    for line_number, raw_line in enumerate(text.splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped or (stripped.startswith("[") and stripped.endswith("]")):
            continue
        if stripped.startswith("#"):
            finish_section()
            current_values = {}
            current_key_lines = {}
            candidate = stripped[1:]
            if candidate not in SECTION_TYPES:
                current_type = None
                diagnostics.append(
                    Diagnostic(source_label, line_number, candidate or None, None, None, "unknown section type", ", ".join(f"#{item}" for item in SECTION_TYPES))
                )
            else:
                current_type = candidate
                current_line = line_number
            continue
        if current_type is None:
            diagnostics.append(
                Diagnostic(source_label, line_number, None, None, None, "key/value line appears outside a valid section", "#Section followed by key = value")
            )
            continue
        if "=" not in stripped:
            diagnostic(line_number, None, "invalid assignment", "key = value")
            continue
        key, raw_value = (part.strip() for part in stripped.split("=", 1))
        schema = section_schema(current_type)
        if key not in schema:
            diagnostic(line_number, key or None, "unknown key", ", ".join(sorted(schema)))
            continue
        if key in current_values:
            diagnostic(line_number, key, "duplicate key in section")
            continue
        try:
            current_values[key] = schema[key](raw_value)
            current_key_lines[key] = line_number
        except ValueError as exc:
            diagnostic(line_number, key, "invalid value", str(exc))
    finish_section()

    duplicates: dict[tuple[str, str], Definition] = {}
    for definition in definitions:
        key = (definition.section_type, definition.identifier)
        if key in duplicates:
            diagnostics.append(
                Diagnostic(
                    definition.file,
                    definition.line,
                    definition.section_type,
                    definition.identifier,
                    "id",
                    "duplicate section type and ID in one layer",
                )
            )
        else:
            duplicates[key] = definition
    if diagnostics:
        raise DefinitionError(diagnostics)
    return definitions


def parse_definition_file(
    path: Path,
    layer: str,
    mod_root: Path,
    scenario_root: Path | None = None,
    allow_file: bool = False,
) -> list[Definition]:
    return parse_definitions(
        path.read_text(encoding="utf-8"), path, layer, mod_root, scenario_root, allow_file
    )


def merge_layers(layer_definitions: Iterable[tuple[str, list[Definition]]]) -> dict[str, Any]:
    provided = list(layer_definitions)
    indexes = [LAYER_ORDER.index(layer) for layer, _definitions in provided]
    if indexes != sorted(indexes) or len(indexes) != len(set(indexes)):
        raise ValueError("Definition layers must be unique and ordered default, scenario, custom")

    active: dict[tuple[str, str], Definition] = {}
    disabled = []
    for layer, definitions in provided:
        if any(definition.layer != layer for definition in definitions):
            raise ValueError(f"Definition layer metadata does not match {layer}")
        for definition in definitions:
            key = (definition.section_type, definition.identifier)
            if definition.values.get("disabled", False):
                active.pop(key, None)
                disabled.append(definition.as_dict() | {"section_type": definition.section_type})
            else:
                active.pop(key, None)
                active[key] = definition

    by_type: dict[str, list[dict[str, Any]]] = {
        section_type.lower() + "s": [] for section_type in SECTION_TYPES
    }
    for definition in active.values():
        by_type[definition.section_type.lower() + "s"].append(definition.as_dict())

    diagnostics: list[Diagnostic] = []
    packs = {item["id"] for item in by_type["packs"]}
    assets = {item["id"] for item in by_type["assets"]}
    environments = {item["id"] for item in by_type["environments"]}
    for item in by_type["assets"]:
        if item["values"]["pack"] not in packs:
            source = item["source"]
            diagnostics.append(Diagnostic(source["file"], source["line"], "Asset", item["id"], "pack", f"unknown pack reference {item['values']['pack']}"))
    for item in by_type["rules"]:
        if item["values"]["asset"] not in assets:
            source = item["source"]
            diagnostics.append(Diagnostic(source["file"], source["line"], "Rule", item["id"], "asset", f"unknown asset reference {item['values']['asset']}"))
    for item in by_type["profiles"]:
        environment = item["values"].get("environment")
        if environment is not None and environment not in environments:
            source = item["source"]
            diagnostics.append(Diagnostic(source["file"], source["line"], "Profile", item["id"], "environment", f"unknown environment reference {environment}"))
    if diagnostics:
        raise DefinitionError(diagnostics)

    return {
        "schema": "c3x.renderer_definition_catalog.v0",
        "layer_order": [layer for layer, _definitions in provided],
        **by_type,
        "disabled": disabled,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse and merge C3X renderer definition layers")
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--custom", type=Path)
    parser.add_argument("--mod-root", type=Path, required=True)
    parser.add_argument("--scenario-root", type=Path)
    parser.add_argument("--allow-file", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    layers = [("default", args.default), ("scenario", args.scenario), ("custom", args.custom)]
    try:
        parsed = [
            (
                layer,
                parse_definition_file(
                    path, layer, args.mod_root, args.scenario_root, args.allow_file
                ),
            )
            for layer, path in layers
            if path is not None
        ]
        catalog = merge_layers(parsed)
        write_json(args.output, catalog)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
