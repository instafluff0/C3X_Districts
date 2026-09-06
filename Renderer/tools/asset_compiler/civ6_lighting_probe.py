#!/usr/bin/env python3
"""Inventory Civ VI lighting, emissive, and ambient-effect source evidence."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = RENDERER_ROOT / "docs" / "civ6_lighting_probe.json"
GAME_LIGHTING_ARTDEF = Path("Base/ArtDefs/GameLighting.artdef")
WATER_ARTDEFS = (
    Path("Base/ArtDefs/Water.artdef"),
    Path("Base/ArtDefs/WaterMaterials.artdef"),
    Path("Base/ArtDefs/Wave.artdef"),
)
PRIMARY_PACKAGES = (
    Path("Base/Platforms/Windows/BLPs/lighting/default_lighting.blp"),
    Path("Base/Platforms/Windows/BLPs/Light.blp"),
    Path("Base/Platforms/Windows/BLPs/VFX_FireFX.blp"),
)
ATTACHMENT_PACKAGES = (
    Path("Base/Platforms/Windows/BLPs/landmarks/city_buildings.blp"),
    Path("Base/Platforms/Windows/BLPs/landmarks/tilebases.blp"),
    Path("Base/Platforms/Windows/BLPs/landmarks/hero_buildings.blp"),
)
PACKAGE_NAMES = {"default_lighting.blp", "light.blp", "vfx_firefx.blp"}
LIGHTING_TOKENS = (
    "LightRig",
    "m_vSunDirection",
    "m_vSunIntensity",
    "ApplyLightMapWeight",
    "DL_",
)
EFFECT_TOKENS = (
    "AnalyticLight",
    "Brazier",
    "ChimneySmoke",
    "Emissive",
    "Fire",
    "Flame",
    "Flicker",
    "Glow",
    "Lantern",
    "Light",
    "Smoke",
    "Spark",
    "Steam",
    "Torch",
)
TEXTURE_TOKENS = (
    "emissive",
    "fire",
    "flame",
    "glow",
    "light",
    "smoke",
    "spark",
    "steam",
    "torch",
)
ASCII_STRING = re.compile(rb"[A-Za-z_][A-Za-z0-9_./-]{3,}")


def text_attribute(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return element.get("text") or (element.text.strip() if element.text else None)


def parse_float_fields(element: ET.Element) -> dict[str, float]:
    fields: dict[str, float] = {}
    for value in element.findall("./m_Fields/m_Values/Element"):
        name = text_attribute(value.find("./m_ParamName"))
        raw = value.findtext("./m_fValue")
        if name and raw is not None:
            fields[name] = float(raw)
    return fields


def parse_game_lighting(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    rigs: list[dict[str, Any]] = []
    profiles: list[str] = []
    for profile in root.iter("Element"):
        cube_lights = next(
            (
                collection
                for collection in profile.findall("./m_ChildCollections/Element")
                if text_attribute(collection.find("./m_CollectionName")) == "CubeLights"
            ),
            None,
        )
        if cube_lights is None:
            continue
        profile_name = text_attribute(profile.find("./m_Name")) or "unnamed"
        profiles.append(profile_name)
        for element in cube_lights.findall("./Element"):
            binding = element.find(
                "./m_Fields/m_Values/Element[@class='AssetObjects..BLPEntryValue']"
            )
            if binding is None or text_attribute(binding.find("./m_XLPClass")) != "GameLighting":
                continue
            curves: list[dict[str, float]] = []
            for collection in element.findall("./m_ChildCollections/Element"):
                if text_attribute(collection.find("./m_CollectionName")) != "WeightCurve":
                    continue
                for point in collection.findall("./Element"):
                    values = parse_float_fields(point)
                    if "Time" in values and "Weight" in values:
                        curves.append({"time": values["Time"], "weight": values["Weight"]})
            rigs.append(
                {
                    "profile": profile_name,
                    "phase": text_attribute(element.find("./m_Name")),
                    "entry": text_attribute(binding.find("./m_EntryName")),
                    "xlp_class": text_attribute(binding.find("./m_XLPClass")),
                    "xlp_path": text_attribute(binding.find("./m_XLPPath")),
                    "blp_package": text_attribute(binding.find("./m_BLPPackage")),
                    "library": text_attribute(binding.find("./m_LibraryName")),
                    "weight_curve": curves,
                }
            )
    return {"profiles": profiles, "rigs": rigs}


def parse_artdef_bindings(assets_root: Path, relative_paths: Iterable[Path]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = assets_root / relative
        item: dict[str, Any] = {"path": relative.as_posix(), "exists": path.is_file(), "bindings": []}
        if path.is_file():
            root = ET.parse(path).getroot()
            for value in root.findall(".//Element[@class='AssetObjects..BLPEntryValue']"):
                item["bindings"].append(
                    {
                        "entry": text_attribute(value.find("./m_EntryName")),
                        "xlp_class": text_attribute(value.find("./m_XLPClass")),
                        "xlp_path": text_attribute(value.find("./m_XLPPath")),
                        "blp_package": text_attribute(value.find("./m_BLPPackage")),
                        "library": text_attribute(value.find("./m_LibraryName")),
                        "parameter": text_attribute(value.find("./m_ParamName")),
                    }
                )
            item["bindings"].sort(key=lambda binding: tuple(str(value or "") for value in binding.values()))
        evidence.append(item)
    return evidence


def extract_matching_strings(path: Path, tokens: Iterable[str]) -> list[str]:
    lowered = tuple(token.lower() for token in tokens)
    values = {
        match.group().decode("ascii")
        for match in ASCII_STRING.finditer(path.read_bytes())
        if any(token in match.group().decode("ascii").lower() for token in lowered)
    }
    return sorted(values, key=lambda value: (value.lower(), value))


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def package_evidence(assets_root: Path, relative_paths: Iterable[Path]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for relative in relative_paths:
        path = assets_root / relative
        tokens = LIGHTING_TOKENS + EFFECT_TOKENS
        evidence.append(
            {
                "path": relative.as_posix(),
                "exists": path.is_file(),
                "size": path.stat().st_size if path.is_file() else None,
                "matching_strings": extract_matching_strings(path, tokens) if path.is_file() else [],
            }
        )
    return evidence


def discover_named_packages(assets_root: Path) -> list[str]:
    return sorted(
        rel(path, assets_root)
        for path in assets_root.rglob("*.blp")
        if path.name.lower() in PACKAGE_NAMES
    )


def discover_effect_textures(assets_root: Path) -> list[str]:
    candidates: list[str] = []
    for path in assets_root.rglob("*"):
        if not path.is_file() or "SHARED_DATA" not in path.parts:
            continue
        name = path.name.lower()
        if any(token in name for token in TEXTURE_TOKENS):
            candidates.append(rel(path, assets_root))
    return sorted(candidates)


def build_report(assets_root: Path) -> dict[str, Any]:
    artdef = assets_root / GAME_LIGHTING_ARTDEF
    if not artdef.is_file():
        raise FileNotFoundError(f"Missing Civ VI GameLighting ArtDef: {artdef}")
    lighting = parse_game_lighting(artdef)
    return {
        "schema": "c3x.civ6_lighting_probe.v0",
        "read_policy": "metadata and printable package strings only; no cooked payload extraction",
        "source_paths_are_relative_to": "Civ VI Assets root",
        "global_lighting": {
            "artdef": GAME_LIGHTING_ARTDEF.as_posix(),
            **lighting,
        },
        "primary_package_evidence": package_evidence(assets_root, PRIMARY_PACKAGES),
        "model_attachment_evidence": package_evidence(assets_root, ATTACHMENT_PACKAGES),
        "water_artdef_evidence": parse_artdef_bindings(assets_root, WATER_ARTDEFS),
        "all_named_lighting_packages": discover_named_packages(assets_root),
        "shared_effect_texture_candidates": discover_effect_textures(assets_root),
        "supported_vertical_slice": {
            "environment": {
                "source_class": "GameLighting",
                "source_entries": ["Sunrise_LightRig", "Noon_LightRig", "Night_LightRig"],
                "evidence": "confirmed ArtDef bindings and weight curves",
                "runtime_conversion": "continuous source-independent C3X hour/season environment",
            },
            "water": {
                "source_class": "Water",
                "evidence": "confirmed structured Water/WaterMaterials/Wave ArtDef bindings",
                "runtime_conversion": "bounded source-independent Fresnel/specular response",
            },
            "analytic_light": {
                "source_resource": "DL_OrangeGlow",
                "evidence": "confirmed printable Light package resource name",
                "runtime_conversion": "generic authored point light",
            },
            "ambient_effect": {
                "source_resource": "Brazier_Fire_light",
                "evidence": "confirmed printable VFX resource name",
                "runtime_conversion": "generic authored flame/light fixture used for scheduling proof",
            },
            "model_attachment": {
                "source_instance_family": "Brazier*/FX_Brazier*",
                "evidence": "inferred resource-to-model relationship from repeated landmark package names",
                "runtime_conversion": "not converted; fixture uses an explicit authored local transform",
            },
            "typed_parameters": {
                "evidence": "unresolved",
                "runtime_conversion": "no source parameters or transforms inferred from names",
            },
        },
        "interpretation": {
            "confirmed": [
                "GameLighting.artdef binds named time-of-day phases to cooked GameLighting entries and weight curves.",
                "Cooked Light and VFX packages contain named analytic-light, glow, fire, smoke, steam, and torch resources.",
                "Shared-data trees contain concrete effect and emissive texture payload candidates.",
            ],
            "inferred": [
                "Effect/light names repeated in landmark packages are likely model-side attachments resolved against shared Light/VFX libraries.",
                "The engine evaluates ArtDef curves, scripts, materials, and attachment transforms at runtime.",
            ],
            "unresolved": [
                "Decode typed package records for attachment transforms, sockets, activation policies, and exact material texture roles.",
                "Determine which model emissives are embedded in landmark BLPs versus referenced from SHARED_DATA.",
                "Decode exact typed light/effect parameters beyond the supported authored fixture slice.",
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.assets_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output} with {len(report['global_lighting']['rigs'])} light phases, "
        f"{len(report['all_named_lighting_packages'])} named packages, and "
        f"{len(report['shared_effect_texture_candidates'])} texture candidates."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
