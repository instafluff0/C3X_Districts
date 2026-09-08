#!/usr/bin/env python3
"""Validate the source-faithful per-system studies and composed Lab pickup."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local(name: str) -> Path:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"nonportable path: {name}")
    result = (ROOT / path).resolve()
    result.relative_to(ROOT / "Renderer")
    return result


def main() -> int:
    state = json.loads((HERE / "LAB_STATE_OF_ART.json").read_text())
    if state["schema"] != "c3x.lab_state_of_art.v2":
        raise ValueError("unexpected state-of-art schema")
    expected = {"mountain", "grass_plains_tundra_hills", "forest", "warrior"}
    actual = {study["id"] for study in state["studies"]}
    if actual != expected or len(state["studies"]) != len(expected):
        raise ValueError("isolated study inventory is incomplete or duplicated")
    quality = state["quality_contract"]
    for key in (
        "upstream_assets_are_authority", "authored_metadata_is_preserved",
        "uniform_object_scaling_only", "arbitrary_height_shortening_forbidden",
        "confirmed_source_and_lab_inference_are_separate",
    ):
        if quality.get(key) is not True:
            raise ValueError(f"quality contract disabled: {key}")
    composition = state["composition_contract"]
    if not composition["combined_scene_authoritative"] or not composition["city_excludes_trees"]:
        raise ValueError("accepted natural-scene composition contract changed")
    if composition["cities_rendered"] or composition["cities_changed"]:
        raise ValueError("city exclusion contract changed")
    if composition.get("blockers"):
        raise ValueError("accepted natural-scene composition has unresolved visual blockers")
    for study in state["studies"]:
        for key in ("fixture", "module", "source", "shader", "audit", "report"):
            if not local(study[key]).is_file():
                raise ValueError(f"missing {study['id']} {key}")
        for key in ("review_image", "raw_image"):
            row = study[key]
            path = local(row["path"])
            if not path.is_file() or digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
                raise ValueError(f"drifted {study['id']} {key}")
        report = json.loads(local(study["report"]).read_text())
        output = report["outputs"]
        if len(output) != 1 or output[0]["sha256"] != study["raw_image"]["sha256"]:
            raise ValueError(f"report mismatch: {study['id']}")
        fixture = json.loads(local(study["fixture"]).read_text())
        if fixture["settings"] != study["settings"]:
            raise ValueError(f"settings mismatch: {study['id']}")
        if fixture["settings"]["samples"] != 4 or fixture["settings"]["anisotropy"] != 16:
            raise ValueError(f"quality sampling regressed: {study['id']}")
        module = json.loads(local(study["module"]).read_text())
        if module["source"] != study["source"] or module["shader"] != study["shader"]:
            raise ValueError(f"module entry-point mismatch: {study['id']}")
    composed = state["composed_witness"]
    for key in (
        "fixture", "terrain_module", "mountain_module", "forest_module",
        "hydrology_module", "terrain_shader", "mountain_source", "mountain_shader",
        "forest_source", "forest_shader", "audit", "report", "repeat_report",
    ):
        if not local(composed[key]).is_file():
            raise ValueError(f"missing composed {key}")
    for key in ("review_image", "raw_image", "comparison"):
        row = composed[key]
        path = local(row["path"])
        if not path.is_file() or digest(path) != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"drifted composed {key}")
    report = json.loads(local(composed["report"]).read_text())
    output = next(row for row in report["outputs"] if row["image"] == composed["raw_image"]["path"])
    if output["sha256"] != composed["raw_image"]["sha256"]:
        raise ValueError("composed report mismatch")
    repeat = json.loads(local(composed["repeat_report"]).read_text())
    if ([row["sha256"] for row in repeat["outputs"]] !=
            [row["sha256"] for row in report["outputs"]] or
            composed["exact_repeat_variants"] != len(repeat["outputs"])):
        raise ValueError("composed exact-repeat mismatch")
    fixture = json.loads(local(composed["fixture"]).read_text())
    if fixture["tile_count"] != 100 or fixture["settings"] != composed["settings"]:
        raise ValueError("composed fixture/settings drifted")
    settings = fixture["settings"]
    if (settings["samples"], settings["anisotropy"], settings["render_scale"],
            settings["mip_bias"]) != (4, 16, 2, -1.0):
        raise ValueError("composed high-definition sampling regressed")
    terrain = json.loads(local(composed["terrain_module"]).read_text())
    mountain = json.loads(local(composed["mountain_module"]).read_text())
    forest = json.loads(local(composed["forest_module"]).read_text())
    hydrology = json.loads(local(composed["hydrology_module"]).read_text())
    if (terrain.get("source") != "Renderer/terrain_lab/v2/systems/relief/beauty_terrain.cpp" or
            terrain.get("shader") != composed["terrain_shader"]):
        raise ValueError("composed terrain is not using the exact accepted provider")
    if (mountain.get("source") != composed["mountain_source"] or
            mountain.get("shader") != composed["mountain_shader"]):
        raise ValueError("composed mountain is not using the exact accepted provider")
    if forest.get("source") != composed["forest_source"] or forest.get("shader") != composed["forest_shader"]:
        raise ValueError("composed forest is not using the accepted source-material path")
    if hydrology.get("suppress_relief") != 1 or hydrology.get("river_corridor") != 1:
        raise ValueError("retained hydrology restored superseded relief or lost rivers")
    terrain_source = local(terrain["source"]).read_text()
    mountain_source = local(composed["mountain_source"]).read_text()
    terrain_shader = local(composed["terrain_shader"]).read_text()
    mountain_shader = local(composed["mountain_shader"]).read_text()
    for token in ("composed_source_macro", "weighted_cells[10]", "geometry_flags"):
        if token not in terrain_source:
            raise ValueError("composed high-detail terrain contract missing: " + token)
    for token in ("std::array<HeightField, 5>", "constexpr unsigned grid = 128",
                  "height_lod0.dds", "blend_lod0.dds"):
        if token not in mountain_source:
            raise ValueError("composed authored mountain contract missing: " + token)
    if ("BEAUTY_COMPOSED_SHADOWS" not in terrain_shader or
            "BEAUTY_COMPOSED_SHADOWS" not in mountain_shader):
        raise ValueError("exact relief providers are not shared-shadow receivers")
    comparison_evidence = json.loads(
        local(composed["comparison"]["path"]).with_name("comparison.json").read_text()
    )
    if (comparison_evidence["resampling"] or comparison_evidence["cities_changed"] or
            comparison_evidence["candidate"]["sha256"] != composed["review_image"]["sha256"] or
            comparison_evidence["comparison"]["sha256"] != composed["comparison"]["sha256"]):
        raise ValueError("lossless composed comparison contract drifted")
    forest_source = local(composed["forest_source"]).read_text()
    for token in ("recipe.count", "material.paths[6]", "placement_column", "placement_row",
                  "draw.geometry_flags = material.paths[6].empty() ? 3u : 7u",
                  "draw.alpha_texture_slot"):
        if token not in forest_source:
            raise ValueError("composed forest contract missing: " + token)
    objects = json.loads((ROOT / "Renderer/packs/BeautyStudies/manifest.json").read_text())
    if (objects["forest_recipe_count"], objects["forest_recipe_weight"],
            objects["object_count"], objects["material_count"]) != (25, 180, 31, 28):
        raise ValueError("BeautyStudies source bundle inventory drifted")
    if "cities" not in state.get("excluded_from_current_update", {}):
        raise ValueError("city exclusion from the current update is not explicit")
    print("PASS pickup status: 4 isolated witnesses plus accepted 100-tile exact-provider composition; directional tree shadows and retained hydrology proven; cities unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
