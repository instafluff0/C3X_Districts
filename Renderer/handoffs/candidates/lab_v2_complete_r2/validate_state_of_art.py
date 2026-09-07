#!/usr/bin/env python3
"""Validate the isolated, source-faithful Lab state-of-the-art pickup."""

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
    if state["schema"] != "c3x.lab_state_of_art.v1":
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
    if composition["combined_scene_authoritative"] or not composition["city_excludes_trees"]:
        raise ValueError("obsolete combined-scene composition was re-authorized")
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
    objects = json.loads((ROOT / "Renderer/packs/BeautyStudies/manifest.json").read_text())
    if (objects["forest_recipe_count"], objects["forest_recipe_weight"],
            objects["object_count"], objects["material_count"]) != (25, 180, 31, 28):
        raise ValueError("BeautyStudies source bundle inventory drifted")
    if "cities" not in state.get("excluded_from_current_update", {}):
        raise ValueError("city exclusion from the current update is not explicit")
    print("PASS isolated Lab state of the art: 4 source-faithful Metal witnesses; cities unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
