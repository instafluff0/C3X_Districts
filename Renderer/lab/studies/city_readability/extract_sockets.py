#!/usr/bin/env python3
"""Recover flame, smoke and night-light attachment points for every city model.

Civ VI city buildings carry named attachment bones (torches, braziers, furnace
fires, chimney smoke, lamps). The original city import kept them as evidence,
but the compiled city library drops them, and the intermediate packs for most
selected buildings were later removed. Each compiled model's asset ID is a hash
of its installed package and entry, so this offline step recovers the entry,
decodes only its skeleton and attachment records, and writes their operational
positions in the compiled model's own frame. Source names never reach runtime
data; the output is a generic list per model.

    python3 Renderer/lab/studies/city_readability/extract_sockets.py
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
from Renderer.tools.asset_compiler.clutter_blp_extractor import landmark_base_model
from Renderer.tools.asset_compiler.compound_landmark_importer import (
    _decode_attachment_points, _decode_skeletons, _decode_states)
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage

RUNTIME = ROOT / "Renderer/packs/CityCompositionRuntime/manifest.json"
OUT = ROOT / "Renderer/packs/CityAccentsLab/sockets.json"
SOURCE_Z = 0.648266978876  # Lab city designs keep source proportion (vertical metric 1)
KINDS = ("flame", "smoke", "night_light")


def bind_position(bones: list[dict], index: int) -> list[float]:
    """World (model) position of a bone's rest pose, composing its parents."""
    def quat_rotate(q, v):
        x, y, z, w = q
        # v' = v + 2w(q x v) + 2 q x (q x v)
        cx, cy, cz = y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]
        cx2, cy2, cz2 = y * cz - z * cy, z * cx - x * cz, x * cy - y * cx
        return [v[0] + 2 * (w * cx + cx2), v[1] + 2 * (w * cy + cy2), v[2] + 2 * (w * cz + cz2)]

    def apply(bone, point):
        rest = bone["rest"]
        s = rest.get("scale_shear", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        p = [s[0] * point[0] + s[1] * point[1] + s[2] * point[2],
             s[3] * point[0] + s[4] * point[1] + s[5] * point[2],
             s[6] * point[0] + s[7] * point[1] + s[8] * point[2]]
        p = quat_rotate(rest.get("orientation", [0, 0, 0, 1]), p)
        return [p[j] + rest.get("position", [0, 0, 0])[j] for j in range(3)]

    point = [0.0, 0.0, 0.0]
    while index >= 0:
        bone = bones[index]
        point = apply(bone, point)
        index = bone.get("parent", -1)
    return point


def semantic(point: dict) -> str | None:
    value = point.get("semantic")
    if value in KINDS:
        return value
    name = (point.get("source_name") or point.get("name") or "").lower()
    if re.search(r"smoke|steam|chimney", name):
        return "smoke"
    if re.search(r"fire|torch|brazier|flame|furnace", name):
        return "flame"
    if re.search(r"light|lamp|lantern", name):
        return "night_light"
    return None


def operational(point: dict) -> bool:
    return point.get("state_hint") in (None, "operational", "worked")


def from_package(package_relative: str, entry: str, units: float) -> list[dict]:
    package = IndexedStaticPackage(DEFAULT_ASSETS_ROOT / package_relative, entry)
    package.select_direct_string(entry)
    _landmark, user_data, base_model = landmark_base_model(package)
    skeletons, _ = _decode_skeletons(package, base_model, units, allow_unvalidated=True)
    points, evidence = _decode_attachment_points(package, user_data, skeletons)
    return [dict(point, skeleton_data=skeletons[point["skeleton"]]["bones"])
            for point in points if point.get("skeleton") is not None and point.get("bone") is not None]


def from_pack(pack: Path, asset: str) -> list[dict]:
    manifest = json.loads((pack / "manifest.json").read_text())
    landmark = json.loads((pack / manifest["assets"][asset]["landmark"]).read_text())
    skeletons = [json.loads((pack / name).read_text())["bones"]
                 for name in landmark["components"].get("skeletons", [])]
    return [dict(point, skeleton_data=skeletons[point["skeleton"]])
            for point in landmark.get("attachment_points", [])
            if point.get("skeleton") is not None and point.get("bone") is not None]


def component_sources(wanted: set[str]) -> dict[str, tuple[str, str]]:
    found = {}
    for blp in DEFAULT_ASSETS_ROOT.glob("**/landmarks/*.blp"):
        relative = blp.relative_to(DEFAULT_ASSETS_ROOT).as_posix()
        for raw in set(re.findall(rb"[A-Za-z0-9_\-\.]{4,}", blp.read_bytes())):
            entry = raw.decode()
            digest = hashlib.sha256((relative + "\0" + entry).encode()).hexdigest()[:16]
            if digest in wanted:
                found[digest] = (relative, entry)
    return found


def accent_models() -> list[dict]:
    """Accent models are centred on their own bounds by recompose.py."""
    from Renderer.lab.shared.cities.assets import component
    pack = Path("Renderer/packs/CityAccentsLab")
    manifest = json.loads((ROOT / pack / "manifest.json").read_text())
    models = []
    for asset in sorted(manifest["assets"]):
        body = component(asset, pack)
        models.append({"asset": asset, "low": body["lo"], "high": body["hi"]})
    return models


def build() -> dict:
    models = json.loads(RUNTIME.read_text())["models"] + accent_models()
    wanted = {m["asset"].split("/")[-1] for m in models if m["asset"].startswith("city/component/")}
    sources = component_sources(wanted)
    result = {}
    counts = {kind: 0 for kind in KINDS}
    for index, model in enumerate(models):
        asset = model["asset"]
        try:
            if asset.startswith("city/component/"):
                package, entry = sources[asset.split("/")[-1]]
                points, z_factor = from_package(package, entry, 100.0), SOURCE_Z
            elif asset.startswith("city/palace/"):
                points, z_factor = from_pack(ROOT / "Renderer/packs/CityPalacesNormalized", asset), SOURCE_Z
            elif asset.startswith("city/accent/"):
                points, z_factor = from_pack(ROOT / "Renderer/packs/CityAccentsLab", asset), SOURCE_Z
            else:
                continue
        except (KeyError, ValueError, OSError) as error:
            print("SKIP", index, asset, error, flush=True)
            continue
        center = [(model["low"][j] + model["high"][j]) / 2 for j in (0, 1)]
        sockets = []
        for point in points:
            kind = semantic(point)
            if not kind or not operational(point):
                continue
            x, y, z = bind_position(point["skeleton_data"], point["bone"])
            sockets.append({"kind": kind, "position": [x - center[0], y - center[1], z * z_factor]})
            counts[kind] += 1
        if sockets:
            result[asset] = sockets
    OUT.write_text(json.dumps({"schema": "c3x.lab.city_sockets.v1",
                               "frame": "compiled model space (x,y centred, z source units scaled as the model)",
                               "models": result}, indent=1) + "\n")
    print("SOCKETS", len(result), "models", counts, flush=True)
    return result


if __name__ == "__main__":
    build()
