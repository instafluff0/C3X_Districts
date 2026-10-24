#!/usr/bin/env python3
"""Civ VI wonder effects (fires, smoke, lamps, blinkers) for the Renderer64 Lab.

A WonderMovie model's TimelineSet triggers spawn effects by name (FNV-1a of a
VFX or Light package entry) at an attachment point. Its timelines are the build
movie, the completion, the idle loop and the pillaged state. A finished wonder
shows the idle loop's effects and the permanent ones (start time -1), never the
construction dust or the pillage smoke. Each effect is placed at its attachment
bone in the finished pose and classified by name into the city effect layer's
kinds (flame, smoke, light).

    python3 Renderer/lab/studies/wonders/effects.py list [--source KEY ...]
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.wonders import kits, stages
from Renderer.tools.asset_compiler import clutter_blp_extractor as cbe
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.resource_pack_builder import fnv1a32


@lru_cache(maxsize=None)
def effect_names() -> dict:
    """FNV-1a hash -> name of every VFX and Light package entry (base, DLC, the mods')."""
    names = set()
    folders = [DEFAULT_ASSETS_ROOT] + [kits.WORKSHOP / item[0] for item in kits.MODS.values()]
    for folder in folders:
        for path in folder.rglob("*.blp"):
            lowered = path.name.lower()
            if "vfx" in lowered or "light" in lowered or "fx" in lowered:
                names |= {m.decode() for m in re.findall(rb"[A-Za-z0-9_\-. ]{4,80}", path.read_bytes())}
    table = {}
    for name in names:
        for variant in (name, name.lower(), name.upper()):
            table[fnv1a32(variant)] = name
    return table


def triggers(key: str) -> list[dict]:
    """The model's effect triggers: timeline, start time, attachment index, effect name."""
    record = kits.record(key)
    saved = cbe.TYPE_CITY_BLOCK
    cbe.TYPE_CITY_BLOCK = "WonderMoviePackageEntry"
    try:
        path = stages.package_path(key)
        try:
            package = IndexedStaticPackage(path, record["entry"])
        except ValueError:
            package = IndexedStaticPackage(path, record["entry"], minimum_temp_support=2)
        _landmark, _user, base = cbe.landmark_base_model(package)
    finally:
        cbe.TYPE_CITY_BLOCK = saved
    names = effect_names()
    timelines = []
    for _offset, pointer in package.pointer_fields(base, "TimelineSet::Timeline"):
        raw = package.bytes_for(pointer)
        timelines = [struct.unpack_from("<IIIIfI", raw, o) for o in range(0, len(raw), 24)]
    out = []
    for _offset, pointer in package.pointer_fields(base, "TimelineSet::Trigger"):
        raw = package.bytes_for(pointer)
        for index in range(len(raw) // 28):
            # kind, ?, start time, attachment index, attachment name hash, effect name hash, ?
            kind, _a, start, attachment, point_hash, name_hash, _c = struct.unpack_from("<IIfIIII", raw, index * 28)
            if kind != 1:
                continue
            timeline = next((t for t, (_x, _y, first, count, _d, _e) in enumerate(timelines)
                             if first <= index < first + count), None)
            out.append({"timeline": timeline, "timelines": len(timelines), "start": start, "attachment": attachment,
                        "name": names.get(name_hash, f"#{name_hash:08x}"), "bone_hash": point_hash})
    return out


def kind_of(name: str) -> str | None:
    """Effect class from its name: what the city effect layer can draw."""
    n = name.lower()
    if any(w in n for w in ("dust", "pil_", "_pil", "debris", "construct", "build", "spark", "reveal", "launch", "wake")):
        return None
    if any(w in n for w in ("fire", "flame", "torch", "brazier", "candle", "burn")):
        return "flame"
    if any(w in n for w in ("smoke", "steam", "chimney", "exhaust")):
        return "smoke"
    if any(w in n for w in ("light", "lamp", "lantern", "glow", "blink", "beacon")):
        return "light"
    if any(w in n for w in ("water", "fountain", "spray", "mist", "splash")):
        return "water"
    return "other"


def finished_effects(key: str) -> list[dict]:
    """Effects of the finished wonder (idle loop and permanent ones), each at its
    bone's position in the finished pose (kit units, model space). The trigger names
    its bone by FNV-1a hash; the importer's attachment index can point at another
    bone, so the hash decides. Effects whose bone the import lacks (the finished
    model's own FX bones, e.g. the Great Library's and Zeus' braziers) come without a
    position."""
    _asset, document, _transform = kits.landmarks(key)[0]
    bones = {}
    for skeleton, path in enumerate(document["components"]["skeletons"]):
        for bone, value in enumerate(kits.dk.load(kits.dk.SOURCES / path)["bones"]):
            if value.get("name"):
                bones.setdefault(fnv1a32(value["name"]), (skeleton, bone))
    staged = kits.posable(key)
    movie_skeleton = stages.movie(key)[4] if staged else None
    worlds = stages.world_matrices(key, 1.0) if staged else None
    rest = {}
    result = []
    for trigger in triggers(key):
        last = trigger["timelines"] - 1
        idle = trigger["timeline"] is not None and 0 < trigger["timeline"] < last
        permanent = trigger["start"] < 0 and trigger["timeline"] != last
        found = bones.get(trigger["bone_hash"])
        if not (idle or permanent):
            continue
        if found is None:
            # Its bone belongs to a model the import lacks: kits.effect_records places
            # it at a peak of the wonder's baked light map, if it has one.
            result.append({**trigger, "kind": kind_of(trigger["name"]), "position": None})
            continue
        skeleton, bone = found
        if staged and skeleton == movie_skeleton:
            world = worlds[bone]
        else:
            if skeleton not in rest:
                rest[skeleton] = kits.dk._skeleton_worlds(kits.dk.load(kits.dk.SOURCES / document["components"]["skeletons"][skeleton]))
            world = rest[skeleton][bone]
        result.append({**trigger, "kind": kind_of(trigger["name"]), "position": [world[12], world[13], world[14]]})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("list",))
    parser.add_argument("--source", action="append")
    args = parser.parse_args()
    for key in args.source or [k for k in kits.keys() if kits.is_movie(k)]:
        try:
            effects = finished_effects(key)
        except (ValueError, KeyError, IndexError) as exc:
            print(key, "skipped:", repr(exc)[:120])
            continue
        summary = {}
        for effect in effects:
            summary.setdefault((effect["kind"], effect["name"]), 0)
            summary[(effect["kind"], effect["name"])] += 1
        print(key, len(effects), sorted(summary.items(), key=lambda x: (str(x[0][0]), x[0][1]))[:14], flush=True)
    return 0


if __name__ == "__main__":
    from Renderer.lab.studies.wonders import effects as module
    raise SystemExit(module.main())
