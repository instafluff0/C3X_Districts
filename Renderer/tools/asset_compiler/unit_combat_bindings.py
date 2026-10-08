#!/usr/bin/env python3
"""Resolve a unit's combat effects into generic unit-pack metadata.

Inputs are importer evidence only: the source timing table
(`c3x.unit_effect_timing.v0`, release markers with sockets), a source map
(`civ6_combat_effect_map.json`: effect-name globs to generic profiles and
impact sets) and Civ III's own action timing
(`Renderer/inventory/civ3_unit_action_timing.json`). The output names only
generic profiles and impact sets, so the runtime never sees source names:

- releases: normalized attack-clip time, profile and model-space socket point;
- impact set: the munition family (`shell`, `naval`, `bomb`, `stone`,
  `bullet`, `arrow`, `melee`, `torpedo`, `missile`);
- bearing: the line of fire off the bow in degrees (ships fire broadsides);
- sync: Civ III's loudest attack sound and the clip's first release, both as
  normalized phases, so the clip can be warped onto the native report.

Units without a named munition fall back to their Civ III role: a Nuclear
Weapon or Cruise Missile ability (BIQ unit abilities), else the unit class.
Every unit that attacks gets a munition: C3X draws no native 2D effects.
"""
from __future__ import annotations

import fnmatch
import struct

import numpy as np

CLASS_FALLBACK = {0: "melee", 1: "naval", 2: "bomb"}  # Civ III unit class: land, sea, air


def role_fallback(role: dict | None) -> str | None:
    """Munition from Civ III's own unit data (BIQ semantics row)."""
    if not role:
        return None
    if role.get("nuclear_weapon"):
        return "nuclear"
    if role.get("cruise_missile"):
        return "missile"
    return CLASS_FALLBACK.get(role.get("unit_class"))
MAX_RELEASES = 16


def first_match(rules, name):
    return next((value for pattern, value in rules if fnmatch.fnmatchcase(name or "", pattern)), None)


def rig_world(blob: bytes, frame: int, bone: int) -> np.ndarray:
    """Bone world transform (row-vector) at an action frame: the payload
    palette is inverse-bind x world, and the C3XRIG1 trailer stores the
    inverse binds in skeleton order."""
    version, n, ni, nb, nf = struct.unpack_from("<5I", blob, 8)
    stride = 88 if version == 2 else 64
    palettes = 32 + n * stride + ni * 4
    trailer = palettes + nf * nb * 64
    if blob[trailer:trailer + 8] != b"C3XRIG1\0":
        raise ValueError("payload has no rig trailer")
    count = struct.unpack_from("<I", blob, trailer + 8)[0]
    binds = trailer + 48 + count * 4 + count * 4
    inverse_bind = np.frombuffer(blob, "<f4", 16, binds + bone * 64).reshape(4, 4).astype(float)
    frame = min(nf - 1, max(0, frame))
    palette = np.frombuffer(blob, "<f4", 16, palettes + (frame * nb + bone) * 64).reshape(4, 4).astype(float)
    # A collapsed (zero-scale) bone has no inverse; its translation still holds.
    return np.linalg.pinv(inverse_bind) @ palette


def resolve(timing: dict, mapping: dict, assets: list, attack_parts: dict, read_mesh) -> dict:
    """Releases, impact set and bearing from one unit's timing evidence.

    `assets` lists the attack action's component ids in part order,
    `attack_parts` is the binding's attack action and `read_mesh(path)`
    returns a part payload's bytes."""
    sockets = {(c["asset"], name): socket for node in timing.get("nodes", []) for c in node["components"]
               for socket in c["sockets"] for name in socket["names"]}
    releases, impact_set = [], None
    for release in timing.get("releases", []):
        if release.get("after_timeline_end") or release.get("asset") not in assets:
            continue
        if release["kind"] == "impact":
            impact = release.get("impact") or {}
            impact_set = impact_set or mapping["impact_set_by_impact"].get(f"{impact.get('collection')}/{impact.get('element')}")
            continue
        if release["kind"] != "effect":
            continue
        impact_set = first_match(mapping["impact_set_by_release"], release["effect"]) or impact_set
        profile = first_match(mapping["release_profiles"], release["effect"])
        socket = sockets.get((release["asset"], release.get("socket")))
        if profile is None or socket is None:
            continue
        blob = read_mesh(attack_parts[f"part{assets.index(release['asset'])}"]["mesh"])
        frames = struct.unpack_from("<5I", blob, 8)[4]
        world = np.array(socket["local"]["matrix"], float).reshape(4, 4) @ \
            rig_world(blob, round(release["normalized"] * (frames - 1)), release["payload_bone_index"])
        if not np.all(np.isfinite(world[3, :3])):
            continue
        releases.append({"normalized": float(release["normalized"]), "profile": profile,
                         "position": [float(v) for v in world[3, :3]]})
    releases.sort(key=lambda r: r["normalized"])
    # Without a named munition, the weapon that fires decides.
    impact_set = impact_set or next((mapping.get("impact_set_by_profile", {}).get(r["profile"]) for r in releases
                                     if r["profile"] in mapping.get("impact_set_by_profile", {})), None)
    attack = (timing.get("nodes") or [{}])[0].get("attack") or {}
    # Ships fire broadsides: the clip's side gives the bearing of the line of fire.
    bearing = first_match(mapping["attack_bearing_by_state"], attack.get("state")) or 0
    return {"releases": releases[:MAX_RELEASES], "impact_set": impact_set, "bearing": int(bearing)}


def publish(binding: dict, combat: dict | None, civ3: dict | None, role: dict | None) -> dict:
    """Write combat metadata into a unit binding (pack JSON) and return it.

    The binding parser has no arrays: releases use `release_count` and
    `release<n>` objects in the attack action."""
    combat = combat or {"releases": [], "impact_set": None, "bearing": 0}
    impact_set = combat["impact_set"] or role_fallback(role)
    published = {}
    if impact_set:
        published["impact_set"] = impact_set
    if combat["bearing"]:
        published["attack_bearing"] = combat["bearing"]
    attack = civ3.get("ATTACK1") if civ3 else None
    if combat["releases"] and attack and attack.get("sync_s") is not None and attack["duration_s"] > 0:
        sync, first = min(.95, attack["sync_s"] / attack["duration_s"]), combat["releases"][0]["normalized"]
        if warp_is_gentle(sync, first):
            published["attack_sync"] = round(sync, 4)
            published["attack_first_release"] = round(first, 4)
    victory = civ3.get("VICTORY") if civ3 else None
    if impact_set == "bomb" and victory and victory.get("sync_s") is not None:
        published["bomb_blast_s"] = round(victory["sync_s"], 4)
    binding.update(published)
    if combat["releases"] and isinstance(binding.get("attack"), dict):
        binding["attack"]["release_count"] = len(combat["releases"])
        for index, release in enumerate(combat["releases"]):
            x, y, z = (round(v, 5) for v in release["position"])
            binding["attack"][f"release{index}"] = {"phase": round(release["normalized"], 5),
                                                    "profile": release["profile"], "x": x, "y": y, "z": z}
            published[f"release{index}"] = binding["attack"][f"release{index}"]
    return published


def warp_is_gentle(sync: float, first: float) -> bool:
    """Sync only when the clip keeps a natural pace: each segment plays at
    0.4x-2.5x and at most 30% of the wind-up is skipped. Otherwise releases
    keep their own clip time (a far-off report would freeze or rush it)."""
    if not 0 < sync < 1 or not 0 <= first <= 1:
        return False
    if first >= sync:
        rate = (1 - first) / (1 - sync)
        return rate >= .4 and 1 - rate <= .3
    return first / sync >= .4 and (1 - first) / (1 - sync) <= 2.5


def clip_phase(phase: float, sync: float | None, first: float | None) -> float:
    """Civ III action phase -> attack-clip phase so the clip's first release
    lands on Civ III's sync point: a later release skips into the clip (inside
    the action blend), an earlier one slows the wind-up."""
    if sync is None or first is None or not 0 < sync < 1:
        return phase
    if first >= sync:
        return 1 - (1 - phase) * (1 - first) / (1 - sync)
    if phase < sync:
        return phase * first / sync
    return first + (phase - sync) * (1 - first) / (1 - sync)


def native_phase(clip: float, sync: float | None, first: float | None) -> float:
    """Inverse of `clip_phase`."""
    if sync is None or first is None or not 0 < sync < 1:
        return clip
    if first >= sync:
        return 1 - (1 - clip) * (1 - sync) / (1 - first)
    if clip < first:
        return clip * sync / first
    return sync + (clip - first) * (1 - sync) / (1 - first)
