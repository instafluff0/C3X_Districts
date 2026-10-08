#!/usr/bin/env python3
"""Validate and compile deterministic, source-independent sprite effect graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


RENDERER_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path(__file__).with_name("effect_graph_profiles.json")
DEFAULT_OUTPUT = RENDERER_ROOT / "preview/out/effects/effect_graphs.json"
DEFAULT_TEXTURE_PACKS = (
    RENDERER_ROOT / "packs/AmbientEffectsNormalized/manifest.json",
    RENDERER_ROOT / "packs/CombatEffectsNormalized/manifest.json",
)
BLENDS = {"alpha", "additive", "premultiplied"}
FRAME_MODES = {"lifetime", "variant"}
ROTATIONS = {"none", "random", "emitter"}
ZOOMS = {"normal", "reduced"}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _finite_number(value: Any, *, positive: bool = False, nonnegative: bool = False) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return False
    return (not positive or value > 0) and (not nonnegative or value >= 0)


def texture_catalog(paths: tuple[Path, ...] = DEFAULT_TEXTURE_PACKS) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for asset_id, texture in document.get("textures", {}).items():
            if asset_id in result:
                raise ValueError(f"duplicate effect texture ID: {asset_id}")
            result[asset_id] = {"manifest": str(path), **texture}
    return result


def _validate_curve(curve: Any, label: str) -> None:
    if not isinstance(curve, list) or len(curve) < 2:
        raise ValueError(f"{label} must have at least two points")
    times = []
    for point in curve:
        if not isinstance(point, list) or len(point) != 2 or not all(_finite_number(v) for v in point):
            raise ValueError(f"{label} contains an invalid point")
        times.append(point[0])
        if not 0 <= point[0] <= 1 or not 0 <= point[1] <= 1:
            raise ValueError(f"{label} values must be normalized")
    if times != sorted(times) or times[0] != 0 or times[-1] != 1:
        raise ValueError(f"{label} must span monotonically from 0 to 1")


def _validate_optional(emitter: dict[str, Any], label: str, duration: float) -> None:
    """Optional one-shot fields; absent fields keep the original continuous behavior."""
    if not _finite_number(emitter.get("start_ms", 0), nonnegative=True) or emitter.get("start_ms", 0) >= duration:
        raise ValueError(f"{label} starts outside its profile")
    burst = emitter.get("burst")
    if burst is not None and (not isinstance(burst, int) or not 1 <= burst <= emitter["max_particles"]):
        raise ValueError(f"{label} burst must be within its particle bound")
    if not _finite_number(emitter.get("burst_spread_ms", 0), nonnegative=True):
        raise ValueError(f"{label} has invalid burst spread")
    if emitter.get("frame_mode", "lifetime") not in FRAME_MODES:
        raise ValueError(f"{label} has unsupported frame mode")
    if emitter.get("rotation", "none") not in ROTATIONS:
        raise ValueError(f"{label} has unsupported rotation")
    spread = emitter.get("spread_tile_per_second", [0, 0, 0, 0])
    if not isinstance(spread, list) or len(spread) != 4 or not all(_finite_number(v) for v in spread) or \
            spread[0] > spread[1] or spread[2] > spread[3]:
        raise ValueError(f"{label} has invalid velocity spread")
    for key in ("gravity_tile_per_second2", "spin_per_second"):
        if not _finite_number(emitter.get(key, 0)):
            raise ValueError(f"{label} has invalid {key}")
    if not _finite_number(emitter.get("intensity", 1), positive=True):
        raise ValueError(f"{label} has invalid intensity")
    pivot = emitter.get("pivot", [0.5, 0.5])
    if not isinstance(pivot, list) or len(pivot) != 2 or not all(_finite_number(v) and 0 <= v <= 1 for v in pivot):
        raise ValueError(f"{label} has invalid pivot")
    tint = emitter.get("tint", [1, 1, 1])
    if not isinstance(tint, list) or len(tint) != 3 or not all(_finite_number(v, nonnegative=True) for v in tint):
        raise ValueError(f"{label} has invalid tint")
    size_curve = emitter.get("size_curve")
    if size_curve is not None:
        if not isinstance(size_curve, list) or len(size_curve) < 2 or not all(
                isinstance(p, list) and len(p) == 2 and _finite_number(p[0]) and _finite_number(p[1], nonnegative=True)
                for p in size_curve) or [p[0] for p in size_curve] != sorted(p[0] for p in size_curve) or \
                size_curve[0][0] != 0 or size_curve[-1][0] != 1:
            raise ValueError(f"{label} has an invalid size curve")


SCALED_FIELDS = ("size_tile", "velocity_tile_per_second", "spread_tile_per_second", "spawn_radius_tile",
                 "gravity_tile_per_second2")


def _expand(profile_id: str, profiles: dict[str, Any], seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Resolve `extends` (another profile's emitters) and `scale` (a uniform
    size for every emitter's sizes, speeds, spread and gravity). The child's
    own keys override; runtime packs only ever see expanded profiles."""
    profile = profiles[profile_id]
    if "extends" not in profile:
        return profile
    base_id = profile["extends"]
    if base_id not in profiles or base_id in seen + (profile_id,):
        raise ValueError(f"{profile_id} extends an unknown or cyclic profile")
    base = _expand(base_id, profiles, seen + (profile_id,))
    scale = profile.get("scale", 1.0)
    if not _finite_number(scale, positive=True):
        raise ValueError(f"{profile_id} has invalid scale")
    emitters = []
    for emitter in base["emitters"]:
        copy = json.loads(json.dumps(emitter))
        for key in SCALED_FIELDS:
            if key in copy:
                copy[key] = [v * scale for v in copy[key]] if isinstance(copy[key], list) else copy[key] * scale
        emitters.append(copy)
    result = {**base, "emitters": emitters, "bounds_tile": [v * scale for v in base["bounds_tile"]]}
    result.update({k: v for k, v in profile.items() if k not in ("extends", "scale")})
    return result


def _validate_repeat(profile: dict[str, Any], label: str) -> None:
    repeat = profile.get("repeat")
    if repeat is None:
        return
    if not isinstance(repeat.get("count"), int) or not 1 <= repeat["count"] <= 16 or \
            not _finite_number(repeat.get("interval_ms", 0), nonnegative=True) or \
            not _finite_number(repeat.get("jitter_tile", 0), nonnegative=True) or \
            not isinstance(repeat.get("spacing_tile", [0, 0, 0]), list) or len(repeat.get("spacing_tile", [0, 0, 0])) != 3 or \
            not all(_finite_number(v) for v in repeat.get("spacing_tile", [0, 0, 0])):
        raise ValueError(f"{label} has an invalid repeat")


def compile_effect_graphs(
    source_path: Path = DEFAULT_SOURCE,
    texture_manifests: tuple[Path, ...] = DEFAULT_TEXTURE_PACKS,
) -> dict[str, Any]:
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get("schema") != "c3x.effect_graph_sources.v0":
        raise ValueError("unsupported effect graph source schema")
    if source.get("runtime_activation") != "not_enabled":
        raise ValueError("offline effect graph sources must not enable runtime rendering")
    textures = texture_catalog(texture_manifests)
    profiles = source.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("effect graph source has no profiles")
    compiled: dict[str, Any] = {}
    texture_refs: set[str] = set()
    profiles = {profile_id: _expand(profile_id, profiles) for profile_id in profiles}
    for profile_id, profile in sorted(profiles.items()):
        _validate_repeat(profile, profile_id)
        duration = profile.get("duration_ms")
        bounds = profile.get("bounds_tile")
        emitters = profile.get("emitters")
        zoom = profile.get("zoom")
        if not _finite_number(duration, positive=True):
            raise ValueError(f"{profile_id} has invalid duration")
        if (
            not isinstance(bounds, list)
            or len(bounds) != 6
            or not all(_finite_number(v) for v in bounds)
            or any(bounds[i] >= bounds[i + 3] for i in range(3))
        ):
            raise ValueError(f"{profile_id} has invalid tile bounds")
        if not isinstance(emitters, list) or not emitters:
            raise ValueError(f"{profile_id} has no emitters")
        if not isinstance(zoom, dict) or set(zoom) != ZOOMS:
            raise ValueError(f"{profile_id} must define normal and reduced zoom")
        for zoom_id, policy in zoom.items():
            if not all(_finite_number(policy.get(k), positive=True) for k in ("density", "size")):
                raise ValueError(f"{profile_id}.{zoom_id} has invalid zoom policy")
        emitter_ids: set[str] = set()
        compiled_emitters = []
        for emitter in emitters:
            emitter_id = emitter.get("id")
            texture_id = emitter.get("texture")
            alpha_id = emitter.get("alpha_texture")
            layout = emitter.get("layout")
            if not isinstance(emitter_id, str) or not emitter_id or emitter_id in emitter_ids:
                raise ValueError(f"{profile_id} has an invalid or duplicate emitter ID")
            emitter_ids.add(emitter_id)
            refs = [texture_id] + ([] if alpha_id is None else [alpha_id])
            if any(ref not in textures for ref in refs):
                raise ValueError(f"{profile_id}.{emitter_id} references an unavailable texture")
            texture_refs.update(refs)
            if emitter.get("blend") not in BLENDS:
                raise ValueError(f"{profile_id}.{emitter_id} has unsupported blend mode")
            if (
                not isinstance(layout, dict)
                or layout.get("order") != "row_major"
                or not all(isinstance(layout.get(k), int) and layout[k] > 0 for k in ("columns", "rows", "frames"))
                or layout["frames"] > layout["columns"] * layout["rows"]
            ):
                raise ValueError(f"{profile_id}.{emitter_id} has invalid atlas layout")
            if not all(_finite_number(emitter.get(k), positive=True) for k in ("rate_per_second", "particle_lifetime_ms")):
                raise ValueError(f"{profile_id}.{emitter_id} has invalid timing")
            if not isinstance(emitter.get("max_particles"), int) or emitter["max_particles"] < 1:
                raise ValueError(f"{profile_id}.{emitter_id} has invalid particle bound")
            if not isinstance(emitter.get("size_tile"), list) or len(emitter["size_tile"]) != 2 or not all(
                _finite_number(v, positive=True) for v in emitter["size_tile"]
            ):
                raise ValueError(f"{profile_id}.{emitter_id} has invalid size")
            velocity = emitter.get("velocity_tile_per_second")
            if not isinstance(velocity, list) or len(velocity) != 3 or not all(_finite_number(v) for v in velocity):
                raise ValueError(f"{profile_id}.{emitter_id} has invalid velocity")
            if not _finite_number(emitter.get("spawn_radius_tile"), nonnegative=True):
                raise ValueError(f"{profile_id}.{emitter_id} has invalid spawn radius")
            _validate_curve(emitter.get("opacity_curve"), f"{profile_id}.{emitter_id}.opacity_curve")
            _validate_optional(emitter, f"{profile_id}.{emitter_id}", duration)
            compiled_emitters.append({
                **emitter,
                "atlas_uv_step": [1.0 / layout["columns"], 1.0 / layout["rows"]],
                "atlas_layout_evidence": "authored_generic_not_source_script_decode",
            })
        compiled[profile_id] = {
            **profile,
            "emitters": compiled_emitters,
            "maximum_live_particles": sum(item["max_particles"] for item in compiled_emitters),
            "graph_hash": hashlib.sha256(_canonical(profile)).hexdigest(),
            "runtime_activation": "not_enabled",
        }
    impact_sets = source.get("impact_sets", {})
    for set_id, outcomes in impact_sets.items():
        # A munition's look per native outcome; units bind a set, never a name.
        # Outcomes: land target hit or missed, a ship hit, any other strike on water, an aircraft struck.
        if set(outcomes) != set(IMPACT_OUTCOMES) or any(value not in compiled for value in outcomes.values()):
            raise ValueError(f"impact set {set_id} must map {', '.join(IMPACT_OUTCOMES)} to compiled profiles")
    return {
        "schema": "c3x.effect_graph_pack.v0",
        "clock": source.get("clock"),
        "profiles": compiled,
        "impact_sets": impact_sets,
        "texture_dependencies": {
            key: {
                "texture": textures[key]["texture"],
                "format": textures[key]["format"],
                "width": textures[key]["width"],
                "height": textures[key]["height"],
            }
            for key in sorted(texture_refs)
        },
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "summary": {
            "profiles": len(compiled),
            "emitters": sum(len(value["emitters"]) for value in compiled.values()),
            "texture_dependencies": len(texture_refs),
            "maximum_particles_across_profiles": max(value["maximum_live_particles"] for value in compiled.values()),
        },
        "source_behavior_claim": "none",
        "runtime_activation": "not_enabled",
    }


IMPACT_OUTCOMES = ("hit", "miss", "water", "ship", "air")


def fnv1a32(text: str) -> int:
    value = 0x811C9DC5
    for byte in text.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def _mix32(value: int) -> int:
    value &= 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def _random01(profile_id: str, instance_id: str, emitter_id: str, ordinal: int, lane: int) -> float:
    """Deterministic [0,1) from integer mixing; the runtime reproduces it exactly."""
    key = fnv1a32(f"{profile_id}\0{instance_id}\0{emitter_id}")
    return _mix32(key ^ _mix32(ordinal * 0x9E3779B1 + lane * 0x85EBCA6B)) / 4294967296.0


def _sample_curve(curve: list[list[float]], phase: float) -> float:
    for left, right in zip(curve, curve[1:]):
        if phase <= right[0]:
            span = right[0] - left[0]
            mix = 0.0 if span <= 0 else (phase - left[0]) / span
            return left[1] + (right[1] - left[1]) * mix
    return curve[-1][1]


def sample_effect(profile_id: str, profile: dict[str, Any], instance_id: str, time_ms: int, zoom: str) -> list[dict[str, Any]]:
    if zoom not in ZOOMS or not isinstance(time_ms, int) or time_ms < 0:
        raise ValueError("effect sample needs a valid zoom and nonnegative integer time")
    duration = int(profile["duration_ms"])
    if not profile["loop"] and time_ms >= duration:
        return []
    # Looping emitters use the unwrapped absolute clock. Modulo-wrapping the
    # emitter age would drop every still-live particle at the profile boundary.
    sample_time = time_ms
    particles = []
    repeat = profile.get("repeat")
    if repeat:
        # A stick or salvo: copies spaced along the event direction (centered on
        # the target) with optional scatter, each `interval_ms` after the last.
        count, spacing = repeat["count"], repeat.get("spacing_tile", [0, 0, 0])
        key = f"{instance_id}#repeat"
        for copy in range(count):
            angle = _random01(profile_id, key, "jitter", copy, 0) * math.tau
            radius = math.sqrt(_random01(profile_id, key, "jitter", copy, 1)) * repeat.get("jitter_tile", 0)
            offset = [(copy - (count - 1) / 2) * spacing[0] + math.cos(angle) * radius,
                      (copy - (count - 1) / 2) * spacing[1] + math.sin(angle) * radius,
                      (copy - (count - 1) / 2) * spacing[2]]
            local = time_ms - round(copy * repeat.get("interval_ms", 0))
            if local < 0:
                continue
            for particle in _sample_once(profile_id, profile, f"{instance_id}#{copy}", local, zoom):
                particle["position_tile"] = [round(p + o, 6) for p, o in zip(particle["position_tile"], offset)]
                particle["position_tile"][2] = max(0.0, particle["position_tile"][2])
                particles.append(particle)
        return particles
    return _sample_once(profile_id, profile, instance_id, sample_time, zoom)


def _sample_once(profile_id: str, profile: dict[str, Any], instance_id: str, sample_time: int, zoom: str) -> list[dict[str, Any]]:
    particles = []
    density = profile["zoom"][zoom]["density"]
    size_scale = profile["zoom"][zoom]["size"]
    for emitter in profile["emitters"]:
        local = sample_time - emitter.get("start_ms", 0)
        lifetime = emitter["particle_lifetime_ms"]
        if local < 0:
            continue
        rand = lambda ordinal, lane: _random01(profile_id, instance_id, emitter["id"], ordinal, lane)
        if "burst" in emitter:
            # A one-shot emitter spawns its whole burst at its start, spread over
            # burst_spread_ms; zoom density thins the burst.
            count = max(1, round(emitter["burst"] * density))
            spawns = [(ordinal, rand(ordinal, 7) * emitter.get("burst_spread_ms", 0)) for ordinal in range(count)]
        else:
            interval = 1000.0 / (emitter["rate_per_second"] * density)
            first = max(0, math.floor((local - lifetime) / interval) + 1)
            last = math.floor(local / interval)
            spawns = [(ordinal, ordinal * interval) for ordinal in range(first, last + 1)][-emitter["max_particles"]:]
        for ordinal, spawn in spawns:
            age = local - spawn
            if age < 0 or age >= lifetime:
                continue
            phase = age / lifetime
            angle = rand(ordinal, 0) * math.tau
            radius = math.sqrt(rand(ordinal, 1)) * emitter["spawn_radius_tile"]
            spread = emitter.get("spread_tile_per_second", [0, 0, 0, 0])
            outward = spread[0] + (spread[1] - spread[0]) * rand(ordinal, 2)
            rise = spread[2] + (spread[3] - spread[2]) * rand(ordinal, 3)
            velocity = emitter["velocity_tile_per_second"]
            seconds = age / 1000.0
            gravity = emitter.get("gravity_tile_per_second2", 0.0)
            frames = emitter["layout"]["frames"]
            if emitter.get("frame_mode", "lifetime") == "variant":
                frame = min(frames - 1, int(rand(ordinal, 4) * frames))
            else:
                frame = min(frames - 1, int(phase * frames))
            column = frame % emitter["layout"]["columns"]
            row = frame // emitter["layout"]["columns"]
            rotation_mode = emitter.get("rotation", "none")
            rotation = (rand(ordinal, 5) * math.tau if rotation_mode == "random" else 0.0) + \
                emitter.get("spin_per_second", 0.0) * (rand(ordinal, 6) * 2 - 1) * seconds
            size = 1.0 if "size_curve" not in emitter else _sample_curve(emitter["size_curve"], phase)
            particles.append({
                "id": f"{instance_id}/{emitter['id']}/{ordinal}",
                "emitter": emitter["id"],
                "texture": emitter["texture"],
                "alpha_texture": emitter.get("alpha_texture"),
                "blend": emitter["blend"],
                "frame": frame,
                "atlas_uv": [
                    round(column * emitter["atlas_uv_step"][0], 6),
                    round(row * emitter["atlas_uv_step"][1], 6),
                    round((column + 1) * emitter["atlas_uv_step"][0], 6),
                    round((row + 1) * emitter["atlas_uv_step"][1], 6),
                ],
                "age_normalized": round(phase, 6),
                "opacity": round(_sample_curve(emitter["opacity_curve"], phase), 6),
                "intensity": emitter.get("intensity", 1.0),
                "tint": emitter.get("tint", [1.0, 1.0, 1.0]),
                "pivot": emitter.get("pivot", [0.5, 0.5]),
                "orientation": "emitter" if rotation_mode == "emitter" else "camera",
                "rotation": round(rotation, 6),
                "position_tile": [
                    round(math.cos(angle) * (radius + outward * seconds) + velocity[0] * seconds, 6),
                    round(math.sin(angle) * (radius + outward * seconds) + velocity[1] * seconds, 6),
                    round(max(0.0, (velocity[2] + rise) * seconds + 0.5 * gravity * seconds * seconds), 6),
                ],
                "size_tile": [round(v * size_scale * size, 6) for v in emitter["size_tile"]],
            })
    return particles


DXGI_FORMATS = {"BC1_UNORM_SRGB": 72, "BC3_UNORM_SRGB": 78, "BC4_UNORM": 80, "BC1_UNORM": 71, "BC3_UNORM": 77}
PACK_MAGIC = b"C3XFX1\0\0"
# Codes shared with render_core/effect_sampler.h (Blend, Rotation).
BLEND_CODES = {"alpha": 0, "additive": 1, "premultiplied": 2}
ROTATION_CODES = {"none": 0, "random": 1, "emitter": 2}


def runtime_aliases(compiled: dict[str, Any]) -> dict[str, str]:
    """`impact/<set>/<outcome>` -> profile, for every impact set outcome."""
    return {f"impact/{set_id}/{outcome}": profile for set_id, outcomes in sorted(compiled.get("impact_sets", {}).items())
            for outcome, profile in sorted(outcomes.items())}


def write_runtime_pack(compiled: dict[str, Any], output: Path, texture_manifests: tuple[Path, ...] = DEFAULT_TEXTURE_PACKS) -> dict[str, Any]:
    """Write the flat runtime effect pack `effects.bin` and its textures.

    Little-endian: magic, version, counts; then texture, profile and emitter
    records, curve points and a UTF-8 string table. Records are fixed size so
    the runtime reader bounds-checks every index. Textures are copied beside it
    under `textures/` by asset ID; a pack built from licensed sources stays
    local like its sources."""
    import shutil
    import struct
    textures = texture_catalog(texture_manifests)
    strings = bytearray()

    def text(value: str) -> tuple[int, int]:
        data = value.encode("utf-8")
        strings.extend(data)
        return len(strings) - len(data), len(data)

    texture_ids = sorted(compiled["texture_dependencies"])
    texture_index = {asset: index for index, asset in enumerate(texture_ids)}
    texture_records = bytearray()
    output.mkdir(parents=True, exist_ok=True)
    for asset in texture_ids:
        entry = textures[asset]
        relative = "textures/" + asset.replace("/", "_") + ".dds"
        source = Path(entry["manifest"]).parent / entry["texture"]
        (output / "textures").mkdir(exist_ok=True)
        shutil.copyfile(source, output / relative)
        texture_records += struct.pack("<2I3I", *text(relative), entry["width"], entry["height"], DXGI_FORMATS[entry["format"]])
    profiles, emitters, points = bytearray(), bytearray(), []

    def curve(values: list[list[float]]) -> tuple[int, int]:
        points.extend(values)
        return len(points) - len(values), len(values)

    emitter_count = 0
    first_emitter = {}

    def profile_record(profile_id: str, profile: dict[str, Any], first: int) -> bytes:
        zoom = profile["zoom"]
        repeat = profile.get("repeat", {})
        return struct.pack("<2I I f 2f 2f 2I I f 3f f", *text(profile_id), int(profile["loop"]), profile["duration_ms"],
                           zoom["normal"]["density"], zoom["reduced"]["density"],
                           zoom["normal"]["size"], zoom["reduced"]["size"], first, len(profile["emitters"]),
                           repeat.get("count", 0), repeat.get("interval_ms", 0), *repeat.get("spacing_tile", [0, 0, 0]),
                           repeat.get("jitter_tile", 0))

    for profile_id, profile in sorted(compiled["profiles"].items()):
        first_emitter[profile_id] = emitter_count
        profiles += profile_record(profile_id, profile, emitter_count)
        for e in profile["emitters"]:
            alpha = texture_index[e["alpha_texture"]] if e.get("alpha_texture") else 0xFFFFFFFF
            emitters += struct.pack(
                "<2I 2I 5I I 4f i I 2f 3f 4f 3f 4f 2I 2I 2f",
                *text(e["id"]), texture_index[e["texture"]], alpha, BLEND_CODES[e["blend"]],
                e["layout"]["columns"], e["layout"]["rows"], e["layout"]["frames"], int(e.get("frame_mode") == "variant"),
                ROTATION_CODES[e.get("rotation", "none")],
                e["rate_per_second"], e["particle_lifetime_ms"], e.get("start_ms", 0), e.get("burst_spread_ms", 0),
                e.get("burst", 0), e["max_particles"], *e["size_tile"], *e["velocity_tile_per_second"],
                *e.get("spread_tile_per_second", [0, 0, 0, 0]), *e.get("tint", [1, 1, 1]),
                e["spawn_radius_tile"], e.get("gravity_tile_per_second2", 0), e.get("spin_per_second", 0), e.get("intensity", 1),
                *curve(e["opacity_curve"]), *(curve(e["size_curve"]) if "size_curve" in e else (0, 0)),
                *e.get("pivot", [0.5, 0.5]))
            emitter_count += 1
    # The runtime resolves a munition's outcome by name: impact/<set>/<outcome>
    # shares its target profile's emitters.
    aliases = runtime_aliases(compiled)
    for alias, target in aliases.items():
        profiles += profile_record(alias, compiled["profiles"][target], first_emitter[target])
    body = struct.pack("<6I", 1, len(texture_ids), len(compiled["profiles"]) + len(aliases), emitter_count, len(points), len(strings))
    data = PACK_MAGIC + body + texture_records + profiles + emitters + b"".join(struct.pack("<2f", *p) for p in points) + bytes(strings)
    (output / "effects.bin").write_bytes(data)
    return {"textures": len(texture_ids), "profiles": len(compiled["profiles"]) + len(aliases), "emitters": emitter_count,
            "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-pack", type=Path, help="also write a flat runtime pack (effects.bin + textures) here")
    args = parser.parse_args()
    try:
        result = compile_effect_graphs(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.runtime_pack:
            print("runtime pack", write_runtime_pack(result, args.runtime_pack))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        f"Compiled {result['summary']['profiles']} generic effect profiles / "
        f"{result['summary']['emitters']} emitters at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
