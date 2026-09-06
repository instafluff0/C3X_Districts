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
    for profile_id, profile in sorted(profiles.items()):
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
    return {
        "schema": "c3x.effect_graph_pack.v0",
        "clock": source.get("clock"),
        "profiles": compiled,
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


def _random01(profile_id: str, instance_id: str, emitter_id: str, ordinal: int, lane: int) -> float:
    digest = hashlib.sha256(f"{profile_id}\0{instance_id}\0{emitter_id}\0{ordinal}\0{lane}".encode()).digest()
    return int.from_bytes(digest[:8], "little") / float(2**64)


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
    density = profile["zoom"][zoom]["density"]
    size_scale = profile["zoom"][zoom]["size"]
    for emitter in profile["emitters"]:
        interval = 1000.0 / (emitter["rate_per_second"] * density)
        first = max(0, math.floor((sample_time - emitter["particle_lifetime_ms"]) / interval) + 1)
        last = math.floor(sample_time / interval)
        ordinals = list(range(first, last + 1))[-emitter["max_particles"] :]
        for ordinal in ordinals:
            spawn = ordinal * interval
            age = sample_time - spawn
            if age < 0 or age >= emitter["particle_lifetime_ms"]:
                continue
            phase = age / emitter["particle_lifetime_ms"]
            angle = _random01(profile_id, instance_id, emitter["id"], ordinal, 0) * math.tau
            radius = math.sqrt(_random01(profile_id, instance_id, emitter["id"], ordinal, 1)) * emitter["spawn_radius_tile"]
            velocity = emitter["velocity_tile_per_second"]
            seconds = age / 1000.0
            frame = min(emitter["layout"]["frames"] - 1, int(phase * emitter["layout"]["frames"]))
            column = frame % emitter["layout"]["columns"]
            row = frame // emitter["layout"]["columns"]
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
                "position_tile": [
                    round(math.cos(angle) * radius + velocity[0] * seconds, 6),
                    round(math.sin(angle) * radius + velocity[1] * seconds, 6),
                    round(velocity[2] * seconds, 6),
                ],
                "size_tile": [round(v * size_scale, 6) for v in emitter["size_tile"]],
            })
    return particles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = compile_effect_graphs(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(
        f"Compiled {result['summary']['profiles']} generic effect profiles / "
        f"{result['summary']['emitters']} emitters at {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
