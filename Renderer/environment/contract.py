"""Validation and deterministic fixture evaluation for the M6.4 environment contract."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTIVATION_POLICIES = {"always", "night", "twilight-and-night", "hour-range"}
MISSING_POLICIES = {"non-emissive", "omit-attachment", "owner-fallback"}


class EnvironmentContractError(ValueError):
    pass


@dataclass(frozen=True)
class EnvironmentState:
    hour: float
    sun_intensity: float
    moon_intensity: float
    ambient: tuple[float, float, float]
    exposure: float
    night_activation: float
    emissive_scale: float
    water_fresnel: float
    water_specular: float


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _smoothstep(low: float, high: float, value: float) -> float:
    amount = _clamp01((value - low) / (high - low))
    return amount * amount * (3.0 - 2.0 * amount)


def evaluate_environment(hour: float, season: int = 0) -> EnvironmentState:
    if not math.isfinite(hour):
        raise EnvironmentContractError("hour must be finite")
    hour %= 24.0
    daylight = _clamp01(_smoothstep(5.0, 7.0, hour) * (1.0 - _smoothstep(17.0, 19.0, hour)))
    sun_elevation = max(0.0, math.sin((hour - 5.0) * math.pi / 14.0))
    night = _clamp01(1.0 - daylight)
    warm = _clamp01(1.0 - abs(sun_elevation - 0.18) / 0.32) * _smoothstep(0.02, 0.16, sun_elevation)
    dusk_mix = warm * _smoothstep(12.0, 18.0, hour)
    dawn_mix = warm * (1.0 - _smoothstep(12.0, 18.0, hour))
    sun_intensity = _clamp01(sun_elevation * (0.45 + 0.35 * daylight) + 0.20 * daylight)
    ambient = [
        0.18 + 0.55 * daylight + 0.20 * dusk_mix + 0.040 * dawn_mix,
        0.23 + 0.55 * daylight + 0.070 * dusk_mix + 0.020 * dawn_mix,
        0.42 + 0.40 * daylight - 0.12 * dusk_mix + 0.030 * dawn_mix,
    ]
    if season == 1:
        ambient[0] *= 1.04
        ambient[1] *= 0.94
    elif season == 2:
        ambient[0] *= 0.88
        ambient[2] *= 1.08
    return EnvironmentState(
        hour=hour,
        sun_intensity=sun_intensity,
        moon_intensity=night * (0.18 + 0.16 * (1.0 - daylight)),
        ambient=tuple(ambient),
        exposure=0.79 + 0.21 * daylight,
        night_activation=night,
        emissive_scale=0.25 + 1.10 * night,
        water_fresnel=0.04 + 0.08 * night,
        water_specular=_clamp01(0.20 + 0.42 * sun_intensity + 0.30 * night * 0.30),
    )


def activation(policy: str, state: EnvironmentState, hour_range: list[float] | None = None) -> float:
    if policy == "always":
        return 1.0
    if policy == "night":
        return state.night_activation
    if policy == "twilight-and-night":
        return _clamp01(state.night_activation * 1.35)
    if policy != "hour-range" or not hour_range or len(hour_range) != 2:
        raise EnvironmentContractError("hour-range activation requires two hours")
    start, end = (value % 24.0 for value in hour_range)
    return float(start <= state.hour <= end if start <= end else state.hour >= start or state.hour <= end)


def attachment_phase(attachment: dict[str, Any], presentation_ticks: int) -> int:
    period = attachment.get("period_ticks", 0)
    if not isinstance(period, int) or period <= 0:
        return 0
    seed = attachment.get("stable_phase_seed", 0)
    if not isinstance(seed, int) or seed < 0:
        raise EnvironmentContractError("stable_phase_seed must be a nonnegative integer")
    return ((max(0, presentation_ticks) % period + seed % period) % period) * 1_000_000 // period


def attachment_status(
    attachment: dict[str, Any], state: EnvironmentState, presentation_ticks: int, *,
    visible: bool = True, resources_available: bool = True, owner_replaced: bool = True,
    current_states: set[str] | None = None,
) -> dict[str, Any]:
    missing_policy = attachment["missing_policy"]
    if not resources_available or not owner_replaced:
        return {"active": False, "animated": False, "phase_millionths": 0, "degrade": missing_policy}
    required = set(attachment.get("state_requirements", []))
    if not visible or not required.issubset(current_states or set()):
        return {"active": False, "animated": False, "phase_millionths": 0, "degrade": "omit-attachment"}
    amount = activation(attachment["activation"], state, attachment.get("active_hours"))
    animated = bool(attachment.get("animated")) and amount > 0.0
    return {
        "active": amount > 0.0,
        "animated": animated,
        "phase_millionths": attachment_phase(attachment, presentation_ticks) if animated else 0,
        "degrade": None,
    }


def _vector(value: Any, count: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != count or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value):
        raise EnvironmentContractError(f"{label} must be a finite {count}-component array")


def _transform(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise EnvironmentContractError(f"{label} must be an object")
    _vector(value.get("translation"), 3, f"{label}.translation")
    _vector(value.get("rotation_degrees"), 3, f"{label}.rotation_degrees")
    _vector(value.get("scale"), 3, f"{label}.scale")


def validate_fixture(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != "c3x.environment_fixture.v0":
        raise EnvironmentContractError("fixture schema must be c3x.environment_fixture.v0")
    sizes = data.get("viewports")
    if not isinstance(sizes, list) or len(sizes) < 2:
        raise EnvironmentContractError("at least two viewport fixtures are required")
    for index, size in enumerate(sizes):
        if not isinstance(size, list) or len(size) != 2 or not all(isinstance(item, int) and 1 <= item <= 8192 for item in size):
            raise EnvironmentContractError(f"viewports[{index}] is invalid")
    hours = data.get("hours")
    if hours != {"noon": 12, "sunset": 18, "midnight": 0, "sunrise": 6}:
        raise EnvironmentContractError("fixture must name the canonical four M6.4 hours")
    material = data.get("material")
    if not isinstance(material, dict) or material.get("schema") != "c3x.material.v0":
        raise EnvironmentContractError("material must use c3x.material.v0")
    emissive = material.get("emissive")
    if not isinstance(emissive, dict) or emissive.get("activation") not in ACTIVATION_POLICIES:
        raise EnvironmentContractError("material emissive activation is missing or invalid")
    _vector(emissive.get("color"), 3, "material.emissive.color")
    if emissive.get("missing_policy") not in MISSING_POLICIES:
        raise EnvironmentContractError("material emissive missing_policy is invalid")
    lights = data.get("analytic_lights")
    attachments = data.get("ambient_attachments")
    if not isinstance(lights, list) or not lights or not isinstance(attachments, list) or not attachments:
        raise EnvironmentContractError("one analytic light and one ambient attachment are required")
    light_ids = set()
    for index, light in enumerate(lights):
        if light.get("type") not in {"point", "spot", "directional"}:
            raise EnvironmentContractError(f"analytic_lights[{index}].type is invalid")
        _transform(light.get("local_transform"), f"analytic_lights[{index}].local_transform")
        _vector(light.get("color"), 3, f"analytic_lights[{index}].color")
        if not isinstance(light.get("radius"), (int, float)) or light["radius"] <= 0:
            raise EnvironmentContractError(f"analytic_lights[{index}].radius must be positive")
        light_ids.add(light.get("id"))
    for index, item in enumerate(attachments):
        _transform(item.get("local_transform"), f"ambient_attachments[{index}].local_transform")
        bounds = item.get("bounds")
        if not isinstance(bounds, dict):
            raise EnvironmentContractError(f"ambient_attachments[{index}].bounds must be an object")
        _vector(bounds.get("center"), 3, f"ambient_attachments[{index}].bounds.center")
        if not isinstance(bounds.get("radius"), (int, float)) or bounds["radius"] <= 0:
            raise EnvironmentContractError(f"ambient_attachments[{index}].bounds.radius must be positive")
        if item.get("activation") not in ACTIVATION_POLICIES or item.get("missing_policy") not in MISSING_POLICIES:
            raise EnvironmentContractError(f"ambient_attachments[{index}] activation or missing policy is invalid")
        if item.get("light_id") not in light_ids:
            raise EnvironmentContractError(f"ambient_attachments[{index}].light_id is unresolved")
        if item.get("animated") and (not isinstance(item.get("period_ticks"), int) or item["period_ticks"] <= 0):
            raise EnvironmentContractError(f"ambient_attachments[{index}].period_ticks must be positive")
    retained = set(data.get("retained_civ3_layers", []))
    if retained != {"fog", "labels", "selection", "minimap", "hud", "ui"}:
        raise EnvironmentContractError("retained Civ III layer ownership is incomplete")
    return data


def load_fixture(path: Path) -> dict[str, Any]:
    return validate_fixture(json.loads(path.read_text(encoding="utf-8")))
