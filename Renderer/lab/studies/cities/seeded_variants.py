"""Offline, deterministic house substitutions for city recipe auditions.

Seed zero is the approved comparison layout. Other seeds keep the civic core,
skyline accents, plot centers, and population counts while changing a few
ordinary houses within the same source-art family. Runtime only needs to pick
one compiled generic recipe by stable city identity after promotion.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import footprint


def _box(item: dict) -> list[float]:
    body = component(item["asset"], Path(item["pack"]))
    return footprint({"low": body["lo"], "high": body["hi"]}, item)


def _candidate(old: dict, other: dict) -> dict | None:
    if old["asset"] == other["asset"] or other.get("skyline_role"):
        return None
    old_body = component(old["asset"], Path(old["pack"]))
    new_body = component(other["asset"], Path(other["pack"]))
    old_height = (old_body["hi"][2] - old_body["lo"][2]) * old["scale"]
    new_height = new_body["hi"][2] - new_body["lo"][2]
    if new_height <= 0:
        return None
    scale = old_height / new_height
    if not .67 <= scale / other["scale"] <= 1.5:
        return None
    return {**old, "asset": other["asset"], "pack": other["pack"],
            "scale": round(scale, 6), "rotation": other["rotation"]}


def _vary(houses: list[dict], core: dict, slots: range, size: int,
          rng: random.Random) -> int:
    alternatives = [item for item in houses if not item.get("skyline_role")]
    targets = [index for index in slots if not houses[index].get("skyline_role")]
    rng.shuffle(targets)
    changed = 0
    for index in targets:
        if changed >= (1, 2, 3)[size]:
            break
        original = houses[index]
        old_box = _box(original)
        choices = alternatives[:]
        rng.shuffle(choices)
        for source in choices:
            item = _candidate(original, source)
            if item is None:
                continue
            bounds = _box(item)
            # Keep the replacement on its original plot. The small allowance
            # permits a different roof outline without enlarging city sprawl.
            if (bounds[0] < old_box[0] - .06 or bounds[1] < old_box[1] - .06 or
                    bounds[2] > old_box[2] + .06 or bounds[3] > old_box[3] + .06):
                continue
            if size == 0 and any(abs(edge) > .53 for edge in bounds):
                continue
            occupied = [_box(core)] + [_box(other) for slot, other in enumerate(houses)
                                       if slot != index]
            if any(overlaps(bounds, box) for box in occupied):
                continue
            houses[index] = item
            changed += 1
            break
    return changed


def apply_house_variation(design: dict, variation_seed: int) -> dict[str, list[int]]:
    """Vary ordinary houses; preserve each authored plot and all special art."""
    counts: dict[str, list[int]] = {}
    if variation_seed == 0:
        return counts
    tiers = design["tier_designs"]
    identity = f"{design['culture_name']}|{design.get('source_art_era', design['era_name'])}"
    for key in ("houses", "capital_houses"):
        if not all(key in tier for tier in tiers):
            continue
        original = [tier[key] for tier in tiers]
        growing = all(original[size + 1][:len(original[size])] == original[size]
                      for size in (0, 1))
        changed_by_size = []
        if growing:
            houses = [item.copy() for item in original[2]]
            previous = 0
            for size, tier in enumerate(tiers):
                end = len(original[size])
                seed = hashlib.sha256(
                    f"{identity}|{key}|{size}|{variation_seed}".encode()).digest()
                core = tier["palace"] if key == "capital_houses" else tier["base_centerpiece"]
                changed_by_size.append(_vary(houses, core, range(previous, end),
                                             size, random.Random(int.from_bytes(seed[:8], "big"))))
                previous = end
            for size, tier in enumerate(tiers):
                tier[key] = [item.copy() for item in houses[:len(original[size])]]
        else:
            for size, tier in enumerate(tiers):
                houses = [item.copy() for item in tier[key]]
                seed = hashlib.sha256(
                    f"{identity}|{key}|{size}|{variation_seed}".encode()).digest()
                core = tier["palace"] if key == "capital_houses" else tier["base_centerpiece"]
                changed_by_size.append(_vary(houses, core, range(len(houses)),
                                             size, random.Random(int.from_bytes(seed[:8], "big"))))
                tier[key] = houses
        counts[key] = changed_by_size
    design["houses"] = tiers[2]["houses"]
    design["variation_seed"] = variation_seed
    return counts
