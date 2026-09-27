"""Deterministic, source-art skyline accents for Civ III Industrial/Modern cities.

Only the offline recipe compiler uses Civ VI ArtEra selectors. Runtime receives
ordinary normalized instances and never branches on a source civilization.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps


ROOT = Path(__file__).resolve().parents[4]
MODERN_PACK = Path("Renderer/lab/out/cities/all-era-source-auditions/modern/foundation-free-pack")
MODERN_REPORT = ROOT / "Renderer/lab/out/cities/all-era-source-auditions/modern/source-report.json"
ACCENTS = {"industrial": ((2, 0), (4, 0)),
           "modern": ((2, 1), (4, 2))}


def palette() -> tuple[list[str], list[str]]:
    """Use individual modern houses and high-rise blocks, never baked city slabs."""
    report = json.loads(MODERN_REPORT.read_text(encoding="utf-8"))
    regular, towers = [], []
    for pool in report["pools"]:
        for record in pool["selected"]:
            entry = record["entry"]
            if pool["source_culture"] == "DEFAULT" and "_Bld_" in entry:
                body = component(record["asset_id"], MODERN_PACK)
                if body["hi"][2] - body["lo"][2] < .19:
                    regular.append(record["asset_id"])
            if pool["source_culture"] == "ModernGlass" and "_Block_SQ_" in entry:
                body = component(record["asset_id"], MODERN_PACK)
                if body["hi"][2] - body["lo"][2] >= .19:
                    towers.append(record["asset_id"])
    if len(regular) < 4 or len(towers) < 3:
        raise ValueError("Modern source palette lacks ordinary or high-rise buildings")
    return sorted(set(regular)), sorted(set(towers))


def _box(item: dict) -> list[float]:
    # The shared footprint helper applies the same source-centering convention
    # as the sheet and native composition compiler.
    from Renderer.lab.studies.cities.build_layouts import footprint
    body = component(item["asset"], Path(item["pack"]))
    return footprint({"low": body["lo"], "high": body["hi"]}, item)


def _replacement(old: dict, role: str, candidates: list[str],
                 occupied: list[list[float]], rng: random.Random) -> dict | None:
    if math.hypot(*old["offset"]) > .56:
        return None
    alternatives = candidates[:]
    rng.shuffle(alternatives)
    for asset in alternatives:
        body = component(asset, MODERN_PACK)
        dz = body["hi"][2] - body["lo"][2]
        desired_height = .46 if role == "modern_infill" else .68
        minimum_height = .28 if role == "modern_infill" else .48
        for step in range(13):
            height = desired_height - step * (desired_height - minimum_height) / 12
            item = {"asset": asset, "pack": MODERN_PACK.as_posix(),
                    "scale": round(height / dz, 6), "rotation": 0.0,
                    "offset": old["offset"][:], "skyline_role": role}
            box = _box(item)
            if (any(abs(edge) > .70 for edge in box) or
                    any(overlaps(box, other) for other in occupied)):
                continue
            return item
    return None


def _behind_palace(offset: list[float], palace: list[float]) -> bool:
    dx, dy = offset[0] - palace[0], offset[1] - palace[1]
    up = -(dx + dy) / 2
    return up >= .10 and abs(dx - dy) / 2 <= 1.73 * up


def _capital_tower_pair(tiers: list[dict], towers: list[str],
                        rng: random.Random) -> tuple[tuple[int, dict], tuple[int, dict]]:
    """Reserve a compatible City/Metropolis pair before filling nearby plots."""
    metro = tiers[2]["capital_houses"]
    palace = tiers[2]["palace"]
    city_limit = len(tiers[1]["capital_houses"])
    slots = [index for index, item in enumerate(metro)
             if _behind_palace(item["offset"], palace["offset"])]
    first_slots = [index for index in slots if index < city_limit]
    rng.shuffle(first_slots)
    for first_slot in first_slots:
        first_assets = towers[:]
        rng.shuffle(first_assets)
        for asset in first_assets:
            occupied = [_box(item) for index, item in enumerate(metro)
                        if index != first_slot] + [_box(palace)]
            first = _replacement(metro[first_slot], "skyscraper", [asset], occupied, rng)
            if first is None:
                continue
            second_slots = [index for index in slots if index != first_slot]
            rng.shuffle(second_slots)
            for second_slot in second_slots:
                occupied = [_box(item) for index, item in enumerate(metro)
                            if index not in (first_slot, second_slot)]
                occupied.extend((_box(palace), _box(first)))
                second = _replacement(metro[second_slot], "skyscraper",
                                      [candidate for candidate in towers if candidate != asset],
                                      occupied, rng)
                if second is not None:
                    return (first_slot, first), (second_slot, second)
    raise ValueError("No compatible glass plots behind the palace")


def apply_skyline(design: dict, target_era: str, variation_seed: int = 0) -> None:
    """Swap inner growth plots; Towns stay intact and Metropolises add more."""
    if target_era not in ACCENTS:
        return
    regular, towers = palette()
    tiers = design["tier_designs"]
    identity = design.get("source_culture", design["culture_name"])
    if target_era == "modern":
        # The ordinary city has no palace. Its glass landmark owns the center
        # plot, rather than hovering beside the old civic building.
        city = tiers[1]
        seed = int.from_bytes(hashlib.sha256(
            f"{identity}|{target_era}|center|{variation_seed}".encode()).digest()[:8], "big")
        occupied = [_box(item) for item in tiers[2]["houses"]]
        landmark = _replacement(city["base_centerpiece"], "skyscraper", towers,
                                occupied, random.Random(seed))
        if landmark is None:
            raise ValueError(f"No central glass plot for {identity} {target_era}")
        for size in (1, 2):
            tiers[size]["base_centerpiece"] = landmark.copy()
    for key in ("houses", "capital_houses"):
        if key == "capital_houses" and not all(key in tier for tier in tiers):
            continue
        planned = None
        if key == "capital_houses" and target_era == "modern":
            seed = int.from_bytes(hashlib.sha256(
                f"{identity}|{target_era}|capital-pair|{variation_seed}".encode()).digest()[:8], "big")
            planned = _capital_tower_pair(tiers, towers, random.Random(seed))
        for size in (1, 2):
            houses = list(tiers[size][key])
            if size == 2:
                houses[:len(tiers[1][key])] = tiers[1][key]
            if planned is not None:
                slot, tower = planned[size - 1]
                houses[slot] = tower
            modern_total, sky_total = ACCENTS[target_era][size - 1]
            previous_modern = ACCENTS[target_era][size - 2][0] if size == 2 else 0
            previous_sky = ACCENTS[target_era][size - 2][1] if size == 2 else 0
            if key == "houses" and target_era == "modern" and size == 1:
                previous_sky = 1  # The replacement centerpiece is the first tower.
            if planned is not None:
                previous_sky = sky_total  # This tier's reserved tower is already placed.
            # Put the glass towers on the most central available plots first,
            # then mingle ordinary modern buildings with nearby local houses.
            roles = (["skyscraper"] * (sky_total - previous_sky) +
                     ["modern_infill"] * (modern_total - previous_modern))
            center = (tiers[size]["palace"] if key == "capital_houses" else
                      tiers[size]["base_centerpiece"])
            seed = int.from_bytes(hashlib.sha256(
                f"{identity}|{target_era}|{key}|{size}|{variation_seed}".encode()).digest()[:8], "big")
            rng = random.Random(seed)
            for role in roles:
                towers_here = [item["offset"] for item in houses
                               if item.get("skyline_role") == "skyscraper"]
                if key == "houses" and target_era == "modern":
                    towers_here.append(center["offset"])
                eligible = [index for index, item in enumerate(houses)
                            if not item.get("skyline_role")]
                if key == "capital_houses" and target_era == "modern":
                    # In screen space, the palace's rear arc spans roughly
                    # ten to two o'clock. Reserve it for glass towers.
                    eligible = [slot for slot in eligible if
                                _behind_palace(houses[slot]["offset"], center["offset"])
                                == (role == "skyscraper")]
                def downtown_score(slot):
                    x, y = houses[slot]["offset"]
                    radius = (x*x + y*y) ** .5
                    if not towers_here:
                        return radius, slot
                    separation = min(((x-tx)**2 + (y-ty)**2) ** .5
                                     for tx, ty in towers_here)
                    return .65*radius + .35*separation, slot
                eligible.sort(key=downtown_score)
                core = eligible[:5 if size == 1 or role == "skyscraper" else 8]
                fringe = eligible[len(core):]
                rng.shuffle(core)
                used = {item["asset"] for item in houses if item.get("skyline_role")}
                if center.get("skyline_role"):
                    used.add(center["asset"])
                candidates = [asset for asset in
                              (regular if role == "modern_infill" else towers)
                              if asset not in used]
                for slot in core + fringe:
                    occupied = [_box(item) for index, item in enumerate(houses)
                                if index != slot] + [_box(center)]
                    if size == 1:
                        occupied.extend(_box(item) for item in
                                        tiers[2][key][len(houses):])
                        if planned is not None:
                            occupied.append(_box(planned[1][1]))
                    replacement = _replacement(houses[slot], role, candidates,
                                               occupied, rng)
                    if replacement:
                        houses[slot] = replacement
                        break
                else:
                    raise ValueError(f"No {role} plot for {identity} {target_era} {key} tier {size} seed {variation_seed}")
            tiers[size][key] = houses
    design["houses"] = tiers[2]["houses"]
    design["population_counts"] = [len(tier["houses"]) for tier in tiers]
