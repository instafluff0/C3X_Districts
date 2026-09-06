"""Q2-owned fixture preparation; no renderer or shared-contract replacement.

All cases are synthetic Lab inputs, not captured game state. The 14 baseline
families come from the inventory. Supplementary base/real combinations are
conservative stress cases, not assertions that every combination is reachable.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OWNED = ROOT / "Renderer/terrain_lab/v2/fixtures/terrain"
INVENTORY = ROOT / "Renderer/inventory/vanilla_conquests_biq_semantics.json"
BASE = {4: 0, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}


def policy(a: int, b: int) -> dict:
    """Proposed base transition; external surfaces retain their own ownership."""
    if (a >= 11) != (b >= 11):
        return {"rule": "shore-mediated", "requires": ["Q3-hydrology"]}
    if a >= 11 and b >= 11:
        return {"rule": "smooth", "requires": ["Q3-hydrology"]}
    if a in (5, 6, 10) or b in (5, 6, 10):
        return {"rule": "shoulder", "requires": ["Q4-relief"]}
    return {"rule": "smooth", "requires": []}


def matrix() -> dict:
    families = json.loads(INVENTORY.read_text())["terrain_types"]
    codes = [f["biq_index"] for f in families]
    pairs = [{"families": [a, b], **policy(a, b)}
             for a, b in itertools.combinations_with_replacement(codes, 2)]
    return {
        "schema": "c3x.q2.terrain_fixture_recipe.v1",
        "state": "prepared_not_rendered",
        "provenance": "synthetic_lab_only",
        "inventory": INVENTORY.relative_to(ROOT).as_posix(),
        "families": [{"code": f["biq_index"], "name": f["name"],
                      "base": BASE.get(f["biq_index"], f["biq_index"])} for f in families],
        "pairs": pairs,
        "axes": {"column": [1, 1], "row": [1, -1]},
        "reversed": [False, True],
        "wrap_origin_x": [-2, 98, 198],
        "map_size": [100, 100],
        "visible_size": [4, 4],
        "halo": 2,
        "three_way": [[0, 1, 2], [0, 2, 3], [2, 9, 11], [2, 5, 6], [11, 12, 13]],
        "base_stress": [{"real": r, "base": b} for r in (5, 6, 7, 8, 10) for b in range(4)],
        "zoom": [1.0, 0.5],
        "hours": [12, 18, 0, 6],
        "controls": ["complete", "base_only", "weights", "height", "normal", "roughness",
                     "detail_off", "detail_on", "no_clutter", "no_cast_shadows"],
        "scroll_pixels": [[0, 0], [1, 0], [0, 1], [16, 8]],
        "gates": {
            "edge_height_normal_weights": "same physical sample from both incident tiles; max absolute delta <= 1e-6",
            "wrap": "same canonical sample across all three aliases; max absolute delta <= 1e-6",
            "visual": "direct unsharpened final-size inspection; no cracks, diamonds, speckling or sparkle",
            "detail": "measure albedo, height, normal and roughness controls independently; macro identity retained",
            "scope": "fixture coverage is not rendered acceptance; pair rules are proposed until composition proves them",
        },
    }


def cases():
    spec = matrix()
    for pair, axis, reverse, origin in itertools.product(
            spec["pairs"], spec["axes"], spec["reversed"], spec["wrap_origin_x"]):
        a, b = pair["families"]
        yield {"id": f"pair_{a:02}_{b:02}_{axis}_r{int(reverse)}_x{origin}",
               "families": [a, b], "axis": axis, "reverse": reverse, "origin": origin}
    for trio, axis, reverse, origin in itertools.product(
            spec["three_way"], spec["axes"], spec["reversed"], spec["wrap_origin_x"]):
        yield {"id": f"junction_{'_'.join(map(str, trio))}_{axis}_r{int(reverse)}_x{origin}",
               "families": trio, "axis": axis, "reverse": reverse, "origin": origin}
    for state in spec["base_stress"]:
        yield {"id": f"base_{state['base']}_real_{state['real']}",
               "families": [state["real"], 3], "axis": "column", "reverse": False,
               "origin": 98, "base_override": state["base"]}


def csv_fixture(case: dict) -> str:
    """Existing BIQ-window V2 syntax with two-cell halo and canonical raw X.

    Alias origins intentionally canonicalize to identical captured coordinates.
    Raw (unwrapped) origin stays in the recipe for SurfaceSample contract tests.
    """
    records = []
    for halo in (False, True):
        for row in range(-2, 6):
            for column in range(-2, 6):
                if (not (0 <= row < 4 and 0 <= column < 4)) != halo:
                    continue
                primary, secondary = (column, row) if case["axis"] == "column" else (row, column)
                side = int(primary >= 2) ^ int(case["reverse"])
                family = case["families"][side]
                if len(case["families"]) == 3 and secondary >= 2:
                    family = case["families"][2]
                base = BASE.get(family, family)
                if side == 0 and "base_override" in case:
                    base = case["base_override"]
                x = (case["origin"] + column + row) % 100
                y = 50 + column - row
                records.append(f"{column},{row},{x},{y},{base},{family},0,0,0")
    return "C3X_BIQ_TERRAIN_WINDOW_V2,4,4,16,0,0,100,100,48\n" + "\n".join(records) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="emit a named case beneath the Q2 fixture directory")
    args = parser.parse_args()
    OWNED.mkdir(parents=True, exist_ok=True)
    if args.case:
        case = next((c for c in cases() if c["id"] == args.case), None)
        if case is None:
            parser.error("unknown case")
        target = OWNED / (case["id"] + ".csv")
        target.write_text(csv_fixture(case))
    else:
        target = OWNED / "matrix_v1.json"
        target.write_text(json.dumps(matrix(), indent=2) + "\n")
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
