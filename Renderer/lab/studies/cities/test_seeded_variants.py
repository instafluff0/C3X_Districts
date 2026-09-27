"""Regression checks for the three review-only city recipe candidates."""

import json
import unittest

from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import ROOT
from Renderer.lab.studies.cities.seeded_variants import _box, apply_house_variation


OUT = ROOT / "Renderer/lab/out/cities/all-era-source-auditions"


def _design(path, era, source_era):
    layouts = json.loads(path.read_text(encoding="utf-8"))
    if era == "classical":
        return next(item for item in layouts["designs"]
                    if (item["culture"], item["era"]) == (1, 1))
    return next(item for item in layouts["designs"]
                if item.get("source_art_era") == source_era)


def _collisions(tier, key):
    center = tier["palace"] if key == "capital_houses" else tier["base_centerpiece"]
    boxes = [_box(center)] + [_box(item) for item in tier[key]]
    return {(first, second) for first in range(len(boxes))
            for second in range(first + 1, len(boxes))
            if overlaps(boxes[first], boxes[second])}


class SeededCityVariantTests(unittest.TestCase):
    def test_all_source_pairs_keep_three_distinct_safe_candidates(self):
        manifest = json.loads((OUT / "review/seeded-variant-manifest.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["source_pairs"]), 46)
        self.assertEqual(len(manifest["civ3_candidates"]), 20)
        self.assertEqual(manifest["candidate_seeds"], [0, 1, 2])
        for mapped in manifest["civ3_candidates"]:
            self.assertEqual([entry["seed"] for entry in mapped["variants"]], [0, 1, 2])
            if mapped["era"] == "modern":
                for entry in mapped["variants"]:
                    design = json.loads((OUT / entry["design"]).read_text(encoding="utf-8"))
                    self.assertEqual(design["target_civ3_era"], "modern")
                    for key in ("houses", "capital_houses"):
                        for tier in design["tier_designs"][1:]:
                            accents = [item["asset"] for item in tier[key]
                                       if item.get("skyline_role")]
                            if key == "houses":
                                accents.append(tier["base_centerpiece"]["asset"])
                            self.assertEqual(len(accents), len(set(accents)))
            else:
                self.assertTrue(all((OUT / entry["layout"]).is_file()
                                    for entry in mapped["variants"]))
        for pair in manifest["source_pairs"]:
            designs = [_design(OUT / entry["layout"], pair["era_audition"],
                               pair["source_art_era"]) for entry in pair["variants"]]
            base = designs[0]
            for seed, design in enumerate(designs[1:], 1):
                self.assertEqual(design["variation_seed"], seed)
                for key in ("houses", "capital_houses"):
                    self.assertTrue(all(amount > 0 for amount in
                                        pair["variants"][seed]["house_changes"][key]))
                    for size, (before, after) in enumerate(zip(base["tier_designs"],
                                                             design["tier_designs"])):
                        self.assertEqual(len(before[key]), len(after[key]))
                        self.assertEqual([item["offset"] for item in before[key]],
                                         [item["offset"] for item in after[key]])
                        self.assertEqual(before["palace"], after["palace"])
                        self.assertLessEqual(_collisions(after, key),
                                             _collisions(before, key),
                                             (pair["era_audition"], pair["source_culture"],
                                              seed, size, key))
                        self.assertNotEqual([item["asset"] for item in before[key]],
                                            [item["asset"] for item in after[key]])

    def test_same_seed_repeats_the_same_house_recipe(self):
        source = OUT / "review/classical/mediterranean/layouts.json"
        first = _design(source, "classical", "ARTERA_CLASSICAL")
        second = _design(source, "classical", "ARTERA_CLASSICAL")
        self.assertEqual(apply_house_variation(first, 1),
                         apply_house_variation(second, 1))
        self.assertEqual(first["tier_designs"], second["tier_designs"])


if __name__ == "__main__":
    unittest.main()
