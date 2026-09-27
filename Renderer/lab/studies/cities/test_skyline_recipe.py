"""Check seeded city accents and vegetation for every late source family."""

import json
import math
import unittest
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.medieval_family_review import generate
from Renderer.lab.studies.cities.skyline_recipe import _box, apply_skyline, palette
from Renderer.lab.studies.cities.build_layouts import ROOT


OUT = ROOT / "Renderer/lab/out/cities/all-era-source-auditions"


class SkylineRecipeTests(unittest.TestCase):
    def test_skyscrapers_come_from_modern_glass_family(self):
        report = json.loads((OUT / "modern" / "source-report.json").read_text())
        glass = {record["asset_id"] for pool in report["pools"]
                 if pool["source_culture"] == "ModernGlass"
                 for record in pool["selected"]}
        regular, towers = palette()
        self.assertTrue(regular)
        self.assertTrue(towers)
        self.assertLessEqual(set(towers), glass)

    def test_all_late_source_families_have_seeded_collision_free_accents(self):
        palaces = json.loads((ROOT / "Renderer/packs/CityPalacesNormalized/palace_catalog.json").read_text())
        tree_pack = ROOT / "Renderer/lab/out/cities/medieval-art/farm-tree-pack"
        signatures = []
        for era in ("industrial", "modern"):
            # Both Civ III late eras retain a culture's Industrial outer
            # houses; Modern adds glass towers through the composition.
            report_path, pack = (OUT / "industrial" / "source-report.json",
                                 OUT / "industrial" / "foundation-free-pack")
            report = json.loads(report_path.read_text())
            for pool in report["pools"]:
                variants = []
                for seed in range(6):
                    palace = None
                    for culture, art_era in ((pool["source_culture"], pool["source_art_era"]),
                                             (pool["source_culture"], "DEFAULT"),
                                             ("DEFAULT", "DEFAULT")):
                        matches = [record["asset_id"] for record in palaces["palaces"]
                                   if any(selector["culture"] == culture and
                                          selector["era"] == art_era
                                          for selector in record["source_selectors"])]
                        if len(matches) == 1:
                            palace = matches[0]
                            break
                    self.assertIsNotNone(palace)
                    layouts, _ = generate(pool, pack, OUT / "flat-palaces",
                                          palace, tree_pack, use_curated=False,
                                          variation_seed=seed)
                    design = next(item for item in layouts["designs"]
                                  if item.get("source_art_era") == pool["source_art_era"])
                    apply_skyline(design, era, seed)
                    for key in ("houses", "capital_houses"):
                        for size, tier in enumerate(design["tier_designs"]):
                            houses = tier[key]
                            roles = [item.get("skyline_role") for item in houses]
                            self.assertEqual(roles.count("modern_infill"),
                                             (0, 2, 4)[size])
                            self.assertEqual(roles.count("skyscraper"),
                                             ((0, 1, 2)[size] if key == "capital_houses" else
                                              (0, 0, 1)[size]) if era == "modern" else 0)
                            accents = [item for item in houses if item.get("skyline_role")]
                            if key == "houses" and era == "modern" and size:
                                accents.append(tier["base_centerpiece"])
                                self.assertEqual(tier["base_centerpiece"].get("skyline_role"),
                                                 "skyscraper")
                            self.assertEqual(len({item["asset"] for item in accents}),
                                             len(accents), (era, pool["source_culture"],
                                                            seed, key, size))
                            if key == "capital_houses":
                                px, py = tier["palace"]["offset"]
                                for item in accents:
                                    if item.get("skyline_role") != "skyscraper":
                                        continue
                                    dx = item["offset"][0] - px
                                    dy = item["offset"][1] - py
                                    up = -(dx + dy) / 2
                                    self.assertGreaterEqual(up, .10)
                                    self.assertLessEqual(abs(dx - dy) / 2, 1.73 * up)
                            if size == 1:
                                for item in houses:
                                    if item.get("skyline_role"):
                                        self.assertLessEqual(math.hypot(*item["offset"]), .56,
                                                             (era, pool["source_culture"], seed, key))
                            for item in houses:
                                role = item.get("skyline_role")
                                if role:
                                    body = component(item["asset"], Path(item["pack"]))
                                    height = (body["hi"][2] - body["lo"][2]) * item["scale"]
                                    self.assertGreaterEqual(height + .001,
                                                            .28 if role == "modern_infill" else .48)
                            occupied = [_box(tier["palace"] if key == "capital_houses"
                                             else tier["base_centerpiece"])]
                            for item in houses:
                                box = _box(item)
                                self.assertFalse(any(overlaps(box, other) for other in occupied),
                                                 (era, pool["source_culture"], seed, size))
                                occupied.append(box)
                        city = [item for item in design["tier_designs"][1][key]
                                if item.get("skyline_role")]
                        metro = design["tier_designs"][2][key]
                        self.assertTrue(all(item in metro for item in city))
                    tree_offsets = tuple(tuple(item["offset"])
                                         for item in design["tier_designs"][2]["decorations"])
                    self.assertGreaterEqual(len(tree_offsets), 2)
                    variants.append((tuple((item["asset"], tuple(item["offset"]))
                                           for item in design["tier_designs"][2]["houses"]
                                           if item.get("skyline_role")), tree_offsets))
                self.assertNotEqual(variants[0], variants[1])
                signatures.append(variants[0])
        self.assertEqual(len(signatures), 6)


if __name__ == "__main__":
    unittest.main()
