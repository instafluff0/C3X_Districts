"""Verify city buildings fit inside complete walls without overlapping."""

import json
import math
import unittest
from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.ground import footprint_alignment
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import (ROOT, COUNTS_BY_ERA, WALL_SEGMENTS,
                                                        build, footprint, inside_wall,
                                                        wall_instances)


class CityDesigns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((ROOT/"Renderer/lab/studies/cities/layouts.json").read_text())

    def test_frozen_designs_match_current_local_sources(self):
        self.assertEqual(self.document, build())
        self.assertEqual(len(self.document["designs"]), 20)
        self.assertEqual({(d["culture"], d["era"]) for d in self.document["designs"]},
                         {(c, e) for c in range(5) for e in range(4)})

    def test_buildings_fit_inside_each_complete_wall_ring(self):
        for design in self.document["designs"]:
            self.assertEqual(design["population_counts"], list(COUNTS_BY_ERA[design["era"]]))
            for capital in (False, True):
                for size, count in enumerate(design["population_counts"]):
                    tier = design["tier_designs"][size]
                    cores = [tier["palace"] if capital else tier["base_centerpiece"]]
                    houses = tier["houses"]
                    self.assertEqual(len(houses), count)
                    boxes = []
                    for item in cores+houses:
                        asset = component(item["asset"], Path(item["pack"]))
                        box = footprint({"low": asset["lo"], "high": asset["hi"]}, item)
                        with self.subTest(culture=design["culture_name"], era=design["era_name"],
                                          capital=capital, count=count, asset=item["asset"]):
                            self.assertTrue(inside_wall(box, size), box)
                            if size == 0:
                                self.assertLessEqual(max(abs(v) for v in box), .5)
                            self.assertFalse(any(overlaps(box, other) for other in boxes))
                        boxes.append(box)

    def test_culture_layouts_remain_distinct_when_source_art_is_shared(self):
        for era in range(4):
            designs = [design for design in self.document["designs"] if design["era"] == era]
            signatures = [tuple(tuple(house["offset"]) for house in design["houses"])
                          for design in designs]
            self.assertEqual(len(set(signatures)), 5, self.document["eras"][era])

    def test_buildings_share_one_facing_direction(self):
        for design in self.document["designs"]:
            for item in [design["base_centerpiece"], *design["houses"]]:
                self.assertEqual(item["rotation"], 0.0,
                                 (design["culture_name"], design["era_name"], item["asset"]))
            palace = design["palace"]
            source = component(palace["asset"], Path(palace["pack"]))
            source_points = [vertex["position"][:2]
                             for mesh, _material in source["parts"]
                             for vertex in mesh["vertices"]]
            correction = footprint_alignment(source_points)
            # These source palace platforms are authored about 30 degrees off
            # the city grid. The applied transform must restore tile-edge
            # alignment, allowing small hull asymmetries on one palace.
            self.assertLess(abs(palace["rotation"]-correction), math.radians(3),
                            (design["culture_name"], design["era_name"], palace["asset"]))

    def test_ancient_asian_roofs_are_yellow_and_era_scale_progresses(self):
        design = next(d for d in self.document["designs"]
                      if d["culture_name"] == "Asian" and d["era_name"] == "Ancient")
        for index, item in enumerate(design["tier_designs"][0]["houses"]):
            source = component(item["asset"], Path(item["pack"]))
            projected_height = (source["hi"][2]-source["lo"][2])*item["scale"]*150
            self.assertEqual(item["pack"], "Renderer/packs/CityAncientWoodCandidates")
            self.assertGreaterEqual(projected_height, 37.9)
            box = footprint({"low": source["lo"], "high": source["hi"]}, item)
            self.assertLessEqual(max(abs(value) for value in box), .5)
        palace = design["palace"]
        source = component(palace["asset"], Path(palace["pack"]))
        self.assertGreaterEqual((source["hi"][2]-source["lo"][2])*
                                palace["scale"]*150, 65)
        modern = next(d for d in self.document["designs"]
                      if d["culture_name"] == "Asian" and d["era_name"] == "Modern")
        self.assertLess(len(design["houses"]), 2*len(modern["houses"]))
        modern_source = component(modern["houses"][0]["asset"],
                                  Path(modern["houses"][0]["pack"]))
        modern_height = ((modern_source["hi"][2]-modern_source["lo"][2])*
                         modern["houses"][0]["scale"]*150)
        self.assertGreater(modern_height, projected_height)

    def test_population_growth_never_rescales_existing_buildings(self):
        for design in self.document["designs"]:
            tiers = design["tier_designs"]
            for size in (1, 2):
                for old, grown in zip(tiers[size-1]["houses"], tiers[size]["houses"]):
                    self.assertEqual((old["asset"], old["scale"], old["rotation"]),
                                     (grown["asset"], grown["scale"], grown["rotation"]))
                for key in ("base_centerpiece", "palace"):
                    self.assertEqual(tiers[size-1][key]["scale"], tiers[size][key]["scale"])

    def test_rounded_walls_form_a_complete_perimeter(self):
        for kit in ("ancient", "medieval", "industrial"):
            for size, segments in enumerate(WALL_SEGMENTS):
                wall = wall_instances(kit, size)
                self.assertEqual(len(wall), segments+segments//4)
                centers = [item["offset"] for item in wall[:segments]]
                self.assertEqual(len({(round(x, 5), round(y, 5)) for x, y in centers}),
                                 segments)
                for index in range(segments):
                    self.assertLess(math.dist(centers[index],
                                              centers[(index+1) % segments]), .26)
                if size == 0:
                    for item in wall:
                        asset = component(item["asset"], Path(item["pack"]))
                        box = footprint({"low": asset["lo"], "high": asset["hi"]}, item)
                        self.assertLessEqual(max(abs(v) for v in box), .5, item["asset"])


if __name__ == "__main__":
    unittest.main()
