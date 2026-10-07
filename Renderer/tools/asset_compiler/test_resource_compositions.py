import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Renderer.tools.asset_compiler import build_resource_compositions as compositions
from Renderer.tools.asset_compiler import resource_composition_sources as sources


def placement(asset, kind="model", count=1, scale=1.0, variation=0.0, center=False):
    return {"asset": asset, "kind": kind, "count": count, "scale": scale, "scale_variation": variation,
            "center": center}


SETTING = {"keep_inside": .2, "centre": (.5, .5), "spread": .05, "clusters": 2, "arrangement": "random",
           "cluster_separation": .2, "decal_world_scale": .2, "decal_lift": .01, "world_scale": .16,
           "cluster_radius": .75, "spacing": .8, "sink": .1, "ground_fit": .6, "accent_scale": .5}


class ResourceCompositionTests(unittest.TestCase):
    def test_requested_sources_cover_resources_alternates_decals_and_accents(self):
        profiles = {"resources": {"a": {"source": "R_A", "decal_source": "R_SALT"}},
                    "alternates": {"description": "text", "a": {"x": {"source": "R_B", "accent_source": "R_C"}}}}
        self.assertEqual(sources.requested_sources(profiles), ["R_A", "R_B", "R_C", "R_SALT"])

    def test_variant_conditions_are_generic(self):
        self.assertEqual(sources._condition({"Feature": "FEATURE_JUNGLE", "Terrain": ""}),
                         {"feature": "jungle", "terrain": None, "hills": False})
        self.assertEqual(sources._condition({"Feature": "FEATURE_FOREST", "Terrain": "TERRAIN_TUNDRA_HILLS"}),
                         {"feature": "forest", "terrain": "tundra", "hills": True})

    def test_variant_sets_compare_without_order_or_ancillaries(self):
        base = [placement("m1"), placement("d1", "decal")]
        reordered = [placement("d1", "decal"), placement("m1"), placement(None, "ancillary", 9)]
        self.assertEqual(compositions.effective(base), compositions.effective(reordered))
        self.assertNotEqual(compositions.effective(base), compositions.effective(base[:1]))

    def test_baked_layout_is_deterministic_and_keeps_authored_burial(self):
        meta = {"radius": .5, "height": .2}
        bake = lambda: compositions.bake_variant("k", SETTING, [placement("m1")] * 3, [], model=lambda _: meta,
                                                 decal_cell_ids=None, accent=placement("crystal", scale=2.0))
        first = bake()
        self.assertEqual(first, bake())
        accent, *bodies = first
        # The accent heads the outcrop; every body only adds the profile's extra sink.
        self.assertEqual(accent["model"], "crystal")
        self.assertAlmostEqual(accent["scale"], .16 * .5 * 2.0)
        for body in first:
            self.assertAlmostEqual(body["lift"], -.1 * .2 * body["scale"])
        self.assertEqual(len(bodies), 3)

    def test_cluster_groups_are_centred_and_mountain_feet_curve_with_the_base(self):
        import random
        base = {**SETTING, "spread": 0.0, "keep_inside": 0.05, "cluster_separation": .2}
        flank = compositions.cluster_centres({**base, "arrangement": "flank"}, random.Random(1))
        self.assertAlmostEqual(sum(u - v for u, v in flank), 0.0)       # centred across the view
        self.assertAlmostEqual(sum(u + v for u, v in flank) / 2, 1.0)   # on the layout centre
        scatter = compositions.cluster_centres(base, random.Random(1))
        self.assertAlmostEqual(sum(u for u, _ in scatter) / 2, .5)
        self.assertAlmostEqual(sum(v for _, v in scatter) / 2, .5)
        left, middle, right = compositions.cluster_centres(
            {**base, "arrangement": "flank", "clusters": 3, "arc": .5, "centre": (.8, .8)}, random.Random(1))
        self.assertEqual(middle, (.8, .8))
        self.assertAlmostEqual(left[0] - left[1], -(right[0] - right[1]))   # symmetric across the view
        self.assertAlmostEqual(left[0] + left[1], right[0] + right[1])
        self.assertLess(left[0] + left[1], 1.6)                             # outer clusters rise toward the mountain

    def test_every_cluster_has_a_decal_under_its_rocks(self):
        setting = {**SETTING, "arrangement": "flank", "clusters": 3, "centre": (.8, .8), "keep_inside": .05,
                   "spread": 0.0, "cluster_weights": [.3, .4, .3], "cluster_scales": [.8, 1.0, .8], "decal_scale": .85}
        decals = [placement("d1", "decal", scale=1.0), placement("d2", "decal", scale=1.0)]
        layout = compositions.bake_variant("m", setting, [placement("m1")] * 10, decals,
                                           model=lambda _: {"radius": .03, "height": .1},
                                           decal_cell_ids=lambda _: [("t", 0)])
        under = [item for item in layout if "decal" in item]
        rocks = [item for item in layout if "model" in item]
        centres = compositions.cluster_centres(setting, __import__("random").Random(0))
        self.assertEqual(len(under), 3)
        self.assertEqual(sorted((round(d["u"], 6), round(d["v"], 6)) for d in under),
                         sorted((round(u, 6), round(v, 6)) for u, v in centres))
        self.assertGreater(under[1]["scale"], under[0]["scale"])
        for rock in rocks:
            self.assertTrue(any(math.hypot(rock["u"] - d["u"], rock["v"] - d["v"]) <= d["scale"] for d in under))

    def test_planted_rows_run_along_u_centred_on_the_layout(self):
        setting = {**SETTING, "planting": "rows", "rows": 3, "row_spacing": .1, "row_length": .3, "spread": 0.0,
                   "keep_inside": .05, "arrangement": "flank"}
        layout = compositions.bake_variant("vines", setting, [placement("vine")] * 9, [],
                                           model=lambda _: {"radius": .03, "height": .1}, decal_cell_ids=None)
        rows = sorted({round(item["v"], 1) for item in layout})
        self.assertEqual(rows, [.4, .5, .6])                       # three rows across v
        for row in rows:
            us = sorted(item["u"] for item in layout if round(item["v"], 1) == row)
            self.assertEqual(len(us), 3)
            self.assertLess(us[-1] - us[0], .3)                    # spread along u within the row length
        self.assertAlmostEqual(sum(item["u"] for item in layout) / 9, .5, delta=.02)

    def test_herd_groups_animated_subjects_at_the_layout_centre(self):
        setting = {**SETTING, "subject": "horses", "herd": 3, "subject_radius": .06, "herd_spread": 1.3,
                   "yaw_jitter": .5, "subject_fit": .6, "spread": 0.0, "arrangement": "flank"}
        registered = []
        layout = compositions.bake_variant("herd", setting, [], [],
                                           model=lambda asset: registered.append(asset) or {"radius": 0, "height": 0},
                                           decal_cell_ids=None)
        self.assertIn("animated/horses", registered)
        self.assertEqual([item["model"] for item in layout], ["animated/horses"] * 3)
        u = sum(item["u"] for item in layout) / 3
        v = sum(item["v"] for item in layout) / 3
        self.assertAlmostEqual(u, .5, delta=.03)
        self.assertAlmostEqual(v, .5, delta=.03)
        for item in layout:
            self.assertLessEqual(abs(item["rotation"]), .5)
            self.assertAlmostEqual(item["ground_fit"], .06 * item["scale"] * .6)
            self.assertAlmostEqual(math.hypot(item["u"] - .5, item["v"] - .5), .06 * 1.3, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
