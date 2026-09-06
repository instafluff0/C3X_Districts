from __future__ import annotations

import math
import unittest
from pathlib import Path

from Renderer.preview.render_textured_patch import BACKGROUND
from Renderer.terrain import production_terrain as terrain


FIXTURE = Path(__file__).resolve().parents[1] / "samples" / "scenes" / "m6_1_terrain.fixture.json"


class ProductionTerrainTests(unittest.TestCase):
    def scene(self, viewport: str = "small", environment: str = "summer_noon"):
        return terrain.load_fixture(FIXTURE, viewport, environment)

    def test_closed_inventory_has_complete_selector_dispositions(self):
        report = terrain.validate_selector_coverage()
        self.assertEqual(report["terrain_types"], 14)
        self.assertEqual(report["contracts"], 11)
        self.assertGreater(report["selector_cells_accounted"], 2000)
        self.assertGreater(report["dispositions"]["mapped"], 0)
        self.assertGreater(report["dispositions"]["m7_vanilla_fallback"], 0)
        self.assertEqual(report["dispositions"]["retained_civ3"], 1)

    def test_two_sizes_are_deterministic_nonblank_and_terrain_only(self):
        renderer = terrain.ProductionTerrainRenderer()
        small = renderer.render(self.scene())
        repeated = renderer.render(self.scene())
        large = renderer.render(self.scene("large_scrolled"))
        self.assertEqual(small.canvas.pixels, repeated.canvas.pixels)
        self.assertGreater(small.canvas.non_background_pixels(BACKGROUND), 5000)
        self.assertGreater(large.canvas.non_background_pixels(BACKGROUND), 5000)
        self.assertEqual(len(small.stats["rendered_ids"]), 16)
        self.assertIn("feature/polar_ice", small.stats["logical_dependencies"]["terrain:0:3:0"])
        self.assertTrue(any("transition/water/shore" in values for values in small.stats["logical_dependencies"].values()))
        self.assertEqual(small.stats["retained_civ3_instance_ids"], ["resource:0:0:0", "city:0:0:0", "unit:0:0:0"])
        self.assertEqual(small.stats["retained_civ3_passes"], 1)
        self.assertTrue(all(owner is None or owner.startswith("terrain:") for owner in small.owner_buffer))

    def test_connected_lattice_wrap_topology_variants_and_anchors_survive_scroll(self):
        renderer = terrain.ProductionTerrainRenderer()
        small_scene = self.scene()
        large_scene = self.scene("large_scrolled")
        small = renderer.render(small_scene)
        large = renderer.render(large_scene)
        self.assertEqual(small.stats["shared_vertices"], 25)
        self.assertLess(small.stats["shared_vertices"], len(small_scene["tiles"]) * 4)
        self.assertTrue(small.stats["topology"]["terrain:0:0:0"]["adjacency_mask"] & 8)
        self.assertTrue(small.stats["topology"]["terrain:3:0:0"]["adjacency_mask"] & 2)
        self.assertEqual(
            {key: value["variant"] for key, value in small.stats["topology"].items()},
            {key: value["variant"] for key, value in large.stats["topology"].items()},
        )
        for tile in large_scene["tiles"]:
            self.assertEqual(large.stats["authoritative_anchors"][tile["terrain"]["id"]], tile["anchor_px"])

    def test_hour_and_season_change_lighting_not_topology(self):
        renderer = terrain.ProductionTerrainRenderer()
        noon = renderer.render(self.scene(environment="summer_noon"))
        night = renderer.render(self.scene(environment="winter_night"))
        spring = renderer.render(self.scene(environment="spring_morning"))
        self.assertNotEqual(noon.canvas.pixels, night.canvas.pixels)
        self.assertNotEqual(noon.canvas.pixels, spring.canvas.pixels)
        self.assertEqual(noon.stats["topology"], night.stats["topology"])
        self.assertGreater(noon.stats["lighting"]["day_factor"], night.stats["lighting"]["day_factor"])

    def test_missing_and_corrupt_assets_fallback_per_item(self):
        renderer = terrain.ProductionTerrainRenderer()
        available = set(terrain.LOGICAL_ASSETS) - {"terrain/grassland/base"}
        frame = renderer.render(
            self.scene(), available_assets=available, corrupt_assets={"terrain/ocean/base"}
        )
        self.assertEqual(set(frame.stats["fallback_ids"]), {"terrain:0:0:0", "terrain:3:0:0", "terrain:2:3:0"})
        self.assertEqual(len(frame.stats["rendered_ids"]), 13)
        self.assertEqual(len(frame.stats["retained_civ3_instance_ids"]), 3)
        self.assertFalse(any(owner in frame.stats["fallback_ids"] for owner in frame.owner_buffer if owner))

    def test_budgets_cache_diagnostics_and_reset_are_bounded(self):
        renderer = terrain.ProductionTerrainRenderer(material_cache_capacity=4, max_visible_tiles=16)
        first = renderer.render(self.scene())
        self.assertLessEqual(first.stats["cache"]["resident"], 4)
        self.assertGreater(first.stats["cache"]["evictions"], 0)
        generation = first.stats["generation"]
        renderer.reset()
        second = renderer.render(self.scene())
        self.assertEqual(second.stats["generation"], generation + 1)
        self.assertLessEqual(second.stats["cache"]["resident"], 4)
        with self.assertRaisesRegex(ValueError, "tile budget exceeded"):
            terrain.ProductionTerrainRenderer(max_visible_tiles=15).render(self.scene())

    def test_map_rect_clipping_and_relief_depth(self):
        frame = terrain.ProductionTerrainRenderer().render(self.scene())
        for y in range(416, 480):
            start = y * 640
            self.assertTrue(all(pixel == BACKGROUND for pixel in frame.canvas.pixels[start : start + 640]))
            self.assertTrue(all(owner is None for owner in frame.owner_buffer[start : start + 640]))
        finite = [value for value in frame.depth_buffer if math.isfinite(value)]
        self.assertTrue(finite)
        self.assertGreater(max(finite), min(finite))
        self.assertGreater(frame.stats["triangles_submitted"], 0)


if __name__ == "__main__":
    unittest.main()
