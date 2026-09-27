from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler.wall_mesh_filter import (
    remove_uv_island_components,
    trim_skirt_and_ground,
)


def vertex(x: float, z: float, u: float, v: float) -> dict:
    return {"position": [x, 0.0, z], "normal": [0.0, -1.0, 0.0], "uv0": [u, v]}


class WallMeshFilterTests(unittest.TestCase):
    def test_removes_only_disconnected_spike_uv_island(self) -> None:
        mesh = {
            "topology": {"primitive": "triangles", "indices": [0, 1, 2, 3, 4, 5]},
            "vertices": [
                vertex(0, 0, .4, .3), vertex(1, 0, .5, .3), vertex(0, 1, .4, .4),
                vertex(3, 0, .03, .57), vertex(4, 0, .2, .57), vertex(3, 1, .03, .6),
            ],
            "skin": None,
        }
        clean, removed = remove_uv_island_components(mesh, (.02, .24, .56, .61))
        self.assertEqual(removed, 1)
        self.assertEqual(clean["topology"]["indices"], [0, 1, 2])
        self.assertEqual(clean["vertices"], mesh["vertices"][:3])
        self.assertEqual(len(mesh["vertices"]), 6)

    def test_skirt_cut_preserves_horizontal_scale_and_interpolates_uv(self) -> None:
        mesh = {
            "topology": {"primitive": "triangles", "indices": [0, 1, 2, 1, 3, 2]},
            "vertices": [
                vertex(0, -.1, 0, 0), vertex(1, -.1, 1, 0),
                vertex(0, .1, 0, 1), vertex(1, .1, 1, 1),
            ],
            "skin": None,
        }
        clean, metrics = trim_skirt_and_ground(mesh, .02)
        positions = [item["position"] for item in clean["vertices"]]
        self.assertEqual(metrics["discarded_triangles"], 0)
        self.assertEqual(metrics["split_triangles"], 1)
        self.assertEqual(min(p[2] for p in positions), 0.0)
        self.assertAlmostEqual(max(p[2] for p in positions), .08)
        self.assertEqual(min(p[0] for p in positions), 0)
        self.assertEqual(max(p[0] for p in positions), 1)
        cut_vertices = [item for item in clean["vertices"] if item["position"][2] == 0]
        self.assertEqual(len(cut_vertices), 3)
        for item in cut_vertices:
            self.assertAlmostEqual(item["uv0"][1], .6)


if __name__ == "__main__":
    unittest.main()
