from __future__ import annotations

import unittest

from Renderer.tools.asset_compiler.unit_model_extractor import _remap_skin_palette


class UnitModelExtractorTests(unittest.TestCase):
    def test_local_skin_palette_is_remapped_to_skeleton_indices(self) -> None:
        mesh = {"vertices": [{"skin": {"bone_indices": [0, 2], "bone_weights": [0.75, 0.25]}}]}
        _remap_skin_palette(mesh, [4, 1, 7], 8)
        self.assertEqual([4, 7], mesh["vertices"][0]["skin"]["bone_indices"])

    def test_local_skin_palette_rejects_out_of_range_vertex_index(self) -> None:
        mesh = {"vertices": [{"skin": {"bone_indices": [2], "bone_weights": [1.0]}}]}
        with self.assertRaisesRegex(ValueError, "outside its local skin palette"):
            _remap_skin_palette(mesh, [0, 1], 3)


if __name__ == "__main__":
    unittest.main()
