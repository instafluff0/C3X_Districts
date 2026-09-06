from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

from Renderer.tools.asset_compiler.normalized_skeleton_to_cn6 import write_model_companion
from Renderer.tools.asset_compiler.test_normalized_skin import fixture_skeleton


class SkeletonCompanionTests(unittest.TestCase):
    def test_writes_source_scale_skeleton_and_minimal_mesh(self) -> None:
        skeleton = fixture_skeleton()
        skeleton["bones"][1]["local"]["position"] = [0.25, 0.0, 0.0]
        skeleton["bones"][1]["inverse_bind_matrix"][12] = -0.25
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "skeleton.json"
            output = root / "companion.cn6"
            source.write_text(json.dumps(skeleton), encoding="utf-8")
            report = write_model_companion(source, output, 100.0)
            text = output.read_text(encoding="utf-8")
        self.assertEqual(report["bones"], 2)
        self.assertIn('1 "Child" 0 25 ', text)
        self.assertIn("-25 ", text)
        self.assertIn("meshes:1", text)
        self.assertIn("0 1 2 0", text)
        vertex_lines = text.split("vertices\n", 1)[1].split("triangles\n", 1)[0].splitlines()
        self.assertEqual(len(vertex_lines), 3)
        self.assertTrue(all(len(line.split()) == 34 for line in vertex_lines))

    def test_rejects_invalid_source_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            write_model_companion(Path("missing"), Path("unused"), 0.0)


if __name__ == "__main__":
    unittest.main()
