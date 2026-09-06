from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.effect_graph_compiler import (
    DEFAULT_SOURCE,
    compile_effect_graphs,
    sample_effect,
)


class EffectGraphCompilerTests(unittest.TestCase):
    def test_checked_profiles_resolve_and_sample_deterministically(self) -> None:
        result = compile_effect_graphs()
        self.assertEqual(6, result["summary"]["profiles"])
        self.assertGreaterEqual(result["summary"]["emitters"], 9)
        self.assertEqual("none", result["source_behavior_claim"])
        profile = result["profiles"]["infrastructure/pollution_radiation"]
        first = sample_effect("infrastructure/pollution_radiation", profile, "tile/42", 1750, "normal")
        repeated = sample_effect("infrastructure/pollution_radiation", profile, "tile/42", 1750, "normal")
        reduced = sample_effect("infrastructure/pollution_radiation", profile, "tile/42", 1750, "reduced")
        self.assertEqual(first, repeated)
        self.assertLessEqual(len(first), profile["maximum_live_particles"])
        self.assertLessEqual(len(reduced), len(first))
        before_wrap = sample_effect("infrastructure/pollution_radiation", profile, "tile/42", 2799, "normal")
        after_wrap = sample_effect("infrastructure/pollution_radiation", profile, "tile/42", 2800, "normal")
        self.assertTrue({item["id"] for item in before_wrap} & {item["id"] for item in after_wrap})
        self.assertTrue(all(0.0 <= item["opacity"] <= 1.0 for item in first))
        self.assertTrue(all(len(item["atlas_uv"]) == 4 for item in first))

    def test_missing_texture_fails_closed(self) -> None:
        source = json.loads(DEFAULT_SOURCE.read_text(encoding="utf-8"))
        source["profiles"]["ambient/fire_small"]["emitters"][0]["texture"] = "missing/texture"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unavailable texture"):
                compile_effect_graphs(path)

    def test_non_looping_profile_cleans_up(self) -> None:
        result = compile_effect_graphs()
        profile = result["profiles"]["combat/land_impact"]
        self.assertEqual([], sample_effect("combat/land_impact", profile, "impact/7", 1100, "normal"))


if __name__ == "__main__":
    unittest.main()
