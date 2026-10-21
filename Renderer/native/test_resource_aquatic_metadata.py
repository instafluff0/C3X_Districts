"""Aquatic resource bodies are chosen by pack metadata, never by resource name.

Renderer64 draws fish and whales in an underwater layer that the water surface
refracts, a little below the waterline. The runtime picked those bodies by the
binding names "fish" and "whales", so when compositions began placing the
derived school "fish.few" the fish lost both and swam on top of the water, with
ground shadows on the waves. Scenarios bring arbitrary resources, so each
binding now carries "medium": "water" and its "submersion" (tiles), and the
runtime reads them. Such a body is submerged only on a water tile: the water
surface composites the underwater layer, so a fish in a marsh keeps to the top.
"""
import json
import re
import unittest

from Renderer.lab.platform import ROOT

PACKS = ROOT / "Renderer/packs"
# Civ III terrain bits of the water families (coast 11, sea 12, ocean 13).
WATER = (1 << 11) | (1 << 12) | (1 << 13)


class ResourceAquaticMetadataTests(unittest.TestCase):
    def test_bodies_placed_on_water_are_aquatic_in_their_binding(self):
        bindings = json.loads((PACKS / "ResourceAnimationRuntime/bindings.json").read_text())["bindings"]
        layouts = json.loads((PACKS / "ResourceCompositions/composition_report.json").read_text())["layouts"]
        placed = {}
        for variants in layouts.values():
            for variant in variants:
                for item in variant["instances"]:
                    model = item.get("model") or ""
                    if model.startswith("animated/"):
                        placed.setdefault(model[9:], set()).add(variant["terrain_mask"])
        self.assertIn("fish.few", placed)
        for name, masks in placed.items():
            binding = bindings[name]
            on_water = any(mask & WATER and not (mask & ~WATER) for mask in masks)
            self.assertEqual(binding.get("medium") == "water", on_water, name)
            if on_water:
                self.assertTrue(0 < binding["submersion"] < 1, name)
            else:
                self.assertNotIn("submersion", binding, name)

    def test_runtime_reads_the_medium_instead_of_names(self):
        renderer = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        self.assertIsNone(re.search(r'animation\.name\s*==', renderer))
        self.assertIn('json_string_after(definitions,"medium",record,medium) && medium == "water"', renderer)
        self.assertIn('json_number_after(definitions,"submersion",record,animation.submersion)', renderer)
        self.assertIn("float submersion=pose_only && anchor.submerged?animation.submersion:0.f;", renderer)
        self.assertEqual(renderer.count("aquatic_resource.push_back(anchor.submerged);"), 2)
        # Both anchor kinds submerge an aquatic body on Civ III's water terrains only.
        self.assertEqual(len(re.findall(r"anchor\.submerged *= *(found->|animation\.)aquatic *&& *"
                                        r"tile\.terrain_type *>= *11;", renderer)), 2)


if __name__ == "__main__":
    unittest.main()
