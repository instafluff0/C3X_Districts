"""Production resources are the approved baked compositions.

The runtime's default resource pack must be the folder the "resources" asset
preparation job builds, and that job must bake without Lab-only alternates; a
Lab study still selects a candidate pack through C3X_RENDERER_RESOURCE_PACK.
The previous static ResourceNormalized bundle is no longer the production pack.
"""
import re
import unittest

from Renderer.lab.platform import ROOT


class ResourceProductionPackTests(unittest.TestCase):
    def test_runtime_default_is_the_prepared_composition_pack(self):
        renderer = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        block = renderer[renderer.index('char resource_pack[128]='):]
        block = block[:block.index("resource_assets_ready = load_runtime_bundle(")]
        defaults = set(re.findall(r'"(Resource\w+)"', block))
        self.assertEqual(defaults, {"ResourceCompositions"})
        self.assertIn("C3X_RENDERER_RESOURCE_PACK", block)
        preparation = (ROOT / "Renderer/lab/asset_preparation.py").read_text()
        job = preparation[preparation.index("    def build_resources(stage):"):]
        job = job[:job.index("\n    def ", 1)]
        self.assertIn('stage / "Renderer/packs/ResourceCompositions", alternates=False', job)


if __name__ == "__main__":
    unittest.main()
