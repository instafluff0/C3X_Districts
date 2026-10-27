"""Attached effects stay visible on static preview frames, behind what hides them.

A zoom (and the first frames after a jump) shows previews resampled from
another zoom while the new static layer builds. Flames, smoke, night lights
and volcano plumes were skipped on every preview, so they vanished for a
whole zoom and returned once the camera settled (the user, October 11). Drawn
without depth instead, a torch behind a house shone on its roof at every zoom
step (the user, 2026-10-27). Previews now test them against the reconstructed
depth with the live water's bias toward the camera; the winter flurry too.
"""
import unittest
from Renderer.lab.platform import ROOT


class PreviewEffectsTests(unittest.TestCase):
    def test_preview_frames_draw_effects_behind_the_scene_depth(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        # The old code skipped the layer on any preview, then drew it without depth.
        self.assertNotIn('if(!static_preview && !effect_visible[geometry_city].empty()', source)
        self.assertNotIn('static_preview?nullptr:glow.linear.depth', source)
        effects = source[source.index('bool const effects_drawn='):source.index('return fail("city_effects");')]
        self.assertIn('(static_preview && skip_preview_effects) || effect_visible[geometry_city].empty() ||', effects)
        self.assertIn('draw_layer(effect_visible,geometry_city,settings,full,glow.linear.target,glow.linear.depth,', effects)
        self.assertIn('"C3X_RENDERER_PREVIEW_EFFECTS"', source)
        # Inexact preview depth: the effects and the flurry take the water's bias.
        bias = 'if(static_preview && !preview_depth_exact)if(auto* biased=preview_bias_raster())renderer.rasterizer_state=biased;'
        block = source[source.index('auto* const effect_raster=renderer.rasterizer_state;'):source.index('return fail("winter_flurry");')]
        self.assertEqual(block.count(bias), 2)
        self.assertIn('bool const snow_drawn=draw_layer(snow,geometry_city,settings,full,glow.linear.target,glow.linear.depth,', block)
        self.assertEqual(block.count('renderer.rasterizer_state=effect_raster;'), 2)
        self.assertIn('desc.DepthBias=-INT(4.0*16777216.0/16384.0);', source)


if __name__ == '__main__':
    unittest.main()
