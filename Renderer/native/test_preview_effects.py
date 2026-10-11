"""Attached effects stay visible on static preview frames.

A zoom (and the first frames after a jump) shows previews resampled from
another zoom while the new static layer builds. Flames, smoke, night lights
and volcano plumes were skipped on every preview, so they vanished for a
whole zoom and returned once the camera settled (the user, October 11). Their
shaders read no scene depth; previews draw them without the resampled depth,
which made them flash when tested against it.
"""
import unittest
from Renderer.lab.platform import ROOT


class PreviewEffectsTests(unittest.TestCase):
    def test_preview_frames_draw_effects_without_the_resampled_depth(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        # The old code skipped the layer on any preview.
        self.assertNotIn('if(!static_preview && !effect_visible[geometry_city].empty()', source)
        self.assertIn('if(!(static_preview && skip_preview_effects) && !effect_visible[geometry_city].empty()', source)
        self.assertIn('glow.linear.target,static_preview?nullptr:glow.linear.depth,false,float(scene_scale)))return fail("city_effects");', source)
        self.assertIn('"C3X_RENDERER_PREVIEW_EFFECTS"', source)
        # The effect shaders must not start reading scene depth: previews bind none.
        shader = (ROOT / 'Renderer/native/city_fidelity/city.hlsl').read_text()
        effects = shader[shader.index('float4 q8_city_effect('):]
        effects = effects[:effects.index('#endif')]
        for token in ('SV_Depth', 'scene_depth', 'depth_texture'):
            self.assertNotIn(token, effects)


if __name__ == '__main__':
    unittest.main()
