"""Near-water overlays stay above the live water on static preview frames.

Roads, bridges, improvements and features on river and coastal tiles are drawn
after the animated water. An exact frame draws the water into the live layer
and then composites the retained overlay layer over it. A preview frame
(resampled from another zoom) baked the sources' overlay layers into the
static image instead, under the live layer, and its water is biased four
pixel-depth units toward the camera: the two roads crossing a river vanished
on every such preview frame of a zoom (the user's recording, October 11).
"""
import unittest
from Renderer.lab.platform import ROOT


class PreviewOverlayOrderTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()

    def test_overlays_are_not_baked_under_the_live_layer(self):
        preview = self.source[self.source.index('        // Preview: resample the best retained rasters for this projection.'):]
        preview = preview[:preview.index('        ++preview_frames;static_preview=true;')]
        # The old code drew the overlay layers into static_cache here.
        self.assertNotIn('static_resample.draw(context,static_cache,primary_overlay', preview)
        self.assertIn('preview_overlay_sources[0]=primary_overlay;preview_overlay_sources[1]=secondary_overlay;', preview)

    def test_overlays_are_composited_after_the_live_water(self):
        branch = self.source[self.source.index('            if(!composited && static_preview && preview_overlays){'):]
        branch = branch[:branch.index('                composited=true;++overlay_preview_hits;')]
        water = branch.index('draw_layer(water_visible,geometry_water,settings,full,glow.linear.target,glow.linear.depth')
        river = branch.index('draw_layer(water_visible,geometry_river,settings,full,glow.linear.target,glow.linear.depth')
        overlay = branch.index('static_resample.draw(context,glow.linear,overlay_primary,')
        wave = branch.index('draw_layer(water_visible,geometry_wave,')
        self.assertLess(water, overlay)
        self.assertLess(river, overlay)
        self.assertLess(overlay, wave, 'the same order as an exact frame: water, river, overlays, waves')
        # The live water is biased four pixel-depth units; the overlays must
        # be shifted by the same amount to keep their depth order with it.
        self.assertIn("desc.DepthBias=-INT(4.0*16777216.0/16384.0);", branch)
        self.assertIn('overlay_primary.depth_shift-=4.f/16384.f;overlay_secondary.depth_shift-=4.f/16384.f;', branch)


if __name__ == '__main__':
    unittest.main()
