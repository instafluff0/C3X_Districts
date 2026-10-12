"""A zoom reaches its destination's look once, during the zoom.

Small objects render differently at each zoom (coal piles are about half
again as heavy at 1x as at 2x), so a static preview stretched from another
zoom's raster changed look when the exact image arrived after the zoom
settled: resources and mines seemed to flicker (the user, October 11).
The zoom lane now refines toward the destination while the zoom moves; the
promoted destination raster becomes the preview source. The shadow field is
fitted to the destination at once, or settling would refit it and redraw the
raster (a second change). Returning to 1x, previews prefer the canonical
raster. Measured on the busy save it halved zoom frame rates in the VM and
did not remove the change (review 79), so it is off by default:
C3X_RENDERER_ZOOM_EARLY=1 enables it.
"""
import unittest
from Renderer.lab.platform import ROOT


class ZoomEarlyRefinementTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()

    def test_destination_refines_while_the_zoom_moves(self):
        # The old gate refined only after the zoom settled.
        self.assertNotIn('else if(budget!=0 && !moving){static_decision.refine=1;', self.source)
        self.assertIn('else if(budget!=0 && (!moving || (options.zoom_early && lane==1 && goal==hint))){', self.source)

    def test_shadow_field_is_the_destination_field(self):
        self.assertIn('if(body_requirements_valid && field>0 && wanted>field && zoom_moving() && '
                      '!sandbox_perf_options().zoom_early)wanted=field;', self.source)

    def test_previews_returning_to_1x_use_the_canonical_raster(self):
        preview = self.source[self.source.index('        // Preview: resample the best retained rasters for this projection.'):]
        first = preview.index('if(options.zoom_early && lane==1 && hint==1.f && home_cover==2){primary=&home_source;primary_slot=&home;}')
        self.assertLess(first, preview.index('else if(front_cover==2){primary=&front_source;'))

    def test_switch(self):
        self.assertIn('zoom_early=GetEnvironmentVariableA("C3X_RENDERER_ZOOM_EARLY",value,sizeof(value)) && value[0]==\'1\';', self.source)
        self.assertIn('bool zoom_early=false;', self.source)


if __name__ == '__main__':
    unittest.main()
