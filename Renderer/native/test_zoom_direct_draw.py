"""Per-frame zoom drawing (camera decoupling plan item 4).

A moving zoom showed static previews: the cached image of another zoom,
resampled (blurred when magnified), replaced by the refined raster once the
zoom settled (a visible pop). With per-frame zoom drawing each moving frame,
and the frame that reaches the destination, draws the visible static raster
at its own zoom. The water mirror, whose redraw costs several times the
static draw in the VM (review 77), waits for the zoom to settle.
"""
import unittest
from Renderer.lab.platform import ROOT


def compose_static(source):
    body = source[source.index('        bool moving=previewable && zoom_moving();'):]
    return body[:body.index('        // Preview: resample the best retained rasters for this projection.')]


class ZoomDirectDrawTests(unittest.TestCase):
    def test_moving_zoom_frames_draw_at_their_zoom(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        compose = compose_static(source)
        self.assertIn('bool direct=options.zoom_direct && previewable &&\n'
                      '            (moving || (lane==1 && static_rasters.states[front_index].projection!=zoom));', compose)
        # An unbounded refine at the current zoom promotes a complete raster,
        # so the frame takes the shifted (exact) path, not the preview.
        direct = compose[compose.index('        if(direct){'):]
        direct = direct[:direct.index('        }\n') + 10]
        self.assertIn('double unbounded=-1.;', direct)
        self.assertIn('refine(zoom,unbounded)', direct)
        self.assertIn('refine_worked=false;', direct, 'a direct draw must not shrink the refinement budget')
        # The direct draw follows the budgeted refinement, which waits while moving.
        self.assertLess(compose.index('else if(budget!=0 && (!moving ||'), compose.index('        if(direct){'))
        # No background guard-band fill behind a moving zoom.
        self.assertIn('if(shifted && previewable && !direct && budget>0', compose)

    def test_the_mirror_waits_for_the_zoom_to_settle(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        mirror = source[source.index('            bool redraw=!reflection_valid || reflection_key!=key;'):]
        mirror = mirror[:mirror.index('            if(!redraw){')]
        self.assertIn('if(redraw && reflection_valid && sandbox_perf_options().zoom_direct && zoom_moving())'
                      '{redraw=false;++reflection_skips;}', mirror)

    def test_switch(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        self.assertIn('zoom_direct=GetEnvironmentVariableA("C3X_RENDERER_ZOOM_DIRECT",value,sizeof(value)) && value[0]==', source)


if __name__ == '__main__':
    unittest.main()
