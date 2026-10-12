"""Zoom previews keep near-water overlays consistent with their pixels.

Roads, mines, resources and features on river and coastal tiles live in a
retained overlay layer per static slot. A preview used that layer only while
its revision equalled the slot's, but invalidate() (a changed environment key
at every zoom step) bumps the revision without touching the pixels. Those
previews drew the overlays live instead, with the preview water's four
pixel-depth bias toward the camera, so the partly buried rocks of a coal pile
beside a mine showed through the hill for one to three frames per zoom step
(the user's recording, October 11; review 79). The live overlays of a preview
whose sources map one-to-one (a bootstrap at this zoom) took the same bias.
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def helper(source):
    body = source[source.index('    static bool overlay_matches_pixels('):]
    body = body[:body.index('\n    }\n') + 6]
    body = body.replace('    static bool overlay_matches_pixels(', 'bool overlay_matches_pixels(')
    return body.replace('c3x_renderer::render_core::StaticRasterState<c3x_renderer::render_core::LinearTarget>',
                        'StaticRasterState<Target>')


class PreviewOverlayStalenessTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()

    def test_an_invalidated_slot_keeps_its_overlay_layer(self):
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include "Renderer/sandbox/static_raster_state.h"
using namespace c3x_renderer::render_core;
struct Target {void reset(){} std::size_t bytes()const{return 0;}};
struct OverlaySlot {std::uint64_t revision=~0ull,invalidations=0;};
''' + helper(self.source) + r'''
int main(){
 StaticRasterStates<Target> rasters;auto& slot=rasters.states[2];
 // A strip writes pixels and the layer together (write_slot).
 ++slot.revision;OverlaySlot layer;layer.revision=slot.revision;layer.invalidations=slot.invalidations;
 assert(overlay_matches_pixels(layer,slot));
 // A changed environment key: stale, pixels unchanged.
 rasters.invalidate(2,raster_environment);rasters.invalidate(2,raster_shadow);
 assert(slot.stale && overlay_matches_pixels(layer,slot));
 // New pixels without the layer (a strip whose layer write failed).
 ++slot.revision;assert(!overlay_matches_pixels(layer,slot));
 // A retired layer never matches.
 OverlaySlot retired;assert(!overlay_matches_pixels(retired,slot));
 // A discarded slot's pixels are gone.
 OverlaySlot again;again.revision=slot.revision;again.invalidations=slot.invalidations;
 rasters.discard(2,raster_scene);assert(!overlay_matches_pixels(again,slot));
}
''')

    def test_previews_use_the_relaxed_match_and_bias_only_resampled_depth(self):
        preview = self.source[self.source.index('        // Preview: resample the best retained rasters for this projection.'):]
        preview = preview[:preview.index('        ++preview_frames;static_preview=true;')]
        # The old check refused any invalidated slot's layer.
        self.assertNotIn('if(retained.revision!=slot->revision ||', preview)
        self.assertIn('if(!overlay_matches_pixels(retained,*slot) ||', preview)
        self.assertIn('preview_depth_exact=one_to_one(*primary) && (!secondary || one_to_one(*secondary));', preview)
        # Every place a layer's revision is stamped also stamps its invalidations.
        stamps = re.findall(r'(\w+)\.revision=(?:kept|overlays)\?(\w+)\.revision:~0ull;\1\.invalidations=\2\.invalidations;', self.source)
        self.assertEqual(len(stamps), 3, stamps)
        water = self.source[self.source.index('            if(!composited && static_preview && preview_overlays){'):]
        water = water[:water.index('        mark_dynamic(dynamic_water_scene);')]
        self.assertEqual(water.count('preview_water_raster && !preview_depth_exact)renderer.rasterizer_state=preview_water_raster;'), 2)
        self.assertIn('if(!preview_depth_exact){overlay_primary.depth_shift-=4.f/16384.f;', water)


if __name__ == '__main__':
    unittest.main()
