"""Fog coverage reaches every tile the widest outward zoom shows.

The GPU fog pass scales captured anchors about the view center for the
current zoom. Below 1x, tiles outside the canonical viewport are on screen, so
their fog quads must exist; previously they were culled at the 1x viewport and
the outer ring of explored-but-unseen and unexplored tiles stayed unfogged.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class VisibilityZoomExtentTests(unittest.TestCase):
    def test_outer_ring_tiles_receive_fog_quads(self):
        run_cpp(r'''
#include "Renderer/native/render_core/visibility_coverage.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
int main(){
    c3x_renderer_frame_v1 frame{};frame.world_width_tiles=frame.world_height_tiles=64;
    frame.target_width=1024;frame.target_height=512;frame.tile_width=128;frame.tile_height=64;
    std::vector<c3x_renderer_tile_v1> tiles;
    // Explored, currently unseen ocean at the 1x center, just outside the 1x
    // viewport (visible at 0.5x) and beyond anything 0.5x can show.
    auto add=[&](int x,int y,int ax,int ay){c3x_renderer_tile_v1 t{};t.tile_x=x;t.tile_y=y;t.anchor_x=ax;t.anchor_y=ay;
        t.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;tiles.push_back(t);};
    add(10,10,448,224);    // inside 1x
    add(20,20,1200,224);   // right of the 1x view, inside the 0.5x view (center 512, reach 1536)
    add(30,30,-400,224);   // left of the 1x view, inside the 0.5x view (reach -512)
    add(40,40,448,-200);   // above the 1x view, inside the 0.5x view (reach -256)
    add(50,50,1700,224);   // beyond the widest zoom
    frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
    VisibilityCoverage coverage;assert(coverage.capture(frame));
    auto has=[&](float x,float y){for(auto const& t:coverage.tiles)if(t.x==x&&t.y==y)return true;return false;};
    assert(has(448,224)&&has(1200,224)&&has(-400,224)&&has(448,-200));
    assert(!has(1700,224));
    // A wide capture can repeat an anchor in the outer ring with different
    // neighbors; that must not fail the frame. Inside the 1x view it still does.
    auto conflict=[&](int ax,int ay){auto copy=tiles;c3x_renderer_tile_v1 t{};t.tile_x=60;t.tile_y=2;t.anchor_x=ax;t.anchor_y=ay;
        t.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_VISIBLE;
        copy.push_back(t);auto f=frame;f.tiles=copy.data();f.tile_count=unsigned(copy.size());VisibilityCoverage c;return c.capture(f);};
    assert(conflict(1200,224));
    assert(!conflict(448,224));
    std::printf("PASS visibility zoom extent: outer_ring_fogged=1 beyond_widest_culled=1\n");
}
''')


if __name__ == '__main__':
    unittest.main()
