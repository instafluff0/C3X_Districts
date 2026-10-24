"""Natural-wonder labels: captured facts and C3X's draw_map_tile_text placement at every zoom."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WonderLabelTests(unittest.TestCase):
    def test_capture_and_native_placement(self):
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include "Renderer/native/render_core/wonder_labels.h"
int main(){
 using c3x_renderer::render_core::WonderLabels;
 c3x_renderer_tile_v1 tiles[4]={};
 for(auto& t:tiles){t.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
  t.natural_wonder_flags=C3X_RENDERER_NATURAL_WONDER_NAME;std::memcpy(t.natural_wonder,"Old Faithful",13);}
 tiles[0].tile_x=4;tiles[0].tile_y=6;tiles[0].anchor_x=320;tiles[0].anchor_y=200;
 tiles[1].tile_x=5;tiles[1].tile_y=7;tiles[1].tile_flags&=~C3X_RENDERER_TILE_EXPLORED;       // unexplored: no label
 tiles[2].tile_x=6;tiles[2].tile_y=8;tiles[2].natural_wonder_flags=0;                         // name hidden by C3X
 tiles[3].tile_x=6;tiles[3].tile_y=9;                                                       // off the parity lattice
 c3x_renderer_frame_v1 frame{};frame.tiles=tiles;frame.tile_count=4;frame.tile_width=128;frame.tile_height=64;
 frame.target_width=1280;frame.target_height=720;
 WonderLabels labels;assert(labels.capture(frame) && labels.labels.size()==1 && labels.labels[0].name=="Old Faithful");
 // At 1x: draw_map_tile_text(x, y, 88, 24) centres in 128-8 at y-24+88+4, text 10 above that.
 auto box=WonderLabels::box(frame,1.f,320,200);
 assert(box.left==324 && box.width==120 && box.top==200-24+88+4-10);
 // At 2x the anchor scales about the view centre and the tile is 256 wide; text keeps its size.
 auto zoomed=WonderLabels::box(frame,2.f,320,200);
 assert(zoomed.left==640+(320-640)*2+8 && zoomed.width==240 && zoomed.top==360+(200-360)*2-48+176+8-10);
 // C3X's zoomed-out 64 px tiles.
 frame.tile_width=64;auto small=WonderLabels::box(frame,1.f,320,200);
 assert(small.left==322 && small.width==60 && small.top==200-12+44+2-10);
}
''')


if __name__ == "__main__":
    unittest.main()
