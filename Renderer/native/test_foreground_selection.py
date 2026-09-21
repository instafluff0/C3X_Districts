"""Camera reuse must preserve the contributors that a cold view would select."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ForegroundSelectionTests(unittest.TestCase):
    def test_camera_crosses_prefetch_boundary(self):
        run_cpp(r'''
#include "Renderer/native/render_core/foreground_selection.h"
#include <cassert>
#include <climits>
using namespace c3x_renderer::render_core;
int main(){
 ForegroundSelection select{2240,1260,128,64,4,0,true,false};
 c3x_renderer_tile_v1 old{};old.tile_flags=C3X_RENDERER_TILE_PREFETCH;
 old.anchor_x=-640;old.anchor_y=0;
 auto current=old;--current.anchor_x;
 assert(select.selects(old)&&!select.selects(current));
 assert(!select.preserves(old,current));
 old.anchor_x=2240+512;current=old;++current.anchor_x;
 assert(!select.preserves(old,current));
 old.anchor_x=0;old.anchor_y=-320;current=old;--current.anchor_y;
 assert(!select.preserves(old,current));
 old.anchor_y=1260+256;current=old;++current.anchor_y;
 assert(!select.preserves(old,current));
 old.anchor_x=old.anchor_y=0;current=old;current.anchor_x=57;
 assert(select.preserves(old,current)); // Ordinary interior reuse remains valid.
 old.tile_flags=current.tile_flags=C3X_RENDERER_TILE_RENDER;
 old.anchor_x=INT_MIN;current.anchor_x=INT_MAX;
 assert(select.preserves(old,current)); // Native visible ownership wins.
 old.tile_flags=C3X_RENDERER_TILE_TOPOLOGY_HALO;
 assert(!select.selects(old));
 old.tile_flags=C3X_RENDERER_TILE_PREFETCH;old.anchor_x=0;old.anchor_y=0;
 select.offload=true;assert(!select.selects(old));
 select.guard=2;assert(select.selects(old));
 current=old;current.anchor_x=2240+257;assert(!select.preserves(old,current));
 auto changed=select;changed.ring=2;assert(!(changed==select));
 changed=select;changed.width=1920;assert(!(changed==select));
}
''')
