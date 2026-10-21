"""The city view's halo stays on the native lattice, so its world window builds.

With custom zoom off (the city view, or zoom disabled), the injected capture
adds a 12-tile halo of uncaptured neighbours. It used to place them with Civ
III's tile_to_screen_coords, which folds any tile more than half a map from
TileX_Min/TileY_Min by a whole map size, even on an axis that does not wrap.
On a 100x100 map the city view's 40 rows plus the halo crossed that fold, the
folded rows left the world, and WorldWindow::build rejected the whole frame:
the city view then drew no hidden ring, so unexplored tiles next to explored
land were hard-edged black holes (a goody hut's map reveal, 2026-10-21).

The harness runs the production function with host C mocks (the native
projection reproduced exactly), then hands its tiles to the renderer's own
WorldWindow.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_injected_world_local import ROOT, production_function


PRELUDE = r'''
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Renderer/native/c3x_renderer_api.h"
#define __fastcall
#define __ 0
#define C3X_NATIVE_CAPTURE_MARGIN 1
typedef struct { int left, top, right, bottom; } RECT;
typedef struct Tile { int unused; } Tile;
typedef struct Map_Renderer { int unused; } Map_Renderer;
typedef struct Map { int Width, Height, Flags; } Map;
typedef struct Bic { Map Map; bool is_zoomed_out; } Bic;
typedef struct Main_Screen_Form { int camera_x, camera_y, TileX_Min, TileX_Max, TileY_Min, TileY_Max; } Main_Screen_Form;
typedef struct State {
    bool custom_renderer_capture_failed, custom_renderer_capture_world_topology;
    int (*custom_renderer_native_image) (int, void *, void *, void *, void *, int);
    Map_Renderer *custom_renderer_target;
    struct c3x_renderer_tile_v1 *custom_renderer_tiles;
    int custom_renderer_tile_count, custom_renderer_zoom_tile_width, custom_renderer_zoom_native_tile_width;
} State;
static State state, *is = &state;
static Bic bic, *p_bic_data = &bic;
static Main_Screen_Form screen, *p_main_screen_form = &screen;
static Tile tiles[5000], null_tile, *p_null_tile = &null_tile;
static struct c3x_renderer_tile_v1 records[8192];
bool custom_renderer_zoom_enabled (void) { return false; }
void sync_custom_renderer_zoom_to_native (void) {}
RECT custom_renderer_capture_bounds (Main_Screen_Form * s) { RECT r = {s->TileX_Min, s->TileY_Min, s->TileX_Max, s->TileY_Max}; return r; }
bool Map_in_range (Map * m, int edx, int x, int y) { (void)edx; return x >= 0 && y >= 0 && x < m->Width && y < m->Height; }
void log_custom_renderer_event (char const * stage, int result) { (void)result; fprintf (stderr, "event %s\n", stage); }
unsigned capture_custom_renderer_visibility (Tile * t, int viewer, int x, int y)
{ (void)t; (void)viewer; (void)x; (void)y; return C3X_RENDERER_TILE_VISIBILITY_KNOWN | C3X_RENDERER_TILE_EXPLORED; }
// Native tile_at: a 16-bit index, out of range reads the null tile.
Tile * tile_at (int x, int y)
{
    unsigned index = ((unsigned)(bic.Map.Width >> 1) * (unsigned)y + ((unsigned)x >> 1 & 0x7fff)) & 0xffff;
    return index < (unsigned)(bic.Map.Width * bic.Map.Height / 2) ? &tiles[index] : p_null_tile;
}
// Native Main_Screen_Form::tile_to_screen_coords (GOG), folding included.
void Main_Screen_Form_tile_to_screen_coords (Main_Screen_Form * s, int edx, int tile_x, int tile_y, int * out_x, int * out_y)
{
    (void)edx;
    int w = bic.is_zoomed_out ? 64 : 128, h = w / 2;
    tile_x -= s->TileX_Min;
    if (bic.Map.Flags & 1) { if (tile_x < 0) tile_x += bic.Map.Width; else if (bic.Map.Width <= tile_x) tile_x -= bic.Map.Width; }
    tile_y -= s->TileY_Min;
    if (bic.Map.Flags & 2) { if (tile_y < 0) tile_y += bic.Map.Height; else if (bic.Map.Height <= tile_y) tile_y -= bic.Map.Height; }
    if (bic.Map.Width / 2 < tile_x) tile_x -= bic.Map.Width;
    if (bic.Map.Height / 2 < tile_y) tile_y -= bic.Map.Height;
    *out_x = (w >> 1) * (tile_x - 1) - s->camera_x % (w >> 1);
    *out_y = (h >> 1) * (tile_y - 1) - s->camera_y % (h >> 1);
}
bool capture_custom_renderer_tile (int viewer, int px, int py, Map_Renderer * target, int mask, int x, int y, Tile * tile, bool topology_only)
{
    (void)viewer; (void)target; (void)mask; (void)tile;
    struct c3x_renderer_tile_v1 * r = &is->custom_renderer_tiles[is->custom_renderer_tile_count++];
    memset (r, 0, sizeof *r);
    r->tile_x = x; r->tile_y = y; r->anchor_x = px; r->anchor_y = py;
    r->tile_flags = capture_custom_renderer_visibility (tile, 0, x, y) |
        (topology_only ? C3X_RENDERER_TILE_TOPOLOGY_HALO : C3X_RENDERER_TILE_RENDER);
    return true;
}
'''

# A 100x100 map wrapping east-west only, the city view at 1x: Civ III's m21
# traversal draws 36 columns and 40 rows with linear anchors.
CASE = r'''
int main (void)
{
    bic.Map.Width = bic.Map.Height = 100; bic.Map.Flags = 1;
    screen.camera_x = 4128; screen.camera_y = 684;
    screen.TileX_Min = 64; screen.TileX_Max = 99; screen.TileY_Min = 21; screen.TileY_Max = 60;
    state.custom_renderer_tiles = records; state.custom_renderer_capture_world_topology = true;
    for (int y = screen.TileY_Min; y <= screen.TileY_Max; y++)
        for (int x = screen.TileX_Min; x <= screen.TileX_Max; x++) {
            if ((x + y) & 1) continue;
            int px = 64 * (x - screen.TileX_Min - 1) - screen.camera_x % 64;
            int py = 32 * (y - screen.TileY_Min - 1) - screen.camera_y % 32;
            capture_custom_renderer_tile (1, px, py, NULL, 0, x, y, &tiles[0], false);
        }
    int native = state.custom_renderer_tile_count;
    capture_custom_renderer_topology (1, 0);
    printf ("%d %d\n", native, state.custom_renderer_tile_count);
    for (int i = 0; i < state.custom_renderer_tile_count; i++)
        printf ("%d %d %d %d %u\n", records[i].tile_x, records[i].tile_y, records[i].anchor_x, records[i].anchor_y, records[i].tile_flags);
    return 0;
}
'''


class CityViewHaloTests(unittest.TestCase):
    def test_city_view_halo_keeps_the_world_window(self):
        compiler = shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("Host clang is unavailable")
        source = (ROOT / "injected_code.c").read_text()
        program = PRELUDE + production_function(source, "wrap_tile_coords") + \
            production_function(source, "capture_custom_renderer_topology") + CASE
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "halo.c").write_text(program)
            built = subprocess.run([compiler, "-std=c17", "-O1", "-Wall", "-Wno-unused-parameter", "-I", str(ROOT),
                                    str(path / "halo.c"), "-o", str(path / "halo")], text=True, capture_output=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            ran = subprocess.run([str(path / "halo")], text=True, capture_output=True, timeout=10)
            self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
        lines = ran.stdout.split("\n")
        native, total = map(int, lines[0].split())
        self.assertGreater(total, native, "the halo must add the uncaptured neighbourhood")
        records = ",".join("{%s}" % ",".join(line.split()) for line in lines[1:] if line)
        run_cpp(r'''
#include "Renderer/native/render_core/world_window.h"
#include <cassert>
#include <cstdio>
#include <vector>
using namespace c3x_renderer::render_core;
int main(){
 int captured[][5]={''' + records + r'''};
 std::vector<c3x_renderer_tile_v1> tiles;
 for(auto const& r:captured){c3x_renderer_tile_v1 t{};t.tile_x=r[0];t.tile_y=r[1];t.anchor_x=r[2];t.anchor_y=r[3];
  t.tile_flags=unsigned(r[4]);tiles.push_back(t);}
 c3x_renderer_frame_v1 f{};f.tile_width=128;f.tile_height=64;f.target_width=2240;f.target_height=1192;
 f.world_width_tiles=f.world_height_tiles=100;f.world_wrap_x=1;f.tiles=tiles.data();f.tile_count=unsigned(tiles.size());
 // Every captured tile sits on one lattice; nothing folds out of the world.
 WorldWindow window;
 bool built=window.build(f,[](int,int,c3x_renderer_tile_v1&){return false;});
 std::printf("halo tiles=%zu window valid=%d\n",tiles.size(),int(built));
 assert(built);
}
''')


if __name__ == "__main__":
    unittest.main()
