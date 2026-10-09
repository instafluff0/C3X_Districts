"""The resident world is selected by a block-anchored window, not the capture.

A selection taken from Civ III's capture changed at every camera step (the
capture reaches 512 px in x and 256 px in y past the view), so casters and
contributors churned, static rasters were repaired and refinement restarted
(performance review, sections 25-26). The window reaches the static guard
band plus shadow reach past the view, is snapped to 8x8-coordinate blocks,
and draws tiles the capture lacks from the retained world.
"""
import unittest
from pathlib import Path

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class WorldWindowTests(unittest.TestCase):
    def test_window_is_stable_within_a_block_and_keeps_capture_order_for_output(self):
        run_cpp(r'''
#include "Renderer/native/render_core/world_window.h"
#include "Renderer/native/render_core/foreground_selection.h"
#include <cassert>
#include <cstdio>
#include <set>
#include <vector>
using c3x_renderer::render_core::WorldWindow;
using c3x_renderer::render_core::ForegroundSelection;
constexpr int TW=128,TH=64,HW=64,HH=32,W=1000,H=600,WORLD_W=200,WORLD_H=200;
// Civ III's capture: RENDER over the view, PREFETCH 8 coordinates past it
// (the appearance halo), topology-only tiles 4 more. Camera = view origin.
std::vector<c3x_renderer_tile_v1> capture(int cam_x,int cam_y){
 std::vector<c3x_renderer_tile_v1> tiles;
 int x0=cam_x/HW,y0=cam_y/HH,x1=(cam_x+W)/HW,y1=(cam_y+H)/HH;
 for(int pass=0;pass<2;++pass)for(int y=y0-12;y<=y1+12;++y)for(int x=x0-12;x<=x1+12;++x){
  if((x+y)&1)continue;
  bool view=x>=x0-1&&x<=x1&&y>=y0-1&&y<=y1;if((pass==0)!=view)continue;
  bool appearance=view||(x>=x0-8&&x<=x1+8&&y>=y0-8&&y<=y1+8);
  c3x_renderer_tile_v1 t{};t.tile_x=((x%WORLD_W)+WORLD_W)%WORLD_W;t.tile_y=y;if(y<0||y>=WORLD_H)continue;
  t.anchor_x=x*HW-cam_x;t.anchor_y=y*HH-cam_y;t.terrain_type=(x*7+y*3)%11;
  t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|
   (view?C3X_RENDERER_TILE_RENDER:appearance?C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO:C3X_RENDERER_TILE_TOPOLOGY_HALO);
  tiles.push_back(t);}
 return tiles;}
c3x_renderer_frame_v1 frame_for(std::vector<c3x_renderer_tile_v1>& tiles){
 c3x_renderer_frame_v1 f{};f.tile_width=TW;f.tile_height=TH;f.target_width=W;f.target_height=H;
 f.world_width_tiles=WORLD_W;f.world_height_tiles=WORLD_H;f.world_wrap_x=1;f.tiles=tiles.data();f.tile_count=unsigned(tiles.size());return f;}
// The retained world: every tile explored, content from coordinates.
bool retained(int x,int y,c3x_renderer_tile_v1& t){if(x==7&&y==21)return false;t={};t.terrain_type=(x*7+y*3)%11;
 t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED|C3X_RENDERER_TILE_PREFETCH;return true;}
struct View {std::vector<std::pair<int,int>> occurrences;std::array<std::int64_t,4> box;std::vector<int> terrain;};
View window_at(int cam_x,int cam_y,WorldWindow& w,std::vector<c3x_renderer_tile_v1>& tiles){
 tiles=capture(cam_x,cam_y);auto f=frame_for(tiles);
 bool ok=w.build(f,[](int x,int y,c3x_renderer_tile_v1& t){return retained(x,y,t);});assert(ok&&w.valid);
 View v;v.box=w.box;
 ForegroundSelection s{W,H,TW,TH,2,0,true,false};s.windowed=true;s.box={w.box[0],w.box[1],w.box[2],w.box[3]};s.camera_x=w.camera_x;s.camera_y=w.camera_y;
 for(unsigned i=0;i<w.tiles.size();++i){auto const& t=w.tiles[i];bool sel=s.selects(t);
  assert(sel==(i<w.window_count));                                   // the window part is the selection
  if(!sel)continue;
  long long ox=(t.anchor_x-w.camera_x)/HW,oy=(t.anchor_y-w.camera_y)/HH;
  v.occurrences.push_back({int(ox),int(oy)});v.terrain.push_back(t.terrain_type);}
 return v;}
int main(){
 WorldWindow w;std::vector<c3x_renderer_tile_v1> tiles;
 // The window covers the view plus 8 (x) and 12 (y) coordinates, block-snapped.
 auto a=window_at(64*40,32*60,w,tiles);
 assert(a.box[0]%(8*HW)==0&&a.box[1]%(8*HH)==0&&a.box[2]%(8*HW)==0&&a.box[3]%(8*HH)==0);
 assert(a.box[0]<=64*40-8*HW&&a.box[2]>=64*40+W+8*HW&&a.box[1]<=32*60-12*HH&&a.box[3]>=32*60+H+12*HH);
 assert(w.synthesized>0);                                              // past the capture: retained world
 // Row-major order, no duplicates, the retained gap (7,21) absent.
 std::set<std::pair<int,int>> seen;
 for(std::size_t i=0;i<a.occurrences.size();++i){assert(seen.insert(a.occurrences[i]).second);
  if(i)assert(a.occurrences[i-1].second<a.occurrences[i].second||(a.occurrences[i-1].second==a.occurrences[i].second&&a.occurrences[i-1].first<a.occurrences[i].first));}
 // Camera steps inside the block window keep the same occurrences, content
 // and order; anchors move uniformly. The capture changes on every step.
 for(int step=1;step<=3;++step){auto b=window_at(64*40+step*64*2,32*60,w,tiles);
  assert(b.box==a.box&&b.occurrences==a.occurrences&&b.terrain==a.terrain);}
 // Crossing a block changes it.
 auto c=window_at(64*40+64*8,32*60,w,tiles);assert(c.box!=a.box);
 // Output stays in capture order: every RENDER tile maps back to its index.
 tiles=capture(64*40,32*60);auto f=frame_for(tiles);w.build(f,[](int x,int y,c3x_renderer_tile_v1& t){return retained(x,y,t);});
 std::vector<c3x_renderer_u32> flags(w.tiles.size());for(unsigned i=0;i<w.tiles.size();++i)flags[i]=1000+i;
 std::vector<c3x_renderer_u32> native;w.native_flags(flags,f.tile_count,native);assert(native.size()==f.tile_count);
 for(unsigned n=0;n<f.tile_count;++n)if(tiles[n].tile_flags&C3X_RENDERER_TILE_RENDER){
  unsigned found=~0u;for(unsigned i=0;i<w.native_index.size();++i)if(w.native_index[i]==n)found=i;
  assert(found!=~0u&&native[n]==1000+found);}
  // Prefetch and topology tiles are drawn from but never owned: Civ III
  // rejects a frame that claims a tile it did not capture to draw.
  else assert(native[n]==0);
 // Frames whose anchors are off the linear lattice are left as they are.
 tiles[3].anchor_x+=5;f=frame_for(tiles);assert(!w.build(f,[](int x,int y,c3x_renderer_tile_v1& t){return retained(x,y,t);})&&!w.valid);
 // Selection identity is the box, not the camera.
 ForegroundSelection s1{W,H,TW,TH,2,0,true,false},s2=s1;s1.windowed=s2.windowed=true;s1.box=s2.box={0,0,512,256};
 s1.camera_x=10;s2.camera_x=-74;assert(s1==s2);s2.box[2]=1024;assert(!(s1==s2));
 // A moved window is the same rule: the incremental diff handles a block
 // crossing as entering and leaving tiles instead of a full rebuild.
 assert(s1.same_rule(s2));s2.ring=4;assert(!s1.same_rule(s2));
 std::printf("PASS world window: block-stable occurrences, retained tiles, capture-order output\n");
}
''')


if __name__ == '__main__':
    unittest.main()


class CaptureMarginTests(unittest.TestCase):
    def test_margin_tiles_are_appearance_only(self):
        # The window's capture margin was captured as RENDER tiles, so Civ
        # III's HUD pass drew unit status boxes and city labels for every
        # off-screen unit and city in it: about 100 box outlines per map
        # redraw instead of 22, which backed up the bridge's hit-test queue
        # and tripled the native map pass (performance review, section 30).
        # Tiles past the zoom envelope are prefetch-only, as in the halo.
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('capture_custom_renderer_topology (int viewer, int visibility_mask)')
        body=source[start:source.index('int halo = 12;',start)]
        self.assertIn('RECT bounds = custom_renderer_capture_bounds (screen), envelope = bounds;',body)
        outside=body[body.index('if (x < envelope.left || x > envelope.right || y < envelope.top || y > envelope.bottom) {'):]
        outside=outside[:outside.index('}')]
        self.assertIn('& C3X_RENDERER_TILE_VISIBILITY_BITS)',outside)
        self.assertIn('C3X_RENDERER_TILE_TOPOLOGY_HALO | C3X_RENDERER_TILE_PREFETCH;',outside)
        self.assertNotIn('TILE_RENDER',outside)


class WorldWaterVisibilityTests(unittest.TestCase):
    def test_water_visibility_follows_retained_world_state_not_frame_coverage(self):
        # Water and river records carry water_visible. It came from the
        # current frame's fog coverage, which spans only the captured view:
        # off-coverage water read as hidden and flipped as the camera moved,
        # editing the resident set at most steps (a new revision, so shadows,
        # static proofs and visibility rebuilt), and with the world window the
        # camera job and visual frames disagreed on every step (review,
        # section 28). The tile's retained visibility is the same for both.
        from Renderer.lab.platform import ROOT
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        block = source[source.index("        // Visibility is deliberately absent from geometry identities."):]
        block = block[:block.index("            resource_visibility_revision=topology_cache.visibility_sequence();")]
        self.assertIn("resource_visibility_revision==topology_cache.visibility_sequence()", block)
        self.assertIn("auto record=topology_cache.retained(topology_cache.key(item.tile_x,item.tile_y));", block)
        self.assertNotIn("visibility_coverage.state(", block)
        self.assertNotIn("visibility_coverage.revision", block)
