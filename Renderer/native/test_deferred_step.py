"""A step inside the resident world is drawn once, at its adoption (stage 2c).

Civ III draws its own layers for an adopted step from the step's tile
ownership (replacement flags) before Renderer64 draws the step, so a deferred
step reports ownership from the last drawn window. Flags follow tile content
in window order; the capture order and placement bits change at every step.
"""
import unittest
from pathlib import Path

from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class DeferredStepTests(unittest.TestCase):
    def test_ownership_follows_window_content_across_captures(self):
        run_cpp(r'''
#include "Renderer/native/render_core/deferred_step.h"
#include <cassert>
#include <map>
#include <vector>
using c3x_renderer::render_core::WorldWindow;
using c3x_renderer::render_core::DeferredStep;
constexpr int TW=128,TH=64,HW=64,HH=32,W=1000,H=600,WORLD_W=200,WORLD_H=200;
int roads[WORLD_W][WORLD_H]={};bool unexplored[WORLD_W][WORLD_H]={};
// Civ III's capture: RENDER over the view, PREFETCH 8 coordinates past it,
// topology-only tiles 4 more. Odd steps list the capture in reverse order.
std::vector<c3x_renderer_tile_v1> capture(int cam_x,int cam_y,bool reverse){
 std::vector<c3x_renderer_tile_v1> tiles;
 int x0=cam_x/HW,y0=cam_y/HH,x1=(cam_x+W)/HW,y1=(cam_y+H)/HH;
 for(int y=y0-12;y<=y1+12;++y)for(int x=x0-12;x<=x1+12;++x){
  if((x+y)&1||y<0||y>=WORLD_H)continue;
  bool view=x>=x0-1&&x<=x1&&y>=y0-1&&y<=y1;
  bool appearance=view||(x>=x0-8&&x<=x1+8&&y>=y0-8&&y<=y1+8);
  c3x_renderer_tile_v1 t{};t.tile_x=((x%WORLD_W)+WORLD_W)%WORLD_W;t.tile_y=y;
  t.anchor_x=x*HW-cam_x;t.anchor_y=y*HH-cam_y;t.terrain_type=(x*7+y*3)%11;t.road_mask=roads[t.tile_x][y];
  t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|(unexplored[t.tile_x][y]?0u:C3X_RENDERER_TILE_EXPLORED)|
   (view?C3X_RENDERER_TILE_RENDER:appearance?C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO:C3X_RENDERER_TILE_TOPOLOGY_HALO);
  // Civ III captures city and overlay facts with the view only.
  if(view)t.tile_flags|=C3X_RENDERER_TILE_CITY_BODY_KNOWN|C3X_RENDERER_TILE_NATIVE_OVERLAYS_KNOWN;
  tiles.push_back(t);}
 if(reverse)std::vector<c3x_renderer_tile_v1>(tiles.rbegin(),tiles.rend()).swap(tiles);
 return tiles;}
bool retained(int x,int y,c3x_renderer_tile_v1& t){t={};t.terrain_type=(x*7+y*3)%11;t.road_mask=roads[x][y];
 t.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|(unexplored[x][y]?0u:C3X_RENDERER_TILE_EXPLORED)|C3X_RENDERER_TILE_PREFETCH;return true;}
struct Step {std::vector<c3x_renderer_tile_v1> tiles;c3x_renderer_frame_v1 frame{};WorldWindow window;};
void build(Step& s,int cam_x,int cam_y,bool reverse=false){
 s.tiles=capture(cam_x,cam_y,reverse);
 s.frame.tile_width=TW;s.frame.tile_height=TH;s.frame.target_width=W;s.frame.target_height=H;
 s.frame.world_width_tiles=WORLD_W;s.frame.world_height_tiles=WORLD_H;s.frame.world_wrap_x=1;
 s.frame.tiles=s.tiles.data();s.frame.tile_count=unsigned(s.tiles.size());
 bool ok=s.window.build(s.frame,[](int x,int y,c3x_renderer_tile_v1& t){return retained(x,y,t);});assert(ok);}
// The renderer's flags: content-derived, for every window tile (any placement).
c3x_renderer_u32 owned(c3x_renderer_tile_v1 const& t){
 return C3X_RENDERER_TILE_CUSTOM_TERRAIN_REPLACED|(t.road_mask?C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED:0u)|
  ((t.terrain_type&1)?C3X_RENDERER_TILE_CUSTOM_FEATURE_REPLACED:0u);}
std::vector<c3x_renderer_u32> drawn_flags(Step const& s){
 std::vector<c3x_renderer_u32> flags(s.window.tiles.size(),0u);
 for(unsigned i=0;i<s.window.window_count;++i)flags[i]=owned(s.window.tiles[i]);
 return flags;}
bool same(c3x_renderer_tile_v1 const& a,c3x_renderer_tile_v1 const& b){
 return a.tile_x==b.tile_x&&a.tile_y==b.tile_y&&a.terrain_type==b.terrain_type&&a.road_mask==b.road_mask&&a.tile_flags==b.tile_flags;}
int main(){
 Step drawn;build(drawn,64*40,32*60);
 DeferredStep::Drawn memory;memory.remember(drawn.window,drawn_flags(drawn),1,1);
 assert(memory.valid()&&memory.window.tiles.size()==drawn.window.window_count);
 // Steps inside the block: ownership is the drawn flags in the new capture's
 // order, for its RENDER tiles only, and equals what a draw would report.
 for(int step=1;step<=3;++step)for(bool reverse:{false,true}){
  Step next;build(next,64*40+step*64*2,32*60,reverse);
  std::vector<c3x_renderer_u32> out;
  assert(DeferredStep::ownership(memory,next.window,next.frame.tile_count,same,out));
  std::vector<c3x_renderer_u32> expected;next.window.native_flags(drawn_flags(next),next.frame.tile_count,expected);
  assert(out==expected&&out.size()==next.frame.tile_count);
  unsigned owned_tiles=0;
  for(unsigned i=0;i<out.size();++i){auto const& t=next.tiles[i];
   if(!(t.tile_flags&C3X_RENDERER_TILE_RENDER))assert(!out[i]);
   else {assert(out[i]==owned(t));++owned_tiles;}}
  assert(owned_tiles>100);
 }
 // A content change inside the window refuses (the draw decides ownership).
 roads[60][70]=5;{Step next;build(next,64*40+64*2,32*60);std::vector<c3x_renderer_u32> out;
  assert(!DeferredStep::ownership(memory,next.window,next.frame.tile_count,same,out));}
 roads[60][70]=0;
 // So does a tile becoming unexplored (the window's tile set changes).
 unexplored[52][60]=true;{Step next;build(next,64*40+64*2,32*60);std::vector<c3x_renderer_u32> out;
  assert(next.window.window_count!=drawn.window.window_count);
  assert(!DeferredStep::ownership(memory,next.window,next.frame.tile_count,same,out));}
 unexplored[52][60]=false;
 // A block crossing changes the resident set.
 {Step next;build(next,64*40+64*8,32*60);std::vector<c3x_renderer_u32> out;
  assert(next.window.box!=drawn.window.box);
  assert(!DeferredStep::ownership(memory,next.window,next.frame.tile_count,same,out));}
 // No remembered draw, or flags that do not cover the window: refuse.
 DeferredStep::Drawn none;{Step next;build(next,64*40+64*2,32*60);std::vector<c3x_renderer_u32> out;
  assert(!DeferredStep::ownership(none,next.window,next.frame.tile_count,same,out));
  DeferredStep::Drawn partial;partial.remember(drawn.window,std::vector<c3x_renderer_u32>(3,0u),1,1);
  assert(!partial.valid()&&!DeferredStep::ownership(partial,next.window,next.frame.tile_count,same,out));}
}
''')

    def test_adoption_draws_a_deferred_step_before_publishing_it(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        adoption = source.split('}else if(command==Command::gpu_render){', 1)[1].split('session.publish(initial', 1)[0]
        # The placeholder never reaches the session: the step is drawn first,
        # and a failed draw publishes nothing.
        self.assertIn('if(ready && gpu_publication.deferred)ready=draw_deferred_step();', adoption)
        self.assertLess(adoption.index('draw_deferred_step()'), adoption.index('retain_visual_map(job_frame)'))
        # A drawn step is current: no second draw for the adoption.
        self.assertIn('if(map_sample.prepare && !drawn_now)', adoption)
        job = source.split('bool supported=renderer_state.scene_surface_requested', 1)[1].split('capture_gpu(ready,output', 1)[0]
        self.assertIn('defer_scroll_step(ready)', job)

    def test_steps_report_ownership_by_content_not_the_drawn_mask(self):
        # A draw clears ownership of tiles Civ III did not capture to draw
        # (RENDER). A margin tile entering the view owns by its content, so a
        # deferred step reports from the flags before that mask (a game run
        # found 61-92 entering tiles per step reported as unowned).
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        remember = source.split('void remember_drawn_window(){', 1)[1].split('}', 1)[0]
        self.assertIn('renderer_state.window_content_flags', remember)
        self.assertNotIn('replacement_tile_flags', remember)
        built = source.split('geometry_cache.content_replacement_flags=build_replacement;', 1)[1]
        self.assertLess(built.index('window_content_flags=build_replacement;'),
                        built.index('build_replacement[i] = 0;'))
        covered = source.split('build_replacement=retained_replacements;', 1)[1]
        self.assertLess(covered.index('window_content_flags=retained_replacements;'),
                        covered.index('}else build_replacement[i]=0;'))


if __name__ == '__main__':
    unittest.main()
