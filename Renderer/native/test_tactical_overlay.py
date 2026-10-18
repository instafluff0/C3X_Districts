import unittest
from pathlib import Path
from Renderer.native.native_cpp_test import run_cpp

class TacticalInputTests(unittest.TestCase):
    def test_copied_semantics_clipping_and_bounds(self):
        run_cpp(r'''
#include "Renderer/native/tactical_overlay.h"
#include <cassert>
int main(){using namespace c3x_renderer::tactical;Input a;a.line(-100,20,40,20);a.ring(50,60,128,true);a.label(80,40,"12*",24);
 auto b=a;a.primitives.clear();assert(b.primitives.size()==5&&b.animated);
 auto r=b.extent({0,0,100,80});assert(r[0]==0&&r[2]==100&&r[3]==80);
 bool rejected=false;try{b.line(NAN,0,1,1);}catch(...){rejected=true;}assert(rejected);
 Input c;c.ring(0,0,128,false);assert(!c.animated);
 // Cursor.flc's white stroke centreline, not the larger earlier 44x21.5.
 assert(c.primitives[0].shape[2]==43.5f && c.primitives[0].shape[3]==20.6f);
 Input small;small.ring(0,0,64,false);assert(small.primitives[0].shape[2]==21.75f&&small.primitives[0].shape[3]==10.3f);
 Input custom;custom.ring(0,0,192,false);assert(custom.primitives[0].shape[2]==43.5f);
 // Native centres the selected cursor exactly like the route target's.
 Input selected;selected.ring(7,9,128,true);assert(selected.primitives[0].shape[0]==7.f&&selected.primitives[0].shape[1]==9.f);
 assert(c.primitives[0].shape[1]==0.f);
 assert(cursor_phase(0)==cursor_phase(5));
 assert(cursor_phase(1.25)<cursor_phase(2.5) && cursor_phase(3.75)<cursor_phase(2.5));
 assert(cursor_phase(2.5)>1.57f && cursor_phase(2.5)<1.58f);
 return 0;}
''')

    def test_turn_count_matches_native_text(self):
        # Civ III centres the count on the tile in white, with its map canvas
        # text effect: a black copy at (+2,+2). Digits advance like its bold
        # 16 px font, not one whole atlas cell each.
        run_cpp(r'''
#include "Renderer/native/tactical_overlay.h"
#include <cassert>
#include <cmath>
int main(){using namespace c3x_renderer::tactical;
 Input a;a.label(100,50,"12",20);assert(a.primitives.size()==2);
 auto centre=[&](int i){auto const& s=a.primitives[i].shape;return s[0]+s[2]*.5f;};
 float pitch=centre(1)-centre(0);assert(std::abs(pitch-19.f*.556f*54.f/64.f)<.001f);
 assert(std::abs((centre(0)+centre(1))*.5f-100)<.001f);
 for(auto const& p:a.primitives){
  assert(p.color[0]==1&&p.color[1]==1&&p.color[2]==1&&p.color[3]==1);
  assert(std::abs(p.shape[1]+p.shape[3]*.5f-50)<.001f);
  // Room for the shadow below/right; no outline needs any above/left.
  assert(p.bounds[2]>=p.shape[0]+p.shape[2]+2&&p.bounds[3]>=p.shape[1]+p.shape[3]+2);
 }
 return 0;}
''')

    def test_route_target_stands_on_the_selected_units_ground(self):
        # The resident selected ring is lifted by its body's ground. The held
        # route lifts its own tile-centre geometry by that same function: the
        # target ring, its count and each line endpoint.
        run_cpp(r'''
#include "Renderer/native/tactical_overlay.h"
#include <cassert>
#include <cmath>
int main(){using namespace c3x_renderer::tactical;
 auto ground=[](float x,float y){return x*.01f+y*.1f;};
 Input route;route.line(10,20,110,70);route.ring(110,70,128,false);route.label(110,70,"3",20);
 route.lift(ground);
 auto const& line=route.primitives[0];auto const& target=route.primitives[1];auto const& count=route.primitives[2];
 assert(line.shape[1]==20-ground(10,20)&&line.shape[3]==70-ground(110,70));
 auto near=[](float a,float b){return std::abs(a-b)<.0001f;};
 assert(near(line.bounds[1],line.shape[1]-3.1f)&&near(line.bounds[3],line.shape[3]+3.1f));
 Input selected;selected.ring(110,70-ground(110,70),128,true);
 assert(target.shape[0]==selected.primitives[0].shape[0]&&target.shape[1]==selected.primitives[0].shape[1]);
 assert(near(target.bounds[1],selected.primitives[0].bounds[1])&&near(target.bounds[3],selected.primitives[0].bounds[3]));
 // The count moves with the ring it labels, not by its own glyph cell.
 Input flat;flat.label(110,70,"3",20);
 assert(count.shape[1]==flat.primitives[0].shape[1]-ground(110,70)&&count.shape[0]==flat.primitives[0].shape[0]);
 Input alone;alone.label(110,70,"3",20);auto before=alone.primitives[0];alone.lift(ground);
 assert(std::abs(alone.primitives[0].shape[1]-(before.shape[1]-ground(110,70)))<.0001f);
 return 0;}
''')

    def test_renderer64_lifts_only_the_held_route_by_unit_ground(self):
        root = Path(__file__).resolve().parents[2]
        units = (root / "Renderer/sandbox/direct_units.h").read_text()
        scene = (root / "Renderer/sandbox/resident_scene.cpp").read_text()
        worker = (root / "Renderer/native/c3x_renderer.cpp").read_text()
        self.assertIn("cursors.ring(float(x+4),float(y+4)-ground_lift(frame,float(x),float(y)),", units)
        self.assertIn("return units.ground_frame.tile_count?units.ground_lift(units.ground_frame,x,y):0.f;", scene)
        fresh = scene.split("bool c3x_renderer64_render_fresh(", 1)[1].split("if(!sandbox_fresh.draw(", 1)[0]
        self.assertIn("units.ground_frame=frame;units.ground_tile=first;", fresh)
        tactical = worker.split("}else if(command==Command::tactical){", 1)[1].split("auto area=tactical_input.extent(", 1)[0]
        self.assertIn("if(tactical_input.world_overlay)tactical_input.lift(c3x_renderer64_ground_lift);", tactical)

    def test_owner_disc_footprint_follows_projection(self):
        # The approved soft disc for Civ III's P_SHOW_TEAM_COLOR_DISC: 30x15 at
        # 128px tiles, fading out by 1.15 radii, inside the primitive bounds.
        run_cpp(r'''
#include "Renderer/native/tactical_overlay.h"
#include <cassert>
int main(){using namespace c3x_renderer::tactical;
 Input a;a.owner_disc(10,20,128,{.8f,.1f,.1f,.45f});auto const& p=a.primitives[0];
 assert(p.shape[0]==10&&p.shape[1]==20&&p.shape[2]==30.f&&p.shape[3]==15.f);
 assert(p.style[0]==4.f&&p.color[3]==.45f&&!a.animated);
 assert(p.bounds[0]<=10-30*1.15f&&p.bounds[2]>=10+30*1.15f&&p.bounds[1]<=20-15*1.15f&&p.bounds[3]>=20+15*1.15f);
 Input b;b.owner_disc(0,0,192,{1,1,1,1});assert(b.primitives[0].shape[2]==45.f&&b.primitives[0].shape[3]==22.5f);
 bool rejected=false;try{b.owner_disc(0,0,NAN,{1,1,1,1});}catch(...){rejected=true;}assert(rejected);
 return 0;}
''')

    def test_native_route_fold_uses_copied_map_anchors(self):
        run_cpp(r'''
#include "Renderer/native/tactical_overlay.h"
#include <cassert>
int main(){using namespace c3x_renderer::tactical;
 RouteAnchors anchors;
 // Actual 60x60 world / 2240x1260 window witness: an adjacent vertical
 // segment arrived as y=-948 -> 908 instead of 972 -> 908.
 int copied[][4]={{992,-948,992,972},{992,908,992,908},{928,-916,928,1004},
                  {-1840,700,2000,700},{100,100,100,100},{100,100,3940,100}};
 anchors.assign(copied,6);copied[0][3]=0;
 auto a=anchors.resolve(992,-948,{992,-948},2240,1260);
 auto b=anchors.resolve(992,908,{992,908},2240,1260);
 assert(a[1]==972 && b[1]==908);
 Input route;route.line(a[0],a[1],b[0],b[1]);
 auto area=route.extent({0,0,2240,1260});assert(area[1]==904 && area[3]==976);
 assert(anchors.resolve(928,-916,{0,0},2240,1260)[1]==1004);
 assert(anchors.resolve(-1840,700,{0,0},2240,1260)[0]==2000);
 assert(anchors.resolve(100,100,{0,0},2240,1260)[0]==100);
 // Captured output already incorporates custom zoom; no second transform.
 int zoomed[]={992,-948,1376,1314};anchors.assign(zoomed,1);
 assert(anchors.resolve(992,-948,{1488,-1422},2240,1260)[1]==1314);
 auto missing=anchors.resolve(12,34,{18,51},2240,1260);assert(missing[0]==18&&missing[1]==51);
 anchors.assign(nullptr,0);assert(anchors.resolve(992,-948,{992,-948},2240,1260)[1]==-948);
 bool refused=false;try{anchors.assign(zoomed,8193);}catch(...){refused=true;}assert(refused);
}''')

    def test_production_gpu_coverage_and_packed_composition(self):
        from Renderer.lab.platform import native_command_result
        result = native_command_result("Renderer/native", "call BUILD.bat tactical")
        self.assertEqual(result["status"], "pass", result["output_tail"])
        self.assertIn("PASS tactical GPU:", result["output_tail"])
