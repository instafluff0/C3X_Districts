"""The static layer frees its zoom lane and previews after zoom rests at 1x.

Both zoom lanes kept front and back slots, each with a retained water-overlay
layer, plus two preview images: about 0.55 GB at this resolution, kept for
the whole session after the first zoom (performance review, section 20).
After ten seconds at a complete, settled 1x layer, the zoom lane's slots and
overlays and both previews are released; the next zoom re-creates them.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class StaticZoomLaneReleaseTests(unittest.TestCase):
    def test_release_frees_zoom_lane_and_previews_only_when_settled(self):
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        release = method(source, "    void release_zoom_lane(")
        compose = source[source.index("    bool compose_static(ViewportShaderSettings& settings,int w,int h){"):]
        gate = compose[compose.index("        auto key=static_key();"):compose.index("        float hint=zoom_destination();")]
        for condition in ("lane==1", "zoom_destination()!=1.f", "static_rasters.refining()",
                          "!static_rasters.front(0).fresh(key)", "std::chrono::seconds(10)", "release_zoom_lane();"):
            self.assertIn(condition, gate)
        run_cpp(r'''
#include <array>
#include <cassert>
#include <cstdint>
#include "Renderer/sandbox/static_raster_state.h"
#include <cstdio>
struct Target {int* color=nullptr;int allocated=0;void reset(){color=nullptr;++allocated;}std::size_t bytes()const{return color?1:0;}};
struct Trace {int writes=0;void write(char const*,char const*,bool){++writes;}};
struct Renderer {Trace trace;} renderer;
struct Inputs {bool cleared=false;void clear(){cleared=true;}};
struct Pipeline {
 using StaticRasters=c3x_renderer::render_core::StaticRasterStates<Target>;
 using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
 StaticRasters static_rasters;std::array<Inputs,4> raster_inputs;
 struct OverlaySlot {Target layer;std::uint64_t revision=~0ull;};std::array<OverlaySlot,4> overlay_slots;
 std::array<StaticState,2> bootstrap;std::array<StaticState const*,2> bootstrap_ring{};
''' + release + r'''
};
int main(){
 Pipeline p;int memory=0;
 for(unsigned i=0;i<4;++i){p.static_rasters.states[i].region.color=&memory;p.static_rasters.states[i].valid=true;
  p.static_rasters.states[i].covered={0,0,8,8};p.overlay_slots[i].layer.color=&memory;p.overlay_slots[i].revision=7;}
 for(auto& image:p.bootstrap){image.region.color=&memory;image.valid=true;image.covered={0,0,4,4};}
 p.bootstrap_ring[0]=&p.static_rasters.states[0];
 p.release_zoom_lane();
 // The 1x lane (slots 0 and 1) is untouched.
 for(unsigned i=0;i<2;++i){auto const& s=p.static_rasters.states[i];
  assert(s.region.color && s.valid && !s.covered.empty() && p.overlay_slots[i].layer.color && p.overlay_slots[i].revision==7 && !p.raster_inputs[i].cleared);}
 // The zoom lane loses its targets, coverage and proofs; previews are gone.
 for(unsigned i=2;i<4;++i){auto const& s=p.static_rasters.states[i];
  assert(!s.region.color && !s.valid && s.covered.empty() && !p.overlay_slots[i].layer.color && p.overlay_slots[i].revision==~0ull && p.raster_inputs[i].cleared);}
 for(auto const& image:p.bootstrap)assert(!image.region.color && !image.valid && image.covered.empty());
 assert(!p.bootstrap_ring[0] && !p.bootstrap_ring[1] && renderer.trace.writes==1);
 return 0;
}
''')

    def test_zoomed_view_frees_only_the_idle_1x_back_slot(self):
        """Zoomed at 2x-3x on the busy save, the 1x lane still held both slots
        and overlays (about 210 MB) beside the zoom lane (review 87). While the
        zoom lane is selected the 1x lane is read only through its front, the
        previews' home source; its back slot and overlay are freed then, and
        the next 1x refinement or recenter allocates them again."""
        source = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()
        compose = source[source.index("    bool compose_static(ViewportShaderSettings& settings,int w,int h){"):]
        gate = compose[compose.index("        auto key=static_key();"):compose.index("        float hint=zoom_destination();")]
        run_cpp(r'''
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "Renderer/sandbox/static_raster_state.h"
struct Target {int* color=nullptr;std::size_t bytes()const{return color?52u<<20:0;}void reset(){color=nullptr;}};
struct Trace {int writes=0;char last[64]={};void write(char const* stage,char const*,bool){++writes;std::snprintf(last,sizeof(last),"%s",stage);}};
struct Renderer {Trace trace;} renderer;
struct Inputs {bool cleared=false;void clear(){cleared=true;}};
struct Pipeline {
 using StaticRasters=c3x_renderer::render_core::StaticRasterStates<Target>;
 using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
 StaticRasters static_rasters;std::array<Inputs,4> raster_inputs;
 struct OverlaySlot {Target layer;std::uint64_t revision=~0ull;};std::array<OverlaySlot,4> overlay_slots;
 std::array<StaticState,2> bootstrap;int released_lanes=0;float destination=1.f;
 std::chrono::steady_clock::time_point zoom_lane_used=std::chrono::steady_clock::now();
 c3x_renderer::render_core::StaticRasterKey static_key()const{return {};}
 float zoom_destination()const{return destination;}
 void release_zoom_lane(){++released_lanes;}
 void gate(unsigned lane){
''' + gate + r'''
 }
};
int main(){
 Pipeline p;int memory=0;
 for(unsigned i=0;i<4;++i){p.static_rasters.states[i].region.color=&memory;p.static_rasters.states[i].valid=true;
  p.static_rasters.states[i].covered={0,0,8,8};p.overlay_slots[i].layer.color=&memory;p.overlay_slots[i].revision=7;}
 // At 1x the back slot stays: scrolling recentres into it.
 p.gate(0);assert(p.static_rasters.states[1].region.color && p.overlay_slots[1].layer.color && renderer.trace.writes==0);
 // Zoomed: the idle 1x back slot and its overlay go; the 1x front (the
 // previews' home source) and the zoom lane's slots stay.
 p.destination=2.f;p.gate(1);
 auto const& back=p.static_rasters.states[1];
 assert(!back.region.color && !back.valid && back.covered.empty() && !p.overlay_slots[1].layer.color &&
  p.overlay_slots[1].revision==~0ull && p.raster_inputs[1].cleared);
 for(unsigned i:{0u,2u,3u})assert(p.static_rasters.states[i].region.color && p.static_rasters.states[i].valid &&
  p.overlay_slots[i].layer.color && p.overlay_slots[i].revision==7 && !p.raster_inputs[i].cleared);
 assert(renderer.trace.writes==1 && !std::strcmp(renderer.trace.last,"static-home-back-released"));
 p.gate(1);assert(renderer.trace.writes==1); // nothing left to free
 // After a 1x promotion the other slot is the back slot.
 p.static_rasters.promote(0);assert(p.static_rasters.front_slot[0]==1);
 p.static_rasters.states[1].region.color=&memory;p.gate(1);
 assert(!p.static_rasters.states[0].region.color && p.static_rasters.states[1].region.color && p.released_lanes==0);
 std::puts("PASS static zoom: the idle 1x back slot is freed while zoomed");
 return 0;
}
''')


if __name__ == '__main__':
    unittest.main()
