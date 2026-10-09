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


if __name__ == '__main__':
    unittest.main()
