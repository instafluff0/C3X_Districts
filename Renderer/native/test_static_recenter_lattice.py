"""Execute the production static-raster recenter at ladder and animating zooms.

Scrolling past a retained raster's guard band copies the overlapping pixels
into the lane's other slot. At a ladder zoom k/8 most camera steps are not a
whole number of raster pixels; the new slot must still land on the source
raster's lattice so the copy is exact, and its retained overlay layer must
move with it.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class StaticRecenterLatticeTests(unittest.TestCase):
    def test_fractional_camera_steps_recenter_on_the_source_lattice(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        recenter = method(source, '    bool recenter(')
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include "Renderer/sandbox/scroll_region.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
struct D3D11_RECT {long left,top,right,bottom;};
struct Target {unsigned width=0,height=0;int id=0;int* samples=&id;int* depth_samples=&id;void reset(){}std::size_t bytes()const{return 0;}};
using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
using StaticRect=StaticState::Rect;
struct ViewportShaderSettings {};
struct Options {bool legacy=false;};
struct Harness {
    c3x_renderer::render_core::StaticRasterStates<Target> static_rasters;
    struct Inputs {bool complete=true;void clear(){complete=true;}};
    std::array<Inputs,4> raster_inputs;
    struct OverlaySlot {Target layer;std::uint64_t revision=~0ull;};
    std::array<OverlaySlot,4> overlay_slots;
    struct Context {void OMSetRenderTargets(int,void*,void*){}} context;
    struct {Context* context;} renderer{&context};
    struct Work {void draw(int){}} work;
    struct Copy {int target,source,dx,dy;};
    struct Restore {
        std::vector<Copy> copies;
        bool draw(Context*,Target& target,int* color,int*,int dx,int dy,std::vector<D3D11_RECT> const&,void*,
                unsigned,unsigned,bool,bool,int,D3D11_RECT const*,int scale,float shift){
            assert(scale==1&&shift==0.f);copies.push_back({target.id,*color,dx,dy});return true;}
    } static_restore;
    Options options;Options const& sandbox_perf_options()const{return options;}
    int camera_x=0,camera_y=0;
    static constexpr int region_margin_x=320,region_margin_y=192;
    unsigned region_width_px=2888,region_height_px=1584,scene_samples=1,recenter_copies=0;
    float projection_zoom=1;bool overlays=true;
    struct {int recenter=0;} static_decision;
    bool overlay_enabled()const{return overlays;}
    bool overlay_target(unsigned){return true;}
    bool ensure_linear_target(Target&,unsigned,unsigned,unsigned,bool){return true;}
    struct ZoomScope {Harness& h;float old;ZoomScope(Harness& o,float z):h(o),old(o.projection_zoom){o.projection_zoom=z;}~ZoomScope(){h.projection_zoom=old;}};
    ViewportShaderSettings slot_settings(StaticState const&,ViewportShaderSettings const&)const{return {};}
    bool raster_dependencies(Inputs&,ViewportShaderSettings const&,D3D11_RECT,bool){return true;}
''' + recenter + r'''
};
int main(){
    unsigned cases=0;
    for(float zoom:{.5f,.625f,.75f,.875f,1.f,1.5f,.6231f})for(int step:{701,-703,661,-663,997}){
        Harness h;unsigned lane=zoom==1.f?0u:1u;h.static_rasters.select_lane(lane);
        unsigned from=h.static_rasters.front_slot[lane],to=h.static_rasters.back_index(lane);
        for(unsigned i=0;i<4;++i){h.static_rasters.states[i].region.id=int(10+i);h.overlay_slots[i].layer.id=int(20+i);}
        auto& source=h.static_rasters.states[from];
        source.valid=true;source.projection=zoom;source.camera_x=1001;source.camera_y=-503;
        source.covered={0,0,2888,1584};h.overlay_slots[from].revision=source.revision;
        // A diagonal scroll that has just left the guard band.
        h.camera_x=source.camera_x+step;h.camera_y=source.camera_y+step/2;
        auto shift=c3x_renderer::render_core::StaticRegionShift::between(zoom,h.camera_x,h.camera_y,
            source.camera_x,source.camera_y,Harness::region_margin_x,Harness::region_margin_y);
        if(shift.reusable)continue;
        bool ladder=std::abs(zoom*8-std::round(zoom*8))<1e-4;
        bool moved=h.recenter(lane,ViewportShaderSettings{},shift,2240,1192);
        if(!ladder){assert(!moved&&h.static_restore.copies.empty());++cases;continue;}
        assert(moved&&h.recenter_copies==1);
        auto& destination=h.static_rasters.states[to];
        assert(h.static_rasters.front_slot[lane]==to&&destination.valid&&destination.projection==zoom);
        // The anchor is on the source lattice: its raster offset is whole.
        double px=double(zoom)*(destination.camera_x-source.camera_x),py=double(zoom)*(destination.camera_y-source.camera_y);
        assert(px==std::floor(px)&&py==std::floor(py));
        // The anchor stays within half a lattice step of the camera.
        assert(std::abs(destination.camera_x-h.camera_x)<=4&&std::abs(destination.camera_y-h.camera_y)<=4);
        // Color/depth and the retained overlay layer move by the same whole shift.
        assert(h.static_restore.copies.size()==2);
        auto const& color=h.static_restore.copies[0];auto const& layer=h.static_restore.copies[1];
        assert(color.target==int(10+to)&&color.source==int(10+from)&&color.dx==int(px)&&color.dy==int(py));
        assert(layer.target==int(20+to)&&layer.source==int(20+from)&&layer.dx==color.dx&&layer.dy==color.dy);
        assert(h.overlay_slots[to].revision==destination.revision);
        // The current view is then a reusable retained shift of the new slot.
        assert(shift.reusable&&std::abs(shift.snap_x)<=.5&&std::abs(shift.snap_y)<=.5);
        ++cases;
    }
    assert(cases==35);
    std::printf("PASS static recenter lattice: cases=%u ladder_whole_shift=1 overlay_moves=1 animating_refused=1\n",cases);
}
''')


if __name__ == '__main__':
    unittest.main()
