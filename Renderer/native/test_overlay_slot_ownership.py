"""Execute the production static strip writer's retained-overlay bookkeeping.

Each static raster slot owns a retained layer of near-water overlays that is
usable only while its revision equals the slot's. Bootstrap images reuse the
same strip writer with their own state and index 0, so they must never retire
or write slot 0's layer.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class OverlaySlotOwnershipTests(unittest.TestCase):
    def test_bootstrap_writes_leave_slot_overlays_and_slot_writes_extend_them(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        writer = method(source, '    bool write_slot(')
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include <array>
#include <cassert>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <vector>
struct D3D11_RECT {long left,top,right,bottom;};
struct Target {unsigned width=64,height=48;int* target=nullptr;int* depth=nullptr;void reset(){}std::size_t bytes()const{return 0;}};
using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
using StaticRect=StaticState::Rect;
struct ViewportShaderSettings {};
struct Record {bool water_dependent=false;};
struct GeometryDrawReference {Record const& r;explicit GeometryDrawReference(Record const& v):r(v){}};
constexpr unsigned layers=4;
struct GeometryDrawView {using Records=std::array<std::vector<Record>,layers>;};
struct Harness {
    c3x_renderer::render_core::StaticRasterStates<Target> static_rasters;
    StaticState bootstrap_image;
    struct OverlaySlot {Target layer;std::uint64_t revision=~0ull;};
    std::array<OverlaySlot,4> overlay_slots;
    std::uint64_t overlay_retired_mismatch=0,overlay_retired_failure=0;
    struct {bool water_scene_active=true;
        bool chunk_intersects_region(GeometryDrawReference,ViewportShaderSettings const&,D3D11_RECT,bool)const{return true;}
        struct {void OMSetRenderTargets(int,void*,void*){}} context_value;decltype(context_value)* context=&context_value;} renderer;
    std::vector<Record> world{{false},{true},{true}};
    unsigned overlay_writes=0,static_writes=0;float projection_zoom=1;
    static constexpr int region_margin_x=8,region_margin_y=8;
    struct ZoomScope {ZoomScope(Harness&,float){}};
    struct TargetScope {TargetScope(Harness&,std::array<long,5>){}};
    struct RasterInputs {bool complete=true;};std::array<RasterInputs,4> raster_inputs;
    bool overlay_enabled()const{return true;}
    // Receiver field in source pixels; the slot view maps it unchanged here.
    StaticRect shadow_field{0,0,64,24};
    StaticRect source_region_rect(long l,long t,long r,long b,ViewportShaderSettings const&,int)const{return {int(l),int(t),int(r),int(b)};}
    ViewportShaderSettings slot_settings(StaticState const&,ViewportShaderSettings const&)const{return {};}
    D3D11_RECT source_bounds(ViewportShaderSettings const&,D3D11_RECT r,bool)const{return r;}
    template<class Visit>void contributors(ViewportShaderSettings const&,D3D11_RECT,bool,Visit visit)const{
        for(auto const& r:world)visit(r.water_dependent?1u:0u,r);}
    bool draw_scene(GeometryDrawView::Records const&,ViewportShaderSettings const&,D3D11_RECT,int*,int*,bool,float){++static_writes;return true;}
    bool draw_static_borders(Target const&,ViewportShaderSettings const&,D3D11_RECT,GeometryDrawView::Records const&,float){return true;}
    bool raster_dependencies(RasterInputs&,ViewportShaderSettings const&,D3D11_RECT,bool){return true;}
    bool write_overlay_strip(unsigned,StaticState const&,ViewportShaderSettings const&,D3D11_RECT,GeometryDrawView::Records& records,float){
        assert(records[1].size()==2&&records[0].empty());++overlay_writes;return true;}
''' + writer.replace('TargetScope scope_target(*this,{slot.region.target,slot.region.width,slot.region.height,\n            float(region_margin_x),float(region_margin_y)});',
                     'TargetScope scope_target(*this,{0,0,0,0,0});') + r'''
};
int main(){
    Harness h;ViewportShaderSettings screen;
    auto& slot=h.static_rasters.states[0];auto& retained=h.overlay_slots[0];
    retained.revision=slot.revision;
    // A real slot strip extends its layer and carries the revision forward.
    assert(h.write_slot(0,slot,screen,{0,0,64,16},1.f,false));
    assert(h.overlay_writes==1&&retained.revision==slot.revision);
    // A bootstrap image written through index 0 leaves slot 0's layer alone.
    auto before=retained.revision;
    assert(h.write_slot(0,h.bootstrap_image,screen,{0,0,64,48},1.f,false));
    assert(h.overlay_writes==1&&retained.revision==before&&retained.revision==slot.revision);
    assert(h.overlay_retired_mismatch==0&&h.overlay_retired_failure==0);
    // The slot remains extendable afterwards.
    assert(h.write_slot(0,slot,screen,{0,16,64,32},1.f,false));
    // Strip pixels outside the shadow receiver field are recorded (bootstrap
    // images are not static slots and record nothing).
    assert((slot.unshadowed.left==0&&slot.unshadowed.top==24&&slot.unshadowed.right==64&&slot.unshadowed.bottom==32));
    assert(h.bootstrap_image.unshadowed.empty());
    assert(h.overlay_writes==2&&retained.revision==slot.revision&&h.static_writes==3);
    std::printf("PASS retained overlay slot ownership: bootstrap_isolated=1 slot_extends=1\n");
}
''')


if __name__ == '__main__':
    unittest.main()
