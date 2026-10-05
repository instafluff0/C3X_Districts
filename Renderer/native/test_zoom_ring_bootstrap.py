"""Execute the production zoom-out bootstrap against its preview mapping.

Zooming out exposes a ring around the displayed raster. The bootstrap draws
only that ring; its undrawn hole must never be sampled, i.e. every screen pixel
whose bootstrap texel (with bilinear/depth taps) falls in the hole must be
covered by the displayed raster, which the preview always prefers.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class ZoomRingBootstrapTests(unittest.TestCase):
    def test_ring_bootstrap_hole_is_always_behind_the_displayed_raster(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        methods = '\n'.join(method(source, signature) for signature in (
            '    bool render_bootstrap(', '    int resample_source('))
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
struct D3D11_RECT{long left,top,right,bottom;};
constexpr unsigned D3D11_CLEAR_DEPTH=1,D3D11_CLEAR_STENCIL=2;
struct Target {unsigned width=0,height=0;int plane=0;int* samples=&plane;int* depth_samples=&plane;int* target=&plane;int* depth=&plane;
 void reset(){}std::size_t bytes()const{return 0;}};
struct Context {void ClearRenderTargetView(int*,float const*){}void ClearDepthStencilView(int*,unsigned,int,int){}};
namespace c3x_renderer{namespace render_core{
struct LinearResample{struct Source{int* color=nullptr;int* depth=nullptr;float map[4]{},covered[4]{},size[2]{},depth_shift=0;};};}}
using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
using StaticRect=StaticState::Rect;
struct ViewportShaderSettings{float depth_translation=0;};
struct Options{float bootstrap_scale=1.f;};
struct Pipeline {
 Context context;
 struct{Context* context;int content_view_width=2240,content_view_height=1192;}renderer{&context};
 std::array<StaticState,2> bootstrap;std::array<StaticState const*,2> bootstrap_ring{};std::array<std::uint64_t,2> bootstrap_ring_revision{};
 std::array<std::array<std::uint64_t,6>,2> bootstrap_stamp{};
 bool overlay_enabled()const{return true;}
 std::array<std::uint64_t,6> bootstrap_identity()const{return {};}
 Options options;Options const& sandbox_perf_options()const{return options;}
 struct Work{void clear(int*){}}work;
 int camera_x=0,camera_y=0;static constexpr int region_margin_x=320,region_margin_y=192;
 unsigned region_width_px=2248+640,region_height_px=1200+384;
 float projection_zoom=1,destination=1;unsigned bootstrap_draws=0;
 struct{double boot_draw_ms=0,boot_deps_ms=0;long long boot_area=0;}static_decision;
 StaticRect hole{};std::vector<StaticRect> strips;
 c3x_renderer::render_core::StaticRasterKey static_key()const{return {};}
 float zoom_destination()const{return destination;}
 struct ZoomScope{ZoomScope(Pipeline&,float){}};
 ViewportShaderSettings slot_settings(StaticState const&,ViewportShaderSettings s)const{return s;}
 bool ensure_linear_target(Target& t,unsigned w,unsigned h,unsigned,bool){t.width=w;t.height=h;return true;}
 // Production strip growth draws the needed rectangle minus the initial area.
 bool extend_coverage(unsigned,StaticState& slot,ViewportShaderSettings const&,StaticRect needed,double&,int,float,bool){
  hole=slot.covered;strips.clear();
  if(slot.covered.empty()){strips.push_back(needed);slot.covered=needed;return true;}
  auto a=slot.covered;
  if(needed.bottom>a.bottom){strips.push_back({a.left,a.bottom,a.right,needed.bottom});a.bottom=needed.bottom;}
  if(needed.top<a.top){strips.push_back({a.left,needed.top,a.right,a.top});a.top=needed.top;}
  if(needed.left<a.left){strips.push_back({needed.left,a.top,a.left,a.bottom});a.left=needed.left;}
  if(needed.right>a.right){strips.push_back({a.right,a.top,needed.right,a.bottom});a.right=needed.right;}
  slot.covered=a;return true;}
''' + methods + r'''
};
int main(){unsigned checked=0;
 for(auto step:std::vector<std::array<float,2>>{{.875f,.75f},{.75f,.625f},{.625f,.5f},{1.f,.875f}})
 for(bool partial:{false,true}){
  Pipeline p;StaticState front;ViewportShaderSettings settings;
  front.valid=true;front.projection=step[0];front.camera_x=37;front.camera_y=-19;
  front.covered={0,0,int(p.region_width_px),int(p.region_height_px)};
  // A front whose guard band is still being filled covers less.
  if(partial)front.covered={300,150,int(p.region_width_px)-200,int(p.region_height_px)-100};
  p.camera_x=40;p.camera_y=-20;p.destination=step[1];p.projection_zoom=step[0]-.01f;
  int w=2248,h=1200;
  assert(p.render_bootstrap(1,settings,w,h,&front));
  assert(p.bootstrap_ring[1]==&front && !p.hole.empty());
  // The ring is drawn exactly once: hole + strips == needed.
  long long drawn=0;for(auto const& r:p.strips)drawn+=r.area();
  assert(drawn+p.hole.area()==p.bootstrap[1].covered.area());
  assert(drawn*100<p.bootstrap[1].covered.area()*70);
  // Every intermediate animation frame down to the destination.
  for(float zoom=step[0]-.01f;zoom>=step[1]-1e-4f;zoom-=.01f){
   p.projection_zoom=zoom;c3x_renderer::render_core::LinearResample::Source f,b;
   int front_cover=p.resample_source(front,w,h,settings,f);
   p.resample_source(p.bootstrap[1],w,h,settings,b);
   if(front_cover==0)continue;
   for(int y=0;y<h;y+=5)for(int x=0;x<w;x+=5){float sx=x+.5f,sy=y+.5f;
    float p0x=sx*f.map[0]+f.map[2],p0y=sy*f.map[1]+f.map[3];
    bool in_front=p0x>=f.covered[0]&&p0y>=f.covered[1]&&p0x<=f.covered[2]&&p0y<=f.covered[3];
    float p1x=sx*b.map[0]+b.map[2],p1y=sy*b.map[1]+b.map[3];
    // Texels within reach of the hole (bilinear + depth neighbors).
    bool near_hole=p1x>=p.hole.left-2&&p1y>=p.hole.top-2&&p1x<=p.hole.right+2&&p1y<=p.hole.bottom+2;
    if(near_hole)assert(in_front);++checked;}
  }
  // Without a displayed raster the whole view is drawn.
  Pipeline full;full.camera_x=40;full.camera_y=-20;full.destination=step[1];full.projection_zoom=step[1];
  assert(full.render_bootstrap(1,settings,w,h,nullptr));assert(full.bootstrap_ring[1]==nullptr&&full.hole.empty());
 }
 std::printf("PASS zoom ring bootstrap: samples=%u hole_behind_front=1 ring_exact=1\n",checked);
}
''')


if __name__ == '__main__':
    unittest.main()
