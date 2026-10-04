"""Execute the complete production terrain selection across semantic edits.

GPU submission is replaced by tagged color/depth planes. These tests check every
published intermediate selection, including both zoom lanes and the bootstrap;
the D3D pixel oracles and captured game run separately verify GPU composition.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method


class StaticSceneTransitionTests(unittest.TestCase):
    def test_no_old_terrain_plane_can_join_current_dynamic_layers(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        methods = '\n'.join(method(source, signature) for signature in (
            '    bool render_bootstrap(', '    int resample_source(',
            '    bool static_pixels_current(', '    bool compose_static('))
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include "Renderer/sandbox/scroll_region.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
using LONG=int;
struct D3D11_RECT{int left,top,right,bottom;};
constexpr unsigned D3D11_CLEAR_DEPTH=1,D3D11_CLEAR_STENCIL=2,DXGI_FORMAT_R16G16B16A16_FLOAT=0;
using Plane=std::array<int,16>;
struct Target {
 unsigned width=16,height=16;Plane colors{},depths{};
 Plane* samples=&colors;Plane* depth_samples=&depths;Plane* color=&colors;Plane* resolved=&colors;
 Plane* target=&colors;Plane* depth=&depths;
 void reset(){}std::size_t bytes()const{return 128;}
};
struct Context {
 void ClearRenderTargetView(Plane* p,float const*){p->fill(0);}
 void ClearDepthStencilView(Plane* p,unsigned,int,int){p->fill(0);}
 void OMSetRenderTargets(int,void*,void*){}
 void CopyResource(Plane* d,Plane* s){*d=*s;}
 void ResolveSubresource(Plane* d,int,Plane* s,int,int){*d=*s;}
};
namespace c3x_renderer{namespace render_core{
struct LinearResample {
 struct Source{Plane* color=nullptr;Plane* depth=nullptr;float map[4]{},covered[4]{},size[2]{},depth_shift=0;};
 bool ensure(int){return true;}
 bool draw(Context*,Target& target,Source const& source,Source const* secondary){
  assert(source.color&&source.depth);target.colors=*source.color;target.depths=*source.depth;return true;
 }
};}}
using StaticRasters=c3x_renderer::render_core::StaticRasterStates<Target>;
using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
using StaticRect=StaticState::Rect;
struct ViewportShaderSettings{float translation[2]{},depth_translation=0;};
struct Options{bool legacy=false;float bootstrap_scale=.5f;};
struct Pipeline {
 Context context;
 struct{Context* context;int device=1,content_view_width=4,content_view_height=4;std::int64_t scene_depth_origin=0;}renderer{&context};
 StaticRasters static_rasters;std::array<StaticState,2> bootstrap;
 struct RasterInputs{int version=0;bool complete=true;void clear(){version=0;complete=true;}};
 std::array<RasterInputs,4> raster_inputs;std::array<RasterInputs,2> bootstrap_inputs;
 Options options;Options const& sandbox_perf_options(){return options;}
 struct{bool atlas_complete=false;}shadow;
 struct Work{bool enabled=false;struct Counts{std::uint64_t target_pixels=0;}counts;
  void draw(unsigned){}void clear(Plane*){}void copy(Plane*,bool){}Counts& row(){return counts;}}work;
 struct Restore{bool draw(Context*,Target& target,Plane* color,Plane* depth,int,int,std::vector<int>,void*,unsigned,unsigned,
  bool,bool,int,void*,int,float=0){target.colors=*color;target.depths=*depth;return true;}}static_restore;
 c3x_renderer::render_core::LinearResample static_resample;
 Target static_cache;std::array<std::uint64_t,12> restore_key{};
 int camera_x=0,camera_y=0,region_margin_x=6,region_margin_y=6;
 unsigned region_width_px=16,region_height_px=16,scene_samples=1;
 float projection_zoom=1;std::array<unsigned,2> lane_still{{3,3}};
 bool refine_worked=false;unsigned refine_restarts=0,refine_slices=0,cache_full_draws=0,refine_promotions=0,cache_scrolls=0,preview_frames=0,bootstrap_draws=0;
 int version=1;Plane world{};bool allow_repair=false;unsigned repairs=0;
 c3x_renderer::render_core::StaticRasterKey key{};
 auto static_key(){return key;}float zoom_destination(){return projection_zoom;}
 double refinement_budget(unsigned){return 500.;}
 struct ZoomScope{Pipeline& p;float old;ZoomScope(Pipeline& p,float zoom):p(p),old(p.projection_zoom){p.projection_zoom=zoom;}~ZoomScope(){p.projection_zoom=old;}};
 ViewportShaderSettings slot_settings(StaticState const&,ViewportShaderSettings const& s){return s;}
 bool raster_dependencies(RasterInputs& inputs,ViewportShaderSettings const&,D3D11_RECT,bool append){
  if(append){inputs.version=version;return true;}return inputs.complete&&inputs.version==version;
 }
 bool ensure_linear_target(Target&,unsigned,unsigned,unsigned,bool){return true;}
 void paint(StaticState& slot){slot.region.colors=world;for(unsigned i=0;i<16;++i)slot.region.depths[i]=world[i]+1000;}
 bool repair_front(unsigned index,StaticState& slot,ViewportShaderSettings const&){
  if(!allow_repair)return false;paint(slot);raster_inputs[index].version=version;++slot.revision;++repairs;return true;
 }
 bool recenter(unsigned,ViewportShaderSettings const&,c3x_renderer::render_core::StaticRegionShift&,int,int){return false;}
 bool reset_slot(unsigned index,float zoom,ViewportShaderSettings const&){auto& s=static_rasters.states[index];
  s.valid=s.stale=false;s.refining=true;s.covered={};s.projection=zoom;s.camera_x=camera_x;s.camera_y=camera_y;s.key=key;
  ++s.revision;raster_inputs[index].clear();return true;
 }
 bool extend_coverage(unsigned index,StaticState& slot,ViewportShaderSettings const&,StaticRect needed,double& budget,int,float,bool track){
  if(budget==0)return true;paint(slot);slot.covered=needed;++slot.revision;
  if(track)raster_inputs[index].version=version;
  if(budget>0)budget=0;return true;
 }
''' + methods + r'''
 void edit(int kind){++version;for(unsigned i=0;i<16;++i)world[i]=version*100+int(i);
  // Hide/reveal and city foundations change only part of the colored plane.
  if(kind==1)for(unsigned i=0;i<8;++i)world[i]=0;
  if(kind==2)world[8]=7777;
 }
 void seed(){for(unsigned i=0;i<4;++i){auto& s=static_rasters.states[i];
  s.valid=true;s.projection=i<2?1.f:projection_zoom;s.covered={0,0,16,16};paint(s);raster_inputs[i].version=version;}
  for(unsigned i=0;i<2;++i){auto& s=bootstrap[i];s.valid=true;s.projection=projection_zoom;s.covered={0,0,16,16};
   paint(s);bootstrap_inputs[i].version=version;}
 }
 void check(){ViewportShaderSettings settings;assert(compose_static(settings,4,4));
  assert(static_cache.colors==world);
  for(unsigned i=0;i<16;++i)assert(static_cache.depths[i]==world[i]+1000);
 }
};
int main(){unsigned frames=0;
 for(float zoom:{.5f,1.f,1.25f,3.f})for(bool repair:{false,true})for(int change=0;change<4;++change){
  Pipeline p;p.projection_zoom=zoom;p.edit(0);p.seed();p.allow_repair=repair;
  p.check();++frames;
  p.edit(change);
  // An environment change used to skip content checks entirely.
  if(change==3)++p.key[1];
  for(unsigned n=0;n<10;++n){p.check();++frames;} // slow shadow/refinement preparation
  p.shadow.atlas_complete=true;p.check();++frames;
  p.projection_zoom=zoom==1.f?1.25f:1.f;p.check();++frames; // return to cached other lane
  p.projection_zoom=zoom;p.check();++frames;
  // Another reveal while only bootstrap pixels can be displayed.
  p.shadow.atlas_complete=false;p.edit(1);p.check();++frames;p.edit(2);p.check();++frames;
 }
 std::printf("PASS semantic terrain transitions: intermediate_frames=%u zooms=4 repair_and_bootstrap=1 color_depth_coherent=1 cached_lane_return=1\n",frames);
}
''')


if __name__ == '__main__':
    unittest.main()
