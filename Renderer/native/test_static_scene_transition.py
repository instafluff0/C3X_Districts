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
    def test_default_fallback_preserves_native_pixel_resolution(self):
        source = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        options = source[source.index('struct SandboxPerfOptions {'):source.index('inline SandboxPerfOptions const&')]
        run_cpp(r'''
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>
char const* override_scale=nullptr;
unsigned GetEnvironmentVariableA(char const* name,char* value,unsigned){
 if(std::strcmp(name,"C3X_RENDERER_BOOTSTRAP_SCALE") || !override_scale)return 0;
 std::strcpy(value,override_scale);return unsigned(std::strlen(value));
}
''' + options + r'''
int main(){assert(SandboxPerfOptions{}.bootstrap_scale==1.f);
 override_scale="nan";assert(SandboxPerfOptions{}.bootstrap_scale==1.f);
 override_scale="0.5";assert(SandboxPerfOptions{}.bootstrap_scale==.5f);
 override_scale="0";assert(SandboxPerfOptions{}.bootstrap_scale==0.f);
}
''')

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
 bool require_native_scale=false;
 bool ensure(int){return true;}
 bool draw(Context*,Target& target,Source const& source,Source const* secondary){
  assert(source.color&&source.depth);
  if(require_native_scale){assert(source.map[0]==1.f&&source.map[1]==1.f);
   if(secondary)assert(secondary->map[0]==1.f&&secondary->map[1]==1.f);}
  target.colors=*source.color;target.depths=*source.depth;return true;
 }
};}}
using StaticRasters=c3x_renderer::render_core::StaticRasterStates<Target>;
using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
using StaticRect=StaticState::Rect;
struct ViewportShaderSettings{float translation[2]{},depth_translation=0,projection=1;};
struct Options{bool legacy=false;float bootstrap_scale=1.f;};
struct Pipeline {
 Context context;
 struct{Context* context;int device=1,content_view_width=4,content_view_height=4;std::int64_t scene_depth_origin=0;}renderer{&context};
 StaticRasters static_rasters;std::array<StaticState,2> bootstrap;
 struct RasterInputs{int version=0;bool complete=true;void clear(){version=0;complete=true;}};
 std::array<RasterInputs,4> raster_inputs;std::array<RasterInputs,2> bootstrap_inputs;
 Options options;Options const& sandbox_perf_options(){return options;}
 struct{bool atlas_complete=true;}shadow;
 struct Work{bool enabled=false;struct Counts{std::uint64_t target_pixels=0;}counts;
  void draw(unsigned){}void clear(Plane*){}void copy(Plane*,bool){}Counts& row(){return counts;}}work;
 struct Restore{bool draw(Context*,Target& target,Plane* color,Plane* depth,int,int,std::vector<int>,void*,unsigned,unsigned,
  bool,bool,int,void*,int,float=0){target.colors=*color;target.depths=*depth;return true;}}static_restore;
 c3x_renderer::render_core::LinearResample static_resample;
 Target static_cache;std::array<std::uint64_t,12> restore_key{};
 int camera_x=0,camera_y=0,region_margin_x=6,region_margin_y=6;
 unsigned region_width_px=16,region_height_px=16,scene_samples=1;
 float projection_zoom=1,destination=0;std::array<unsigned,2> lane_still{{3,3}};
 bool refine_worked=false;unsigned refine_restarts=0,refine_slices=0,cache_full_draws=0,refine_promotions=0,cache_scrolls=0,preview_frames=0,bootstrap_draws=0;
 int version=1;Plane world{};bool allow_repair=false;unsigned repairs=0;
 c3x_renderer::render_core::StaticRasterKey key{};
 auto static_key(){return key;}float zoom_destination(){return destination>0?destination:projection_zoom;}
 double available_budget=0;double refinement_budget(unsigned){return available_budget;}
 struct ZoomScope{Pipeline& p;float old;ZoomScope(Pipeline& p,float zoom):p(p),old(p.projection_zoom){p.projection_zoom=zoom;}~ZoomScope(){p.projection_zoom=old;}};
 ViewportShaderSettings slot_settings(StaticState const& slot,ViewportShaderSettings s){s.projection=slot.projection;return s;}
 bool raster_dependencies(RasterInputs& inputs,ViewportShaderSettings const& settings,D3D11_RECT,bool append){
  assert(settings.projection==projection_zoom);
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
 // A reveal can finish the hidden canonical lane before the visible zoom lane.
 // Neither it nor an old zoom preview may soften a settled view during repair.
 for(float zoom:{.5f,.625f,.75f,.875f,1.25f,1.5f,1.75f,2.f,2.5f,3.f}){
  Pipeline p;p.projection_zoom=p.destination=zoom;p.edit(0);p.seed();
  auto visible=StaticRasters::lane_of(zoom);
  p.static_rasters.front(visible).valid=false;
  p.bootstrap[visible].valid=false;p.available_budget=0;
  p.static_resample.require_native_scale=true;p.check();
  assert(p.bootstrap[visible].valid&&p.bootstrap[visible].projection==zoom);
 }

 // Outward bootstrap covers the destination before intermediate zoom frames.
 for(float goal:{.5f,.625f,.75f,.875f}){
  Pipeline p;p.projection_zoom=1;p.destination=goal;ViewportShaderSettings settings;
  assert(p.render_bootstrap(1,settings,4,4));assert(p.bootstrap[1].projection==goal);
  p.projection_zoom=goal;c3x_renderer::render_core::LinearResample::Source source;
  assert(p.resample_source(p.bootstrap[1],4,4,settings,source)==2);
  assert(source.map[0]==1&&source.map[1]==1);
 }

 for(float zoom:{.5f,1.f,1.25f,3.f})for(bool repair:{false,true})for(int change=0;change<4;++change){
  Pipeline p;p.projection_zoom=zoom;p.edit(0);p.seed();p.allow_repair=repair;
  p.check();++frames;
  p.edit(change);
  // An environment change used to skip content checks entirely.
  if(change==3)++p.key[1];
  for(unsigned n=0;n<10;++n){p.check();++frames;} // exhausted per-frame raster budget
  p.available_budget=500;p.check();++frames;
  p.projection_zoom=zoom==1.f?1.25f:1.f;p.check();++frames; // return to cached other lane
  p.projection_zoom=zoom;p.check();++frames;
  // Another reveal while only bootstrap pixels can be displayed.
  p.available_budget=0;p.edit(1);p.check();++frames;p.edit(2);p.check();++frames;
 }
 std::printf("PASS semantic terrain transitions: intermediate_frames=%u zooms=4 repair_and_bootstrap=1 color_depth_coherent=1 cached_lane_return=1\n",frames);
}
''')


if __name__ == '__main__':
    unittest.main()
