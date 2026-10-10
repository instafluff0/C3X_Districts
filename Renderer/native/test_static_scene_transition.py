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
            '    bool static_pixels_current(', '    bool zoom_moving(', '    bool compose_static('))
        run_cpp(r'''
#include "Renderer/sandbox/static_raster_state.h"
#include "Renderer/sandbox/scroll_region.h"
#include <algorithm>
#include <chrono>
#include <array>
#include <cassert>
#include <climits>
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
 // Settled views: native or minified (source texels per screen pixel >= 1), never magnified.
 bool require_native_scale=false;
 bool ensure(int){return true;}
 bool draw(Context*,Target& target,Source const& source,Source const* secondary,int layers=-1){
  if(layers>=0)return true; // overlay composite: no color/depth change in this model
  assert(source.color&&source.depth);
  if(require_native_scale){assert(source.map[0]>=1.f&&source.map[1]>=1.f);
   if(secondary)assert(secondary->map[0]>=1.f&&secondary->map[1]>=1.f);}
  target.colors=*source.color;target.depths=*source.depth;return true;
 }
};}}
using StaticRasters=c3x_renderer::render_core::StaticRasterStates<Target>;
using StaticState=c3x_renderer::render_core::StaticRasterState<Target>;
using StaticRect=StaticState::Rect;
struct ViewportShaderSettings{float translation[2]{},depth_translation=0,projection=1;};
struct Revisions{struct Checkpoint{int owner=0,sequence=0;};Checkpoint checkpoint()const{return {};}};
struct Options{bool legacy=false;float bootstrap_scale=1.f,zoom_refine_hold=.01f;};
struct Pipeline {
 Context context;
 struct{Context* context;int device=1,content_view_width=4,content_view_height=4;std::int64_t scene_depth_origin=0;
  bool borrowed_scene_frame=false;Revisions raster_dependency_revisions;}renderer{&context};
 StaticRasters static_rasters;std::array<StaticState,2> bootstrap;std::array<StaticState const*,2> bootstrap_ring{};std::array<std::uint64_t,2> bootstrap_ring_revision{};
 struct RasterInputs{using Revisions=::Revisions;int version=0;bool complete=true;void clear(){version=0;complete=true;}std::size_t bytes()const{return 0;}
  void carry(Revisions const&,int,Revisions::Checkpoint){}};
 int raster_validation_key(ViewportShaderSettings const&,D3D11_RECT){return 0;}
 std::chrono::steady_clock::time_point zoom_lane_used{};void release_zoom_lane(){}
 std::array<RasterInputs,4> raster_inputs;std::array<RasterInputs,2> bootstrap_inputs;
 std::array<std::array<std::uint64_t,6>,2> bootstrap_stamp{};
 bool restore_overlays=false,preview_overlays=false;
 struct OverlaySlot{Target layer;std::uint64_t revision=~0ull;};std::array<OverlaySlot,4> overlay_slots;
 bool overlay_enabled()const{return false;}
 std::array<std::uint64_t,6> bootstrap_identity()const{return {std::uint64_t(version),0,0,0,0,0};}
 Options options;Options const& sandbox_perf_options()const{return options;}
 struct ShadowChange{std::uint64_t serial=0;std::array<int,4> source{};};
 struct{bool atlas_complete=true;std::uint64_t change_serial=0,change_floor=0;std::vector<ShadowChange> shadow_changes;}shadow;
 std::vector<std::array<int,4>> shadow_dirty,repaired_dirty;float resident_basis_x=0,resident_basis_y=0;int wrap_pixels=0;
 struct Work{bool enabled=false;struct Counts{std::uint64_t target_pixels=0;}counts;
  void draw(unsigned){}void clear(Plane*){}void copy(Plane*,bool){}Counts& row(){return counts;}}work;
 struct Restore{bool draw(Context*,Target& target,Plane* color,Plane* depth,int,int,std::vector<int>,void*,unsigned,unsigned,
  bool,bool,int,void*,int,float=0){target.colors=*color;target.depths=*depth;return true;}}static_restore;
 c3x_renderer::render_core::LinearResample static_resample;
 Target static_cache;std::array<std::uint64_t,12> restore_key{};
 struct OverlayFrame{bool valid=false;unsigned slot=0;int move_x=0,move_y=0;float depth_shift=0;} overlay_frame;
 double last_static_ms=0;bool static_preview=false;
 struct {unsigned lane=0;int reusable=-1,recenter=0,shifted=0,refine=0,sync=0,preview=0,front_cover=0,home_cover=0,boot_cover=0;
  long long missing=0;int camera_x=0,camera_y=0,slot_x=0,slot_y=0;unsigned entry=0,home_entry=0,repair=0,key_diff=0;std::size_t input_bytes=0;double boot_draw_ms=0,boot_deps_ms=0;long long boot_area=0;
  double proof_ms=0,repair_ms=0,recenter_ms=0,refine_ms=0,strip_ms=0,restore_ms=0;} static_decision;
 bool layout_reset=false;
 int camera_x=0,camera_y=0,region_margin_x=6,region_margin_y=6;
 unsigned region_width_px=16,region_height_px=16,scene_samples=1;
 float projection_zoom=1,destination=0;std::array<unsigned,2> lane_still{{3,3}};
 bool refine_worked=false;unsigned refine_restarts=0,refine_slices=0,cache_full_draws=0,refine_promotions=0,cache_scrolls=0,preview_frames=0,bootstrap_draws=0;
 int version=1;Plane world{};bool allow_repair=false;unsigned repairs=0;
 c3x_renderer::render_core::StaticRasterKey key{};
 auto static_key(){return key;}float zoom_destination()const{return destination>0?destination:projection_zoom;}
 double available_budget=0;double refinement_budget(unsigned){return available_budget;}
 struct ZoomScope{Pipeline& p;float old;ZoomScope(Pipeline& p,float zoom):p(p),old(p.projection_zoom){p.projection_zoom=zoom;}~ZoomScope(){p.projection_zoom=old;}};
 ViewportShaderSettings slot_settings(StaticState const& slot,ViewportShaderSettings s){s.projection=slot.projection;return s;}
 bool raster_dependencies(RasterInputs& inputs,ViewportShaderSettings const& settings,D3D11_RECT,bool append){
  assert(settings.projection==projection_zoom);
  if(append){inputs.version=version;return true;}return inputs.complete&&inputs.version==version;
 }
 bool ensure_linear_target(Target&,unsigned,unsigned,unsigned,bool){return true;}
 void paint(StaticState& slot){slot.region.colors=world;for(unsigned i=0;i<16;++i)slot.region.depths[i]=world[i]+1000;}
 StaticRect shadow_field{INT_MIN/4,INT_MIN/4,INT_MAX/4,INT_MAX/4},repaired_ring;bool hidden=false;
 bool canonical_hidden()const{return hidden;}
 StaticRect source_region_rect(std::int64_t l,std::int64_t t,std::int64_t r,std::int64_t b,ViewportShaderSettings const&,int)const{
  return {int(l),int(t),int(r),int(b)};}
 bool repair_front(unsigned index,StaticState& slot,ViewportShaderSettings const&,StaticRect ring=StaticRect()){
  repaired_dirty=shadow_dirty;repaired_ring=ring;if(!allow_repair)return false;paint(slot);raster_inputs[index].version=version;++slot.revision;++repairs;return true;
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
   paint(s);bootstrap_inputs[i].version=version;bootstrap_stamp[i]=bootstrap_identity();}
 }
 void check(){ViewportShaderSettings settings;assert(compose_static(settings,4,4));
  // Retained overlays may composite only from the slot this frame restored;
  // a resampled preview never composites them (and biases live water).
  assert(!overlay_frame.valid||(overlay_frame.slot<4&&static_rasters.states[overlay_frame.slot].valid));
  assert(!(overlay_frame.valid&&static_preview));
  assert(static_cache.colors==world);
  for(unsigned i=0;i<16;++i)assert(static_cache.depths[i]==world[i]+1000);
 }
};
int main(){unsigned frames=0;
 // A reveal can finish the hidden canonical lane before the visible zoom lane.
 // Neither it nor an old zoom preview may soften (magnify) a settled view
 // during repair; a current 1x raster may serve a zoomed-out view, minified.
 for(float zoom:{.5f,.625f,.75f,.875f,1.25f,1.5f,1.75f,2.f,2.5f,3.f}){
  Pipeline p;p.projection_zoom=p.destination=zoom;p.edit(0);p.seed();
  auto visible=StaticRasters::lane_of(zoom);
  p.static_rasters.front(visible).valid=false;
  p.bootstrap[visible].valid=false;p.available_budget=0;
  p.static_resample.require_native_scale=true;p.check();
  if(zoom>1.f)assert(p.bootstrap[visible].valid&&p.bootstrap[visible].projection==zoom);
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
 // A shadow caster that enters the field after a repair changes baked
 // pixels without changing any contributor proof; its journaled footprint
 // repairs the retained raster once, then later frames prove again.
 for(float zoom:{1.f,3.f}){
  Pipeline p;p.projection_zoom=p.destination=zoom;p.edit(0);p.seed();p.allow_repair=true;p.check();
  auto lane=StaticRasters::lane_of(zoom);auto repairs=p.repairs;
  for(unsigned i=0;i<16;++i)p.world[i]+=50000;
  p.shadow.shadow_changes.push_back({1,{0,0,8,8}});p.shadow.change_serial=1;
  p.check();++frames;assert(p.repairs==repairs+1&&p.static_rasters.front(lane).shadow_serial==1);
  p.check();++frames;assert(p.repairs==repairs+1);
 }
 // Footprints are chunk coordinates: repairs add the resident basis that
 // draw records use, plus both wrapped copies on a wrapping map.
 {Pipeline p;p.projection_zoom=p.destination=1.f;p.edit(0);p.seed();p.allow_repair=true;p.check();
  p.resident_basis_x=-5280;p.resident_basis_y=-1900;p.wrap_pixels=7680;
  for(unsigned i=0;i<16;++i)p.world[i]+=50000;
  p.shadow.shadow_changes.push_back({1,{10,20,30,40}});p.shadow.change_serial=1;p.check();
  assert(p.repaired_dirty.size()==3);
  assert((p.repaired_dirty[0]==std::array<int,4>{10-5280-7680,20-1900,30-5280-7680,40-1900}));
  assert((p.repaired_dirty[1]==std::array<int,4>{10-5280,20-1900,30-5280,40-1900}));
  assert((p.repaired_dirty[2]==std::array<int,4>{10-5280+7680,20-1900,30-5280+7680,40-1900}));}
 // Pixels drawn while the receiver field was narrower (the hidden canonical
 // lane during a zoom-in) repair once the displayed lane's field covers
 // them; a ring too broad to repair refines behind the current pixels.
 {Pipeline p;p.projection_zoom=p.destination=1.f;p.edit(0);p.seed();p.allow_repair=true;p.check();
  auto repairs=p.repairs;p.static_rasters.front(0).unshadowed={0,0,16,3};
  p.shadow_field={0,8,16,16};p.check();assert(p.repairs==repairs); // still outside the field
  p.shadow_field={-100,-100,100,100};p.hidden=true;p.check();assert(p.repairs==repairs); // field is the destination's
  p.hidden=false;p.shadow.atlas_complete=false;p.check();assert(p.repairs==repairs); // pages not drawn yet
  p.shadow.atlas_complete=true;for(unsigned i=0;i<16;++i)p.world[i]+=50000;
  p.check();assert(p.repairs==repairs+1&&p.repaired_ring.top==0&&p.repaired_ring.bottom==3&&p.repaired_ring.right==16);
  assert(p.static_rasters.front(0).unshadowed.empty());
  p.check();assert(p.repairs==repairs+1);
  p.static_rasters.front(0).unshadowed={0,0,16,16};for(unsigned i=0;i<16;++i)p.world[i]+=50000;
  p.available_budget=500;p.check();assert(p.repairs==repairs+1);
  assert(p.static_rasters.front(0).valid&&p.static_rasters.front(0).unshadowed.empty());}
 // Refinement toward a zoom destination waits while the zoom visibly moves:
 // under Parallels its GPU cost made transition frames 30-180 ms (review, 43).
 // Within the hold of the destination, and with the hold off, it refines.
 for(float hold:{.01f,0.f}){
  Pipeline p;p.options.zoom_refine_hold=hold;p.projection_zoom=1.25f;p.edit(0);p.seed();
  p.destination=1.5f;p.lane_still={{0,0}};p.available_budget=500;
  p.check();assert((p.refine_slices==0)==(hold>0));
  p.projection_zoom=1.4951f;p.check();assert(p.refine_slices>0);
 }
 // A zoom that stops short of its destination refines toward where it is.
 {Pipeline p;p.projection_zoom=1.25f;p.edit(0);p.seed();p.destination=1.5f;p.available_budget=500;
  p.static_rasters.front(1).projection=1.1f;p.check();assert(p.refine_slices>0);}
 std::printf("PASS semantic terrain transitions: intermediate_frames=%u zooms=4 repair_and_bootstrap=1 color_depth_coherent=1 cached_lane_return=1 shadow_journal_repair=1 unshadowed_ring=1 zoom_refine_hold=1\n",frames);
}
''')


if __name__ == '__main__':
    unittest.main()
