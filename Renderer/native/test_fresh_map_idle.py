"""Execute the resident fresh-map callback without a GPU or display clock."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.source_fidelity.prepare import function


class FreshMapIdleTests(unittest.TestCase):
    def test_camera_import_samples_current_pose_before_publishing(self):
        source = Path(__file__).with_name("c3x_renderer.cpp").read_text()
        refresh = source.split("// The native overlay stream may adopt this camera", 1)[1].split("#endif", 1)[0]
        refresh = refresh[refresh.index("camera_scene_complete=true;"):]
        run_cpp(r'''
#include <cassert>
#include <functional>
struct Texture {};
struct Worker {
 bool camera_scene_complete=false;void retire_completed_scene(){}
 long long visual_ticks=1000,visual_frequency=1000;
 struct {long long resumed=0;void resume_motion(long long t,long long){resumed=t;}} unit_instances;
 struct Frame {bool ready=false;struct {Texture value;Texture* Get(){return &value;}} front;} frame;
 Frame* prepared_map=&frame;
 Texture saved;
 void advance_visual_clock(){visual_ticks=1500;}
 Texture* adopt(bool resource_ready){
  auto initial=&saved;
  struct {std::function<void(long long,long long,float)> prepare;} map_sample;
  map_sample.prepare=[&](long long t,long long f,float z){
   assert(camera_scene_complete&&t==1500&&f==1000&&z==1.f);
   assert(unit_instances.resumed==t);
   frame.ready=resource_ready;
  };
''' + refresh + r'''
  return initial;
 }
};
int main(){
 Worker worker;
 assert(worker.adopt(true)==worker.frame.front.Get()); // never re-import the old pose
 assert(worker.adopt(false)==&worker.saved); // pending assets retain a complete image
}
''')

    def test_retained_prepare_observes_visible_facts_before_gpu_work(self):
        source = Path(__file__).with_name("c3x_renderer.cpp").read_text()
        prepared = source.split("    struct PreparedMapFrame {", 1)[1].split(
            "    // One preparation owner", 1)[0]
        callback = source.split("        if(fresh_map){", 1)[1].split(
            "\n        }\n#endif", 1)[0]
        loading = source.split('} else if(command==Command::prepare_world_loading){', 1)[1].split(
            '} else if (command == Command::require_world_changes)', 1)[0]
        retirement = loading.split('#ifdef C3X_RENDERER64_FRESH', 1)[1].split('#endif', 1)[0]
        eligibility = function(source, "frame_has_resource_animation")
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "Renderer/native/input_recording/codec.h"
#include "Renderer/native/render_core/dynamic_scene_input.h"
#include "Renderer/native/render_core/unit_instances.h"
#include "Renderer/native/render_core/unit_hud_anchors.h"
#include "Renderer/native/render_core/unit_arrival_visibility.h"
#include "Renderer/native/render_core/water_material_frame.h"
#include "Renderer/native/render_core/retained_scene_view.h"
#include <optional>
using namespace c3x_renderer::render_core;
struct D3D11_TEXTURE2D_DESC {unsigned Width=128,Height=64;};
struct ID3D11Texture2D {void GetDesc(D3D11_TEXTURE2D_DESC* d){*d={};}};
struct ID3D11RenderTargetView {};
namespace Microsoft {namespace WRL {template<class T>struct ComPtr {
 T* value=nullptr;ComPtr()=default;ComPtr(T* p):value(p){}
 T* Get()const{return value;}T* operator->()const{return value;}
 explicit operator bool()const{return value!=nullptr;}T** operator&(){return &value;}
 ComPtr& operator=(T* p){value=p;return *this;}
};}}
bool FAILED(int result){return result<0;}
struct Device {
 ID3D11Texture2D texture;ID3D11RenderTargetView target;
 unsigned allocations=0;
 int CreateTexture2D(D3D11_TEXTURE2D_DESC*,void*,ID3D11Texture2D** out){++allocations;*out=&texture;return 0;}
 int CreateRenderTargetView(ID3D11Texture2D*,void*,ID3D11RenderTargetView** out){*out=&target;return 0;}
};
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* value){value->QuadPart=1;}
unsigned GetEnvironmentVariableA(char const*,char*,unsigned){return 0;}
namespace c3x_renderer{namespace render_core{unsigned cached_environment(char const* n,char* b,unsigned s){return GetEnvironmentVariableA(n,b,s);}}}
unsigned renders=0,mesh_prepares=0;
std::uint64_t c3x_renderer64_unit_selection_revision(){return 1;}
int c3x_renderer64_prepare_unit_meshes(){++mesh_prepares;return C3X_RENDERER_RESULT_OK;}
bool c3x_renderer64_render_fresh(c3x_renderer_frame_v1 const&,ID3D11RenderTargetView*,float){++renders;return true;}
// Retained static refinement complete: idle frames depend only on scene facts.
bool c3x_renderer64_static_refinement_pending(float){return false;}
namespace c3x_gpu_images {struct RetainedComposition {
 struct SampledImage {
  enum class Kind {unchanged,bgra,frozen,held};Kind kind=Kind::unchanged;std::uint64_t generation=0;
  static SampledImage frozen(){return {Kind::frozen};}
  static SampledImage held(){return {Kind::held};}
  struct Rect {int left,top,right,bottom;};
  static SampledImage bgra(ID3D11Texture2D*,Rect,float,std::uint64_t generation=0){return {Kind::bgra,generation};}
 };
 struct Sample {
  std::function<SampledImage(long long,long long)> canonical;
  std::function<SampledImage(long long,long long,float)> projected;
  std::function<void(long long,long long,float)> prepare;
  std::uint64_t source_generation=0;
  std::shared_ptr<UnitHudAnchors> unit_anchors;
  template<class F>Sample(F f):canonical(std::move(f)){}
  SampledImage operator()(long long t,long long f){return canonical(t,f);}
 };
};}
struct Clip {std::string name="idle";bool ambient=true,loop=true;double duration=1.;unsigned frames=16;};
struct Unit {std::vector<std::string> keys={"warrior"};std::vector<Clip> actions={Clip{},Clip{"move"}};};
struct RendererState {
 struct {std::uint64_t geometry=1,complete=1;} cached_signature;
 std::uint64_t tile_geometry_epoch=1;std::int64_t gpu_serial=0;unsigned device_generation=1;
 std::int64_t route_map_serial=0;
 std::uint64_t route_frame_sequence=1;
 struct {int value=1;} geometry_viewport_settings;
 Device owned,*device=&owned;ID3D11Texture2D initial;ID3D11Texture2D* gpu_map_texture=&initial;
 struct {std::vector<Unit> units=std::vector<Unit>(1);} unit_bodies;
 std::vector<UnitInstances::ScenePose> fresh_unit_poses;
 UnitArrivalVisibility arrival_visibility;
 unsigned moving_resources=0,visible_wave_animations=0,visible_water_animations=0,asset_prepares=0;
 bool water_scene_active=true,wave_ready=true,visibility_pass=true;
 std::vector<int> resource_animations;
 int resource_animation_for(c3x_renderer_tile_v1 const& tile)const{return tile.resource_id==101?0:-1;}
 bool animated_composition_for(c3x_renderer_tile_v1 const&)const{return false;}
 bool assets_pending=false,reject_selection=false;unsigned selections=0;
 bool borrowed_scene_frame=false,borrowed_scene_stale=false; // camera-job snapshot frames (review 4v)
 // Contributor selection has separate executable geometry fixtures. This
 // adapter fixture supplies their visible-tile result to test ownership/idle.
 bool select_frame_units(c3x_renderer_frame_v1 const& frame,
  std::vector<UnitInstances::ScenePose> const& candidates,
  std::vector<UnitInstances::ScenePose>& selected,float){
  ++selections;if(reject_selection)return false;
  selected.clear();for(auto const& pose:candidates)for(unsigned i=0;i<frame.tile_count;++i){
   auto const& tile=frame.tiles[i];if(tile.tile_x==pose.tile_x && tile.tile_y==pose.tile_y &&
     (tile.tile_flags&C3X_RENDERER_TILE_VISIBLE)){selected.push_back(pose);break;}
  }return true;
 }
 unsigned ambient_count()const{return moving_resources+visible_wave_animations+visible_water_animations;}
 int prepare_frame_unit_assets(std::vector<UnitInstances::ScenePose> const&){++asset_prepares;
  return assets_pending?C3X_RENDERER_RESULT_PENDING:C3X_RENDERER_RESULT_OK;}
 struct {int level=1;LARGE_INTEGER frequency{1000};void write(char const*,char const*,bool){}double milliseconds(long long){return 0.;}} trace;
''' + eligibility + r'''
};
struct Worker {
 RendererState renderer_state;UnitInstances unit_instances;DynamicSceneInputs dynamic_inputs;
 bool camera_active=false,camera_scene_complete=true;
 using View=RetainedSceneView<decltype(RendererState::cached_signature),std::uint64_t>;
 std::optional<View> completed_scene;
 bool completed_scene_usable()const{return bool(completed_scene);}
 bool completed_scene_borrowable()const{return (camera_active||!camera_scene_complete)&&completed_scene_usable();}
 auto borrow_completed_scene(){return View::Borrow(
  (camera_active||!camera_scene_complete)&&completed_scene?&*completed_scene:nullptr,
  std::tie(renderer_state.cached_signature,renderer_state.tile_geometry_epoch));}
 void retire_completed_scene(){completed_scene.reset();}
 c3x_renderer_frame_v1 job_frame{};c3x_renderer_camera_identity_v1 job_camera_identity{};
 struct Publication {bool projection_matches=true;bool matches_projection(c3x_renderer_frame_v1 const&,
  c3x_renderer_camera_identity_v1 const&)const{return projection_matches;}} gpu_publication;
 unsigned visual_map_samples=0;
 struct PreparedMapFrame {''' + prepared + r'''
 std::shared_ptr<PreparedMapFrame> prepared_map;
 void retire_loading_view(){''' + retirement + r'''}
 c3x_gpu_images::RetainedComposition::Sample make(){
  using Sampled=c3x_gpu_images::RetainedComposition::SampledImage;
  auto capture=dynamic_inputs.capture(job_frame,job_camera_identity);
  auto selected=dynamic_inputs.capture(job_frame,job_camera_identity);assert(capture&&selected);
  long long origin=1000;int x=0,y=0,w=128,h=64;
''' + callback + r'''
 }
 void ambient(){renderer_state.visible_water_animations=0;
  for(unsigned i=0;i<job_frame.tile_count;++i)
   renderer_state.visible_water_animations+=water_scene_tile(job_frame.tiles[i],job_frame,true);
 }
 void body(int id,int x,int y,unsigned flags=C3X_RENDERER_UNIT_STATE_CAPTURED,unsigned color=0){
  c3x_renderer_unit_state_v1 state{};state.struct_size=sizeof(state);state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;
  state.unit_id=id;state.tile_x=x;state.tile_y=y;state.action=1;state.max_hp=3;
  state.visible=1;state.presentation_frequency=1000;assert(unit_instances.state(state));
  c3x_renderer_unit_v1 draw{};draw.struct_size=sizeof(draw);draw.unit_id=id;draw.action=1;draw.frame_count=16;
  draw.sprite_width=draw.sprite_height=191;draw.projection_scale_milli=1000;draw.presentation_frequency=1000;
  draw.display_color_rgb=color;
  std::strcpy(draw.unit_key,"warrior");UnitInstances::Selection selected;
  bool accepted=unit_instances.capture(draw,flags,renderer_state.unit_bodies.units,[](int){return "idle";},selected);
  assert(accepted==!(flags&C3X_RENDERER_UNIT_HIDDEN));
 }
};
int main(){
 using Kind=c3x_gpu_images::RetainedComposition::SampledImage::Kind;
 Worker worker;c3x_renderer_tile_v1 tile{};tile.tile_x=tile.tile_y=4;tile.terrain_type=tile.real_terrain_type=12;
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN;
 auto& frame=worker.job_frame;frame.api_version=C3X_RENDERER_API_VERSION;frame.struct_size=sizeof(frame);
 frame.tiles=&tile;frame.tile_count=1;frame.target_width=frame.tile_width=128;frame.target_height=frame.tile_height=64;
 frame.presentation_time_ticks=1000;frame.presentation_frequency=1000;
 worker.ambient();auto sample=worker.make();++worker.renderer_state.gpu_serial;
 worker.renderer_state.visible_water_animations=1; // A cached different view grants no eligibility.
 unsigned imports=0;long long tick=1000;
 auto step=[&](Kind expected,float zoom=1.f){tick+=100;
  sample.prepare(tick,1000,zoom);auto canonical=sample(tick,1000);
  assert(canonical.kind==(zoom==1.f||expected==Kind::frozen||expected==Kind::held?expected:Kind::unchanged));
  auto projected=sample.projected(tick,1000,zoom);
  assert(projected.kind==(expected==Kind::unchanged?Kind::held:expected));
  imports+=projected.kind==Kind::bgra;
 };
 // Unknown water and its later clock samples do not touch assets or GPU output.
 for(int i=0;i<3;++i)step(Kind::unchanged);
 assert(!renders&&!imports&&!mesh_prepares&&!worker.renderer_state.asset_prepares&&!worker.renderer_state.owned.allocations);
 // Offscreen-only explored water remains idle after its authoritative adoption.
 tile.tile_flags|=C3X_RENDERER_TILE_EXPLORED;tile.anchor_x=128;
 ++worker.renderer_state.cached_signature.complete;worker.ambient();step(Kind::bgra);
 unsigned before=renders,assets=worker.renderer_state.asset_prepares;
 worker.renderer_state.visible_water_animations=1;
 for(int i=0;i<3;++i)step(Kind::unchanged);
 assert(renders==before&&worker.renderer_state.asset_prepares==assets);
 // Prefetch alone never makes the current view animate either.
 tile.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_EXPLORED;tile.anchor_x=0;
 ++worker.renderer_state.cached_signature.complete;worker.ambient();step(Kind::bgra);before=renders;
 step(Kind::unchanged);assert(renders==before);
 // Actual on-screen explored water continually adopts the current cosmetic clock.
 tile.tile_flags=C3X_RENDERER_TILE_RENDER|C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
 ++worker.renderer_state.cached_signature.complete;worker.ambient();
 for(int i=0;i<3;++i)step(Kind::bgra);
 assert(worker.prepared_map->input.presentation_time_ticks==tick);
 // Settle a visible static land map. A later native body needs just one adoption.
 tile.terrain_type=tile.real_terrain_type=2;tile.tile_flags|=C3X_RENDERER_TILE_VISIBLE;
 ++worker.renderer_state.cached_signature.complete;worker.ambient();step(Kind::bgra);
 before=renders;worker.body(7,4,4);step(Kind::bgra);step(Kind::unchanged);
 assert(renders==before+1&&worker.prepared_map->poses.size()==1);
 // Repeat native captures change ordering/generation, not the displayed body proof.
 worker.body(7,4,4);before=renders;step(Kind::unchanged);assert(renders==before);
 worker.body(7,4,4,C3X_RENDERER_UNIT_STATE_CAPTURED,0xabcdef);
 before=renders;step(Kind::bgra);step(Kind::unchanged);assert(renders==before+1);
 // Hidden/offscreen-only records never wake GPU work on this static view.
 worker.body(8,6,6);worker.body(9,4,4,C3X_RENDERER_UNIT_STATE_CAPTURED|C3X_RENDERER_UNIT_HIDDEN);
 before=renders;assets=worker.renderer_state.asset_prepares;step(Kind::unchanged);
 assert(renders==before&&worker.renderer_state.asset_prepares==assets);
 // Changed visible body facts are adopted, and removing that body is adopted once.
 worker.body(7,4,4,C3X_RENDERER_UNIT_STATE_CAPTURED|C3X_RENDERER_UNIT_SELECTED);
 step(Kind::bgra);step(Kind::bgra); // admitted selected idle animation
 worker.unit_instances.forget(7);step(Kind::bgra);before=renders;step(Kind::unchanged);assert(renders==before);
 // An unfinished visible asset cannot settle merely because the next pose matches.
 worker.body(7,4,4);worker.renderer_state.assets_pending=true;
 before=renders;step(Kind::held);step(Kind::held);assert(renders==before&&worker.prepared_map->poses.empty());
 worker.renderer_state.assets_pending=false;step(Kind::bgra);step(Kind::unchanged);
 // Traveling native units and ambient resources keep their existing delivery.
 c3x_renderer_unit_move_v1 move{};move.struct_size=sizeof(move);move.unit_id=7;
 move.old_x=move.old_y=4;move.new_x=6;move.new_y=4;move.action=2;
 move.source_visible=move.target_visible=1;move.presentation_frequency=1000;move.presentation_time_ticks=tick;
 assert(worker.unit_instances.begin_motion(move,100,100,false,false));
 step(Kind::bgra);assert(worker.prepared_map->poses[0].travelling);step(Kind::bgra);
 // A reveal preparation holds the displayed pose, but must not permanently
 // retire the retained callback while its ordered native import is pending.
 auto held_pose=worker.prepared_map->poses[0].draw.body_x;before=renders;
 worker.camera_active=true;worker.camera_scene_complete=false;worker.unit_instances.pause_motion(tick);
 step(Kind::held);step(Kind::held);assert(renders==before);
 // Cancelled scratch is not a completed scene, even between worker jobs.
 worker.camera_active=false;
 step(Kind::held);assert(renders==before);
 step(Kind::held);assert(renders==before);
 worker.camera_scene_complete=true;worker.unit_instances.resume_motion(tick,1000);
 ++worker.renderer_state.cached_signature.complete;
 step(Kind::bgra);assert(renders==before+1&&worker.prepared_map->poses[0].draw.body_x>held_pose);
 // A retained completed view advances actors during preparation and through
 // cancellation gaps, while partial signatures never become the frame input.
 worker.completed_scene.emplace(std::tie(worker.renderer_state.cached_signature,worker.renderer_state.tile_geometry_epoch));
 worker.camera_active=true;worker.camera_scene_complete=false;
 auto completed_signature=worker.prepared_map->signature;
 worker.renderer_state.cached_signature.complete+=100;
 auto pending_signature=worker.renderer_state.cached_signature.complete;
 before=renders;step(Kind::bgra);
 assert(renders==before+1&&worker.prepared_map->signature==completed_signature);
 assert(worker.renderer_state.cached_signature.complete==pending_signature);
 worker.camera_active=false;step(Kind::bgra); // cancellation cannot expose scratch
 assert(worker.prepared_map->signature==completed_signature);
 worker.camera_scene_complete=true;worker.retire_completed_scene();step(Kind::bgra);
 assert(worker.prepared_map->signature==pending_signature);
 worker.unit_instances.forget(7);step(Kind::bgra);
 worker.renderer_state.moving_resources=1;worker.renderer_state.resource_animations={1};tile.resource_id=101;
 ++worker.renderer_state.cached_signature.complete;step(Kind::bgra);step(Kind::bgra);
 worker.renderer_state.moving_resources=0;step(Kind::unchanged);
 // Projection zoom and compatible authoritative content each get one new front.
 step(Kind::bgra,.75f);before=renders;step(Kind::unchanged,.75f);assert(renders==before);
 ++worker.renderer_state.tile_geometry_epoch;step(Kind::bgra,.75f);step(Kind::unchanged,.75f);
 ++worker.renderer_state.cached_signature.complete;worker.gpu_publication.projection_matches=false;
 before=renders;step(Kind::unchanged,.75f);assert(renders==before);
 worker.gpu_publication.projection_matches=true;step(Kind::bgra,.75f);step(Kind::unchanged,.75f);
 // Valid world preparation releases the only strong job owner. Its actual
 // callbacks freeze without consulting the incompatible shared selection
 // context, while the immutable capture and source-generation proof survive.
 std::weak_ptr<Worker::PreparedMapFrame> retired=worker.prepared_map;
 auto old=sample;assert(old.source_generation==1);
 worker.retire_loading_view();assert(retired.expired()&&!worker.prepared_map);
 worker.renderer_state.reject_selection=true;
 auto selections=worker.renderer_state.selections;before=renders;assets=worker.renderer_state.asset_prepares;
 step(Kind::frozen,.75f);
 assert(worker.renderer_state.selections==selections && renders==before && worker.renderer_state.asset_prepares==assets);
 assert(old.source_generation==1 && old(tick,1000).kind==Kind::frozen && old.projected(tick,1000,.75f).kind==Kind::frozen);
 // A genuine subsequent foreground publication owns a new sampler. Actual
 // selection errors still throw; retiring an old job never suppresses them.
 worker.renderer_state.route_frame_sequence=2;sample=worker.make();++worker.renderer_state.gpu_serial;
 bool failed=false;try{sample.prepare(tick,1000,.75f);}catch(std::runtime_error const& error){
  failed=std::string(error.what())=="unit contribution selection failed";}
 assert(failed && worker.renderer_state.selections==selections+1 && renders==before);
 worker.renderer_state.reject_selection=false;step(Kind::bgra,.75f);
 assert(sample.source_generation==2 && sample.projected(tick,1000,.75f).generation==2);
 assert(old.projected(tick,1000,.75f).kind==Kind::frozen);
 // Capture retirement freezes the dependency; idleness alone never freezes it.
 worker.dynamic_inputs.invalidate();step(Kind::frozen,.75f);
 assert(imports==renders&&worker.visual_map_samples==renders);
}
''')

    def test_fresh_metadata_reports_actual_animation_demand(self):
        source = Path(__file__).with_name("c3x_renderer.cpp").read_text()
        from Renderer.native.test_fresh_preparation_cancellation import block_at
        start = source.index("int camera_ready_view_locked(")
        poll = block_at(source, start)
        metadata = source.split("gpu_metadata.visible_animation_count=job_frame.visible_animation_count+", 1)[1].split(
            "gpu_metadata.clip_left=", 1)[0]
        run_cpp(r'''
#include <cassert>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/gpu_frame_api.h"
struct Worker {
 long long camera_ticket=7;int camera_result=C3X_RENDERER_RESULT_OK;bool camera_active=false,camera_ready_prepared=true;
 struct {c3x_renderer_camera_identity_v1 identity{};c3x_renderer_frame_v1 frame{};c3x_renderer_output_v1 output{};
  struct {bool texture=true;}resident;int phase_x=0,phase_y=0;bool fresh=true;}camera_ready;
 c3x_renderer_frame_v1 job_frame{};
 struct {c3x_renderer_frame_v1 frame{};c3x_renderer_output_v1 output{};bool fresh=true;}gpu_publication;
 c3x_renderer_output_v1 gpu_metadata{};
 ''' + poll + r'''
 int inspect(long long ticket,c3x_renderer_gpu_camera_view_v1& view){return camera_ready_view_locked(ticket,view);}
 void published(){gpu_metadata.visible_animation_count=job_frame.visible_animation_count+''' + metadata + r'''
 }
};
int main(){
 Worker worker;c3x_renderer_gpu_camera_view_v1 view{};
 assert(worker.inspect(7,view)==C3X_RENDERER_RESULT_OK);
 assert(!view.camera.output.visible_animation_count&&!view.camera.output.request_continuous_redraw);
 worker.published();assert(!worker.gpu_metadata.visible_animation_count&&!worker.gpu_metadata.request_continuous_redraw);
 worker.camera_ready.output.visible_animation_count=2;worker.camera_ready.output.request_continuous_redraw=1;
 assert(worker.inspect(7,view)==C3X_RENDERER_RESULT_OK&&view.camera.output.visible_animation_count==2);
 worker.job_frame.visible_animation_count=3;worker.gpu_publication.frame.visible_animation_count=4;
 worker.gpu_publication.output.visible_animation_count=6;worker.published();
 assert(worker.gpu_metadata.visible_animation_count==5&&worker.gpu_metadata.request_continuous_redraw==1);
}
''')
