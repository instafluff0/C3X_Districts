"""Execute interturn readiness through the real injected world-preparation helper."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_preparation_cancellation import block_at


class RequiredInterturnPreparationTests(unittest.TestCase):
    def test_no_display_prepares_world_and_preserves_retired_scope(self):
        source = (ROOT / "injected_code.c").read_text()
        start = source.index("\tif (is->current_config.enable_custom_rendering && p_main_screen_form != NULL &&",
                             source.index("patch_perform_interturn_in_main_loop ()"))
        boundary = block_at(source, start)
        start = source.index("prepare_custom_renderer_loading_world ()")
        helper = "int prepare_custom_renderer_loading_world()" + block_at(source, source.index("{", start))
        helper = helper.replace("(void *)(*p_GetProcAddress)", "(c3x_renderer_seed_world_fn)(*p_GetProcAddress)")
        run_cpp(r'''
#include "Renderer/native/render_core/world_input_capture.h"
#include <cassert>
#include <cstring>
#include <vector>
using HMODULE=void*;
constexpr int IS_OK=1,IS_INIT_FAILED=2,IS_LOADING=3;
constexpr int DNCM_OFF=0,SCM_OFF=0,CS_SUMMER=0,CS_SPRING=3;
struct Counter {long long QuadPart=1000;};
struct Map {int Width=16,Height=16,Flags=1;void* Tiles=reinterpret_cast<void*>(5);
 struct {void* spotlight_on_city=nullptr;}Renderer;};
struct Bic {struct Map Map;}bic,replacement_bic;Bic* p_bic_data=&bic;
struct Main_Screen_Form {int Player_CivID=2;}screen,other_screen;Main_Screen_Form* p_main_screen_form=&screen;
struct State {
 struct {bool enable_custom_rendering=true;int day_night_cycle_mode=0,seasonal_cycle_mode=0;}current_config;
 int custom_renderer_init_state=IS_OK,custom_renderer_viewer_civ_id=2;
 bool custom_renderer_display_valid=false,custom_renderer_capture_world_topology=true;
 void* custom_renderer_native_map=reinterpret_cast<void*>(1);void* custom_renderer_seed_world=reinterpret_cast<void*>(2);
 HMODULE custom_renderer_module=reinterpret_cast<void*>(3);
 long long custom_renderer_map_epoch=1,custom_renderer_viewer_epoch=2,custom_renderer_display_viewer_epoch=0,
  custom_renderer_seeded_viewer_epoch=2,custom_renderer_visibility_revision=1,custom_renderer_world_topology_revision=3;
 int custom_renderer_dirty_flags=0;bool custom_renderer_redraw_pending=false;
 bool custom_renderer_draw_in_progress=false,custom_renderer_frame_active=false,custom_renderer_capture_only=false,
  custom_renderer_loading_world_capture=false,custom_renderer_world_audit_needed=false,
  day_night_cycle_unstarted=true,seasonal_cycle_unstarted=true;
 int current_day_night_cycle=18,current_seasonal_cycle=2;
 Counter custom_renderer_qpc_frequency;
 unsigned const* custom_renderer_world_topology=nullptr;unsigned custom_renderer_world_topology_count=128;
}state;State* is=&state;
unsigned topology[128]{},debug=0;auto p_debug_mode_bits=&debug;
bool owner=true,available=true,export_available=true,topology_ok=true,world_changed=false;
unsigned loads=0,captures=0,prepares=0;int mutation=0,answer=C3X_RENDERER_RESULT_OK,result=99;char event[80]{};
std::vector<long long> prepared_revisions;
bool ensure_custom_renderer_loaded(){++loads;if(mutation==20)state.custom_renderer_init_state=IS_LOADING;return available;}
bool custom_renderer_native_probe_on(){return owner;}
bool is_online_game(){return false;}
int clamp(int a,int b,int v){return v<a?a:v>b?b:v;}
bool QueryPerformanceFrequency(Counter* f){f->QuadPart=1000;return true;}
bool capture_custom_renderer_world_topology(){
 ++captures;assert(state.custom_renderer_world_audit_needed);
 state.custom_renderer_world_topology=topology;
 if(world_changed)++state.custom_renderer_world_topology_revision;
 state.custom_renderer_world_audit_needed=false;return topology_ok;
}
void log_custom_renderer_event(char const* name,int code){std::strcpy(event,name);result=code;}
int prepare(c3x_renderer_camera_request_v1 const* request){
 ++prepares;assert(state.custom_renderer_loading_world_capture);
 assert(!state.custom_renderer_display_viewer_epoch&&!state.custom_renderer_seeded_viewer_epoch);
 assert(!state.custom_renderer_draw_in_progress&&!state.custom_renderer_frame_active&&!state.custom_renderer_capture_only);
 assert(state.custom_renderer_redraw_pending&&(state.custom_renderer_dirty_flags&C3X_RENDERER_DIRTY_SCENE));
 assert(c3x_renderer::render_core::valid_loading_world(request));
 assert(request->frame->tiles==nullptr&&!request->frame->tile_count);
 prepared_revisions.push_back(request->identity.scene_epoch);
 switch(mutation){
 case 1:state.custom_renderer_module=reinterpret_cast<void*>(4);state.custom_renderer_init_state=IS_LOADING;state.custom_renderer_seeded_viewer_epoch=777;break;
 case 2:++state.custom_renderer_map_epoch;break;
 case 3:++state.custom_renderer_viewer_epoch;break;
 case 4:state.current_config.enable_custom_rendering=false;break;
 case 5:p_main_screen_form=&other_screen;break;
 case 6:p_bic_data=&replacement_bic;break;
 case 7:bic.Map.Tiles=reinterpret_cast<void*>(6);break;
 case 8:state.custom_renderer_init_state=IS_LOADING;break;
 case 9:++state.custom_renderer_viewer_civ_id;break;
 case 10:owner=false;break;
 case 11:bic.Map.Renderer.spotlight_on_city=reinterpret_cast<void*>(1);break;
 case 12:++state.custom_renderer_visibility_revision;break;
 case 13:++state.custom_renderer_world_topology_revision;break;
 case 14:++bic.Map.Width;break;
 case 15:bic.Map.Flags^=1;break;
 case 16:++screen.Player_CivID;break;
 case 17:state.custom_renderer_init_state=IS_INIT_FAILED;break;
 }
 return answer;
}
c3x_renderer_seed_world_fn find(HMODULE,char const* name){
 assert(!std::strcmp(name,"c3x_renderer_prepare_world_loading"));
 if(mutation==19){state.custom_renderer_module=reinterpret_cast<void*>(4);state.custom_renderer_init_state=IS_LOADING;}
 return export_available?prepare:nullptr;
}
auto p_GetProcAddress=find;
''' + helper + r'''
void boundary(){
''' + boundary + r'''
}
void reset(){state={};bic={};replacement_bic={};screen={};other_screen={};p_main_screen_form=&screen;p_bic_data=&bic;
 debug=0;loads=captures=prepares=0;owner=available=export_available=topology_ok=true;world_changed=false;
 mutation=0;answer=C3X_RENDERER_RESULT_OK;result=99;event[0]=0;prepared_revisions.clear();}
int main(){
 // Real preparation succeeds without a draw, native anchors or any display.
 for(bool displayed:{false,true}){reset();state.custom_renderer_display_valid=displayed;boundary();
  assert(prepares==1&&captures==1&&state.custom_renderer_init_state==IS_OK&&result==C3X_RENDERER_RESULT_OK);
  assert(state.custom_renderer_seeded_viewer_epoch==2&&!state.custom_renderer_display_viewer_epoch);
  assert(state.custom_renderer_display_valid==displayed&&!state.custom_renderer_loading_world_capture);
  assert(!std::strcmp(event,"required-interturn-preparation-complete"));}
 // Every completed turn audits again, even when the viewer epoch is unchanged.
 reset();boundary();world_changed=true;boundary();world_changed=false;boundary();
 assert((prepared_revisions==std::vector<long long>{3,4,4})&&prepares==3&&captures==3);
 for(int failure:{C3X_RENDERER_RESULT_ERROR,C3X_RENDERER_RESULT_BAD_ARGUMENT,C3X_RENDERER_RESULT_DEVICE_ERROR,C3X_RENDERER_RESULT_PENDING}){
  reset();answer=failure;boundary();assert(prepares==1&&result==failure&&state.custom_renderer_init_state==IS_INIT_FAILED);
  assert(!state.custom_renderer_seeded_viewer_epoch&&!state.custom_renderer_loading_world_capture);
  assert(!std::strcmp(event,"required-interturn-preparation-failed"));}
 reset();answer=C3X_RENDERER_RESULT_SUPERSEDED;boundary();
 assert(result==C3X_RENDERER_RESULT_SUPERSEDED&&state.custom_renderer_init_state==IS_OK&&!state.custom_renderer_seeded_viewer_epoch);
 for(int stale=1;stale<=17;++stale)for(int completion:{C3X_RENDERER_RESULT_OK,C3X_RENDERER_RESULT_ERROR}){
  reset();mutation=stale;answer=completion;boundary();
  assert(prepares==1&&result==C3X_RENDERER_RESULT_SUPERSEDED&&!state.custom_renderer_loading_world_capture);
  assert(!std::strcmp(event,"required-interturn-preparation-retired"));
  assert(state.custom_renderer_init_state==((stale==1||stale==8)?IS_LOADING:stale==17?IS_INIT_FAILED:IS_OK));
  assert(state.custom_renderer_seeded_viewer_epoch==(stale==1?777:0));}
 // Early failures still fail closed; early replacements preserve their state.
 for(int failure=0;failure<3;++failure){reset();
  if(failure==0)export_available=false;if(failure==1)topology_ok=false;if(failure==2)available=false;
  boundary();assert(!prepares&&state.custom_renderer_init_state==IS_INIT_FAILED);
  assert(!std::strcmp(event,"required-interturn-preparation-failed"));}
 reset();mutation=19;export_available=false;boundary();assert(result==C3X_RENDERER_RESULT_SUPERSEDED&&state.custom_renderer_init_state==IS_LOADING);
 reset();mutation=20;available=false;boundary();assert(result==C3X_RENDERER_RESULT_SUPERSEDED&&state.custom_renderer_init_state==IS_LOADING);
 for(int skip=0;skip<9;++skip){reset();
  if(skip==0)state.current_config.enable_custom_rendering=false;if(skip==1)state.custom_renderer_draw_in_progress=true;
  if(skip==2)state.custom_renderer_frame_active=true;if(skip==3)state.custom_renderer_capture_only=true;
  if(skip==4)state.custom_renderer_native_map=nullptr;if(skip==5)bic.Map.Tiles=nullptr;
  if(skip==6)owner=false;if(skip==7)state.custom_renderer_viewer_epoch=0;if(skip==8)state.custom_renderer_module=nullptr;
  boundary();assert(!prepares&&!loads&&result==99&&state.custom_renderer_init_state==IS_OK);}
}
''')


if __name__ == "__main__":
    unittest.main()
