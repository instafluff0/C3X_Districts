"""Execute the existing startup hooks with early native-authority preparation."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp
from Renderer.native.test_fresh_preparation_cancellation import block_at


class LoadingWorldHooksTests(unittest.TestCase):
    def test_actual_era_and_technology_wrappers_suppress_early_music(self):
        source = (ROOT / 'injected_code.c').read_text()
        def extract(name, signature):
            start = source.index(name + ' (')
            return signature + block_at(source, source.index('{', start)).replace('this', 'self')
        era = extract('patch_Leader_enter_new_era', 'void era(Leader* self,int edx,bool param_1,bool no_online_sync)')
        unlock = extract('patch_Leader_unlock_technology', 'void unlock(Leader* self,int edx,int tech_id,bool param_2,bool param_3,bool param_4)')
        # Return-address inspection belongs to the game's x86 ABI. Retain the
        # existing unrelated maintenance branch with an explicit host address.
        unlock = unlock.replace('int * p_stack = (int *)&tech_id;\n\tint ret_addr = p_stack[-1];', 'int ret_addr = 0;')
        music = extract('patch_initialize_map_music', 'void music(int civ_id,int era_id,bool param_3)')
        reveal = extract('patch_Map_process_after_placing', 'void reveal(Map* self,int edx,bool param_1)')
        run_cpp(r'''
#include <cassert>
#include <vector>
constexpr int __=0,C3X_RENDERER_DIRTY_SCENE=4;
constexpr int ADDR_UNLOCK_TECH_AT_INIT_1=1,ADDR_UNLOCK_TECH_AT_INIT_2=2,ADDR_UNLOCK_TECH_AT_INIT_3=3;
struct Map{};struct Leader{int ID=2;}leader;
struct {struct {bool enable_custom_rendering=true,patch_maintenance_persisting_for_obsolete_buildings=false;
 int ai_multi_city_start=0;}current_config;
 bool custom_renderer_loading_players_ready=false,custom_renderer_loading_technology=false,showing_hotseat_replay=false;
 bool custom_renderer_world_audit_needed=false,custom_renderer_redraw_pending=false;
 unsigned custom_renderer_dirty_flags=0,custom_renderer_presented_frames=0;void* custom_renderer_module=(void*)1;}state,*is=&state;
struct {bool is_now_loading_game=false;int Player_CivID=2;}form,*p_main_screen_form=&form;
struct {struct MapData:Map{void* Tiles=(void*)1;}Map;int field_84C=1,field_BD8[6]{},ImprovementsCount=0;
 struct {int ObsoleteID=0;}Improvements[1];}bic,*p_bic_data=&bic;
int turn=0;auto p_current_turn_no=&turn;int era_calls=0,tech_calls=0,music_calls=0,reveal_calls=0,prepares=0,names=0,depth=0;
bool era_unlock=false,unlock_era=false,disable_in_era=false;std::vector<int> era_args,tech_args;
bool is_online_game(){return false;}
void set_up_ai_multi_city_start(Map*,int){}
void Map_process_after_placing(Map*,int,bool){++reveal_calls;}
void initialize_map_music(int,int,bool){++music_calls;}
int prepare_custom_renderer_loading_world(){++prepares;return 1;}
void apply_era_specific_names(Leader*){++names;}
void Leader_recompute_buildings_maintenance(Leader*){assert(false);}
void Leader_enter_new_era(Leader*,int,bool,bool);
void Leader_unlock_technology(Leader*,int,int,bool,bool,bool);
''' + music + reveal + era + unlock + r'''
void Leader_enter_new_era(Leader* self,int edx,bool a,bool b){
 ++era_calls;era_args={self->ID,edx,int(a),int(b)};
 music(self->ID,0,true);
 if(era_unlock && !depth){++depth;unlock(self,0,7,false,true,false);--depth;}
 if(disable_in_era)state.current_config.enable_custom_rendering=false;
}
void Leader_unlock_technology(Leader* self,int edx,int tech,bool a,bool b,bool c){
 ++tech_calls;tech_args={self->ID,edx,tech,int(a),int(b),int(c)};
 music(self->ID,0,true);
 reveal(&bic.Map,0,true);
 if(unlock_era && !depth){++depth;era(self,0,false,true);--depth;}
}
void reset(){state={};form={};bic={};leader={};era_calls=tech_calls=music_calls=reveal_calls=prepares=names=depth=0;
 era_unlock=unlock_era=disable_in_era=false;era_args.clear();tech_args.clear();}
int main(){
 // Direct initial-era entry precedes later AI initialization. Music alone is
 // insufficient to certify this stage, even with the renderer already loaded.
 reset();era(&leader,91,true,false);
 assert(!state.custom_renderer_loading_players_ready && !state.custom_renderer_loading_technology);
 assert(era_calls==1 && names==1 && music_calls==1 && prepares==0);
 assert((era_args==std::vector<int>{2,0,1,0}) && state.custom_renderer_world_audit_needed && state.custom_renderer_redraw_pending);
 // Nested era -> unlock -> reveal-map technology cannot consume an existing
 // players-ready marker; the final native reveal may consume it afterwards.
 reset();era_unlock=true;state.custom_renderer_loading_players_ready=true;era(&leader,0,false,true);
 assert(tech_calls==1 && reveal_calls==1 && prepares==0 && state.custom_renderer_loading_players_ready);
 assert(!state.custom_renderer_loading_technology && (tech_args==std::vector<int>{2,0,7,0,1,0}));
 reveal(&bic.Map,0,false);assert(prepares==1 && !state.custom_renderer_loading_players_ready);
 // Reverse nesting retains the outer technology guard through era music.
 reset();unlock_era=true;unlock(&leader,0,7,false,true,false);
 assert(era_calls==1 && tech_calls==1 && !state.custom_renderer_loading_players_ready && !state.custom_renderer_loading_technology && !prepares);
 reset();state.custom_renderer_loading_technology=true;era_unlock=true;era(&leader,0,false,true);
 assert(state.custom_renderer_loading_technology && !state.custom_renderer_loading_players_ready && !prepares);
 // After all leaders have finished, the ordinary final native music arms the
 // marker and the offline final reveal captures exactly once.
 reset();era_unlock=true;era(&leader,0,false,true);music(2,0,true);
 assert(state.custom_renderer_loading_players_ready);reveal(&bic.Map,0,false);
 assert(prepares==1 && !state.custom_renderer_loading_players_ready && !state.custom_renderer_loading_technology);
 reset();music(3,0,true);assert(!state.custom_renderer_loading_players_ready);
 // Renderer disabled keeps native arguments, existing naming and prior flag.
 for(bool prior:{false,true}){
  reset();state.current_config.enable_custom_rendering=false;state.custom_renderer_loading_technology=prior;
  era(&leader,77,true,false);assert(state.custom_renderer_loading_technology==prior && names==1 && era_calls==1);
  assert((era_args==std::vector<int>{2,0,1,0}) && !state.custom_renderer_world_audit_needed && !prepares);
 }
 reset();disable_in_era=true;era(&leader,0,true,true);
 assert(!state.current_config.enable_custom_rendering && !state.custom_renderer_loading_technology && names==1);
}
''')

    def test_order_config_off_technology_and_final_scenario_stages(self):
        source = (ROOT / 'injected_code.c').read_text()
        start = source.index('patch_Map_process_after_placing (')
        reveal = 'void reveal(Map* self,int edx,bool param_1)' + block_at(source, source.index("{", start)).replace('this', 'self')
        start = source.index('patch_initialize_map_music (')
        music = 'void music(int civ_id,int era_id,bool param_3)' + block_at(source, source.index("{", start))
        start = source.index('patch_Map_place_scenario_things (')
        tail = source[start:source.index('on_open_advisor (', start)]
        start = tail.index('\tif (is->current_config.enable_custom_rendering &&')
        placement = 'void placement(Map* self){' + block_at(tail, start).replace('this', 'self') + '}'
        start = source.index('patch_move_game_data (')
        restore = source[start:source.index('patch_MappedFile_deinit_after_saving_or_loading', start)]
        condition = restore[restore.rfind('\tif (is->current_config.enable_custom_rendering &&'):restore.rfind('\treturn tr;')]
        run_cpp(r'''
#include <cassert>
#include <vector>
constexpr int __=0;
struct Map{};
struct {struct {bool enable_custom_rendering=true;int ai_multi_city_start=1;}current_config;
 bool custom_renderer_loading_players_ready=false,custom_renderer_loading_technology=false,showing_hotseat_replay=false;
 unsigned custom_renderer_presented_frames=0;void* custom_renderer_module=(void*)1;}state,*is=&state;
struct {bool is_now_loading_game=false;int Player_CivID=2;}form,*p_main_screen_form=&form;
struct {struct MapData:Map {void* Tiles=(void*)1;}Map;int field_84C=1,field_BD8[6]{};}bic,*p_bic_data=&bic;
int turn=0;auto p_current_turn_no=&turn;bool online=false;std::vector<int> calls;
bool is_online_game(){return online;}
void set_up_ai_multi_city_start(Map*,int){calls.push_back(1);}
void Map_process_after_placing(Map*,int,bool){calls.push_back(2);}
void initialize_map_music(int,int,bool){calls.push_back(3);}
int prepare_custom_renderer_loading_world(){calls.push_back(4);return 1;}
''' + reveal + music + placement + '\nvoid restored(bool save_else_load,bool renderer_restore_ok){\n' + condition + r'''
}
void reset(){state={};form={};bic={};online=false;calls.clear();}
int main(){
 reset();state.current_config.enable_custom_rendering=false;
 music(2,0,true);reveal(&bic.Map,0,false);placement(&bic.Map);restored(false,true);
 assert((calls==std::vector<int>{3,1,2}) && !state.custom_renderer_loading_players_ready);
 reset();state.custom_renderer_loading_technology=true;music(2,0,true);
 state.custom_renderer_loading_players_ready=true;reveal(&bic.Map,0,true);
 assert((calls==std::vector<int>{3,1,2}) && state.custom_renderer_loading_players_ready);
 // Offline placement has completed before the final native reveal.
 reset();music(2,0,true);assert(state.custom_renderer_loading_players_ready);
 reveal(&bic.Map,0,false);assert((calls==std::vector<int>{3,1,2,4}) && !state.custom_renderer_loading_players_ready);
 // Online scenarios reveal first, then place their actual cities/improvements.
 reset();online=true;music(2,0,true);reveal(&bic.Map,0,false);
 assert(calls.back()==2 && state.custom_renderer_loading_players_ready);
 placement(&bic.Map);assert(calls.back()==4 && !state.custom_renderer_loading_players_ready);
 // Debug reveal changes viewer after this stage; retain its actual later capture.
 reset();bic.field_BD8[5]=1;music(2,0,true);reveal(&bic.Map,0,false);assert(calls.back()==2);
 reset();state.showing_hotseat_replay=true;music(2,0,true);assert(calls.empty() && !state.custom_renderer_loading_players_ready);
 reset();restored(false,true);assert(calls.empty());form.is_now_loading_game=true;
 restored(true,true);restored(false,false);assert(calls.empty());
 restored(false,true);assert((calls==std::vector<int>{4}));
}
''')


    def test_loading_scope_success_failure_and_superseded_lifetime(self):
        source = (ROOT / 'injected_code.c').read_text()
        start = source.index('prepare_custom_renderer_loading_world ()')
        helper = 'int prepare_custom_renderer_loading_world()' + block_at(source, source.index('{', start))
        # Production C permits the void* function conversion; the host fixture is C++.
        helper = helper.replace('(void *)(*p_GetProcAddress)', '(c3x_renderer_seed_world_fn)(*p_GetProcAddress)')
        run_cpp(r"""
#include "Renderer/native/render_core/world_input_capture.h"
#include <cassert>
#include <cstring>
using HMODULE=void*;
constexpr int DNCM_OFF=0,SCM_OFF=0,CS_SUMMER=0,CS_SPRING=3,IS_INIT_FAILED=9;
struct Counter {long long QuadPart=0;};
struct State {
 struct {bool enable_custom_rendering=true;int day_night_cycle_mode=0,seasonal_cycle_mode=0;}current_config;
 bool custom_renderer_capture_world_topology=true,custom_renderer_draw_in_progress=false,
  custom_renderer_frame_active=false,custom_renderer_loading_world_capture=false,
  custom_renderer_capture_only=false,custom_renderer_world_audit_needed=false,
  day_night_cycle_unstarted=true,seasonal_cycle_unstarted=true;
 int custom_renderer_viewer_civ_id=-1,current_day_night_cycle=18,current_seasonal_cycle=2,custom_renderer_init_state=0;
 c3x_renderer_i64 custom_renderer_seeded_viewer_epoch=0,custom_renderer_viewer_epoch=0,
  custom_renderer_map_epoch=0,custom_renderer_visibility_revision=1,custom_renderer_world_topology_revision=3;
 Counter custom_renderer_qpc_frequency;HMODULE custom_renderer_module=(void*)1;
 unsigned const* custom_renderer_world_topology=nullptr;unsigned custom_renderer_world_topology_count=128;
}state,*is=&state;
struct Map {int Width=16,Height=16,Flags=1;void* Tiles=(void*)1;
 struct {void* spotlight_on_city=nullptr;}Renderer;};
struct {struct Map Map;}bic,*p_bic_data=&bic;
struct Main_Screen_Form {int Player_CivID=2;}form,*p_main_screen_form=&form;
unsigned topology[128]{};unsigned debug=0;auto p_debug_mode_bits=&debug;
bool available=true,owner=true,topology_ok=true,export_available=true,mutate=false;
int loads=0,captures=0,calls=0,mutation=0,answer=C3X_RENDERER_RESULT_OK;
bool ensure_custom_renderer_loaded(){++loads;return available;}
bool custom_renderer_native_probe_on(){return owner;}
bool is_online_game(){return false;}
int clamp(int a,int b,int v){return v<a?a:v>b?b:v;}
bool QueryPerformanceFrequency(Counter* f){f->QuadPart=1000;return true;}
bool capture_custom_renderer_world_topology(){++captures;state.custom_renderer_world_topology=topology;return topology_ok;}
void log_custom_renderer_event(char const*,int){}
int prepare(c3x_renderer_camera_request_v1 const* r){
 ++calls;assert(state.custom_renderer_loading_world_capture);
 assert(!state.custom_renderer_frame_active && !state.custom_renderer_draw_in_progress && !state.custom_renderer_capture_only);
 assert(c3x_renderer::render_core::valid_loading_world(r));
 assert(r->frame->hour==12 && r->frame->season==CS_SUMMER);
 if(mutate)++state.custom_renderer_visibility_revision;
 switch(mutation){
 case 1:state.current_config.enable_custom_rendering=false;break;
 case 2:bic.Map.Tiles=(void*)2;break;
 case 3:form.Player_CivID=3;break;
 case 4:state.custom_renderer_module=(void*)2;break;
 case 5:++state.custom_renderer_map_epoch;break;
 case 6:state.custom_renderer_viewer_civ_id=3;break;
 case 7:++bic.Map.Width;break;
 case 8:p_main_screen_form=nullptr;break;
 case 9:state.custom_renderer_init_state=7;break;
 case 10:owner=false;break;
 case 11:bic.Map.Renderer.spotlight_on_city=(void*)1;break;
 }
 return answer;
}
c3x_renderer_seed_world_fn find(HMODULE,char const* name){
 assert(!std::strcmp(name,"c3x_renderer_prepare_world_loading"));return export_available?prepare:nullptr;
}
auto p_GetProcAddress=find;
""" + helper + r"""
void reset(){state={};bic={};form={};debug=0;available=owner=topology_ok=export_available=true;
 mutate=false;loads=captures=calls=mutation=0;p_main_screen_form=&form;answer=C3X_RENDERER_RESULT_OK;}
int main(){
 reset();state.current_config.enable_custom_rendering=false;
 assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_OK && loads==0 && calls==0);
 reset();assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_OK);
 assert(calls==1 && state.custom_renderer_seeded_viewer_epoch==state.custom_renderer_viewer_epoch && !state.custom_renderer_loading_world_capture);
 assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_OK && calls==1);
 for(int failure:{C3X_RENDERER_RESULT_ERROR,C3X_RENDERER_RESULT_BAD_ARGUMENT,C3X_RENDERER_RESULT_DEVICE_ERROR}){
  reset();answer=failure;assert(prepare_custom_renderer_loading_world()==failure);
  assert(!state.custom_renderer_loading_world_capture && !state.custom_renderer_seeded_viewer_epoch && state.custom_renderer_init_state==IS_INIT_FAILED);
 }
 reset();answer=C3X_RENDERER_RESULT_SUPERSEDED;assert(prepare_custom_renderer_loading_world()==answer);
 assert(!state.custom_renderer_loading_world_capture && !state.custom_renderer_seeded_viewer_epoch && !state.custom_renderer_init_state);
 reset();mutate=true;assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_SUPERSEDED);
 assert(!state.custom_renderer_loading_world_capture && !state.custom_renderer_seeded_viewer_epoch);
 // A completion belongs only to the same enabled renderer/native scope.
 for(int stale=1;stale<=11;++stale)for(int completion:{C3X_RENDERER_RESULT_OK,C3X_RENDERER_RESULT_ERROR}){
  reset();mutation=stale;answer=completion;
  assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_SUPERSEDED);
  assert(!state.custom_renderer_loading_world_capture && !state.custom_renderer_seeded_viewer_epoch);
  assert(state.custom_renderer_init_state==(stale==9?7:0));
 }
 reset();export_available=false;assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_BAD_ARGUMENT && !calls);
 reset();topology_ok=false;assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_ERROR && !calls);
 reset();state.custom_renderer_frame_active=true;
 assert(prepare_custom_renderer_loading_world()==C3X_RENDERER_RESULT_BAD_ARGUMENT && state.custom_renderer_frame_active && !calls);
}
""")


if __name__ == '__main__':
    unittest.main()
