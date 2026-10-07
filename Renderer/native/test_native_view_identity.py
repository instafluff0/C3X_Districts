"""Execute the injected native identity observations and ordinary-call dispatch."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class NativeViewIdentityTests(unittest.TestCase):
    def test_default_handoff_requires_capabilities_and_allows_diagnostic_opt_out(self):
        source = (ROOT / 'injected_code.c').read_text()
        selection = source.split('char async_option[8] = {0};', 1)[1]
        selection = '#ifdef Main_Screen_Form_move_camera' + selection.split('#ifdef Main_Screen_Form_move_camera', 1)[1].split('#endif', 1)[0] + '#endif'
        run_cpp(r"""
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <initializer_list>
#include <cstring>
struct State {
 struct {bool enable_custom_rendering=true;} current_config;
 int custom_renderer_init_state=1;bool custom_renderer_redraw_pending=false,custom_renderer_loading_world_capture=false;
 bool custom_renderer_camera_exact=false,custom_renderer_unit_representatives_dirty=false,custom_renderer_scroll_request=false;
 bool combat_unit_display_override_active=false,custom_renderer_trace_input=false;
 c3x_renderer_native_navigation_fn custom_renderer_navigation=nullptr;
 c3x_renderer_visual_clock_fn custom_renderer_visual_clock=nullptr;
 bool custom_renderer_async_enabled=false;
 void *custom_renderer_render_view=this, *custom_renderer_camera_begin=this,
      *custom_renderer_camera_poll=this, *custom_renderer_camera_present=this,
      *custom_renderer_camera_cancel=this;
};
#define Main_Screen_Form_move_camera available
void select(State* is, char const* async_option) {
""" + selection + r"""
}
#undef Main_Screen_Form_move_camera
void unsupported(State* is, char const* async_option) {
""" + selection + r"""
}
int main() {
 State state;select(&state, "");assert(state.custom_renderer_async_enabled);
 select(&state, "0");assert(!state.custom_renderer_async_enabled);
 select(&state, "1");assert(state.custom_renderer_async_enabled);
 for(auto member:{&State::custom_renderer_render_view,&State::custom_renderer_camera_begin,
                  &State::custom_renderer_camera_poll,&State::custom_renderer_camera_present,
                  &State::custom_renderer_camera_cancel}) {
  State missing;missing.*member=nullptr;select(&missing, "");assert(!missing.custom_renderer_async_enabled);
 }
 unsupported(&state, "");assert(!state.custom_renderer_async_enabled);
}
""")

    def test_capture_lifecycle_visibility_and_legacy_dispatch(self):
        source = (ROOT / 'injected_code.c').read_text()
        world = 'bool\ncapture_custom_renderer_world_topology ()' + source.split('capture_custom_renderer_world_topology ()', 1)[1].split('\nvoid\n', 1)[0]
        viewer = source.split('\t// A complete capture belongs to one authoritative native viewer.', 1)[1].split('\tint const max_tiles', 1)[0]
        retire = source.split('\t// No publication survives unload;', 1)[1].split('\tis->custom_renderer_capture_world_topology = false;', 1)[0]
        retire = retire.split('\n', 1)[1]
        dispatch = '\tstruct c3x_renderer_camera_request_v1 request = {0};' + source.split('\tstruct c3x_renderer_camera_request_v1 request = {0};', 1)[1].split('\tif (is->custom_renderer_presented_frames == 0)', 1)[0]
        dispatch = dispatch.replace('(void *)(*p_GetProcAddress)',
            '(DWORD (*)(char const *, char *, DWORD))(*p_GetProcAddress)')
        dispatch = dispatch.replace('(DWORD (*)(char const *, char *, DWORD))(*p_GetProcAddress) (is->kernel32, "Sleep")',
            '(void (*)(DWORD))(*p_GetProcAddress) (is->kernel32, "Sleep")')
        dispatch = dispatch.replace('c3x_renderer_world_status_fn query_world = (DWORD (*)(char const *, char *, DWORD))',
                                    'c3x_renderer_world_status_fn query_world = (c3x_renderer_world_status_fn)')
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
using DWORD=unsigned;
#define WINAPI
bool legacy_recovery=false;
DWORD environment(char const*,char* value,DWORD capacity){
 if(!legacy_recovery || capacity<2)return 0;
 value[0]='1';value[1]=0;return 1;
}
unsigned sleeps=0;void sleep_fake(DWORD ms){assert(ms==5);++sleeps;}
void* lookup(void*,char const* name){return std::strcmp(name,"Sleep")==0?
 reinterpret_cast<void*>(&sleep_fake):reinterpret_cast<void*>(&environment);}
auto p_GetProcAddress=&lookup;
void log_custom_renderer_event(char const*,int){}
constexpr int IS_OK=1;void notify_custom_renderer_unit_selection(bool){}
struct LARGE_INTEGER {long long QuadPart=0;};
void QueryPerformanceCounter(LARGE_INTEGER* p){p->QuadPart=1000;}
void debug(char const*){} auto p_OutputDebugStringA=&debug;
std::size_t fail_at=~std::size_t(0),largest_request=0;
unsigned fail_nth=0;
struct Memory {void* p;template<class T>operator T*()const{return static_cast<T*>(p);}};
Memory allocate(void* p,std::size_t bytes){largest_request=std::max(largest_request,bytes);return {bytes>=fail_at || (fail_nth && !--fail_nth)?nullptr:std::realloc(p,bytes)};}
#define realloc allocate
#define malloc(bytes) allocate(nullptr,bytes)
struct Tile;
struct Vtable {int(*m49_Get_Square_RealType)(Tile*);int(*m50_Get_Square_BaseType)(Tile*);int(*m37_Get_River_Code)(Tile*);int(*m43_Get_field_30)(Tile*);};
constexpr int SQ_Mountains=6;
struct Tile {Vtable* vtable;struct {int FOWStatus=0,Visibility=0,Fog_Of_War=0,V3=0,field_D0_Visibility=0;void* active_tile_effect=nullptr;}Body;int ground=2,base=2,river=0,field30=0;};
Vtable vtable{[](Tile*t){return t->ground;},[](Tile*t){return t->base;},[](Tile*t){return t->river;},[](Tile*t){return t->field30;}};
struct MapData{int Width=4,Height=4;int Renderer=0;};using Map=MapData;
struct Bic {MapData Map;bool is_zoomed_out=false;} bic;Bic* p_bic_data=&bic;
Tile null_tile{&vtable};Tile* p_null_tile=&null_tile;
std::vector<Tile> tiles(5000,Tile{&vtable});int absent=-1;
Tile* tile_at(int x,int y){int at=(y*bic.Map.Width+x)/2;return at==absent?p_null_tile:&tiles.at(at);}
unsigned modern_calls=0,legacy_calls=0,resident_calls=0;c3x_renderer_camera_identity_v1 received{};
std::vector<int> order;
int resident_result=C3X_RENDERER_RESULT_OK,polls_until_ready=-1;bool probe=true;
bool custom_renderer_native_probe_on(){return probe;}
int resident(int action,void* image,c3x_renderer_camera_request_v1 const* r,c3x_renderer_camera_view_v1* v){
 v->frame.presentation_time_ticks=55;
 assert(action==C3X_NATIVE_MAP_PREPARE&&image&&r);order.push_back(3);++resident_calls;received=r->identity;
 if(resident_result==C3X_RENDERER_RESULT_PENDING && polls_until_ready>=0 && polls_until_ready--==0)return C3X_RENDERER_RESULT_OK;
 return resident_result;
}
int modern(c3x_renderer_camera_request_v1 const* r,c3x_renderer_output_v1*){
 assert(r->version==C3X_RENDERER_CAMERA_VIEW_VERSION && r->struct_size==sizeof(*r));order.push_back(4);received=r->identity;++modern_calls;return 7;
}
int legacy(c3x_renderer_frame_v1 const*,c3x_renderer_output_v1*){order.push_back(5);++legacy_calls;return 9;}
using HMODULE=void*;using byte=unsigned char;
constexpr int __=0;
struct LoadingForm {int camera_x=0,camera_y=0;struct GUIData {int field_574[4]{};}GUI;} form;auto p_main_screen_form=&form;
void Main_GUI_label_loading_bar(LoadingForm::GUIData*,int,int,char const*){}
struct State {
 struct {bool enable_custom_rendering=true;}current_config;
 HMODULE custom_renderer_module=nullptr;void* custom_renderer_target=nullptr;
 bool custom_renderer_draw_in_progress=false,custom_renderer_frame_active=false,custom_renderer_capture_only=false,custom_renderer_capture_failed=false;
 c3x_renderer_tile_v1* custom_renderer_tiles=nullptr;
 unsigned custom_renderer_presented_frames=1;
 void* kernel32=nullptr;
 bool custom_renderer_async_drawing=false,custom_renderer_display_valid=true;
 struct {int camera_x=0,camera_y=0,native_width=128;}custom_renderer_display_view;
 c3x_renderer_camera_present_view_fn custom_renderer_camera_present=nullptr;
 long long custom_renderer_display_clock=0,custom_renderer_camera_ticket=0;
 c3x_renderer_render_view_fn custom_renderer_render_view=modern;
 c3x_renderer_render_fn custom_renderer_render=legacy;
 c3x_renderer_native_map_view_fn custom_renderer_native_map=nullptr;
 c3x_renderer_native_lifetime_fn custom_renderer_native_lifetime=nullptr;
 int* custom_renderer_city_site_grades=nullptr;int custom_renderer_city_site_grade_count=0;
 unsigned* custom_renderer_world_topology=nullptr;
 unsigned long long* custom_renderer_world_visibility=nullptr;
 int custom_renderer_world_topology_count=0,custom_renderer_tile_count=0,custom_renderer_viewer_civ_id=-1;
 long long custom_renderer_world_topology_revision=0,custom_renderer_visibility_revision=0;
 long long custom_renderer_map_epoch=0,custom_renderer_viewer_epoch=0,custom_renderer_display_viewer_epoch=0;
 bool custom_renderer_world_audit_needed=true;
 bool custom_renderer_capture_world_topology=false;
 c3x_renderer_world_reconcile_fn custom_renderer_world_reconcile=nullptr;
 unsigned custom_renderer_requested_frames=0;LARGE_INTEGER custom_renderer_qpc_frequency{1000};
 unsigned custom_renderer_dirty_flags=0;bool custom_renderer_redraw_pending=false;
} state;State* is=&state;
// The helpers have their own extracted-body tests. Here their controlled
// results exercise the actual composite dispatch's ordering and early return.
int seed_result=1,bootstrap_result=1;
int seed_custom_renderer_initial_world(c3x_renderer_camera_request_v1 const* request){
 assert(request&&request->version==C3X_RENDERER_CAMERA_VIEW_VERSION&&request->frame);
 assert(request->identity.map_epoch==state.custom_renderer_map_epoch&&
        request->identity.viewer_epoch==state.custom_renderer_viewer_epoch&&
        request->identity.visibility_epoch==state.custom_renderer_visibility_revision&&
        request->identity.scene_epoch==request->frame->world_topology_revision);
 order.push_back(1);return seed_result;
}
int bootstrap_custom_renderer_initial_units(){order.push_back(2);return bootstrap_result;}
''' + world + '\nbool viewer(int visible_to_civ_id){\n' + viewer + '\nreturn true;}\nvoid retire(){\n' + retire + r'''
}
int demand(){void* image=is;c3x_renderer_frame_v1 frame={};frame.presentation_time_ticks=77;frame.world_topology_revision=is->custom_renderer_world_topology_revision;c3x_renderer_output_v1 output={};
''' + dispatch + r'''
 return render_result;
}
int main(){
 assert(viewer(3) && state.custom_renderer_viewer_epoch==1);
 assert(viewer(3) && state.custom_renderer_viewer_epoch==1);
 state.custom_renderer_tile_count=3;assert(!viewer(4) && state.custom_renderer_viewer_civ_id==3);
 state.custom_renderer_tile_count=0;assert(viewer(4) && state.custom_renderer_viewer_epoch==2);
 assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_count==8 && state.custom_renderer_world_topology_revision==1 && state.custom_renderer_visibility_revision==1);
 auto topology=state.custom_renderer_world_topology_revision,visibility=state.custom_renderer_visibility_revision;
 assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==topology && state.custom_renderer_visibility_revision==visibility);
 tiles[7].Body.Visibility=1;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_visibility_revision==visibility); // No continuous scan.
 state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==topology && state.custom_renderer_visibility_revision==++visibility);
 tiles[7].Body.FOWStatus=0x12345678;state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_visibility_revision==++visibility); // All sources of current visibility and explored state are observed.
 tiles[6].Body.V3=8;state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());assert(state.custom_renderer_visibility_revision==++visibility);
 tiles[6].Body.field_D0_Visibility=16;state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());assert(state.custom_renderer_visibility_revision==++visibility);
 tiles[6].Body.Fog_Of_War=32;state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());assert(state.custom_renderer_visibility_revision==++visibility);
 tiles[2].ground=4;state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==++topology && state.custom_renderer_visibility_revision==visibility);
 // Civ III's snow-capped mountain (field_30 0x100000 on a mountain) is topology
 // bit 26; a bare mountain or the same bit on other terrain is not.
 tiles[3].base=6;tiles[3].field30=0x100000;tiles[4].base=6;tiles[5].field30=0x100000;
 state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==++topology);
 assert((state.custom_renderer_world_topology[3]>>26&1u)==1 && (state.custom_renderer_world_topology[4]>>26&1u)==0 &&
        (state.custom_renderer_world_topology[5]>>26&1u)==0);
 tiles[3]=tiles[4]=tiles[5]=Tile{&vtable};
 state.custom_renderer_world_audit_needed=true;assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_revision==++topology);
 assert(demand()==C3X_RENDERER_RESULT_BAD_ARGUMENT && !modern_calls && !legacy_calls && state.custom_renderer_map_epoch==1);
 bic.Map={100,100};assert(capture_custom_renderer_world_topology());
 assert(state.custom_renderer_world_topology_count==5000 && largest_request==5000*sizeof(unsigned long long));
 // Either allocation may fail during shrink. Returning to the old map must
 // retain both complete old owners; reallocating topology first used to shrink it.
 auto* old_topology=state.custom_renderer_world_topology;
 auto* old_visibility=state.custom_renderer_world_visibility;
 for(unsigned allocation: {1u,2u}){
  bic.Map={4,4};fail_nth=allocation;
  assert(!capture_custom_renderer_world_topology());
  assert(state.custom_renderer_world_topology==old_topology && state.custom_renderer_world_visibility==old_visibility);
  assert(state.custom_renderer_world_topology_count==5000);
  bic.Map={100,100};assert(capture_custom_renderer_world_topology());
 }
 // An allocation failure cannot publish a new count with an undersized visibility owner.
 bic.Map={2048,2048};fail_at=16u*1024u*1024u;
 assert(!capture_custom_renderer_world_topology() && largest_request==fail_at && state.custom_renderer_world_topology_count==5000);
 fail_at=~std::size_t(0);bic.Map={4,4};assert(capture_custom_renderer_world_topology());
 absent=3;state.custom_renderer_world_audit_needed=true;assert(!capture_custom_renderer_world_topology() && !state.custom_renderer_world_topology_count);
 absent=-1;assert(capture_custom_renderer_world_topology());
 auto count=state.custom_renderer_world_topology_count;
 for(auto dims: {MapData{2049,4},MapData{4,2049},MapData{3,4},MapData{0,4}}){
  bic.Map=dims;assert(!capture_custom_renderer_world_topology() && state.custom_renderer_world_topology_count==count);
 }
 retire();assert(!state.custom_renderer_world_visibility && !state.custom_renderer_visibility_revision);
 assert(state.custom_renderer_map_epoch==2 && !state.custom_renderer_viewer_epoch && state.custom_renderer_viewer_civ_id==-1);
 legacy_recovery=true;state.custom_renderer_render_view=nullptr;
 assert(demand()==9 && legacy_calls==1 && !modern_calls);
 legacy_recovery=false;
 bic.Map={4,4};assert(capture_custom_renderer_world_topology() && !state.custom_renderer_world_visibility); // Older DLL compatibility.
 // The real map dispatch carries the same epochs through the resident owner,
 // and only an admission rejection permits the existing CPU path.
 state.custom_renderer_render_view=modern;state.custom_renderer_native_map=resident;
 state.custom_renderer_native_lifetime=[](int,void*,int){return 1;};
 assert(demand()==C3X_RENDERER_RESULT_OK&&resident_calls==1&&!modern_calls&&legacy_calls==1);
 assert(state.custom_renderer_display_clock==55&&received.map_epoch==state.custom_renderer_map_epoch);
 resident_result=C3X_RENDERER_RESULT_DEVICE_ERROR;
 assert(demand()==C3X_RENDERER_RESULT_DEVICE_ERROR&&!modern_calls&&legacy_calls==1);
 resident_result=C3X_RENDERER_RESULT_BAD_ARGUMENT;
 assert(demand()==C3X_RENDERER_RESULT_BAD_ARGUMENT&&!modern_calls&&resident_calls==3);
 resident_result=C3X_RENDERER_RESULT_PENDING;
 state.custom_renderer_display_viewer_epoch=state.custom_renderer_viewer_epoch;
 assert(demand()==0&&state.custom_renderer_redraw_pending&&
        (state.custom_renderer_dirty_flags&C3X_RENDERER_DIRTY_SCENE));
 probe=false;assert(demand()==C3X_RENDERER_RESULT_BAD_ARGUMENT&&!modern_calls&&resident_calls==4);
 // A first-view camera may never race initial world/body admission. Failure
 // requests a later native redraw and leaves both modern and legacy paths idle.
 probe=true;resident_result=C3X_RENDERER_RESULT_OK;
 state.custom_renderer_capture_world_topology=true;
 auto before_resident=resident_calls;
 seed_result=C3X_RENDERER_RESULT_BAD_ARGUMENT;order.clear();
 state.custom_renderer_dirty_flags=0;state.custom_renderer_redraw_pending=false;
 assert(demand()==0&&order==std::vector<int>{1}&&resident_calls==before_resident);
 assert(state.custom_renderer_redraw_pending&&(state.custom_renderer_dirty_flags&C3X_RENDERER_DIRTY_SCENE));
 seed_result=C3X_RENDERER_RESULT_OK;bootstrap_result=C3X_RENDERER_RESULT_PENDING;order.clear();
 state.custom_renderer_dirty_flags=0;state.custom_renderer_redraw_pending=false;
 assert(demand()==0&&order==std::vector<int>({1,2})&&resident_calls==before_resident);
 assert(state.custom_renderer_redraw_pending&&(state.custom_renderer_dirty_flags&C3X_RENDERER_DIRTY_SCENE));
 bootstrap_result=C3X_RENDERER_RESULT_OK;order.clear();
 assert(demand()==C3X_RENDERER_RESULT_OK&&order==std::vector<int>({1,2,3}));
 // Frozen/legacy profiles skip both initial helpers, preserving their dispatch.
 state.custom_renderer_capture_world_topology=false;order.clear();
 assert(demand()==C3X_RENDERER_RESULT_OK&&order==std::vector<int>{3});
 // Execute the entire production prepare/poll loop, not just its condition.
 for(int change=0;change<5;++change){
  form.camera_x=form.camera_y=0;bic.is_zoomed_out=false;state.custom_renderer_display_valid=true;
  state.custom_renderer_display_view={};state.custom_renderer_display_viewer_epoch=state.custom_renderer_viewer_epoch;
  if(change==0)form.camera_x=64;if(change==1)form.camera_y=32;
  if(change==2)bic.is_zoomed_out=true;if(change==3)state.custom_renderer_display_valid=false;
  if(change==4)state.custom_renderer_display_viewer_epoch++;
  resident_result=C3X_RENDERER_RESULT_PENDING;polls_until_ready=3;
  auto calls=resident_calls,waits=sleeps;
  assert(demand()==C3X_RENDERER_RESULT_OK&&resident_calls==calls+4&&sleeps==waits+3);
  assert(state.custom_renderer_display_clock==55);
 }
 std::puts("PASS native map barrier: five delayed completions; same-view pending remains asynchronous");
 std::free(state.custom_renderer_world_topology);
}
''')

    def test_current_camera_publication_and_request_only_capture(self):
        source = (ROOT / 'injected_code.c').read_text()
        helpers = 'struct custom_renderer_native_view\ncustom_renderer_native_view' + source.split('struct custom_renderer_native_view\ncustom_renderer_native_view', 1)[1].split('void __fastcall\npatch_Map_Renderer_m71_Draw_Tiles', 1)[0]
        # The navigation transaction fixture supplies a wrapped native camera;
        # zoom clamping and the Win32 timer have their own executable fixture.
        helpers = helpers.split('#ifdef Main_Screen_Form_scroll_at_mouse', 1)[0]
        poll_begin = helpers.index('void\npoll_custom_renderer_combat_zoom ()')
        poll_end = helpers.index('#ifdef Animator_update_display\nvoid __fastcall\npatch_Animator_update_display', poll_begin)
        helpers = helpers[:poll_begin] + helpers[poll_end:]
        begin = helpers.index('// Preserve native wrapping and city centering;')
        end = helpers.index('#ifdef Main_Screen_Form_move_camera\nvoid __fastcall\npatch_Main_Screen_Form_move_camera', begin)
        helpers = helpers[:begin] + helpers[end:]
        body = source.split('void __fastcall\npatch_Map_Renderer_m71_Draw_Tiles', 1)[1]
        start = '\tif (custom_renderer_zoom_enabled ()) sync_custom_renderer_zoom_to_native ();' + body.split('\tif (custom_renderer_zoom_enabled ()) sync_custom_renderer_zoom_to_native ();', 1)[1].split('\tis->custom_renderer_draw_in_progress = true;', 1)[0]
        finish = '\t// Keep the completed native view identity' + body.split('\t// Keep the completed native view identity', 1)[1].split('\tis->custom_renderer_frame_active = false;', 1)[0]
        program = r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstring>
#include <cstdio>
#include <algorithm>
#define __fastcall
#define __ 0
#define Main_Screen_Form_move_camera native_move
int resolved_x=-1,resolved_y=-1;
void log_custom_renderer_test_route_resolved(int x,int y){resolved_x=x;resolved_y=y;}
struct RECT {int left=0,top=0,right=2240,bottom=1192;};
struct JGL_Image;
struct ImageVtable {int(*m54_Get_Width)(JGL_Image*);int(*m55_Get_Height)(JGL_Image*);};
struct JGL_Image {ImageVtable* vtable;RECT Clip_Rect,Image_Rect;};
ImageVtable image_vtable{[](JGL_Image*){return 2240;},[](JGL_Image*){return 1192;}};
JGL_Image image{&image_vtable};
struct JGL {JGL_Image* Image=&image;};
struct Map_Renderer;
struct Vtable {void* m21_Draw_Tiles_by_Flags;};
struct Map_Renderer {Vtable* vtable;struct JGL JGL;void* spotlight_on_city=nullptr;};
struct PCX_Image {void* vtable;struct JGL JGL;};
struct MapData {Map_Renderer Renderer;};
struct Bic {MapData Map;bool is_zoomed_out=false;} bic;
Bic* p_bic_data=&bic;
struct Animator {int fields[32]{};int* field_18E4=fields;int Units2_Count=0;};
struct Main_Screen_Form {struct {char field_574[4]{};}GUI;struct {PCX_Image Canvas{};}Base_Data;Animator animator;bool is_now_loading_game=false,turn_end_flag=false;int Player_CivID=2;int camera_x=0,camera_y=0,TileX_Min=0,TileX_Max=20,TileY_Min=0,TileY_Max=20;} screen;
Main_Screen_Form* p_main_screen_form=&screen;
struct Clock {long long QuadPart=0;};
using CaptureFn=int(*)(void*,c3x_renderer_camera_request_v1 const*,long long*);
CaptureFn early_capture=nullptr;
CaptureFn get_capture(void*,char const*){return early_capture;}
auto p_GetProcAddress=get_capture;

struct State {
 void* custom_renderer_module=nullptr;
 struct {bool enable_custom_rendering=true;} current_config;
 int custom_renderer_init_state=1;bool custom_renderer_redraw_pending=false,custom_renderer_loading_world_capture=false;
 bool custom_renderer_camera_exact=false,custom_renderer_unit_representatives_dirty=false,custom_renderer_scroll_request=false;
 bool combat_unit_display_override_active=false,custom_renderer_trace_input=false;
 c3x_renderer_native_navigation_fn custom_renderer_navigation=nullptr;
 c3x_renderer_visual_clock_fn custom_renderer_visual_clock=nullptr;
 bool custom_renderer_async_enabled=true,custom_renderer_display_valid=false;
 bool custom_renderer_draw_in_progress=false,custom_renderer_async_drawing=false,custom_renderer_async_presented=false;
 bool custom_renderer_capture_only=false,custom_renderer_capture_failed=false,custom_renderer_capture_world_topology=true;
 int custom_renderer_zoom_tile_width=128,custom_renderer_tile_count=0,custom_renderer_viewer_civ_id=2;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
 unsigned custom_renderer_presented_frames=0;
 c3x_renderer_native_image_fn custom_renderer_native_image=nullptr;
 long long custom_renderer_camera_ticket=0,custom_renderer_display_clock=0;
 long long custom_renderer_map_epoch=1,custom_renderer_viewer_epoch=2,custom_renderer_visibility_revision=3;
 struct custom_renderer_native_view custom_renderer_display_view{},custom_renderer_queued_view{};
 c3x_renderer_tile_v1 storage[2]{},*custom_renderer_tiles=storage;
 unsigned custom_renderer_visible_animation_count=0;
 Clock custom_renderer_animation_timestamp,custom_renderer_qpc_frequency;
 c3x_renderer_camera_begin_view_fn custom_renderer_camera_begin;
 c3x_renderer_camera_poll_view_fn custom_renderer_camera_poll;
 c3x_renderer_camera_cancel_fn custom_renderer_camera_cancel;
 int custom_renderer_capture_cover=0,custom_renderer_zoom_target_width=128;
} state;State* is=&state;
unsigned debug_mode_bits=0;auto p_debug_mode_bits=&debug_mode_bits;
bool online=false;bool is_online_game(){return online;}
bool custom_renderer_zoom_enabled(){return !screen.is_now_loading_game;}
int custom_renderer_capture_cover_width(bool){return 112;}
void log_custom_renderer_event(char const*,int){}
void debug(char const*){}auto p_OutputDebugStringA=debug;
constexpr int IS_OK=1;void poll_custom_renderer_combat_zoom(){}
void notify_custom_renderer_unit_selection(bool){}
void sync_custom_renderer_zoom_to_native(){}
void native_move(Main_Screen_Form* s,int,int x,int y,int,bool){
 s->camera_x=(x%8192+8192)%8192;s->camera_y=(y%4096+4096)%4096;
 s->TileX_Min=s->camera_x/64;s->TileX_Max=s->TileX_Min+20;
 s->TileY_Min=s->camera_y/32;s->TileY_Max=s->TileY_Min+20;
}
void move_custom_renderer_camera(Main_Screen_Form* s,int e,int x,int y,int r,bool b){native_move(s,e,x,y,r,b);}
unsigned captures=0,begins=0,cancels=0;int queued_x=-1,poll_status=C3X_RENDERER_RESULT_PENDING;
void capture(Map_Renderer* target,int,int viewer,int,int,Map_Renderer* output,void* clip,int x,int y,int flags){
 assert(state.custom_renderer_capture_only && !clip && x==-1 && y==-1 && flags==9 && output==target && viewer==((debug_mode_bits&8)&&!online?0:2));
 ++captures;state.custom_renderer_tile_count=2;
 state.storage[0]={};state.storage[0].visibility_mask=8;state.storage[0].anchor_x=-screen.camera_x;
 image.Clip_Rect.left=7; // The real traversal may change clipping; the bridge restores it.
}
void capture_custom_renderer_topology(int viewer,int mask){assert(viewer==((debug_mode_bits&8)&&!online?0:2) && mask==8);}
bool prepare_custom_renderer_frame(c3x_renderer_frame_v1* frame){
 *frame={};frame->tiles=state.storage;frame->tile_count=2;
 frame->world_topology_revision=4;frame->presentation_time_ticks=state.custom_renderer_animation_timestamp.QuadPart;return true;
}
int begin(c3x_renderer_camera_request_v1 const* r,long long* ticket){
 assert(r->identity.map_epoch==1 && r->identity.viewer_epoch==2 && r->identity.visibility_epoch==3 && r->identity.scene_epoch==4);
 queued_x=-r->frame->tiles[0].anchor_x;*ticket=++begins;return C3X_RENDERER_RESULT_PENDING;
}
int poll(long long,c3x_renderer_camera_view_v1*){return poll_status;}
int cancel(long long){++cancels;return C3X_RENDERER_RESULT_OK;}
''' + helpers.replace('this', 'screen_arg') + r'''
int displayed_x=-1,animator_x=-1,animator_y=-1;
void call(bool success=true){Map_Renderer* screen_arg=&bic.Map.Renderer;int param_1=2;
 // Native Animator_update computes canvas and wrap copies before calling m71.
 animator_x=screen.camera_x;animator_y=screen.camera_y;
''' + start.replace('this', 'screen_arg') + r'''
 assert(screen.camera_x==animator_x && screen.camera_y==animator_y);
 state.custom_renderer_async_presented=success;
 if(success){displayed_x=screen.camera_x;state.custom_renderer_display_clock=state.custom_renderer_animation_timestamp.QuadPart;}
''' + finish.replace('this', 'screen_arg') + r'''
 assert(screen.camera_x==animator_x && screen.camera_y==animator_y);
}
int main(){
 Vtable vt{reinterpret_cast<void*>(&capture)};bic.Map.Renderer.vtable=&vt;
 state.custom_renderer_camera_begin=begin;state.custom_renderer_camera_poll=poll;state.custom_renderer_camera_cancel=cancel;
 call();assert(displayed_x==0 && state.custom_renderer_display_valid && begins==0);
 // Every native movement is now an exact barrier. The real animator observes
 // these fields between movement and m71; the previous fixture omitted it.
 patch_Main_Screen_Form_move_camera(&screen,0,32,0,1,false);
 assert(screen.camera_x==32 && !state.custom_renderer_display_valid);
 call();assert(displayed_x==32 && begins==0 && captures==0);
 for(int i=0;i<20;++i){
  patch_Main_Screen_Form_move_camera(&screen,0,screen.camera_x+32,0,1,false);
  assert(screen.camera_x==64+i*32); // Input cannot be hidden behind publication.
  call();assert(displayed_x==screen.camera_x && begins==0);
 }
 // Stationary ambient updates still use the bounded caller-driven queue.
 state.custom_renderer_qpc_frequency.QuadPart=150;
 state.custom_renderer_visible_animation_count=1;
 state.custom_renderer_animation_timestamp.QuadPart=30;
 capture_custom_renderer_native_view(&bic.Map.Renderer,2,&state.custom_renderer_display_view,false);
 assert(begins==1 && state.custom_renderer_camera_ticket && queued_x==672);
 // Movement cancels old ambient work even if it is never ready. Native bounds,
 // picking and unit culling immediately see the new native camera.
 patch_Main_Screen_Form_move_camera(&screen,0,640,0,1,false);
 assert(cancels==1 && !state.custom_renderer_camera_ticket && screen.camera_x==640);
 assert(screen.TileX_Min==10 && !state.custom_renderer_display_valid);
 state.custom_renderer_visible_animation_count=0;
 call();assert(displayed_x==640 && !state.custom_renderer_camera_ticket);
 // Camera changes that bypass the move hook still invalidate old queued output.
 // A completed old ticket cannot rewrite the native animator's current camera.
 state.custom_renderer_visible_animation_count=1;
 state.custom_renderer_animation_timestamp.QuadPart+=30;
 capture_custom_renderer_native_view(&bic.Map.Renderer,2,&state.custom_renderer_display_view,false);
 assert(state.custom_renderer_camera_ticket);
 native_move(&screen,0,704,64,1,false);poll_status=C3X_RENDERER_RESULT_OK;
 state.custom_renderer_visible_animation_count=0;
 call();assert(cancels==2 && displayed_x==704 && screen.camera_y==64);
 assert(!state.custom_renderer_camera_ticket && screen.TileX_Min==11);
 // Projection changes likewise reject a ready old-view publication.
 state.custom_renderer_visible_animation_count=1;
 state.custom_renderer_animation_timestamp.QuadPart+=30;
 capture_custom_renderer_native_view(&bic.Map.Renderer,2,&state.custom_renderer_display_view,false);
 assert(state.custom_renderer_camera_ticket);
 state.custom_renderer_zoom_tile_width=144;state.custom_renderer_visible_animation_count=0;
 call();assert(cancels==3 && !state.custom_renderer_camera_ticket && displayed_x==704);
 // Exact programmatic recenter, zoom, wrap and repeated reversal remain native.
 patch_Main_Screen_Form_move_camera(&screen,0,1000,320,0,true);
 call();assert(screen.camera_x==1000 && displayed_x==1000);
 state.custom_renderer_zoom_tile_width=160;call();
 assert(state.custom_renderer_display_view.tile_width==160);
 patch_Main_Screen_Form_move_camera(&screen,0,8180,0,0,true);call();
 patch_Main_Screen_Form_move_camera(&screen,0,screen.camera_x+32,0,1,false);
 assert(screen.camera_x==20);call();assert(displayed_x==20);
 patch_Main_Screen_Form_move_camera(&screen,0,screen.camera_x-32,0,1,false);
 assert(screen.camera_x==8180);call();assert(displayed_x==8180);
 // A pending same-view publication must not turn the next poll into a
 // first-map barrier. Repeat to cover the entire delayed refresh interval.
 unsigned before=begins;
 for(int n=0;n<20;++n){call(false);assert(state.custom_renderer_display_valid && begins==before);}
 native_move(&screen,0,500,100,1,false);call(false);
 assert(!state.custom_renderer_display_valid && begins==before);
 state.custom_renderer_async_enabled=false;call();assert(!state.custom_renderer_display_valid && !state.custom_renderer_camera_ticket);
 // The accepted movement sight capture bypasses the ambient rate gate and
 // starts copied preparation without adopting a canvas or advancing animation.
 state.custom_renderer_visible_animation_count=0;state.custom_renderer_camera_ticket=0;
 early_capture=[](void* canvas,c3x_renderer_camera_request_v1 const* r,long long* ticket)->int{
  assert(canvas==&image&&r->frame&&state.custom_renderer_capture_only);*ticket=99;return C3X_RENDERER_RESULT_PENDING;
 };
 assert(capture_custom_renderer_native_view(&bic.Map.Renderer,2,&state.custom_renderer_display_view,2));
 assert(!state.custom_renderer_camera_ticket&&!state.custom_renderer_capture_only);
 // Saved-game loading disables navigation, but the complete publication still
 // needs its camera identity when native startup clears the main canvas.
 state.custom_renderer_async_enabled=true;screen.is_now_loading_game=true;
 native_move(&screen,0,3616,394,0,true);call();
 assert(!state.custom_renderer_display_valid && state.custom_renderer_display_view.camera_x==3616 &&
        state.custom_renderer_display_view.camera_y==394);
}
'''
        run_cpp(program)
        # Same extracted hooks with the live GOG Animator inlead enabled.
        enabled = program.replace('#define Main_Screen_Form_move_camera native_move',
                                  '#define Main_Screen_Form_move_camera native_move\n#define Animator_update_display native_animator\n#define Main_Screen_Form_center_camera native_center')
        enabled = enabled.replace('struct Clock {', 'struct City {struct {struct {int Status2=0;}Data;}Base;} city;auto p_city_form=&city;\nstruct Clock {')
        enabled = enabled.replace('unsigned captures=0', 'void native_center(Main_Screen_Form*,int,int,int,int,bool,bool);\nunsigned captures=0')
        enabled = enabled.replace('void native_move(', 'unsigned native_calls=0,native_work=0;int overlay_x=0;\nvoid native_animator(Animator* a,int){++native_calls;if(screen.turn_end_flag || a->Units2_Count || a->fields[10] || a->fields[13]){++native_work;overlay_x=screen.camera_x;a->fields[10]=0;}}\nvoid native_move(')
        enabled = enabled.replace('s->camera_x=(x%8192', 's->animator.fields[10]=1;s->camera_x=(x%8192')
        enabled = enabled[:enabled.index('int main(){')]+r'''
void native_center(Main_Screen_Form* s,int,int x,int y,int reason,bool bounds,bool){
 assert(!state.current_config.enable_custom_rendering || state.custom_renderer_camera_exact);patch_Main_Screen_Form_move_camera(s,0,x,y,reason,bounds);
}
struct custom_renderer_native_view desired{};bool nav_pending=false,nav_ready=false;unsigned barriers=0;
int navigation(int action,void*,struct custom_renderer_native_view* v,c3x_renderer_camera_request_v1 const*){
 if(action==C3X_NAV_REQUEST){desired=*v;nav_pending=true;return C3X_RENDERER_RESULT_PENDING;}
 if(action==C3X_NAV_DISCARD){nav_pending=false;return C3X_RENDERER_RESULT_SUPERSEDED;}
 if(!nav_pending)return C3X_RENDERER_RESULT_SUPERSEDED;
 if(action==C3X_NAV_POLL&&!nav_ready)return C3X_RENDERER_RESULT_PENDING;
 barriers+=action==C3X_NAV_BARRIER;*v=desired;nav_pending=false;return C3X_RENDERER_RESULT_OK;
}
int main(){
 Vtable vt{reinterpret_cast<void*>(&capture)};bic.Map.Renderer.vtable=&vt;
 state.custom_renderer_camera_begin=begin;state.custom_renderer_camera_poll=poll;state.custom_renderer_camera_cancel=cancel;
 state.custom_renderer_navigation=navigation;state.custom_renderer_display_valid=true;
 state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 unsigned no_op_captures=captures;
 patch_Main_Screen_Form_move_camera(&screen,0,0,0,1,false);
 assert(captures==no_op_captures&&!nav_pending&&state.custom_renderer_display_valid);
 // Zoom bounds are re-clamped while a native map/HUD refresh is pending.
 // The same camera must not become an invalid first-map transaction.
 assert(screen.animator.fields[10]);
 patch_Main_Screen_Form_move_camera(&screen,0,0,0,1,false);
 assert(captures==no_op_captures&&!nav_pending&&state.custom_renderer_display_valid);
 screen.animator.fields[10]=0; // The stub marks dirty even for a native no-op.
 // The requested camera is normalized by native code, but input, unit culling,
 // wrap canvases and picking still observe the displayed camera while pending.
 patch_Main_Screen_Form_move_camera(&screen,0,96,64,1,false);
 assert(state.custom_renderer_unit_representatives_dirty);
 assert(resolved_x==96&&resolved_y==64);
 assert(nav_pending&&screen.camera_x==0&&screen.camera_y==0&&desired.camera_x==96&&desired.min_x==1);
 for(int i=0;i<20;++i){patch_Animator_update_display(&screen.animator,0);assert(screen.camera_x==0&&overlay_x==0);}
 assert(native_calls==20&&native_work==0); // Native early returns, never skipped calls.
 nav_ready=true;patch_Animator_update_display(&screen.animator,0);
 assert(screen.camera_x==96&&screen.camera_y==64&&screen.TileX_Min==1&&overlay_x==96&&native_work==1);
 state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);nav_ready=false;
 // A native action starts while a later pan is pending: it advances once at the
 // exact requested view, not at the old view and not after a postponed action.
 patch_Main_Screen_Form_move_camera(&screen,0,160,96,1,false);screen.animator.Units2_Count=1;
 patch_Animator_update_display(&screen.animator,0);
 assert(barriers==1&&screen.camera_x==160&&overlay_x==160&&native_work==2);
 screen.animator.Units2_Count=0;
 state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 patch_Main_Screen_Form_move_camera(&screen,0,224,96,1,false);assert(nav_pending);
 // Selecting a new unit/programmatic recenter supersedes the pending pan and
 // preserves immediate vanilla movement. It is never delayed by this fast path.
 patch_Main_Screen_Form_center_camera(&screen,0,1200,320,1,false,false);
 assert(!state.custom_renderer_camera_exact);
 assert(!nav_pending&&screen.camera_x==1200&&screen.camera_y==320&&!state.custom_renderer_display_valid);
 patch_Animator_update_display(&screen.animator,0);assert(overlay_x==1200);
 state.custom_renderer_display_valid=true;state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 patch_Main_Screen_Form_move_camera(&screen,0,8260,-32,1,false);assert(nav_pending&&desired.camera_x==68&&desired.camera_y==4064);
 // Combat owns the displayed camera even between clips (empty native queue).
 state.combat_unit_display_override_active=true;patch_Animator_update_display(&screen.animator,0);
 assert(!nav_pending&&screen.camera_x==1200&&screen.camera_y==320);
 state.combat_unit_display_override_active=false;screen.animator.fields[10]=0;
 patch_Main_Screen_Form_move_camera(&screen,0,8260,-32,1,false);assert(nav_pending);
 // Config-off delegates immediately with no renderer navigation side effects.
 state.current_config.enable_custom_rendering=false;patch_Animator_update_display(&screen.animator,0);
 assert(nav_pending&&screen.camera_x==1200&&screen.camera_y==320&&barriers==1);
 unsigned before=captures;patch_Main_Screen_Form_move_camera(&screen,0,200,100,1,false);
 assert(screen.camera_x==200&&captures==before);
 // Config-off observed before Animator still settles the normalized intent.
 state.current_config.enable_custom_rendering=true;state.custom_renderer_display_valid=true;
 screen.animator.fields[10]=0;state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 patch_Main_Screen_Form_move_camera(&screen,0,300,100,1,false);assert(nav_pending);
 state.current_config.enable_custom_rendering=false;settle_custom_renderer_navigation(C3X_NAV_BARRIER);
 assert(!nav_pending&&screen.camera_x==300&&screen.animator.fields[10]);
 // A viewer switch must not carry the previous viewer's requested camera.
 state.current_config.enable_custom_rendering=true;screen.animator.fields[10]=0;
 state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 patch_Main_Screen_Form_move_camera(&screen,0,400,100,1,false);assert(nav_pending);
 ++screen.Player_CivID;nav_ready=true;patch_Animator_update_display(&screen.animator,0);
 assert(!nav_pending&&screen.camera_x==300);
 // Real player turns keep turn_end_flag set even with no directed action.
 // Native updates continue on every poll, using one displayed camera until
 // the completed map is ready. The turn flag cannot bypass this transaction.
 screen.Player_CivID=2;state.custom_renderer_viewer_civ_id=2;
 screen.turn_end_flag=true;screen.animator.fields[10]=0;nav_ready=false;
 state.custom_renderer_display_valid=true;state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 auto before_work=native_work;auto before_camera=screen.camera_x;
 patch_Main_Screen_Form_move_camera(&screen,0,500,100,1,false);
 assert(nav_pending&&screen.camera_x==before_camera);
 for(int n=0;n<20;++n){patch_Animator_update_display(&screen.animator,0);assert(screen.camera_x==before_camera&&overlay_x==before_camera);}
 assert(native_work==before_work+20);
 nav_ready=true;patch_Animator_update_display(&screen.animator,0);
 assert(screen.camera_x==500&&overlay_x==500&&native_work==before_work+21);
 // Loading and gameplay centering use the same native path without GPU copies.
 static unsigned loading_copies=0;
 state.custom_renderer_native_image=[](int,void*,void*,void const*,void const*,unsigned)->int {
  ++loading_copies;return 1;
 };
 for(bool custom:{false,true})for(bool loading:{false,true})for(char bar:{0,1}){
  state.current_config.enable_custom_rendering=custom;screen.is_now_loading_game=loading;
  screen.GUI.field_574[3]=bar;
  patch_Main_Screen_Form_center_camera(&screen,0,600,100,0,false,false);
  assert(!loading_copies&&!state.custom_renderer_camera_exact&&screen.camera_x==600);
 }
 // A debug pan captures viewer zero and is adopted in that same scope.
 screen.is_now_loading_game=false;screen.GUI.field_574[3]=0;
 state.custom_renderer_display_valid=true;screen.animator.fields[10]=0;
 state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 debug_mode_bits=8;state.custom_renderer_viewer_civ_id=0;
 patch_Main_Screen_Form_move_camera(&screen,0,700,100,1,false);
 assert(nav_pending&&screen.camera_x==600);nav_ready=true;
 patch_Animator_update_display(&screen.animator,0);assert(screen.camera_x==700&&!nav_pending);
 // Online games must retain the real player even with stale debug bits.
 online=true;state.custom_renderer_viewer_civ_id=2;screen.animator.fields[10]=0;
 state.custom_renderer_display_view=custom_renderer_native_view(&bic.Map.Renderer);
 patch_Main_Screen_Form_move_camera(&screen,0,800,100,1,false);
 assert(nav_pending);patch_Animator_update_display(&screen.animator,0);assert(screen.camera_x==800);


}
'''
        run_cpp(enabled)
