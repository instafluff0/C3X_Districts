"""Startup leaves camera/UI sequencing to Civ III; native draws own publication."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class LoadingViewTests(unittest.TestCase):
    def test_changed_camera_or_native_projection_waits_before_overlay_paint(self):
        source = (ROOT / 'injected_code.c').read_text()
        start = source.index('\tbool changed_view =', source.index('composite_custom_renderer_frame ()'))
        stop = source.index('\n\t\tvoid (WINAPI * sleep_ms)', start)
        decision = source[start:stop]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
using HMODULE=void*;
constexpr int C3X_RENDERER_RESULT_PENDING=2;
struct {int camera_x=20,camera_y=30;}screen,*p_main_screen_form=&screen;
struct {bool is_zoomed_out=false;}bic,*p_bic_data=&bic;
struct {struct{int camera_x=20,camera_y=30,native_width=128;}custom_renderer_display_view;
 int custom_renderer_display_viewer_epoch=1,custom_renderer_viewer_epoch=1;
 bool custom_renderer_display_valid=true;void* custom_renderer_module=nullptr;}state,*is=&state;
bool waits(int resident_result){
''' + decision + r'''
 return true;
 }return false;
}
int main(){
 unsigned cases=0;
 for(int ready=0;ready<2;++ready)for(int change=0;change<6;++change){
  state={};screen={};bic={};
  if(change==1)++screen.camera_x;
  if(change==2)--screen.camera_y;
  if(change==3)bic.is_zoomed_out=true;
  if(change==4)++state.custom_renderer_viewer_epoch;
  if(change==5)state.custom_renderer_display_valid=false;
  assert(waits(ready?1:2)==(!ready&&change!=0));++cases;
 }
 // The return from city 0.5x must wait as well; a ready map never blocks.
 state={};screen={};bic={};state.custom_renderer_display_view.native_width=64;
 assert(waits(2)&&!waits(1));
 // Deferred scroll still sees the displayed native camera and stays asynchronous.
 state={};assert(!waits(2));
 std::printf("PASS map/overlay boundary: cases=%u city_Z_entry_exit=1 deferred_scroll_async=1\n",cases);
}
''')

    def test_city_repaint_keeps_completed_screen_until_matching_map(self):
        source = (ROOT / 'injected_code.c').read_text()
        begin = source.index('patch_JGL_Graphsy_present (void * graph, int edx, RECT * rect)\n{\n')
        guard = source[begin:source.index('\t// The process-owned module', begin)]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
constexpr int IS_OK=1;
struct RECT {int left,top,right,bottom;};
struct {int camera_x=12,camera_y=34;}screen,*p_main_screen_form=&screen;
struct {bool is_zoomed_out=false;struct{struct{void* spotlight_on_city=&screen;}Renderer;}Map;}bic,*p_bic_data=&bic;
struct {struct{bool enable_custom_rendering=true;}current_config;int custom_renderer_init_state=IS_OK;
 int custom_renderer_presented_frames=1;struct{int camera_x=12,camera_y=34,native_width=128;}custom_renderer_display_view;}state,*is=&state;
unsigned presented=0;
int ''' + guard + r'''
 assert(graph==&screen&&edx==17&&rect);++presented;return 917;
}
int main(){RECT rect{};unsigned cases=0;
 for(int change=0;change<4;++change)for(int bypass=0;bypass<7;++bypass){
  state={};screen={};bic={};p_main_screen_form=&screen;p_bic_data=&bic;
  if(change==1)screen.camera_x++;if(change==2)screen.camera_y++;if(change==3)bic.is_zoomed_out=true;
  if(bypass==1)state.current_config.enable_custom_rendering=false;
  if(bypass==2)state.custom_renderer_init_state=0;
  if(bypass==3)state.custom_renderer_presented_frames=0;
  if(bypass==4)bic.Map.Renderer.spotlight_on_city=nullptr;
  if(bypass==5)p_main_screen_form=nullptr;if(bypass==6)p_bic_data=nullptr;
  auto before=presented;bool hold=change&&bypass==0;
  assert(patch_JGL_Graphsy_present(&screen,17,&rect)==(hold?0:917));assert(presented==before+!hold);++cases;
 }
 state={};bic={};p_main_screen_form=&screen;p_bic_data=&bic;screen={};bic.is_zoomed_out=true;
 assert(patch_JGL_Graphsy_present(&screen,17,&rect)==0);
 state.custom_renderer_display_view.native_width=64;
 assert(patch_JGL_Graphsy_present(&screen,17,&rect)==917); // exact map completes; no timer or new input needed
 std::printf("PASS city repaint publication: cases=%u native_off_delegation=1 matching_map_releases=1\n",cases);
}
''')

    def test_loading_has_no_speculative_camera_or_progress_ui(self):
        source = (ROOT / 'injected_code.c').read_text()
        self.assertNotIn('prepare_custom_renderer_loading_view', source)
        load = source.split('patch_load_scenario (', 1)[1].split('// Initialize Trade Net X', 1)[0]
        self.assertIn('ensure_custom_renderer_loaded ();', load)
        self.assertNotIn('Main_GUI_label_loading_bar', load)
        hook = 'void patch_MappedFile_deinit_after_saving_or_loading' + source.split(
            'patch_MappedFile_deinit_after_saving_or_loading', 1)[1].split('bool __fastcall', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <cstddef>
struct MappedFile{};
struct {void* accessing_save_file;} state,*is=&state;
unsigned deinits=0;
void MappedFile_deinit(MappedFile* file){assert(file&&!state.accessing_save_file);++deinits;}
''' + hook.replace('this', 'file') + r'''
int main(){MappedFile file;state.accessing_save_file=&file;
 patch_MappedFile_deinit_after_saving_or_loading(&file);assert(deinits==1);}
''')

    def test_offscreen_preparation_requires_exploration(self):
        source = (ROOT / 'injected_code.c').read_text()
        decision = source.split('bool prepare_appearance =', 1)[1].split(';', 1)[0]
        run_cpp(r'''
#include <cassert>
constexpr unsigned C3X_RENDERER_TILE_EXPLORED=2;
unsigned visibility=0;int calls=0;
unsigned capture_custom_renderer_visibility(void*,int,int,int){++calls;return visibility;}
bool prepare(int dx,int dy){int warm_min_x=-8,warm_max_x=8,warm_min_y=-8,warm_max_y=8;
 void* tile=nullptr;int viewer=2,x=dx,y=dy;
 return ''' + decision + r''';
}
int main(){
 assert(!prepare(0,0));visibility=2;assert(prepare(0,0));
 assert(prepare(-8,8));assert(!prepare(-9,0));assert(calls==3);
 visibility=0;assert(!prepare(8,-8));
}
''')

    def test_world_pages_admit_the_native_debug_viewer(self):
        source = (ROOT / 'injected_code.c').read_text().split(
            'capture_custom_renderer_world_page (struct c3x_renderer_world_page_v1 * page)\n{', 1)[1]
        # Execute the complete callback: page validation and the initial seed's
        # narrow loading exception must remain coupled to the ordinary guards.
        callback = 'int capture_custom_renderer_world_page(c3x_renderer_world_page_v1* page) {\n' + source.split('\n}\n', 1)[0] + '\n}\n'
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstddef>
#include <initializer_list>
constexpr int IS_OK=1;
struct Map_Renderer {};
struct Tile {struct {unsigned Fog_Of_War=0,FOWStatus=0,V3=0,Visibility=0,field_D0_Visibility=0;}Body;};
struct Map {int Width=4,Height=4,Flags=1;Tile* Tiles;Map_Renderer Renderer;}map;
struct {Map Map;}bic;auto p_bic_data=&bic;
Tile tiles[8],null_tile;auto p_null_tile=&null_tile;
unsigned topology[8]={};unsigned long long visibility[8]={};
c3x_renderer_tile_v1 captured[1]{},output[8]{};
unsigned thread=9;
unsigned current_thread(){return thread;}
struct {struct {bool enable_custom_rendering=true;}current_config;
 int custom_renderer_init_state=IS_OK,custom_renderer_viewer_civ_id=2;
 bool custom_renderer_draw_in_progress=false,custom_renderer_frame_active=false,
 custom_renderer_capture_only=false,custom_renderer_display_valid=true,
 custom_renderer_initial_world_capture=false,custom_renderer_loading_world_capture=false,custom_renderer_capture_failed=false;
 unsigned (*custom_renderer_probe_thread_id)()=current_thread;
 unsigned custom_renderer_probe_owner=9;
 Map_Renderer* custom_renderer_target=&bic.Map.Renderer;
 c3x_renderer_tile_v1* custom_renderer_tiles=captured;
 int custom_renderer_tile_count=1,custom_renderer_world_topology_count=8;
 unsigned* custom_renderer_world_topology=topology;
 unsigned long long* custom_renderer_world_visibility=visibility;
 long long custom_renderer_map_epoch=1,custom_renderer_viewer_epoch=2,
 custom_renderer_display_viewer_epoch=1,custom_renderer_world_topology_revision=3,
 custom_renderer_visibility_revision=4;}state,*is=&state;
struct {bool is_now_loading_game=false;int Player_CivID=2;}form,*p_main_screen_form=&form;
unsigned debug=0;auto p_debug_mode_bits=&debug;bool online=false;
bool is_online_game(){return online;}
int reads=0;
Tile* tile_at(int x,int y){return &tiles[(y*bic.Map.Width+x)/2];}
bool read_custom_renderer_world_record(c3x_renderer_tile_v1* record,int viewer,int mask,
 int x,int y,Tile*){
 assert(viewer==state.custom_renderer_viewer_civ_id&&mask==(state.custom_renderer_loading_world_capture?0:77));
 *record={};record->tile_x=x;record->tile_y=y;++reads;return true;
}
''' + callback + r'''
c3x_renderer_world_page_v1 page{};
void reset(){
 state={};form={};debug=0;online=false;thread=9;
 bic.Map.Width=4;bic.Map.Height=4;bic.Map.Flags=1;bic.Map.Tiles=tiles;
 captured[0].visibility_mask=77;reads=0;
 page={};page.struct_size=sizeof page;page.tiles=output;page.capacity=8;
 page.identity={1,2,4,3};page.frame.world_width_tiles=4;page.frame.world_height_tiles=4;
 page.frame.world_wrap_x=1;page.frame.world_topology_revision=3;
}
int admit(){return capture_custom_renderer_world_page(&page);}
void initial(){
 reset();state.custom_renderer_initial_world_capture=true;
 state.custom_renderer_draw_in_progress=state.custom_renderer_frame_active=true;
 state.custom_renderer_display_valid=false;form.is_now_loading_game=true;
}
int main(){
 reset();
 assert(admit()==1);debug=8;assert(admit()==4);
 state.custom_renderer_viewer_civ_id=0;assert(admit()==1);
 online=true;assert(admit()==4);state.custom_renderer_viewer_civ_id=2;assert(admit()==1);
 online=false;debug=0;form.is_now_loading_game=true;assert(admit()==4);
 form.is_now_loading_game=false;state.current_config.enable_custom_rendering=false;assert(admit()==4);
 // Ordinary reads never borrow loading, drawing, or unpublished-view access.
 for(auto flag:{&decltype(state)::custom_renderer_draw_in_progress,
               &decltype(state)::custom_renderer_frame_active,
               &decltype(state)::custom_renderer_capture_only}){
  reset();state.*flag=true;assert(admit()==4&&!reads);
 }
 reset();state.custom_renderer_display_valid=false;assert(admit()==4&&!reads);
 reset();assert(admit()==1&&reads==8);
 initial();assert(admit()==1&&reads==8);
 reset();form.is_now_loading_game=true;state.custom_renderer_display_valid=false;
 state.custom_renderer_loading_world_capture=true;state.custom_renderer_tiles=nullptr;state.custom_renderer_tile_count=0;
 assert(admit()==1&&reads==8);
 reset();form.is_now_loading_game=true;state.custom_renderer_loading_world_capture=true;thread=10;assert(admit()==4&&!reads);
 reset();form.is_now_loading_game=true;state.custom_renderer_loading_world_capture=true;state.custom_renderer_draw_in_progress=true;assert(admit()==4&&!reads);
 // The flag alone cannot relax any guard: only a completed capture on its
 // owner thread, still within the new viewer's synchronous seed, can do so.
 initial();thread=10;assert(admit()==4&&!reads);
 initial();state.custom_renderer_probe_thread_id=nullptr;assert(admit()==4&&!reads);
 initial();state.custom_renderer_capture_only=true;assert(admit()==4&&!reads);
 initial();state.custom_renderer_capture_failed=true;assert(admit()==4&&!reads);
 initial();state.custom_renderer_target=nullptr;assert(admit()==4&&!reads);
 initial();state.custom_renderer_tiles=nullptr;assert(admit()==4&&!reads);
 initial();state.custom_renderer_tile_count=0;assert(admit()==4&&!reads);
 initial();state.custom_renderer_display_viewer_epoch=2;assert(admit()==4&&!reads);
 initial();state.custom_renderer_draw_in_progress=false;assert(admit()==4&&!reads);
 initial();state.custom_renderer_frame_active=false;assert(admit()==4&&!reads);
 initial();state.custom_renderer_viewer_civ_id=0;assert(admit()==4&&!reads);
 initial();bic.Map.Tiles=nullptr;assert(admit()==4&&!reads);
 initial();state.custom_renderer_init_state=0;assert(admit()==4&&!reads);
 initial();state.current_config.enable_custom_rendering=false;assert(admit()==4&&!reads);
 // Identity and indexed full-world shape remain mandatory in the exception.
 initial();++page.identity.map_epoch;assert(admit()==5&&!reads);
 initial();++page.identity.viewer_epoch;assert(admit()==5&&!reads);
 initial();++page.identity.scene_epoch;assert(admit()==5&&!reads);
 initial();++page.identity.visibility_epoch;assert(admit()==5&&!reads);
 initial();++page.frame.world_topology_revision;assert(admit()==5&&!reads);
 initial();page.frame.world_width_tiles=6;assert(admit()==5&&!reads);
 initial();page.frame.world_wrap_x=0;assert(admit()==5&&!reads);
 initial();page.first=0xfffffffeu;assert(admit()==5&&!reads);
 initial();state.custom_renderer_world_topology_count=7;assert(admit()==4&&!reads);
 reset();page.capacity=129;assert(admit()==2&&!reads);
 assert(capture_custom_renderer_world_page(nullptr)==2);
}
''')

    def test_native_capture_survives_debug_pass_masks(self):
        source = (ROOT / 'injected_code.c').read_text().split(
            'patch_Map_Renderer_m19_Draw_Tile_by_XY_and_Flags', 1)[1]
        block = source[source.index('\tif (is->current_config.enable_custom_rendering && is->custom_renderer_frame_active'):]
        block = block.split('\n\t// Custom rendering owns', 1)[0].replace('this', 'renderer')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <initializer_list>
#define __fastcall
#define __ 0
constexpr int C3X_RENDERER_RESULT_ERROR=0;
struct Map_Renderer;
struct Vtable {void (*m21_Draw_Tiles_by_Flags)(Map_Renderer*,int,int,int,int,Map_Renderer*,void*,int,int,int);};
struct Map_Renderer {Vtable* vtable;};
struct {struct {bool enable_custom_rendering=true;}current_config;
 bool custom_renderer_frame_active=true,custom_renderer_composited=false,
 custom_renderer_capture_only=false,custom_renderer_capture_failed=false;}state,*is=&state;
int captures=0,composites=0,tiles=0,errors=0;
void capture(Map_Renderer* renderer,int,int viewer,int x,int y,Map_Renderer* target,void* clip,int tx,int ty,int flags){
 assert(state.custom_renderer_capture_only&&state.custom_renderer_composited);
 assert(renderer==target&&viewer==0&&x==-1&&y==-1&&!clip&&tx==-1&&ty==-1&&flags==9);
 ++captures;tiles=12;
}
void capture_custom_renderer_topology(int viewer,int mask){assert(viewer==0&&mask==15&&tiles==12);}
void composite_custom_renderer_frame(){assert(!state.custom_renderer_capture_only&&tiles==12);++composites;}
void log_custom_renderer_event(char const*,int){++errors;}
void draw(Map_Renderer* renderer,int param_8){int param_1=0,param_5=15;auto map_renderer=renderer;
''' + block + r'''
}
int main(){Vtable vt{capture};Map_Renderer renderer{&vt};
 // Native debug mode applies ~Flags to its first four passes. Test default,
 // individual hidden layers, and every layer hidden; final fog is unmasked.
 for(int mask:{0,0x1ae80,9,0x1fef0,0x1ffff}){
  state.custom_renderer_composited=false;captures=composites=tiles=errors=0;
  for(int pass:{9&~mask,4&~mask,0x1fef0&~mask,2&~mask,0x100})
   for(int tile=0;tile<12;++tile)draw(&renderer,pass);
  assert(captures==1&&composites==1&&!errors&&!state.custom_renderer_capture_only);
 }
 state.custom_renderer_composited=false;state.custom_renderer_capture_failed=true;
 draw(&renderer,9);assert(errors==1&&composites==1);
 state.custom_renderer_composited=false;state.current_config.enable_custom_rendering=false;
 draw(&renderer,9);assert(!state.custom_renderer_composited&&errors==1);
}
''')


if __name__ == '__main__':
    unittest.main()
