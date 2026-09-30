"""Startup leaves camera/UI sequencing to Civ III; native draws own publication."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class LoadingViewTests(unittest.TestCase):
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
        gate = source.split('\tMap * map =', 1)[0]
        run_cpp(r'''
#include <cassert>
constexpr int IS_OK=1,C3X_RENDERER_RESULT_PENDING=4;
struct {struct {bool enable_custom_rendering=true;}current_config;
 int custom_renderer_init_state=IS_OK,custom_renderer_viewer_civ_id=2;
 bool custom_renderer_draw_in_progress=false,custom_renderer_frame_active=false,
 custom_renderer_capture_only=false,custom_renderer_display_valid=true;}state,*is=&state;
struct {bool is_now_loading_game=false;int Player_CivID=2;}form,*p_main_screen_form=&form;
unsigned debug=0;auto p_debug_mode_bits=&debug;bool online=false;
bool is_online_game(){return online;}
int admit(){''' + gate + r'''
 return 1;
}
int main(){
 assert(admit()==1);debug=8;assert(admit()==4);
 state.custom_renderer_viewer_civ_id=0;assert(admit()==1);
 online=true;assert(admit()==4);state.custom_renderer_viewer_civ_id=2;assert(admit()==1);
 online=false;debug=0;form.is_now_loading_game=true;assert(admit()==4);
 form.is_now_loading_game=false;state.current_config.enable_custom_rendering=false;assert(admit()==4);
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
