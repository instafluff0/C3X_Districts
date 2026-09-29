"""Loading waits for the first camera publication; ordinary map drawing never waits."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class LoadingViewTests(unittest.TestCase):
    def test_only_complete_loading_camera_prepares_view(self):
        source = (ROOT / 'injected_code.c').read_text()
        helper = 'void\nprepare_custom_renderer_loading_view' + source.split('void\nprepare_custom_renderer_loading_view', 1)[1].split('\nstruct custom_renderer_native_view', 1)[0]
        # The injected ABI is 32-bit; preserve pointer width in this host fixture.
        helper = helper.replace('(Map_Renderer *, int, int, int, int)', '(Map_Renderer *, int, int, std::intptr_t, int)')
        helper = helper.replace('(int)&form->Base_Data.Canvas', '(std::intptr_t)&form->Base_Data.Canvas')
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#define __fastcall
#define __ 0
struct PCX_Image {struct {void* Image=this;}JGL;};
struct Main_GUI {int field_574[4]{};};
struct Main_Screen_Form {Main_GUI GUI;struct {PCX_Image Canvas;}Base_Data;
 int camera_x=0,camera_y=0,Player_CivID=2,TileX_Min=0,TileX_Max=20,TileY_Min=0,TileY_Max=20;} screen;
auto p_main_screen_form=&screen;
struct Map_Renderer;
struct Vtable {void(*m71_Draw_Tiles)(Map_Renderer*,int,int,std::intptr_t,int);};
struct Map_Renderer : PCX_Image {Vtable* vtable;};
struct Bic {struct {Map_Renderer Renderer;int Width=100,Height=100;}Map;} bic;
auto p_bic_data=&bic;
struct State {struct {bool enable_custom_rendering=true;}current_config;
 unsigned custom_renderer_presented_frames=0;bool custom_renderer_draw_in_progress=false,custom_renderer_async_presented=false;} state;
auto is=&state;
unsigned draws=0,labels=0;int last_result=-1;
void Main_GUI_label_loading_bar(Main_GUI* gui,int,int increment,char const* text){
 assert(gui==&screen.GUI && increment==0 && std::strcmp(text,"Preparing map")==0);++labels;
}
void debug(char const*){}auto p_OutputDebugStringA=&debug;
void log_custom_renderer_event(char const*,int result){last_result=result;}
void draw(Map_Renderer* renderer,int,int viewer,std::intptr_t canvas,int flags){
 assert(renderer==&bic.Map.Renderer && viewer==2 && canvas==(std::intptr_t)&screen.Base_Data.Canvas && flags==0);
 assert(screen.GUI.field_574[3] && !state.custom_renderer_draw_in_progress);
 ++draws;++state.custom_renderer_presented_frames;state.custom_renderer_async_presented=true;
}
''' + helper + r'''
int main(){
 Vtable vt{draw};bic.Map.Renderer.vtable=&vt;
 prepare_custom_renderer_loading_view(&screen);assert(!draws&&!labels); // Regular gameplay cannot block.
 screen.GUI.field_574[3]=1;state.current_config.enable_custom_rendering=false;
 prepare_custom_renderer_loading_view(&screen);assert(!draws); // Vanilla remains untouched.
 state.current_config.enable_custom_rendering=true;
 Main_Screen_Form other;prepare_custom_renderer_loading_view(&other);assert(!draws);
 for(int* field:{&screen.Player_CivID,&screen.TileX_Max,&screen.TileY_Max,&bic.Map.Width,&bic.Map.Height}){
  int saved=*field;*field=0;prepare_custom_renderer_loading_view(&screen);assert(!draws);*field=saved;
 }
 state.custom_renderer_draw_in_progress=true;prepare_custom_renderer_loading_view(&screen);assert(!draws);
 state.custom_renderer_draw_in_progress=false;
 void* image=bic.Map.Renderer.JGL.Image;bic.Map.Renderer.JGL.Image=nullptr;
 prepare_custom_renderer_loading_view(&screen);assert(!draws);bic.Map.Renderer.JGL.Image=image;
 image=screen.Base_Data.Canvas.JGL.Image;screen.Base_Data.Canvas.JGL.Image=nullptr;
 prepare_custom_renderer_loading_view(&screen);assert(!draws);screen.Base_Data.Canvas.JGL.Image=image;
 prepare_custom_renderer_loading_view(&screen);assert(draws==1 && labels==1 && last_result==C3X_RENDERER_RESULT_OK);
 screen.camera_x=128;prepare_custom_renderer_loading_view(&screen);assert(draws==2); // Native loading can still recenter.
 screen.GUI.field_574[3]=0;prepare_custom_renderer_loading_view(&screen);assert(draws==2); // Gameplay never enters preparation.
 screen.GUI.field_574[3]=1;state.custom_renderer_presented_frames=0;prepare_custom_renderer_loading_view(&screen);assert(draws==3); // New map.
}
''')

    def test_save_loader_warms_native_start_view_and_restores_saved_camera(self):
        source = (ROOT / 'injected_code.c').read_text()
        hook = 'void patch_MappedFile_deinit_after_saving_or_loading' + source.split('patch_MappedFile_deinit_after_saving_or_loading', 1)[1].split('bool __fastcall', 1)[0]
        run_cpp(r'''
#include <cassert>
#include <vector>
#define __ 0
#define Main_Screen_Form_center_camera native_center
struct MappedFile{};
unsigned deinits=0;
void MappedFile_deinit(MappedFile*){++deinits;}
struct Unit {struct {int X=73,Y=31;}Body;};
struct Map_Renderer{};
struct Main_Screen_Form {bool is_now_loading_game=true;
 struct {int field_574[4]{0,0,0,1};}GUI;
 int Player_CivID=1,camera_x=10,camera_y=20;Unit* Current_Unit=nullptr;}form;
auto p_main_screen_form=&form;
struct Bic {struct {int Width=100,Height=100;int Starting_Locations[32]{0,1586};Map_Renderer Renderer;}Map;}bic;
auto p_bic_data=&bic;
struct State {void* accessing_save_file=&form;struct {bool enable_custom_rendering=true;}current_config;}state;
auto is=&state;
struct custom_renderer_native_view {int x,y;};
struct custom_renderer_native_view custom_renderer_native_view(Map_Renderer*){return {form.camera_x,form.camera_y};}
void apply_custom_renderer_native_view(struct custom_renderer_native_view* v){form.camera_x=v->x;form.camera_y=v->y;}
bool Map_in_range(decltype(bic.Map)*,int,int x,int y){return x>=0&&x<100&&y>=0&&y<100;}
void native_center(Main_Screen_Form* f,int,int x,int y,int reason,bool bounds,bool force){
 assert(f==&form&&reason==0&&bounds&&!force);f->camera_x=x;f->camera_y=y;
}
std::vector<int> views;
void prepare_custom_renderer_loading_view(Main_Screen_Form* f){views.push_back(f->camera_x);views.push_back(f->camera_y);}
''' + hook.replace('this', 'file_arg') + r'''
int main(){
 MappedFile file;
 patch_MappedFile_deinit_after_saving_or_loading(&file);
 assert((views==std::vector<int>{10,20,73,31})); // Saved view and native startup view, with no saved-camera mutation.
 assert(form.camera_x==10&&form.camera_y==20&&!state.accessing_save_file&&deinits==1);
 views.clear();Unit unit;unit.Body.X=45;unit.Body.Y=67;form.Current_Unit=&unit;
 patch_MappedFile_deinit_after_saving_or_loading(&file);
 assert((views==std::vector<int>{10,20,45,67}));
 views.clear();state.current_config.enable_custom_rendering=false;
 patch_MappedFile_deinit_after_saving_or_loading(&file);assert(views.empty()&&deinits==3);
 state.current_config.enable_custom_rendering=true;form.is_now_loading_game=false;
 patch_MappedFile_deinit_after_saving_or_loading(&file);assert(views.empty()&&deinits==4); // Saving never prepares.
 form.is_now_loading_game=true;form.GUI.field_574[3]=0;
 patch_MappedFile_deinit_after_saving_or_loading(&file);assert(views.empty()&&deinits==5);
}
''')


if __name__ == '__main__':
    unittest.main()
