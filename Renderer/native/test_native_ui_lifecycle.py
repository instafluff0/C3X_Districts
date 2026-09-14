"""Run the actual popup and command-button scopes through nested native callbacks."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT=Path(__file__).resolve().parents[2]

class NativeUiLifecycleTests(unittest.TestCase):
    def test_popup_lifetime_and_complete_button_reconstruction(self):
        source=(ROOT/'injected_code.c').read_text()
        popup=source[source.index('struct pause_for_popup {'):source.index('void\npop_up_in_game_error')]
        buttons=source[source.index('void __fastcall\npatch_Main_GUI_set_up_unit_command_buttons'):source.index('void \nclear_highlighted_worker_tiles_and_redraw')]
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#define __fastcall
#define __ 0
struct LARGE_INTEGER {long long QuadPart;};
struct Base_Form {};
struct Table {void (*m02_Show_Disabled)(Base_Form*);void (*m01_Show_Enabled)(Base_Form*,int,int);};
struct Button {struct {int Status2=0;} Base_Data;Table* vtable;int* Images[4]{};int field_5FC[14]{};};
struct Command_Button {struct Button Button;int Command;};
struct Main_GUI {Command_Button Unit_Command_Buttons[42];};
struct Unit_Body {int X=1,Y=2,CivID=3;};struct Unit {Unit_Body Body;};
struct Screen {Unit* Current_Unit;} screen;auto p_main_screen_form=&screen;
struct Bic {int Map;} bic;auto p_bic_data=&bic;
constexpr int IS_OK=1,UCV_Build_City=1,CLV_CITY_TOO_CLOSE=1,CL_CITY_TOO_CLOSE_BUTTON_TOOLTIP=0;
struct State {
 bool paused_for_popup=false,custom_renderer_modal=false;long long time_spent_paused_during_popup=0;
 int show_popup_was_called=0,disabled_command_img_state=IS_OK,disabled_build_city_button_img=0;
 struct {bool enable_districts=true;int minimum_city_separation=2;} current_config;
 char* c3x_labels[1];
} state;auto is=&state;
long long now=0;void QueryPerformanceCounter(LARGE_INTEGER* q){q->QuadPart=++now;}
int popup_calls=0,button_phases=0,tooltip_calls=0;
int show_popup(void*,int,int,int);
void phase(){assert(state.custom_renderer_modal);++button_phases;}
void recompute_resources_if_necessary(){phase();}
void Main_GUI_set_up_unit_command_buttons(Main_GUI*);
void set_up_stack_bombard_buttons(Main_GUI*){phase();}
void set_up_stack_worker_buttons(Main_GUI*){phase();}
void set_up_district_buttons(Main_GUI*){phase();}
int patch_Map_check_city_location(int*,int,int,int,int,bool){phase();return CLV_CITY_TOO_CLOSE;}
void Button_set_tooltip(Button*,int,char* text){phase();assert(!std::strcmp(text,"Wait 2 tiles"));++tooltip_calls;}
void hide(Base_Form*){phase();}void show(Base_Form*,int,int){phase();}
''' + popup.replace('this','self') + buttons.replace('this','self') + r'''
int show_popup(void* self,int,int nesting,int){
 ++popup_calls;assert(state.paused_for_popup);
 // The constructor returns before the rest of native layout and its dialog.
 auto constructor=[](){assert(state.paused_for_popup);};constructor();
 assert(state.paused_for_popup);
 if(nesting)assert(patch_show_popup(self,0,0,0)==73);
 assert(state.paused_for_popup);return 73;
}
void Main_GUI_set_up_unit_command_buttons(Main_GUI* gui){
 phase();assert(patch_show_popup(gui,0,1,0)==73);
 assert(state.custom_renderer_modal && !state.paused_for_popup);phase();
}
int main(){
 assert(patch_show_popup(&state,0,1,0)==73);
 assert(popup_calls==2 && !state.paused_for_popup && state.show_popup_was_called);
 assert(state.time_spent_paused_during_popup==1); // Nested pause counted once.
 state.paused_for_popup=true;patch_show_popup(&state,0,0,0);assert(state.paused_for_popup);
 assert(state.time_spent_paused_during_popup==1);state.paused_for_popup=false;
 Table table{hide,show};Main_GUI gui{};Unit unit;screen.Current_Unit=&unit;
 char label[]="Wait $NUM0 tiles";state.c3x_labels[0]=label;
 gui.Unit_Command_Buttons[0].Button.vtable=&table;
 gui.Unit_Command_Buttons[0].Button.Base_Data.Status2=1;
 gui.Unit_Command_Buttons[0].Command=UCV_Build_City;
 patch_Main_GUI_set_up_unit_command_buttons(&gui);
 assert(!state.custom_renderer_modal && button_phases==10 && tooltip_calls==1);
 state.custom_renderer_modal=true;patch_Main_GUI_set_up_unit_command_buttons(&gui);
 assert(state.custom_renderer_modal && button_phases==20 && tooltip_calls==2);
 assert(!state.paused_for_popup);
 std::puts("UI scopes: complete popup lifetime, nested restoration and all command-button phases pass");
}
''')

if __name__=='__main__':unittest.main()
