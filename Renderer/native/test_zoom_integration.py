"""Native delegation, display telemetry, and world-view input boundaries."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


def function(source, name):
    start = source.index('\n' + name + ' (') + 1
    end = source.index('\n}\n', start) + 3
    return source[start:end].replace('this', 'self')


class ZoomIntegrationTests(unittest.TestCase):
    def test_notification_scope_delegates_when_disabled_and_balances(self):
        body = function((ROOT/'injected_code.c').read_text(), 'patch_Main_GUI_draw_notifications')
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include <cstddef>
#include "Renderer/native/c3x_renderer_api.h"
struct Main_GUI{};struct Screen{struct{struct{struct{struct{void* Image=nullptr;}JGL;}Canvas;}Data;}Units_Control;}screen,*p_main_screen_form=&screen;
struct State{struct{bool enable_custom_rendering=false,enable_custom_rendering_zoom=true;}current_config;
 c3x_renderer_native_image_fn custom_renderer_native_image=nullptr;}state,*is=&state;
bool probe=true,scoped=false;int calls=0,begins=0,ends=0;bool custom_renderer_native_probe_on(){return probe;}
void Main_GUI_draw_notifications(Main_GUI*,int edx){assert(edx==73);++calls;
 assert(scoped==(state.current_config.enable_custom_rendering&&state.current_config.enable_custom_rendering_zoom&&probe&&state.custom_renderer_native_image));}
int native(int op,void*,void*,void const*,void const*,unsigned){
 if(op==C3X_NATIVE_FIXED_UI_BEGIN){assert(!scoped);scoped=true;++begins;return 1;}
 assert(op==C3X_NATIVE_FIXED_UI_END&&scoped);scoped=false;++ends;return 1;
}
void '''+body+r'''
int main(){Main_GUI gui;for(int i=0;i<16;++i){state.current_config.enable_custom_rendering=bool(i&1);
 state.current_config.enable_custom_rendering_zoom=bool(i&2);probe=bool(i&4);state.custom_renderer_native_image=i&8?native:nullptr;
 patch_Main_GUI_draw_notifications(&gui,73);assert(!scoped&&begins==ends);}
 assert(calls==16&&begins==1);}
''')

    def test_map_message_scope_preserves_native_call_and_attachment(self):
        source=(ROOT/'injected_code.c').read_text()
        scope=function(source,'custom_renderer_hud_scope')
        draw=function(source,'patch_MapMessage_draw').replace('(unsigned)self','unsigned(reinterpret_cast<uintptr_t>(self))')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include "Renderer/native/c3x_renderer_api.h"
struct JGL_Image{} image,map_image,unrelated;
struct PCX_Image{struct {JGL_Image* Image=&image;}JGL;};
struct Bic{struct{struct{PCX_Image canvas;void* spotlight_on_city=nullptr;}Renderer;}Map;}bic,*p_bic_data=&bic;
struct Screen{struct{PCX_Image Canvas;}Base_Data;struct{struct{PCX_Image Canvas;}Data;}Units_Control;}screen,*p_main_screen_form=&screen;
struct State{struct{bool enable_custom_rendering=false,enable_custom_rendering_zoom=true;}current_config;
 bool custom_renderer_trace_input=false;
 c3x_renderer_native_image_fn custom_renderer_native_image=nullptr;}state,*is=&state;
void debug(char const*){} auto p_OutputDebugStringA=debug;
struct RECT{int left,top,right,bottom;};struct MapMessage{char padding[0x28];RECT rect;};
bool probe=true,active=false;int begins=0,ends=0,calls=0;
bool custom_renderer_native_probe_on(){return probe;}
int native(int op,void* canvas,void*,void const* from,void const*,unsigned){
 if(op==C3X_NATIVE_HUD_END){assert(active);active=false;++ends;return 1;}
 assert(op==C3X_NATIVE_HUD_BEGIN&&(canvas==&image||canvas==&map_image)&&!active);
 auto anchor=static_cast<int const*>(from);assert(anchor[0]==210&&anchor[1]==86);active=true;++begins;return 1;
}
void MapMessage_draw(MapMessage*,int edx,PCX_Image* canvas,int shade){assert(edx==91&&shade==7&&canvas);++calls;}
int '''+scope+r'''
void '''+draw+r'''
int main(){PCX_Image canvas;MapMessage message{};message.rect={180,70,236,84};state.custom_renderer_native_image=native;
 for(int flags=0;flags<16;++flags){state.current_config.enable_custom_rendering=bool(flags&1);
  state.current_config.enable_custom_rendering_zoom=bool(flags&2);probe=bool(flags&4);canvas.JGL.Image=flags&8?&image:&unrelated;
  patch_MapMessage_draw(&message,91,&canvas,7);assert(!active&&begins==ends);
 }
 assert(calls==16&&begins==1&&message.rect.right-message.rect.left==56);
 bic.Map.Renderer.canvas.JGL.Image=&map_image;canvas.JGL.Image=&map_image;
 patch_MapMessage_draw(&message,91,&canvas,7);assert(calls==17&&begins==2&&begins==ends);
}
''')

    def test_city_spotlight_keeps_native_projection(self):
        body=function((ROOT/'injected_code.c').read_text(),'custom_renderer_zoom_enabled')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
struct {struct{bool enable_custom_rendering=true,enable_custom_rendering_zoom=true;}current_config;}state,*is=&state;
struct {bool is_now_loading_game=false;}screen,*p_main_screen_form=&screen;
struct {struct{struct{void* spotlight_on_city=nullptr;}Renderer;}Map;}bic,*p_bic_data=&bic;
int players=1,*p_player_bits=&players;
bool '''+body+r'''
int main(){assert(custom_renderer_zoom_enabled());
 bic.Map.Renderer.spotlight_on_city=&screen;assert(!custom_renderer_zoom_enabled());
 bic.Map.Renderer.spotlight_on_city=nullptr;assert(custom_renderer_zoom_enabled());
 state.current_config.enable_custom_rendering=false;assert(!custom_renderer_zoom_enabled());}
''')

    def test_city_center_keeps_same_pixel_attachment_and_native_off_behavior(self):
        body=function((ROOT/'injected_code.c').read_text(),'get_city_screen_center_y')
        run_cpp(r'''
#include <cassert>
struct City{struct{int X,Y;}Body;};
struct {struct{bool enable_custom_rendering;int city_work_radius;}current_config;}state,*is=&state;
struct {bool is_zoomed_out;}bic,*p_bic_data=&bic;
int '''+body+r'''
int main(){for(int radius=2;radius<=5;++radius)for(int parity=0;parity<2;++parity){
 City city{{80+parity,60+parity}};state.current_config.city_work_radius=radius;
 state.current_config.enable_custom_rendering=true;bic.is_zoomed_out=false;
 int normal=get_city_screen_center_y(&city);bic.is_zoomed_out=true;int reduced=get_city_screen_center_y(&city);
 assert((normal-city.Body.Y)*32==(reduced-city.Body.Y)*16);
 assert((normal&1)==(city.Body.X&1)&&(reduced&1)==(city.Body.X&1));
 state.current_config.enable_custom_rendering=false;assert(get_city_screen_center_y(&city)==city.Body.Y+7);
 bic.is_zoomed_out=false;assert(get_city_screen_center_y(&city)==normal);
}}
''')

    def test_city_pan_is_ignored_but_centering_and_config_off_delegate(self):
        body=function((ROOT/'injected_code.c').read_text(),'patch_Main_Screen_Form_move_camera')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __ 0
constexpr int C3X_NAV_DISCARD=1;
struct custom_renderer_native_view{int camera_x=0,camera_y=0;};
struct PCX_Image{struct{void* Image=nullptr;}JGL;};
struct Map_Renderer:PCX_Image{void* spotlight_on_city=nullptr;};
struct{struct{Map_Renderer Renderer;}Map;}bic,*p_bic_data=&bic;
struct Main_Screen_Form{int Player_CivID=1;struct{char field_18E4[64];}animator;}screen;
struct State{struct{bool enable_custom_rendering=true;}current_config;
 bool custom_renderer_camera_exact=false,custom_renderer_async_enabled=false,custom_renderer_display_valid=false;
 long long custom_renderer_camera_ticket=17;void(*custom_renderer_camera_cancel)(long long)=nullptr;
 void(*custom_renderer_navigation)(int,void*,struct custom_renderer_native_view*,void*)=nullptr;
}state,*is=&state;
int calls=0,last_edx=0,cancels=0;
void cancel(long long){++cancels;}
void Main_Screen_Form_move_camera(Main_Screen_Form* value,int edx,int x,int y,int,bool){
 assert(value==&screen&&x==123&&y==456);++calls;last_edx=edx;}
struct custom_renderer_native_view custom_renderer_native_view(Map_Renderer*){return {};}
bool capture_custom_renderer_native_view(Map_Renderer*,int,struct custom_renderer_native_view*,bool){return false;}
void apply_custom_renderer_native_view(struct custom_renderer_native_view*){assert(false);}
void '''+body+r'''
int main(){state.custom_renderer_camera_cancel=cancel;bic.Map.Renderer.spotlight_on_city=&screen;
 patch_Main_Screen_Form_move_camera(&screen,91,123,456,1,false);assert(calls==0&&cancels==0);
 state.current_config.enable_custom_rendering=false;
 patch_Main_Screen_Form_move_camera(&screen,91,123,456,1,false);assert(calls==1&&last_edx==91&&cancels==0);
 state.current_config.enable_custom_rendering=true;
 patch_Main_Screen_Form_move_camera(&screen,91,123,456,0,true);assert(calls==2);
 state.custom_renderer_camera_exact=true;
 patch_Main_Screen_Form_move_camera(&screen,91,123,456,1,false);assert(calls==3);
 state.custom_renderer_camera_exact=false;bic.Map.Renderer.spotlight_on_city=nullptr;
 patch_Main_Screen_Form_move_camera(&screen,91,123,456,1,false);assert(calls==4);
}
''')

    def test_inverse_pick_uses_one_presented_sample_for_both_axes(self):
        body = function((ROOT/'injected_code.c').read_text(), 'custom_renderer_zoom_inverse_point')
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include <cmath>
#include <cstddef>
#include "Renderer/native/c3x_renderer_api.h"
bool enabled=true;int q=65536,queries=0,syncs=0;
int native(int op,void*,void*,void const*,void const*,unsigned){assert(op==C3X_NATIVE_ZOOM_PRESENTED);++queries;return q;}
struct State{c3x_renderer_native_image_fn custom_renderer_native_image=native;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;}state,*is=&state;
struct Bic{int ScreenWidth=2240,ScreenHeight=1260;}bic,*p_bic_data=&bic;
bool custom_renderer_zoom_enabled(){return enabled;}
void sync_custom_renderer_zoom_to_native(){++syncs;}
int custom_renderer_zoom_inverse_coordinate(int v,long long){return v;}
void '''+body+r'''
int main(){for(q=65536;q<=98304;q+=137)for(int px:{-120,0,517,1120,1720,2240})for(int py:{0,331,630,1100}){
 int x=int(std::round(1120+(px-1120)*q/65536.)),y=int(std::round(630+(py-630)*q/65536.));
 auto before=queries;custom_renderer_zoom_inverse_point(&x,&y);assert(queries==before+1&&std::abs(x-px)<=1&&std::abs(y-py)<=1);
 }
 enabled=false;int x=91,y=173,before=queries;custom_renderer_zoom_inverse_point(&x,&y);assert(x==91&&y==173&&queries==before);
 enabled=true;for(int value:{0,-1,65535,98305}){q=value;x=91;y=173;custom_renderer_zoom_inverse_point(&x,&y);assert(x==91&&y==173);}
}
''')


if __name__ == '__main__':
    unittest.main()
