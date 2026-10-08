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
    def test_zoom_reuses_the_prepared_capture_envelope(self):
        body=function((ROOT/'injected_code.c').read_text(),'advance_custom_renderer_zoom')
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <cstddef>
#include "Renderer/native/c3x_renderer_api.h"
#define ARRAY_LEN(a) (int(sizeof(a)/sizeof((a)[0])))
struct Main_Screen_Form{bool is_now_loading_game=false;int camera_x=0,camera_y=0;}screen;
struct{struct{struct{void* spotlight_on_city=nullptr;}Renderer;}Map;}bic,*p_bic_data=&bic;
struct State{struct{bool enable_custom_rendering=true,enable_custom_rendering_zoom=true;}current_config;
 int custom_renderer_zoom_target_width=128;unsigned custom_renderer_dirty_flags=0;
 bool custom_renderer_redraw_pending=false,custom_renderer_unit_representatives_dirty=false,combat_unit_display_override_active=false;
 c3x_renderer_native_image_fn custom_renderer_native_image=nullptr;}state,*is=&state;
int players=1,*p_player_bits=&players;bool accept=true;unsigned calls=0,target=0;
int native(int op,void*,void*,void const*,void const*,unsigned q){assert(op==C3X_NATIVE_ZOOM_TARGET);++calls;target=q;return accept?1:0;}
void sync_custom_renderer_zoom_to_native(){} void debug(char const*){} auto p_OutputDebugStringA=debug;
bool '''+body+r'''
int main(){int levels[]={64,80,96,112,128,160,192,224,256,320,384};
 for(int from=0;from<11;++from)for(int to=0;to<11;++to){state=State{};state.custom_renderer_native_image=native;
  state.custom_renderer_zoom_target_width=levels[from];unsigned before=calls;
  assert(advance_custom_renderer_zoom(&screen,to-from,false));
  assert(state.custom_renderer_zoom_target_width==levels[to]&&calls==before+(to!=from));
  bool capture=false;
  assert(state.custom_renderer_redraw_pending==capture&&state.custom_renderer_unit_representatives_dirty==capture);
  assert(state.custom_renderer_dirty_flags==(capture?C3X_RENDERER_DIRTY_SCENE:0));
  if(to!=from)assert(target==unsigned(levels[to]*65536/128));
 }
 state=State{};state.custom_renderer_native_image=native;accept=false;
 assert(advance_custom_renderer_zoom(&screen,-1,false));assert(state.custom_renderer_zoom_target_width==128&&!state.custom_renderer_dirty_flags);
 state.current_config.enable_custom_rendering=false;unsigned before=calls;
 assert(!advance_custom_renderer_zoom(&screen,-1,false)&&calls==before);
}
''')

    def test_bridge_target_and_helper_presented_range(self):
        owner=(ROOT/'Renderer/native/native_composition_owner.h').read_text()
        target=owner[owner.index('        if(op==C3X_NATIVE_ZOOM_TARGET){'):owner.index('        if(op==C3X_NATIVE_HUD_BEGIN||')]
        helper=(ROOT/'Renderer/native/helper_trial/scene_client.h').read_text()
        presented=helper[helper.index('    unsigned presented_zoom()const{'):helper.index('    SceneClient(SceneClient const&)=delete;')]
        run_cpp(r'''
#include <cassert>
#include <cstdint>
#include "Renderer/native/scene_projection.h"
#include "Renderer/native/gpu_image_commands.h"
using namespace c3x_gpu_images;
constexpr int C3X_NATIVE_ZOOM_TARGET=129;
using LONG=std::int32_t;
LONG InterlockedCompareExchange(volatile LONG* address,LONG,LONG){return *address;}
struct Client{unsigned queued=0,flushed=0,value=0;
 void submit(Command const* command,int count){assert(count==1&&command->kind==Kind::zoom_target);value=command->color;++queued;}
 void flush(){++flushed;}
}transport;
struct Owner {Client* client=&transport;int operation(int op,unsigned color){
''' + target + r'''
return 0;}};
struct Helper {struct Wire {volatile LONG presented_zoom_q16,presented_pan;} state{0,0};Wire* wire=&state;
''' + presented + r'''
};
int main(){Owner owner;Helper helper;
 for(unsigned scale:{32768u,40960u,49152u,57344u,65536u,81920u,98304u,104857u,114688u,131072u,163840u,196608u}){
  unsigned prior=transport.queued;assert(owner.operation(129,scale)==1);
  assert(transport.queued==prior+1&&transport.flushed==transport.queued&&transport.value==scale);
  helper.state.presented_zoom_q16=LONG(scale);assert(helper.presented_zoom()==scale);
 }
 for(unsigned bad:{0u,32767u,196609u,~0u}){
  unsigned prior=transport.queued;assert(owner.operation(129,bad)==-1&&transport.queued==prior);
  helper.state.presented_zoom_q16=LONG(bad);assert(helper.presented_zoom()==65536);
 }
 helper.wire=nullptr;assert(helper.presented_zoom()==65536);
}
''')

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
bool probe=true,active=false;int begins=0,ends=0,calls=0,unit_begins=0;
bool custom_renderer_native_probe_on(){return probe;}
int native(int op,void* canvas,void*,void const* from,void const*,unsigned){
 if(op==C3X_NATIVE_HUD_END){assert(active);active=false;++ends;return 1;}
 assert((op==C3X_NATIVE_HUD_BEGIN||op==C3X_NATIVE_UNIT_HUD_BEGIN)&&(canvas==&image||canvas==&map_image)&&!active);
 if(op==C3X_NATIVE_UNIT_HUD_BEGIN){assert(static_cast<int const*>(from)[2]==7);++unit_begins;}
 auto anchor=static_cast<int const*>(from);assert(anchor[0]==210&&anchor[1]==86);active=true;++begins;return 1;
}
void MapMessage_draw(MapMessage*,int edx,PCX_Image* canvas,int shade){assert(edx==91&&shade==7&&canvas);++calls;}
void custom_renderer_hud_layout_offset(int,int,int*x,int*y){*x=*y=0;}
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
 state.current_config.enable_custom_rendering_zoom=false;
 assert(custom_renderer_hud_scope(&image,210,86,1,7)==1);
 assert(custom_renderer_hud_scope(nullptr,0,0,0,-1)==1);
 assert(unit_begins==1&&begins==ends&&!active);
 assert(custom_renderer_hud_scope(&image,210,86,99,-1)==0);
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
#include <cstdio>
#define __ 0
constexpr int C3X_NAV_DISCARD=1,C3X_NAV_PENDING=5,C3X_RENDERER_RESULT_PENDING=4,C3X_NATIVE_ZOOM_PRESENTED=130;
struct custom_renderer_native_view{int camera_x=0,camera_y=0;};
struct LARGE_INTEGER{long long QuadPart;};bool QueryPerformanceCounter(LARGE_INTEGER* p){p->QuadPart=1;return true;}
void debug(char const*){}auto p_OutputDebugStringA=debug;
struct PCX_Image{struct{void* Image=nullptr;}JGL;};
struct Map_Renderer:PCX_Image{void* spotlight_on_city=nullptr;};
struct{struct{Map_Renderer Renderer;}Map;}bic,*p_bic_data=&bic;
struct Main_Screen_Form{int Player_CivID=1,camera_x=0,camera_y=0;struct{char field_18E4[64];}animator;}screen;
struct State{struct{bool enable_custom_rendering=true;}current_config;
 bool custom_renderer_camera_exact=false,custom_renderer_async_enabled=false,custom_renderer_display_valid=false,custom_renderer_unit_representatives_dirty=false;
 long long custom_renderer_camera_ticket=17;void(*custom_renderer_camera_cancel)(long long)=nullptr;
 bool custom_renderer_scroll_request=false,custom_renderer_trace_input=false;
 int(*custom_renderer_navigation)(int,void*,struct custom_renderer_native_view*,void*)=nullptr;
 int(*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=nullptr;
 int custom_renderer_capture_cover=0,custom_renderer_zoom_target_width=128;
}state,*is=&state;
bool custom_renderer_zoom_enabled(){return false;}
int custom_renderer_capture_cover_width(bool){return 112;}
int calls=0,last_edx=0,cancels=0;
void cancel(long long){++cancels;}
void Main_Screen_Form_move_camera(Main_Screen_Form* value,int edx,int x,int y,int,bool){
 assert(value==&screen&&x==123&&y==456);++calls;last_edx=edx;}
void move_custom_renderer_camera(Main_Screen_Form* p,int e,int x,int y,int r,bool b){Main_Screen_Form_move_camera(p,e,x,y,r,b);}
struct custom_renderer_native_view custom_renderer_native_view(Map_Renderer*){return {};}
bool custom_renderer_same_projection(struct custom_renderer_native_view const*,struct custom_renderer_native_view const*){return true;}
bool capture_custom_renderer_native_view(Map_Renderer*,int,struct custom_renderer_native_view*,bool){return false;}
void log_custom_renderer_test_route_resolved(int,int){}
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
bool enabled=true;int q=65536,queries=0,syncs=0,pan=0;
int native(int op,void*,void*,void const*,void const*,unsigned){
 if(op==C3X_NATIVE_PAN_PRESENTED)return pan;assert(op==C3X_NATIVE_ZOOM_PRESENTED);++queries;return q;}
struct State{struct{bool enable_custom_rendering=true;}current_config;c3x_renderer_native_image_fn custom_renderer_native_image=native;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;}state,*is=&state;
struct Bic{int ScreenWidth=2240,ScreenHeight=1260;}bic,*p_bic_data=&bic;
bool custom_renderer_zoom_enabled(){return enabled;}
void sync_custom_renderer_zoom_to_native(){++syncs;}
int custom_renderer_zoom_inverse_coordinate(int v,long long){return v;}
void '''+body+r'''
int main(){for(q=32768;q<=196608;q+=137)for(int px:{-120,0,517,1120,1720,2240})for(int py:{0,331,630,1100}){
 int x=int(std::round(1120+(px-1120)*q/65536.)),y=int(std::round(630+(py-630)*q/65536.));
 auto before=queries;custom_renderer_zoom_inverse_point(&x,&y);assert(queries==before+1&&std::abs(x-px)<=1&&std::abs(y-py)<=1);
 }
 enabled=false;int x=91,y=173,before=queries;custom_renderer_zoom_inverse_point(&x,&y);assert(x==91&&y==173&&queries==before);
 enabled=true;for(int value:{0,-1,32767,196609}){q=value;x=91;y=173;custom_renderer_zoom_inverse_point(&x,&y);assert(x==91&&y==173);}
 // A sliding camera step shifts the shown world; picks undo it first.
 q=65536;pan=int((unsigned(-37)&0xffffu)|(unsigned(52)<<16));x=500;y=400;custom_renderer_zoom_inverse_point(&x,&y);
 assert(x==537&&y==348);pan=0;
}
''')


if __name__ == '__main__':
    unittest.main()
