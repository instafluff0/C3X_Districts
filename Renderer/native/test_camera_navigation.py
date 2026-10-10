"""Execute the production zoom bounds, minimap scope and Civ III edge scrolling."""
from pathlib import Path
import re
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_zoom_integration import function

ROOT = Path(__file__).resolve().parents[2]


class CameraNavigationTests(unittest.TestCase):
    def test_combat_zoom_poll_consumes_only_zoom(self):
        source = (ROOT / 'injected_code.c').read_text()
        body = 'bool ' + function(source, 'custom_renderer_minimap_zoom_due') + '\nvoid ' + function(source, 'poll_custom_renderer_combat_zoom')
        body = body.replace('(void *)(*p_GetProcAddress)', '(Peek)(*p_GetProcAddress)')
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#include <vector>
using HWND=void*;using UINT=unsigned;using BOOL=int;
#define WINAPI
constexpr int WM_MOUSEWHEEL=522,WM_KEYDOWN=256,PM_REMOVE=1,PM_NOREMOVE=0;
constexpr int VK_Z=90,VK_CONTROL=17,VK_MENU=18,C3X_NATIVE_ZOOM_PRESENTED=130;
struct MSG {HWND hwnd;UINT message,wParam;};using LPMSG=MSG*;
using Peek=BOOL(*)(LPMSG,HWND,UINT,UINT,UINT);
struct Base_Form;void redraw(Base_Form*);
struct VTable {void(*m73_call_m22_Draw)(Base_Form*)=redraw;} vtable;
struct Base_Form {VTable* vtable=&::vtable;};
struct Main_Screen_Form {struct {Base_Form Base;} GUI;} screen,*p_main_screen_form=&screen;
int scale=131072,samples=0,draws=0,targets=0,steps=0,polls=0;bool zoom=true,ctrl=false,alt=false;
void redraw(Base_Form* p){assert(p==(Base_Form*)&screen.GUI);++draws;}
int image(int op,void*,void*,void const*,void const*,unsigned){assert(op==130);++samples;return scale;}
struct State {struct {bool enable_custom_rendering=true;}current_config;
 bool combat_unit_display_override_active=true,custom_renderer_modal=false,paused_for_popup=false;
 int custom_renderer_zoom_wheel_remainder=0,custom_renderer_minimap_zoom=65536,custom_renderer_zoom_target_width=0;
 int(*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=image;
 void* user32=nullptr;
}state,*is=&state;
#define p_native_modal_depth (&modal)
int modal=0;HWND focus=&screen;
HWND GetFocus(){return focus;}
int key_state(int key){return (key==VK_CONTROL?ctrl:alt)?0x8000:0;}auto p_GetAsyncKeyState=key_state;
bool custom_renderer_zoom_enabled(){return zoom;}int int_abs(int x){return x<0?-x:x;}
bool advance_custom_renderer_zoom(Main_Screen_Form* p,int delta,bool wrap){assert(p==&screen);++targets;steps+=delta;return true;}
std::vector<MSG> queue;
BOOL peek(LPMSG out,HWND hwnd,UINT lo,UINT hi,UINT flags){
 ++polls;for(auto it=queue.begin();it!=queue.end();++it)if(it->hwnd==hwnd&&it->message>=lo&&it->message<=hi){
  *out=*it;if(flags==PM_REMOVE)queue.erase(it);return true;
 }return false;
}
Peek proc(void*,char const*){return peek;}auto p_GetProcAddress=proc;
void wheel(int delta){queue.push_back({&screen,WM_MOUSEWHEEL,unsigned(delta)<<16});}
'''+body+r'''
int main(){
 for(int gate=0;gate<7;++gate){
  state.current_config.enable_custom_rendering=gate!=0;state.combat_unit_display_override_active=gate!=1;
  zoom=gate!=2;state.custom_renderer_modal=gate==3;state.paused_for_popup=gate==4;modal=gate==5;
  focus=gate==6?nullptr:&screen;poll_custom_renderer_combat_zoom();assert(!polls&&!targets&&!draws);
 }
 focus=&screen;wheel(40);wheel(40);wheel(40);wheel(-240);
 queue.push_back({&state,WM_MOUSEWHEEL,120u<<16}); // Another window's input stays native.
 queue.push_back({&screen,513,0}); // Click stays native.
 queue.push_back({&screen,WM_KEYDOWN,88});queue.push_back({&screen,WM_KEYDOWN,VK_Z});
 poll_custom_renderer_combat_zoom();assert(targets==2&&steps==-1&&queue.size()==4&&!state.custom_renderer_zoom_wheel_remainder);
 assert(draws==1&&samples==1&&state.custom_renderer_minimap_zoom==131072);
 queue.erase(queue.begin()+2); // Native processing owns the preceding non-Z key.
 ctrl=true;poll_custom_renderer_combat_zoom();assert(targets==2&&queue.size()==3);
 ctrl=false;alt=true;poll_custom_renderer_combat_zoom();assert(targets==2&&queue.size()==3);
 alt=false;poll_custom_renderer_combat_zoom();assert(targets==3&&steps==-2&&queue.size()==2);
 assert(draws==1);scale=196608;poll_custom_renderer_combat_zoom();assert(draws==2&&state.custom_renderer_minimap_zoom==196608);
 for(int n=0;n<40;++n)wheel(120);
 auto before=targets;poll_custom_renderer_combat_zoom();assert(targets==before+32&&queue.size()==10);
 poll_custom_renderer_combat_zoom();assert(targets==before+40&&queue.size()==2);
}
''')

    def test_display_geometry_and_native_fallback(self):
        source = (ROOT / 'injected_code.c').read_text()
        bodies = '\n'.join(result + ' ' + function(source, name) for result, name in [
            ('RECT', 'custom_renderer_visible_map_rect'),
            ('void', 'patch_Navigator_Data_draw_viewport'),
            ('void', 'move_custom_renderer_camera'),
            ('void', 'patch_Main_Screen_Form_scroll_at_mouse')])
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <initializer_list>
using LONG=long;
struct RECT {LONG left,top,right,bottom;};
struct Map_Renderer {int field_3E98[3]={0,0,0},field_3EA4=2240,field_3EA8=1260;};
struct Map {int Width=100,Height=80,Flags=0;Map_Renderer Renderer;};
struct Bic {Map Map;bool is_zoomed_out=false;int ScreenWidth=2240,ScreenHeight=1260;}bic,*p_bic_data=&bic;
struct Main_Screen_Form {int camera_x=900,camera_y=300,TileX_Min=7,TileX_Max=37,TileY_Min=4,TileY_Max=24;
 struct {int field_18E4[20]={};}animator;}screen,*p_main_screen_form=&screen;
struct Navigator_Data {RECT Rect={10,20,410,220};int Mini_Map_Width2=400,Mini_Map_Height2=200;} nav;
struct State {struct{bool enable_custom_rendering=true;}current_config;
 int (*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned);
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
 bool custom_renderer_trace_input=false,combat_unit_display_override_active=false,custom_renderer_scroll_request=false;
 unsigned custom_renderer_view_timer=1;
}state,*is=&state;
void debug(char const*){}auto p_OutputDebugStringA=debug;
constexpr int C3X_NATIVE_ZOOM_PRESENTED=130;int scale=65536,samples=0;
bool zoom=true,city=false;
bool custom_renderer_zoom_enabled(){return state.current_config.enable_custom_rendering&&zoom&&!city;}
void sync_custom_renderer_zoom_to_native(){}
int custom_renderer_zoom_inverse_coordinate(int p,long long){
 int center=0; // Separate below: normal native basis in this fixture.
 return p;
}
int sample(int op,void*,void*,void const*,void const*,unsigned){assert(op==130);++samples;return scale;}
int native_calls=0,scroll_calls=0;RECT observed{};int marker=0;RECT observed_nav{};int observed_width=0,observed_height=0;
void Navigator_Data_draw_viewport(Navigator_Data* n,int edx){assert(n==&nav&&edx==91);++native_calls;
 observed={screen.TileX_Min,screen.TileY_Min,screen.TileX_Max,screen.TileY_Max};
 observed_nav=n->Rect;observed_width=n->Mini_Map_Width2;observed_height=n->Mini_Map_Height2;}
bool scroll_tagged=false;
void Main_Screen_Form_scroll_at_mouse(Main_Screen_Form* p,int e){assert(p==&screen&&e==91);scroll_tagged=state.custom_renderer_scroll_request;++scroll_calls;}
void Main_Screen_Form_move_camera(Main_Screen_Form* p,int e,int x,int y,int reason,bool update){
 assert(p==&screen&&e==91);++native_calls;auto& m=bic.Map;auto&r=m.Renderer;
 int hw=bic.is_zoomed_out?32:64,hh=hw/2,ox=p->camera_x,oy=p->camera_y;
 p->camera_x=(m.Flags&1)?(x%(m.Width*hw)+m.Width*hw)%(m.Width*hw):std::clamp(x,hw,std::max(hw,m.Width*hw-r.field_3EA4+r.field_3E98[1]));
 int lo=m.Flags&4?0:hh,hi=(m.Height+(m.Flags&4?1:0))*hh-r.field_3EA8+r.field_3E98[2];
 p->camera_y=(m.Flags&2)?(y%(m.Height*hh)+m.Height*hh)%(m.Height*hh):std::clamp(y,lo,std::max(lo,hi));
 if(update||ox!=p->camera_x||oy!=p->camera_y){
 p->TileX_Min=(p->camera_x+r.field_3E98[1])/hw-1;p->TileX_Max=(p->camera_x+r.field_3EA4)/hw+1;
 p->TileY_Min=(p->camera_y+r.field_3E98[2])/hh-1;p->TileY_Max=(p->camera_y+r.field_3EA8)/hh+1;
 *(bool*)(p->animator.field_18E4+10)=true;}
}
''' + bodies + r'''
int main(){state.custom_renderer_native_image=sample;
 for(int q:{32768,40960,49152,57344,65536,81920,98304,114688,131072,163840,196608}){
  scale=q;int before=samples;RECT r=custom_renderer_visible_map_rect();assert(samples==before+1);
  assert(std::abs((r.right-r.left)-2240.*65536/q)<=1.1);
  assert(std::abs((r.bottom-r.top)-1260.*65536/q)<=1.1);
  for(int flags:{0,1,2,3,4,5}){
   bic.Map.Flags=flags;
   for(int target:{-10000,10000}){
    move_custom_renderer_camera(&screen,91,target,target,1,false);
    if(!(flags&1))assert(screen.camera_x==(target<0?64-r.left:6400-r.right));
    if(!(flags&2))assert(screen.camera_y==(target<0?(flags&4?0:32)-r.top:2560+(flags&4?32:0)-r.bottom));
    assert(screen.TileX_Min==screen.camera_x/64-1&&screen.TileY_Min==screen.camera_y/32-1);
    auto x=screen.camera_x,y=screen.camera_y;*(bool*)(screen.animator.field_18E4+10)=false;
    move_custom_renderer_camera(&screen,91,x,y,1,false);
    assert(screen.camera_x==x&&screen.camera_y==y&&!*(bool*)(screen.animator.field_18E4+10));
   }
  }
  screen.camera_x=1700;screen.camera_y=500;
  RECT saved{screen.TileX_Min,screen.TileY_Min,screen.TileX_Max,screen.TileY_Max};
  patch_Navigator_Data_draw_viewport(&nav,91);
  assert(observed.left==(1700+r.left)/64&&observed.right==(1700+r.right)/64);
  assert(observed.top==(500+r.top)/32&&observed.bottom==(500+r.bottom)/32);
  assert(screen.TileX_Min==saved.left&&screen.TileX_Max==saved.right&&screen.TileY_Min==saved.top&&screen.TileY_Max==saved.bottom);
 }
 scale=65536;bic.Map.Width=20;bic.Map.Height=20;int previous_calls=native_calls;
 patch_Navigator_Data_draw_viewport(&nav,91);assert(native_calls==previous_calls);
 bic.Map.Height=80;patch_Navigator_Data_draw_viewport(&nav,91);
 assert(native_calls==previous_calls+1&&observed.left==0&&observed.right==19);
 assert(observed_nav.left==13&&observed_nav.top==20&&observed_width==394&&observed_height==200);
 assert(nav.Rect.left==10&&nav.Rect.top==20&&nav.Mini_Map_Width2==400&&nav.Mini_Map_Height2==200);
 state.current_config.enable_custom_rendering=false;int before=samples;
 patch_Navigator_Data_draw_viewport(&nav,91);assert(samples==before&&observed.left==screen.TileX_Min);
 // Civ III chooses edge-scroll steps. With the renderer on, its request is
 // tagged as a scroll so an in-flight step finishes before the next.
 patch_Main_Screen_Form_scroll_at_mouse(&screen,91);assert(scroll_calls==1&&!scroll_tagged);
 state.current_config.enable_custom_rendering=true;patch_Main_Screen_Form_scroll_at_mouse(&screen,91);
 assert(scroll_calls==2&&scroll_tagged&&!state.custom_renderer_scroll_request);
 state.custom_renderer_view_timer=0;patch_Main_Screen_Form_scroll_at_mouse(&screen,91);assert(scroll_calls==3);
 // The custom combat display keeps its battlefield camera fixed.
 state.combat_unit_display_override_active=true;patch_Main_Screen_Form_scroll_at_mouse(&screen,91);assert(scroll_calls==3);
 state.combat_unit_display_override_active=false;
 for(int bad:{0,-1,32767,196609}){scale=bad;auto r=custom_renderer_visible_map_rect();assert(r.left==0&&r.right==2240);}
 zoom=false;scale=196608;auto r=custom_renderer_visible_map_rect();assert(r.top==0&&r.bottom==1260);
 zoom=true;city=true;r=custom_renderer_visible_map_rect();assert(r.left==0&&r.right==2240);
}
''')

    def test_zoomed_visibility_preserves_native_centering_decisions(self):
        source = (ROOT / 'injected_code.c').read_text()
        native = (ROOT / 'ref/Civ3Conquests_master.exe.c').read_text()
        start = native.index('Main_Screen_Form::is_tile_on_screen(Main_Screen_Form *this')
        native = native[start:native.index('\n}\n', start) + 3]
        native = native.replace('Main_Screen_Form::is_tile_on_screen', 'Main_Screen_Form_is_tile_on_screen')
        native = native.replace('Main_Screen_Form *this,int x', 'Main_Screen_Form *this,int edx,int x')
        native = native.replace('this', 'self').replace('Map::wrap_horiz', 'wrap_horiz').replace('Map::wrap_vert', 'wrap_vert')
        patch = function(source, 'patch_Main_Screen_Form_is_tile_on_screen')
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include <cstddef>
using uint=unsigned;
struct RECT{int left,top,right,bottom;};
struct Map_Renderer{int field_3E98[3]={0,0,0},field_3EA4=2240,field_3EA8=1260;};
struct Map{int Width=100,Height=80,Flags=0;Map_Renderer Renderer;};
struct Bic{Map Map;bool is_zoomed_out=false;int ScreenWidth=2240,ScreenHeight=1260;}bic_data,*p_bic_data=&bic_data;
struct Main_Screen_Form{int camera_x=640,camera_y=320,TileX_Min=9,TileX_Max=46,TileY_Min=9,TileY_Max=50;}screen;
struct{struct{bool enable_custom_rendering=true;}current_config;void* custom_renderer_hud_canvas=nullptr;int custom_renderer_capture_cover=64;}state,*is=&state;
bool zoom=true;bool custom_renderer_zoom_enabled(){return zoom;}
RECT visible{0,0,2240,1260};RECT custom_renderer_visible_map_rect(){return visible;}
int wrap_horiz(Map*m,int x){return !(m->Flags&1)?x:x<0?x+m->Width:x>=m->Width?x-m->Width:x;}
int wrap_vert(Map*m,int y){return !(m->Flags&2)?y:y<0?y+m->Height:y>=m->Height?y-m->Height:y;}
bool ''' + native + '\nbool ' + patch + r'''
int main(){
 // At 1x every native visibility decision and margin is unchanged.
 for(int flags:{0,1,2,3}){bic_data.Map.Flags=flags;
  for(int x=0;x<100;++x)for(int y=0;y<80;++y)for(int margin:{0,1,4}){
   bool expected=Main_Screen_Form_is_tile_on_screen(&screen,91,x,y,margin,margin+2);
   assert(patch_Main_Screen_Form_is_tile_on_screen(&screen,91,x,y,margin,margin+2)==expected);
   assert(screen.TileX_Min==9&&screen.TileX_Max==46&&screen.TileY_Min==9&&screen.TileY_Max==50);
  }
 }
 // Off-screen city labels are prepared for outward zoom without changing
 // ordinary unit visibility, native traversal bounds, or centering rules.
 bic_data.Map.Flags=0;visible={0,0,2240,1260};
 assert(!patch_Main_Screen_Form_is_tile_on_screen(&screen,91,58,30,0,0));
 state.custom_renderer_hud_canvas=&screen;
 assert(patch_Main_Screen_Form_is_tile_on_screen(&screen,91,58,30,0,0));
 assert(screen.TileX_Min==9&&screen.TileX_Max==46&&screen.TileY_Min==9&&screen.TileY_Max==50);
 // Labels follow the captured envelope: a 1x-only capture keeps them in view.
 state.custom_renderer_capture_cover=128;assert(!patch_Main_Screen_Form_is_tile_on_screen(&screen,91,58,30,0,0));
 state.custom_renderer_capture_cover=64;
 state.custom_renderer_hud_canvas=nullptr;
 // A unit inside the canonical capture but cropped off by zoom must recenter.
 visible={747,420,1493,840};bic_data.Map.Flags=0;
 assert(Main_Screen_Form_is_tile_on_screen(&screen,91,16,24,4,6));
 assert(!patch_Main_Screen_Form_is_tile_on_screen(&screen,91,16,24,4,6));
 assert(patch_Main_Screen_Form_is_tile_on_screen(&screen,91,28,28,4,6));
 assert(!patch_Main_Screen_Form_is_tile_on_screen(&screen,91,28,29,0,0)); // parity
 // Wrapped visible copies still use the native seam logic.
 bic_data.Map.Flags=3;screen.camera_x=5400;screen.camera_y=1900;
 assert(patch_Main_Screen_Form_is_tile_on_screen(&screen,91,2,78,0,0));
 assert(!patch_Main_Screen_Form_is_tile_on_screen(&screen,91,60,40,0,0));
 // Wider than one world period: compare every tile against explicit copies,
 // including a narrow wrapped axis beside a fully covered axis.
 bic_data.Map.Width=60;bic_data.Map.Height=60;
 for(int flags:{0,1,2,3})for(int cx:{-1800,640,5400})for(int cy:{-1400,320,2900})
 for(auto area:{RECT{-1120,-630,3360,1890},RECT{-1120,420,3360,840}}){
  bic_data.Map.Flags=flags;screen.camera_x=cx;screen.camera_y=cy;visible=area;
  for(int x=0;x<60;++x)for(int y=0;y<60;++y)for(int margin:{0,1,4}){
   int mx=margin*(area.right-area.left)/2240,my=margin*(area.bottom-area.top)/1260;
   int lo_x=(cx+area.left)/64-1+mx,hi_x=(cx+area.right)/64+1-mx;
   int lo_y=(cy+area.top)/32-1+my,hi_y=(cy+area.bottom)/32+1-my;
   bool inside=false;
   for(int ix=-4;ix<=4;++ix)for(int iy=-4;iy<=4;++iy){
    if((!(flags&1)&&ix)||(!(flags&2)&&iy))continue;
    inside|=x+ix*60>lo_x&&x+ix*60<hi_x&&y+iy*60>lo_y&&y+iy*60<hi_y;
   }
   assert(patch_Main_Screen_Form_is_tile_on_screen(&screen,91,x,y,margin,margin)==(inside&&!((x^y)&1)));
  }
 }
 state.current_config.enable_custom_rendering=false;
 assert(patch_Main_Screen_Form_is_tile_on_screen(&screen,91,16,24,4,6)==Main_Screen_Form_is_tile_on_screen(&screen,91,16,24,4,6));
 state.current_config.enable_custom_rendering=true;zoom=false;
 assert(patch_Main_Screen_Form_is_tile_on_screen(&screen,91,16,24,4,6)==Main_Screen_Form_is_tile_on_screen(&screen,91,16,24,4,6));
}
''')

    def test_view_timer_follows_zoom_and_never_scrolls(self):
        # Edge scrolling is Civ III's own (scroll_at_mouse). The renderer's
        # 16 ms timer only redraws the minimap box for the presented zoom and
        # re-clamps the camera after a zoom-out at an expanded map edge.
        source = (ROOT / 'injected_code.c').read_text()
        body = function(source, 'custom_renderer_view_timer')
        due = 'bool ' + function(source, 'custom_renderer_minimap_zoom_due')
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <initializer_list>
using HWND=int;using UINT=unsigned;using UINT_PTR=unsigned;using DWORD=unsigned;
constexpr int __=0,C3X_NATIVE_ZOOM_PRESENTED=130;
struct Animator{int Units2_Count=0,field_18E4[20]={};};
struct Base_Form;void gui_draw(Base_Form*);struct FormVtable{void(*m73_call_m22_Draw)(Base_Form*)=gui_draw;}vtable;
struct Base_Form{FormVtable* vtable=&::vtable;};
struct Main_Screen_Form{struct{Base_Form Base;bool is_enabled=true;}GUI;bool is_now_loading_game=false,turn_end_flag=true;Animator animator;
 int camera_x=500,camera_y=300;}screen,*p_main_screen_form=&screen;
struct{struct{struct{void* spotlight_on_city=nullptr;}Renderer;}Map;}bic,*p_bic_data=&bic;
struct{struct{struct{int Status2=0;}Data;}Base;}city,*p_city_form=&city;
int players=1,*p_player_bits=&players,scale=65536,moves=0,draws=0,gui_draws=0,last_x=0,last_y=0;
int modal_depth=0,inhibited=0,ending=0;
#define p_native_modal_depth (&modal_depth)
#define p_native_timer_inhibited (&inhibited)
#define p_native_game_ending (&ending)
struct State{struct{bool enable_custom_rendering=true;}current_config;
 bool combat_unit_display_override_active=false;
 unsigned custom_renderer_view_timer=17;bool custom_renderer_view_timer_running=false;
 bool custom_renderer_modal=false,paused_for_popup=false,custom_renderer_draw_in_progress=false;
 int custom_renderer_minimap_zoom=65536,custom_renderer_zoom_target_width=0;
 int (*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=nullptr;
}state,*is=&state;
int int_abs(int x){return x<0?-x:x;}
bool custom_renderer_zoom_enabled(){return true;}
int query(int op,void*,void*,void const*,void const*,unsigned){assert(op==C3X_NATIVE_ZOOM_PRESENTED);return scale;}
void patch_Main_Screen_Form_move_camera(Main_Screen_Form*p,int,int x,int y,int r,bool b){
 assert(p==&screen&&r==1&&!b);last_x=x;last_y=y;++moves;}
void patch_Animator_update_display(Animator*,int){++draws;}
void gui_draw(Base_Form*){++gui_draws;}
''' + due + r'''
void ''' + body + r'''
void tick(){custom_renderer_view_timer(0,0,17,0);assert(!state.custom_renderer_view_timer_running);}
int main(){state.custom_renderer_native_image=query;
 // Steady zoom: no camera movement, wherever the cursor is.
 for(int n=0;n<100;++n)tick();
 assert(moves==0&&draws==0&&gui_draws==0);
 // Zoom in redraws the minimap box only.
 scale=131072;tick();assert(moves==0&&gui_draws==1&&state.custom_renderer_minimap_zoom==131072);
 tick();assert(gui_draws==1);
 // Zoom out also re-clamps at the current camera, once.
 scale=65536;tick();assert(moves==1&&last_x==500&&last_y==300&&draws==1&&gui_draws==2);
 tick();assert(moves==1);
 // Blocked states change nothing.
 for(int k=0;k<13;++k){
  state.custom_renderer_modal=k==0;state.paused_for_popup=k==1;screen.is_now_loading_game=k==2;screen.turn_end_flag=k!=3;
  city.Base.Data.Status2=k==4;screen.animator.Units2_Count=k==5;state.combat_unit_display_override_active=k==6;
  screen.GUI.is_enabled=k!=7;*(bool*)(screen.animator.field_18E4+0xD)=k==8;state.custom_renderer_draw_in_progress=k==9;
  modal_depth=k==10;inhibited=k==11;ending=k==12;
  scale=k%2?32768:131072;int before=moves,drawn=gui_draws;tick();assert(moves==before&&gui_draws==drawn);
 }
 state.current_config.enable_custom_rendering=false;screen.turn_end_flag=true;scale=32768;
 int before=moves;tick();assert(moves==before);
 // An animating zoom: the presented scale changes on every frame. Redrawing
 // the native GUI (and re-clamping) for each one queued Civ III interface work
 // ahead of the zoom's frames (review, 44). The box follows within 1% of the
 // target, then once more on arrival.
 state.current_config.enable_custom_rendering=true;
 state.custom_renderer_modal=state.paused_for_popup=screen.is_now_loading_game=false;city.Base.Data.Status2=0;
 screen.animator.Units2_Count=0;state.combat_unit_display_override_active=false;screen.GUI.is_enabled=true;
 *(bool*)(screen.animator.field_18E4+0xD)=false;state.custom_renderer_draw_in_progress=false;modal_depth=inhibited=ending=0;
 scale=65536;state.custom_renderer_minimap_zoom=65536;state.custom_renderer_zoom_target_width=256;
 int drawn=gui_draws,moved=moves;
 for(int s:{70000,90000,110000,120000})scale=s,tick();
 assert(gui_draws==drawn);
 scale=130000;tick();assert(gui_draws==drawn+1);
 scale=130900;tick();assert(gui_draws==drawn+1);
 scale=131072;tick();tick();assert(gui_draws==drawn+2&&moves==moved);
 // Zooming out re-clamps the same way: near the target, then on arrival.
 state.custom_renderer_zoom_target_width=128;drawn=gui_draws;
 for(int s:{120000,100000,80000})scale=s,tick();
 assert(gui_draws==drawn&&moves==moved);
 scale=66000;tick();assert(gui_draws==drawn+1&&moves==moved+1);
 scale=65600;tick();assert(gui_draws==drawn+1&&moves==moved+1);
 scale=65536;tick();assert(gui_draws==drawn+2&&moves==moved+2);
}
''')

if __name__ == '__main__':
    unittest.main()
