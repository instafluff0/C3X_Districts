"""Validate the injected main-map zoom transform and its integration contracts."""
from Renderer.native.native_cpp_test import run_cpp
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
FP_ONE = 65536


def c_div(numerator: int, denominator: int) -> int:
    """C integer division, which truncates toward zero."""
    quotient = abs(numerator) // abs(denominator)
    return -quotient if (numerator < 0) != (denominator < 0) else quotient


def transform(value: int, width: int, native_width: int, translation: int) -> int:
    value_fp = c_div(value * width * FP_ONE, native_width) + translation
    return c_div(value_fp + FP_ONE // 2, FP_ONE) if value_fp >= 0 else c_div(value_fp - FP_ONE // 2, FP_ONE)


def inverse(value: int, width: int, native_width: int, translation: int) -> int:
    value_fp = c_div((value * FP_ONE - translation) * native_width, width)
    return c_div(value_fp + FP_ONE // 2, FP_ONE) if value_fp >= 0 else c_div(value_fp - FP_ONE // 2, FP_ONE)


class CustomZoomTests(unittest.TestCase):
    def test_transient_message_anchor_preserves_font_and_dirty_rect(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('RECT * __fastcall\npatch_MapMessage_compute_rect')
        wrapper=source[start:source.index('\n}\n',start)+3]
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#include <cstddef>
#define __fastcall
struct RECT {int left,top,right,bottom;};
struct MapMessage {RECT native,output;};
struct {struct {bool enable_custom_rendering=false;} current_config;} state,*is=&state;
int width=128,calls=0,transforms=0;
RECT* MapMessage_compute_rect(MapMessage* p){++calls;p->output=p->native;return &p->output;}
bool custom_renderer_zoom_transform_active(){return is->current_config.enable_custom_rendering&&width!=128;}
void custom_renderer_zoom_transform_point(int* x,int* y){++transforms;*x=*x*width/128-240;*y=*y*width/128+80;}
'''+wrapper.replace('this','self')+r'''
int main(){
 for(int zoom:{128,160,192,224,256,320,384})for(int text_width:{80,81,234})for(int lower:{0,32}){
  width=zoom;MapMessage p{{640-text_width/2-2,320+lower-18-2,640-text_width/2-2+text_width,320+lower-2},{}};
  for(bool enabled:{false,true}){is->current_config.enable_custom_rendering=enabled;
   for(int repeat=0;repeat<10;++repeat){auto before=calls;RECT* r=patch_MapMessage_compute_rect(&p);
    assert(r==&p.output&&calls==before+1&&r->right-r->left==text_width&&r->bottom-r->top==18);
    int x=r->left+text_width/2+2,y=r->bottom+2;
    assert(x==((enabled&&zoom!=128)?640*zoom/128-240:640));
    assert(y==((enabled&&zoom!=128)?(320+lower)*zoom/128+80:320+lower));
   }
  }
 }
 MapMessage hidden{{0,0,0,0},{}};auto before=transforms;
 assert(patch_MapMessage_compute_rect(&hidden)->right==0&&transforms==before);
}
''')

    def test_settler_boundary_callsite_projects_only_when_enabled(self):
        source=(ROOT/'injected_code.c').read_text()
        start=source.index('void __fastcall\npatch_OpenGLRenderer_draw_settler_boundary')
        wrapper=source[start:source.index('\n}\n',start)+3]
        run_cpp(r'''
#include <cassert>
#include <initializer_list>
#define __fastcall
struct OpenGLRenderer{};
struct {struct {bool enable_custom_rendering=false;} current_config;} state,*is=&state;
int width=128,tx=0,ty=0,calls=0,transforms=0,points[4];
void custom_renderer_zoom_transform_point(int* x,int* y){++transforms;*x=*x*width/128+tx;*y=*y*width/128+ty;}
void patch_OpenGLRenderer_draw_line(OpenGLRenderer*,int edx,int x1,int y1,int x2,int y2){
 assert(edx==73);++calls;points[0]=x1;points[1]=y1;points[2]=x2;points[3]=y2;
}
'''+wrapper.replace('this','self')+r'''
int main(){OpenGLRenderer context;
 for(int zoom:{128,160,192,224,256,320,384}){width=zoom;tx=-256;ty=64;
  is->current_config.enable_custom_rendering=false;transforms=0;
  patch_OpenGLRenderer_draw_settler_boundary(&context,73,256,128,512,256);
  assert(!transforms&&points[0]==256&&points[1]==128&&points[2]==512&&points[3]==256);
  is->current_config.enable_custom_rendering=true;
  patch_OpenGLRenderer_draw_settler_boundary(&context,73,256,128,512,256);
  assert(transforms==2&&points[0]==2*zoom-256&&points[1]==zoom+64&&points[2]==4*zoom-256&&points[3]==2*zoom+64);
 }
 assert(calls==14);
}
'''.replace('#include <cassert>','#include <cassert>\n#include <initializer_list>'))

    def test_repeated_zoom_keeps_exact_native_camera(self):
        source = (ROOT / "injected_code.c").read_text()
        function = source[source.index("bool\nadvance_custom_renderer_zoom ("):source.index("int __fastcall\npatch_Main_Screen_Form_handle_key_down")]
        run_cpp(r'''
#include <cassert>
#include <cstdio>
#include <array>
#define Main_Screen_Form_move_camera native_move
#define ARRAY_LEN(x) (sizeof(x)/sizeof((x)[0]))
constexpr int VK_Z=90,__=0,C3X_RENDERER_DIRTY_SCENE=2,C3X_RENDERER_DIRTY_ALL=255,C3X_NATIVE_ZOOM_TARGET=129;
struct Main_Screen_Form {bool is_now_loading_game=false;int camera_x=0,camera_y=0;};
struct State {struct {bool enable_custom_rendering=true,enable_custom_rendering_zoom=true;} current_config;
 bool combat_unit_display_override_active=false,custom_renderer_unit_representatives_dirty=false;
 int custom_renderer_zoom_target_width=128;
 int (*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=nullptr;
 int custom_renderer_zoom_tile_width=128,custom_renderer_dirty_flags=0;bool custom_renderer_redraw_pending=false;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;} state,*is=&state;
struct Bic {struct{struct{void* spotlight_on_city=nullptr;}Renderer;}Map;int ScreenWidth=2240,ScreenHeight=1260;} bic,*p_bic_data=&bic;
int players=1,*p_player_bits=&players,moves=0;
void sync_custom_renderer_zoom_to_native(){}
void debug(char const*){}auto p_OutputDebugStringA=debug;
int target(int op,void*,void*,void const*,void const*,unsigned q){assert(op==129&&q>=32768&&q<=196608);return 1;}
void native_move(Main_Screen_Form* screen,int,int x,int y,int reason,bool bounds){
 assert(x==screen->camera_x&&y==screen->camera_y&&reason==0&&bounds);++moves;
}
''' + function.replace('this', 'screen') + r'''
int main(){
 for(int x:{-129,0,3616,3679})for(int y:{-17,0,394,427}){
  Main_Screen_Form screen;state.custom_renderer_native_image=target;screen.camera_x=x;screen.camera_y=y;
  state.custom_renderer_zoom_tile_width=128;
  state.custom_renderer_zoom_translate_x_fp=state.custom_renderer_zoom_translate_y_fp=0;
  for(int n=0;n<220;++n){assert(advance_custom_renderer_zoom_from_key(&screen,0,VK_Z));
   assert(screen.camera_x==x&&screen.camera_y==y);
  }
  assert(state.custom_renderer_zoom_tile_width==128&&state.custom_renderer_zoom_target_width==128&&moves==0);
  assert(state.custom_renderer_zoom_translate_x_fp==0&&state.custom_renderer_zoom_translate_y_fp==0);
  state.current_config.enable_custom_rendering=false;auto before=moves;
  assert(!advance_custom_renderer_zoom_from_key(&screen,0,VK_Z)&&moves==before);
  state.current_config.enable_custom_rendering=true;
 }
}
''')

    def test_cursor_anchor_and_inverse_pick(self) -> None:
        for native_width in (64, 128):
            translation = 0
            old_width = native_width
            cursor = 517
            for width in (384, 320, 256, 224, 192, 160, 128):
                cursor_fp = cursor * FP_ONE
                translation = cursor_fp - c_div((cursor_fp - translation) * width, old_width)
                self.assertEqual(transform(inverse(cursor, width, native_width, translation), width, native_width, translation), cursor)
                for point in (-4096, -3, 0, 91, 2048, 8192):
                    projected = transform(point, width, native_width, translation)
                    self.assertLessEqual(abs(inverse(projected, width, native_width, translation) - point), 1)
                old_width = width

    def test_injected_contract_is_complete(self) -> None:
        source = (ROOT / "injected_code.c").read_text()
        header = (ROOT / "C3X.h").read_text()
        api = (ROOT / "Renderer/native/c3x_renderer_api.h").read_text()
        for marker in (
            "int levels[11] = {64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384}",
            "advance_custom_renderer_zoom_from_key",
            "patch_Main_Screen_Form_get_tile_coords_under_mouse",
            "patch_Sprite_draw_on_map",
            "prepare_custom_renderer_zoom_tiles",
            "C3X_RENDERER_TILE_PREFETCH",
            "frame.tile_width = custom_renderer_zoom_enabled ()",
            "draw.projection_scale_milli = is->custom_renderer_zoom_tile_width * 1000 / 128",
            "if (is->current_config.enable_custom_rendering && canvas != NULL && canvas == is->custom_renderer_unit_canvas) return 0",
            "patch_Main_Screen_Form_city_hud_coords",
            "patch_Unit_draw_map_status",
            "patch_Animator_draw_map_unit_cursor",
            "patch_Sprite_draw_map_unit_marker",
        ):
            self.assertTrue(marker in source, marker)
        self.assertIn("bool enable_custom_rendering_zoom", header)
        self.assertIn("projection_scale_milli", api)

    def test_native_input_and_hud_share_the_projection(self):
        source = (ROOT / "injected_code.c").read_text()
        functions = source[source.index("int\ncustom_renderer_zoom_transform_coordinate"):source.index("// Temporary, event-bounded diagnosis")]
        start=functions.index('RECT * __fastcall\npatch_MapMessage_compute_rect')
        functions=functions[:start]+functions[functions.index('\n}\n',start)+3:]
        functions = functions[:functions.index('// A separate traversal envelope:')] + functions[functions.index('void __fastcall\npatch_Main_Screen_Form_city_hud_coords'):]
        functions = functions.replace("this", "screen").replace("int * anchors = malloc (", "int * anchors = (int*)malloc (")
        program = r'''
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#define __fastcall
#define __cdecl
#define __stdcall
constexpr int __=0;
struct State {int custom_renderer_tile_count=0;c3x_renderer_tile_v1* custom_renderer_tiles=nullptr;
 bool custom_renderer_trace_input=false;
 bool custom_renderer_unit_bootstrap=false;
 struct {bool enable_custom_rendering=false;} current_config;
 c3x_renderer_native_image_fn custom_renderer_native_image=nullptr;
 void* custom_renderer_hud_canvas=nullptr;int custom_renderer_zoom_native_tile_width=128,custom_renderer_zoom_tile_width=128,custom_renderer_zoom_target_width=128;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
} state,*is=&state;
struct CityForm {struct {struct {int Status2=0;} Data;} Base;} city,*p_city_form=&city;
using JGL_Image=void;
struct Main_Screen_Form {int mouse_x=0,mouse_y=0;struct {struct {struct {struct {JGL_Image* Image=nullptr;} JGL;} Canvas;} Data;} Units_Control;struct {struct {struct {JGL_Image* Image=nullptr;} JGL;} Canvas;} Base_Data;} main_screen,*p_main_screen_form=&main_screen;
struct Unit{};struct Animator{int field_18E4[32]{};};struct PCX_Image{struct {void* Image=nullptr;} JGL;};struct PCX_Color_Table{};
int native_routes=0;
void Main_Screen_Form_update_in_go_to_mode(Main_Screen_Form*,int){++native_routes;}
void Main_Screen_Form_draw_route_cursor(int,int){}
void debug(char const*){}auto p_OutputDebugStringA=debug;
struct Sprite{int Width=95,Height=63;};
struct Bic{bool is_zoomed_out=false;int ScreenWidth=2240,ScreenHeight=1260;} bic,*p_bic_data=&bic;
int status_calls=0,cursor_calls=0,marker_calls=0,overlay_x=0,overlay_y=0;
int ring_calls=0,route_begins=0,route_ends=0,capable=1;std::vector<int> route_anchors;
int native_image(int op,void*,void*,void const* from,void const* to,unsigned count){
 if(op==C3X_NATIVE_ZOOM_PRESENTED)return 65536;
 if(op==C3X_NATIVE_TACTICAL_CAPABLE)return capable;
 if(op==C3X_NATIVE_TACTICAL_ROUTE_BEGIN){++route_begins;auto a=(int const*)to;route_anchors.assign(a,a+4*count);return 1;}
 if(op==C3X_NATIVE_TACTICAL_ROUTE_END){++route_ends;return 1;}
 assert(op==C3X_NATIVE_TACTICAL_RING);
 auto ring=static_cast<int const*>(from);assert(ring[2]==(bic.is_zoomed_out?64:128)&&ring[3]==1);
 overlay_x=ring[0];overlay_y=ring[1];++ring_calls;return 1;
}
int custom_renderer_hud_scope(void*,int,int,unsigned){return 0;}
void custom_renderer_hud_layout_offset(int,int,int*x,int*y){*x=*y=0;}
PCX_Image canvas;PCX_Color_Table palette;
void Unit_draw_status(Unit*,int,PCX_Image* c,int x,int y,bool stack){
 assert(c==&canvas&&stack);++status_calls;overlay_x=x;overlay_y=y;
}
void Animator_draw_unit_cursor(Animator*,int,int x,int y){++cursor_calls;overlay_x=x;overlay_y=y;}
int Sprite_draw_scaled_color(Sprite*,int,PCX_Image* c,int x,int y,int color,int sx,int sy,int divisor,PCX_Color_Table* p){
 assert(c==&canvas&&color==123&&sx==1&&sy==1&&divisor==(bic.is_zoomed_out?2:1)&&p==&palette);
 ++marker_calls;overlay_x=x;overlay_y=y;return 37;
}
bool enabled=true;
bool custom_renderer_zoom_enabled(){return enabled;}
void sync_custom_renderer_zoom_to_native(){}
int passed_x=0,passed_y=0,anchor_x=0,anchor_y=0;
int Main_Screen_Form_get_tile_coords_under_mouse(Main_Screen_Form*,int,int x,int y,int* tx,int* ty){
 passed_x=x;passed_y=y;*tx=x/64;*ty=y/32;return 0;
}
void Main_Screen_Form_tile_to_screen_coords(Main_Screen_Form*,int,int,int,int* x,int* y){*x=anchor_x;*y=anchor_y;}
''' + functions.replace('(unsigned)screen', '(unsigned)(uintptr_t)screen') + r'''
int main(){
 Main_Screen_Form screen;
 state.current_config.enable_custom_rendering=true;
 state.custom_renderer_native_image=native_image;
 for(int basis:{64,128})for(int zoom:{128,160,192,224,256,320,384})for(int camera:{-317,0,803}){
  bic.is_zoomed_out=basis==64;
  state.custom_renderer_zoom_native_tile_width=basis;state.custom_renderer_zoom_tile_width=zoom;
  state.custom_renderer_zoom_translate_x_fp=-517LL*65536*(zoom-basis)/basis;
  state.custom_renderer_zoom_translate_y_fp=-319LL*65536*(zoom-basis)/basis;
  int x=431,y=279,tx=0,ty=0;
  custom_renderer_zoom_transform_point(&x,&y);
  // Native down/hover/up/right and hold callback receive raw display pixels;
  // all use the same hooked picker, including stored hover coordinates.
  screen.mouse_x=x;screen.mouse_y=y;
  int pressed_x=-1,pressed_y=-1;
  for(int event=0;event<5;++event){
   assert(patch_Main_Screen_Form_get_tile_coords_under_mouse(&screen,0,screen.mouse_x,screen.mouse_y,&tx,&ty)==0);
   assert(passed_x==431&&passed_y==279);
   if(event==0){pressed_x=tx;pressed_y=ty;}
   assert(tx==pressed_x&&ty==pressed_y); // no spurious drag to another tile
   assert(screen.mouse_x==x&&screen.mouse_y==y); // native HUD hit tests keep raw pixels
  }
  patch_Main_Screen_Form_get_tile_coords_for_map_clip(&screen,0,x,y,&tx,&ty);
  assert(passed_x==x&&passed_y==y);
  city.Base.Data.Status2=1;
  patch_Main_Screen_Form_get_tile_coords_under_mouse(&screen,0,x,y,&tx,&ty);
  assert(passed_x==x&&passed_y==y);city.Base.Data.Status2=0;
  enabled=false;patch_Main_Screen_Form_get_tile_coords_under_mouse(&screen,0,x,y,&tx,&ty);
  assert(passed_x==x&&passed_y==y);enabled=true;
  anchor_x=423-camera;anchor_y=192+camera;
  int expected_x=anchor_x+basis/2,expected_y=anchor_y+basis*7/16;
  custom_renderer_zoom_transform_point(&expected_x,&expected_y);
  patch_Main_Screen_Form_city_hud_coords(&screen,0,0,0,&x,&y);
  assert(x+basis/2==expected_x&&y+basis*7/16+10==expected_y+10);
  Main_Screen_Form_tile_to_screen_coords(&screen,0,0,0,&x,&y);
  assert(x==anchor_x&&y==anchor_y); // all other native anchor callers unchanged
  Unit unit;Animator animator;Sprite sprite;
  int center_x=431-camera,center_y=279+camera,projected_x=center_x,projected_y=center_y;
  custom_renderer_zoom_transform_point(&projected_x,&projected_y);
  // Preserve the old translated-center result without translating tick_anim.
  int offset=basis/4;
  patch_Unit_draw_map_status(&unit,0,&canvas,center_x-offset,center_y-offset,true);
  assert(overlay_x==projected_x-offset&&overlay_y==projected_y-offset);
  patch_Animator_draw_map_unit_cursor(&animator,0,center_x,center_y);
  assert(overlay_x==projected_x&&overlay_y==projected_y);
  int hw=sprite.Width/(basis==64?4:2),hh=sprite.Height/(basis==64?4:2);
  assert(patch_Sprite_draw_map_unit_marker(&sprite,0,&canvas,center_x-hw,center_y-hh,123,1,1,basis==64?2:1,&palette)==37);
  assert(overlay_x==projected_x-hw&&overlay_y==projected_y-hh);
  enabled=false;
  patch_Main_Screen_Form_city_hud_coords(&screen,0,0,0,&x,&y);
  assert(x==anchor_x&&y==anchor_y);
  patch_Unit_draw_map_status(&unit,0,&canvas,center_x-offset,center_y-offset,true);
  assert(overlay_x==center_x-offset&&overlay_y==center_y-offset);
  patch_Animator_draw_map_unit_cursor(&animator,0,center_x,center_y);
  assert(overlay_x==center_x&&overlay_y==center_y);
  assert(patch_Sprite_draw_map_unit_marker(&sprite,0,&canvas,center_x-hw,center_y-hh,123,1,1,basis==64?2:1,&palette)==37);
  assert(overlay_x==center_x-hw&&overlay_y==center_y-hh);enabled=true;
 }
 assert(ring_calls==84);
 // Config-off and missing capability immediately delegate; captured anchor
 // transport cannot affect native pathfinding or input ordering.
 state.current_config.enable_custom_rendering=false;
 patch_Main_Screen_Form_update_in_go_to_mode(&screen,0);
 assert(native_routes==1&&route_begins==0);
 state.current_config.enable_custom_rendering=true;capable=0;
 patch_Main_Screen_Form_update_in_go_to_mode(&screen,0);
 assert(native_routes==2&&route_begins==0);capable=1;
 c3x_renderer_tile_v1 tiles[2]={};tiles[0].tile_flags=C3X_RENDERER_TILE_RENDER;
 tiles[0].anchor_x=928;tiles[0].anchor_y=940; // native folded row vs actual map row
 tiles[1].tile_flags=C3X_RENDERER_TILE_PREFETCH;
 state.custom_renderer_tiles=tiles;state.custom_renderer_tile_count=2;
 enabled=false;bic.is_zoomed_out=false;anchor_x=928;anchor_y=-980;
 patch_Main_Screen_Form_update_in_go_to_mode(&screen,0);
 assert(native_routes==3&&route_begins==1&&route_ends==1);
 assert((route_anchors==std::vector<int>{992,-948,992,972}));
 tiles[0].anchor_y=0;assert(route_anchors[3]==972); // copied before scratch release

 // A city on an outer polar row uses a captured occurrence rather than
 // the native half-world fold; repeated wrapped copies choose the nearest.
 c3x_renderer_tile_v1 city_tiles[3]={};
 for(auto& t:city_tiles){t.tile_flags=C3X_RENDERER_TILE_RENDER;t.tile_x=4;t.tile_y=6;}
 city_tiles[0].anchor_x=-2800;city_tiles[0].anchor_y=600;
 city_tiles[1].anchor_x=1200;city_tiles[1].anchor_y=600;
 city_tiles[2].anchor_x=1200;city_tiles[2].anchor_y=-1800;
 state.custom_renderer_tiles=city_tiles;state.custom_renderer_tile_count=3;
 state.custom_renderer_zoom_tile_width=128;
 state.custom_renderer_zoom_translate_x_fp=state.custom_renderer_zoom_translate_y_fp=0;
 enabled=true;
 for(int target:{64,80,96,112})for(int basis:{64,128}){
  state.custom_renderer_zoom_target_width=target;state.custom_renderer_zoom_native_tile_width=basis;
  bic.is_zoomed_out=basis==64;anchor_x=928;anchor_y=-980;
  int x=0,y=0;patch_Main_Screen_Form_city_hud_coords(&screen,0,4,6,&x,&y);
  assert(x+basis/2==1264&&y+basis*7/16==656);
 }

}
'''
        run_cpp("#include <initializer_list>\n" + program)
        for retired in ("custom_renderer_zoom_inverse_point (&param_1, &param_2)",
                        "custom_renderer_zoom_inverse_point (&local_x, &local_y)",
                        "custom_renderer_zoom_inverse_point (&native_x, &native_y)"):
            self.assertNotIn(retired, source)

    def test_projection_hooks_are_installed(self):
        import csv
        rows = {row[4].strip(): row for row in csv.reader((ROOT / "civ_prog_objects.csv").read_text().replace("\t", " ").splitlines(), skipinitialspace=True) if len(row) >= 6}
        for name in ("Main_Screen_Form_get_tile_coords_under_mouse",):
            self.assertEqual(rows[name][0].strip(), "inlead", name)
        clip_rows = [row for row in csv.reader((ROOT / "civ_prog_objects.csv").read_text().replace("\t", " ").splitlines(), skipinitialspace=True)
                     if len(row) >= 6 and row[4].strip() == "Main_Screen_Form_get_tile_coords_for_map_clip"]
        self.assertEqual([int(row[1].strip(), 0) for row in clip_rows], [0x4C32A8, 0x4C32C4, 0x4C4F44, 0x4C4F60])
        self.assertTrue(all(row[0].strip() == "repl call" for row in clip_rows))
        self.assertEqual(rows["Main_Screen_Form_tile_to_screen_coords"][0].strip(), "define")
        expected = {
            "Main_Screen_Form_city_hud_coords": [0x4E571E],
            "Unit_draw_map_status": [0x5CC41D, 0x5CC9EB],
            "Animator_draw_map_unit_cursor": [0x5CC2B1, 0x5CC7F8],
            "Sprite_draw_map_unit_marker": [0x5CC29D, 0x5CC7D8],
        }
        all_rows = list(csv.reader((ROOT / "civ_prog_objects.csv").read_text().replace("\t", " ").splitlines(), skipinitialspace=True))
        for name, addresses in expected.items():
            selected = [r for r in all_rows if len(r) >= 6 and r[4].strip() == name]
            self.assertEqual([int(r[1].strip(), 0) for r in selected], addresses)
            self.assertTrue(all(r[0].strip() == "repl call" and int(r[2], 0) == int(r[3], 0) == 0 for r in selected))
        for name, address in {"Unit_draw_status": 0x5BA750, "Animator_draw_unit_cursor": 0x4F03E0, "Sprite_draw_scaled_color": 0x5F84B0}.items():
            self.assertEqual((rows[name][0].strip(), int(rows[name][1], 0)), ("define", address))
        source=(ROOT / "injected_code.c").read_text()
        header=(ROOT / "C3X.h").read_text()
        for retired in ("custom_renderer_zoom_native_hud_context", "custom_renderer_zoom_unit_tick_"):
            self.assertNotIn(retired, source)
            self.assertNotIn(retired, header)


    def test_unit_capture_scope_preserves_native_offsets(self):
        source=(ROOT / "injected_code.c").read_text()
        start=source.index("void __fastcall\npatch_Unit_tick_anim")
        body=source[start:source.index("int __fastcall\npatch_Sprite_draw_unit_body_normal", start)].replace("this", "unit")
        run_cpp(r'''
#include <cassert>
#include <cstddef>
#define __fastcall
constexpr int __=0,IS_OK=1;
struct Unit{struct {int X=2,Y=4,ID=7,army_top_defender_id=-1;}Body;};struct PCX_Image{struct {void* Image=nullptr;} JGL;};
constexpr unsigned C3X_RENDERER_TILE_VISIBLE=1;constexpr int UTA_Army=1,C3X_RENDERER_UNIT_STATE_OBSERVE=1;
void notify_custom_renderer_unit_state(Unit*,int){}
struct Tile{};Tile tile;bool visible=true;Tile* tile_at(int,int){return &tile;}
unsigned capture_custom_renderer_visibility(Tile*,int,int,int){return visible?C3X_RENDERER_TILE_VISIBLE:0;}
bool Unit_has_ability(Unit*,int,int){return false;}
struct Screen {int Player_CivID=1;} screen,*p_main_screen_form=&screen;
Unit outer,inner;PCX_Image previous,canvas;
struct State {
 Unit* custom_renderer_unit_context=&outer;PCX_Image* custom_renderer_unit_canvas=&previous;
 struct {bool enable_custom_rendering=true;} current_config;
 void (*custom_renderer_unit_forget)(int)=nullptr;
 void* custom_renderer_unit_draw=&outer;int custom_renderer_init_state=IS_OK;
} state,*is=&state;
int calls=0;
void Unit_tick_anim(Unit* unit,int,PCX_Image* target,int x,int y,bool status){
 assert(unit==&inner&&target==&canvas&&x==-413&&y==291&&status);
 assert(state.custom_renderer_unit_context==(state.current_config.enable_custom_rendering?&inner:nullptr));
 assert(state.custom_renderer_unit_canvas==&canvas);++calls;
}
''' + body + r'''
int main(){
 patch_Unit_tick_anim(&inner,0,&canvas,-413,291,true);
 assert(calls==1&&state.custom_renderer_unit_context==&outer&&state.custom_renderer_unit_canvas==&previous);
 visible=false;patch_Unit_tick_anim(&inner,0,&canvas,-413,291,true);
 assert(calls==1&&state.custom_renderer_unit_context==&outer&&state.custom_renderer_unit_canvas==&previous);
 state.current_config.enable_custom_rendering=false;
 patch_Unit_tick_anim(&inner,0,&canvas,-413,291,true);
 assert(calls==2&&state.custom_renderer_unit_context==&outer&&state.custom_renderer_unit_canvas==&previous);
}
''')

    def test_actual_injected_cycle_and_close_zoom_survive_sync(self) -> None:
        source = (ROOT / "injected_code.c").read_text()
        key = "bool\nadvance_custom_renderer_zoom (" + source.split(
            "bool\nadvance_custom_renderer_zoom (", 1)[1].split(
            "\nint __fastcall\npatch_Main_Screen_Form_handle_key_down", 1)[0]
        # 'this' is a valid identifier in the injected C, not in C++.
        key = key.replace("this", "screen")
        sync = "void\nsync_custom_renderer_zoom_to_native" + source.split(
            "void\nsync_custom_renderer_zoom_to_native", 1)[1].split(
            "\nint\ncustom_renderer_zoom_transform_coordinate", 1)[0]
        program = r'''
#include <cstdint>
#include <cassert>
#include <cstdio>
#define ARRAY_LEN(a) (sizeof(a)/sizeof((a)[0]))
enum {VK_Z=90,C3X_RENDERER_DIRTY_SCENE=2,C3X_RENDERER_DIRTY_ALL=255,__=0,C3X_NATIVE_ZOOM_TARGET=129};
#define Main_Screen_Form_move_camera native_move
#define Main_Screen_Form_process_mouse_wheel native_wheel
#define __fastcall
struct Main_Screen_Form {bool is_now_loading_game=false;int camera_x=3616,camera_y=394;};
struct State {
 struct {bool enable_custom_rendering=true,enable_custom_rendering_zoom=true;} current_config;
 bool combat_unit_display_override_active=false,custom_renderer_unit_representatives_dirty=false;
 int custom_renderer_zoom_target_width=128;
 int (*custom_renderer_native_image)(int,void*,void*,void const*,void const*,unsigned)=nullptr;
 int custom_renderer_zoom_native_tile_width=0,custom_renderer_zoom_tile_width=0,custom_renderer_zoom_wheel_remainder=0;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
 int custom_renderer_dirty_flags=0;bool custom_renderer_redraw_pending=false;
} state,*is=&state;
struct Bic {struct{struct{void* spotlight_on_city=nullptr;}Renderer;}Map;bool is_zoomed_out=false;int ScreenWidth=1024,ScreenHeight=768;} bic,*p_bic_data=&bic;
int player_bits=1,*p_player_bits=&player_bits,redraws=0,targets=0;
int target(int op,void*,void*,void const*,void const*,unsigned q){assert(op==129&&q>=32768&&q<=196608);++targets;return 1;}
void debug(char const*){} auto p_OutputDebugStringA=debug;
void native_move(Main_Screen_Form* s,int,int x,int y,int reason,bool bounds){assert(x==s->camera_x&&y==s->camera_y&&reason==0&&bounds);++redraws;}
int wheel_calls=0;
void native_wheel(Main_Screen_Form*,int edx,int delta,int x,int y){
 assert(edx==73&&delta==120&&x==413&&y==211);++wheel_calls;
}
''' + sync + key + r'''
int main(){
 Main_Screen_Form screen;state.custom_renderer_native_image=target;
 sync_custom_renderer_zoom_to_native();assert(state.custom_renderer_zoom_tile_width==128);
 for(int cycle=0;cycle<3;cycle++)for(int width:{112,96,80,64,384,320,256,224,192,160,128}){
  assert(advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
  assert(state.custom_renderer_zoom_target_width==width&&state.custom_renderer_zoom_tile_width==128);
  auto x=state.custom_renderer_zoom_translate_x_fp,y=state.custom_renderer_zoom_translate_y_fp;
  sync_custom_renderer_zoom_to_native(); // Close zoom must not reset during rendering.
  assert(state.custom_renderer_zoom_target_width==width&&state.custom_renderer_zoom_tile_width==128);
  assert(state.custom_renderer_zoom_translate_x_fp==x && state.custom_renderer_zoom_translate_y_fp==y);
 }
 assert(redraws==0);
 state.current_config.enable_custom_rendering_zoom=false;
 assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));assert(redraws==0);
 state.current_config.enable_custom_rendering_zoom=true;
 screen.is_now_loading_game=true;assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 screen.is_now_loading_game=false;player_bits=0;assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 player_bits=1;assert(!advance_custom_renderer_zoom_from_key(&screen,'x',88));
 bic.is_zoomed_out=true;sync_custom_renderer_zoom_to_native();assert(state.custom_renderer_zoom_tile_width==128);
 assert(state.custom_renderer_zoom_native_tile_width==64);
 assert(state.custom_renderer_zoom_translate_x_fp==-512LL*65536);
 assert(state.custom_renderer_zoom_translate_y_fp==-384LL*65536);
 assert(advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 sync_custom_renderer_zoom_to_native();assert(state.custom_renderer_zoom_target_width==112&&state.custom_renderer_zoom_tile_width==128);
 state.custom_renderer_zoom_tile_width=256;sync_custom_renderer_zoom_to_native();
 assert(state.custom_renderer_zoom_tile_width==128 && state.custom_renderer_zoom_translate_x_fp==-512LL*65536);
 for(int retired:{64,96}){state.custom_renderer_zoom_tile_width=retired;sync_custom_renderer_zoom_to_native();
  assert(state.custom_renderer_zoom_tile_width==128);}
 // Configuration-off never consumes Z or changes the native camera state.
 state.current_config.enable_custom_rendering=false;
 auto before=state.custom_renderer_zoom_translate_x_fp;
 assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 assert(bic.is_zoomed_out && state.custom_renderer_zoom_translate_x_fp==before);
 state.current_config.enable_custom_rendering=true;
 bic.is_zoomed_out=false;sync_custom_renderer_zoom_to_native();assert(state.custom_renderer_zoom_tile_width==128);
 // Wheel keeps native arguments and behavior whenever custom zoom is unavailable.
 for(int unavailable=0;unavailable<4;++unavailable){
  state.current_config.enable_custom_rendering=unavailable!=0;
  state.current_config.enable_custom_rendering_zoom=unavailable!=1;
  player_bits=unavailable!=2;screen.is_now_loading_game=unavailable==3;
  auto before=redraws;patch_Main_Screen_Form_process_mouse_wheel(&screen,73,120,413,211);
  assert(wheel_calls==unavailable+1&&redraws==before&&state.custom_renderer_zoom_wheel_remainder==0);
 }
 state.current_config.enable_custom_rendering=state.current_config.enable_custom_rendering_zoom=true;
 player_bits=1;screen.is_now_loading_game=false;
 auto wheel=[&](int delta){patch_Main_Screen_Form_process_mouse_wheel(&screen,73,delta,413,211);};
 state.combat_unit_display_override_active=true; // Wheel remains accepted during native combat.
 wheel(40);wheel(40);assert(state.custom_renderer_zoom_tile_width==128);
 wheel(40);assert(state.custom_renderer_zoom_target_width==160&&state.custom_renderer_zoom_tile_width==128&&state.custom_renderer_zoom_wheel_remainder==0);
 wheel(240);assert(state.custom_renderer_zoom_target_width==224&&state.custom_renderer_zoom_tile_width==128);
 wheel(1200);assert(state.custom_renderer_zoom_target_width==384);
 auto at_limit=targets;wheel(120);assert(targets==at_limit&&state.custom_renderer_zoom_target_width==384&&state.custom_renderer_zoom_tile_width==128);
 wheel(-120);assert(state.custom_renderer_zoom_target_width==320&&state.custom_renderer_zoom_tile_width==128);
 wheel(-1200);assert(state.custom_renderer_zoom_target_width==64&&state.custom_renderer_zoom_tile_width==128);
 at_limit=targets;wheel(-120);wheel(0);assert(targets==at_limit);
 bic.Map.Renderer.spotlight_on_city=&screen;
 auto before_city=targets;wheel(120);assert(targets==before_city);
 assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 assert(state.custom_renderer_zoom_target_width==64&&state.custom_renderer_zoom_wheel_remainder==0);
 bic.Map.Renderer.spotlight_on_city=nullptr;
 assert(state.custom_renderer_dirty_flags==C3X_RENDERER_DIRTY_SCENE&&state.custom_renderer_redraw_pending&&redraws==0);
 assert(screen.camera_x==3616&&screen.camera_y==394&&wheel_calls==4);
 assert(state.custom_renderer_zoom_translate_x_fp==0&&state.custom_renderer_zoom_translate_y_fp==0);
}
'''
        program = "#include <initializer_list>\n" + program
        run_cpp(program)

    def test_supported_zoom_evidence_rejects_reordering_and_inexact_repeats(self):
        import tempfile
        from pathlib import Path
        from Renderer.native.compare_zoom_benchmark import run
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            lines=[]
            for cycle in range(2):
                for width in (128,192,160):
                    lines.append(f"ZOOM cycle={cycle} width={width} result=1")
                    if cycle:lines.append(f"ZOOM parity width={width} changed=0 error=0 status=pass")
            lines.append("BIQ 100x100 viewport: 0 fallback")
            log=folder/"benchmark.log"
            log.write_text("\n".join(lines))
            self.assertEqual(len(run("supported",folder,"zoom")[1]),6)
            log.write_text("\n".join(lines).replace("cycle=1 width=192", "cycle=1 width=160"))
            with self.assertRaisesRegex(ValueError,"sequence"):run("reordered",folder,"zoom")
            log.write_text("\n".join(lines).replace("error=0", "error=1",1))
            with self.assertRaisesRegex(ValueError,"exact"):run("inexact",folder,"zoom")

    def test_key_handler_only_publishes_a_zoom_target(self) -> None:
        source = (ROOT / "injected_code.c").read_text()
        handler = source.split("advance_custom_renderer_zoom (", 1)[1]
        handler = handler.split("patch_Main_Screen_Form_handle_key_down", 1)[0]
        self.assertIn("if (levels[next] < 128 || levels[current] < 128)", handler)
        self.assertIn("stage=zoom-target", handler)
        self.assertNotIn("Main_Screen_Form_move_camera", handler)
        self.assertNotIn("m73_call_m22_Draw", handler)
        self.assertNotIn("custom_renderer_original_mouse_wheel", handler)
        self.assertNotIn("ensure_custom_renderer_mouse_wheel_hook", source)
        self.assertNotIn("*slot = (int)patch_Main_Screen_Form_process_mouse_wheel", source)
        self.assertNotIn("abs (", handler)


if __name__ == "__main__":
    unittest.main()
