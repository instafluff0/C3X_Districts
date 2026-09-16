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
    def test_z_key_cycles_toward_zoom_out_then_wraps(self) -> None:
        levels = [128, 160, 192]
        current = 128
        observed = []
        for _ in levels:
            index = levels.index(current)
            current = levels[index - 1] if index > 0 else levels[-1]
            observed.append(current)
        self.assertEqual(observed, [192, 160, 128])
        self.assertEqual(levels[0], 128)

    def test_cursor_anchor_and_inverse_pick(self) -> None:
        for native_width in (64, 128):
            translation = 0
            old_width = native_width
            cursor = 517
            for width in (192, 160, 128):
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
            "int levels[3] = {128, 160, 192}",
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
        functions = functions.replace("this", "screen")
        program = r'''
#include <cassert>
#define __fastcall
constexpr int __=0;
struct State {
 int custom_renderer_zoom_native_tile_width=128,custom_renderer_zoom_tile_width=128;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
} state,*is=&state;
struct CityForm {struct {struct {int Status2=0;} Data;} Base;} city,*p_city_form=&city;
struct Main_Screen_Form {int mouse_x=0,mouse_y=0;};
struct Unit{};struct Animator{};struct PCX_Image{};struct PCX_Color_Table{};
struct Sprite{int Width=95,Height=63;};
struct Bic{bool is_zoomed_out=false;} bic,*p_bic_data=&bic;
int status_calls=0,cursor_calls=0,marker_calls=0,overlay_x=0,overlay_y=0;
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
''' + functions + r'''
int main(){
 Main_Screen_Form screen;
 for(int basis:{64,128})for(int zoom:{128,160,192})for(int camera:{-317,0,803}){
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
struct Unit{};struct PCX_Image{};
Unit outer,inner;PCX_Image previous,canvas;
struct State {
 Unit* custom_renderer_unit_context=&outer;PCX_Image* custom_renderer_unit_canvas=&previous;
 struct {bool enable_custom_rendering=true;} current_config;
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
 state.current_config.enable_custom_rendering=false;
 patch_Unit_tick_anim(&inner,0,&canvas,-413,291,true);
 assert(calls==2&&state.custom_renderer_unit_context==&outer&&state.custom_renderer_unit_canvas==&previous);
}
''')

    def test_actual_injected_cycle_and_close_zoom_survive_sync(self) -> None:
        source = (ROOT / "injected_code.c").read_text()
        key = "bool\nadvance_custom_renderer_zoom_from_key" + source.split(
            "bool\nadvance_custom_renderer_zoom_from_key", 1)[1].split(
            "\nint __fastcall\npatch_Main_Screen_Form_handle_key_down", 1)[0]
        # 'this' is a valid identifier in the injected C, not in C++.
        key = key.replace("this", "screen")
        sync = "void\nsync_custom_renderer_zoom_to_native" + source.split(
            "void\nsync_custom_renderer_zoom_to_native", 1)[1].split(
            "\nint\ncustom_renderer_zoom_transform_coordinate", 1)[0]
        program = r'''
#include <cassert>
#include <cstdio>
#define ARRAY_LEN(a) (sizeof(a)/sizeof((a)[0]))
enum {VK_Z=90,C3X_RENDERER_DIRTY_ALL=255,__=0};
struct Main_Screen_Form {bool is_now_loading_game=false;int TileX_Max=100,TileX_Min=0,TileY_Max=100,TileY_Min=0;};
struct State {
 struct {bool enable_custom_rendering=true,enable_custom_rendering_zoom=true;} current_config;
 int custom_renderer_zoom_native_tile_width=0,custom_renderer_zoom_tile_width=0;
 long long custom_renderer_zoom_translate_x_fp=0,custom_renderer_zoom_translate_y_fp=0;
 int custom_renderer_dirty_flags=0;bool custom_renderer_redraw_pending=false;
} state,*is=&state;
struct Bic {bool is_zoomed_out=false;int ScreenWidth=1024,ScreenHeight=768;} bic,*p_bic_data=&bic;
int player_bits=1,*p_player_bits=&player_bits,redraws=0;
void debug(char const*){} auto p_OutputDebugStringA=debug;
void Main_Screen_Form_bring_tile_into_view(Main_Screen_Form*,int,int,int,int,bool,bool){++redraws;}
''' + sync + key + r'''
int main(){
 Main_Screen_Form screen;
 sync_custom_renderer_zoom_to_native();assert(state.custom_renderer_zoom_tile_width==128);
 for(int cycle=0;cycle<3;cycle++)for(int width:{192,160,128}){
  assert(advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
  assert(state.custom_renderer_zoom_tile_width==width);
  auto x=state.custom_renderer_zoom_translate_x_fp,y=state.custom_renderer_zoom_translate_y_fp;
  sync_custom_renderer_zoom_to_native(); // Close zoom must not reset during rendering.
  assert(state.custom_renderer_zoom_tile_width==width);
  assert(state.custom_renderer_zoom_translate_x_fp==x && state.custom_renderer_zoom_translate_y_fp==y);
 }
 assert(redraws==9);
 state.current_config.enable_custom_rendering_zoom=false;
 assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));assert(redraws==9);
 state.current_config.enable_custom_rendering_zoom=true;
 screen.is_now_loading_game=true;assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 screen.is_now_loading_game=false;player_bits=0;assert(!advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 player_bits=1;assert(!advance_custom_renderer_zoom_from_key(&screen,'x',88));
 bic.is_zoomed_out=true;sync_custom_renderer_zoom_to_native();assert(state.custom_renderer_zoom_tile_width==128);
 assert(state.custom_renderer_zoom_native_tile_width==64);
 assert(state.custom_renderer_zoom_translate_x_fp==-512LL*65536);
 assert(state.custom_renderer_zoom_translate_y_fp==-384LL*65536);
 assert(advance_custom_renderer_zoom_from_key(&screen,'z',VK_Z));
 sync_custom_renderer_zoom_to_native();assert(state.custom_renderer_zoom_tile_width==192);
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

    def test_key_handler_queues_native_redraw_without_indirect_dispatch(self) -> None:
        source = (ROOT / "injected_code.c").read_text()
        handler = source.split("advance_custom_renderer_zoom_from_key", 1)[1]
        handler = handler.split("patch_Main_Screen_Form_handle_key_down", 1)[0]
        self.assertIn("custom_renderer_redraw_pending = true", handler)
        self.assertIn("stage=zoom-key", handler)
        self.assertIn("Main_Screen_Form_bring_tile_into_view", handler)
        self.assertIn("center_x - 1, center_y - 1, 0, true, false", handler)
        self.assertNotIn("m73_call_m22_Draw", handler)
        self.assertNotIn("custom_renderer_original_mouse_wheel", handler)
        self.assertNotIn("ensure_custom_renderer_mouse_wheel_hook", source)
        self.assertNotIn("*slot = (int)patch_Main_Screen_Form_process_mouse_wheel", source)
        self.assertNotIn("abs (", handler)


if __name__ == "__main__":
    unittest.main()
