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
            "custom_renderer_zoom_inverse_point (&param_1, &param_2)",
            "custom_renderer_zoom_inverse_point (&local_x, &local_y)",
            "patch_Sprite_draw_on_map",
            "prepare_custom_renderer_zoom_tiles",
            "C3X_RENDERER_TILE_PREFETCH",
            "frame.tile_width = custom_renderer_zoom_enabled ()",
            "draw.projection_scale_milli = is->custom_renderer_zoom_tile_width * 1000 / 128",
            "if (custom_renderer_zoom_enabled () && canvas != NULL && canvas == is->custom_renderer_unit_canvas) return 0",
            "patch_Main_Screen_Form_tile_to_screen_coords",
            "custom_renderer_zoom_native_hud_context",
            "custom_renderer_zoom_unit_tick_translated",
            "offset_x -= is->custom_renderer_zoom_unit_tick_delta_x",
            "offset_y -= is->custom_renderer_zoom_unit_tick_delta_y",
            "draw.body_x -= is->custom_renderer_zoom_unit_tick_delta_x",
            "draw.body_y -= is->custom_renderer_zoom_unit_tick_delta_y",
        ):
            self.assertIn(marker, source)
        self.assertIn("bool enable_custom_rendering_zoom", header)
        self.assertIn("projection_scale_milli", api)

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
