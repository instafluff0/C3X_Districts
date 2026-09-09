"""Validate the injected main-map zoom transform and its integration contracts."""
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
        levels = [64, 80, 96, 112, 128]
        current = 128
        observed = []
        for _ in levels:
            index = levels.index(current)
            current = levels[index - 1] if index > 0 else levels[-1]
            observed.append(current)
        self.assertEqual(observed, [112, 96, 80, 64, 128])

    def test_cursor_anchor_and_inverse_pick(self) -> None:
        for native_width in (64, 128):
            translation = 0
            old_width = native_width
            cursor = 517
            for width in ((80, 96, 112, 128) if native_width == 64 else (112, 96, 80, 64)):
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
            "int levels[5] = {64, 80, 96, 112, 128}",
            "advance_custom_renderer_zoom_from_key",
            "custom_renderer_zoom_inverse_point (&param_1, &param_2)",
            "custom_renderer_zoom_inverse_point (&local_x, &local_y)",
            "patch_Sprite_draw_on_map",
            "prepare_custom_renderer_zoom_tiles",
            "C3X_RENDERER_TILE_PREFETCH",
            "frame.tile_width = custom_renderer_zoom_enabled ()",
            "draw.projection_scale_milli = is->custom_renderer_zoom_tile_width * 1000 / 128",
            "if (custom_renderer_zoom_enabled ()) return 0",
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
