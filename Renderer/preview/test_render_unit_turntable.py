from __future__ import annotations

import struct
import unittest

from Renderer.preview.render_unit_turntable import apply_owner_tint, decode_bc3


class UnitTurntableTests(unittest.TestCase):
    def test_bc3_decoder_reads_opaque_color_and_alpha_endpoints(self) -> None:
        alpha = bytes((255, 0)) + (0).to_bytes(6, "little")
        color = struct.pack("<HHI", 0xF800, 0x07E0, 0)
        self.assertEqual((255, 0, 0, 255), decode_bc3(alpha + color, 0, 0))

    def test_exact_civ3_primary_ramp_preserves_unmasked_texel(self) -> None:
        ramp = [[255 - index * 12, 10, 10] for index in range(16)]
        self.assertEqual((90, 100, 110), apply_owner_tint((90, 100, 110), 255, owner_ramp=ramp))

    def test_exact_civ3_display_color_modulates_without_erasing_source_value(self) -> None:
        ramp = [[255 - index * 12, 10, 10] for index in range(16)]
        result = apply_owner_tint((128, 128, 128), 0, owner_ramp=ramp)
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], 70)
        self.assertLess(result[0], 170)

    def test_owner_color_modes_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            apply_owner_tint((128, 128, 128), 0, (1, 2, 3), [[1, 2, 3]] * 16)

    def test_owner_tint_strength_is_bounded_and_supports_readability_accents(self) -> None:
        ramp = [[255 - index * 12, 10, 10] for index in range(16)]
        subtle = apply_owner_tint((128, 128, 128), 0, owner_ramp=ramp, tint_strength=0.25)
        strong = apply_owner_tint((128, 128, 128), 0, owner_ramp=ramp, tint_strength=1.0)
        self.assertGreater(strong[0] - strong[1], subtle[0] - subtle[1])
        with self.assertRaisesRegex(ValueError, "between zero and one"):
            apply_owner_tint((128, 128, 128), 0, owner_ramp=ramp, tint_strength=1.1)

    def test_material_can_select_any_preserved_runtime_palette_slot(self) -> None:
        colors = [[0, 0, 0] for _ in range(64)]
        colors[22] = [240, 20, 20]
        result = apply_owner_tint(
            (128, 128, 128),
            0,
            owner_ramp=colors,
            representative_palette_index=22,
        )
        self.assertGreater(result[0], result[1])
        with self.assertRaisesRegex(ValueError, "outside"):
            apply_owner_tint(
                (128, 128, 128),
                0,
                owner_ramp=[[0, 0, 0]] * 16,
                representative_palette_index=22,
            )


if __name__ == "__main__":
    unittest.main()
