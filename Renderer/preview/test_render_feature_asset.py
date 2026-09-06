from __future__ import annotations

import struct
import unittest

from Renderer.preview.render_feature_asset import DdsBc1Texture, decode_bc1


class Bc1Tests(unittest.TestCase):
    def test_decodes_four_color_block(self) -> None:
        block = struct.pack("<HHI", 0xF800, 0x001F, 0)
        self.assertEqual(decode_bc1(block, 0, 0), (255, 0, 0, 255))

    def test_decodes_transparent_selector(self) -> None:
        selectors = 3
        block = struct.pack("<HHI", 0x001F, 0xF800, selectors)
        self.assertEqual(decode_bc1(block, 0, 0), (0, 0, 0, 0))

    def test_texture_addressing_can_wrap_independently(self) -> None:
        dds = bytearray(148)
        dds[:4] = b"DDS "
        dds[84:88] = b"DX10"
        struct.pack_into("<I", dds, 12, 4)
        struct.pack_into("<I", dds, 16, 4)
        struct.pack_into("<I", dds, 28, 1)
        struct.pack_into("<I", dds, 128, 71)
        dds.extend(struct.pack("<HHI", 0xF800, 0x001F, 0))
        texture = DdsBc1Texture(bytes(dds), "wrap", "clamp")
        self.assertEqual(texture.sample(-0.25, 0.25), texture.sample(0.75, 0.25))
        self.assertEqual(texture.sample(1.25, 2.0), texture.sample(0.25, 1.0))


if __name__ == "__main__":
    unittest.main()
