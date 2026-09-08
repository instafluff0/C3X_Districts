import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import hill_height_import


def legacy_r8(width: int, height: int, pixels: bytes) -> bytes:
    header = bytearray(128)
    header[:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<II", header, 12, height, width)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x40)
    struct.pack_into("<I", header, 88, 8)
    struct.pack_into("<I", header, 92, 0xff)
    return bytes(header) + pixels


class HillHeightImportTests(unittest.TestCase):
    def test_legacy_r8_becomes_generic_dx10_r8_without_resampling(self) -> None:
        pixels = bytes(range(16))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.dds"
            output = root / "output.dds"
            source.write_bytes(legacy_r8(4, 4, pixels))
            report = hill_height_import.import_height(source, output)
            normalized = output.read_bytes()
            self.assertEqual(normalized[:4], b"DDS ")
            self.assertEqual(normalized[84:88], b"DX10")
            self.assertEqual(struct.unpack_from("<I", normalized, 128)[0], 61)
            self.assertEqual(normalized[148:], pixels)
            self.assertEqual((report["width"], report["height"]), (4, 4))
            self.assertEqual(report["redistribution"], "local-only")

    def test_rejects_ambiguous_or_truncated_payloads(self) -> None:
        with self.assertRaisesRegex(ValueError, "single-channel"):
            bad = bytearray(legacy_r8(4, 4, bytes(16)))
            struct.pack_into("<I", bad, 92, 0x0f)
            hill_height_import.decode_r8_dds(bytes(bad))
        with self.assertRaisesRegex(ValueError, "payload size"):
            hill_height_import.decode_r8_dds(legacy_r8(4, 4, bytes(15)))


if __name__ == "__main__":
    unittest.main()
