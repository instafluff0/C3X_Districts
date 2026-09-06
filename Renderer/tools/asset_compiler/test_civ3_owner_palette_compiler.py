from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.civ3_owner_palette_compiler import (
    COLORS_PER_TABLE,
    TABLE_COUNT,
    compile_owner_palettes,
    load_owner_color_table,
    read_pcx_palette,
)


def _write_palette(path: Path, seed: int) -> None:
    header = bytearray(128)
    header[0] = 0x0A
    header[3] = 8
    header[65] = 1
    palette = bytes((seed + index) & 255 for index in range(768))
    path.write_bytes(bytes(header) + b"\x00" + b"\x0c" + palette)


class Civ3OwnerPaletteCompilerTests(unittest.TestCase):
    def test_reads_trailing_pcx_palette(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ntp00.pcx"
            _write_palette(path, 7)
            colors = read_pcx_palette(path)
            self.assertEqual(256, len(colors))
            self.assertEqual([7, 8, 9], colors[0])
            self.assertEqual([10, 11, 12], colors[1])

    def test_partial_high_priority_root_overrides_per_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            scenario = root / "scenario"
            output = root / "pack"
            base.mkdir()
            scenario.mkdir()
            for table_id in range(TABLE_COUNT):
                _write_palette(base / f"ntp{table_id:02d}.pcx", table_id)
            _write_palette(scenario / "NTP06.PCX", 200)
            report = compile_owner_palettes([base, scenario], output)
            self.assertEqual(1, report["scenario_override_count"])
            self.assertEqual([200, 201, 202], load_owner_color_table(output, 6)["colors"][0])
            self.assertEqual([5, 6, 7], load_owner_color_table(output, 5)["colors"][0])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(TABLE_COUNT * COLORS_PER_TABLE * 4, (output / manifest["gpu_lut"]["path"]).stat().st_size)
            self.assertNotIn(str(base), (output / "owner_colors.json").read_text(encoding="utf-8"))

    def test_missing_table_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_palette(root / "ntp00.pcx", 0)
            with self.assertRaisesRegex(ValueError, "ntp01.pcx"):
                compile_owner_palettes([root], root / "pack")

    def test_runtime_loader_rejects_corrupt_gpu_lut(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for table_id in range(TABLE_COUNT):
                _write_palette(source / f"ntp{table_id:02d}.pcx", table_id)
            pack = root / "pack"
            compile_owner_palettes([source], pack)
            (pack / "owner_colors.rgba8").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "byte size"):
                load_owner_color_table(pack, 0)


if __name__ == "__main__":
    unittest.main()
