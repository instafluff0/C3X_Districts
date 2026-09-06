from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Renderer.preview.render_owner_palette_sheet import render_palette_sheet
from Renderer.tools.asset_compiler.civ3_owner_palette_compiler import TABLE_COUNT, compile_owner_palettes
from Renderer.tools.asset_compiler.test_civ3_owner_palette_compiler import _write_palette


class OwnerPaletteSheetTests(unittest.TestCase):
    def test_renders_all_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for table_id in range(TABLE_COUNT):
                _write_palette(source / f"ntp{table_id:02d}.pcx", table_id)
            pack = root / "pack"
            output = root / "sheet.png"
            compile_owner_palettes([source], pack)
            report = render_palette_sheet(pack, output)
            self.assertEqual(32, report["table_count"])
            self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
