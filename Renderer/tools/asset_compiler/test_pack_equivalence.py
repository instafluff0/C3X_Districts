from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler import pack_equivalence


def write_pack(root: Path, assets: dict[str, tuple[str, bytes]]) -> None:
    manifest_assets = {}
    for logical_id, (file_name, payload) in assets.items():
        material = f"materials/{file_name}.json"
        texture = f"textures/{file_name}.dds"
        (root / "materials").mkdir(parents=True, exist_ok=True)
        (root / "textures").mkdir(parents=True, exist_ok=True)
        (root / texture).write_bytes(payload)
        (root / material).write_text(
            json.dumps({"schema": "c3x.material.v0", "base_color": {"texture": texture}}),
            encoding="utf-8",
        )
        manifest_assets[logical_id] = {"type": "terrain", "material": material}
    (root / "manifest.json").write_text(
        json.dumps({"schema": "c3x.asset_pack.v0", "name": root.name, "assets": manifest_assets}),
        encoding="utf-8",
    )


class PackEquivalenceTests(unittest.TestCase):
    def test_reports_replaced_inherited_missing_and_added_ids_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            alternate = root / "alternate"
            baseline.mkdir()
            alternate.mkdir()
            write_pack(baseline, {
                "terrain/grassland/base": ("grass", b"baseline-grass"),
                "terrain/plains/base": ("plains", b"same-plains"),
                "terrain/ocean/base": ("ocean", b"baseline-ocean"),
            })
            write_pack(alternate, {
                "terrain/grassland/base": ("grass", b"alternate-grass"),
                "terrain/plains/base": ("plains", b"same-plains"),
                "terrain/marsh/base": ("marsh", b"alternate-marsh"),
            })

            first = pack_equivalence.compare_packs(baseline, alternate)
            second = pack_equivalence.compare_packs(baseline, alternate)

        self.assertEqual(first, second)
        self.assertEqual(
            ["terrain/grassland/base", "terrain/plains/base"],
            first["logical_ids"]["shared"],
        )
        self.assertEqual(["terrain/grassland/base"], first["logical_ids"]["replaced"])
        self.assertEqual(["terrain/plains/base"], first["logical_ids"]["inherited"])
        self.assertEqual(["terrain/ocean/base"], first["logical_ids"]["missing"])
        self.assertEqual(["terrain/marsh/base"], first["logical_ids"]["added"])
        self.assertEqual(first["report_sha256"], second["report_sha256"])

    def test_rejects_runtime_file_reference_that_escapes_pack_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            alternate = root / "alternate"
            baseline.mkdir()
            alternate.mkdir()
            write_pack(baseline, {"terrain/grassland/base": ("grass", b"grass")})
            write_pack(alternate, {"terrain/grassland/base": ("grass", b"grass")})
            manifest = json.loads((alternate / "manifest.json").read_text(encoding="utf-8"))
            manifest["assets"]["terrain/grassland/base"]["material"] = "../outside.json"
            (alternate / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                pack_equivalence.compare_packs(baseline, alternate)


if __name__ == "__main__":
    unittest.main()
