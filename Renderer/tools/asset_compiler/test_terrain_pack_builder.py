from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Renderer.tools.asset_compiler import c3x_asset_compiler
from Renderer.tools.asset_compiler import terrain_pack_builder


class TerrainPackBuilderTests(unittest.TestCase):
    def test_mapping_and_fallback_contract_cover_production_families(self) -> None:
        mapped = set(terrain_pack_builder.MATERIAL_TARGETS)
        self.assertEqual(len(mapped), 14)
        self.assertEqual(mapped | set(terrain_pack_builder.EXPLICIT_FALLBACKS).intersection({
            "flood_plain", "hills", "mountains", "forest", "jungle", "marsh", "volcano", "ocean"
        }), {
            "desert", "plains", "grassland", "tundra", "flood_plain", "hills",
            "mountains", "forest", "jungle", "marsh", "volcano", "coast", "sea", "ocean",
        })
        self.assertTrue({"transitions", "polar_ice", "landmarks"}.issubset(terrain_pack_builder.EXPLICIT_FALLBACKS))

    def test_compiled_pack_is_generic_and_complete_for_mapped_materials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "source.blp"
            package.write_bytes(b"fixture")
            mesh = Path("Renderer/samples/geometry/flat_terrain_patch.json")
            pack = root / "pack"
            report = root / "out" / "report.json"
            dds = c3x_asset_compiler.make_dds_dx10_header({
                "width": 4, "height": 4, "mip_count": 1, "dxgi_format": 78,
            }) + bytes.fromhex("ffff000000000000e007e00700000000")

            def fake_binding(_package: Path, target: str, occurrence=None):
                return {"target": target, "roles": []}

            def fake_extract(_package: Path, binding: dict, role: str, output: Path):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(dds)
                return {
                    "width": 4, "height": 4, "mip_count": 1,
                    "dxgi_format": 78 if role == "base_color" else 80,
                    "format_name": "BC3_UNORM_SRGB" if role == "base_color" else "BC4_UNORM",
                    "color_space": "srgb" if role == "base_color" else "linear",
                    "logical_name": f'{binding["target"]}_{role}', "dds_sha256": f"fixture-{role}",
                }

            def fake_relief(_package: Path, output: Path):
                output.mkdir(parents=True, exist_ok=True)
                r8 = terrain_pack_builder.terrain_relief_builder.make_r8_dds(4, 4, bytes(range(16)))
                for role in terrain_pack_builder.terrain_relief_builder.RELIEF_OUTPUTS:
                    (output / f"{role}.dds").write_bytes(r8)
                return {"schema": "c3x.terrain_relief_probe.v0", "extracted": []}

            def fake_authored_relief(_package: Path, output: Path):
                (output / "relief").mkdir(parents=True, exist_ok=True)
                for name, kind in (("hills", "hills"), ("mountains", "mountains")):
                    (output / "relief" / f"{name}.json").write_text(
                        json.dumps({"schema": "c3x.relief_set.v0", "kind": kind}) + "\n",
                        encoding="utf-8",
                    )
                return {
                    "schema": "c3x.authored_relief_compile.v0",
                    "runtime_sets": {"hills": "relief/hills.json", "mountains": "relief/mountains.json"},
                    "source_evidence": [],
                    "compiled_texture_count": 0,
                }

            with mock.patch.object(terrain_pack_builder.civblp_material_resolver, "resolve_file", side_effect=fake_binding), \
                 mock.patch.object(terrain_pack_builder.grassland_pack_builder, "extract_embedded_texture_role", side_effect=fake_extract), \
                 mock.patch.object(terrain_pack_builder.terrain_relief_builder, "extract_relief_resources", side_effect=fake_relief), \
                 mock.patch.object(terrain_pack_builder.terrain_relief_builder, "compile_authored_relief_sets", side_effect=fake_authored_relief), \
                 mock.patch.object(terrain_pack_builder.terrain_relief_builder, "validate_authored_relief_sets", return_value=[]):
                result = terrain_pack_builder.build_local_terrain_pack(package, mesh, pack, report)

            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            coverage = json.loads((pack / "runtime_coverage.json").read_text(encoding="utf-8"))
            self.assertEqual(result["mapped_count"], 14)
            self.assertEqual(len(manifest["assets"]), 14)
            self.assertEqual(set(terrain_pack_builder.AUXILIARY_MATERIAL_TARGETS), set(manifest["material_library"]))
            self.assertEqual(
                {"hills": "relief/hills.json", "mountains": "relief/mountains.json"},
                manifest["relief_sets"],
            )
            self.assertEqual(len(coverage["fallbacks"]), 4)
            hills = json.loads((pack / "materials" / "hills.json").read_text(encoding="utf-8"))
            self.assertEqual("R8_UNORM", hills["relief"]["format"])
            self.assertEqual("connected_hills", hills["relief"]["profile"])
            self.assertEqual("relief/hills.json", hills["authored_relief_set"])
            self.assertEqual("BC4_UNORM", hills["height"]["format"])
            self.assertEqual("BC4_UNORM", hills["specular"]["format"])
            grassland = json.loads((pack / "materials" / "grassland.json").read_text(encoding="utf-8"))
            mountains = json.loads((pack / "materials" / "mountains.json").read_text(encoding="utf-8"))
            desert = json.loads((pack / "materials" / "desert.json").read_text(encoding="utf-8"))
            coast = json.loads((pack / "materials" / "coast.json").read_text(encoding="utf-8"))
            self.assertEqual("BC3_UNORM_SRGB", grassland["elevated"]["base_color"]["format"])
            self.assertEqual("BC4_UNORM", grassland["elevated"]["height"]["format"])
            self.assertEqual("BC3_UNORM_SRGB", mountains["elevated"]["base_color"]["format"])
            self.assertEqual("relief/mountains.json", mountains["authored_relief_set"])
            self.assertEqual(
                {"snow", "desert_base", "desert_stripe_1", "desert_stripe_2", "desert_stripe_3"},
                set(mountains["authored_layers"]),
            )
            self.assertTrue(all(
                layer["base_color"]["format"] == "BC3_UNORM_SRGB"
                for layer in mountains["authored_layers"].values()
            ))
            self.assertEqual({"beach", "cliff", "cliff_white"}, set(coast["authored_layers"]))
            self.assertNotIn("elevated", desert)
            self.assertEqual(
                "ART_DEF_TERRAIN_MATERIAL_MTN_TOP",
                next(item for item in result["source_evidence"] if item["terrain"] == "mountains")["elevated_source_target"],
            )
            serialized = json.dumps(manifest).lower()
            self.assertNotIn("civ6", serialized)
            self.assertNotIn(".blp", serialized)
            self.assertFalse(any(Path(value["material"]).is_absolute() for value in manifest["assets"].values()))
            self.assertIsNone(manifest["water"])


if __name__ == "__main__":
    unittest.main()
