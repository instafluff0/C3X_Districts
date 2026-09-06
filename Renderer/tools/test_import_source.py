from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Renderer.tools import import_source


def make_source_tree(root: Path, overlay: bool = False) -> None:
    relative_root = Path("Platforms/Windows/BLPs/terrain") if overlay else Path(
        "Base/Platforms/Windows/BLPs/terrain"
    )
    package_root = root / relative_root
    package_root.mkdir(parents=True)
    (package_root / "TerrainMaterialSet_Base.blp").write_bytes(b"material")
    (package_root / "TerrainElementSet_Base.blp").write_bytes(b"relief")


def permission_record(path: Path) -> None:
    path.write_text(json.dumps({
        "schema": import_source.PERMISSION_SCHEMA,
        "source_name": "Synthetic skin",
        "rights_holder": "Fixture Author",
        "grant_reference": "fixture://permission",
        "permissions": ["conversion", "cross-game-use"],
        "redistribution": "local-only",
    }), encoding="utf-8")


def fake_build(package: Path, mesh: Path, pack: Path, report: Path, **kwargs):
    (pack / "materials").mkdir(parents=True, exist_ok=True)
    (pack / "textures").mkdir(parents=True, exist_ok=True)
    (pack / "textures" / "grass.dds").write_bytes(b"fixture")
    (pack / "materials" / "grass.json").write_text(json.dumps({
        "schema": "c3x.material.v0",
        "base_color": {"texture": "textures/grass.dds"},
    }), encoding="utf-8")
    (pack / "manifest.json").write_text(json.dumps({
        "schema": "c3x.asset_pack.v0",
        "name": "SyntheticTerrain",
        "assets": {"terrain/grassland/base": {
            "type": "terrain", "material": "materials/grass.json",
        }},
    }), encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema": "fixture"}), encoding="utf-8")
    return {"mapped_count": 1}


class ImportSourceTests(unittest.TestCase):
    def test_overlay_requires_documented_conversion_permission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            overlay = root / "overlay"
            make_source_tree(source)
            make_source_tree(overlay, overlay=True)
            with self.assertRaisesRegex(ValueError, "permission-record"):
                import_source.build_variant(
                    source, root / "output", "synthetic-overlay", overlay_root=overlay
                )

    def test_local_testing_acknowledgement_is_non_distributable_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            overlay = root / "overlay"
            baseline = root / "baseline"
            output = root / "output"
            make_source_tree(source)
            make_source_tree(overlay, overlay=True)
            baseline.mkdir()
            fake_build(Path(), Path(), baseline, baseline / "provenance" / "build.json")
            with mock.patch.object(
                import_source.terrain_pack_builder,
                "build_local_terrain_pack",
                side_effect=fake_build,
            ):
                result = import_source.build_variant(
                    source,
                    output,
                    "synthetic-overlay",
                    overlay_root=overlay,
                    baseline_pack=baseline,
                    local_testing_only=True,
                )

            self.assertEqual("prohibited", result["source"]["permission"]["redistribution"])
            self.assertEqual(
                "explicit-user-local-testing-only",
                result["source"]["permission"]["basis"],
            )

    def test_builds_overlay_in_staging_and_records_source_aware_provenance_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            overlay = root / "overlay"
            output = root / "output"
            baseline = root / "baseline"
            permission = root / "permission.json"
            make_source_tree(source)
            make_source_tree(overlay, overlay=True)
            permission_record(permission)
            baseline.mkdir()
            fake_build(Path(), Path(), baseline, baseline / "provenance" / "build.json")
            with mock.patch.object(
                import_source.terrain_pack_builder,
                "build_local_terrain_pack",
                side_effect=fake_build,
            ) as builder:
                result = import_source.build_variant(
                    source,
                    output,
                    "synthetic-overlay",
                    overlay_root=overlay,
                    baseline_pack=baseline,
                    permission_record=permission,
                )

            self.assertEqual(output.resolve(), result["output"])
            self.assertTrue((output / "manifest.json").is_file())
            provenance = json.loads(
                (output / "provenance" / "import.json").read_text(encoding="utf-8")
            )
            self.assertEqual("overlay", provenance["source_kind"])
            self.assertEqual("baseline", provenance["relief_source_kind"])
            self.assertEqual(["relief", "water"], provenance["inherited_components"])
            self.assertTrue(provenance["runtime_source_independent"])
            self.assertTrue((output / "provenance" / "equivalence_report.json").is_file())
            self.assertEqual(
                import_source.OVERLAY_MATERIAL_OCCURRENCES,
                builder.call_args.kwargs["material_occurrences"],
            )
            self.assertEqual([], import_source.grassland_pack_builder.validate_runtime_independence(output))

    def test_refuses_to_silently_overwrite_an_existing_variant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            make_source_tree(source)
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "--replace"):
                import_source.build_variant(source, output, "vanilla")


if __name__ == "__main__":
    unittest.main()
