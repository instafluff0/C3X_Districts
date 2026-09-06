from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Renderer.tools.asset_compiler import shore_blp_extractor as extractor


class ShorePackTests(unittest.TestCase):
    def test_inventory_contains_verified_cliff_and_ice_sets(self) -> None:
        groups: dict[str, int] = {}
        for spec in extractor.SHORE_SPECS:
            groups[spec["group"]] = groups.get(spec["group"], 0) + 1
        self.assertEqual(
            {"cliff_large": 4, "cliff_small": 2, "polar_ice": 16, "river_rock": 5},
            groups,
        )
        self.assertEqual(17, len(extractor.SOURCE_EXCLUSIONS))

    def test_build_writes_source_agnostic_runtime_manifest(self) -> None:
        class FakePackage:
            data = b"package"
            table_offset = 12
            allocations = [object(), object()]
            stripe_bases = {0: 100, 1: 200}

        def fake_build(_package, _shared, pack, spec):
            mesh = f"meshes/features/{spec['stem']}.json"
            material = f"materials/features/{spec['stem']}.json"
            (pack / mesh).parent.mkdir(parents=True, exist_ok=True)
            (pack / material).parent.mkdir(parents=True, exist_ok=True)
            (pack / mesh).write_text("{}\n", encoding="utf-8")
            (pack / material).write_text("{}\n", encoding="utf-8")
            return {"type": "feature", "mesh": mesh, "material": material}, {
                "selected_asset": spec["source_name"],
                "normalized_asset_id": spec["asset_id"],
            }

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = root / "pack"
            report = root / "report.json"
            with (
                mock.patch.object(extractor, "StaticPackage", return_value=FakePackage()),
                mock.patch.object(extractor, "build_feature", side_effect=fake_build),
                mock.patch.object(extractor, "validate_runtime_independence", return_value=[]),
            ):
                result = extractor.build_shore_pack(
                    root / "environment" / "clutter.blp", root / "SHARED_DATA", pack, report
                )

            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(27, len(manifest["assets"]))
            self.assertEqual(16, len(manifest["feature_sets"]["polar_ice"]["variants"]))
            self.assertEqual("verified_subset", manifest["feature_sets"]["cliff_small"]["status"])
            self.assertEqual(5, len(manifest["feature_sets"]["river_rock"]["variants"]))
            self.assertNotIn("TER_", json.dumps(manifest))
            self.assertEqual("passed", result["runtime_independence"])
            self.assertEqual(17, len(result["excluded_source_candidates"]))


if __name__ == "__main__":
    unittest.main()
