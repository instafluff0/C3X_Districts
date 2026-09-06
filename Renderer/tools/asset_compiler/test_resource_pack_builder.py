from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from Renderer.tools.asset_compiler import indexed_static_package
from Renderer.tools.asset_compiler import resource_pack_builder


class ResourcePackTests(unittest.TestCase):
    def test_specialized_routes_cover_compound_and_decal_resource_assets(self) -> None:
        routes = resource_pack_builder.load_special_asset_routes()
        self.assertEqual(
            routes["RES_Elephant_01"],
            {"pack": "CompoundLandmarksNormalized", "asset": "resource/elephants/variant_01"},
        )
        self.assertEqual(
            routes["RES_Oil_Land_Decal01"],
            {"pack": "DecalsNormalized", "asset": "resource/oil/decal_01"},
        )
        self.assertEqual(len(routes), 30)

    def test_landmark_routes_include_oasis_and_animated_resources(self) -> None:
        routes = resource_pack_builder.load_landmark_routes()
        self.assertEqual(
            routes["FEATURE_Oasis_OB"],
            {"pack": "CompoundLandmarksNormalized", "asset": "terrain/feature/oasis"},
        )
        self.assertEqual(routes["RES_Fish"]["asset"], "resource/fish")
        self.assertEqual(routes["RES_Whale_01"]["asset"], "resource/whales")

    def test_single_subject_profile_is_data_driven(self) -> None:
        document = resource_pack_builder.load_resource_presentation()
        profile = document["profiles"]["single_primary_subject"]
        self.assertEqual(profile["subject_count"], 1)
        self.assertEqual(profile["ancillary_policy"], "omit")
        self.assertEqual(
            document["resource_bindings"]["resource/cattle"]["profile"],
            "single_primary_subject",
        )

    def test_build_filters_unsupported_placements_and_registers_clips(self) -> None:
        inventory = {
            "resources": [
                {
                    "resource_id": "resource/test",
                    "placements": [
                        {
                            "asset": {"package": "environment/clutter", "entry": "Good"},
                            "clutter_set": "SOURCE_SET",
                            "name": "SOURCE_PLACEMENT",
                            "scale": 0.5,
                            "count": 2,
                        },
                        {
                            "asset": {"package": "environment/clutter", "entry": "Bad"},
                            "clutter_set": "SOURCE_SET",
                            "name": "SOURCE_PLACEMENT_2",
                            "scale": 1.0,
                            "count": 1,
                        },
                    ],
                }
            ]
        }
        probe = {
            "schema": "probe",
            "assets": [
                {
                    "source": {"package": "environment/clutter", "entry": "Good"},
                    "status": "normalized",
                    "manifest_key": "resource/assets/good",
                    "manifest_asset": {"type": "feature", "mesh": "meshes/good.json"},
                },
                {
                    "source": {"package": "environment/clutter", "entry": "Bad"},
                    "status": "unsupported",
                    "reason": "unsupported fixture",
                },
            ],
            "summary": {"candidates": 2, "normalized": 1, "unsupported": 1},
        }
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory)
            (pack / "animations").mkdir()
            (pack / "animations" / "fish_ambient.c3anim").write_bytes(b"fish")
            with mock.patch.object(resource_pack_builder, "probe_static_assets", return_value=probe):
                report = resource_pack_builder.build_static_pack(inventory, pack, pack)
            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))

        resource = manifest["resources"]["resource/test"]
        self.assertEqual(resource["placements"], [{"asset": "resource/assets/good", "count": 2, "scale": 0.5}])
        self.assertEqual(resource["omitted_source_entries"][0]["entry"], "Bad")
        self.assertNotIn("SOURCE_SET", json.dumps(manifest))
        self.assertEqual(list(manifest["animations"]), ["resource/fish"])
        self.assertEqual(report["summary"]["manifest_assets"], 1)
        self.assertEqual(report["summary"]["complete_static_resources"], 0)

    def test_build_emits_one_deduplicated_primary_subject_candidate(self) -> None:
        inventory = {
            "resources": [
                {
                    "resource_id": "resource/test",
                    "placements": [
                        {
                            "asset": {"package": "environment/clutter", "entry": "Animal"},
                            "clutter_set": "SOURCE_SET",
                            "name": "FIRST",
                            "scale": 1.0,
                            "count": 4,
                        },
                        {
                            "asset": {"package": "environment/clutter", "entry": "Animal"},
                            "clutter_set": "SOURCE_SET",
                            "name": "SECOND",
                            "scale": 0.8,
                            "count": 2,
                        },
                    ],
                    "landmarks": [],
                }
            ]
        }
        probe = {
            "schema": "probe",
            "assets": [
                {
                    "source": {"package": "environment/clutter", "entry": "Animal"},
                    "status": "normalized",
                    "manifest_key": "resource/assets/animal",
                    "manifest_asset": {"type": "feature", "mesh": "meshes/animal.json"},
                }
            ],
            "summary": {"candidates": 1, "normalized": 1, "routed": 0, "unsupported": 0},
        }
        presentation = {
            "schema": "c3x.resource_presentation_profiles.v0",
            "default_profile": "cluster",
            "profiles": {
                "cluster": {"composition": "source_authored_cluster"},
                "one": {
                    "composition": "single_primary_subject",
                    "subject_count": 1,
                    "ancillary_policy": "omit",
                },
            },
            "resource_bindings": {
                "resource/test": {
                    "profile": "one",
                    "primary_source_entries": ["Animal"],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory)
            presentation_path = pack / "presentation.json"
            presentation_path.write_text(json.dumps(presentation), encoding="utf-8")
            with mock.patch.object(resource_pack_builder, "probe_static_assets", return_value=probe):
                resource_pack_builder.build_static_pack(
                    inventory, pack, pack, presentation_path
                )
            manifest = json.loads((pack / "manifest.json").read_text(encoding="utf-8"))

        compiled = manifest["resources"]["resource/test"]["presentation"]
        self.assertEqual(compiled["composition"], "single_primary_subject")
        self.assertEqual(compiled["subject_count"], 1)
        self.assertEqual(
            compiled["subject_candidates"], [{"asset": "resource/assets/animal"}]
        )


class IndexedPackageTests(unittest.TestCase):
    def test_type_name_caches_reflected_resolution(self) -> None:
        package = indexed_static_package.IndexedStaticPackage.__new__(
            indexed_static_package.IndexedStaticPackage
        )
        package.allocations = [{"type_pointer": 7}]
        package.package = b"fixture"
        package.stripe_bases = {0: 0, 1: 0}
        package._type_cache = {}
        with mock.patch.object(
            indexed_static_package.civblp_probe,
            "resolve_type_name",
            return_value="FixtureType",
        ) as resolve:
            self.assertEqual(package.type_name(1), "FixtureType")
            self.assertEqual(package.type_name(1), "FixtureType")
        resolve.assert_called_once_with(7, b"fixture", package.allocations, package.stripe_bases)
        self.assertIsNone(package.type_name(0))
        self.assertIsNone(package.type_name(2))


if __name__ == "__main__":
    unittest.main()
