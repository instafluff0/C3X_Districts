from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.dedicated_object_pipelines import (
    DEFAULT_CITY_CONTRACT,
    DEFAULT_UNIT_CONTRACT,
    _city_axis_evidence,
    _unit_target_graph,
    load_contract,
    validate_separation,
)


class DedicatedObjectPipelineTests(unittest.TestCase):
    def test_checked_contracts_are_separate_and_category_specific(self) -> None:
        city = load_contract(DEFAULT_CITY_CONTRACT, "city")
        unit = load_contract(DEFAULT_UNIT_CONTRACT, "unit")
        validate_separation(city, unit)
        self.assertEqual("generated_compound_city", city["composition_mode"])
        self.assertEqual("animated_formation", unit["composition_mode"])
        self.assertIn("compound_landmark_importer", {stage["adapter"] for stage in city["stages"]})
        self.assertIn("normalized_animation", {stage["adapter"] for stage in unit["stages"]})
        collision = copy.deepcopy(unit)
        collision["runtime_namespace"] = city["runtime_namespace"]
        with self.assertRaisesRegex(ValueError, "separate normalized namespaces"):
            validate_separation(city, collision)

    def test_city_evidence_preserves_civilization_era_and_growth_axes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artdefs = root / "Base" / "ArtDefs"
            artdefs.mkdir(parents=True)
            artdefs.joinpath("CityGenerators.artdef").write_text(
                """
                <AssetObjects..ArtDefSet><m_RootCollections><Element>
                  <m_CollectionName text="Generator"/><Element><m_Fields><m_Values>
                    <Element><m_ElementName text="ARTERA_ANCIENT"/><m_RootCollectionName text="ArtEra"/><m_ParamName text="Era"/></Element>
                  </m_Values></m_Fields><m_ChildCollections><Element><m_CollectionName text="GrowthStage"/>
                    <Element><m_Fields><m_Values><Element><m_nValue>7</m_nValue><m_ParamName text="Var_Population"/></Element></m_Values></m_Fields><m_ChildCollections/><m_Name text="7"/></Element>
                  </Element></m_ChildCollections><m_Name text="City"/></Element>
                </Element></m_RootCollections></AssetObjects..ArtDefSet>
                """,
                encoding="utf-8",
            )
            artdefs.joinpath("Cultures.artdef").write_text(
                """
                <AssetObjects..ArtDefSet><m_RootCollections><Element>
                  <m_CollectionName text="Culture"/><Element><m_Fields><m_Values><Element><m_Values>
                    <Element><m_ElementName text="CIVILIZATION_TEST"/><m_RootCollectionName text="Civilization"/><m_ParamName text="Civilizations"/></Element>
                  </m_Values><m_ParamName text="Civilizations"/></Element></m_Values></m_Fields><m_ChildCollections/><m_Name text="TestCulture"/></Element>
                </Element></m_RootCollections></AssetObjects..ArtDefSet>
                """,
                encoding="utf-8",
            )
            evidence = _city_axis_evidence(root)
        self.assertEqual([7], evidence["growth_populations"])
        self.assertEqual(["ARTERA_ANCIENT"], evidence["art_eras"])
        self.assertEqual(["CIVILIZATION_TEST"], evidence["civilizations"])
        self.assertEqual(1, evidence["culture_profiles"])

    def test_unit_graph_traverses_only_typed_unit_dependencies(self) -> None:
        unit_id = "Base/ArtDefs/Units.artdef#Units/UNIT_TEST"
        member_id = "Base/ArtDefs/Units.artdef#UnitMemberTypes/TestMember"
        external_id = "Base/ArtDefs/VFX.artdef#MaterialTypes/MEAT"
        reference = {
            "collection_path": ["Members"],
            "parameter": "Type",
            "target_name": "TestMember",
            "target_root": "UnitMemberTypes",
            "target_path": "Units.artdef",
            "template": "Units",
        }
        external = {
            "collection_path": [],
            "parameter": "VFXMaterialType",
            "target_name": "MEAT",
            "target_root": "MaterialTypes",
            "target_path": "VFX.artdef",
            "template": "VFX",
        }
        index = {
            "nodes": {
                unit_id: {
                    "id": unit_id,
                    "path": "Base/ArtDefs/Units.artdef",
                    "document": "Units.artdef",
                    "content_root": "Base",
                    "root_collection": "Units",
                    "name": "UNIT_TEST",
                    "references": [reference],
                    "terminals": [],
                },
                member_id: {
                    "id": member_id,
                    "path": "Base/ArtDefs/Units.artdef",
                    "document": "Units.artdef",
                    "content_root": "Base",
                    "root_collection": "UnitMemberTypes",
                    "name": "TestMember",
                    "references": [external],
                    "terminals": [],
                },
                external_id: {
                    "id": external_id,
                    "path": "Base/ArtDefs/VFX.artdef",
                    "document": "VFX.artdef",
                    "content_root": "Base",
                    "root_collection": "MaterialTypes",
                    "name": "MEAT",
                    "references": [],
                    "terminals": [],
                },
            },
            "by_root_name": {
                ("Units", "UNIT_TEST"): [unit_id],
                ("UnitMemberTypes", "TestMember"): [member_id],
                ("MaterialTypes", "MEAT"): [external_id],
            },
        }
        graph = _unit_target_graph(index, "UNIT_TEST")
        self.assertEqual({unit_id, member_id}, set(graph["nodes"]))
        self.assertEqual(1, len(graph["edges"]))
        self.assertFalse(graph["unresolved_internal_edges"])


if __name__ == "__main__":
    unittest.main()
