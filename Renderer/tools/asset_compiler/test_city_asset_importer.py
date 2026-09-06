from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.city_asset_importer import (
    DEFAULT_STRATEGY,
    _component_id,
    build_candidate_pools,
    load_strategy,
    parse_city_generator_blocks,
)


class CityAssetImporterTests(unittest.TestCase):
    def test_city_generator_block_parser_preserves_three_typed_fields(self) -> None:
        document = """
        <AssetObjects..ArtDefSet><m_RootCollections><Element>
          <m_CollectionName text="GeneratorBlockList"/><Element><m_Fields><m_Values/></m_Fields>
          <m_ChildCollections><Element><m_CollectionName text="Block"/>
            <Element><m_Fields><m_Values>
              <Element><m_ElementName text="TestCulture"/><m_ParamName text="Tag_Culture"/></Element>
              <Element><m_ElementName text="ARTERA_TEST"/><m_ParamName text="Tag_Era"/></Element>
              <Element><m_EntryName text="Building_A"/><m_XLPClass text="CityBuildings"/><m_BLPPackage text="landmarks/cities"/><m_ParamName text="Asset_CityBlock"/></Element>
            </m_Values></m_Fields><m_ChildCollections/><m_Name text="Block1"/></Element>
          </Element></m_ChildCollections><m_Name text="TestList"/></Element></Element>
        </m_RootCollections></AssetObjects..ArtDefSet>
        """
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "CityGenerators.artdef"
            path.write_text(document, encoding="utf-8")
            records = parse_city_generator_blocks(path, Path("Base/ArtDefs/CityGenerators.artdef"))
        self.assertEqual(1, len(records))
        self.assertEqual("TestCulture", records[0]["source_culture"])
        self.assertEqual("ARTERA_TEST", records[0]["source_art_era"])
        self.assertEqual("Building_A", records[0]["entry"])
        self.assertEqual("landmarks/cities", records[0]["package"])

    def test_checked_strategy_covers_five_cultures_four_eras_and_three_sizes(self) -> None:
        strategy = load_strategy(DEFAULT_STRATEGY)
        self.assertEqual(set(range(5)), {item["civ3_culture_group"] for item in strategy["styles"]})
        self.assertEqual(set(range(4)), {item["civ3_era"] for item in strategy["eras"]})
        self.assertEqual(["town", "city", "metropolis"], [item["id"] for item in strategy["runtime"]["size_recipes"]])
        broken = copy.deepcopy(strategy)
        broken["styles"][0]["source_culture_by_era"].pop("modern")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "strategy.json"
            import json

            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete source culture"):
                load_strategy(path)

    def test_candidate_pools_are_deduplicated_and_complete(self) -> None:
        strategy = load_strategy(DEFAULT_STRATEGY)
        blocks = []
        for style in strategy["styles"]:
            for era in strategy["eras"]:
                culture = style["source_culture_by_era"][era["id"]]
                for index in range(4):
                    blocks.append(
                        {
                            "source_culture": culture,
                            "source_art_era": era["source_art_era"],
                            "package_path": f"Base/package_{culture}.blp",
                            "entry": f"Component_{culture}_{era['id']}_{index}",
                        }
                    )
                blocks.append(dict(blocks[-1]))
        pools = build_candidate_pools(blocks, strategy)
        self.assertEqual(20, len(pools))
        self.assertTrue(all(len(pool["candidates"]) >= 4 for pool in pools))
        self.assertEqual(20, len({pool["id"] for pool in pools}))

    def test_component_id_hides_source_names_but_is_stable(self) -> None:
        first = _component_id("Base/secret.blp", "SourceBuilding")
        self.assertEqual(first, _component_id("Base/secret.blp", "SourceBuilding"))
        self.assertRegex(first, r"^city/component/[0-9a-f]{16}$")
        self.assertNotIn("SourceBuilding", first)


if __name__ == "__main__":
    unittest.main()
