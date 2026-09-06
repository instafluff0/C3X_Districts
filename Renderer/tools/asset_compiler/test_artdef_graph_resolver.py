from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.artdef_graph_resolver import (
    _resolve_package,
    build_resource_improvement_graphs,
)


def _artdef(collections: str) -> str:
    return textwrap.dedent(
        f"""\
        <AssetObjects..ArtDefSet>
          <m_RootCollections>{collections}</m_RootCollections>
        </AssetObjects..ArtDefSet>
        """
    ).lstrip()


def _root(collection: str, items: str) -> str:
    return f'<Element><m_CollectionName text="{collection}"/>{items}</Element>'


def _item(name: str, values: str = "", children: str = "") -> str:
    return f"""
    <Element>
      <m_Fields><m_Values>{values}</m_Values></m_Fields>
      <m_ChildCollections>{children}</m_ChildCollections>
      <m_Name text="{name}"/>
    </Element>
    """


def _terminal(entry: str, package: str) -> str:
    return f"""
    <Element class="AssetObjects..BLPEntryValue">
      <m_EntryName text="{entry}"/>
      <m_XLPClass text="Landmark"/>
      <m_BLPPackage text="{package}"/>
      <m_LibraryName text="Landmark"/>
      <m_ParamName text="Asset"/>
    </Element>
    """


def _implicit_clutter(name: str) -> str:
    child = _item(
        "clutter",
        f"""
        <Element><m_StringValue text="{name}"/><m_ParamName text="XrefName"/></Element>
        <Element><m_ElementName text=""/><m_RootCollectionName text=""/>
          <m_ArtDefPath text=""/><m_ParamName text="Xref"/></Element>
        """,
    )
    return f'<Element><m_CollectionName text="Clutter"/>{child}</Element>'


def _improvement_reference(name: str) -> str:
    return f"""
    <Element>
      <m_ElementName text="{name}"/>
      <m_RootCollectionName text="Improvement"/>
      <m_ArtDefPath text="Improvements.artdef"/>
      <m_ParamName text="Improvement"/>
    </Element>
    """


class ArtDefGraphResolverTests(unittest.TestCase):
    def test_resolves_forward_xrefs_incoming_specialized_graphs_and_package_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            assets = Path(temporary) / "Assets"
            base_artdefs = assets / "Base" / "ArtDefs"
            dlc_artdefs = assets / "DLC" / "Pack" / "ArtDefs"
            base_blps = assets / "Base" / "Platforms" / "Windows" / "BLPs" / "environment"
            dlc_blps = assets / "DLC" / "Pack" / "platforms" / "windows" / "BLPs" / "environment"
            for path in (base_artdefs, dlc_artdefs, base_blps, dlc_blps):
                path.mkdir(parents=True)

            base_artdefs.joinpath("Resources.artdef").write_text(
                _artdef(_root("Resource", _item("RESOURCE_TEST", children=_implicit_clutter("CLUTTER_TEST")))),
                encoding="utf-8",
            )
            dlc_artdefs.joinpath("Resources_Shared.artdef").write_text(
                _artdef(_root("Resource", _item("RESOURCE_TEST", children=_implicit_clutter("CLUTTER_TEST")))),
                encoding="utf-8",
            )
            base_artdefs.joinpath("Clutter.artdef").write_text(
                _artdef(_root("ClutterSets", _item("CLUTTER_TEST", _terminal("ASSET_TEST", "environment/clutter")))),
                encoding="utf-8",
            )
            dlc_artdefs.joinpath("Clutter.artdef").write_text(
                _artdef(_root("ClutterSets", _item("CLUTTER_TEST", _terminal("ASSET_TEST", "environment/clutter")))),
                encoding="utf-8",
            )
            base_artdefs.joinpath("Improvements.artdef").write_text(
                _artdef(_root("Improvement", _item("IMPROVEMENT_FARM"))),
                encoding="utf-8",
            )
            base_artdefs.joinpath("Farms.artdef").write_text(
                _artdef(
                    _root(
                        "Farms",
                        _item(
                            "FARM_TEST",
                            _improvement_reference("IMPROVEMENT_FARM")
                            + _terminal("FARM_ASSET", "landmarks/farms"),
                        ),
                    )
                ),
                encoding="utf-8",
            )

            base_blps.joinpath("clutter.blp").write_bytes(b"CIVBLP\0ASSET_TEST\0")
            # A local partial package must not hide the inherited Base entry.
            dlc_blps.joinpath("clutter.blp").write_bytes(b"CIVBLP\0OTHER_ENTRY\0")
            farms = base_blps.parent / "landmarks" / "farms.blp"
            farms.parent.mkdir(parents=True)
            farms.write_bytes(b"CIVBLP\0FARM_ASSET\0")

            mapping = Path(temporary) / "mapping.json"
            mapping.write_text(
                json.dumps(
                    {
                        "schema": "c3x.civ3_to_civ6_resource_mapping.v0",
                        "mappings": [
                            {
                                "civ3_id": "GOOD_TEST",
                                "target_kind": "resource",
                                "civ6_artdef": "RESOURCE_TEST",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = build_resource_improvement_graphs(assets, mapping)

        self.assertEqual(1, report["summary"]["resource_graphs"])
        self.assertEqual(1, report["summary"]["improvement_graphs"])
        self.assertEqual(0, report["summary"]["unresolved_visual_edges"])
        self.assertEqual(0, report["summary"]["unresolved_visual_terminals"])
        resource = next(graph for graph in report["graphs"] if graph["graph_id"] == "resource/test")
        self.assertEqual(2, len(resource["definitions"]))
        self.assertIn("ClutterSets", {node["root_collection"] for node in resource["nodes"]})
        fallback = next(
            terminal
            for terminal in resource["terminals"]
            if terminal["source"].startswith("DLC/Pack/")
        )
        self.assertEqual("base_fallback", fallback["package_resolution"])
        improvement = next(
            graph for graph in report["graphs"] if graph["graph_id"] == "improvement/farm"
        )
        self.assertEqual(1, len(improvement["associated_visual_roots"]))
        self.assertIn("Farms", {node["root_collection"] for node in improvement["nodes"]})

    def test_package_resolution_fails_closed_when_entry_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "missing.blp"
            package.write_bytes(b"CIVBLP\0OTHER\0")
            result = _resolve_package(
                [
                    {
                        "path": package,
                        "relative": "Base/Platforms/Windows/BLPs/missing.blp",
                        "logical": "missing",
                        "content_root": "Base",
                    }
                ],
                "missing",
                "Base",
                "EXPECTED",
                {},
            )
        self.assertEqual("missing_entry", result["status"])


if __name__ == "__main__":
    unittest.main()
