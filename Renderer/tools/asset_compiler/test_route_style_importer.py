from __future__ import annotations

import json
import struct
import tempfile
import textwrap
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.route_style_importer import (
    _normalized_uv,
    decode_route_material_slots,
    load_mapping,
    read_route_style,
)


def _artdef(body: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<AssetObjects..ArtDefSet><m_RootCollections>"
        + body
        + "</m_RootCollections></AssetObjects..ArtDefSet>"
    )


def _piece_collection(name: str, material: str, start: tuple[int, int]) -> str:
    return f"""
    <Element><m_CollectionName text="RoutePieces"/>
      <Element><m_Fields><m_Values>
        <Element><m_x>{start[0]}</m_x><m_y>{start[1]}</m_y><m_ParamName text="DefaultUV_X1Y1"/></Element>
        <Element><m_x>63</m_x><m_y>63</m_y><m_ParamName text="DefaultUV_X2Y2"/></Element>
        <Element><m_EntryName text="{material}"/><m_ParamName text="DefaultMaterial"/></Element>
      </m_Values></m_Fields><m_ChildCollections/><m_Name text="{name}"/></Element>
    </Element>
    """


class RouteStyleImporterTests(unittest.TestCase):
    def test_material_slots_require_base_and_height_but_allow_missing_specular(self) -> None:
        raw = bytearray(96)
        struct.pack_into("<4I", raw, 0x48, 0, 1, 2, 3)
        textures = {
            0: {"index": 0, "name": "base", "class": "Decal_BaseColor"},
            1: {"index": 1, "name": "height", "class": "Decal_Heightmap"},
            2: {"index": 2, "name": "placeholder", "class": "Decal_FOWColor"},
            3: {"index": 3, "name": "fog", "class": "Decal_FOWColor"},
        }
        channels, evidence = decode_route_material_slots(bytes(raw), textures.__getitem__)
        self.assertEqual({"base_color", "height", "fog_color"}, set(channels))
        self.assertEqual("class_mismatch", evidence["specular"]["status"])
        struct.pack_into("<I", raw, 0x4C, 2)
        with self.assertRaisesRegex(ValueError, "height class"):
            decode_route_material_slots(bytes(raw), textures.__getitem__)

    def test_uv_normalization_preserves_source_direction(self) -> None:
        self.assertEqual([0.61176471, 0.74901961], _normalized_uv([156, 191], (256, 256)))
        self.assertEqual([0.00392157, 0.50588235], _normalized_uv([1, 129], (256, 256)))
        with self.assertRaisesRegex(ValueError, "outside"):
            _normalized_uv([256, 0], (256, 256))

    def test_artdef_merge_resolves_base_piece_in_expansion_style(self) -> None:
        base = _artdef(_piece_collection("BASE_FADE", "BASE_MATERIAL", (4, 4)))
        expansion = _artdef(
            _piece_collection("RAIL_FADE", "RAIL_MATERIAL", (8, 8))
            + """
            <Element><m_CollectionName text="RouteTypes"/>
              <Element><m_Fields><m_Values><Element><m_nValue>5</m_nValue><m_ParamName text="Priority"/></Element></m_Values></m_Fields><m_ChildCollections/><m_Name text="RType_Rail"/></Element>
            </Element>
            <Element><m_CollectionName text="GameCoreRouteTranslator"/>
              <Element><m_Fields><m_Values><Element><m_ElementName text="RType_Rail"/><m_ParamName text="RouteSystem Type"/></Element></m_Values></m_Fields><m_ChildCollections/><m_Name text="ROUTE_RAIL"/></Element>
            </Element>
            <Element><m_CollectionName text="Route Descriptions"/>
              <Element><m_Fields><m_Values>
                <Element><m_bValue>true</m_bValue><m_ParamName text="TileUVs"/></Element>
                <Element><m_ElementName text="RType_Rail"/><m_ParamName text="Type"/></Element>
                <Element><m_fValue>6</m_fValue><m_ParamName text="Width"/></Element>
                <Element><m_fValue>4</m_fValue><m_ParamName text="BlockerWidth"/></Element>
              </m_Values></m_Fields><m_ChildCollections>
                <Element><m_CollectionName text="Route Segments"/>
                  <Element><m_Fields><m_Values>
                    <Element><m_Value text="FADEOUT"/><m_ParamName text="Segment Type"/></Element>
                    <Element><m_Value text="NORMAL"/><m_ParamName text="State"/></Element>
                  </m_Values></m_Fields><m_ChildCollections>
                    <Element><m_CollectionName text="Layers"/>
                      <Element><m_Fields><m_Values><Element><m_fValue>5</m_fValue><m_ParamName text="Height"/></Element><Element><m_ElementName text="BASE_FADE"/><m_ParamName text="RoutePiece"/></Element></m_Values></m_Fields><m_ChildCollections/><m_Name text="base"/></Element>
                      <Element><m_Fields><m_Values><Element><m_fValue>17</m_fValue><m_ParamName text="Height"/></Element><Element><m_ElementName text="RAIL_FADE"/><m_ParamName text="RoutePiece"/></Element></m_Values></m_Fields><m_ChildCollections/><m_Name text="rail"/></Element>
                    </Element>
                  </m_ChildCollections><m_Name text="fade"/></Element>
                </Element>
              </m_ChildCollections><m_Name text="RType_Rail"/></Element>
            </Element>
            """
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "base.artdef"
            expansion_path = root / "expansion.artdef"
            base_path.write_text(textwrap.dedent(base), encoding="utf-8")
            expansion_path.write_text(textwrap.dedent(expansion), encoding="utf-8")
            style = read_route_style([base_path, expansion_path], "ROUTE_RAIL")
        self.assertEqual(["BASE_FADE", "RAIL_FADE"], [item["source_piece"] for item in style["pieces"]])
        self.assertEqual(2, len(style["segments"][0]["layers"]))

    def test_mapping_rejects_duplicate_runtime_ids(self) -> None:
        mapping = {
            "schema": "c3x.source_route_style_mapping.v0",
            "catalogs": [{
                "artdefs": ["Base/routes.artdef"],
                "material_sources": [{"source_package": "Base/routes.blp"}],
                "source_units_per_tile": 100,
                "styles": [
                    {"source_route": "A", "asset_id": "route/test", "route_kind": "road", "style_stage": "a"},
                    {"source_route": "B", "asset_id": "route/test", "route_kind": "road", "style_stage": "b"},
                ],
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate normalized"):
                load_mapping(path)


if __name__ == "__main__":
    unittest.main()
