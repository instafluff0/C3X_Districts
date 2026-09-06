import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.dune_source_probe import (
    SAND_ASSETS,
    parse_dune_style,
    parse_sand_placements,
)


def field(kind: str, parameter: str, value_tag: str, value: str) -> str:
    attribute = f' text="{value}"' if value_tag in ("m_EntryName", "m_Value") else ""
    body = "" if attribute else value
    return (
        f'<Element class="AssetObjects..{kind}"><{value_tag}{attribute}>{body}</{value_tag}>'
        f'<m_ParamName text="{parameter}"/></Element>'
    )


class DuneSourceProbeTests(unittest.TestCase):
    def test_artdef_parsers_select_dune_controls_and_five_sand_decals(self) -> None:
        dune_fields = "".join((
            field("BLPEntryValue", "DesertHillsMtl", "m_EntryName", "ART_DEF_TERRAIN_MATERIAL_DESERT_HILLS"),
            field("FloatValue", "DuneBase", "m_fValue", "0.0"),
            field("FloatValue", "DuneHeight", "m_fValue", "4.0"),
            field("FloatValue", "DuneWidth", "m_fValue", "4.0"),
            field("FloatValue", "DuneNoise", "m_fValue", "0.6"),
            field("FloatValue", "DuneAngle", "m_fValue", "0.300001"),
        ))
        terrain_xml = (
            "<AssetObjects..ArtDefSet>"
            '<Element><m_CollectionName text="DuneDesertHills"/>'
            f'<Element><m_Fields><m_Values>{dune_fields}</m_Values></m_Fields>'
            '<m_Name text="Default"/></Element></Element>'
            "</AssetObjects..ArtDefSet>"
        )
        placements = []
        for index, asset in enumerate(SAND_ASSETS):
            values = "".join((
                field("BLPEntryValue", "Asset", "m_EntryName", asset),
                field("FloatValue", "Scale", "m_fValue", "5.5"),
                field("IntValue", "Count", "m_nValue", "3"),
                field("FloatValue", "ScaleVariation", "m_fValue", "0.15"),
                field("BoolValue", "ShowDecal", "m_bValue", "true"),
                field("IntValue", "Priority", "m_nValue", "3"),
                field("StringValue", "RotateMode", "m_Value", "RotateZ"),
                field("BoolValue", "AllowOverlap", "m_bValue", "true"),
            ))
            placements.append(
                f'<Element><m_Fields><m_Values>{values}</m_Values></m_Fields>'
                f'<m_Name text="Desert Sand {index}"/></Element>'
            )
        clutter_xml = (
            "<AssetObjects..ArtDefSet><Element>"
            + "".join(placements)
            + '<m_Name text="CLUTTER_DESERT"/></Element></AssetObjects..ArtDefSet>'
        )
        with tempfile.TemporaryDirectory() as temporary:
            terrain_path = Path(temporary) / "TerrainStyle.artdef"
            clutter_path = Path(temporary) / "Clutter.artdef"
            terrain_path.write_text(terrain_xml, encoding="utf-8")
            clutter_path.write_text(clutter_xml, encoding="utf-8")
            style = parse_dune_style(terrain_path)
            sand = parse_sand_placements(clutter_path)

        self.assertEqual("ART_DEF_TERRAIN_MATERIAL_DESERT_HILLS", style["material"])
        self.assertEqual(4.0, style["parameters"]["DuneHeight"])
        self.assertEqual(list(SAND_ASSETS), [entry["asset"] for entry in sand])
        self.assertTrue(all(entry["show_decal"] for entry in sand))


if __name__ == "__main__":
    unittest.main()
