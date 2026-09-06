#!/usr/bin/env python3
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import civ6_lighting_probe as probe


ARTDEF = """<?xml version="1.0" encoding="UTF-8"?>
<AssetObjects..ArtDefSet><m_RootCollections><Element>
<m_CollectionName text="TimeOfDay"/><Element>
<m_ChildCollections><Element><m_CollectionName text="CubeLights"/><Element>
<m_Fields><m_Values><Element class="AssetObjects..BLPEntryValue">
<m_EntryName text="Night_LightRig"/><m_XLPClass text="GameLighting"/>
<m_XLPPath text="default_lighting.xlp"/><m_BLPPackage text="lighting/default_lighting"/>
<m_LibraryName text="GameLighting"/><m_ParamName text="Light"/>
</Element></m_Values></m_Fields>
<m_ChildCollections><Element><m_CollectionName text="WeightCurve"/>
<Element><m_Fields><m_Values>
<Element class="AssetObjects..FloatValue"><m_fValue>23.0</m_fValue><m_ParamName text="Time"/></Element>
<Element class="AssetObjects..FloatValue"><m_fValue>1.0</m_fValue><m_ParamName text="Weight"/></Element>
</m_Values></m_Fields></Element></Element></m_ChildCollections>
<m_Name text="Night"/></Element></Element></m_ChildCollections>
<m_Name text="DEFAULT_LIGHTING"/></Element></Element></m_RootCollections></AssetObjects..ArtDefSet>
"""


class Civ6LightingProbeTests(unittest.TestCase):
    def make_assets(self, root: Path) -> None:
        artdef = root / probe.GAME_LIGHTING_ARTDEF
        artdef.parent.mkdir(parents=True)
        artdef.write_text(ARTDEF, encoding="utf-8")
        payloads = {
            probe.PRIMARY_PACKAGES[0]: b"Night_LightRig\x00m_vSunDirection\x00",
            probe.PRIMARY_PACKAGES[1]: b"DL_OrangeGlow\x00ApplyLightMapWeight\x00",
            probe.PRIMARY_PACKAGES[2]: b"FX_Light_Flicker\x00ChimneySmoke\x00",
            probe.ATTACHMENT_PACKAGES[0]: b"FX_ChimneySmoke001\x00",
        }
        for relative, content in payloads.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        texture = root / "Base/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_Fire_Glow"
        texture.parent.mkdir(parents=True, exist_ok=True)
        texture.write_bytes(b"payload")
        water_artdef = """<AssetObjects..ArtDefSet><m_RootCollections><Element><m_Fields><m_Values>
        <Element class="AssetObjects..BLPEntryValue"><m_EntryName text="Water/RiverSource"/>
        <m_XLPClass text="Water"/><m_XLPPath text="water.xlp"/><m_BLPPackage text="Water"/>
        <m_LibraryName text="Water"/><m_ParamName text="Material"/></Element>
        </m_Values></m_Fields></Element></m_RootCollections></AssetObjects..ArtDefSet>"""
        for relative in probe.WATER_ARTDEFS:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(water_artdef, encoding="utf-8")

    def test_report_follows_artdef_and_finds_asset_backed_resources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_assets(root)
            report = probe.build_report(root)

        self.assertEqual(report["schema"], "c3x.civ6_lighting_probe.v0")
        self.assertEqual(report["global_lighting"]["profiles"], ["DEFAULT_LIGHTING"])
        self.assertEqual(report["global_lighting"]["rigs"][0]["profile"], "DEFAULT_LIGHTING")
        self.assertEqual(report["global_lighting"]["rigs"][0]["phase"], "Night")
        self.assertEqual(
            report["global_lighting"]["rigs"][0]["weight_curve"],
            [{"time": 23.0, "weight": 1.0}],
        )
        self.assertIn(
            "Base/Platforms/Windows/BLPs/Light.blp",
            report["all_named_lighting_packages"],
        )
        self.assertIn(
            "Base/Platforms/Windows/BLPs/SHARED_DATA/TEXTURE_Fire_Glow",
            report["shared_effect_texture_candidates"],
        )
        self.assertIn(
            "DL_OrangeGlow",
            report["primary_package_evidence"][1]["matching_strings"],
        )
        self.assertEqual("Water", report["water_artdef_evidence"][0]["bindings"][0]["xlp_class"])
        self.assertEqual(
            "inferred resource-to-model relationship from repeated landmark package names",
            report["supported_vertical_slice"]["model_attachment"]["evidence"],
        )
        self.assertEqual("unresolved", report["supported_vertical_slice"]["typed_parameters"]["evidence"])

    def test_report_is_deterministic_and_contains_no_machine_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_assets(root)
            first = probe.build_report(root)
            second = probe.build_report(root)

        encoded = json.dumps(first, sort_keys=True)
        self.assertEqual(encoded, json.dumps(second, sort_keys=True))
        self.assertNotIn(str(root), encoded)

    def test_missing_artdef_is_an_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "GameLighting ArtDef"):
                probe.build_report(Path(tmp))


if __name__ == "__main__":
    unittest.main()
