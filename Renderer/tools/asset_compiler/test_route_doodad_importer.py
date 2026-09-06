from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from Renderer.tools.asset_compiler.route_doodad_importer import (
    load_mapping,
    read_transition_records,
)


def _transition_artdef() -> str:
    return """<?xml version="1.0" encoding="utf-8"?>
<AssetObjects..ArtDefSet>
  <m_RootCollections>
    <Element>
      <m_CollectionName text="Route Transition Doodads"/>
      <Element>
        <m_Fields><m_Values>
          <Element><m_ElementName text="RType_Railroad"/><m_ParamName text="Origin"/></Element>
          <Element><m_ElementName text="RType_Ancient"/><m_ParamName text="Destination"/></Element>
          <Element><m_Value text="BRIDGE"/><m_ParamName text="Type"/></Element>
          <Element><m_EntryName text="RR_Bridge"/><m_ParamName text="Asset"/></Element>
          <Element><m_fValue>100.0</m_fValue><m_ParamName text="TransitionLength"/></Element>
          <Element><m_bValue>true</m_bValue><m_ParamName text="ContourToRoad"/></Element>
          <Element><m_bValue>false</m_bValue><m_ParamName text="ScaleToGap"/></Element>
        </m_Values></m_Fields>
        <m_Name text="RR_ANC"/>
      </Element>
    </Element>
  </m_RootCollections>
</AssetObjects..ArtDefSet>
"""


class RouteDoodadImporterTests(unittest.TestCase):
    def test_transition_normalizes_source_units_and_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "WorldViewRoutes.artdef"
            path.write_text(_transition_artdef(), encoding="utf-8")
            runtime, evidence = read_transition_records(
                path,
                {
                    "RType_Railroad": "route/railroad/default",
                    "RType_Ancient": "route/road/ancient",
                },
                {"RR_Bridge": "route/bridge/railroad"},
                100.0,
            )
        self.assertEqual(
            {
                "origin_style": "route/railroad/default",
                "destination_style": "route/road/ancient",
                "kind": "bridge",
                "asset": "route/bridge/railroad",
                "length_tiles": 1.0,
                "contour_to_route": True,
                "scale_to_gap": False,
            },
            runtime[0],
        )
        self.assertEqual("RR_ANC", evidence[0]["source_record"])

    def test_transition_rejects_unmapped_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "WorldViewRoutes.artdef"
            path.write_text(_transition_artdef(), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unmapped route or body"):
                read_transition_records(
                    path,
                    {
                        "RType_Railroad": "route/railroad/default",
                        "RType_Ancient": "route/road/ancient",
                    },
                    {},
                    100.0,
                )

    def test_mapping_requires_explicit_header_exception(self) -> None:
        mapping = {
            "schema": "c3x.source_route_doodad_mapping.v0",
            "route_types": {"RType_Ancient": "route/road/ancient"},
            "catalogs": [
                {
                    "artdef": "Base/ArtDefs/WorldViewRoutes.artdef",
                    "source_package": "Base/route_doodads.blp",
                    "shared_data": ["Base/SHARED_DATA"],
                    "source_units_per_tile": 100.0,
                    "assets": [
                        {"source_entry": "MED_Bridge", "asset_id": "route/bridge/medieval"}
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mapping.json"
            path.write_text(json.dumps(mapping), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "explicit Boolean"):
                load_mapping(path)


if __name__ == "__main__":
    unittest.main()
