from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from Renderer.inventory import civ3_art_inventory


def write_pcx(path: Path, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(128)
    header[0] = 0x0A
    header[1] = 5
    header[2] = 1
    header[3] = 8
    struct.pack_into("<4H", header, 4, 0, 0, width - 1, height - 1)
    header[65] = 1
    path.write_bytes(header)


def write_flc(path: Path, width: int = 240, height: int = 240, frames: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray(128)
    struct.pack_into("<I5H", header, 0, len(header), 0xAF12, frames, width, height, 8)
    struct.pack_into("<2H", header, 96, 8, frames // 8)
    path.write_bytes(header)


class Civ3ArtInventoryTests(unittest.TestCase):
    def test_layering_fog_units_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp)
            write_pcx(install / "Art" / "Terrain" / "FogOfWar.pcx", 256, 128)
            write_pcx(install / "Art" / "Terrain" / "roads.pcx", 128, 64)
            write_pcx(install / "Art" / "resources.pcx", 640, 320)
            write_pcx(install / "Conquests" / "Art" / "Terrain" / "roads.pcx", 256, 128)
            unit_dir = install / "Conquests" / "Art" / "Units" / "Test Unit"
            write_flc(unit_dir / "TestRun.flc")
            (unit_dir / "Test Unit.ini").write_text(
                "[Animations]\nDEFAULT=TestRun.flc\nRUN=TestRun.flc\n", encoding="latin-1"
            )

            first = civ3_art_inventory.build_inventory(install)
            second = civ3_art_inventory.build_inventory(install)

        self.assertEqual(civ3_art_inventory.canonical_json(first), civ3_art_inventory.canonical_json(second))
        effective = [item for item in first["file_assets"] if item["effective"]]
        fog = next(item for item in effective if item["asset_path"].casefold().endswith("fogofwar.pcx"))
        roads = next(item for item in effective if item["asset_path"].casefold().endswith("roads.pcx"))
        self.assertEqual(fog["render_layer"], "fog_of_war")
        self.assertEqual(roads["source_layer"], "conquests")
        self.assertEqual(roads["image"]["width"], 256)
        self.assertEqual(first["units"][-1]["animations"][0]["frames"], 8)
        self.assertEqual(first["units"][-1]["animations"][0]["direction_count"], 8)
        self.assertEqual(first["units"][-1]["animations"][0]["frames_per_direction"], 1)
        self.assertEqual(first["units"][-1]["animations"][0]["resolution_status"], "resolved")
        self.assertFalse(first["districts_in_scope"])
        self.assertEqual(first["completeness"]["status"], "incomplete")
        ownership = {layer["id"]: layer["default_ownership"] for layer in first["render_layers"]}
        for retained in ("fog_of_war", "territory_border", "map_grid", "selection_highlight", "city_label", "minimap", "hud"):
            self.assertEqual(ownership[retained], "retained_civ3")

    def test_known_transition_names_are_classified(self) -> None:
        self.assertEqual(civ3_art_inventory.classify_terrain_file("xdgc.pcx"), "terrain_transition")
        self.assertEqual(civ3_art_inventory.classify_terrain_file("lxdgc.pcx"), "terrain_transition")
        self.assertEqual(civ3_art_inventory.classify_terrain_file("wCSO.pcx"), "water")
        self.assertEqual(civ3_art_inventory.classify_terrain_file("lwSSS.pcx"), "water")
        self.assertEqual(civ3_art_inventory.classify_terrain_file("FogOfWar.pcx"), "fog_of_war")

    def test_unknown_terrain_file_remains_visible_as_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp)
            write_pcx(install / "Art" / "Terrain" / "mystery.pcx", 32, 32)
            inventory = civ3_art_inventory.build_inventory(install)
        self.assertEqual(inventory["summary"]["unclassified_effective_files"], 1)
        self.assertEqual(inventory["completeness"]["unclassified_effective_files"], ["Art/Terrain/mystery.pcx"])

    def test_contract_biq_and_runtime_evidence_close_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            install = Path(tmp)
            write_pcx(install / "Art" / "Terrain" / "FogOfWar.pcx", 256, 128)
            unit_dir = install / "Art" / "Units" / "Test Unit"
            write_flc(unit_dir / "TestRun.flc", frames=16)
            (unit_dir / "Test Unit.ini").write_text(
                "[Animations]\nDEFAULT=TestRun.flc\nRUN=TestRun.flc\n", encoding="latin-1"
            )
            contracts = {
                "contracts": [
                    {
                        "id": "fog_test",
                        "basenames": ["FogOfWar.pcx"],
                        "groups": [
                            {
                                "name": "fog",
                                "origin": [0, 0],
                                "cell": [128, 64],
                                "stride": [128, 64],
                                "columns": 2,
                                "rows": 2,
                            }
                        ],
                        "authored_capacity": 4,
                        "reachable": {"mode": "all_contract_cells"},
                    }
                ]
            }
            semantics = {
                "terrain_types": [],
                "resources": [],
                "unit_types": [
                    {
                        "civilopedia_entry": "PRTO_Test",
                        "art_variants": [
                            {"key": "ANIMNAME_PRTO_Test", "art_folder": "Test Unit", "source_layer": "base"}
                        ],
                    }
                ],
            }
            census = {"unknown_selectors": []}
            inventory = civ3_art_inventory.build_inventory(
                install,
                atlas_contracts=contracts,
                biq_semantics=semantics,
                runtime_census=census,
            )

        self.assertEqual(inventory["completeness"]["status"], "complete")
        atlas = next(item["atlas"] for item in inventory["file_assets"] if item["effective"])
        self.assertEqual(atlas["authored_capacity"], 4)
        self.assertEqual(atlas["reachable_indices"], [0, 1, 2, 3])
        self.assertEqual(inventory["units"][0]["semantic_binding"]["unit_type_ids"], ["PRTO_Test"])
        correlated = inventory["semantic_inventory"]["unit_types"][0]["art_correlation"][0]
        self.assertEqual(correlated["actions"][0]["direction_order"][2], "southeast")

    def test_bundled_contracts_cover_every_stock_atlas_basename(self) -> None:
        contracts = civ3_art_inventory.load_json(civ3_art_inventory.DEFAULT_ATLAS_CONTRACTS)
        self.assertIsNotNone(contracts)
        by_name = civ3_art_inventory.contract_by_basename(contracts)
        expected = {name.casefold() for name in civ3_art_inventory.TERRAIN_FILE_LAYERS}
        expected.update(
            {
                "xtgc.pcx", "xpgc.pcx", "xdgc.pcx", "xdpc.pcx", "xdgp.pcx", "xggc.pcx",
                "wcso.pcx", "wsss.pcx", "wooo.pcx", "lxtgc.pcx", "lxpgc.pcx", "lxdgc.pcx",
                "lxdpc.pcx", "lxdgp.pcx", "lxggc.pcx", "lwcso.pcx", "lwsss.pcx", "lwooo.pcx",
                "resources.pcx", "resources_shadows.pcx", "airandharb.pcx", "barracks.pcx",
                "city icons.pcx", "destroy.pcx", "ramer.pcx", "amerwall.pcx",
            }
        )
        self.assertTrue(expected <= set(by_name))
        self.assertEqual(len(by_name), 76)

    def test_bundled_biq_snapshot_is_primary_record_complete(self) -> None:
        snapshot = civ3_art_inventory.load_json(civ3_art_inventory.DEFAULT_BIQ_SEMANTICS)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["counts"]["primary_unit_types"], 124)
        self.assertEqual(snapshot["counts"]["resources"], 26)
        self.assertEqual(snapshot["counts"]["terrain_types"], 14)
        self.assertEqual(len(snapshot["unit_types"]), 124)
        self.assertTrue(all(unit["art_variants"] for unit in snapshot["unit_types"]))
        self.assertNotIn("C:\\", civ3_art_inventory.canonical_json(snapshot))


if __name__ == "__main__":
    unittest.main()
