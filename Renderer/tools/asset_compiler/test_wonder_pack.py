#!/usr/bin/env python3
"""Contract tests for the WonderRuntime pack (docs/constructed_wonder_rendering.md).

The built pack (Renderer/packs/WonderRuntime) is checked against the contract:
the resident texture budget, the index, the group libraries, the authority format,
season masks, the night-light lamp modes and the absence of source identifiers.
The pure helpers (texture caps, lamp modes) are checked on synthetic data.
"""
from __future__ import annotations

import json
import re
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from Renderer.lab.studies.city_readability import runtime_pack as rp
from Renderer.tools.asset_compiler import build_resource_compositions as rc
from Renderer.tools.asset_compiler import build_wonder_pack as wp

PACK = wp.OUTPUT
AUTHORITY = re.compile(r"^(wonder|district)\|[^|]+\|(complete|build:0\.\d+)(\+alt)?\|(0|2|5|8)\|r?$")
SOURCE_WORDS = re.compile(r"civ6|civ ?vi|firaxis|blp|artdef|won_|building_", re.IGNORECASE)


def read_index(data: bytes) -> list[dict]:
    if data[:8] != wp.MAGIC:
        raise ValueError("not a wonder index")
    at = 8

    def take(fmt):
        nonlocal at
        values = struct.unpack_from(fmt, data, at)
        at += struct.calcsize(fmt)
        return values

    def text():
        (length,) = take("<I")
        nonlocal at
        value = data[at:at + length].decode()
        at += length
        return value
    (count,) = take("<I")
    groups = []
    for _ in range(count):
        key, path = text(), text()
        size, textures = take("<II")
        groups.append({"key": key, "path": path, "bytes": size, "textures": [text() for _ in range(textures)]})
    if at != len(data):
        raise ValueError("trailing index bytes")
    return groups


class Helpers(unittest.TestCase):
    def test_capped_drops_top_mips(self):
        # A 1024x512 BC1 texture with its full chain, capped to 256 on its longer side.
        width, height, mips, block = 1024, 512, 11, 8
        layout = rc.mip_offsets(width, height, mips, block)
        payload = bytes((i * 7) & 255 for i in range(layout[-1][0] + layout[-1][1] * layout[-1][2] * block))
        data = rc.dds_header(width, height, mips, 72, layout[0][1] * layout[0][2] * block) + payload
        out = wp.capped(data, 256)
        w, h, m, dxgi, body = rc.dds(self._file(out))
        self.assertEqual((w, h, m, dxgi), (256, 128, 9, 72))
        self.assertEqual(body, payload[layout[2][0]:])
        self.assertIs(wp.capped(data, 2048), data)

    def _file(self, data: bytes) -> Path:
        import tempfile
        handle = tempfile.NamedTemporaryFile(suffix=".dds", delete=False)
        handle.write(data)
        handle.close()
        self.addCleanup(Path(handle.name).unlink)
        return Path(handle.name)

    def test_lamp_modes(self):
        flame = [0.1, 0.2, 0.3, 0.0, 0.12, 0.24, 5.0, 1.3]
        self.assertEqual(wp.lamp(flame), flame)
        steady_red = wp.lamp([0, 0, 0, wp.LAB_LAMP, .12, .12, 2 * 16 + 3, 1.6])
        self.assertEqual(steady_red[3], wp.NIGHT_LIGHT)
        self.assertEqual(int(steady_red[6]) // 64, 1 + 2)
        blinking_warm = wp.lamp([0, 0, 0, wp.LAB_BLINK, .12, .12, 5, 1.6])
        self.assertEqual(int(blinking_warm[6]) // 64, 1 + 0 + 4)
        self.assertLess(blinking_warm[6] % 64, 64)


@unittest.skipUnless((PACK / "index.bin").is_file(), "build the pack first (build_wonder_pack.py)")
class Pack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups = read_index((PACK / "index.bin").read_bytes())
        cls.report = json.loads((PACK / "pack_report.json").read_text())
        cls.libraries = {g["key"]: rp.decode(PACK / g["path"]) for g in cls.groups}

    def test_texture_budget(self):
        total = sum((PACK / "textures" / p.name).stat().st_size - 148 for p in (PACK / "textures").iterdir())
        self.assertLessEqual(total, wp.BUDGET)
        self.assertEqual(total, self.report["texture_bytes"])

    def test_index_matches_files(self):
        listed = {g["path"] for g in self.groups}
        self.assertEqual(listed, {f"groups/{p.name}" for p in (PACK / "groups").iterdir()})
        for group in self.groups:
            self.assertEqual((PACK / group["path"]).stat().st_size, group["bytes"])
            for name in group["textures"]:
                self.assertTrue((PACK / "textures" / name).is_file(), name)

    def test_every_wonder_and_era(self):
        ledger = json.loads(wp.LEDGER.read_text())["mappings"]
        keys = {g["key"] for g in self.groups}
        for entry in ledger:
            alias = entry["c3x_name"].lower()
            self.assertIn(f"wonder|{alias}|complete", keys)
            for built in wp.BUILDS:
                self.assertIn(f"wonder|{alias}|build:{built}", keys)
        for era in range(len(wp.plan.WD_ERAS)):
            self.assertIn(f"district|{era}|complete", keys)

    def test_authorities_and_masks(self):
        for key, library in self.libraries.items():
            masks = set()
            for template in library["templates"]:
                self.assertRegex(template["authority"], AUTHORITY)
                self.assertTrue(template["authority"].startswith(key + "|"))
                masks.add(int(template["authority"].split("|")[3]))
                self.assertFalse(template["owns_walls"])
            # Foliage is either absent or present in all three seasonal forms.
            self.assertIn(masks & {2, 5, 8}, (set(), {2, 5, 8}), key)
            self.assertIn(0, masks, key)

    def test_no_source_identifiers(self):
        for group in self.groups:
            self.assertNotRegex(group["key"], SOURCE_WORDS)
        for library in self.libraries.values():
            for material in library["materials"]:
                for texture in material["textures"]:
                    self.assertTrue(not texture or texture.startswith(wp.TEXTURE_PREFIX), texture)

    def test_groups_fit_a_tile(self):
        # A tile's prepared objects must stay under half the preparation byte limit:
        # every group's vertices (all season variants together) within the cap.
        for group, library in self.libraries.items():
            used = {i["model"] for t in library["templates"] for i in t["instances"]}
            vertices = sum(library["models"][m]["vertex_count"] for m in used)
            self.assertLessEqual(vertices, wp.GROUP_VERTICES, group)

    def test_effects_use_production_kinds(self):
        for library in self.libraries.values():
            for template in library["templates"]:
                for instance in template["instances"]:
                    for effect in instance["effects"]:
                        self.assertIn(effect[3], (0.0, 1.0, 2.0))
                        if effect[3] != 2.0:
                            self.assertLess(effect[6], 64)

    def test_stamps_carry_no_effects_or_waves(self):
        # A follow-river stamp moves and turns with its river: effects or waves authored
        # in the kit's own frame would not follow, so the pack carries none.
        from Renderer.tools.asset_compiler import build_natural_wonder_pack as natural
        kits = natural.read_wonders((PACK / "stamps.bin").read_bytes())["kits"]
        self.assertTrue(kits)
        for kit in kits:
            self.assertTrue(kit["flags"] & wp.FOLLOW_RIVER_FLAG and kit["flags"] & wp.STAMP_FLAG)
            self.assertFalse(kit["effects"], kit["key"])
            self.assertFalse(kit["waves"], kit["key"])

    def test_foundations_only_where_asked(self):
        # The city terrace lays a skirt and top plates even on gentle relief, so only
        # kits whose metadata asks for one carry it (none of the approved picks yet).
        asked = {entry["c3x_name"].lower() for entry in json.loads(wp.LEDGER.read_text())["mappings"]
                 if any(part.get("foundation") for part in wp.plan.PICKS[entry["c3x_key"]][0]["parts"])}
        for key, library in self.libraries.items():
            for template in library["templates"]:
                self.assertEqual(bool(template["foundation"]), key.split("|")[1] in asked and
                                 template["authority"].split("|")[3] == "0", template["authority"])


if __name__ == "__main__":
    unittest.main()
