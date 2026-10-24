import struct
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

from Renderer.tools.asset_compiler import build_natural_wonder_pack as pack
from Renderer.tools.asset_compiler import build_resource_compositions as rc


def instance(model: str, u: float = .5, v: float = .5) -> dict:
    return {"model": model, "u": u, "v": v, "rotation": 0.0, "scale": 1.0, "lift": 0.0, "ground_fit": 0.0}


def bc1_palette(c0: int, c1: int, four_colour: bool) -> list[tuple[float, float, float]]:
    def rgb(value: int) -> tuple[float, float, float]:
        return ((value >> 11) & 31) / 31, ((value >> 5) & 63) / 63, (value & 31) / 31
    a, b = rgb(c0), rgb(c1)
    if four_colour:
        return [a, b, tuple((2 * x + y) / 3 for x, y in zip(a, b)), tuple((x + 2 * y) / 3 for x, y in zip(a, b))]
    return [a, b, tuple((x + y) / 2 for x, y in zip(a, b)), (0.0, 0.0, 0.0)]


def decode_bc1(block: bytes) -> list[tuple[float, float, float]]:
    c0, c1, bits = struct.unpack("<HHI", block)
    palette = bc1_palette(c0, c1, c0 > c1)
    return [palette[(bits >> (2 * texel)) & 3] for texel in range(16)]


def decode_bc3_colour(block: bytes) -> list[tuple[float, float, float]]:
    c0, c1, bits = struct.unpack("<HHI", block[8:16])
    palette = bc1_palette(c0, c1, True)                 # BC3 colour blocks are always four-colour
    return [palette[(bits >> (2 * texel)) & 3] for texel in range(16)]


class SeasonGroupTests(unittest.TestCase):
    def test_parts_shared_by_every_season_form_one_group(self) -> None:
        rock, crown = instance("rock"), instance("tree", .3, .7)
        groups = pack.season_groups({season: [rock, crown] for season in pack.SEASONS})
        self.assertEqual(groups, [(pack.ALL_SEASONS, [rock, crown])])

    def test_winter_pine_swap_splits_into_season_masks(self) -> None:
        rock, pine, snowy = instance("rock"), instance("forest/pine_01", .2, .2), instance("forest_snow/pine_01", .2, .2)
        groups = dict(pack.season_groups({0: [rock, pine], 1: [rock, pine], 2: [rock, snowy], 3: [rock, pine]}))
        self.assertEqual(groups[pack.ALL_SEASONS], [rock])
        self.assertEqual(groups[0b1011], [pine])          # outside Winter, as forest stands
        self.assertEqual(groups[0b0100], [snowy])         # Winter only

    def test_repeated_placements_keep_their_count(self) -> None:
        tree = instance("tree")
        groups = dict(pack.season_groups({0: [tree, tree], 1: [tree], 2: [tree, tree], 3: [tree, tree]}))
        self.assertEqual(len(groups[pack.ALL_SEASONS]), 1)
        self.assertEqual(len(groups[0b1101]), 1)


class TextureVariantTests(unittest.TestCase):
    def test_three_colour_order_becomes_an_exact_four_colour_bc1_block(self) -> None:
        bits = sum((texel % 4) << (2 * texel) for texel in range(16))
        bc3 = bytes(8) + struct.pack("<HHI", 0x1234, 0xF00D, bits)      # c0 < c1
        bc1 = pack.bc3_colour_as_bc1(bc3)
        self.assertGreater(*struct.unpack_from("<HH", bc1))
        self.assertEqual(decode_bc1(bc1), decode_bc3_colour(bc3))

    def test_equal_and_ordered_endpoints_stay_exact(self) -> None:
        for c0, c1 in ((0xF00D, 0x1234), (0x4321, 0x4321), (0, 0)):
            bc3 = bytes(8) + struct.pack("<HHI", c0, c1, 0xE4E4E4E4)
            self.assertEqual(decode_bc1(pack.bc3_colour_as_bc1(bc3)), decode_bc3_colour(bc3))

    def test_variant_takes_the_summer_size_and_format(self) -> None:
        def chain(width: int, block: int) -> list[bytes]:
            levels, level = [], width
            while level >= 4:
                blocks = (level // 4) ** 2
                levels.append(bytes(range(256)) * (blocks * block // 256) + bytes(range(blocks * block % 256)))
                level //= 2
            return levels
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summer = rc.write_dds(root / "summer.dds", 256, 7, 71, 8, chain(256, 8))
            variant = rc.write_dds(root / "autumn.dds", 512, 8, 77, 16, chain(512, 16))
            matched = pack.matching_variant(summer, variant, root)
            width, height, mips, dxgi, payload = rc.dds(matched)
        self.assertEqual((width, height, mips, dxgi), (256, 256, 7, 71))
        self.assertEqual(len(payload), sum(len(level) for level in chain(256, 8)))


class VeilSlotTests(unittest.TestCase):
    def test_bodies_never_reach_the_veil_slot(self) -> None:
        plain = [list(range(rc.TEXTURE_SLOTS)) for _ in pack.SEASONS]
        pack.check_veil_slot(rc.VEIL_SLOT, plain)                     # seven atlases still fit
        with self.assertRaises(ValueError):
            pack.check_veil_slot(rc.VEIL_SLOT + 1, plain)              # an eighth would be a veil
        autumn = [row[:] for row in plain]
        autumn[1][rc.VEIL_SLOT] = rc.TEXTURE_SLOTS                     # a season atlas swapped into slot 7
        with self.assertRaises(ValueError):
            pack.check_veil_slot(5, autumn)


class WondersBinTests(unittest.TestCase):
    def fake_kits(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(PICKS={"peak": {}, "reef": {"winter_wet": 24.0}}, SNOW_CAPPED={"peak"})

    def test_records_round_trip_through_the_reader(self) -> None:
        kits = self.fake_kits()
        field = np.linspace(0, 1, 16).reshape(4, 4)
        stamp = {"field": field, "ground": field * .5, "special": 1 - field, "span": 2.5, "units": 90.0,
                 "turn": .75, "kind": 4, "effects": [[4.0] + [0.25] * 9, [6.0] + [0.5] * 9]}
        peak = pack.kit_record("peak", {0: {"stamp": stamp, "waves": []}}, [], kits)
        reef = pack.kit_record("reef", {0: {"stamp": None, "waves": [0.1, 0.2, 0.0, 0.5, 2.3]}},
                               [("NW_reef", pack.ALL_SEASONS)], kits)
        data = bytearray(pack.MAGIC) + struct.pack("<IIII", pack.VERSION, 2, 1, rc.TEXTURE_SLOTS)
        for season in pack.SEASONS:
            data += struct.pack(f"<{rc.TEXTURE_SLOTS}I", *range(rc.TEXTURE_SLOTS))
        data += peak + reef + rc.bundle_string("Great Reef") + struct.pack("<I", 1)
        parsed = pack.read_wonders(bytes(data))

        first, second = parsed["kits"]
        self.assertEqual((first["key"], first["flags"], first["region_kind"]), ("peak", pack.FLAG_STAMP | pack.FLAG_SNOW, 4))
        self.assertAlmostEqual(first["span"], 2.5)
        self.assertEqual(first["field_size"], 4)
        self.assertEqual(first["fields"][0][-1], 255)
        self.assertEqual(len(first["effects"]), 2)
        self.assertEqual(first["effects"][0][0], 4.0)             # a geyser keeps its kind
        self.assertEqual(first["effects"][1][0], 9.0)             # spray leaves the flurry's kind 6
        self.assertEqual((second["flags"], second["field_size"], second["snow_floor"]), (0, 0, 24.0))
        self.assertEqual(second["bodies"], [("NW_reef", pack.ALL_SEASONS)])
        self.assertEqual(len(second["waves"]), 1)
        self.assertEqual(parsed["aliases"], [("Great Reef", 1)])

    def test_relief_slots_number_stamped_and_collared_kits_in_pack_order(self) -> None:
        kits = [{"key": "peak", "flags": pack.FLAG_STAMP, "waves": []},
                {"key": "grove", "flags": 0, "waves": []},
                {"key": "bay", "flags": 0, "waves": [[0.0] * 5]},
                {"key": "fjord", "flags": pack.FLAG_STAMP | pack.FLAG_SNOW, "waves": []}]
        self.assertEqual(pack.relief_slots(kits), {"peak": 1, "bay": 2, "fjord": 3})
        with self.assertRaises(ValueError):
            pack.relief_slots([{"key": str(i), "flags": pack.FLAG_STAMP, "waves": []} for i in range(16)])

    def test_reader_rejects_trailing_bytes(self) -> None:
        data = bytearray(pack.MAGIC) + struct.pack("<IIII", pack.VERSION, 0, 0, rc.TEXTURE_SLOTS)
        data += struct.pack(f"<{rc.TEXTURE_SLOTS * 4}I", *([0] * rc.TEXTURE_SLOTS * 4)) + b"\0"
        with self.assertRaises(ValueError):
            pack.read_wonders(bytes(data))


if __name__ == "__main__":
    unittest.main()
