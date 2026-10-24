#!/usr/bin/env python3
"""Civ VI natural-wonder sources as Renderer64 Lab kits beside C3X's natural-wonder sprites (Lab only).

Each proposed source wonder (Civ VI install or a Steam Workshop item) is read
from its own cooked packages: the TerrainStyle height field (NWON terrain
element: height, blend and region-ID fields) becomes a lit relief mesh whose
colour is baked from the Civ VI terrain materials the region IDs select.
Each kit is baked into a Lab-only composition pack and drawn by the production
renderer on the C3X natural wonder's own terrain: its terrain_type tile, its
adjacent_to terrain on neighbouring tiles, and its adjacency direction.

Renderer64 has no per-tile hook for an arbitrary height field yet, so relief
is shown as a rigid kit mesh standing on the tile; terrain-integrated relief is
later M9 work. Nothing here changes production packs, shared tools, the roster
file or ledgers; packs and output are Lab-only and ignored.

    python3 Renderer/lab/studies/natural_wonders/kits.py render [--wonder KEY ...] [--zoom 128 --zoom 256]
    python3 Renderer/lab/studies/natural_wonders/kits.py crops
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from Renderer.tools.asset_compiler import build_resource_compositions as rc
from Renderer.tools.asset_compiler import terrain_relief_builder as relief
from Renderer.tools.asset_compiler.artdef_graph_resolver import DEFAULT_ASSETS_ROOT
from Renderer.inventory.natural_wonder_mapping_inventory import DEFAULT_CONFIG, parse_natural_wonder_config
from Renderer.lab.studies.natural_wonders import assets

CATALOG = ROOT / "Renderer/packs/NaturalWonderCatalogLab"
OUT = ROOT / "Renderer/lab/out/natural_wonders"
WORKSHOP = Path.home() / "Library/Application Support/Steam/steamapps/workshop/content/289070"
# Renderer64 draws its ground and mountains with the Civ V Environment Skin textures.
MATERIALS = ROOT / "Renderer/packs/Civ5EnvironmentSkin/textures"
SPRITES = ROOT / "Art/Districts/Summer/1200/NaturalWonders.pcx"
ALL_TERRAINS = (1 << 14) - 1
# Civ VI's camera-facing (-y) side toward C3X's viewer, as the mine and district studies do.
FACING = 3 * math.pi / 4
# Kits whose source has a front (Angel Falls' falls) turn it from Civ VI's south (screen
# south at FACING) to the C3X adjacency direction: extra turn in the composition's frame.
FRONT_TURN = {"south": 0.0, "southeast": -math.pi / 4, "east": -math.pi / 2, "northeast": -3 * math.pi / 4,
              "north": math.pi, "northwest": 3 * math.pi / 4, "west": math.pi / 2, "southwest": math.pi / 4}


def front_turn(key: str) -> float:
    pick = PICKS[key]
    if pick.get("turn") is not None:
        return pick["turn"]
    if not pick.get("front"):
        return 0.0
    return FRONT_TURN[pick.get("front_dir") or definitions()[key].get("adjacency_dir") or "south"]
# One Civ VI hex is one C3X tile. Neighbouring hexes sit 64 world units apart (the
# two- and three-hex TerrainAsset layouts) and span 256 terrain-element grid units
# (single-hex blend footprints). Imported meshes are 100 world units per tile.
HEX_GRID = 256.0
HEX_UNITS = 64.0
ASSET_TO_HEX = 100.0 / HEX_UNITS
# Every kit is scaled so its footprint spans about this many tiles: C3X's natural-wonder
# sprites fill their tile, while Civ VI wonders at true scale sit small within one hex.
# Multi-hex sources fold onto the one C3X tile the same way. Relief and meshes share a
# mild vertical exaggeration so attached pieces stay seated on their slopes.
FIT_TILES = 1.2
VERTICAL = 1.3
# Terrain-stamp mode (stamp_tree.py): the relief joins Renderer64's terrain mesh as a
# Lab-only stamp beside the volcano stamp. Relief units per tile of height (a unit is
# about .47 px at 1x, so the mesh kits' heights need about 180 per tile), the stamp's turn (Civ VI's south toward the viewer, as the
# kits' FACING), the Lab marker bit and the per-render stamp pack.
STAMP = False
# C3X season index (0 Summer, 1 Autumn, 2 Winter, 3 Spring) for the seasonal variants.
SEASON = 0
SEASON_NAMES = {0: "summer", 1: "autumn", 2: "winter", 3: "spring"}
STAMP_UNITS = 180.0
STAMP_TURN = 3 * math.pi / 4
STAMP_MARKER = 0x40000000
STAMP_PACK = ROOT / "Renderer/packs/NaturalWonderStampLab"
STAMP_TREE = ROOT / "Renderer/lab/out/natural_wonders/tree-stamp"
TERRAIN_ONLY: set[str] = set()      # stamp-mode wonders with no meshes or clutter
# Region colours on terrain-integrated relief (stamp_tree.py's shader blend): ground ids
# ease to the tile's own ground (vegetated slopes), special ids to the wonder's special
# material, chosen per render by shader variant: 1 snow, 2 travertine, 3 banded sandstone,
# 4 stratovolcano (special 0.5 volcanic brown, 1 snow).
# "all" applies a weight over the whole relief (sources without region IDs); green_foot
# eases low rock (below that normalized source height) to ground, as C3X's sprites show.
REGIONS = {
    "angel_falls": dict(kind=3, ground=(217,), special=(115,), special_strength=0.55),
    "yosemite": dict(kind=1, ground=(13, 14)),
    "mount_fuji": dict(kind=4, ground={0: 1.0, 13: 1.0, 14: 0.75}, special={14: 0.25, 115: 0.5, 153: 1.0}),
    "yellowstone": dict(kind=2, special=(120,), special_strength=0.6),
    "mount_everest": dict(kind=1, special=(153, 154)),
    "mount_kilimanjaro": dict(kind=1, special=(153, 154), green_foot=0.30),
    "matterhorn": dict(kind=1, ground=(13, 14), special=(153, 154)),
    "moraine_lake": dict(kind=1, ground=(13, 14), special=(153, 154)),
    "wadi_rum": dict(kind=3, special="all", special_rise=(0.0, 0.12), ground_fade=(0.0, 0.08)),
    "eyjafjallajokull": dict(kind=1, special=(133, 153, 154)),
    "geirangerfjord": dict(kind=1, ground=(13, 14), special=(153, 154)),
}

# Wonders whose tile carries Civ III's snow cap, so the mountain shader snows their peaks.
SNOW_CAPPED = {"mount_kilimanjaro", "mount_fuji", "matterhorn", "mount_everest", "eyjafjallajokull", "geirangerfjord"}
# Attached effects on terrain-integrated wonders (stamp_tree.py): Lab stand-ins for Civ VI's
# particle effects, drawn by Renderer64's city effect layer as animated camera-facing quads.
# Kinds: 4 geyser steam, 5 waterfall (a ribbon draped down the stamped face, standing off it
# by some relief units), 6 plunge-pool spray. Geysers erupt from the highest cones of the
# special region. Widths and heights are tile units.
EFFECT_STEAM, EFFECT_WATERFALL, EFFECT_SPRAY, EFFECT_LAKE, EFFECT_CHANNEL = 4.0, 5.0, 6.0, 7.0, 8.0
# C3X directions as Renderer64 natural-coordinate unit vectors (+x down-right on screen).
NATURAL_DIRECTIONS = {"southeast": (1.0, 0.0), "northwest": (-1.0, 0.0), "northeast": (0.0, 1.0), "southwest": (0.0, -1.0),
                      "south": (.7071, -.7071), "north": (-.7071, .7071), "east": (.7071, .7071), "west": (-.7071, -.7071)}
EFFECTS = {
    "angel_falls": dict(waterfall=("FIREFX_NWON_SUK_ANGELFALLS001", 0.22, 4.0)),
    "yellowstone": dict(central=True, vents=((0.36, 0.9, 1.0), (0.24, 0.6, 0.5), (0.22, 0.55, 0.46), (0.22, 0.5, 0.42),
                               (0.2, 0.48, 0.4))),
}
# Effect clock for the Lab stills (s): mid-eruption for the geyser's cycle.
EFFECT_TIME = 3.0
TEXTURE = 512
# Relief bakes: texels per hex, terrain-material mip size and repeats per hex, and the
# most field texels one relief sub-mesh spans before its vertices are stepped.
BAKE_PER_HEX = 640
MATERIAL_SIZE = 512
# Production tiling per tile: ground 0.43 (mountain.hlsl uv0) and rock 1.8 (MTN_ROCK_SCALE).
GROUND_REPEAT = 0.43
ROCK_REPEAT = 1.8
SUB_TEXELS = 160

BASE_TE = "Base/Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp"
XP1_TE = "DLC/Expansion1/platforms/windows/BLPs/terrain/TerrainElementSet_Base.blp"
XP2_TE = "DLC/Expansion2/Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp"
VIK_TE = "DLC/VikingsLandmarks/Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp"
MOD_TE = "Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp"

# C3X key -> proposed source. Workshop items name their item id; None is the Civ VI install.
# enlarge and rise are Lab readability tuning (footprint and relief height multipliers).
TA = "Platforms/Windows/BLPs/terrain/TerrainAssetSet_Base.blp"
IK_TA = "DLC/Indonesia_Khmer/platforms/windows/BLPs/terrain/TerrainAssetSet_Base.blp"
SAVANNAH = ("3263126211", "Platforms/Windows/BLPs/environment/clutter.blp")
PICKS = {
    "angel_falls": dict(rise=1.3, front=True, waterfall="FIREFX_NWON_SUK_ANGELFALLS001", title="Sukritact's Angel Falls", root="2984762028", package=MOD_TE,
                        element="ART_DEF_TERRAIN_ELEMENT_NWON_SUK_ANGELFALLS",
                        asset=("2984762028", TA, "NWON_Suk_AngelFalls_Tilebase")),
    "yosemite": dict(rise=0.75, keep_east=True, thin=3, piece_scale=1.32, centre=(0.05, 0.05), turn=3 * math.pi / 4,
                     normal_bake=1.3, vertical=1.0, grain=("mtn_base", 7.0, .3), title="Yosemite (its Half Dome side)", root=None, package=BASE_TE, element="ART_DEF_TERRAIN_ELEMENT_NWON_YOSEMITE",
                     asset=(None, "Base/" + TA, "ART_DEF_TERRAIN_ASSET_NWON_YOSEMITE")),
    "mount_fuji": dict(title="Mount Fuji (authored stratovolcano)", root=None, analytic="stratovolcano"),
    "yellowstone": dict(enlarge=1.15, title="Old Faithful (Terra Mirabilis)", root="1461463960", package=MOD_TE,
                        element="TerrainElement_Old_Faithful"),
    "mount_everest": dict(rise=1.21, enlarge=1.4, bulk=(0.14, 0.7), title="Mount Everest", root=None, package=BASE_TE, element="ART_DEF_TERRAIN_ELEMENT_NWON_EVEREST"),
    "zhangjiajie_mountains": dict(extra_pillars=(14, 0.17, (0.7, 1.05), 7), pillar_stretch=1.6, thin=2, fold=0.565, title="Wulingyuan (Terra Mirabilis)", root=None,
                                  asset=(None, IK_TA, "ART_DEF_TERRAIN_ASSET_NWON_HALONGBAY")),
    "mount_kilimanjaro": dict(title="Mount Kilimanjaro", root=None, package=BASE_TE,
                              element="ART_DEF_TERRAIN_ELEMENT_NWON_KILIMANJARO"),
    "great_barrier_reef": dict(bake=1280, winter_wet=24.0, title="Great Barrier Reef", root=None, package=BASE_TE,
                               element="ART_DEF_TERRAIN_ELEMENT_NWON_BARRIERREEF", sea=0, submerged=True,
                               # Sea-tile objects seat on the sea bed; this lift brings the reef's crests to the surface.
                               lift=0.09,
                               asset=(None, "Base/" + TA, "ART_DEF_TERRAIN_ASSET_NWON_BARRIERREEF")),
    "matterhorn": dict(rise=0.89, enlarge=1.12, bulk=(0.1, 0.55), title="Matterhorn", root=None, package=XP1_TE, element="ART_DEF_TERRAIN_ELEMENT_NWON_MATTERHORN"),
    "moraine_lake": dict(title="Lake basin with the Matterhorn's peak behind", root=None, package=BASE_TE,
                         element="ART_DEF_TERRAIN_ELEMENT_NWON_CRATERLAKE", lake=204, compose_land=True, effect_water="lake",
                         lake_level=0.098,
                         # The lake to the front, a mountain behind it (Civ VI hex units, scale, height).
                         compose=(("basin", 0.14, -0.30, 0.34, 0.22, 6.0), ("matterhorn", -0.10, 0.24, 0.70, 1.25))),
    "tropical_rainforest": dict(title="Jungle canopy (Renderer64's Civ V skin jungle trees)", root=None, canopy=True, fold=1.0),
    "wadi_rum": dict(no_brush=True, rise=1.0, enlarge=1.1, sink=0.035, talus=(0.05, 0.2), relief_flat=True, talus_share=0.42, pedestal=True,
                     # Screen layout (tile units: x right, y toward the viewer): the two Mittens side by
                     # side, Merrick Butte to the front right, a lower mesa behind (mesh, x, y, turn, scale).
                     screen_layout=(("f4e0ae90b2bfef11", -0.48, -0.04, 0.0, 0.95), ("f4e0ae90b2bfef11", 0.08, -0.26, math.pi, 0.9),
                                    ("7a0594232bb1f373", 0.50, 0.22, 0.6, 0.95), ("48b222f774c9d781", -0.06, -0.56, 2.0, 0.85)), title="Sukritact's Monument Valley", root="2949095377", package=MOD_TE,
                     element="NWON_Suk_MonumentValley_TerrainElement", cliffs="mtn_desert_stripe01",
                     asset=("2949095377", TA, "NWON_Suk_MonumentValley")),
    "eyjafjallajokull": dict(rise=1.4, title="Eyjafjallajökull", root=None, package=VIK_TE,
                             element="ART_DEF_TERRAIN_ELEMENT_NWON_EYJAFJALLAJOKULL"),
    "ha_long_bay": dict(waves=(0.5, .28), title="Hạ Long Bay", root=None, asset=(None, IK_TA, "ART_DEF_TERRAIN_ASSET_NWON_HALONGBAY")),
    "lofoten_skerries": dict(title="Skerries from the Matterhorn's peak", root=None, sea=179, analytic="skerries", snow_slope=(0.1, 0.45),
                             one_neighbour=True, bake=1280, rock_relief=0.035, waves=(0.03, .5), fold=0.931,
                             # Islets in the sea: Matterhorn peaks (Civ VI hex units, scale, height).
                             detail_bake=1.4,
                             compose=(("matterhorn", -0.10, 0.14, 0.4, 0.6, 0.4), ("matterhorn", 0.23, 0.25, 0.24, 0.6, 2.1),
                                      ("matterhorn", 0.33, 0.12, 0.13, 0.42, 4.4), ("matterhorn", -0.29, -0.18, 0.21, 0.56, 4.0),
                                      ("matterhorn", -0.17, -0.31, 0.13, 0.4, 5.3), ("matterhorn", -0.41, -0.10, 0.11, 0.36, 3.1),
                                      ("matterhorn", -0.33, 0.34, 0.16, 0.48, 0.9), ("matterhorn", 0.09, -0.13, 0.12, 0.38, 2.6),
                                      ("matterhorn", 0.15, -0.26, 0.10, 0.32, 1.7))),
    "geirangerfjord": dict(water_level=0.03, enlarge=1.39, centre=(-0.52, 0.5), open_mouth=0.4, effect_water="channel", title="Lysefjord", root=None, package=VIK_TE,
                           element="ART_DEF_TERRAIN_ELEMENT_NWON_LYSEFJORDEN", sea=179),
    "delicate_arch": dict(enlarge=0.7, front=True, front_dir="southeast", title="Delicate Arch", root=None,
                          asset=(None, "DLC/Expansion1/platforms/windows/BLPs/terrain/TerrainAssetSet_Base.blp",
                                 "ART_DEF_TERRAIN_ASSET_NWON_DELICATEARCH")),
    # Authored from JNR's Savannah models: a rock clump as the kopje, acacias and shrubs around it.
    # (item, package, entry, x, y, yaw, scale) in hex units, x east and y north.
    "savanna": dict(grit=("Rocks", "mtn_base", 6.0), grass=(200, 0.042, 3), title="Savannah biome (JNR), authored", root=None, layout=(
        SAVANNAH + ("Tree_Savannah_01", -0.30, 0.10, 0.3, 1.0),
        SAVANNAH + ("Desert_Rocks_02", 0.02, 0.02, 0.0, 5.5), SAVANNAH + ("Desert_Rocks_01", 0.13, -0.07, 1.1, 4.0), SAVANNAH + ("Tree_Savannah_02", 0.30, 0.20, 1.2, 1.0),
        SAVANNAH + ("Tree_Savannah_03", -0.12, 0.34, 2.0, 0.9), SAVANNAH + ("Tree_Savannah_Green_01", 0.30, -0.22, 0.7, 1.0),
        SAVANNAH + ("Tree_Savannah_04", -0.34, -0.24, 2.6, 0.9), SAVANNAH + ("Tree_Savannah_Green_02", 0.04, -0.36, 4.0, 0.85),
        SAVANNAH + ("Desert_Shrub_01", 0.18, 0.04, 0.0, 1.0), SAVANNAH + ("Desert_Shrub_02", -0.18, -0.08, 1.0, 1.0),
        SAVANNAH + ("Desert_Shrub_02", -0.04, 0.20, 2.0, 1.0), SAVANNAH + ("Desert_Shrub_01", 0.36, -0.02, 3.0, 1.0))),
}

# Wonder clutter sets (CLUTTER_YOSEMITE and kin scatter TER_Scatter_Tree_*; Old Faithful
# scatters pines). With the Civ V skin those trees are the skin's own; here they come from
# Renderer64's extracted skin vegetation, the trees its forest tiles draw, at its
# forest placement scales (leafy 1.0, pines 0.5).
VEGETATION = ROOT / "Renderer/packs/Civ5EnvironmentVegetation"
SCATTER_TREES = (("forest/leafy_v1_01", 1.0), ("forest/leafy_v2_02", 1.0), ("forest/leafy_clump_01", 1.0),
                 ("forest/leafy_v3_03", 1.0), ("forest/leafy_v4_02", 1.0), ("forest/leafy_clump_02", 1.0))
PINES = (("forest/pine_01", 0.5), ("forest/pine_02", 0.5), ("forest/pine_03", 0.5), ("forest/pine_clump_01", 0.6),
         ("forest/pine_clump_02", 0.6))
CLUTTER = {"yellowstone": (PINES, 45),
           "moraine_lake": (PINES, 18)}
# Renderer64's forest tiles draw these models about a third of their placement scale
# (measured against neighbouring forest tiles in the Lab renders).
TREE_SCALE = 0.33

# Region IDs inferred from height/slope statistics across installed wonders
# (13 vegetated slope, 115 rock, 128 high rock, 153 snow, 179 shore/underwater,
# 204 lake bed, 199 reef, 120 salt/travertine, 112 crater, 217/122 dark volcanic
# rock, 133 ice). Inference, not a recovered Civ VI material table.
REGION_MATERIALS = {13: "grasshill_top", 115: "mtn_base", 128: "mtn_top", 153: "mtn_snow",
                    179: "shallows", 204: "shallows", 199: "beach", 120: "salt_base",
                    112: "mtn_dark_base", 217: "mtn_dark_base", 122: "mtn_dark_base", 133: "ice",
                    222: "mtn_desert_stripe01"}
HOST_MATERIALS = {"grassland": "grassland", "plains": "plains", "desert": "desert", "tundra": "tundra",
                  "jungle": "grassland", "coast": "beach", "sea": "shallows"}

# Civ III (base, real) terrain codes for the Lab scene; snow mountains add BIQ bonus 0x10.
TERRAIN = {"desert": (0, 0), "plains": (1, 1), "grassland": (2, 2), "tundra": (3, 3), "floodplain": (4, 4),
           "hill": (2, 5), "hills": (2, 5), "mountain": (2, 6), "mountains": (2, 6), "snow-mountains": (2, 6),
           "forest": (2, 7), "jungle": (2, 8), "marsh": (2, 9), "volcano": (2, 10), "coast": (11, 11),
           "sea": (12, 12), "ocean": (13, 13)}
DIRECTIONS = {"north": (0, -2), "northeast": (1, -1), "east": (2, 0), "southeast": (1, 1),
              "south": (0, 2), "southwest": (-1, 1), "west": (-2, 0), "northwest": (-1, -1)}
# Civ III river edge bits of a tile, by direction (NE, SE, SW, NW edges).
RIVER_EDGE = {"northeast": 2, "southeast": 8, "southwest": 32, "northwest": 128}
BACK = ("north", "northeast", "northwest")


# ---------------------------------------------------------------- source fields

@contextmanager
def _workshop_reader():
    """Workshop terrain-element packages name entries without the ART_DEF_
    prefix, and Terra Mirabilis ships two malformed entries. Widen the entry
    bootstrap and skip invalid records in this process only."""
    pattern, decode = relief.ENTRY_NAME_PATTERN, relief.decode_terrain_element_record
    relief.ENTRY_NAME_PATTERN = re.compile(
        rb"(?:ART_DEF_TERRAIN_ELEMENT_[A-Za-z0-9_]+|NWON_Suk_[A-Za-z0-9]+(?:_TerrainElement)?|TerrainElement_[A-Za-z0-9_]+)\0")

    def tolerant(raw, name, resources):
        try:
            return decode(raw, name, resources)
        except ValueError as exc:
            return {"entry_name": name, "invalid": str(exc), "grid_dimensions": [0, 0], "parameters": {}, "channels": {}}
    relief.decode_terrain_element_record = tolerant
    try:
        yield
    finally:
        relief.ENTRY_NAME_PATTERN, relief.decode_terrain_element_record = pattern, decode


def package_path(pick: dict) -> Path:
    return (DEFAULT_ASSETS_ROOT if pick["root"] is None else WORKSHOP / pick["root"]) / pick["package"]


@lru_cache(maxsize=None)
def _package(path: Path):
    if WORKSHOP in path.parents:
        with _workshop_reader():
            return relief.inspect_terrain_element_package(path)
    return relief.inspect_terrain_element_package(path)


def element_fields(pick: dict) -> dict:
    path = package_path(pick)
    _, elements, report = _package(path)
    entry = elements[pick["element"]]
    if "invalid" in entry:
        raise ValueError(f"{pick['element']}: {entry['invalid']}")
    fields = {"grid": entry["grid_dimensions"][0], "parameters": entry["parameters"]}
    with path.open("rb") as stream:
        for role, channel_list in entry["channels"].items():
            channel = next(item for item in channel_list if item["level"] == 0)
            stream.seek(report["big_data_offset"] + channel["relative_offset"])
            fields[role] = np.frombuffer(stream.read(channel["bytes"]), np.uint8).reshape(
                channel["height"], channel["width"])
    return fields


def has_relief(pick: dict) -> bool:
    return bool(pick.get("element") or pick.get("analytic"))


def stratovolcano_fields(seed: int = 3) -> dict:
    """An authored Fuji-like stratovolcano as element fields: a broad concave cone with a
    wide summit crater, faint radial gullies, vegetated lower slopes (13, thinning to 14
    up the ridges), a rock band (115) and a snow cap (153) whose fingers run down the gullies."""
    res, hexes = 256, 1.7
    axis = (np.arange(res) + .5) / res * hexes - hexes / 2
    x, y = np.meshgrid(axis, axis)
    r, theta = np.hypot(x, y), np.arctan2(y, x)
    radius = 0.64
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, 3)
    wobble = (np.sin(3 * theta + phases[0]) * .5 + np.sin(5 * theta + phases[1]) * .3 + np.sin(9 * theta + phases[2]) * .2)
    t = np.clip(r / (radius * (1 + .025 * wobble)), 0, 1)
    h = (1 - t) ** 1.45
    # Summit: a broad flat rim around a shallow crater.
    rim = (1 - .09) ** 1.45
    h = np.where(h > rim, rim - .03 * _smoothstep(rim + .015, 1.0, h), h)
    gully = np.sin(22 * theta + 2.5 * np.sin(4 * theta + phases[0]))
    h = h * (1 + .018 * gully * t * (1 - t) * 4)
    h = np.clip(h, 0, None) / max(float(h.max()), 1e-6)
    blend = _smoothstep(radius * 1.04, radius * .80, r)
    finger = np.clip(gully, 0, 1) ** 2
    ids = np.full((res, res), 115, np.uint8)
    ids[h < .44 + .08 * finger] = 14
    ids[h < .28 + .04 * wobble] = 13
    ids[h > .66 - .15 * finger * (h > .45)] = 153
    ids[blend < .02] = 0
    return {"grid": hexes * HEX_GRID, "parameters": {"base_height": 0.0, "height_scale": 27.0},
            "height": np.clip(h * 255 + .5, 0, 255).astype(np.uint8), "blend": np.clip(blend * 255 + .5, 0, 255).astype(np.uint8),
            "region_ids": ids}


def _field_world(fields: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """A field's height above its own foot (Civ VI world units), blend (0-1), region ids,
    texel size and extent (hex units)."""
    height = fields["height"].astype(np.float32) / 255 * fields["parameters"]["height_scale"]
    blend = fields.get("blend", np.full_like(fields["height"], 255)).astype(np.float32) / 255
    foot = height[(blend > .02) & (blend < .2)]
    height = height - (float(np.median(foot)) if foot.size else 0.0)
    res = height.shape[0]
    extent = float(fields["grid"]) / HEX_GRID
    ids = fields.get("region_ids")
    return height, blend, ids if ids is not None else np.zeros_like(fields["height"]), extent / res, extent


# Composition sources that are not themselves a pick.
SOURCE_ELEMENTS = {"crater_lake": dict(root=None, package="Base/Platforms/Windows/BLPs/terrain/TerrainElementSet_Base.blp",
                                       element="ART_DEF_TERRAIN_ELEMENT_NWON_CRATERLAKE")}


def composed_fields(pick: dict, base: dict | None) -> dict:
    """Element fields with other wonders' elements placed in them (pick["compose"]: source
    wonder, Civ VI hex offset, scale and height scale). A base of None is open sea (179)."""
    if base is None:
        # Open sea (a sea floor below the islets) or open land.
        res, extent = 512, 2.0
        b_height = np.full((res, res), -6.0 if pick.get("sea") else 0.0, np.float32)
        b_blend = np.zeros((res, res), np.float32)
        b_ids = np.full((res, res), pick.get("sea") or 0, np.uint8)
        texel = extent / res
    else:
        b_height, b_blend, b_ids, texel, extent = _field_world(base)
        b_ids = b_ids.copy()
        res = b_height.shape[0]
    axis = (np.arange(res) + .5) * texel - extent / 2
    gx, gy = np.meshgrid(axis, axis)
    height, blend, ids = b_height * b_blend, b_blend.copy(), b_ids
    for source, ox, oy, scale, rise, *extra in pick["compose"]:
        if source == "basin":
            # An authored lake basin: a low moraine rim (height `extra[0]`, Civ VI units)
            # around a flat raised bed, radii scale x rise (hex units).
            d = np.hypot((gx - ox) / scale, (gy - oy) / rise)
            rim = extra[0] * np.exp(-((d - 1) / .28) ** 2)
            sh = np.where(d < 1, np.maximum(rim, extra[0] * .42), rim)
            sb = _smoothstep(1.6, 1.25, d)
            taller = sh * sb > height
            height = np.where(taller, sh * sb + height * (1 - sb), height)
            ids = np.where(d < .9, pick.get("lake", 204), ids)
            blend = np.maximum(blend, sb)
            continue
        s_height, s_blend, s_ids, s_texel, s_extent = _field_world(element_fields(SOURCE_ELEMENTS.get(source) or PICKS[source]))
        # Sample the source at (position - offset) / scale, turned by the optional extra angle.
        turn = extra[0] if extra else 0.0
        lx, ly = (gx - ox) / scale, (gy - oy) / scale
        lx, ly = lx * math.cos(turn) + ly * math.sin(turn), -lx * math.sin(turn) + ly * math.cos(turn)
        sc = (lx + s_extent / 2) / s_texel - .5
        sr = (ly + s_extent / 2) / s_texel - .5
        inside = (sc >= 0) & (sc <= s_height.shape[1] - 1) & (sr >= 0) & (sr <= s_height.shape[0] - 1)
        sh = _bilinear(s_height, sr, sc) * scale * rise
        sb = np.where(inside, _bilinear(s_blend, sr, sc), 0.0)
        si = s_ids[np.clip(np.round(sr).astype(int), 0, s_ids.shape[0] - 1), np.clip(np.round(sc).astype(int), 0, s_ids.shape[1] - 1)]
        taller = (sh * sb > height) & (sb > .05)
        height = np.where(taller, sh * sb + height * (1 - sb), height)
        if base is None and pick.get("sea"):
            # Islets: bare rock to the waterline, snow on their upper part.
            peak = max(float((sh * sb).max()), 1e-6)
            si = np.where(sh * sb > .55 * peak, 153, 115).astype(np.uint8)
        ids = np.where(taller & (si > 0), si, ids)
        if base is None and pick.get("sea"):
            ids = np.where((sb > .03) & (ids == pick["sea"]), 115, ids)      # no beach skirt on islets
        blend = np.maximum(blend, sb)
    low, high = float(height.min()), float(height.max())
    return {"grid": extent * HEX_GRID, "parameters": {"base_height": low, "height_scale": high - low},
            "height": np.clip((height - low) / max(high - low, 1e-6) * 255 + .5, 0, 255).astype(np.uint8),
            "blend": np.clip(blend * 255 + .5, 0, 255).astype(np.uint8), "region_ids": ids.astype(np.uint8)}


def source_fields(pick: dict) -> dict:
    """The relief's element fields: the source element, an authored shape, or a composition."""
    if pick.get("analytic") == "stratovolcano":
        return stratovolcano_fields()
    base = element_fields(pick) if pick.get("element") and not pick.get("compose_land") else None
    return composed_fields(pick, base) if pick.get("compose") else base


# ---------------------------------------------------------------- BC textures

def _rgb565(values: np.ndarray) -> np.ndarray:
    r = (values >> 11) & 31
    g = (values >> 5) & 63
    b = values & 31
    return np.stack([r * 255 // 31, g * 255 // 63, b * 255 // 31], -1).astype(np.float32)


def decode_colour(path: Path, size: int = 256) -> np.ndarray:
    """sRGB colour of a BC1/BC3 DDS mip no larger than size, as float32 HxWx3."""
    width, height, mips, dxgi, payload = rc.dds(path)
    block = 8 if dxgi in (70, 71, 72) else 16
    level = 0
    while (width >> level) > size and level + 1 < mips:
        level += 1
    offset, bw, bh = rc.mip_offsets(width, height, mips, block)[level]
    blocks = np.frombuffer(payload[offset:offset + bw * bh * block], np.uint8).reshape(bh, bw, block)
    colour = blocks[..., block - 8:]
    c0 = colour[..., 0].astype(np.uint16) | (colour[..., 1].astype(np.uint16) << 8)
    c1 = colour[..., 2].astype(np.uint16) | (colour[..., 3].astype(np.uint16) << 8)
    bits = (colour[..., 4].astype(np.uint32) | (colour[..., 5].astype(np.uint32) << 8) |
            (colour[..., 6].astype(np.uint32) << 16) | (colour[..., 7].astype(np.uint32) << 24))
    e0, e1 = _rgb565(c0), _rgb565(c1)
    four = (c0 > c1)[..., None] | (block == 16)
    palette = np.stack([e0, e1, np.where(four, (2 * e0 + e1) / 3, (e0 + e1) / 2),
                        np.where(four, (e0 + 2 * e1) / 3, 0)], 2)
    index = np.stack([(bits >> (2 * k)) & 3 for k in range(16)], -1).reshape(bh, bw, 4, 4)
    texels = np.take_along_axis(palette[:, :, None, None, :, :].repeat(4, 2).repeat(4, 3),
                                index[..., None, None].repeat(3, -1).astype(np.int64), 4)[..., 0, :]
    return texels.transpose(0, 2, 1, 3, 4).reshape(bh * 4, bw * 4, 3)


def encode_bc3(rgba: np.ndarray) -> bytes:
    """Vectorised BC3: luma-extreme colour endpoints, min/max alpha (sRGB in, sRGB out)."""
    h, w, _ = rgba.shape
    blocks = rgba.reshape(h // 4, 4, w // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(h // 4, w // 4, 16, 4)
    rgb, alpha = blocks[..., :3], blocks[..., 3]
    luma = rgb @ np.array([.3, .59, .11], np.float32)
    hi = np.take_along_axis(rgb, luma.argmax(-1)[..., None, None].repeat(3, -1), 2)[..., 0, :]
    lo = np.take_along_axis(rgb, luma.argmin(-1)[..., None, None].repeat(3, -1), 2)[..., 0, :]

    def code(c):
        c = np.clip(c, 0, 255).astype(np.uint32)
        return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)
    c0, c1 = code(hi), code(lo)
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    e0, e1 = _rgb565(c0.astype(np.uint16)), _rgb565(c1.astype(np.uint16))
    palette = np.stack([e0, e1, (2 * e0 + e1) / 3, (e0 + 2 * e1) / 3], 2)
    distance = ((rgb[:, :, :, None, :] - palette[:, :, None, :, :]) ** 2).sum(-1)
    index = distance.argmin(-1).astype(np.uint32)
    index = np.where((c0 == c1)[..., None], 0, index)
    colour_bits = (index << (2 * np.arange(16, dtype=np.uint32))).sum(-1).astype(np.uint32)
    a0, a1 = alpha.max(-1).round(), alpha.min(-1).round()
    apal = np.stack([a0, a1] + [((7 - k) * a0 + k * a1) / 7 for k in range(1, 7)], -1)
    aindex = np.abs(alpha[..., None] - apal[:, :, None, :]).argmin(-1).astype(np.uint64)
    abits = (aindex << (3 * np.arange(16, dtype=np.uint64))).sum(-1).astype(np.uint64)
    out = np.zeros((h // 4, w // 4, 16), np.uint8)
    out[..., 0], out[..., 1] = a0.astype(np.uint8), a1.astype(np.uint8)
    for k in range(6):
        out[..., 2 + k] = ((abits >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)
    out[..., 8], out[..., 9] = (c0 & 255).astype(np.uint8), (c0 >> 8).astype(np.uint8)
    out[..., 10], out[..., 11] = (c1 & 255).astype(np.uint8), (c1 >> 8).astype(np.uint8)
    for k in range(4):
        out[..., 12 + k] = ((colour_bits >> (8 * k)) & 255).astype(np.uint8)
    return out.tobytes()


def write_bc3(path: Path, rgba: np.ndarray) -> Path:
    levels, image = [], rgba.astype(np.float32)
    while True:
        levels.append(encode_bc3(image))
        if image.shape[0] <= 4:
            break
        image = image.reshape(image.shape[0] // 2, 2, image.shape[1] // 2, 2, 4).mean((1, 3))
    return rc.write_dds(path, rgba.shape[0], len(levels), 78, 16, levels)


@lru_cache(maxsize=None)
def material(name: str) -> np.ndarray:
    return decode_colour(MATERIALS / f"{name}_base_color.dds", MATERIAL_SIZE)


@lru_cache(maxsize=None)
def material_height(name: str) -> np.ndarray:
    """BC4 height detail of a terrain material (0-1), at the colour's resolution."""
    path = MATERIALS / f"{name}_height.dds"
    if not path.is_file():
        return np.full((4, 4), .5, np.float32)
    width, height, mips, dxgi, payload = rc.dds(path)
    if dxgi not in (80, 81):
        return np.full((4, 4), .5, np.float32)
    level = 0
    while (width >> level) > MATERIAL_SIZE and level + 1 < mips:
        level += 1
    offset, bw, bh = rc.mip_offsets(width, height, mips, 8)[level]
    blocks = np.frombuffer(payload[offset:offset + bw * bh * 8], np.uint8).reshape(bh, bw, 8)
    return _bc4(blocks)


def _bc4(blocks: np.ndarray) -> np.ndarray:
    """One BC4 channel (0-1) from its 8-byte blocks, (bh, bw, 8)."""
    bh, bw = blocks.shape[:2]
    r0, r1 = blocks[..., 0].astype(np.float32), blocks[..., 1].astype(np.float32)
    bits = np.zeros((bh, bw), np.uint64)
    for k in range(6):
        bits |= blocks[..., 2 + k].astype(np.uint64) << np.uint64(8 * k)
    eight = r0 > r1
    palette = [r0, r1] + [np.where(eight, ((7 - k) * r0 + k * r1) / 7,
                                   ((5 - k) * r0 + k * r1) / 5 if k < 5 else 0) for k in range(1, 5)] + \
              [np.where(eight, (2 * r0 + 5 * r1) / 7, 0.0), np.where(eight, (r0 + 6 * r1) / 7, 255.0)]
    palette = np.stack(palette, -1)
    index = np.stack([((bits >> np.uint64(3 * k)) & np.uint64(7)).astype(np.int64) for k in range(16)], -1)
    texels = np.take_along_axis(palette, index, -1).reshape(bh, bw, 4, 4)
    return (texels.transpose(0, 2, 1, 3).reshape(bh * 4, bw * 4) / 255).astype(np.float32)


def source_channels(path: Path, size: int = 512) -> list[np.ndarray]:
    """A BC4 or BC5 source map's channels (0-1) at a mip no larger than size."""
    width, height, mips, dxgi, payload = rc.dds(Path(path))
    block = 16 if dxgi in (83, 84) else 8
    level = 0
    while (width >> level) > size and level + 1 < mips:
        level += 1
    offset, bw, bh = rc.mip_offsets(width, height, mips, block)[level]
    blocks = np.frombuffer(payload[offset:offset + bw * bh * block], np.uint8).reshape(bh, bw, block)
    return [_bc4(blocks[..., 0:8])] + ([_bc4(blocks[..., 8:16])] if block == 16 else [])


def normal_baked(colour: Path, normal: Path, cache: Path, strength: float = 1.0, grain: tuple | None = None) -> Path:
    """A source colour map with its normal map's relief baked in: the kit format carries one
    colour texture, and without the normal map large rock reads soft next to Renderer64's
    normal-mapped mountains. Tilted micro-faces darken (crevices, ledges, cracks) independent
    of light direction, since Renderer64's sun moves through the day."""
    target = Path(cache) / f"{Path(colour).stem}_relief_{Path(normal).stem}_{strength:g}_{'_'.join(map(str, grain or ()))}.dds"
    if target.is_file():
        return target
    base = decode_colour(Path(colour), 512)
    size = base.shape[0]
    nx, ny = (_resize(channel * 2 - 1, size) for channel in source_channels(normal, size))
    tilt = np.hypot(nx, ny)
    # Fine relief only: tilt against its local mean keeps each face's tone.
    shade = np.exp(-3.0 * (tilt - _blur(_blur(tilt, 6), 6)))
    shade = 1 + (shade / np.maximum(_blur(_blur(shade, 6), 6), 1e-3) - 1) * strength
    if grain:
        # Fine skin-rock grain at a higher tiling: crisp texel-scale detail the 512 source lacks.
        material_name, repeat, amount = grain
        rock = material(material_name) @ np.array([.2126, .7152, .0722], np.float32)
        t = (np.arange(size) + .5) / size
        fine = _bilinear(rock, ((t * repeat) % 1 * (rock.shape[0] - 1))[:, None] + 0 * t[None, :],
                         ((t * repeat) % 1 * (rock.shape[1] - 1))[None, :] + 0 * t[:, None])
        shade = shade * (1 + (fine / np.maximum(_blur(fine, 8), 1e-3) - 1) * amount)
    rgba = np.dstack([np.clip(base * np.clip(shade, .35, 1.12)[..., None], 0, 255), np.full((size, size), 255.0)])
    return write_bc3(target, rgba.astype(np.float32))


def _sample(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = texture.shape[:2]
    return texture[(v * h).astype(np.int64) % h, (u * w).astype(np.int64) % w]


def _blur(field: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return field
    kernel = np.ones(2 * radius + 1, np.float32) / (2 * radius + 1)
    out = np.apply_along_axis(lambda row: np.convolve(np.pad(row, radius, mode="edge"), kernel, "valid"), 1, field)
    return np.apply_along_axis(lambda col: np.convolve(np.pad(col, radius, mode="edge"), kernel, "valid"), 0, out)


def _bilinear(field: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Field values at fractional texel coordinates (rows and cols broadcast together)."""
    h, w = field.shape[:2]
    r = np.clip(rows, 0, h - 1)
    c = np.clip(cols, 0, w - 1)
    r0, c0 = np.floor(r).astype(np.int64), np.floor(c).astype(np.int64)
    r1, c1 = np.minimum(r0 + 1, h - 1), np.minimum(c0 + 1, w - 1)
    tr, tc = r - r0, c - c0
    if field.ndim == 3:
        tr, tc = tr[..., None], tc[..., None]
    top = field[r0, c0] * (1 - tc) + field[r0, c1] * tc
    bottom = field[r1, c0] * (1 - tc) + field[r1, c1] * tc
    return top * (1 - tr) + bottom * tr


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / (edge1 - edge0), 0, 1)
    return t * t * (3 - 2 * t)


def _resize(field: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resample of a square field to size x size."""
    n = field.shape[0]
    coords = (np.arange(size) + .5) * n / size - .5
    i0 = np.clip(np.floor(coords).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    t = np.clip(coords - i0, 0, 1)
    rows = field[i0] * (1 - t)[:, None] + field[i1] * t[:, None]
    return rows[:, i0] * (1 - t)[None, :] + rows[:, i1] * t[None, :]


# ---------------------------------------------------------------- relief

def relief_part(key: str, pick: dict, terrain_type: str, cache: Path) -> dict:
    """The source height field as meshes in hex units (x east, y north; one Civ VI hex =
    one tile, the element origin at the anchor hex), at the field's full resolution,
    cropped to its footprint and split into sub-meshes that each carry a full-size bake."""
    fields = source_fields(pick)
    res = fields["height"].shape[0]
    height = fields["height"].astype(np.float32) / 255
    blend = fields.get("blend", np.full_like(fields["height"], 255)).astype(np.float32) / 255
    ids = fields.get("region_ids")
    params = fields["parameters"]
    grid = float(fields["grid"])
    units = grid / res                                     # grid units per field texel
    texel_hex = units / HEX_GRID                           # hex units per field texel
    # Civ VI lerps the element into the neighbouring terrain over the blend ramp; a rigid
    # kit cannot reshape that terrain, so the ramp is widened (about 0.08 hex) and eased
    # so flanks meet the ground instead of ending in a wall at the footprint edge.
    ramp = max(1, int(round(0.04 / texel_hex)))
    split = None
    if pick.get("keep_east"):
        # Only the east part of the source (screen right): the cut eases over 0.08 hex.
        columns = (np.arange(res) + .5) * units / HEX_GRID - grid / 2 / HEX_GRID
        inside = np.nonzero((blend > .5).any(axis=0))[0]
        split = float(columns[inside.min()] + columns[inside.max()]) / 2
        blend = blend * _smoothstep(split - .04, split + .04, columns)[None, :]
    blend = _smoothstep(0.0, 1.0, np.minimum(blend, _blur(blend, ramp)))
    # True Civ VI proportions (64 world units and 256 grid units per hex); the 8-bit
    # field is softened by one binomial pass so quantization steps do not facet the light.
    raw = (params.get("base_height", 0.0) + height * params["height_scale"]) / HEX_UNITS * VERTICAL
    z = raw.copy()
    z[1:-1, 1:-1] = (4 * raw[1:-1, 1:-1] + raw[:-2, 1:-1] + raw[2:, 1:-1] + raw[1:-1, :-2] + raw[1:-1, 2:]) / 8
    sea = pick.get("sea")
    if pick.get("submerged"):
        # A reef: its tops just break the water surface, like C3X's reef islets
        # (Renderer64's water hides anything fully submerged).
        z = z - float(z[blend > .5].max()) + 0.008
    elif sea is not None:
        low = z[(ids == sea) if (ids is not None and sea) else (blend < .2)]
        z = z - (np.percentile(low, 70) if low.size else 0.0)
    else:
        rim = z[(blend > .02) & (blend < .2)]
        z = z - (np.median(rim) if rim.size else 0.0)
        if pick.get("lake") is not None or (pick.get("pool") and not STAMP):
            z = z - min(0.0, float(z[blend > .5].min()))
    z = z * pick.get("rise", 1.0)
    if pick.get("bulk"):
        # A broader massif under the peaks at the same height: a smooth body rising to a
        # share of the peak, so slender summits stand on a heavier mountain.
        radius, share = pick["bulk"]
        width = max(1, int(round(radius / texel_hex)))
        body = _blur(_blur(np.maximum(z, 0) * (blend > .3), width), width)
        z = np.maximum(z, body * (float(z.max()) * share / max(float(body.max()), 1e-6)))
    surface = blend * z - (1 - blend) * 0.025
    if STAMP and not pick.get("lake") and not pick.get("pool") and sea is None:
        # Civ VI eases an element into its (often raised) neighbours; on flat C3X ground
        # the cut-off flank reads as a drum. A smooth apron outside the footprint gives
        # the wonder a natural foot without changing the authored core.
        core_height = np.maximum(blend * z, 0)
        apron = _blur(_blur(core_height, max(1, int(round(0.07 / texel_hex)))), max(1, int(round(0.07 / texel_hex))))
        surface = np.maximum(surface, core_height + apron * (1 - blend) * 0.9)
    if pick.get("rock_relief"):
        # The skin rock's height detail as geometry: the kit shader has no normal maps, so
        # lit facets and crevices come from the mesh, as Renderer64's mountains get them
        # from the rock normal map (at the production rock tiling, a little coarser).
        axis = (np.arange(res) + .5) * units / HEX_GRID - grid / 2 / HEX_GRID
        ys_, xs_ = np.nonzero(blend > .5)
        tiles = FIT_TILES / max(np.ptp(axis[xs_]), np.ptp(axis[ys_])) * pick.get("enlarge", 1.0)
        rock = _blur(material_height("mtn_base"), 3)
        repeat = ROCK_REPEAT * .5 * tiles
        detail = _sample(rock, (axis[None, :] * repeat) % 1 + 0 * axis[:, None], (axis[:, None] * repeat) % 1 + 0 * axis[None, :])
        surface = surface + pick["rock_relief"] * (detail - float(detail.mean())) * np.clip(surface / .02, 0, 1) * blend
    gy, gx = np.gradient(surface, texel_hex, texel_hex)
    slope = np.hypot(gx, gy)
    cavity = np.clip((_blur(surface, 4) - surface) / 0.02, -1, 1)
    hex_axis = (np.arange(res) + .5) * units / HEX_GRID - grid / 2 / HEX_GRID

    host_name = HOST_MATERIALS.get(terrain_type, "grassland")
    present = [int(v) for v in np.unique(ids) if v] if ids is not None else []
    one_hot = {value: _blur((ids == value).astype(np.float32), 1) for value in present}
    names = {}
    for value in present:
        name = host_name if value == 13 else REGION_MATERIALS.get(value)
        if name is None:
            name = "mtn_top" if float(height[ids == value].mean()) > .6 else "mtn_base"
        names[value] = name

    # The kit's final tiles per hex (kit() fits the blend core to FIT_TILES), so baked
    # textures repeat at the production terrain's density on screen.
    ys, xs = np.nonzero(blend > .5)
    tiles_per_hex = FIT_TILES / max(np.ptp(hex_axis[xs]), np.ptp(hex_axis[ys])) * pick.get("enlarge", 1.0)
    # Crop to the footprint and split into sub-meshes of at most SUB_TEXELS field texels.
    rows, cols = np.nonzero(blend > .04)
    r0, r1 = max(0, rows.min() - 2), min(res - 1, rows.max() + 2)
    c0, c1 = max(0, cols.min() - 2), min(res - 1, cols.max() + 2)
    span = max(r1 - r0, c1 - c0) * texel_hex
    parts = max(1, math.ceil(span * pick.get("bake", BAKE_PER_HEX) / TEXTURE))
    edges_r = np.linspace(r0, r1, parts + 1).round().astype(int)
    edges_c = np.linspace(c0, c1, parts + 1).round().astype(int)
    meshes = []
    for i in range(parts):
        for j in range(parts):
            ra, rb, ca, cb = edges_r[i], edges_r[i + 1], edges_c[j], edges_c[j + 1]
            if not (blend[ra:rb + 1, ca:cb + 1] > .04).any():
                continue
            mesh = _relief_piece(f"{key}_{i}{j}", ra, rb, ca, cb, surface, blend, slope, cavity, hex_axis,
                                 texel_hex, ids, one_hot, names, host_name, pick, cache, tiles_per_hex)
            if mesh is not None:
                meshes.append(mesh)
    water, scatter_level = None, None
    if pick.get("lake") is not None and ids is not None:
        water = ids == pick["lake"]
    elif pick.get("pool"):
        water = (z < np.percentile(z[blend > .5], 8)) & (blend > .5)
    elif sea and ids is not None and terrain_type not in ("coast", "sea"):
        water = ids == sea
        if pick.get("open_mouth"):
            # The channel's water runs on south from its mouth (rows run south to north)
            # across the shore, so the fjord opens onto the coast as in Civ VI.
            rows = np.nonzero(water.any(axis=1))[0]
            tail = rows[:max(3, len(rows) // 6)]                  # the channel's southmost rows
            width = int(np.median([water[r].sum() for r in tail]))
            middle = float(np.mean([np.nonzero(water[r])[0].mean() for r in tail]))
            reach = int(round(pick["open_mouth"] / texel_hex))
            water = water.copy()
            water[max(0, rows.min() - reach):rows.min(),
                  max(0, int(round(middle - width / 2))):int(round(middle + width / 2))] = True
    if water is not None and water.any():
        level = float(np.percentile(z[water], 85)) if not sea else 0.01
        if STAMP and pick.get("lake") is not None:
            level = pick.get("lake_level", float(np.percentile(z[water], 80)) + 0.006)
        if STAMP and pick.get("water_level") is not None:
            level = pick["water_level"]
        scatter_level = level
        if STAMP and pick.get("pool"):
            level = 0.004                                   # the hollow stays at ground level
        if not (STAMP and pick.get("effect_water")):
            meshes.append(water_mesh(key, water, level, hex_axis, cache))
    ys, xs = np.nonzero(blend > .5)
    core = (hex_axis[xs].min(), hex_axis[xs].max(), hex_axis[ys].min(), hex_axis[ys].max())

    def ground(x: float, y: float) -> float:
        c = int(np.clip(round((x - hex_axis[0]) / texel_hex), 0, res - 1))
        r = int(np.clip(round((y - hex_axis[0]) / texel_hex), 0, res - 1))
        return float(surface[r, c])
    scatter = {"axis": hex_axis, "texel": texel_hex, "surface": surface, "blend": blend, "slope": slope,
               "ids": ids, "height": height, "water": water,
               "water_level": scatter_level if water is not None and water.any() else None}
    return {"meshes": meshes, "core": core, "ground": ground, "peak": float(z.max()), "regions": present,
            "scatter": scatter, "split": split}


def _relief_piece(name, ra, rb, ca, cb, surface, blend, slope, cavity, hex_axis, texel_hex, ids, one_hot, names,
                  host_name, pick, cache, tiles_per_hex) -> dict | None:
    """One relief sub-mesh and its bake. Colour follows Renderer64's mountain recipe so a
    wonder sits beside production mountains: Civ V skin textures at the production tiling,
    snow only on gentle faces, crevice and grain from the rock height, rock tone and the
    cool skylight grade. Flat host ground is cut out so the real Renderer64 ground shows."""
    size = TEXTURE
    t = (np.arange(size) + .5) / size
    rr = ra + t[:, None] * (rb - ra) + 0 * t[None, :]
    cc = ca + t[None, :] * (cb - ca) + 0 * t[:, None]
    wx = (hex_axis[0] + cc * texel_hex) * tiles_per_hex   # tile units, as the shader's world
    wy = (hex_axis[0] + rr * texel_hex) * tiles_per_hex
    gu, gv = wx * GROUND_REPEAT, wy * GROUND_REPEAT
    ru, rv = wx * ROCK_REPEAT, wy * ROCK_REPEAT
    b = _bilinear(blend, rr, cc)
    z = _bilinear(surface, rr, cc)
    steep = _bilinear(slope, rr, cc)
    nz = 1 / np.sqrt(1 + steep * steep)
    cav = _bilinear(cavity, rr, cc)
    colour = _sample(material(host_name), gu, gv).astype(np.float32)
    rocky = np.zeros((size, size), np.float32)              # rock-material coverage
    snowy = np.zeros((size, size), np.float32)
    detail = np.full((size, size), .5, np.float32)
    if one_hot:
        total = np.maximum(_bilinear(sum(one_hot.values()), rr, cc), 1e-6)
        coverage = np.clip(total, 0, 1)
        mixed = np.zeros_like(colour)
        mixed_detail = np.zeros((size, size), np.float32)
        for value, mask in one_hot.items():
            weight = _bilinear(mask, rr, cc) / total
            name_v = names[value]
            if name_v == "mtn_snow":
                # Production rule: snow holds only on gentle faces; the rest is the patchy top rock.
                snow = weight * _smoothstep(*pick.get("snow_slope", (0.45, 0.8)), nz)
                parts = ((snow, "mtn_snow"), (weight - snow, "mtn_top"))
                snowy += snow * coverage
            else:
                parts = ((weight, name_v),)
            for share, layer in parts:
                rock = layer.startswith("mtn_") or layer in ("cliff", "cliff_white")
                u, v = (ru, rv) if rock else (gu, gv)
                mixed += share[..., None] * _sample(material(layer), u, v)
                mixed_detail += share * _sample(material_height(layer), u, v)
                if rock:
                    rocky += share * coverage
        colour = colour * (1 - coverage[..., None]) + mixed * coverage[..., None]
        detail = detail * (1 - coverage) + mixed_detail * coverage
    # Steep faces of other regions expose the cliff rock (Civ VI cliff blending).
    cliff = _smoothstep(1.4, 2.6, steep) * (1 - rocky) * (1 - snowy)
    rock_name = pick.get("cliffs", "mtn_base")
    colour = colour * (1 - cliff[..., None]) + _sample(material(rock_name), ru, rv) * cliff[..., None]
    detail = detail * (1 - cliff) + _sample(material_height(rock_name), ru, rv) * cliff
    rocky = np.clip(rocky + cliff, 0, 1)
    if pick.get("detail_bake"):
        # Renderer64 lights rock through its height-derived normal map; the kit shader has none,
        # so the rock height's fine tilt darkens crevices and ledges here (light-independent).
        dy_, dx_ = np.gradient(detail)
        tilt = np.hypot(dx_, dy_) * 18
        relief_shade = np.exp(-(tilt - _blur(_blur(tilt, 6), 6)))
        relief_shade = np.clip(relief_shade / np.maximum(_blur(_blur(relief_shade, 6), 6), 1e-3), .4, 1.12)
        colour = colour * (1 + (relief_shade - 1) * pick["detail_bake"] * np.clip(rocky + snowy * .5, 0, 1))[..., None]
    # Rock: micro relief, fine grain normalized against its local mean, tone and skylight grade.
    rock_detail = detail
    micro = _smoothstep(0.16, 0.84, rock_detail)
    crevice = (0.919 * (1 - micro) + 1.027 * micro)
    fine = _sample(material(rock_name), ru * 3.7 + .31, rv * 3.7 + .17) @ np.array([.2126, .7152, .0722], np.float32)
    mean = _blur(fine, 6)
    grain = np.clip(1 + 2.0 * (fine - mean) / np.maximum(mean, 5.0), 0.5, 1.18)
    rock_gain = crevice * (1 + (grain - 1) * 0.45 * (1 - snowy)) * (1.15 * (1 - snowy) + snowy)
    colour = colour * (1 + (rock_gain - 1) * rocky)[..., None]
    luma = colour @ np.array([.2126, .7152, .0722], np.float32)
    graded = (colour * 0.72 + luma[..., None] * 0.28) * np.array([0.96, 1.0, 1.07], np.float32)
    colour = colour * (1 - rocky[..., None]) + graded * rocky[..., None]
    # Large-scale occlusion from the relief itself (crevices darken, ridges lift slightly).
    occlusion = np.clip(1 - np.clip(cav, 0, 1) * .30 + np.clip(-cav, 0, 1) * .06, .65, 1.08)
    colour = np.clip(colour * occlusion[..., None], 0, 255)
    # Cut out flat host ground so Renderer64's own ground and its transitions show through.
    host_only = (1 - rocky) * (1 - snowy)
    if one_hot:
        host_only = host_only * (1 - np.clip(_bilinear(sum((m for v, m in one_hot.items() if names[v] != host_name),
                                                           np.zeros_like(blend)), rr, cc), 0, 1))
    flat_ground = (host_only > .9) & (z < 0.012) & (steep < 0.35) & (not pick.get("submerged"))
    if pick.get("submerged"):
        flat_ground = z < -0.004                            # only the reef's crests break the surface
    alpha = np.where((b > .08) & ~flat_ground, 255.0, 0.0)
    if not (alpha > 0).any():
        return None
    texture = write_bc3(cache / f"{name}_relief.dds", np.dstack([colour, alpha]).astype(np.float32))

    # Mesh: one vertex per field texel inside the piece (stepped only above SUB_TEXELS).
    step = max(1, math.ceil(max(rb - ra, cb - ca) / SUB_TEXELS))
    rs = list(range(ra, rb, step)) + [rb]
    cs = list(range(ca, cb, step)) + [cb]
    gy, gx = np.gradient(surface, texel_hex, texel_hex)
    vertex_index = {}
    vertices, indices = [], []
    keep = blend > .05
    for r in rs:
        for c in cs:
            if keep[max(0, r - step):r + step + 1, max(0, c - step):c + step + 1].any():
                normal = np.array([-gx[r, c], -gy[r, c], 1.0])
                normal /= np.linalg.norm(normal)
                vertex_index[(r, c)] = len(vertices)
                vertices.append(([float(hex_axis[c]), float(hex_axis[r]), float(surface[r, c])],
                                 [float(v) for v in normal], [(c - ca) / (cb - ca), (r - ra) / (rb - ra)]))
    for a in range(len(rs) - 1):
        for d in range(len(cs) - 1):
            corners = [(rs[a], cs[d]), (rs[a], cs[d + 1]), (rs[a + 1], cs[d]), (rs[a + 1], cs[d + 1])]
            if any(corner not in vertex_index for corner in corners):
                continue
            if not keep[rs[a]:rs[a + 1] + 1, cs[d]:cs[d + 1] + 1].any():
                continue
            p, q, s, u = (vertex_index[corner] for corner in corners)
            indices += [p, q, s, q, u, s]
    if not indices:
        return None
    return {"key": f"nw/{name}/relief", "texture": texture, "masked": True, "vertices": vertices,
            "indices": indices, "pivot": [0.0, 0.0, 0.0], "yaw": 0.0, "scale": 1.0, "follow": False}


def water_mesh(key: str, water: np.ndarray, level: float, hex_axis: np.ndarray, cache: Path) -> dict:
    """A flat still-water surface over the source's lake, pool or channel (a static Lab stand-in for Civ VI water)."""
    rows, cols = np.nonzero(water)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    side = max(r1 - r0, c1 - c0)
    mask = np.zeros((side, side), np.float32)
    mask[:r1 - r0, :c1 - c0] = water[r0:r1, c0:c1]
    size = 128
    cover = _blur(_resize(mask, size), 1)
    # Shallow, lighter water at the shore deepening toward the middle, with faint ripples.
    depth = _smoothstep(.45, .97, _blur(_resize(mask, size), 6))
    ripple = _blur(np.random.default_rng(5).uniform(-1, 1, (size, size)).astype(np.float32), 2)
    shallow, deep = np.array([74, 134, 138], np.float32), np.array([34, 86, 110], np.float32)
    colour = shallow + (deep - shallow) * depth[..., None] + ripple[..., None] * 9
    rgba = np.dstack([colour, np.where(cover > .45, 255.0, 0.0)])
    texture = write_bc3(cache / f"{key}_water.dds", rgba.astype(np.float32))
    step = hex_axis[1] - hex_axis[0]
    x0, y0 = hex_axis[c0] - step / 2, hex_axis[r0] - step / 2
    x1, y1 = x0 + side * step, y0 + side * step
    vertices = [([x0, y0, level], [0, 0, 1], [0, 0]), ([x1, y0, level], [0, 0, 1], [1, 0]),
                ([x0, y1, level], [0, 0, 1], [0, 1]), ([x1, y1, level], [0, 0, 1], [1, 1])]
    return {"key": f"nw/{key}/water", "texture": texture, "masked": True, "vertices": vertices,
            "indices": [0, 1, 2, 1, 3, 2], "pivot": [0.0, 0.0, 0.0], "yaw": 0.0, "scale": 1.0, "follow": False}


# ---------------------------------------------------------------- source meshes

def _block_rows(payload: bytes, offset: int, bw: int, bh: int, block: int) -> list[bytes]:
    return [payload[offset + row * bw * block: offset + (row + 1) * bw * block] for row in range(bh)]


def square(path: Path, cache: Path, block: int) -> tuple[Path, float, float]:
    """An atlas-ready square texture of 128-512 texels with a full mip chain (block copies,
    no recompression; the short side repeats). Returns the texture and its UV scale."""
    width, height, mips, dxgi, payload = rc.dds(path)
    if width == height and 128 <= width <= rc.CELL and mips >= (width // 4).bit_length():
        return path, 1.0, 1.0
    layout = rc.mip_offsets(width, height, mips, block)
    skip = 0
    while max(width, height) >> skip > rc.CELL:
        skip += 1
    size = max(128, max(width, height) >> skip)
    levels = []
    for level in range((size // 4).bit_length()):
        source = min(skip + level, len(layout) - 1)
        offset, bw, bh = layout[source]
        rows = _block_rows(payload, offset, bw, bh, block)
        side = max(1, (size >> level) // 4)
        levels.append(b"".join((rows[row % bh] * (side // bw + 1))[:side * block] for row in range(side)))
    target = rc.write_dds(cache / f"sq{size}_{path.parent.name}_{path.name}", size, len(levels), dxgi, block, levels)
    return target, (width >> skip) / size, (height >> skip) / size


def _opaque_bc3(path: Path, cache: Path) -> Path:
    target = cache / f"opaque_{path.parent.name}_{path.name}"
    if not target.is_file():
        data = bytearray(path.read_bytes())
        for offset in range(148, len(data) - 15, 16):
            data[offset:offset + 8] = b"\xff\xff\x00\x00\x00\x00\x00\x00"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))
    return target


def source_meshes(key: str, compiler: "assets.Compiler", item: str | None, package: str, entry: str,
                  cache: Path) -> list[dict]:
    """The TerrainAsset (or clutter model) tree as meshes in hex units, each at its own pivot."""
    out, built = [], {}
    for draw in assets.placed_meshes(compiler, compiler.root(item, package, entry)):
        mesh_key = f"nw/{draw['geometry'].rsplit('/', 1)[-1].removesuffix('.json')}"
        if not draw["fitted"]:
            mesh_key += "/" + format(abs(hash(tuple(round(v, 4) for v in draw["linear"]))) % 16 ** 8, "08x")
        if mesh_key not in built:
            linear = draw["linear"]
            mesh = assets.load(assets.SOURCES / draw["geometry"])
            vertices = []
            for vertex in mesh["vertices"]:
                x, y, z0 = vertex["position"]
                nx, ny, nz = vertex["normal"]
                position = [(x * linear[0] + y * linear[4] + z0 * linear[8]) * ASSET_TO_HEX,
                            (x * linear[1] + y * linear[5] + z0 * linear[9]) * ASSET_TO_HEX,
                            (x * linear[2] + y * linear[6] + z0 * linear[10]) * ASSET_TO_HEX * PICKS[key].get("vertical", VERTICAL)]
                normal = [nx * linear[0] + ny * linear[4] + nz * linear[8], nx * linear[1] + ny * linear[5] + nz * linear[9],
                          nx * linear[2] + ny * linear[6] + nz * linear[10]]
                length = math.sqrt(sum(v * v for v in normal)) or 1.0
                vertices.append((position, [v / length for v in normal], list(vertex["uv0"])))
            indices = mesh["topology"]["indices"]
            if any(value < 0 or value > 1 for _, _, uv in vertices for value in uv):
                vertices, indices = rc.within_texture(vertices, indices)
            colour, opacity = draw["colour"], draw["opacity"]
            if opacity is not None and rc.fully_opaque(opacity):
                opacity = None
            masked = False
            if rc.dds(colour)[3] in (77, 78):
                texture, su, sv = square(_opaque_bc3(colour, cache), cache, 16)
                masked = True
            elif opacity is not None:
                try:
                    texture, su, sv = square(rc.masked(colour, opacity, cache), cache, 16)
                    masked = True
                except ValueError:
                    texture, su, sv = square(colour, cache, 8)
            else:
                texture, su, sv = square(colour, cache, 8)
            if PICKS[key].get("normal_bake") and draw.get("normal") is not None and not masked and len(vertices) > 500:
                texture, su, sv = square(normal_baked(colour, draw["normal"], cache, PICKS[key]["normal_bake"],
                                                      PICKS[key].get("grain")), cache, 16)
                masked = True
            built[mesh_key] = {"key": mesh_key, "texture": texture, "masked": masked, "indices": indices,
                               "vertices": [(p, nn, [uv[0] * su, uv[1] * sv]) for p, nn, uv in vertices]}
        pivot = [v * ASSET_TO_HEX for v in draw["pivot"]]
        pivot[2] *= PICKS[key].get("vertical", VERTICAL)
        out.append({**built[mesh_key], "pivot": pivot,
                    "yaw": draw["yaw"], "scale": draw["scale"], "follow": True})
    return out


# ---------------------------------------------------------------- clutter

def clutter_parts(key: str, compiler, scatter: dict, fold: float, budget: int, cache: Path) -> list[dict]:
    """The wonder's clutter trees scattered deterministically over vegetated ground and
    lower rock on gentle slopes (never on snow, high rock or water), each standing on
    the relief. Trees keep Renderer64's forest scale rather than the kit's fit."""
    entries, wanted = CLUTTER.get(key, ((), 0))
    count = min(wanted, budget)
    if not count:
        return []
    ids, blend, slope, height = scatter["ids"], scatter["blend"], scatter["slope"], scatter["height"]
    lower = height < np.percentile(height[blend > .5], 55)
    allowed = blend > .6
    if ids is not None:
        allowed &= (np.isin(ids, (0, 13)) & (slope < 1.0)) | ((ids == 115) & lower & (slope < 1.5))
    else:
        allowed &= lower
    if scatter["water"] is not None:
        allowed &= ~_blur(scatter["water"].astype(np.float32), 3).astype(bool)
    rows, cols = np.nonzero(allowed)
    if not rows.size:
        return []
    rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
    order = rng.permutation(rows.size)
    spacing = 0.04
    chosen = []
    axis = scatter["axis"]
    for index in order:
        x, y = float(axis[cols[index]]), float(axis[rows[index]])
        if all((x - a) ** 2 + (y - b) ** 2 >= spacing ** 2 for a, b in chosen):
            chosen.append((x, y))
            if len(chosen) >= count:
                break
    if SEASON == 2:
        # Winter: the skin's authored snowy pines (same bodies, snow material), as Renderer64's
        # winter stands use; shrubs have no snowy twin and keep shader snow.
        assets_ = json.loads((VEGETATION / "manifest.json").read_text())["assets"]
        entries = [(entry.replace("forest/", "forest_snow/")
                    if f"feature/{entry.replace('forest/', 'forest_snow/')}" in assets_ else entry, base)
                   for entry, base in entries]
    models = {entry: vegetation_mesh(entry, cache) for entry, _ in entries}
    out = []
    for number, (x, y) in enumerate(chosen):
        entry, base = entries[number % len(entries)]
        size = base * TREE_SCALE * float(rng.uniform(0.85, 1.15)) / fold
        out.append({**models[entry], "pivot": [x, y, 0.0], "yaw": float(rng.uniform(0, 2 * math.pi)),
                    "scale": size, "follow": True})
    return out


# A rainforest canopy of Renderer64's skin jungle trees: emergent giants in the middle, a
# stepped dome of leafy trees and palms around them, understory plants at the edge.
# Rings: radius (tiles), count, size over forest scale, models.
CANOPY = ((0.0, 1, 2.0, ("jungle/palm_03",)),
          (0.17, 7, 1.95, ("jungle/leafy_01", "jungle/leafy_02", "jungle/palm_02", "jungle/leafy_02", "jungle/leafy_01",
                           "jungle/palm_03", "jungle/leafy_02")),
          (0.32, 12, 1.6, ("jungle/palm_01", "jungle/leafy_02", "jungle/palm_02", "jungle/leafy_01")),
          (0.44, 18, 1.15, ("jungle/leafy_01", "jungle/plant_01", "jungle/palm_01", "jungle/leafy_02", "jungle/plant_02")))


def canopy_parts(cache: Path) -> list[dict]:
    rng = np.random.default_rng(11)
    out = []
    for radius, count, size, models in CANOPY:
        for n in range(count):
            angle = 2 * math.pi * (n + rng.uniform(-.25, .25)) / count + radius * 7
            distance = radius * rng.uniform(.8, 1.15) if radius > .1 else radius * rng.uniform(0, 1.2)
            mesh = vegetation_mesh(models[n % len(models)], cache)
            out.append({**mesh, "pivot": [distance * math.cos(angle), distance * math.sin(angle), 0.0],
                        "yaw": float(rng.uniform(0, 2 * math.pi)), "scale": size * TREE_SCALE * float(rng.uniform(.9, 1.1)),
                        "follow": True})
    return out


@lru_cache(maxsize=None)
def _vegetation_cached(entry: str, cache: str) -> dict:
    asset = json.loads((VEGETATION / "manifest.json").read_text())["assets"][f"feature/{entry}"]
    mesh = json.loads((VEGETATION / asset["mesh"]).read_text())
    material = json.loads((VEGETATION / asset["material"]).read_text())
    colour = VEGETATION / material["base_color"]["texture"]
    opacity = material.get("opacity", {}).get("texture")
    if opacity:
        texture, su, sv = square(rc.masked(colour, VEGETATION / opacity, Path(cache)), Path(cache), 16)
    else:
        texture, su, sv = square(colour, Path(cache), 8)
    vertices = [(list(v["position"]), list(v["normal"]), [v["uv0"][0] * su, v["uv0"][1] * sv]) for v in mesh["vertices"]]
    return {"key": f"nw/vegetation/{entry}", "texture": texture, "masked": bool(opacity), "vertices": vertices,
            "indices": list(mesh["topology"]["indices"])}


def gritty(texture: Path, material: str, repeat: float, cache: Path) -> Path:
    """A source texture's colour with a Civ V skin rock's grain and crevices tiled over it, for
    low-detail rocks whose grit Civ VI draws from detail maps the kit does not carry."""
    target = Path(cache) / f"{Path(texture).stem}_grit_{material}_{repeat:g}.dds"
    if target.is_file():
        return target
    base = decode_colour(Path(texture), 512)
    rock = decode_colour(MATERIALS / f"{material}_base_color.dds", 512)
    size = 512
    t = (np.arange(size) + .5) / size
    grain = _bilinear(rock, ((t * repeat) % 1 * (rock.shape[0] - 1))[:, None] + 0 * t[None, :],
                      ((t * repeat) % 1 * (rock.shape[1] - 1))[None, :] + 0 * t[:, None])
    colour = _bilinear(base, (t * (base.shape[0] - 1))[:, None] + 0 * t[None, :], (t * (base.shape[1] - 1))[None, :] + 0 * t[:, None])
    light = grain.mean(-1, keepdims=True)
    detail = np.clip(light / max(float(light.mean()), 1e-3), .3, 1.8) ** 1.7
    rgba = np.dstack([np.clip(colour * detail, 0, 255), np.full((size, size), 255.0)])
    return write_bc3(target, rgba.astype(np.float32))


RESOURCE_SOURCES = ROOT / "Renderer/packs/ResourceCompositionSources"


def wheat_grass(entry: str, cache: Path, height: float = 1.0) -> dict:
    """A wheat tuft from Renderer64's resource sources (alpha-cut stalks) as savanna grass:
    its own colour paled toward dry straw."""
    asset = json.loads((RESOURCE_SOURCES / "manifest.json").read_text())["assets"][f"source/{entry}"]
    mesh = json.loads((RESOURCE_SOURCES / asset["mesh"]).read_text())
    material = json.loads((RESOURCE_SOURCES / asset["material"]).read_text())
    target = Path(cache) / f"straw2_{entry}.dds"
    if not target.is_file():
        colour = decode_colour(RESOURCE_SOURCES / material["base_color"]["texture"], 256)
        alpha = _resize(source_channels(RESOURCE_SOURCES / material["opacity"]["texture"], 256)[0], colour.shape[0])
        luma = colour @ np.array([.2126, .7152, .0722], np.float32)
        light = np.clip(luma / max(float(luma[alpha > .5].mean()), 1.0), .55, 1.35)[..., None]
        straw = light * np.array([205, 172, 96], np.float32)            # the savanna plains' gold
        write_bc3(target, np.dstack([np.clip(straw, 0, 255), np.where(alpha > .5, 255.0, 0.0)]).astype(np.float32))
    texture, su, sv = square(target, Path(cache), 16)
    vertices = [([v["position"][0], v["position"][1], v["position"][2] * height], list(v["normal"]),
                 [v["uv0"][0] * su, v["uv0"][1] * sv]) for v in mesh["vertices"]]
    return {"key": f"nw/savanna/{entry}", "texture": texture, "masked": True, "vertices": vertices,
            "indices": list(mesh["topology"]["indices"])}


def wave_ribbons(field: np.ndarray, axis: np.ndarray, level: float, cx: float, cy: float, fold: float,
                 facing: float, scale: float, half: float = .9, size: int = 160) -> list[float]:
    """Renderer64 coastal-wave ribbons round a water wonder's islets (field above `level`), in
    the layout coastal_waves.h builds along a coast: 32 rows along the waterline, 6 columns out
    to .9 of the shader's distance. Distances are scaled by `scale`, so a breaker rolls in from
    .625 x scale tiles out: small, gentle waves for small islets. Points are natural offsets from
    the wonder's tile centre, as a triangle list of (x, y, distance, along, coverage)."""
    offsets = np.linspace(-half, half, size)
    dx, dy = np.meshgrid(offsets, offsets)
    c, s_ = math.cos(facing), math.sin(facing)
    lx, ly = c * dx - s_ * dy, -s_ * dx - c * dy                 # inverse of the kit placement
    step = axis[1] - axis[0]
    cols, rows = (cx + lx / fold - axis[0]) / step, (cy + ly / fold - axis[0]) / step
    inside = (cols >= 0) & (rows >= 0) & (cols <= len(axis) - 1) & (rows <= len(axis) - 1)
    land = inside & (_bilinear(field, rows, cols) > level)
    grid = offsets[1] - offsets[0]
    # Water distance to the nearest rock (tiles), for folding checks.
    points = np.stack([dx[land], dy[land]], 1)
    flat = np.stack([dx.ravel(), dy.ravel()], 1)
    water = np.full(flat.shape[0], 9.0, np.float32)
    for start in range(0, len(points), 512):
        chunk = points[start:start + 512]
        water = np.minimum(water, np.sqrt(((flat[:, None, :] - chunk[None, :, :]) ** 2).sum(-1)).min(1))
    water = water.reshape(land.shape)

    def water_at(x, y):
        return float(_bilinear(water, np.array([[(y + half) / grid]]), np.array([[(x + half) / grid]]))[0, 0])
    # Islets: connected land components (4-neighbour flood fill).
    labels, count = np.zeros(land.shape, int), 0
    for start in zip(*np.nonzero(land)):
        if labels[start]:
            continue
        count += 1
        stack = [start]
        labels[start] = count
        while stack:
            r, q = stack.pop()
            for nr, nq in ((r + 1, q), (r - 1, q), (r, q + 1), (r, q - 1)):
                if 0 <= nr < size and 0 <= nq < size and land[nr, nq] and not labels[nr, nq]:
                    labels[nr, nq] = count
                    stack.append((nr, nq))
    out = []
    rows_n, columns = 32, 6
    for label in range(1, count + 1):
        mask = labels == label
        if mask.sum() * grid * grid < .00008:
            continue                                            # specks
        ox, oy = float(dx[mask].mean()), float(dy[mask].mean())
        # The waterline by rays from the islet's centre (islets are near star-shaped).
        radii = []
        for k in range(rows_n):
            angle = 2 * math.pi * k / rows_n
            ux, uy = math.cos(angle), math.sin(angle)
            r_land = 0.0
            for r_ in np.arange(0, .6, grid * .5):
                x, y = ox + ux * r_, oy + uy * r_
                i, j = int(round((y + half) / grid)), int(round((x + half) / grid))
                if 0 <= i < size and 0 <= j < size and labels[i, j] == label:
                    r_land = r_
            radii.append(r_land + grid * .5)
        radii = np.array(radii)
        radii = (np.roll(radii, 1) + 2 * radii + np.roll(radii, -1)) / 4
        feet = [(ox + math.cos(2 * math.pi * k / rows_n) * radii[k], oy + math.sin(2 * math.pi * k / rows_n) * radii[k])
                for k in range(rows_n)]
        # Short stretches, as along a coast: each its own ribbon (the breaker art spans one
        # ribbon) with its own seed, so the waves round an islet lap out of step.
        perimeter = sum(math.hypot(feet[(k + 1) % rows_n][0] - feet[k][0], feet[(k + 1) % rows_n][1] - feet[k][1])
                        for k in range(rows_n))
        stretches = max(1, min(4, int(round(perimeter / (.32 * scale + .08)))))
        ring = []
        for k in range(rows_n + 1):
            fx, fy = feet[k % rows_n]
            px, py = feet[(k - 1) % rows_n]
            nx_, ny_ = feet[(k + 1) % rows_n]
            tx, ty = nx_ - px, ny_ - py
            length = math.hypot(tx, ty) or 1.0
            nx, ny = ty / length, -tx / length
            if (fx - ox) * nx + (fy - oy) * ny < 0:
                nx, ny = -nx, -ny                                # outward
            column = []
            for col in range(columns + 1):
                d = .002 + .90 * col / columns
                x, y = fx + nx * d * scale, fy + ny * d * scale
                actual = water_at(x, y) / scale
                mask_ = 1.0 if abs(actual - d) < .10 + grid / scale and actual > 0 or col == 0 else 0.0
                column.append([x, y, d, 0.0, mask_])
            ring.append(column)
        per = rows_n // stretches
        for part in range(stretches):
            seed = (label * .618 + part * .377 + scale * 3.1) % 1
            for k in range(part * per, rows_n if part == stretches - 1 else (part + 1) * per):
                span = (rows_n - part * per) if part == stretches - 1 else per
                for col in range(columns):
                    corners = []
                    for kk, cc in ((k, col), (k, col + 1), (k + 1, col), (k + 1, col + 1)):
                        x, y, d, _, m = ring[kk][cc]
                        along = (kk - part * per) / span
                        corners.append((x, y, d, along, (2 + seed) * (1 if m else -1)))
                    a, b, d_, e = corners
                    for point in (a, b, e, a, e, d_):
                        out.extend(point)
    return out


def _oklch(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB to OKLab lightness, chroma and hue (degrees), as Renderer64's season code."""
    c = np.maximum(rgb, 0)
    lms = np.cbrt(np.maximum(c @ np.array([[0.4122214708, 0.2119034982, 0.0883024619],
                                             [0.5363325363, 0.6806995451, 0.2817188376],
                                             [0.0514459929, 0.1073969566, 0.6299787005]], np.float32), 1e-9))
    lab = lms @ np.array([[0.2104542553, 1.9779984951, 0.0259040371],
                          [0.7936177850, -2.4285922050, 0.7827717662],
                          [-0.0040720468, 0.4505937099, -0.8086757660]], np.float32)
    return np.stack([lab[..., 0], np.hypot(lab[..., 1], lab[..., 2]), np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))], -1)


def _from_oklch(lch: np.ndarray) -> np.ndarray:
    h = np.radians(lch[..., 2])
    lab = np.stack([lch[..., 0], lch[..., 1] * np.cos(h), lch[..., 1] * np.sin(h)], -1)
    lms = lab @ np.array([[1, 1, 1], [0.3963377774, -0.1055613458, -0.0894841775],
                          [0.2158037573, -0.0638541728, -1.2914855480]], np.float32)
    return np.maximum((lms ** 3) @ np.array([[4.0767416621, -1.2684380046, -0.0041960863],
                                              [-3.3077115913, 2.6097574011, -0.7034186147],
                                              [0.2309699292, -0.3413193965, 1.7076147010]], np.float32), 0)


# Renderer64's seasonal crown palettes (Renderer/lab/shared/seasons autumn.hlsl and spring.hlsl
# autumn_leaf_color / spring_leaf_color): (lightness, chroma, hue) by tree seed, the leaf
# chroma scale and hue spread. Natural trees get these in objects.hlsl; composition foliage
# (the wonders' own trees and plants) does not, so the kits bake them.
SEASON_CROWNS = {
    1: ([(.34, (.75, .15, 84)), (.48, (.79, .14, 92)), (.6, (.7, .15, 70)), (.82, (.67, .155, 59)),
         (.88, (.6, .13, 54)), (.93, (.51, .125, 47)), (1.01, (.62, .09, 104))], .92, 10.0, .8),
    3: ([(.45, (.6, .13, 127)), (.85, (.58, .125, 130)), (1.01, (.63, .135, 124))], 1.0, 5.0, .7),
}
BROADLEAF_GB = 3.2


def seasonal_foliage(texture: Path, season: int, seed: float, cache: Path) -> Path:
    """A composition foliage texture with Renderer64's crown colour for the season: only when its
    average is broadleaf green (g > 3.2 b, conifers and jungle fail) and only on green texels
    (leafness = (g - r) * 25 + .75), the atlas keeping its own light structure."""
    target = Path(cache) / f"{Path(texture).stem}_season{season}_{int(seed * 100):02d}.dds"
    if target.is_file():
        return target
    width, height, mips, dxgi, payload = rc.dds(Path(texture))
    srgb = decode_colour(Path(texture), 512)
    linear = (srgb / 255) ** 2.2
    average = linear.reshape(-1, 3).mean(0)
    if not average[1] > BROADLEAF_GB * average[2]:
        return Path(texture)
    alpha = np.full(srgb.shape[:2], 255.0, np.float32)
    if dxgi in (77, 78):
        size = srgb.shape[0]
        level = 0
        while (width >> level) > size and level + 1 < mips:
            level += 1
        offset, bw, bh = rc.mip_offsets(width, height, mips, 16)[level]
        blocks = np.frombuffer(payload[offset:offset + bw * bh * 16], np.uint8).reshape(bh, bw, 16)
        alpha = _bc4(blocks[..., 0:8]) * 255
    palette, chroma, spread, detail = SEASON_CROWNS[season]
    pal = next(p for limit, p in palette if seed < limit)
    pal = (pal[0] + (seed * 13.7 % 1 - .5) * .06, pal[1], pal[2] + (seed * 7.31 % 1 - .5) * spread)
    soft = linear * .3 + _blur(linear, 2) * .7
    src, avg = _oklch(soft), _oklch(average[None, None, :])[0, 0]
    lightness = pal[0] + (src[..., 0] - avg[0]) * detail
    chroma_ = pal[1] * chroma * np.clip(.45 + .75 * src[..., 0] / max(avg[0], .05), 0, 1)
    hue = pal[2] + np.clip((src[..., 2] - avg[2] + 180) % 360 - 180, -25, 25) * .3
    graded = _from_oklch(np.stack([np.maximum(lightness, .05), chroma_, hue], -1))
    leafness = np.clip((linear[..., 1] - linear[..., 0]) * 25 + .75, 0, 1)[..., None]
    result = linear * (1 - leafness) + graded * leafness
    write_bc3(target, np.dstack([np.clip(result ** (1 / 2.2) * 255, 0, 255), alpha]).astype(np.float32))
    return target


def golden_grass(entry: str, cache: Path) -> dict:
    """A skin grass clump in dry savanna gold (its own luminance, warm tint)."""
    mesh = vegetation_mesh(entry, cache)
    target = Path(cache) / f"golden_{entry.replace('/', '_')}.dds"
    if not target.is_file():
        colour = decode_colour(Path(mesh["texture"]), 256)
        luma = colour @ np.array([.2126, .7152, .0722], np.float32)
        light = np.clip(luma / max(float(luma.mean()), 1.0), .45, 1.5)[..., None]
        gold = np.clip(light * np.array([214, 178, 98], np.float32), 0, 255)
        write_bc3(target, np.dstack([gold, np.full(luma.shape, 255.0)]).astype(np.float32))
    return {**mesh, "key": mesh["key"] + "/golden", "texture": target, "masked": True}


def vegetation_mesh(entry: str, cache: Path) -> dict:
    """A Renderer64 skin-vegetation model (tile units) as a kit mesh."""
    return dict(_vegetation_cached(entry, str(cache)))


# ---------------------------------------------------------------- waterfall

def waterfall_parts(key: str, compiler, pick: dict, ground, cache: Path) -> list[dict]:
    """A static falling-water sheet from the source's waterfall effect bone down to the relief's foot,
    with a mist patch where it lands (a Lab stand-in for the Civ VI particle effect)."""
    item, package, entry = pick["asset"]
    landmark = assets.load(assets.SOURCES / compiler.manifest["assets"][compiler.root(item, package, entry)]["landmark"])
    top = None
    for path in landmark["components"]["skeletons"]:
        skeleton = assets.load(assets.SOURCES / path)
        names = [bone.get("name") for bone in skeleton["bones"]]
        if pick["waterfall"] in names:
            world = assets._skeleton_worlds(skeleton)[names.index(pick["waterfall"])]
            top = [v * ASSET_TO_HEX for v in world[12:15]]
    if top is None:
        return []
    x, y = top[0], top[1]
    z_top = ground(x, y) + top[2] * VERTICAL
    # Fall toward the viewer (Civ VI south) to where the relief meets the low ground.
    foot_y = y
    while foot_y > y - 0.45 and ground(x, foot_y) > 0.2 * max(ground(x, y), 1e-3):
        foot_y -= 0.01
    z_foot = ground(x, foot_y)
    y -= 0.03
    width = 0.08
    size = 128
    rng = np.random.default_rng(7)
    streaks = np.repeat(rng.uniform(0.75, 1.0, (1, size)), size, 0)
    grain = _blur(rng.uniform(0, 1, (size, size)).astype(np.float32), 1)
    shade = np.clip(streaks * (0.85 + 0.3 * grain), 0, 1)
    edge = 1 - np.abs(np.linspace(-1, 1, size))[None, :] ** 3
    alpha = np.where(grain * edge > 0.32, 255.0, 0.0)
    water = np.dstack([shade * 235, shade * 242, shade * 248, alpha]).astype(np.float32)
    texture = write_bc3(cache / f"{key}_waterfall.dds", water)
    sheet = [([x - width, y, z_top], [0, -1, 0], [0, 0]), ([x + width, y, z_top], [0, -1, 0], [1, 0]),
             ([x - width * 1.4, foot_y, z_foot], [0, -1, 0], [0, 1]), ([x + width * 1.4, foot_y, z_foot], [0, -1, 0], [1, 1])]
    mist = [([x - 0.14, foot_y - 0.10, z_foot + 0.004], [0, 0, 1], [0, 0]), ([x + 0.14, foot_y - 0.10, z_foot + 0.004], [0, 0, 1], [1, 0]),
            ([x - 0.14, foot_y + 0.05, z_foot + 0.004], [0, 0, 1], [0, 1]), ([x + 0.14, foot_y + 0.05, z_foot + 0.004], [0, 0, 1], [1, 1])]
    common = {"texture": texture, "masked": True, "pivot": [0.0, 0.0, 0.0], "yaw": 0.0, "scale": 1.0, "follow": False}
    return [{**common, "key": f"nw/{key}/waterfall", "vertices": sheet, "indices": [0, 2, 1, 1, 2, 3]},
            {**common, "key": f"nw/{key}/mist", "vertices": mist, "indices": [0, 2, 1, 1, 2, 3]}]


# ---------------------------------------------------------------- kits

def kit(key: str, compiler, cache: Path) -> dict:
    """Every part of one wonder in hex units, recentred on its footprint and folded onto one tile."""
    pick = PICKS[key]
    terrain_type = definitions()[key]["terrain_type"]
    parts, report, ground = [], {"source": pick["title"]}, (lambda x, y: 0.0)
    core = None
    if has_relief(pick):
        built = relief_part(key, pick, terrain_type, cache)
        parts += built["meshes"]
        ground, core = built["ground"], built["core"]
        report.update(peak_hex=round(built["peak"], 3), regions=built["regions"])
    if pick.get("asset"):
        item, package, entry = pick["asset"]
        meshes = source_meshes(key, compiler, item, package, entry, cache)
        if pick.get("keep_east") and built.get("split") is not None:
            meshes = [mesh for mesh in meshes if mesh["pivot"][0] >= built["split"] - .02]
        if pick.get("asset_place"):
            # The source's pieces follow its element where a composition moved and scaled it.
            ox, oy, size = pick["asset_place"]
            meshes = [{**mesh, "pivot": [mesh["pivot"][0] * size + ox, mesh["pivot"][1] * size + oy, mesh["pivot"][2] * size],
                       "scale": mesh["scale"] * size} for mesh in meshes]
        if pick.get("min_vertices"):
            meshes = [mesh for mesh in meshes if len(mesh["vertices"]) >= pick["min_vertices"]]
        if pick.get("piece_scale"):
            # The large formations grow about their pivots; small pieces keep their size.
            meshes = [{**mesh, "scale": mesh["scale"] * pick["piece_scale"]} if len(mesh["vertices"]) > 500 else mesh
                      for mesh in meshes]
        if pick.get("thin"):
            # Fewer of the small pieces (trees, shrubs); the large formations all stay.
            meshes = [mesh for n, mesh in enumerate(meshes) if len(mesh["vertices"]) > 500 or n % pick["thin"] == 0]
        parts += meshes
        report["asset_draws"] = len(meshes)
    if pick.get("canopy"):
        parts += canopy_parts(cache)
    if pick.get("extra_pillars"):
        # More of the source's own pillars, packed among the authored ones at varied size and turn.
        count, spacing, (low, high), seed = pick["extra_pillars"]
        pillars = [part for part in parts if len(part["vertices"]) > 1000]
        if pick.get("pillar_stretch"):
            # Wulingyuan's pillars are slender: the source pillars stretched up (bases unchanged).
            stretch = pick["pillar_stretch"]
            tall = {}
            for part in pillars:
                if part["key"] not in tall:
                    tall[part["key"]] = {**part, "key": part["key"] + "/tall",
                                         "vertices": [([p[0], p[1], p[2] * stretch if p[2] > 0 else p[2]], n, uv)
                                                      for p, n, uv in part["vertices"]]}
            parts = [({**tall[part["key"]], "pivot": part["pivot"], "yaw": part["yaw"], "scale": part["scale"]}
                      if part["key"] in tall else part) for part in parts]
            pillars = [part for part in parts if len(part["vertices"]) > 1000]
        taken = [tuple(part["pivot"][:2]) for part in pillars]
        xs, ys = [p[0] for p in taken], [p[1] for p in taken]
        # Over the whole footprint (the authored pillars cluster at the back).
        everything = [part["pivot"] for part in parts]
        xs, ys = [p[0] for p in everything], [p[1] for p in everything]
        rng = np.random.default_rng(seed)
        added = 0
        for _ in range(4000):
            if added >= count:
                break
            x = rng.uniform(min(xs) - .15, max(xs) + .15)
            y = rng.uniform(min(ys) - .15, max(ys) + .15)
            if all(math.hypot(x - a, y - b) >= spacing for a, b in taken):
                source = pillars[added % len(pillars)]
                parts.append({**source, "pivot": [x, y, 0.0], "yaw": float(rng.uniform(0, 2 * math.pi)),
                              "scale": source["scale"] * float(rng.uniform(low, high))})
                taken.append((x, y))
                added += 1
        report["extra_pillars"] = added
    if pick.get("waterfall") and pick.get("asset") and not (STAMP and stamped(key)):
        parts += waterfall_parts(key, compiler, pick, ground, cache)
    for item, package, entry, x, y, yaw, scale in pick.get("layout", ()):
        for mesh in source_meshes(key, compiler, item, package, entry, cache):
            if pick.get("grit") and pick["grit"][0] in entry:
                # A flat-coloured source rock gains the skin rock's grain (cutout atlas, opaque).
                mesh = {**mesh, "texture": gritty(mesh["texture"], *pick["grit"][1:], cache), "masked": True}
            parts.append({**mesh, "pivot": [x + mesh["pivot"][0] * scale, y + mesh["pivot"][1] * scale, 0.0],
                          "yaw": yaw + mesh["yaw"], "scale": scale * mesh["scale"]})
    if not parts:
        raise ValueError("no source parts")
    if core is None:
        # Asset-only kits: the footprint of the placed meshes themselves.
        xs, ys = [], []
        for part in parts:
            radius = max(math.hypot(v[0][0], v[0][1]) for v in part["vertices"]) * part["scale"]
            xs += [part["pivot"][0] - radius, part["pivot"][0] + radius]
            ys += [part["pivot"][1] - radius, part["pivot"][1] + radius]
        core = (min(xs), max(xs), min(ys), max(ys))
    cx, cy = pick.get("centre") or ((core[0] + core[1]) / 2, (core[2] + core[3]) / 2)
    extent = max(core[1] - core[0], core[3] - core[2])
    fold = pick.get("fold") or FIT_TILES / extent * pick.get("enlarge", 1.0)
    if pick.get("screen_layout"):
        facing = FACING + front_turn(key)
        turn_c, turn_s = math.cos(facing), math.sin(facing)
        large = {part["key"].split("/")[-1].split("_")[1]: part for part in parts
                 if len(part["vertices"]) > 500 and part["key"].split("/")[-1].startswith("mod_")}
        placed = []
        for suffix, sx, sy, yaw, size in pick["screen_layout"]:
            dx, dy = (sx + sy) / 2, (sx - sy) / 2               # natural offsets from the tile centre
            lx, ly = dx * turn_c - dy * turn_s, -dx * turn_s - dy * turn_c
            placed.append({**large[suffix], "pivot": [cx + lx / fold, cy + ly / fold, 0.0],
                           "yaw": large[suffix]["yaw"] + yaw, "scale": large[suffix]["scale"] * size})
        small = [part for part in parts if len(part["vertices"]) <= 500 and part["texture"] is not None]
        brush = [part for part in small if len(part["vertices"]) < 120]
        parts = placed + ([] if pick.get("no_brush") else brush)
    waves = None
    if pick.get("waves"):
        level, scale = pick["waves"]
        if has_relief(pick):
            field, axis, level = built["scatter"]["surface"], built["scatter"]["axis"], level / fold
        else:
            # Pieces standing in water (karst pillars): their feet are the waterline.
            low_x, high_x, low_y, high_y = core
            axis = np.linspace(min(low_x, low_y) - .2, max(high_x, high_y) + .2, 256)
            field, step = np.zeros((256, 256), np.float32), axis[1] - axis[0]
            for part in parts:
                if len(part["vertices"]) <= 1000:
                    continue
                local = np.array([v[0] for v in part["vertices"]]) * part["scale"]
                low = local[local[:, 2] <= local[:, 2].min() + .25 * np.ptp(local[:, 2])]
                c_, s_ = math.cos(part["yaw"]), math.sin(part["yaw"])
                cols = np.clip(((part["pivot"][0] + low[:, 0] * c_ - low[:, 1] * s_ - axis[0]) / step).round().astype(int), 0, 255)
                rows = np.clip(((part["pivot"][1] + low[:, 0] * s_ + low[:, 1] * c_ - axis[0]) / step).round().astype(int), 0, 255)
                field[rows, cols] = 1.0
            field = _blur(field, 2) * 6
        waves = wave_ribbons(field, axis, level, cx, cy, fold, FACING + front_turn(key), scale)
        report["wave_points"] = len(waves) // 5
    if pick.get("grass"):
        # Tall dry grass: the skin's grass clumps retinted gold, scattered clear of rocks and trunks.
        count, spacing, seed = pick["grass"]
        rng = np.random.default_rng(seed)
        models = [wheat_grass(f"res_wheat_tuft0{n}", cache, .4) for n in (1, 2, 3, 4)]
        taken = [tuple(part["pivot"][:2]) for part in parts]
        added = 0
        for _ in range(6000):
            if added >= count:
                break
            x, y = rng.uniform(-.6, .6, 2)
            # An even, continuous cover over the tile that softens only at its edge, clear of the
            # rocks and tree trunks.
            fade = 1 - _smoothstep(.5, .82, abs(x) + abs(y) + .05 * math.sin(x * 13 + y * 5))
            if rng.uniform() > fade or any(math.hypot(x - a, y - b) < spacing for a, b in taken):
                continue
            model = models[added % len(models)]
            parts.append({**model, "pivot": [float(x), float(y), 0.0], "yaw": float(rng.uniform(0, 2 * math.pi)),
                          "scale": float(rng.uniform(.26, .34)) / fold, "follow": True})
            taken.append((x, y))
            added += 1
        report["grass"] = added
    if has_relief(pick) and key in CLUTTER:
        trees = clutter_parts(key, compiler, built["scatter"], fold, 240 - len(parts), cache)
        parts += trees
        report["clutter"] = len(trees)
    stamp = None
    if STAMP and stamped(key):
        # The terrain carries the relief; Renderer64 seats every instance on it.
        parts = [part for part in parts if not part["key"].endswith("/relief")]
        if pick.get("talus"):
            # Rubble cones against the large pieces' walls (buttes stand in their own debris).
            height, reach = pick["talus"]
            scatter = dict(built["scatter"])
            axis = scatter["axis"]
            gx, gy = np.meshgrid(axis, axis)
            surface = scatter["surface"].copy()
            if pick.get("relief_flat"):
                surface = np.zeros_like(surface)                 # the floor; only the talus rises
            for part in parts:
                if len(part["vertices"]) <= 500:
                    continue
                radius = .8 * max(math.hypot(v[0][0], v[0][1]) for v in part["vertices"]) * part["scale"]
                distance = np.hypot(gx - part["pivot"][0], gy - part["pivot"][1])
                # The skirt meets the walls at the surrounding talus' height (filling dips between
                # humps), then falls away; the piece sinks so its base stays where it was.
                tall = max(v[0][2] for v in part["vertices"]) * part["scale"]
                top = min(float(surface[distance < radius + reach * .5].max()) + height * .3,
                          ground(part["pivot"][0], part["pivot"][1]) + .3 * tall)
                if pick.get("relief_flat"):
                    top = pick["talus_share"] * tall
                weight = (1 - _smoothstep(radius, radius + reach, distance)) ** 1.4
                surface = np.maximum(surface, top * weight)
                # A pedestal butte stands on its talus top; otherwise it keeps its original base.
                part["sink"] = (0.0 if pick.get("pedestal") else
                                top - (0.0 if pick.get("relief_flat") else ground(part["pivot"][0], part["pivot"][1]))) + pick.get("sink", 0.0)
            scatter["surface"] = surface
            built = {**built, "scatter": scatter}
        stamp = stamp_field(key, built["scatter"], cx, cy, fold)
        stamp["effects"] = effect_records(key, compiler, built["scatter"], stamp, cx, cy, fold)
        report["stamp"] = {"span_tiles": round(stamp["span"], 3), "units": round(stamp["units"], 1)}
        if stamp["effects"]:
            report["effects"] = [[round(v, 3) for v in record] for record in stamp["effects"]]
    if SEASON in SEASON_CROWNS:
        # Composition foliage takes the season's crown colour, as natural trees do in Renderer64.
        graded = {}
        for index, part in enumerate(parts):
            texture = part.get("texture")
            if texture is None or part["key"].startswith("decal/"):
                continue
            # One seed per mesh (the kit keeps one mesh per key); distinct tree meshes vary.
            seed = (sum(map(ord, part["key"])) % 3 + .5) / 3
            if (texture, seed) not in graded:
                graded[(texture, seed)] = seasonal_foliage(Path(texture), SEASON, seed, cache)
            if graded[(texture, seed)] != Path(texture):
                parts[index] = {**part, "texture": graded[(texture, seed)], "masked": True}
        report["seasonal_foliage"] = sum(1 for k, v in graded.items() if v != Path(k[0]))
    instances = []
    for part in parts:
        px, py, pz = part["pivot"]
        if stamp is not None:
            # Renderer64 seats the instance on the stamped terrain under its pivot. Pieces
            # authored around a hollow the terrain cannot carve simply sit on the terrain.
            if part["follow"] and pick.get("seat_pieces"):
                pz = 0.0
            if pick.get("sink") and len(part["vertices"]) > 500:
                pz -= part.get("sink", pick["sink"])
            lift = (pz - (0.0 if part["follow"] else max(0.0, ground(px, py)))) * fold
        else:
            lift = (pz + (ground(px, py) if part["follow"] else 0.0)) * fold
        lx, ly = (px - cx) * fold, (py - cy) * fold
        facing = FACING + front_turn(key)
        cosine, sine = math.cos(facing), math.sin(facing)
        instances.append({"model": part["key"], "u": .5 + lx * cosine - ly * sine, "v": .5 + lx * sine + ly * cosine,
                          "rotation": facing + part["yaw"], "scale": fold * part["scale"],
                          "lift": lift + pick.get("lift", 0.0),
                          "ground_fit": 0.0})
    for record in (stamp or {}).get("effects", []):
        if record[0] != EFFECT_WATERFALL:
            continue
        # Pieces standing on the fall's path (not at its foot) would cut the water in two.
        foot, lip = np.array(record[1:3]), np.array(record[3:5])
        path = lip - foot
        by_key = {part["key"]: part for part in parts}
        kept = []
        for instance in instances:
            mesh = by_key[instance["model"]]
            radius = max(math.hypot(v[0][0], v[0][1]) for v in mesh["vertices"]) * instance["scale"]
            at = np.array([instance["u"] - .5, .5 - instance["v"]])
            t = float(np.clip(np.dot(at - foot, path) / max(float(np.dot(path, path)), 1e-9), 0, 1))
            if .2 < t < .95 and np.linalg.norm(at - (foot + t * path)) < radius + record[6] * .5:
                continue
            # A large cliff whose face crosses the fall gets a notch where the water pours.
            local = np.array([v[0] for v in mesh["vertices"]]) * instance["scale"]
            turn_c, turn_s = math.cos(instance["rotation"]), math.sin(instance["rotation"])
            spot = np.stack([instance["u"] - .5 + local[:, 0] * turn_c - local[:, 1] * turn_s,
                             .5 - (instance["v"] + local[:, 0] * turn_s + local[:, 1] * turn_c)], 1)
            triangles = np.array(mesh["indices"]).reshape(-1, 3)
            centre = spot[triangles].mean(1)
            along = np.clip((centre - foot) @ path / max(float(path @ path), 1e-9), 0, 1)
            across = np.linalg.norm(centre - (foot + along[:, None] * path), axis=1)
            notch = (along > .1) & (along < .95) & (across < record[6] * .75)
            if len(mesh["vertices"]) > 500 and notch.any():
                notched = {**mesh, "key": mesh["key"] + "/notch", "indices": triangles[~notch].reshape(-1).tolist()}
                parts.append(notched)
                instance = {**instance, "model": notched["key"]}
            kept.append(instance)
        report["cleared_from_fall"] = len(instances) - len(kept)
        instances = kept
    report.update(parts=len(parts), fold=round(fold, 3), footprint_hex=round(extent, 2))
    return {"meshes": {part["key"]: part for part in parts}, "instances": instances, "report": report, "stamp": stamp,
            "waves": waves}


def stamped(key: str) -> bool:
    """Land wonders with a height field join the terrain; water wonders keep a relief mesh."""
    return has_relief(PICKS[key]) and definitions()[key]["terrain_type"] not in ("coast", "sea", "ocean")


def stamp_field(key: str, scatter: dict, cx: float, cy: float, fold: float) -> dict:
    """The prepared displacement (levelled, eased blend, non-negative) recentred on the
    footprint, with its span in tiles and peak in relief units."""
    surface, texel = scatter["surface"], scatter["texel"]
    res = surface.shape[0]
    disp = np.maximum(surface, 0.0)
    if PICKS[key].get("stamp_blur"):
        # The terrain mesh cannot draw features much narrower than its grid, while pieces
        # seat on the exact stamp: broaden narrow pedestals so both agree.
        width = max(1, int(round(PICKS[key]["stamp_blur"] / fold / texel)))
        disp = np.maximum(_blur(_blur(disp, width), width), disp * .6)
    shift_c, shift_r = int(round(cx / texel)), int(round(cy / texel))
    shifted = np.zeros_like(disp)
    rs, cs = slice(max(0, shift_r), res + min(0, shift_r)), slice(max(0, shift_c), res + min(0, shift_c))
    rd, cd = slice(max(0, -shift_r), res + min(0, -shift_r)), slice(max(0, -shift_c), res + min(0, -shift_c))
    shifted[rd, cd] = disp[rs, cs]
    peak = float(shifted.max())

    def shift(field):
        out = np.zeros_like(field)
        out[rd, cd] = field[rs, cs]
        return out
    ground, special = region_weights(key, scatter)
    return {"field": shifted / max(peak, 1e-6), "span": res * texel * fold, "units": peak * fold * STAMP_UNITS,
            "turn": STAMP_TURN + front_turn(key), "ground": shift(ground), "special": shift(special),
            "kind": REGIONS.get(key, {}).get("kind", 1)}


def stamp_offset(stamp: dict, kx: float, ky: float, cx: float, cy: float, fold: float) -> tuple[float, float]:
    """A kit point (hex units) as Renderer64 natural coordinates from the tile centre, inverting
    mountain_shape.h's stamp sampling (u = .5 + lx, v = .5 - ly after the stamp's turn)."""
    lx, ly = (kx - cx) * fold / stamp["span"], -(ky - cy) * fold / stamp["span"]
    c, s = math.cos(stamp["turn"]), math.sin(stamp["turn"])
    return (lx * c + ly * s) * stamp["span"], (-lx * s + ly * c) * stamp["span"]


def stamp_height(stamp: dict, dx: float, dy: float) -> float:
    """The stamp's displacement (relief units) at natural offset (dx, dy) from the tile centre."""
    a, b = dx / stamp["span"], dy / stamp["span"]
    c, s = math.cos(stamp["turn"]), math.sin(stamp["turn"])
    u, v = .5 + a * c - b * s, .5 - (a * s + b * c)
    field = stamp["field"]
    if not (0 <= u <= 1 and 0 <= v <= 1):
        return 0.0
    return float(_bilinear(field, np.array([[v * field.shape[0]]]), np.array([[u * field.shape[1]]]))[0, 0]) * stamp["units"]


def effect_records(key: str, compiler, scatter: dict, stamp: dict, cx: float, cy: float, fold: float) -> list[list[float]]:
    """effects.bin records: kind, foot and top offsets from the tile centre (natural
    coordinates), lift above the top (relief units), width, height (tile units), strength, seed."""
    spec, records = EFFECTS.get(key, {}), []
    kind = PICKS[key].get("effect_water")
    if kind and scatter.get("water") is not None and scatter.get("water_level") is not None:
        # Still water for the production water layers: the water's long axis, a generous width
        # (the terrain falling below the water decides the visible shore) and its level in relief
        # units above the natural ground, at the top end (e[5]) and the foot (e[7]).
        rows, cols = np.nonzero(scatter["water"])
        axis = scatter["axis"]
        xy = np.stack([axis[cols], axis[rows]], 1)
        middle = xy.mean(0)
        if kind == "channel":
            major = np.array([0.0, 1.0])                       # Civ VI south (the mouth) to north
        else:
            values, vectors = np.linalg.eigh(np.cov((xy - middle).T))
            major = vectors[:, -1]
        minor = np.array([-major[1], major[0]])
        along, across = (xy - middle) @ major, (xy - middle) @ minor
        ends = [middle + major * float(along.min()), middle + major * float(along.max())]
        if kind == "channel":
            width = 2 * float(np.percentile(np.abs(across - np.median(across)), 80))
            shift = minor * float(np.median(across))
            ends = [end + shift for end in ends]
        else:
            width = 2 * float(np.percentile(np.abs(across), 97))
        (fx, fy), (tx, ty) = (stamp_offset(stamp, *end, cx, cy, fold) for end in ends)
        level = scatter["water_level"] * fold * STAMP_UNITS
        top = foot = level
        if kind == "channel":
            # A level surface just above the channel's typical floor.
            # The floor of each cross-section (the channel bends off a straight centreline).
            length = math.hypot(tx - fx, ty - fy)
            nx_, ny_ = -(ty - fy) / length, (tx - fx) / length
            half = width * fold * .8
            floor = [min(stamp_height(stamp, fx + (tx - fx) * v + nx_ * w, fy + (ty - fy) * v + ny_ * w)
                         for w in np.linspace(-half, half, 15)) for v in np.linspace(0, 1, 41)]
            # Clears the floor and the natural ground's own variation; the head's slope stays dry.
            foot = top = float(np.percentile(floor, 75)) + 6.0
        records.append([EFFECT_CHANNEL if kind == "channel" else EFFECT_LAKE, fx, fy, tx, ty, top,
                        width * fold * (1.6 if kind == "channel" else 1.4), foot, 1.0, 3.0])
    if not spec:
        return records
    if spec.get("waterfall"):
        # The source's falls (its effect bone) face the C3X adjacency direction once the kit
        # turns its front there (Angel Falls: the river's southeast edge). The water leaves the
        # rim above the bone and drapes down that face to the foot.
        bone_name, width, standoff = spec["waterfall"]
        item, package, entry = PICKS[key]["asset"]
        landmark = assets.load(assets.SOURCES / compiler.manifest["assets"][compiler.root(item, package, entry)]["landmark"])
        bone = None
        for path in landmark["components"]["skeletons"]:
            skeleton = assets.load(assets.SOURCES / path)
            names = [b.get("name") for b in skeleton["bones"]]
            if bone_name in names:
                bone = [v * ASSET_TO_HEX for v in assets._skeleton_worlds(skeleton)[names.index(bone_name)][12:15]]
        if bone is not None:
            dx, dy = NATURAL_DIRECTIONS[definitions()[key].get("adjacency_dir") or "south"]
            x, y = stamp_offset(stamp, bone[0], bone[1], cx, cy, fold)
            step = .005
            lip_x, lip_y, lip = x, y, stamp_height(stamp, x, y)
            while math.hypot(x - lip_x, y - lip_y) < .6:
                x, y = x - dx * step, y - dy * step
                height = stamp_height(stamp, x, y)
                if height < lip - 3:
                    break
                if height > lip:
                    lip_x, lip_y, lip = x, y, height
            x, y = lip_x, lip_y
            while stamp_height(stamp, x, y) > .05 * lip and math.hypot(x - lip_x, y - lip_y) < .6:
                x, y = x + dx * step, y + dy * step
            records.append([EFFECT_WATERFALL, x, y, lip_x, lip_y, standoff, width, 0.0, 1.0, 5.0])
            # Spray from the plunge pool, a little out from the foot.
            x, y = x + dx * .07, y + dy * .07
            records.append([EFFECT_SPRAY, x, y, x, y, 0.0, .52, .32, 1.0, 7.0])
    if spec.get("vents"):
        # The highest cones (local maxima) of the special region.
        special = np.isin(scatter["ids"], REGIONS[key]["special"]) if scatter["ids"] is not None else None
        surface, axis = scatter["surface"], scatter["axis"]
        padded = np.pad(surface, 3, mode="edge")
        crest = np.lib.stride_tricks.sliding_window_view(padded, (7, 7)).max(axis=(2, 3))
        cones = sorted(np.argwhere(special & (surface >= crest) & (surface > .01)), key=lambda rc: -surface[rc[0], rc[1]])
        if spec.get("central"):
            # The main geyser is the crater at the heart of the travertine (its lowest point
            # within the highest cones' ring), the cones around it steam.
            rows, cols = np.nonzero(special)
            mr, mc = int(round(rows.mean())), int(round(cols.mean()))
            ring = int(round(.06 / scatter["texel"]))
            window = surface[max(0, mr - ring):mr + ring + 1, max(0, mc - ring):mc + ring + 1]
            low = np.unravel_index(int(np.argmin(window)), window.shape)
            cones = [np.array([max(0, mr - ring) + low[0], max(0, mc - ring) + low[1]])] + cones
        for (width, height, strength), (r, c) in zip(spec["vents"], cones):
            x, y = stamp_offset(stamp, axis[c], axis[r], cx, cy, fold)
            records.append([EFFECT_STEAM, x, y, x, y, 0.0, width, height, strength, float(len(records) * 23)])
    return records


def region_weights(key: str, scatter: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-texel ground and special weights (0-1) for the stamp, from the source's region IDs."""
    spec = REGIONS.get(key, {})
    ids, height, blend = scatter["ids"], scatter["height"], scatter["blend"]
    shape = height.shape

    def weight(name):
        chosen = spec.get(name, ())
        if chosen == "all":
            return np.full(shape, spec.get(f"{name}_strength", 1.0), np.float32)
        if ids is None or not chosen:
            return np.zeros(shape, np.float32)
        if isinstance(chosen, dict):
            out = np.zeros(shape, np.float32)
            for value, strength in chosen.items():
                out[ids == value] = strength
            return out
        return np.isin(ids, chosen).astype(np.float32) * spec.get(f"{name}_strength", 1.0)
    ground, special = weight("ground"), weight("special")
    if spec.get("special_rise"):
        # The special material climbs with the relief (red rubble near the buttes) and its
        # foot eases into the tile's own ground (sand).
        rise = np.clip(scatter["surface"] / max(float(scatter["surface"].max()), 1e-6), 0, 1)
        special = special * _smoothstep(*spec["special_rise"], rise)
        ground = np.maximum(ground, 1 - _smoothstep(*spec["ground_fade"], rise))
    if spec.get("green_foot") and ids is not None:
        low = 1 - _smoothstep(spec["green_foot"] * 0.6, spec["green_foot"], height)
        ground = np.maximum(ground, low * (1 - special))
    blur = lambda f: np.clip(_blur(f, 2), 0, 1) * (blend > .02)
    return blur(ground), blur(special)


def write_stamp(stamp: dict | None, waves: list | None = None) -> None:
    """Install (or clear) the per-render stamp pack the private tree reads."""
    if STAMP_PACK.exists():
        shutil.rmtree(STAMP_PACK)
    if waves:
        STAMP_PACK.mkdir(parents=True)
        (STAMP_PACK / "waves.bin").write_bytes(struct.pack(f"<{len(waves)}f", *waves))
    if stamp is None:
        return
    STAMP_PACK.mkdir(parents=True, exist_ok=True)
    field = np.clip(stamp["field"] * 255 + .5, 0, 255).astype(np.uint8)
    (STAMP_PACK / "height_lod0.dds").write_bytes(relief.make_r8_dds(field.shape[1], field.shape[0], field.tobytes()))
    (STAMP_PACK / "stamp.bin").write_bytes(struct.pack("<3f", stamp["span"], stamp["units"], stamp["turn"]))
    if stamp.get("effects"):
        flat = [value for record in stamp["effects"] for value in record]
        (STAMP_PACK / "effects.bin").write_bytes(struct.pack(f"<{len(flat)}f", *flat))
    for name in ("ground", "special"):
        weights = np.clip(stamp[name] * 255 + .5, 0, 255).astype(np.uint8)
        (STAMP_PACK / f"{name}_lod0.dds").write_bytes(
            relief.make_r8_dds(weights.shape[1], weights.shape[0], weights.tobytes()))


# ---------------------------------------------------------------- bake

def composition_name(key: str) -> str:
    return ("NW_" + key)[:23]


def bake(kits: dict) -> dict:
    """Write the Lab composition pack: one composition per wonder, every part a rigid instance."""
    meshes, compositions = {}, []
    for key, built in kits.items():
        if not built["instances"]:
            continue                                        # terrain-only (stamp mode)
        meshes.update(built["meshes"])
        if len(built["instances"]) > 256:
            raise ValueError(f"{key}: {len(built['instances'])} instances exceed the runtime limit")
        compositions.append((composition_name(key), [(ALL_TERRAINS, built["instances"])]))
    if len(meshes) > 256:
        raise ValueError(f"{len(meshes)} meshes exceed the runtime asset limit; render fewer wonders at once")
    opaque = list(dict.fromkeys(mesh["texture"] for mesh in meshes.values() if not mesh["masked"]))
    cutout = list(dict.fromkeys(mesh["texture"] for mesh in meshes.values() if mesh["masked"]))
    atlases, place = rc.pack_atlases(opaque, rc.MODEL_ATLAS) if opaque else ([], {})
    if cutout:
        more, cut_place = rc.pack_atlases(cutout, rc.MASKED_ATLAS, len(atlases))
        atlases += more
        place.update(cut_place)
    if len(atlases) > rc.TEXTURE_SLOTS:
        raise ValueError(f"{len(atlases)} atlases exceed the texture slots; render fewer wonders at once")
    slots = [f"textures/atlas_{i}.dds" for i in range(len(atlases))]
    padded = slots + ["textures/unused.dds"] * (rc.TEXTURE_SLOTS - len(slots))
    index, payloads = {}, []
    for key, mesh in meshes.items():
        atlas, u0, v0, extent = place[mesh["texture"]]
        edge = .5 / (extent * rc.MODEL_ATLAS[0])
        vertices = [(p, n, (u0 + min(1 - edge, max(edge, uv[0])) * extent,
                            v0 + min(1 - edge, max(edge, uv[1])) * extent)) for p, n, uv in mesh["vertices"]]
        index[key] = len(payloads)
        payloads.append(rc.mesh_payload(key, atlas, vertices, mesh["indices"]))
    blob = bytearray(rc.MAGIC)
    blob.extend(struct.pack("<IIII", 2, rc.TEXTURE_SLOTS, len(payloads), 0))
    for texture in padded:
        blob.extend(rc.bundle_string(texture))
    for payload in payloads:
        blob.extend(payload)
    blob.extend(struct.pack("<I", len(compositions)))
    for name, variants in compositions:
        blob.extend(rc.bundle_string(name))
        blob.extend(struct.pack("<I", len(variants)))
        for mask, instances in variants:
            blob.extend(struct.pack("<II", mask, len(instances)))
            for item in instances:
                blob.extend(struct.pack("<I6f", index[item["model"]], item["u"], item["v"], item["rotation"],
                                        item["scale"], item["lift"], item["ground_fit"]))
    blob.extend(struct.pack("<I", len(compositions)))
    for position, (name, _) in enumerate(compositions):
        blob.extend(rc.bundle_string(name))
        blob.extend(struct.pack("<I", position))
    if CATALOG.exists():
        shutil.rmtree(CATALOG)
    (CATALOG / "textures").mkdir(parents=True)
    for position, data in enumerate(atlases):
        (CATALOG / f"textures/atlas_{position}.dds").write_bytes(data)
    (CATALOG / "textures/unused.dds").write_bytes(rc.dds_header(4, 4, 1, rc.MODEL_ATLAS[3], 8) + bytes(8))
    (CATALOG / "resource_runtime.bin").write_bytes(blob)
    return {"meshes": len(meshes), "atlases": len(atlases), "compositions": [name for name, _ in compositions]}


# ---------------------------------------------------------------- C3X scenes

@lru_cache(maxsize=None)
def definitions() -> dict:
    keys = {"Zhangjiajie Mountains": "zhangjiajie_mountains", "Mount Kilimanjaro": "mount_kilimanjaro",
            "Lofoten Skerries": "lofoten_skerries"}
    out = {}
    for item in parse_natural_wonder_config(DEFAULT_CONFIG):
        key = keys.get(item["name"], item["name"].lower().replace(" ", "_"))
        out[key] = item
    return out


def scene_tiles(key: str) -> dict:
    """{(dx, dy): (base, real, river, bonus)} following the C3X placement rule:
    terrain_type on the wonder tile, adjacent_to on neighbours (the configured
    direction, else the three tiles behind the wonder)."""
    item = definitions()[key]
    terrain, adjacent, direction = item["terrain_type"], item.get("adjacent_to"), item.get("adjacency_dir")
    tiles = {}

    def put(dx, dy, kind, river=0):
        base, real = TERRAIN[kind]
        tiles[(dx, dy)] = (base, real, river, 0x10 if kind == "snow-mountains" else 0)
    water = terrain in ("coast", "sea", "ocean")
    for dy in range(-6, 7):
        for dx in range(-8, 9):
            if (dx + dy) % 2:
                continue
            if water:
                # Land to the north, the wonder's water around it, open sea toward the viewer.
                kind = "grassland" if dy <= -3 else "coast" if dy <= 1 else "sea"
                if terrain == "sea" and dy >= -1:
                    kind = "sea" if (dx, dy) == (0, 0) or dy >= 1 else "coast"
                put(dx, dy, kind)
            else:
                put(dx, dy, terrain)
    if adjacent == "river":
        put(0, 0, terrain, RIVER_EDGE[direction or "southeast"])
        ox, oy = DIRECTIONS["south"]
        put(ox, oy, terrain, RIVER_EDGE["northeast"])
    elif adjacent:
        targets = [direction] if direction else ["north"] if PICKS[key].get("one_neighbour") else list(BACK)
        if adjacent in ("coast", "sea") and direction:
            # A shoreline: the configured side and the rest of the near side are water.
            sx, sy = DIRECTIONS[direction]
            for dx, dy in list(tiles):
                if (dx - sx) * sx + (dy - sy) * sy >= 0 and (dx, dy) != (0, 0):
                    put(dx, dy, adjacent if (abs(dx - sx) + abs(dy - sy)) <= 2 else "sea")
        else:
            for name in targets:
                dx, dy = DIRECTIONS[name]
                put(dx, dy, adjacent)
    put(0, 0, terrain, tiles[(0, 0)][2])
    return tiles


class Cases:
    """Lab tile cases for this study's resource roster cases (this process only)."""

    def __init__(self, keys):
        self.keys = list(keys)

    def key(self, case):
        name = case.removeprefix("native-nw-")
        return name if case.startswith("native-nw-") and name in self.keys else None

    def applies(self, category, case):
        return category == "resources" and self.key(case) is not None

    def terrain(self, category, case, dx, dy, base, real):
        key = self.key(case)
        tile = scene_tiles(key).get((dx, dy))
        if tile is None:
            return 2, 2, 0, 0, 0
        base, real, river, bonus = tile
        if STAMP and stamped(key) and (dx, dy) == (0, 0):
            # The stamp rides the mountain relief path: a mountain over the C3X base terrain.
            real = 6
            bonus |= STAMP_MARKER | (0x10 if key in SNOW_CAPPED else 0)
        if STAMP and PICKS[key].get("waves") and (dx, dy) == (0, 0):
            bonus |= STAMP_MARKER                           # its waves join the coastal waves
        return base, real, river, bonus, 0

    def resources(self, category, case):
        # With tile overlays on, the preview places objects from this list, not the roster.
        key = self.key(case)
        return "" if key in TERRAIN_ONLY else f"0,0,{composition_name(key)}"

    def viewport_for(self, case, zoom):
        return 7 * zoom, 4 * zoom


def install(keys) -> Cases:
    from Renderer import renderer
    from Renderer.lab.studies.resources import roster
    cases = Cases(keys)
    original_native, original_layout = roster.native, roster.native_layout
    data = {**original_native(), "batches": {**original_native()["batches"],
                                              **{f"nw-{key}": [composition_name(key)] for key in keys}}}
    roster.native = lambda: data
    roster.native_layout.cache_clear()

    def native_layout(case):
        key = cases.key(case)
        if key is not None:
            return ((0, 0, composition_name(key), "Grassland"),)
        return original_layout(case)
    roster.native_layout = native_layout
    roster.CASES = roster.cases()
    original_tile_cases = renderer.lab_tile_cases
    renderer.lab_tile_cases = lambda category, case: cases if cases.applies(category, case) else \
        original_tile_cases(category, case)
    return cases


# ---------------------------------------------------------------- render, crops

def season_tag() -> str:
    return f"-{SEASON_NAMES[SEASON]}" if SEASON else ""


def frame(key: str, zoom: int) -> Path:
    return OUT / (("renders-stamp" if STAMP else "renders") + season_tag()) / f"{key}-z{zoom}"


def render(keys, zooms=(128, 256), clip: bool = False) -> dict:
    from Renderer import renderer
    cache = OUT / "texture_cache"
    compiler = assets.Compiler()
    kits, report = {}, {}
    for key in keys:
        try:
            kits[key] = kit(key, compiler, cache)
        except (OSError, ValueError, KeyError, struct.error) as exc:
            report[key] = {"error": str(exc)}
            print(f"SKIP {key}: {exc}", flush=True)
            continue
        report[key] = kits[key]["report"]
        if not kits[key]["instances"]:
            TERRAIN_ONLY.add(key)
        print(key, json.dumps(report[key]), flush=True)
    compiler.save()
    install(list(kits))
    # A pinned copy of the current candidate: peers rebuild the shared one.
    dll = OUT / "dll/C3XRenderer.dll"
    if not dll.is_file():
        dll.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "Renderer/native/build/candidate/C3XRenderer.dll", dll)
    # The composition pack holds at most 256 meshes: bake and render in batches.
    batches, current, count = [], [], 0
    for key, built in kits.items():
        if current and count + len(built["meshes"]) > 240:
            batches.append(current)
            current, count = [], 0
        current.append(key)
        count += len(built["meshes"])
    if current:
        batches.append(current)
    for batch in batches:
        summary = bake({key: kits[key] for key in batch})
        print(json.dumps(summary), flush=True)
        # A batch of terrain-only kits has no compositions; render it without a pack.
        pack = CATALOG.name if summary["compositions"] else ""
        for key in batch:
            render_one(key, zooms, dll, report, kits[key].get("stamp"), pack, clip, kits[key].get("waves"))
    name = ("render_report_stamp" if STAMP else "render_report") + season_tag().replace("-", "_") + ".json"
    if clip:
        name = "render_report_clips.json"
    previous = json.loads((OUT / name).read_text()) if (OUT / name).is_file() else {}
    previous.update(report)
    (OUT / name).write_text(json.dumps(previous, indent=1) + "\n")
    if STAMP:
        write_stamp(None)
    return report


def _vm_idle(limit: float = 600.0) -> None:
    """Wait while another session's Lab fixture runs on the shared VM (renders collide)."""
    import subprocess
    import time
    deadline = time.time() + limit
    while time.time() < deadline:
        lines = subprocess.run(["ps", "-Ao", "args="], capture_output=True, text=True).stdout.splitlines()
        if not any(line.startswith("/Applications/Parallels") and "prlctl exec" in line for line in lines):
            return
        time.sleep(3)


# Natural-wonder touches on top of Renderer64's season shading (Lab only, applied after the
# production overlay in each seasonal shader root): {season: [(files, anchor, addition[, "replace"])]}.
WONDER_SEASON_EDITS = {2: [
    # A geothermal basin (Old Faithful's travertine, region kind 2) stays bare and steaming.
    (("Renderer/native/city_fidelity/mountain.hlsl",),
     "        winter_snow = lerp(ground_cover, rock_cover, rock_albedo_coverage);\n",
     "#if defined(WONDER_REGION_KIND) && WONDER_REGION_KIND == 2\n"
     "        if (input.material.y > 1.5 && input.volcano_owner.z < 0.001)\n"
     "            winter_snow *= 1 - saturate(input.volcano_owner.y * 1.6);\n"
     "#endif\n"),
    # Composition bodies: nothing at or below the waterline takes snow (the reef's coral), and
    # foliage sheds most of it, as the needle snow on natural jungle and broadleaf does.
    (("Renderer/native/city_fidelity/feature.hlsl", "Renderer/native/city_fidelity/city.hlsl",
      "Renderer/native/city_fidelity/rigid_feature.hlsl", "Renderer/native/city_fidelity/resource_body.hlsl"),
     "            albedo = lerp(lerp(albedo, winter_dormant(albedo), WINTER_OBJ_DORMANT), winter_white, saturate(cover));\n",
     "            albedo = lerp(lerp(albedo, winter_dormant(albedo), WINTER_OBJ_DORMANT * wonder_dry), winter_white,\n"
     "                          saturate(cover * wonder_dry * lerp(1.0, 0.45, saturate((albedo.g - albedo.r) * 25 + 0.75))));\n",
     "replace"),
    (("Renderer/native/city_fidelity/feature.hlsl", "Renderer/native/city_fidelity/city.hlsl",
      "Renderer/native/city_fidelity/rigid_feature.hlsl", "Renderer/native/city_fidelity/resource_body.hlsl"),
     "            cover *= 0.7 + 0.6 * winter_noise(winter_world * 23.0);\n",
     "            float wonder_dry = smoothstep(DRY_LOW / 112.0, DRY_HIGH / 112.0, input.q6_world.z);   // not at the waterline\n"),
]}


@lru_cache(maxsize=None)
def shader_variant(kind: int, season: int = 0, wet: float = 0.0) -> Path:
    """The private tree's shaders with WONDER_REGION_KIND defined (one root per special material).
    Other seasons carry Renderer64's integrated seasonal shading (Renderer/lab/shared/seasons,
    the approved Autumn, Winter and Spring looks) with the season index pinned, so the wonders
    use the production season code without its frame constant."""
    from Renderer.lab.studies.natural_wonders import stamp_tree
    from Renderer.lab.shared.seasons import seasons
    # `wet`: composition bodies up to this height (relief units over the water's world height)
    # count as in the water; a reef seated on the sea bed reaches the surface far above it.
    variant = f"kind{kind}" + (f"-s{season}" if season else "") + (f"-wet{int(wet)}" if wet else "")
    stamp_tree.private_tree.variant_root(stamp_tree.NAME, variant, [f"WONDER_REGION_KIND={kind}"])
    root = OUT / "ranges" / f"root-{variant}"
    if season:
        for relative in seasons.TARGETS:
            path = root / relative
            if not path.is_file():
                continue                                    # game-only open water (sandbox)
            text = seasons.apply(relative, path.read_text())
            for index in ("int(C3XSeason.x + 0.5)", "int(game_resource_look.z + 0.5)"):
                text = text.replace(f"#define C3X_SEASON_INDEX {index}", f"#define C3X_SEASON_INDEX {season}")
            if f"#define C3X_SEASON_INDEX {season}" not in text and "C3X_SEASON_INDEX" in text:
                raise ValueError(f"{relative}: season index define not pinned")
            for files, anchor, addition, *mode in WONDER_SEASON_EDITS.get(season, ()):
                if relative in files:
                    if text.count(anchor) != 1:
                        raise ValueError(f"{relative}: wonder season anchor not found once: {anchor.strip()[:60]}")
                    addition = addition.replace("DRY_LOW", f"{2.6 + wet:.1f}").replace("DRY_HIGH", f"{6.0 + wet:.1f}")
                    text = text.replace(anchor, addition if mode == ["replace"] else anchor + addition)
            path.write_text(text)
    return root


def render_one(key: str, zooms, dll: Path, report: dict, stamp: dict | None = None, pack: str = CATALOG.name,
               clip: bool = False, waves: list | None = None) -> None:
    from Renderer import renderer
    preview, env = None, None
    if STAMP:
        dll = STAMP_TREE / "Renderer/native/build/candidate/C3XRenderer.dll"
        preview = OUT / "stamp_preview.exe"
        shutil.copy2(STAMP_TREE / "Renderer/lab/.cache/native_preview.exe", preview)
        root = shader_variant(stamp["kind"] if stamp else 1, SEASON, PICKS[key].get("winter_wet", 0.0) if SEASON == 2 else 0.0)
        env = {"C3X_RENDERER_SHADER_SOURCE_ROOT": "..\\..\\" + renderer.relative(root).replace("/", "\\"),
               "C3X_LAB_EFFECT_TIME": str(EFFECT_TIME), "C3X_RENDERER_PREVIEW_SEASON": str(SEASON)}
        if clip:
            # stamp_tree.py's effect frames: 48 cold redraws at 15 fps from EFFECT_TIME.
            env["C3X_LAB_EFFECT_FRAMES"] = "48"
    for zoom in zooms:
        target = OUT / "clips" / f"{key}{season_tag()}-z{zoom}" if clip else frame(key, zoom)
        for attempt in range(2):
            _vm_idle()
            if STAMP:
                write_stamp(stamp, waves)
            try:
                renderer.native_render("resources", f"native-nw-{key}", 12, zoom, target,
                                       resource_pack=pack, candidate=dll, preview=preview, extra_env=env)
                break
            except ValueError as exc:
                failure = str(exc)
                if clip and any(target.glob("*.effect-047.bmp")):
                    break                                   # the frames loop exits before the usual witness
        else:
            report[key][f"z{zoom}"] = f"render failed: {failure}"
            print(f"FAILED {key} z{zoom}: {failure}", flush=True)
            continue
        census = (target / "native.log").read_text(errors="replace")
        drawn = key in TERRAIN_ONLY or f" {composition_name(key)} replaced=1" in census
        report[key][f"z{zoom}"] = "drawn" if drawn else "rendered, composition not reported drawn"


def crops(keys, zooms=(128, 256)) -> dict:
    """Crop each wonder's tile at the C3X natural-wonder cell framing (tile top - 24 px at 1x) and a context view."""
    from Renderer.lab.studies.resources import audit
    folder = OUT / "crops"
    folder.mkdir(parents=True, exist_ok=True)
    index = {}
    for key in keys:
        for zoom in zooms:
            bmp = frame(key, zoom) / f"native-nw-{key}-h12-z{zoom}.bmp"
            if not bmp.is_file():
                continue
            image = audit.read_bmp(bmp)
            cx, cy = image[0] // 2, image[1] // 2
            tag = ("-stamp" if STAMP else "") + season_tag()
            cell = audit.crop(image, cx - zoom // 2, cy - zoom * 7 // 16, zoom, zoom * 11 // 16)
            audit.write_png(folder / f"{key}-cell{tag}-z{zoom}.png", *cell)
            context = audit.crop(image, cx - zoom * 3 // 2, cy - zoom, zoom * 3, zoom * 15 // 8)
            audit.write_png(folder / f"{key}-context{tag}-z{zoom}.png", *context)
            index.setdefault(key, {})[str(zoom) + tag] = {"cell": f"{key}-cell{tag}-z{zoom}.png",
                                                           "context": f"{key}-context{tag}-z{zoom}.png"}
    previous = json.loads((folder / "index.json").read_text()) if (folder / "index.json").is_file() else {}
    for key, views in index.items():
        previous.setdefault(key, {}).update(views)
    (folder / "index.json").write_text(json.dumps(previous, indent=1) + "\n")
    return index


def clips(keys, zoom: int = 256) -> dict:
    """Loop clips of the attached effects: the 48 effect frames (15 fps), cropped
    around the wonder, as H.264 MP4 for the gallery."""
    import subprocess
    import tempfile
    from Renderer.lab.studies.resources import audit
    folder = OUT / "crops"
    index = {}
    for key in keys:
        source = OUT / "clips" / f"{key}{season_tag()}-z{zoom}"
        frames = sorted(source.glob(f"native-nw-{key}-h12-z{zoom}.bmp.effect-[0-9][0-9][0-9].bmp"))
        if len(frames) < 2:
            continue
        with tempfile.TemporaryDirectory(prefix="nw-clip-") as directory:
            for n, path in enumerate(frames):
                image = audit.read_bmp(path)
                cx, cy = image[0] // 2, image[1] // 2
                view = audit.crop(image, cx - zoom, cy - zoom * 7 // 8, zoom * 2, zoom * 3 // 2)
                audit.write_png(Path(directory) / f"{n:03d}.png", *view)
            target = folder / f"{key}-clip{season_tag()}-z{zoom}.mp4"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15", "-i", str(Path(directory) / "%03d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(target)],
                           check=True)
        index[key] = target.name
        print(key, target.relative_to(ROOT), len(frames), "frames", flush=True)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("render", "crops", "list", "fields", "clips"))
    parser.add_argument("--wonder", action="append")
    parser.add_argument("--zoom", action="append", type=int)
    parser.add_argument("--terrain-stamp", action="store_true",
                        help="put land wonders' relief into the terrain via the private stamp tree")
    parser.add_argument("--season", type=int, choices=(0, 1, 2, 3), default=0,
                        help="C3X season: 0 summer, 1 autumn, 2 winter, 3 spring")
    args = parser.parse_args()
    global STAMP, SEASON
    STAMP = args.terrain_stamp
    SEASON = args.season
    keys = args.wonder or list(PICKS)
    unknown = set(keys) - set(PICKS)
    if unknown:
        parser.error(f"unknown wonders {sorted(unknown)}")
    zooms = tuple(args.zoom or (128, 256))
    if args.action == "list":
        for key in keys:
            print(key, PICKS[key]["title"], scene_tiles(key)[(0, 0)])
    elif args.action == "fields":
        for key in keys:
            fields = element_fields(PICKS[key])
            print(key, fields["grid"], fields["parameters"], {k: v.shape for k, v in fields.items()
                                                              if isinstance(v, np.ndarray)})
    elif args.action == "render":
        render(keys, zooms)
        crops(keys, zooms)
    elif args.action == "clips":
        # Wonders with attached effects, rendered as motion-study clips.
        keys = [key for key in (args.wonder or EFFECTS) if key in EFFECTS or PICKS[key].get("waves")]
        render(keys, (256,), clip=True)
        clips(keys)
    else:
        crops(keys, zooms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
