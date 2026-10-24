"""Proposed Civ VI (and Workshop) art for each default C3X wonder, as Lab catalogs.

`PICKS` holds, per C3X wonder (the ledger's c3x_key), the recommended source
first and then the alternatives worth comparing. A choice is one or more
parts; a part names a source from kits.py and how to cut it from its build
movie (`keep`), turn, fit and place it. Catalogs lay C3X wonders out as columns
and choices as rows on the terrain each wonder needs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
LEDGER = ROOT / "Renderer/inventory/vanilla_c3x_to_civ6_constructed_wonders.json"
FACE_SW, FACE_SE = math.pi, math.pi / 2


def part(source: str, **options) -> dict:
    return {"source": source, **options}


def choice(label: str, *parts: dict, note: str = "") -> dict:
    return {"label": label, "parts": list(parts), "note": note}


# Terrain each C3X wonder stands on in the Lab (its buildable_on rule, else grassland).
TERRAIN = {"pyramids": "Desert", "oracle": "Hills", "great_lighthouse": "Coast", "colossus": "Coast",
           "hoover_dam": "Mountains"}
# Front toward the viewer's left (SW) or right (SE), as the C3X sprite draws it.
FACING = {"great_library": FACE_SW, "newtons_university": FACE_SW, "js_bachs_cathedral": FACE_SE,
          "sistine_chapel": FACE_SW}

PICKS: dict[str, list[dict]] = {
    "pyramids": [choice("Pyramids", part("pyramids"))],
    # Its four torch posts' fires (FX_bone_Fire_Perm_01-04) name bones the package
    # does not carry: placed on the posts' tops, measured in the finished model.
    "hanging_gardens": [choice("Hanging Gardens", part("hanging_gardens",
                                                       fires=((-.05, .21), (-.21, -.05), (.07, -.21), (.21, .05))))],
    "oracle": [choice("Oracle", part("oracle"))],
    "temple_of_artemis": [choice("Temple of Artemis", part("temple_of_artemis"))],
    "mausoleum_of_mausollos": [choice("Mausoleum at Halicarnassus", part("mausoleum"))],
    "statue_of_zeus": [choice("Statue of Zeus", part("statue_of_zeus"))],
    "great_library": [choice("Great Library", part("great_library"))],
    "copernicus_observatory": [
        choice("University (domed observatory tower)", part("university")),
        choice("Ulugh Beg Observatory (Workshop)", part("ulugh_beg")),
        choice("Maya Observatory (Industrial)", part("observatory")),
    ],
    "js_bachs_cathedral": [
        choice("Notre-Dame (Workshop)", part("notre_dame")),
        choice("Gothic Cathedral (Holy Site)", part("cathedral")),
    ],
    "newtons_university": [
        choice("Oxford University", part("oxford_university")),
        choice("National College (Workshop)", part("albro_college")),
    ],
    "seti_program": [
        choice("Arecibo Observatory (Workshop)", part("arecibo")),
        choice("Amundsen-Scott Research Station", part("amundsen_scott")),
    ],
    "shakespeares_theater": [
        choice("Globe Theatre (Workshop)", part("globe_theatre", drop_children=("Tree_", "PROP_Crane"))),
        choice("Bolshoi Theatre", part("bolshoi_theatre")),
    ],
    "sistine_chapel": [
        choice("St. Peter's Basilica (Workshop)", part("st_peters")),
        choice("Hagia Sophia", part("hagia_sophia")),
    ],
    "great_lighthouse": [choice("Great Lighthouse", part("great_lighthouse"))],
    "colossus": [choice("Colossus", part("colossus"))],
    # The dam stands across its tile's river, in a gorge the pack's stamp carves
    # (river_span: build_wonder_pack's gorge, turned to the river by the renderer).
    "hoover_dam": [choice("Three Gorges Dam (Workshop)", part("three_gorges", river_span=True)),
                   choice("Dam + Hydroelectric Dam", part("dam_hydro"))],
    "apollo_program": [
        # The movie also parks a second rocket below ground (for its launch): cut it.
        # The rocket on its tower and the assembly hangar stand at opposite corners
        # of the model: brought together at one scale.
        choice("Space Race rocket on its launch tower",
               part("large_rocket", keep={"z_min": .005, "box": [-1, -1, 1, 0]}, scale=3.4, at=(-.1, .12)),
               part("large_rocket", keep={"z_min": .005, "box": [-1, 0, 1, 1]}, scale=3.4, at=(.24, -.2))),
        choice("Spaceport launch pad", part("spaceport")),
    ],
    "forbidden_palace": [
        choice("Forbidden City", part("forbidden_city")),
        choice("Audience Chamber (Government Plaza)", part("gov_wide")),
    ],
    # C3X draws a low, wide modern office complex (Langley): two of Civ VI's Campus
    # Research Labs as a low-rise campus, one turned as a wing.
    "intelligence_agency": [
        choice("Research Lab campus (two Civ VI Research Labs)",
               part("research_lab", fit=.55, at=(-.17, .14)),
               part("research_lab", fit=.55, turn=math.pi / 2, at=(.2, -.16))),
        choice("Research Lab (Campus) alone", part("research_lab")),
        choice("Intelligence Agency (Government Plaza)", part("gov_spies")),
    ],
    "pentagon": [choice("The Pentagon (Workshop)", part("pentagon_mod")),
                 choice("War Department (Government Plaza)", part("gov_military"))],
    "military_academy": [choice("Military Academy (Encampment)", part("military_academy"))],
    "iron_works": [choice("Ruhr Valley", part("ruhr_valley")),
                   choice("Ironworks (Workshop)", part("albro_ironworks"))],
    "wall_street": [
        choice("Financial Center (Workshop)", part("albro_finance")),
        choice("Stock Exchange among glass towers", part("stock_exchange", fit=.5, at=(.12, .12)),
               part("glass_tower_a", fit=.3, height=.9, at=(-.22, -.18)),
               part("glass_tower_b", fit=.28, height=.75, at=(.2, -.22)),
               part("glass_tower_c", fit=.28, height=.6, at=(-.24, .18))),
    ],
}

GROUPS = (
    ("ancient", "Ancient wonders", ("pyramids", "hanging_gardens", "oracle", "temple_of_artemis",
                                    "mausoleum_of_mausollos", "statue_of_zeus", "great_library")),
    ("renaissance", "Renaissance and modern great wonders", ("copernicus_observatory", "js_bachs_cathedral",
                                                             "newtons_university", "seti_program",
                                                             "shakespeares_theater", "sistine_chapel")),
    ("site", "Coast and mountain wonders", ("great_lighthouse", "colossus", "hoover_dam")),
    ("small", "Small wonders", ("apollo_program", "forbidden_palace", "intelligence_agency", "pentagon",
                                "military_academy", "iron_works", "wall_street")),
)


def ledger() -> dict:
    return {item["c3x_key"]: item for item in json.loads(LEDGER.read_text())["mappings"]}


def short(text: str, limit: int = 23) -> str:
    return text[:limit]


# A construction site for stand-ins without a build movie (district buildings):
# Civ VI's own crane, brick and wood piles, imported with the wonder movies,
# around the rising building while it is under construction.
# Props sit at the tile's east, west and south corners (a building hides anything
# behind it) at about the scale Civ VI's wonder movies give them.
SITE = (part("prop:great_library:PROP_Crane_LG_Anim", scale=2.4, at=(.36, -.36)),
        part("prop:pyramids:WON_YellowBrickPile_C", scale=3.5, at=(-.36, .36)),
        part("prop:pyramids:WON_Woodpiles_Movie_A", scale=2.4, at=(.3, .3)))


def staging(parts: list, built: float) -> list:
    """A choice's parts at a build progress, with a construction site when none
    of them has a working build movie of its own."""
    from Renderer.lab.studies.wonders import kits
    staged = [{**item, "built": built} for item in parts]
    if built < 1.0 and not any(kits.staged(item["source"]) for item in parts):
        staged += [dict(item) for item in SITE]
    return staged


# C3X build progress shown for each recommended wonder (C3X itself has one
# construction sprite; Civ VI's build movie gives every stage).
STAGES = ((.25, "25% built"), (.5, "half built"), (.75, "75% built"), (1.0, "complete"))


def catalogs() -> dict:
    out = {}
    for key, title, wonders in GROUPS:
        stages, picks = [], []
        for col, wonder in enumerate(wonders):
            common = {"wonder": wonder, "terrain": TERRAIN.get(wonder, "Grassland"), "facing": FACING.get(wonder, FACE_SW),
                      "col": col}
            recommended = PICKS[wonder][0]
            for row, (built, label) in enumerate(STAGES):
                stages.append({**common, "name": short(f"S{col}{row} {wonder}"), "label": f"{recommended['label']} · {label}",
                               "row": row, "built": built,
                               "parts": staging(recommended["parts"], built)})
            for row, pick in enumerate(PICKS[wonder]):
                picks.append({**common, "name": short(f"P{col}{row} {wonder}"), "label": pick["label"], "row": row,
                              "parts": pick["parts"]})
        out[f"wonder-stages-{key}"] = {"id": f"wonder-stages-{key}", "title": f"{title}: construction stages",
                                       "cells": stages}
        if any(len(PICKS[wonder]) > 1 for wonder in wonders):
            out[f"wonder-picks-{key}"] = {"id": f"wonder-picks-{key}", "title": f"{title}: candidates", "cells": picks}
    return out


# Raw offsets of a city's neighbours on screen, and the context shots (gameplay
# and close-up zoom at noon, close-up at night).
E, W, S, N, SE, NW, NE, SW = (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, -1), (1, -1), (-1, 1)
CONTEXT_SHOTS = ({"zoom": 128, "hour": 12}, {"zoom": 256, "hour": 12}, {"zoom": 256, "hour": 22})
# City: dx, dy, culture (0 American, 1 European, 2 Roman, 3 Middle Eastern, 4 Asian), era, size, capital, walled.
CONTEXTS = (
    ("ancient", "Ancient Middle Eastern capital", (0, 0, 3, 0, 1, 1, 0), (
        (E, "pyramids", 1.0), (W, "great_library", 1.0), (SE, "hanging_gardens", 1.0),
        (NW, "temple_of_artemis", 1.0), (NE, "oracle", 1.0), (S, "mausoleum_of_mausollos", .5))),
    ("medieval", "Medieval European city", (0, 0, 1, 1, 2, 0, 1), (
        (E, "js_bachs_cathedral", 1.0), (W, "sistine_chapel", 1.0), (SE, "newtons_university", 1.0),
        (NW, "copernicus_observatory", 1.0), (S, "shakespeares_theater", .5))),
    ("modern", "Modern American metropolis", (0, 0, 0, 3, 2, 1, 0), (
        (E, "wall_street", 1.0), (W, "iron_works", 1.0), (SE, "pentagon", 1.0), (NW, "intelligence_agency", 1.0),
        (NE, "military_academy", 1.0), (S, "apollo_program", .5))),
)


def contexts() -> dict:
    out = {}
    for key, title, city, entries in CONTEXTS:
        cells = []
        for index, ((dx, dy), wonder, built) in enumerate(entries):
            pick = PICKS[wonder][0]
            label = f"{pick['label']}" + ("" if built == 1.0 else f" · {round(built * 100)}% built")
            cells.append({"name": short(f"C{index} {wonder}"), "label": label, "row": 0, "col": index,
                          "dx": dx, "dy": dy, "wonder": wonder, "terrain": TERRAIN.get(wonder, "Grassland"),
                          "facing": FACING.get(wonder, FACE_SW), "built": built,
                          "parts": staging(pick["parts"], built)})
        if key in ("ancient", "modern"):
            # The era's default Wonder District on one more neighbour.
            era = "Ancient" if key == "ancient" else "Modern"
            cells.append({"name": short(f"C{len(cells)} wonder district"), "label": f"Wonder District · {era}",
                          "row": 0, "col": len(cells), "dx": SW[0], "dy": SW[1], "wonder": "wonder_district",
                          "terrain": "Grassland", "facing": FACE_SW, "parts": wonder_district(era)})
        out[f"wonder-context-{key}"] = {"id": f"wonder-context-{key}", "title": title, "cities": [city],
                                        "shots": SCENE_SHOTS, "cells": cells}
    return out


# Shots: gameplay and close-up zoom at noon; close-up at night (Civ VI's lit
# windows and lamps); close-up in Autumn, Winter and Spring (seasons: 1, 2, 3).
SEASONS = ((1, "-autumn"), (2, "-winter"), (3, "-spring"))
GRID_SHOTS = ({"zoom": 128, "hour": 12}, {"zoom": 256, "hour": 12}, {"zoom": 256, "hour": 22}) + \
    tuple({"zoom": 256, "hour": 12, "season": season, "tag": tag} for season, tag in SEASONS)
PICK_SHOTS = ({"zoom": 256, "hour": 12}, {"zoom": 256, "hour": 22})
SCENE_SHOTS = CONTEXT_SHOTS + ({"zoom": 256, "hour": 12, "season": 2, "tag": "-winter"},)

# C3X's default Wonder District (WonderDistrict.pcx: a fenced, scaffolded plot by
# era, a crane in the last) for a district no city has started a wonder on yet:
# Civ VI's own construction props on the era's paving, scaffolding along the plot.
WD_ERAS = ("Ancient", "Medieval", "Industrial", "Modern")


def scaffold_ring(half: float = .36, count: int = 5) -> list:
    pieces = []
    for side in range(4):
        for k in range(count):
            t = -half + (k + .5) * 2 * half / count
            u, v = ((t, -half), (half, t), (t, half), (-half, t))[side]
            turn = math.pi / 2 if side % 2 == 0 else 0.0
            pieces.append(part("prop:notre_dame_2024:PROP_Scaffold_A", scale=2.4, turn=turn, at=(u, v)))
    return pieces


def wonder_district(era: str) -> list:
    """The default Wonder District for an era: paving, a scaffold fence and the
    era's construction props (stone and wood; brick; a crane from the Middle Ages)."""
    props = {
        "Ancient": [("prop:ulugh_beg:PROP_Boulderpile_A", 2.4, (-.12, .12)), ("prop:broadway:WON_Woodpiles_Movie_A", 2.4, (.14, .05)),
                    ("prop:forbidden_city:WON_YellowBrickPile_C", 3.5, (.1, .18)), ("prop:notre_dame_2024:PROP_Stone_A", 2.4, (-.1, -.1))],
        "Medieval": [("prop:albro_college:PROP_Crane_LG_Anim", 2.4, (.18, -.14)), ("prop:broadway:WON_RedBrickPile_A", 3.5, (-.1, .14)),
                     ("prop:broadway:WON_Woodpiles_Movie_A", 2.4, (.08, .16)), ("prop:notre_dame_2024:PROP_Stone_A", 2.4, (-.14, -.08))],
        "Industrial": [("prop:albro_finance:PROP_INDCrane_LG_Anim", 2.4, (.18, -.14)), ("prop:ruhr_valley:WON_DarkerRedBrickPile_A", 3.5, (-.1, .14)),
                       ("prop:ruhr_valley:WON_Rhur_Cart", 2.4, (.1, .14)), ("prop:broadway:WON_Woodpiles_Movie_A", 2.4, (-.14, -.06))],
        "Modern": [("prop:amundsen_scott:PROP_MODCrane_LG_Anim", 2.4, (.18, -.14)), ("prop:amundsen_scott:WON_IBeamPile_Movie_A", 2.4, (-.1, .12)),
                   ("prop:amundsen_scott:WON_MetalPile_Movie_A", 2.4, (.1, .16)), ("prop:biosphere:WON_GreyBrickPile_A", 3.5, (-.14, -.06))],
    }[era]
    return [part(f"plot:{era}")] + scaffold_ring() + [part(source, scale=scale, at=at) for source, scale, at in props]


def districts() -> dict:
    cells = [{"name": short(f"D{col} {era}"), "label": f"Wonder District · {era}", "row": 0, "col": col,
              "wonder": "wonder_district", "terrain": "Grassland", "facing": FACE_SW, "parts": wonder_district(era)}
             for col, era in enumerate(WD_ERAS)]
    return {"wonder-district": {"id": "wonder-district", "title": "Wonder District (no wonder started)",
                                "cells": cells, "shots": GRID_SHOTS}}


# Bumped when kits.py changes how wonders are laid on the tile or drawn (2: squared
# to the Civ III tile; 3: night lights and seasons; 4: Civ VI fire and light effects; 5: city
# bodies with Civ VI's full materials and the finished, lit building; 6: light maps, metalness and
# authored fire anchors), so older renders are redone.
LAYOUT = 6


def everything() -> dict:
    specs = {**catalogs(), **contexts(), **districts()}
    for key, spec in specs.items():
        spec.setdefault("shots", PICK_SHOTS if "picks" in key else GRID_SHOTS)
    return {key: {**spec, "layout_version": LAYOUT} for key, spec in specs.items()}
