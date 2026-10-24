"""Catalogs of Civ VI stand-ins for Civ III's tile buildings (Lab only).

Civ III draws these in Map_Renderer::m12_Draw_Tile_Buildings: a barricade
(else a fortress), an airfield, a colony, a radar tower, an outpost and a
scenario victory point. Fortress, barricade and airfield sprites change with
the era; outposts have three sprites. Rows are Civ III eras, columns options,
as in the district survey this study borrows its pipeline from.
"""
from __future__ import annotations

from Renderer.lab.studies.districts.kits import landmark, raw
from Renderer.lab.studies.districts.plan import CITY_WALLS, aerodrome_turn, cell, kit, part, wall_scene

ERAS = ("Ancient", "Medieval", "Industrial", "Modern")
AB = {"Ancient": "An", "Medieval": "Md", "Industrial": "In", "Modern": "Mo"}
# Civ III era -> Civ VI art-era suffix of era-tagged entries.
ALCAZAR = {"Ancient": "_Ancient", "Medieval": "", "Industrial": "_Industrial", "Modern": "_Modern"}
FORT_KEEP = {"Ancient": "IMP_Fort_FortA_Medieval", "Medieval": "IMP_Fort_FortA_Medieval",
             "Industrial": "IMP_Fort_FortA_Industrial", "Modern": "IMP_Fort_FortA_Modern"}
WATCHTOWER = {"Ancient": "DIS_ENC_Tower_Classical", "Medieval": "DIS_ENC_Tower_Classical",
              "Industrial": "DIS_ENC_Tower_Industrial", "Modern": "DIS_ENC_Tower_Modern"}
CAMP_WALL = {"Ancient": "DIS_ENC_Wall_Classical", "Medieval": "DIS_ENC_Wall_Classical",
             "Industrial": "DIS_ENC_Wall_Inustrial_MD", "Modern": "DIS_ENC_Wall_Modern_MD"}


def tile(entry: str, content: str = "Base", package: str = "landmarks/tilebases") -> dict:
    return raw(content, package, entry)


def lone(record: dict, **options) -> list:
    """One component, centred on the tile."""
    return [part(record, centre=True, fixed=True, **options)]


def keep(era: str) -> list:
    """The Fort's keep alone, in the middle of a drafted (walled) tile."""
    return [part(tile(FORT_KEEP[era]), role="centre", fixed=True)]


def ring_records(era: str) -> list:
    """The era's ring pieces as (role, record): its city walls (as Renderer64's
    cities and the district survey's Encampments use) in the Ancient and
    Medieval rows, the Encampment's own walls and watchtowers later. The
    Ancient gate is an open gateway (two posts), which reads as a breach in a
    barricade: that ring stays closed."""
    if era in CITY_WALLS:
        segment, tower, gate = CITY_WALLS[era]
        walls = lambda name: raw("Base", "landmarks/Walls", name)
        return [("wall", walls(segment)), ("tower", walls(tower))] + ([("gate", walls(gate))] if era != "Ancient" else [])
    return [("wall", tile(CAMP_WALL[era])), ("tower", tile(WATCHTOWER[era]))]


def ring(era: str) -> list:
    """A closed ring round the tile (the gallery's drafted barricade)."""
    return [part(record, role=role) for role, record in ring_records(era)]


WALLED = {"walls": "explicit", "density": .3, "props": 2}


def fortress() -> list:
    cells = []
    for row, era in enumerate(ERAS):
        a = ALCAZAR[era]
        cells += [
            cell(f"Fort {AB[era]}", f"{era} · Civ VI Fort", row, 0, [part(landmark("LM_FORT", era))], era=era),
            cell(f"FortKeep {AB[era]}", f"{era} · Fort's keep alone", row, 1, lone(tile(FORT_KEEP[era])), era=era),
            cell(f"AlcA {AB[era]}", f"{era} · Alcázar A", row, 2,
                 [part(landmark("LM_ALCAZAR", era, f"IMP_Alcazar_01{a}"))], era=era),
            cell(f"AlcB {AB[era]}", f"{era} · Alcázar B", row, 3,
                 [part(landmark("LM_ALCAZAR", era, f"IMP_Alcazar_02{a}"))], era=era),
        ]
    cells.append(cell("Pa An", "Ancient · Māori Pā", 0, 4, [part(landmark("LM_MAORI_PA", "Ancient"))], era="Ancient"))
    return cells


def barricade() -> list:
    cells = []
    for row, era in enumerate(ERAS):
        cells += [
            cell(f"BarFortW {AB[era]}", f"{era} · Fort's keep in a walled ring", row, 0,
                 keep(era) + ring(era), era=era, layout=WALLED, draft="force"),
            cell(f"BarFort {AB[era]}", f"{era} · Civ VI Fort, walled", row, 1,
                 [part(landmark("LM_FORT", era))] + ring(era), era=era, layout=WALLED, draft="force"),
            cell(f"BarAlc {AB[era]}", f"{era} · Alcázar A, walled", row, 2,
                 [part(landmark("LM_ALCAZAR", era, f"IMP_Alcazar_01{ALCAZAR[era]}"))] + ring(era), era=era,
                 layout=WALLED, draft="force"),
        ]
    return cells


def barricade_line() -> dict:
    """The 1498 save's Japanese line, as Civ III draws joined barricades: a run
    of tiles joined edge to edge with Civ VI's Great Wall (the district
    survey's topology: a pillar per tile, half walls to edge neighbours)."""
    run = {(-4, 2): "Grassland", (-3, 1): "Grassland", (-2, 0): "Plains", (-1, -1): "Hills",
           (0, -2): "Grassland", (1, -3): "Forest", (2, -2): "Grassland", (3, -1): "Plains"}
    cells, tiles = wall_scene(run)
    for item in cells:
        item["name"] = item["name"].replace("GW ", "BL ")
    tiles = [(x, y, name.replace("GW ", "BL "), terrain) for x, y, name, terrain in tiles]
    return {"title": "Barricades joined into a line (Great Wall pieces)", "snap": False, "cells": cells,
            "tiles": tiles, "shots": ({"zoom": 128, "hour": 12}, {"zoom": 256, "hour": 12}), "crop": (6.5, 3.75)}


def airfield() -> list:
    aero = {"facing": aerodrome_turn(), "scale": .7, "draft": False, "snap": False, "runway": True}
    cells = []
    for row, era in enumerate(("Industrial", "Modern")):
        cells += [
            cell(f"Air {AB[era]}", f"{era} · Civ VI Airstrip", row, 0, [part(landmark("LM_AIRSTRIP", era))], era=era),
            cell(f"Aero {AB[era]}", f"{era} · Aerodrome with hangar", row, 1,
                 kit("DISTRICT_AERODROME", "HANGAR", era, ("HANGAR",)), era=era, **aero),
            cell(f"AeroAir {AB[era]}", f"{era} · Aerodrome with hangar and airport", row, 2,
                 kit("DISTRICT_AERODROME", "HANGAR, AIRPORT", era, ("HANGAR", "AIRPORT")), era=era, **aero),
        ]
    return cells


# Lone towers and markers: Civ III's outpost and radar sprites stand about
# half a tile tall at game zoom; Civ VI's pieces are a fraction of a hex.
TALL = 2.4


def outpost() -> list:
    cells = []
    for row, era in enumerate(ERAS):
        cells.append(cell(f"Watch {AB[era]}", f"{era} · Encampment watchtower", row, 0,
                          lone(tile(WATCHTOWER[era]), scale=TALL), era=era))
        cells.append(cell(f"AirTower {AB[era]}", f"{era} · Airstrip watchtower", row, 1,
                          lone(tile("IMP_Airstrip_Tower"), scale=TALL), era=era))
        if era in ("Ancient", "Medieval"):
            cells.append(cell(f"GWTower {AB[era]}", f"{era} · Great Wall tower", row, 2,
                              lone(raw("Base", "landmarks/Walls", "IMP_Great_Wall_tower"), scale=TALL * .7), era=era))
        else:
            cells.append(cell(f"BarTower {AB[era]}", f"{era} · Industrial watchtower", row, 2,
                              lone(tile("VIL_BAR_IND_Tower"), scale=TALL * .8), era=era))
    return cells


def radar() -> list:
    picks = (("RadarAero", "Aerodrome radio mast", tile("DIS_AERO_Radio_Tower")),
             ("RadarDome", "Amundsen Station radar dome",
              tile("WON_Amundsen_Station_Radar", "DLC/Expansion1")),
             ("RadarCtl", "Aerodrome control tower", tile("DIS_AERO_Tower")),
             ("RadarWatch", "Industrial watchtower", tile("VIL_BAR_IND_Tower")),
             ("RadarSilo", "Missile-silo tower", tile("IMP_SILO_TowerA")))
    return [cell(name, f"Modern · {label}", 0, col, lone(record, scale=TALL * .8), era="Modern")
            for col, (name, label, record) in enumerate(picks)]


def victory() -> list:
    """Civ III's scenario victory point is a small white obelisk."""
    picks = (("VicSphinx", "Sphinx obelisk", tile("IMP_Sphinx_Obelisk"), TALL),
             ("VicAntique", "Antiquity obelisk", raw("Base", "environment/clutter", "IMP_Antiquity_Obelisk"), TALL),
             ("VicMonument", "Monument (classical)", raw("Base", "landmarks/hero_buildings", "AW_Monument_Classical"), 1.0),
             ("VicRome", "Rome city monument", tile("DIS_CTY_ROME_Monument"), 1.4),
             ("VicLeader", "Leader statue", tile("DIS_THR_StatueLeader"), TALL))
    cells = [cell(name, label, 0, col, lone(record, scale=scale)) for col, (name, label, record, scale) in enumerate(picks)]
    cells[1]["state"] = "default"      # clutter pieces have no build states
    return cells


# Colonies: the 2026-10-07 draft's concepts (Renderer/lab/out/colonies_survey),
# rebuilt from the same Civ VI pieces as drafted kits. A colony stands on its
# resource, so the middle and front of the tile stay clear for it.
X1, X2, AU, GC, KV, CR = ("DLC/Expansion1", "DLC/Expansion2", "DLC/Australia", "DLC/GranColombia_Maya",
                          "DLC/KublaiKhan_Vietnam", "DLC/CivRoyaleScenario")
COLONIES = {
    "Ancient": (
        ("A", "Resource camp kit", [("IMP_Camp_AN_Tent", "Base", "anchor"), ("IMP_Camp_AN_Rack_LG", "Base", "building"),
                                    ("IMP_Camp_AN_Rack_SM_A", "Base", "prop"), ("IMP_Camp_AN_Pots", "Base", "prop"),
                                    ("IMP_Camp_AN_Barrels", "Base", "prop"), ("IMP_Camp_AN_Well", "Base", "prop")]),
        ("B", "Thatch storehouse yard (draft pick)", [
            ("IMP_PLNT_ANC_Bld_A", "Base", "building"), ("IMP_PLNT_ANC_Bld_C", "Base", "building"),
            ("IMP_Pasture_AN_Shed", "Base", "building"), ("IMP_Camp_AN_Rack_SM_A", "Base", "prop"),
            ("IMP_Camp_AN_Pots", "Base", "prop"), ("IMP_PLNT_ANC_Barrel_Lg", "Base", "prop"),
            ("IMP_PLNT_ANC_Wall_Rock_A", "Base", "prop")]),
        ("C", "Lodge-and-cache", [("IMP_Cree_Mekewap_BldgA", X1, "building"), ("IMP_Cree_Mekewap_BldgB", X1, "building"),
                                  ("IMP_Cree_Mekewap_BldgC", X1, "building"), ("IMP_Cree_Mekewap_Crate", X1, "prop"),
                                  ("IMP_Cree_Mekewap_Pottery_A", X1, "prop")])),
    "Medieval": (
        ("A", "Timber yard", [("IMP_Lumbermill_Medieval_Storage", "Base", "anchor"),
                              ("IMP_Lumbermill_Medieval_Mill", "Base", "anchor"),
                              ("IMP_Lumbermill_Medieval_Log_Cut_Stack", "Base", "prop"),
                              ("IMP_Lumbermill_Medieval_Saw", "Base", "prop"), ("IMP_Camp_AN_Well", "Base", "prop")]),
        ("B", "Hoist tower and storehouse (draft pick)", [
            ("IMP_Mining_Industry_BldgD", KV, "building"), ("IMP_PLNT_ANC_Bld_A", "Base", "building"),
            ("IMP_Pasture_AN_Shed", "Base", "building"), ("IMP_Fort_Wagon_Industrial", "Base", "prop"),
            ("PROP_Barrel_Classical", "Base", "prop")]),
        ("C", "Colonial trading house", [("IMP_Hacienda_BuildingB", GC, "building"),
                                         ("PROP_Barrel_Classical", "Base", "prop"), ("IMP_PLNT_ANC_Cart", "Base", "prop"),
                                         ("IMP_Fort_Wagon_Industrial", "Base", "prop")])),
    "Industrial": (
        ("A", "Resource camp kit", [("IMP_Camp_IND_Cabin", "Base", "anchor"), ("IMP_Camp_IND_Rack_LG", "Base", "building"),
                                    ("IMP_Camp_IND_Firewood", "Base", "prop"), ("IMP_Camp_IND_Barrel", "Base", "prop"),
                                    ("IMP_Camp_IND_Guntable", "Base", "prop")]),
        ("B", "Headframe and sheds (draft pick)", [
            ("IMP_Mining_Industry_BldgE", KV, "building"), ("IMP_Mining_Industry_BldgC", KV, "building"),
            ("IMP_Fort_BldA_Industrial", "Base", "building"), ("IMP_Mining_Industry_Cart", KV, "prop"),
            ("IMP_Mining_Industry_barrel", KV, "prop")]),
        ("C", "Pump-jack depot", [("IMP_Oil_Well_Shed", "Base", "building"), ("IMP_Oil_Well_TankSmall", "Base", "building"),
                                  ("IMP_Oil_Well_BarrelRed", "Base", "prop"), ("IMP_Oil_Well_BarrelYellow", "Base", "prop")]),
        ("D", "Outback station", [("IMP_Outback_IND_Bld_A", AU, "building"), ("IMP_Outback_IND_Bld_B", AU, "building"),
                                  ("IMP_Outback_IND_Well", AU, "building"), ("IMP_Outback_IND_Truck", AU, "prop")])),
    "Modern": (
        ("A", "Quonset depot (draft pick)", [
            ("IMP_Airstrip_Hangar", "Base", "anchor"), ("IMP_Airstrip_Tent", "Base", "building"),
            ("IMP_Mining_Corp_Tanks", KV, "prop"), ("PROP_Barrel_Modern", "Base", "prop"),
            ("IMP_Fort_Truck_Modern", "Base", "prop")]),
        ("B", "Container yard", [("Crates_CR_Double", CR, "building"), ("Crates_CR_Green", CR, "building"),
                                 ("IMP_Fort_BldB_Modern", "Base", "building"), ("PROP_Barrel_Modern", "Base", "prop"),
                                 ("IMP_Fort_TruckB_Modern", "Base", "prop")]),
        ("C", "Corporate extraction pad", [("IMP_Mining_Corp_Main", KV, "building"),
                                           ("IMP_Mining_Corp_Tanks", KV, "building"),
                                           ("IMP_Mining_Corp_Shed", KV, "building")]),
        ("D", "Fuel and supply point", [("IMP_Oil_Well_TankLarge", "Base", "building"),
                                        ("IMP_Oil_Well_TankSmall", "Base", "building"),
                                        ("IMP_Fort_BldC_Modern", "Base", "building"),
                                        ("IMP_Oil_Well_BarrelRed", "Base", "prop"), ("IMP_Fort_Truck_Modern", "Base", "prop")])),
}
# Packages that hold a piece under another name than landmarks/tilebases.
COLONY_PACKAGES = {AU: "IMP_Outback_Station"}


# The draft's layout: buildings in an arc across the back of the tile (tile
# units, u down-right and v down-left on screen), props at its ends, the middle
# and front left to the resource. Fronts face the viewer.
BACK = ((-.22, -.22), (-.36, .02), (.02, -.36), (-.06, -.40), (-.40, -.06))
ENDS = ((-.30, .26), (.26, -.30), (-.42, .14), (.14, -.42), (-.16, .34))
COLONY_SCALE = 1.8


def colony() -> list:
    import math
    from Renderer.lab.studies.districts.kits import SCALE
    size = SCALE * COLONY_SCALE
    cells = []
    for row, era in enumerate(ERAS):
        for col, (letter, label, pieces) in enumerate(COLONIES[era]):
            back, ends, parts = list(BACK), list(ENDS), []
            for entry, content, role in pieces:
                u, v = (back if role != "prop" else ends).pop(0)
                parts.append(part(tile(entry, content, COLONY_PACKAGES.get(content, "landmarks/tilebases")),
                                  place=(u / size, v / size, math.pi), ground_fit=.03))
            cells.append(cell(f"Col {AB[era]} {letter}", f"{era} · {letter} · {label}", row, col, parts, era=era,
                              facing="tile", snap=False, scale=COLONY_SCALE))
    return cells


# Colony concept A (the user's pick, 2026-10-23) on real resources: the
# pieces stand around the resource's footprint (colonies.layout), and the
# production resource itself is merged into the tile (colonies.merge).
COLONY_RESOURCES = {"Ancient": ("horses", "iron", "wines"), "Medieval": ("iron", "spices", "dyes"),
                    "Industrial": ("coal", "saltpeter", "rubber"), "Modern": ("oil", "aluminum", "uranium")}


def colony_resources() -> list:
    import math
    from Renderer.lab.studies.districts.kits import SCALE, Compiler
    from Renderer.lab.studies.tile_buildings import colonies
    compiler, size, cells = Compiler(), SCALE * COLONY_SCALE, []
    for row, era in enumerate(ERAS):
        _letter, _label, pieces = next(option for option in COLONIES[era] if option[0] == "A")
        records = [(tile(entry, content, COLONY_PACKAGES.get(content, "landmarks/tilebases")), role)
                   for entry, content, role in pieces]
        extents = [colonies.piece_extent(compiler, record, size) for record, _role in records]
        fits = [colonies.fit(role, extent[2], extent[3]) for (_r, role), extent in zip(records, extents)]
        for col, resource in enumerate(COLONY_RESOURCES[era]):
            placed = colonies.layout([(index, role, extent[2] * k, extent[3] * k) for index, ((_r, role), extent, k)
                                      in enumerate(zip(records, extents, fits))], colonies.resource_grid(resource))
            parts = [part(records[index][0], place=((u - .5) / size + extents[index][0],
                                                    (v - .5) / size + extents[index][1], math.pi),
                          ground_fit=.03, scale=factor * fits[index]) for index, u, v, factor in placed]
            cells.append(cell(f"CR {AB[era]} {resource}", f"{era} · A on {resource}", row, col, parts, era=era,
                              facing="tile", snap=False, scale=COLONY_SCALE, resource=resource,
                              dropped=len(records) - len(placed)))
    compiler.save()
    return cells


# The user's picks (2026-10-23), named as the runtime looks them up in the
# site pack: fortress_<era> and the barricade's keep_, wall_, tower_ and
# gate_<era> by the viewer's era (Civ III's rule), airfield_<0|1> (early,
# modern), outpost_<era>, radar, victory and colony pieces
# colony_<era>_<role>_<k> placed at run time around the resource.
OUTPOST = {"Ancient": tile(WATCHTOWER["Ancient"]), "Medieval": raw("Base", "landmarks/Walls", "IMP_Great_Wall_tower"),
           "Industrial": tile(WATCHTOWER["Industrial"]), "Modern": tile(WATCHTOWER["Modern"])}
BARRICADE_SCALE = .8
OUTPOST_SCALE = {"Ancient": TALL, "Medieval": TALL * .7, "Industrial": TALL, "Modern": TALL}


def picks() -> list:
    from Renderer.lab.studies.districts.kits import SCALE, Compiler
    from Renderer.lab.studies.tile_buildings import colonies
    aero = {"facing": aerodrome_turn(), "scale": .7, "draft": False, "snap": False, "runway": True}
    cells = []
    for row, era in enumerate(ERAS):
        cells += [cell(f"fortress_{row}", f"{era} fortress", row, 0, lone(tile(FORT_KEEP[era])), era=era),
                  cell(f"outpost_{row}", f"{era} outpost", row, 2, lone(OUTPOST[era], scale=OUTPOST_SCALE[era]), era=era)]
        # A barricade is the keep inside a ring the runtime lays round its dry
        # ground (objects::place_tile_buildings): the era's pieces, at the
        # drafted barricade's size (.8 of a lone fortress, as the user saw it).
        cells += [cell(f"{role}_{row}", f"{era} barricade {role}", row, 12 + k, lone(record, scale=BARRICADE_SCALE),
                       era=era)
                  for k, (role, record) in enumerate([("keep", tile(FORT_KEEP[era]))] + ring_records(era))]
    cells += [cell("airfield_0", "Early airfield", 0, 3, [part(landmark("LM_AIRSTRIP", "Industrial"))], era="Industrial"),
              cell("airfield_1", "Modern airfield", 1, 3, kit("DISTRICT_AERODROME", "HANGAR, AIRPORT", "Modern",
                                                              ("HANGAR", "AIRPORT")), era="Modern", **aero),
              cell("radar", "Radar tower", 2, 3, lone(tile("WON_Amundsen_Station_Radar", "DLC/Expansion1"),
                                                      scale=TALL * .8), era="Modern"),
              cell("victory", "Victory point", 3, 3,
                   lone(raw("Base", "environment/clutter", "IMP_Antiquity_Obelisk"), scale=TALL), state="default",
                   whiten=True)]
    # Colony pieces, one composition each, centred, at their capped size.
    compiler, size = Compiler(), SCALE * COLONY_SCALE
    for row, era in enumerate(ERAS):
        _letter, _label, pieces = next(option for option in COLONIES[era] if option[0] == "A")
        for k, (entry, content, role) in enumerate(pieces):
            record = tile(entry, content, COLONY_PACKAGES.get(content, "landmarks/tilebases"))
            extent = colonies.piece_extent(compiler, record, size)
            cells.append(cell(f"colony_{row}_{role}_{k}", f"{era} colony {entry}", row, 4 + k,
                              lone(record, scale=colonies.fit(role, extent[2], extent[3])),
                              era=era, facing=3.141592653589793, snap=False, scale=COLONY_SCALE))
    compiler.save()
    return cells


# Drafted cells (walled forts, colonies) skip the district study's city paving.
BARE = ({"zoom": 128, "hour": 12, "ground": False}, {"zoom": 256, "hour": 12, "ground": False})


def catalogs() -> dict:
    found = {
        "tb-fortress": {"title": "Fortress", "cells": fortress()},
        "tb-barricade": {"title": "Barricade", "cells": barricade(), "shots": BARE},
        "tb-barricade-line": barricade_line(),
        "tb-airfield": {"title": "Airfield", "cells": airfield()},
        "tb-outpost": {"title": "Outpost", "cells": outpost()},
        "tb-radar": {"title": "Radar tower", "cells": radar()},
        "tb-victory": {"title": "Victory point", "cells": victory()},
        "tb-colony": {"title": "Colony", "cells": colony(), "shots": BARE},
        "tb-colony-res": {"title": "Colony A around its resource", "cells": colony_resources(), "shots": BARE,
                          "resources": True},
    }
    for key, spec in found.items():
        spec["id"] = key
    return found
