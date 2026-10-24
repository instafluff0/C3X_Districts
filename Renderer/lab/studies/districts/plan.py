"""Catalogs of Civ VI district compositions proposed for the default C3X districts.

Each catalog is one Lab frame: cells on a row x column grid. A cell names the
Civ VI base variant and hero buildings it composes; `kits.base`/`kits.hero`
pick the variant Civ VI itself would choose for that building set and art era.
Rows are usually Civ III eras and columns the C3X building count, so a frame
reads like the C3X sprite sheet it would replace.
"""
from __future__ import annotations

import random

from Renderer.lab.studies.districts.kits import (base, city, city_buildings, city_tops, component, hero, kit_turn,
                                                 landmark, raw)

ERAS = ("Ancient", "Medieval", "Industrial", "Modern")
LATE = ("Industrial", "Modern")
B = "BUILDING_"
AB = {"Ancient": "An", "Medieval": "Md", "Industrial": "In", "Modern": "Mo"}


def part(record: dict, **options) -> dict:
    return {"record": record, **options}


def colosseum() -> dict:
    """Civ VI's Colosseum wonder at district size (Stadium-sized): the shell,
    seating and arena of its wonder-movie model, without the construction
    scaffolds, stone piles and set dressing the movie stages around it."""
    return part(raw("Base", "WonderMovie", "BUILDING_COLOSSEUM"),
                merge={"materials": (1, 5, 6, 7), "radius": .175}, scale=.68, squash=.72, fixed=True)


def kit(district: str, buildings: str, era: str, heroes=(), *, culture="DEFAULT", hero_district=None, extra=()):
    parts = [part(base(district, buildings, era, culture))]
    # The Colosseum and Stadium are drawn squat, the same footprint in every cell.
    parts += [colosseum() if name == "COLOSSEUM" else
              part(hero(hero_district or district, B + name, era, culture), squash=.75, scale=.82, fixed=True)
              if name == "STADIUM" else
              part(hero(hero_district or district, B + name, era, culture)) for name in heroes]
    return parts + list(extra)


def cell(name: str, label: str, row: int, col: int, parts: list, **options) -> dict:
    if len(name) > 23:
        raise ValueError(f"cell name too long: {name}")
    return {"name": name, "label": label, "row": row, "col": col, "parts": parts, **options}


def era_heroes(district: str, era: str, heroes: tuple) -> tuple:
    """One C3X building, the era's Civ VI model: the Colosseum is Civ VI's Stadium in the Modern era."""
    if era == "Modern" and district in ("DISTRICT_ENTERTAINMENT_COMPLEX", "DISTRICT_HIPPODROME"):
        return tuple("STADIUM" if name in ("COLOSSEUM", "TLACHTLI") else name for name in heroes)
    return heroes


TOWERS = ("DIS_CTY_MG_Bld_07", "DIS_CTY_MGG_Bld_05", "DIS_CTY_MG_Bld_01", "DIS_CTY_MGG_Bld_02")
ROW_HOUSES = ("DIS_CTY_INDRH_Bld01", "DIS_CTY_INDRH_Bld03", "DIS_CTY_INDRH_Bld05", "DIS_CTY_INDRH_Bld02")


def city_piece(entry: str) -> dict:
    return part(raw("Base", "landmarks/city_buildings", entry), filler=True)


def era_dressing(district: str, era: str, col: int) -> list:
    """Civ VI city buildings where its own district kit is thin: its Commercial Hub
    is market stalls in every era, while Civ VI's commercial skyline comes from the
    city generator. Industrial hubs gain row houses, Modern hubs glass towers, more
    as the hub grows."""
    if district not in ("DISTRICT_COMMERCIAL_HUB", "DISTRICT_SUGUBA"):
        return []
    if era == "Industrial":
        return [city_piece(entry) for entry in ROW_HOUSES[:1 + col]]
    if era == "Modern":
        return [city_piece(entry) for entry in TOWERS[:1 + col]]
    return []


TREES = ("Tree_B_Lg", "Tree_B_Sm", "Tree_C_Lg", "Tree_C_Sm", "Tree_D_Lg")
CULTURE_NAME = {"AMER": "American", "EURO": "European", "ROMAN": "Roman", "MIDEAST": "Middle Eastern",
                "ASIAN": "Asian"}
HOUSE_ERA = {"Ancient": "Ancient", "Medieval": "Middle Ages", "Industrial": "Industrial", "Modern": "Modern"}
DENSE = {"density": .68, "props": 4}
FAIR = {"density": .46, "copies": 2, "building_scale": 1.5, "props": 3}     # fewer, larger fair tents


def trees(names=TREES) -> list:
    """Civ VI trees, which the drafter scatters at random into the gaps."""
    return [part(component(name), role="plant") for name in names]


def houses(code: str, era: str, count: int = 6, seed: int = 0, max_top: float | None = None) -> list:
    """A seeded selection of the city pack's houses for a culture and era, as
    fillers, so a district blends with the city beside it; max_top keeps only
    low ones (kit units)."""
    pool = city_buildings()["houses"][f"{CULTURE_NAME[code]}|{HOUSE_ERA[era]}"]
    if max_top is not None:
        pool = [entry for entry in pool if city_tops().get(entry, 1.0) <= max_top]
    picks = random.Random(f"{code}|{era}|{seed}").sample(pool, min(count, len(pool)))
    return [part(city(entry), role="filler") for entry in sorted(picks)]


def ferris_wheel(district: str, era: str) -> list:
    """Industrial and Modern Entertainment Complexes add the Water Park's Ferris wheel."""
    if district not in ("DISTRICT_ENTERTAINMENT_COMPLEX", "DISTRICT_HIPPODROME") or era not in LATE:
        return []
    # Scaled so the wheel stands a little taller than the Stadium, not over it.
    return [part(hero("DISTRICT_WATER_ENTERTAINMENT_COMPLEX", B + "FERRIS_WHEEL", era), role="landmark", scale=.65,
                 fixed=True)]


def grid(prefix: str, district: str, columns, eras=ERAS, **options) -> list:
    """Era rows x building-set columns of one district; columns are (label, set, heroes)."""
    cells = []
    for row, era in enumerate(eras):
        for col, (label, buildings, heroes) in enumerate(columns):
            cells.append(cell(f"{prefix} {AB[era]}{col}", f"{era} · {label}", row, col,
                              kit(district, buildings, era, era_heroes(district, era, heroes),
                                  extra=era_dressing(district, era, col) + ferris_wheel(district, era)),
                              era=era, layout_key=f"{prefix}{col}", **options))
    return cells


INDUSTRIAL_FILLERS = ("DIS_PRD_Metal_Tanks_A", "DIS_PRD_Bld_A", "DIS_GEN_IND_Filler_RES_BldF",
                      "DIS_GEN_IND_Filler_RES_BldG", "DIS_GEN_IND_Filler_RES_BldH", "DIS_GEN_IND_Filler_RES_BldI")


def industrial_fillers() -> list:
    """Civ VI's Industrial Zone warehouses and tanks, so a late zone is not one
    building repeated."""
    return [part(component(name), role="filler") for name in INDUSTRIAL_FILLERS]


def aerodrome_turn() -> float:
    import math
    return kit_turn(hero("DISTRICT_AERODROME", B + "AIRPORT", "Modern")) + math.pi


CITY_WALLS = {"Ancient": ("CityWalls_ANC_wall_01", "CityWalls_ANC_sm_tower", "CityWalls_ANC_gate"),
              "Medieval": ("CityWalls_MED_wall", "CityWalls_MED_sm_tower", "CityWalls_MED_gate")}


def encampment_walls(era: str) -> list:
    """Ancient and Medieval Encampments: a closed ring of the era's Civ VI city
    walls (the walls Renderer64's cities use), small towers at the corners and
    the gate facing the viewer, in place of the kit's stake palisade."""
    if era not in CITY_WALLS:
        return []
    segment, tower, gate = CITY_WALLS[era]
    wall = lambda name, role: part(raw("Base", "landmarks/Walls", name), role=role)
    return [wall(segment, "wall"), wall(tower, "tower"), wall(gate, "gate")]


def walled_camp(cells: list) -> list:
    for cell_ in cells:
        era = cell_.get("era")
        if era in CITY_WALLS and cell_["parts"] and cell_["parts"][0]["record"]["name"] == "DISTRICT_ENCAMPMENT":
            cell_["parts"] = cell_["parts"] + encampment_walls(era)
            cell_["layout"] = {**cell_.get("layout", DENSE), "walls": "explicit"}
    return cells


WAREHOUSES = {   # (storage buildings, goods and carts by the doors)
    "Ancient": (("DIS_HBR_Classical_Shack", "DIS_HBR_Classical_Shack_B", "DIS_ENC_Shed"),
                ("PROP_Barrel_Classical", "IMP_Trading_Dome_Crate", "DIS_COM_Crate", "DIS_ENC_Crate_Classical")),
    "Medieval": (("DIS_HBR_Classical_Shack", "DIS_HBR_Classical_Shack_B", "DIS_ENC_Shed"),
                 ("PROP_Barrel_Classical", "IMP_Trading_Dome_Crate", "DIS_COM_Crate", "DIS_ENC_Crate_Classical")),
    "Industrial": (("IMP_Outback_IND_Bld_A", "IMP_Outback_IND_Bld_B", "IMP_Outback_IND_Bld_C", "DIS_PRD_Bld_A"),
                   ("PROP_Crate_Industrial", "PROP_Crate_Industrial", "DIS_PRD_Metal_Tanks_A", "PROP_Barrel_Classical")),
    "Modern": (("DIS_HBR_Modern_Warehouse", "DIS_HBR_Modern_Lg_Bldg"),
               ("DIS_HBR_Modern_Orange_Crate", "DIS_HBR_Modern_Teal_Crate", "DIS_HBR_Modern_Yellow_Crate",
                "DIS_GEN_Car_B", "DIS_GEN_Car_C")),
}


def distribution(era: str) -> list:
    """A Distribution Hub: a yard of the era's warehouses, its goods stacked
    between them (the Granary as the Ancient and Medieval storehouse)."""
    buildings, goods = WAREHOUSES[era]
    parts = [part(component(name), role="filler") for name in buildings]
    if era in ("Ancient", "Medieval"):
        parts.insert(0, part(hero("DISTRICT_CITY_CENTER", B + "GRANARY", era)))
    return parts + [part(component(name), role="prop") for name in goods]


def abandoned() -> list:
    """C3X's Abandoned district: what is left when a district falls outside any
    territory. On land, a ruined block (Civ VI's ruined city pieces) taken back
    by trees and scrub; at sea, a wreck among floating debris."""
    overgrowth = [part(component(name), role="plant") for name in
                  ("Tree_B_Lg", "Tree_C_Lg", "Tree_B_Sm", "Shrub_Round_Lg", "Shrub_Round_Single", "Shrub_semi-circle_A")]
    cells = []
    for col, letter in enumerate("ABC"):
        ruin = part(landmark("LM_POPPED_GOODY", "Ancient", f"DIS_CTY_CR_CityRuined_{letter}_NoCrates"), role="landmark",
                    whole=True, fixed=True, scale=.42)
        cells.append(cell(f"Ruin {letter}", f"Land · ruined block {letter}, overgrown", 0, col, [ruin] + overgrowth,
                          layout={"density": 0, "trees": 14, "props": 0}))
    wreck = part(landmark("RES_SHIPWRECK", "Ancient", "IMP_Shipwreck_Base"), role="landmark", whole=True, fixed=True, scale=.9)
    debris = part(landmark("LM_WATER_DOODADS", "Ancient", "CR_Destroyer_Destroyed_Water_01"), role="landmark", whole=True,
                  fixed=True, scale=.5)
    cells.append(cell("Wreck", "Sea · a wreck among floating debris", 0, 3, [wreck, debris], terrain="Sea",
                      layout={"density": 0, "trees": 0, "props": 0}))
    return cells


def stacks(count: int) -> list:
    """Civ VI's tall factory smokestacks (the city pack's industrial accent)."""
    return [part(component("DIS_PRD_FactoryStack"), role="building") for _ in range(count)]


def data_center(district: str, lab: bool) -> list:
    """A dense modern tech park: Civ VI's glass towers and modern city
    buildings round a campus or spaceport pad, with the Research Lab."""
    parts = [part(base(district, "EMPTY", "Modern"))]
    if lab:
        parts.append(part(hero("DISTRICT_CAMPUS", B + "RESEARCH_LAB", "Modern")))
    towers = city_buildings()["glass"]                 # glass towers only: no ornate stone blocks
    return parts + [part(city(entry), role="filler") for entry in towers] + trees(("Tree_B_Sm", "Tree_C_Sm"))


SKI_SCALE = .4


def ski_resort() -> list:
    """Civ VI's Ski Resort laid out on the Civ III mountain itself, in tile
    units around the peak at the tile centre (+u+v is toward the viewer): the
    lodge on the front foot, the gondola climbing from beside it to just
    behind the summit (Civ VI's authored slope taken out, every vertex seated
    on the drawn mountain), a cabin on each flank and a slalom of flags down the
    left slope. No trees of its own: the mountain keeps only what the map draws."""
    import math
    from Renderer.lab.studies.districts.kits import SCALE
    size = SCALE * SKI_SCALE
    piece = lambda name, u, v, turn=0.0, **options: part(component(name), place=(u / size, v / size, turn), **options)
    parts = [piece("IMP_Ski_Resort_LiftBldA", 0.0, .33, math.pi / 2, ramp=(0.0, -.43, .156), conform=True),
             piece("IMP_Ski_Resort_BldA", .17, .3),
             piece("IMP_Ski_Resort_BldB", -.24, .06),
             piece("IMP_Ski_Resort_BldC", .22, -.08),
             piece("IMP_Ski_Resort_SkiRack", .07, .42), piece("IMP_Ski_Resort_SkiRack", .1, .44),
             piece("IMP_Ski_Resort_SnowboardRED", .05, .44), piece("IMP_Ski_Resort_SnowboardPURPLE", .12, .41)]
    for k in range(8):     # slalom gates down the left slope
        t = k / 7
        side = .035 if k % 2 else -.035
        parts.append(piece("IMP_Ski_Resort_Flag", -.1 - .18 * t + side * .7, -.02 + .32 * t + side * .7, scale=1.8))
    return parts


def catalogs() -> dict:
    global AERODROME
    AERODROME = {"facing": aerodrome_turn(), "scale": .7, "draft": False, "snap": False, "runway": True}
    c = {}

    c["campus"] = {"title": "Campus", "layout": {"density": .34, "copies": 2, "props": 3},
                   "cells": grid("Campus", "DISTRICT_CAMPUS", [
        ("no buildings", "EMPTY", ()),
        ("Library", "LIBRARY", ("LIBRARY",)),
        ("Library + University", "LIBRARY, UNIVERSITY", ("LIBRARY", "UNIVERSITY"))])}

    # SAM Missile Battery needs Rocketry, so a second Encampment building only
    # exists in the Modern row; it borrows the Armory's slot.
    enc = grid("Enc", "DISTRICT_ENCAMPMENT", [
        ("no buildings", "EMPTY", ()), ("Barracks", "BARRACKS", ("BARRACKS",))])
    silo = landmark("LM_MISSILE_SILO", "Modern")
    armory = hero("DISTRICT_ENCAMPMENT", B + "ARMORY", "Modern")
    enc += [cell("Enc M2", "Modern · Barracks + SAM (silo in Armory slot)", 3, 2,
                 kit("DISTRICT_ENCAMPMENT", "BARRACKS, ARMORY", "Modern", ("BARRACKS",),
                     extra=[part(silo, move_to=armory, scale=.7)]), era="Modern"),
            cell("Enc M3", "Modern · Barracks + Armory (reference)", 3, 3,
                 kit("DISTRICT_ENCAMPMENT", "BARRACKS, ARMORY", "Modern", ("BARRACKS", "ARMORY")), era="Modern"),
            cell("Enc I2", "Industrial · Barracks + Armory (reference)", 2, 2,
                 kit("DISTRICT_ENCAMPMENT", "BARRACKS, ARMORY", "Industrial", ("BARRACKS", "ARMORY")), era="Industrial"),
            cell("Enc fort M", "Modern · Civ VI Fort (alternative)", 1, 3, [part(landmark("LM_FORT", "Modern"))]),
            cell("Enc fort I", "Medieval · Civ VI Fort (alternative)", 0, 3, [part(landmark("LM_FORT", "Medieval"))])]
    c["encampment"] = {"title": "Encampment", "layout": DENSE, "cells": walled_camp(enc)}

    worship = [("Cathedral", "CATHEDRAL", "CATHEDRAL"), ("Meeting House", "MEETING", "MEETING_HOUSE"),
               ("Mosque", "MOSQUE", "MOSQUE"), ("Dar-e Mehr", "DAR_E_MEHR", "DAR_E_MEHR"),
               ("Synagogue", "SYNAGOGUE", "SYNAGOGUE"), ("Pagoda", "PAGODA", "PAGODA"), ("Wat", "WAT", "WAT"),
               ("Stupa", "STUPA", "STUPA"), ("Gurdwara", "GUADWARA", "GURDWARA")]
    holy = []
    for index, (label, token, building) in enumerate(worship):
        holy.append(cell(f"Worship {index}", f"Medieval · Temple + {label}", index // 4, index % 4,
                         kit("DISTRICT_HOLY_SITE", f"SHRINE, TEMPLE, {token}", "Medieval",
                             ("TEMPLE", building)) + trees(), era="Medieval"))
    holy += [cell("Worship stave", "Medieval · Stave Church (temple slot) + Cathedral", 2, 1,
                  kit("DISTRICT_HOLY_SITE", "SHRINE, STAVE, CATHEDRAL", "Medieval",
                      ("STAVE_CHURCH", "CATHEDRAL")) + trees(), era="Medieval"),
             cell("Worship prasat", "Medieval · Prasat (temple slot) + Pagoda", 2, 2,
                  kit("DISTRICT_HOLY_SITE", "SHRINE, PRASAT, PAGODA", "Medieval", ("PRASAT", "PAGODA")) + trees(),
                  era="Medieval"),
             cell("Lavra M", "Medieval · Lavra base + Temple + Cathedral", 2, 3,
                  kit("DISTRICT_LAVRA", "SHRINE, TEMPLE, CATHEDRAL", "Medieval", ("TEMPLE", "CATHEDRAL")) + trees(),
                  era="Medieval"),
             cell("Worship I mosque", "Industrial · Temple + Mosque", 3, 0,
                  kit("DISTRICT_HOLY_SITE", "SHRINE, TEMPLE, MOSQUE", "Industrial", ("TEMPLE", "MOSQUE")) + trees(),
                  era="Industrial"),
             cell("Worship I pagoda", "Industrial · Prasat + Pagoda", 3, 1,
                  kit("DISTRICT_HOLY_SITE", "SHRINE, PRASAT, PAGODA", "Industrial", ("PRASAT", "PAGODA")) + trees(),
                  era="Industrial"),
             cell("Worship I meeting", "Industrial · Temple + Meeting House", 3, 2,
                  kit("DISTRICT_HOLY_SITE", "SHRINE, TEMPLE, MEETING", "Industrial",
                      ("TEMPLE", "MEETING_HOUSE")) + trees(), era="Industrial"),
             cell("Ziggurat", "Ancient · Sumerian Ziggurat (alternative)", 3, 3, [part(landmark("LM_ZIGGURAT", "Ancient"))])]
    c["holy-culture"] = {"title": "Holy Site worship buildings by culture", "layout": {"density": .32, "trees": 7},
                         "cells": holy}

    ent = grid("Ent", "DISTRICT_ENTERTAINMENT_COMPLEX", [
        ("no buildings", "EMPTY", ()), ("Colosseum (Stadium in Modern)", "ARENA", ("COLOSSEUM",))])
    for col, (label, district, buildings, heroes) in enumerate((
            ("Hippodrome + Colosseum", "DISTRICT_HIPPODROME", "Arena", ("COLOSSEUM",)),
            ("Arena (Civ VI jousting ground)", "DISTRICT_ENTERTAINMENT_COMPLEX", "ARENA", ("ARENA",))), start=2):
        for row, era in enumerate(ERAS):
            try:
                ent.append(cell(f"Ent alt {AB[era]}{col}", f"{era} · {label}", row, col,
                                kit(district, buildings, era,
                                    era_heroes(district, era, heroes) if "COLOSSEUM" in heroes else heroes,
                                    extra=ferris_wheel(district, era)), era=era, layout_key=f"Ent alt {col}"))
            except KeyError:
                pass
    c["entertainment"] = {"title": "Entertainment Complex", "layout": FAIR, "cells": ent}

    com = []
    # Poland's Sukiennice package fails a size check in this install, so it is left out.
    options = (("Grand Bazaar (Ottoman bank)", "DISTRICT_COMMERCIAL_HUB",
                ("MARKET", ("MARKET",)), ("MARKET, BAZAAR", ("MARKET", "GRAND_BAZAAR")),
                ("MARKET, BAZAAR, STOCK_EXCHANGE", ("MARKET", "GRAND_BAZAAR", "STOCK_EXCHANGE"))),
               ("Suguba (Malian market)", "DISTRICT_SUGUBA",
                ("MARKET", ("MARKET",)), ("MARKET, BANK", ("MARKET", "BANK")),
                ("MARKET, BANK, STOCK_EXCHANGE", ("MARKET", "BANK", "STOCK_EXCHANGE"))))
    for block, (label, district, *stages) in enumerate(options):
        for e, era in enumerate(("Medieval", "Industrial")):
            for col, (buildings, heroes) in enumerate(stages):
                try:
                    com.append(cell(f"ComC {block}{AB[era]}{col}", f"{era} · {label} · {len(heroes)} building(s)",
                                    block * 2 + e, col, kit(district, buildings, era, heroes), era=era))
                except KeyError:
                    pass
    c["commercial-culture"] = {"title": "Commercial Hub culture variants", "layout": DENSE, "cells": com}

    ind = []
    for row, era in enumerate(LATE):
        # The Factory reads through the coal plant's brick works and stacks; the
        # Manufacturing Plant adds the oil plant's complex (both Civ VI Energy Grid art).
        for col, (label, sets, heroes, chimneys) in enumerate((
                ("no buildings", "EMPTY", (), 0), ("no buildings, Workshop dressing", "WORKSHOP", ("WORKSHOP",), 0),
                ("Factory (coal-plant works)", "WORKSHOP, FACTORY, COAL_POWER_PLANT",
                 ("WORKSHOP", "COAL_POWER_PLANT"), 1),
                ("Factory + Manufacturing Plant (oil-plant complex)", "WORKSHOP, FACTORY, FOSSIL_FUEL_POWER_PLANT",
                 ("WORKSHOP", "COAL_POWER_PLANT", "FOSSIL_FUEL_POWER_PLANT"), 1))):
            ind.append(cell(f"IZ {AB[era]}{col}", f"{era} · {label}", row, col,
                            kit("DISTRICT_INDUSTRIAL_ZONE", sets, era, heroes,
                                extra=stacks(chimneys) + industrial_fillers()), era=era, layout_key=f"IZ {col}"))
    c["industrial"] = {"title": "Industrial Zone", "layout": DENSE, "cells": ind}

    energy = []
    for row, era in enumerate(LATE):
        plants = (("no plant", "WORKSHOP, FACTORY, COAL_POWER_PLANT", None),
                  ("Coal Plant", "WORKSHOP, FACTORY, COAL_POWER_PLANT", "COAL_POWER_PLANT"),
                  ("Nuclear Plant", "WORKSHOP, FACTORY, POWER_PLANT", "POWER_PLANT"),
                  ("Oil Plant (alternative)", "WORKSHOP, FACTORY, FOSSIL_FUEL_POWER_PLANT", "FOSSIL_FUEL_POWER_PLANT"))
        for col, (label, buildings, plant) in enumerate(plants):
            energy.append(cell(f"Grid {AB[era]}{col}", f"{era} · {label}", row, col,
                               kit("DISTRICT_INDUSTRIAL_ZONE", buildings, era, (plant,) if plant else ()), era=era))
    energy += [cell("Grid hydro", "Modern · Hydro Plant (Civ VI Dam + Hydroelectric Dam)", 2, 0,
                    kit("DISTRICT_DAM", "HYDROELECTRIC_DAM", "Modern", ("HYDROELECTRIC_DAM",)), era="Modern"),
               cell("Grid solar", "Modern · Solar Plant (Civ VI Solar Farm)", 2, 1,
                    [part(landmark("LM_SOLAR_FARM", "Modern"))]),
               cell("Grid wind", "Modern · Wind Farm (alternative)", 2, 2, [part(landmark("LM_WIND_FARM", "Modern"))]),
               cell("Grid geothermal", "Modern · Geothermal Plant (alternative)", 2, 3,
                    [part(landmark("LM_GEOTHERMAL_PLANT", "Modern"))])]
    c["energy-grid"] = {"title": "Energy Grid", "layout": DENSE, "cells": energy}

    data = []
    for row, (label, district) in enumerate((("Modern Campus with glass towers", "DISTRICT_CAMPUS"),
                                             ("Spaceport pad with glass towers", "DISTRICT_SPACEPORT"))):
        for col in range(2):
            data.append(cell(f"Data {row}{col}", f"{label} · {'Research Lab' if col else 'no Research Lab'}",
                             row, col, data_center(district, col), era="Modern"))
    c["data-center"] = {"title": "Data Center candidates", "layout": {"density": .72, "props": 3}, "cells": data}

    # Civ VI's own Aerodrome layout, not re-drafted: the whole kit turned so the
    # airport's tarmac lines up with the tile, at a size that fits it.
    aero = grid("Aero", "DISTRICT_AERODROME", [
        ("no buildings", "EMPTY", ()), ("Airport (Hangar + Airport)", "HANGAR, AIRPORT", ("HANGAR", "AIRPORT")),
        ("Hangar only (alternative)", "HANGAR", ("HANGAR",))], eras=LATE, **AERODROME)
    aero.append(cell("Aero strip", "Modern · Civ VI Airstrip (alternative)", 1, 3,
                     [part(landmark("LM_AIRSTRIP", "Modern"))]))
    c["aerodrome"] = {"title": "Aerodrome", "snap": False, "cells": aero}

    c["port"] = {"title": "Port", "cells": grid("Port", "DISTRICT_HARBOR", [
        ("no buildings", "EMPTY", ()), ("Harbor = Lighthouse", "LIGHTHOUSE", ("LIGHTHOUSE",)),
        ("Harbor + Commercial Dock = Shipyard", "LIGHTHOUSE, SHIPYARD", ("LIGHTHOUSE", "SHIPYARD")),
        ("+ Seaport (alternative 3rd stage)", "LIGHTHOUSE, SHIPYARD, SEAPORT", ("LIGHTHOUSE", "SHIPYARD", "SEAPORT"))],
        terrain="Coast")}

    port = []
    for row, (label, district) in enumerate((("Cothon (Carthaginian)", "DISTRICT_COTHON"),
                                             ("Royal Navy Dockyard (English)", "DISTRICT_ROYAL_NAVY_DOCKYARD"))):
        for col, era in enumerate(ERAS):
            port.append(cell(f"PortC {row}{AB[era]}", f"{era} · {label} · Harbor + Commercial Dock", row, col,
                             kit(district, "LIGHTHOUSE, SHIPYARD", era, ("LIGHTHOUSE", "SHIPYARD")),
                             era=era, terrain="Coast"))
    c["port-culture"] = {"title": "Port culture variants", "cells": port}

    coastal = [cell("Rig base", "no platform = the Offshore Oil Rig's deck without its derrick", 0, 0,
                    [part(landmark("LM_OFFSHORE_OIL_RIG", "Industrial"), clip_z=.085)], terrain="Sea"),
               cell("Rig I", "Industrial · Offshore Platform = Civ VI Offshore Oil Rig", 0, 1,
                    [part(landmark("LM_OFFSHORE_OIL_RIG", "Industrial"))], terrain="Sea"),
               cell("Rig M", "Modern · Offshore Oil Rig", 0, 2, [part(landmark("LM_OFFSHORE_OIL_RIG", "Modern"))],
                    terrain="Sea"),
               cell("Offshore wind", "Modern · Offshore Wind Farm (alternative)", 0, 3,
                    [part(landmark("LM_OFFSHORE_WIND_FARM", "Modern"))], terrain="Sea")]
    for col, (label, buildings, heroes) in enumerate((
            ("Water Park base", "EMPTY", ()), ("Ferris Wheel", "FERRIS_WHEEL", ("FERRIS_WHEEL",)),
            ("Ferris Wheel + Aquarium", "FERRIS_WHEEL, AQUARIUM", ("FERRIS_WHEEL", "AQUARIUM")),
            ("+ Aquatics Center", "FERRIS_WHEEL, AQUARIUM, AQUATICS_CENTER", ("FERRIS_WHEEL", "AQUARIUM", "AQUATICS_CENTER")))):
        # C3X builds the Water Park in the water, as Civ VI does.
        coastal.append(cell(f"Water {col}", f"Modern · {label}", 1, col,
                            kit("DISTRICT_WATER_ENTERTAINMENT_COMPLEX", buildings, "Modern", heroes), era="Modern",
                            terrain="Sea"))
    coastal.append(cell("Beach resort", "Beach Resort (alternative)", 1, 4, [part(landmark("LM_BEACH_RESORT", "Modern"))]))
    c["coastal"] = {"title": "Offshore Extraction Zone and Water Park", "cells": coastal}

    leisure = [cell(f"Park {AB[era]}", f"{era} · City Park", 0, col, [part(landmark("LM_CITY_PARK", era))])
               for col, era in enumerate(ERAS)]
    leisure += [cell("Park forest", "Industrial · City Park on forest", 1, 0,
                     [part(landmark("LM_CITY_PARK", "Industrial"))], terrain="Forest"),
                cell("National park", "National Park (alternative)", 1, 1, [part(landmark("NATIONAL_PARK", "Modern"))]),
                cell("Golf course", "Industrial · Golf Course (alternative)", 1, 2,
                     [part(landmark("LM_GOLF_COURSE", "Industrial"))]),
                cell("Preserve", "Preserve + Grove + Sanctuary (alternative)", 1, 3,
                     kit("DISTRICT_PRESERVE", "Grove, Sanctuary", "Modern", ("GROVE", "SANCTUARY")))]
    leisure += [cell("Ski I", "Ski Resort on a snow-capped mountain", 2, 0, ski_resort(), terrain="Mountains",
                     scale=SKI_SCALE, facing="tile", snow=True, snow_cover=.7),
                cell("Ski hockey", "Ice Hockey Rink (alternative)", 2, 1,
                     [part(landmark("LM_ICE_HOCKEY_RINK", "Modern"))], terrain="Tundra")]
    c["leisure"] = {"title": "Park and Ski Resort", "cells": leisure}

    # Warehouses and stacked goods, as C3X's sprite shows: one storage yard, its
    # buildings from Civ VI's harbour, encampment and industrial kits by era.
    c["distribution"] = {"title": "Distribution Hub", "draft": "force", "layout": {"density": .66, "copies": 4, "props": 5, "building_scale": 1.4},
                         "cells": [cell(f"Dist {AB[era]}", f"{era} · warehouses and goods", 0, col, distribution(era),
                                        era=era, layout_key="Dist") for col, era in enumerate(ERAS)]}
    c["abandoned"] = {"title": "Abandoned district", "draft": "force", "cells": abandoned()}

    nbh = []
    for col, entry in enumerate(("DIS_NBH_Industrial_Base_01", "DIS_NBH_Industrial_Base_03", "DIS_NBH_Industrial_Base_05")):
        nbh.append(cell(f"Nbh I{col}", f"Industrial · Civ VI Neighborhood variant {col + 1}", 0, col,
                        [part(raw("Base", "landmarks/tilebases", entry))]))
    for col, entry in enumerate(("DIS_NBH_Modern_Base_01", "DIS_NBH_Modern_Base_03", "DIS_NBH_Modern_Base_05")):
        nbh.append(cell(f"Nbh M{col}", f"Modern · Civ VI Neighborhood variant {col + 1}", 1, col,
                        [part(raw("Base", "landmarks/tilebases", entry))]))
    c["neighborhood"] = {"title": "Civ VI Neighborhood bases", "cells": nbh}

    topo = []
    for col, era in enumerate(("Ancient", "Medieval", "Industrial")):
        topo.append(cell(f"Canal {AB[era]}", f"{era} · Canal (straight)", 0, col,
                         [part(next(r for r in _canals(era) if "Straight" in r["entry"]))], era=era))
    for col, entry in enumerate(("MED_Bridge", "IND_Bridge", "MOD_Bridge")):
        topo.append(cell(f"Bridge {col}", f"Bridge kit {entry}", 1, col,
                         [part(raw("Base", "route_doodads", entry), centre=True)]))
    # Civ VI's Great Wall is procedural wall geometry (Walls.artdef GreatWallSet), not a landmark kit.
    topo += [cell("State built", "Medieval Campus · completed", 3, 0,
                  kit("DISTRICT_CAMPUS", "LIBRARY", "Medieval", ("LIBRARY",)), era="Medieval"),
             cell("State building", "Medieval Campus · under construction", 3, 1,
                  kit("DISTRICT_CAMPUS", "LIBRARY", "Medieval", ("LIBRARY",)), era="Medieval", state="construction"),
             cell("State pillaged", "Medieval Campus · pillaged (C3X 'Abandoned')", 3, 2,
                  kit("DISTRICT_CAMPUS", "LIBRARY", "Medieval", ("LIBRARY",)), era="Medieval", state="pillaged")]
    c["topology"] = {"title": "Canal, Bridge, Great Wall and district states", "cells": topo}

    lone = (("Barracks", "DISTRICT_ENCAMPMENT", "BARRACKS", "Medieval"),
            ("Barracks (Industrial)", "DISTRICT_ENCAMPMENT", "BARRACKS", "Modern"),
            ("Armory", "DISTRICT_ENCAMPMENT", "ARMORY", "Medieval"),
            ("Temple", "DISTRICT_HOLY_SITE", "TEMPLE", "Medieval"),
            ("Cathedral", "DISTRICT_HOLY_SITE", "CATHEDRAL", "Medieval"),
            ("Library", "DISTRICT_CAMPUS", "LIBRARY", "Medieval"),
            ("University", "DISTRICT_CAMPUS", "UNIVERSITY", "Medieval"),
            ("Research Lab", "DISTRICT_CAMPUS", "RESEARCH_LAB", "Modern"),
            ("Arena (jousting ground)", "DISTRICT_ENTERTAINMENT_COMPLEX", "ARENA", "Medieval"),
            ("Market", "DISTRICT_COMMERCIAL_HUB", "MARKET", "Medieval"),
            ("Bank", "DISTRICT_COMMERCIAL_HUB", "BANK", "Medieval"),
            ("Stock Exchange", "DISTRICT_COMMERCIAL_HUB", "STOCK_EXCHANGE", "Modern"))
    c["buildings-a"] = {"title": "Hero buildings I", "cells": [
        cell(f"Hero a{index}", label, index // 4, index % 4, [part(hero(district, B + name, era), centre=True)])
        for index, (label, district, name, era) in enumerate(lone)]}
    lone = (("Factory", "DISTRICT_INDUSTRIAL_ZONE", "FACTORY", "Industrial"),
            ("Electronics Factory", "DISTRICT_INDUSTRIAL_ZONE", "ELECTRONICS_FACTORY", "Modern"),
            ("Coal Power Plant", "DISTRICT_INDUSTRIAL_ZONE", "COAL_POWER_PLANT", "Industrial"),
            ("Nuclear Power Plant", "DISTRICT_INDUSTRIAL_ZONE", "POWER_PLANT", "Modern"),
            ("Oil Power Plant", "DISTRICT_INDUSTRIAL_ZONE", "FOSSIL_FUEL_POWER_PLANT", "Modern"),
            ("Airport", "DISTRICT_AERODROME", "AIRPORT", "Modern"),
            ("Hangar", "DISTRICT_AERODROME", "HANGAR", "Modern"),
            ("Lighthouse", "DISTRICT_HARBOR", "LIGHTHOUSE", "Medieval"),
            ("Shipyard", "DISTRICT_HARBOR", "SHIPYARD", "Industrial"),
            ("Seaport", "DISTRICT_HARBOR", "SEAPORT", "Modern"),
            ("Food Market", "DISTRICT_NEIGHBORHOOD", "FOOD_MARKET", "Modern"),
            ("Shopping Mall", "DISTRICT_NEIGHBORHOOD", "SHOPPING_MALL", "Modern"))
    c["buildings-b"] = {"title": "Hero buildings II", "cells": [
        cell(f"Hero b{index}", label, index // 4, index % 4, [part(hero(district, B + name, era), centre=True)])
        for index, (label, district, name, era) in enumerate(lone)]}
    c.update(contexts())
    # Land districts are re-drafted onto the square tile (kits._draft). Coastal
    # kits keep Civ VI's layout so their docks still meet the water; canal,
    # bridge, Great Wall and lone-building catalogs are already tile-shaped.
    for key in ("campus", "encampment", "holy-culture", "entertainment", "commercial-culture", "industrial",
                "energy-grid", "data-center", "leisure", "distribution", "neighborhood",
                "context-ancient", "context-medieval", "context-industrial", "context-modern"):
        c[key]["draft"] = "force" if c[key].get("draft") == "force" else True
        c[key].setdefault("layout", DENSE)
    for state in c["topology"]["cells"]:
        if state["name"].startswith("State"):
            state["draft"] = True
    c.update(great_wall())
    c.update(sheets())
    for key, spec in c.items():
        spec["id"] = key
    return c


# Gameplay and close-up zoom at noon and night, plus a close-up without the
# city-paving ground cover.
CONTEXT_SHOTS = ({"zoom": 128, "hour": 12}, {"zoom": 256, "hour": 12}, {"zoom": 128, "hour": 22},
                 {"zoom": 256, "hour": 22}, {"tag": "-bare", "zoom": 256, "hour": 12, "ground": False})
# Raw offsets of a city's eight neighbours: E, W, S, N, SE, NW, NE, SW on screen.
E, W, S, N, SE, NW, NE, SW = (2, 0), (-2, 0), (0, 2), (0, -2), (1, 1), (-1, -1), (1, -1), (-1, 1)


def contexts() -> dict:
    """Districts on the tiles around a production city, to judge size, shadows
    and light against accepted art. A Town-size city on one neighbour shows the
    Neighborhood-from-city-art proposal."""
    def scene(culture, era, size, entries):
        cells = []
        for index, (offset, label, parts, *options) in enumerate(entries):
            # A stable per-tile choice between the two facings, as the game would make.
            cells.append(cell(f"Ctx {index}", label, 0, index, parts, era=era, dx=offset[0], dy=offset[1],
                              **{"facing": "SW" if index % 2 == 0 else "SE", **(options[0] if options else {})}))
        return cells

    hub = lambda col: {"layout": HUB_GROWTH[col]}
    temple = {"layout": HOLY}
    nbh = lambda variant: {"draft": "force", "layout": NEIGHBORHOODS[variant][1]}

    return {
        "context-ancient": {"title": "Ancient Middle Eastern town with districts", "shots": CONTEXT_SHOTS, "cities": [(0, 0, 3, 0, 0, 0, 0)], "cells": scene(3, "Ancient", 0, [
            (E, "Holy Site · Temple", holy("MIDEAST", "Ancient", 1), temple),
            (W, "Encampment · Barracks", kit("DISTRICT_ENCAMPMENT", "BARRACKS", "Ancient", ("BARRACKS",)) +
             encampment_walls("Ancient"), {"layout": {**DENSE, "walls": "explicit"}}),
            (SE, "Campus · Library", kit("DISTRICT_CAMPUS", "LIBRARY", "Ancient", ("LIBRARY",))),
            (NE, "Commercial Hub · Marketplace", commercial("MIDEAST", "Ancient", 1), hub(1)),
            (S, "Neighborhood · variant 1", neighborhood("MIDEAST", "Ancient", 0), nbh(0))])},
        "context-medieval": {"title": "Medieval European city with districts", "shots": CONTEXT_SHOTS, "cities": [(0, 0, 1, 1, 1, 0, 0)], "cells": scene(1, "Medieval", 1, [
            (E, "Campus · Library + University",
             kit("DISTRICT_CAMPUS", "LIBRARY, UNIVERSITY", "Medieval", ("LIBRARY", "UNIVERSITY"))),
            (W, "Holy Site · Temple + Cathedral", holy("EURO", "Medieval", 2), temple),
            (SE, "Commercial Hub · Marketplace + Bank", commercial("EURO", "Medieval", 2), hub(2)),
            (S, "Neighborhood · variant 2", neighborhood("EURO", "Medieval", 1), nbh(1)),
            (NW, "Encampment · Barracks", kit("DISTRICT_ENCAMPMENT", "BARRACKS", "Medieval", ("BARRACKS",)) +
             encampment_walls("Medieval"), {"layout": {**DENSE, "walls": "explicit"}}),
            (NE, "Entertainment Complex · Colosseum",
             kit("DISTRICT_ENTERTAINMENT_COMPLEX", "ARENA", "Medieval", ("COLOSSEUM",)))])},
        "context-industrial": {"title": "Industrial Roman city with districts", "shots": CONTEXT_SHOTS, "cities": [(0, 0, 2, 2, 1, 0, 0)], "cells": scene(2, "Industrial", 1, [
            (E, "Industrial Zone · Factory", kit("DISTRICT_INDUSTRIAL_ZONE", "WORKSHOP, FACTORY, COAL_POWER_PLANT",
                                                 "Industrial", ("WORKSHOP", "COAL_POWER_PLANT"),
                                                 extra=stacks(1) + industrial_fillers())),
            (W, "Commercial Hub · all three", commercial("ROMAN", "Industrial", 3), hub(3)),
            (S, "Neighborhood · variant 3", neighborhood("ROMAN", "Industrial", 2), nbh(2)),
            (SE, "Campus · Library + University",
             kit("DISTRICT_CAMPUS", "LIBRARY, UNIVERSITY", "Industrial", ("LIBRARY", "UNIVERSITY"))),
            (NW, "Holy Site · Temple + Cathedral", holy("ROMAN", "Industrial", 2), temple),
            (NE, "Energy Grid · Coal Plant",
             kit("DISTRICT_INDUSTRIAL_ZONE", "WORKSHOP, FACTORY, COAL_POWER_PLANT", "Industrial", ("COAL_POWER_PLANT",))),
            (N, "Entertainment Complex · Colosseum",
             kit("DISTRICT_ENTERTAINMENT_COMPLEX", "ARENA", "Industrial", ("COLOSSEUM",))),
            (SW, "Encampment · Barracks", kit("DISTRICT_ENCAMPMENT", "BARRACKS", "Industrial", ("BARRACKS",)))])},
        "context-modern": {"title": "Modern Asian metropolis with districts", "shots": CONTEXT_SHOTS, "cities": [(0, 0, 4, 3, 2, 0, 0)], "cells": scene(4, "Modern", 2, [
            (E, "Industrial Zone · Factory + Manufacturing Plant",
             kit("DISTRICT_INDUSTRIAL_ZONE", "WORKSHOP, FACTORY, FOSSIL_FUEL_POWER_PLANT", "Modern",
                 ("WORKSHOP", "COAL_POWER_PLANT", "FOSSIL_FUEL_POWER_PLANT"), extra=stacks(1) + industrial_fillers())),
            (W, "Commercial Hub · all three", commercial("ASIAN", "Modern", 3), hub(3)),
            (S, "Neighborhood · variant 1", neighborhood("ASIAN", "Modern", 0), nbh(0)),
            (SE, "Campus · Library + University",
             kit("DISTRICT_CAMPUS", "LIBRARY, UNIVERSITY", "Modern", ("LIBRARY", "UNIVERSITY"))),
            (NW, "Aerodrome · Airport", kit("DISTRICT_AERODROME", "HANGAR, AIRPORT", "Modern", ("HANGAR", "AIRPORT")),
             AERODROME),
            (NE, "Energy Grid · Nuclear Plant",
             kit("DISTRICT_INDUSTRIAL_ZONE", "WORKSHOP, FACTORY, POWER_PLANT", "Modern", ("POWER_PLANT",))),
            (N, "Entertainment Complex · Colosseum (Stadium)",
             kit("DISTRICT_ENTERTAINMENT_COMPLEX", "ARENA", "Modern", ("STADIUM",))),
            (SW, "Data Center · Research Lab", data_center("DISTRICT_CAMPUS", True), {"layout": {"density": .72}})])},
    }


def _canals(era: str) -> list:
    from Renderer.lab.studies.districts.kits import census, ART_ERAS
    rows = [row for row in census() if row["name"] == "DISTRICT_CANAL" and row["child"] == "BaseVariants"]
    for art_era in ART_ERAS[era]:
        found = [row for row in rows if row["era"] == art_era]
        if found:
            return found
    return rows


# --- Great Wall --------------------------------------------------------------
# C3X draws a pillar on every Great Wall tile and a half wall toward each edge
# neighbour (NE, SE, SW, NW) that is also Great Wall of the same owner. The 3D
# proposal keeps that topology with Civ VI's Great Wall kit: two wall pieces
# from the tile centre to each shared edge, a tower at the centre, the tower
# with a building at corners, the coastal end tower at chain ends, and the gate
# where a road crosses.
WALL_EDGES = {"NE": ((1, -1), (0.0, -1.0)), "SE": ((1, 1), (1.0, 0.0)),
              "SW": ((-1, 1), (0.0, 1.0)), "NW": ((-1, -1), (-1.0, 0.0))}   # raw offset, (u, v) direction


def wall_tile(mask: tuple, gate: bool = False) -> list:
    import math
    from Renderer.lab.studies.districts.kits import SCALE
    piece = lambda name: raw("Base", "landmarks/Walls", name)
    parts = []
    turns = []
    for edge in mask:
        du, dv = WALL_EDGES[edge][1]
        turn = math.atan2(-du, dv)                 # local +y along the run
        turns.append(turn)
        # Half-wall pieces (0.05 tile, 0.125 at 2.5x): four reach the edge. Each sits on the
        # lowest ground under it, so a run steps up a slope instead of floating over it.
        for k in range(4):
            parts.append(part(piece("IMP_Great_Wall_halfwall"),
                              place=(du * k * .125 / SCALE, dv * k * .125 / SCALE, turn), ground_fit=.05))
    turn = turns[0] if turns else 0.0
    straight = len(mask) == 2 and {mask[0], mask[1]} in ({"NE", "SW"}, {"NW", "SE"})
    centre = ("IMP_Great_Wall_gate" if gate else "IMP_Great_Wall_end" if len(mask) == 1 else
              "IMP_Great_Wall_tower" if straight or not mask else "IMP_Great_Wall_tower_building")
    parts.append(part(piece(centre), place=(0.0, 0.0, turn), ground_fit=.05))
    return parts


def wall_scene(tiles: dict, gates=(), roads=()) -> tuple[list, list]:
    """Cells (one per distinct neighbour mask) and the tile list for a run."""
    cells, placed, names = [], [], {}
    for (x, y), terrain in tiles.items():
        mask = tuple(edge for edge, ((ox, oy), _) in WALL_EDGES.items() if (x + ox, y + oy) in tiles)
        gate = (x, y) in gates
        name = "GW " + ("gate " if gate else "") + ("-".join(mask) or "alone")
        if name not in names:
            names[name] = len(cells)
            cells.append(cell(name, name.removeprefix("GW ").replace("-", " ") or "alone", 0, len(cells),
                              wall_tile(mask, gate), facing="tile"))
        placed.append((x, y, name + ("|road" if gate else ""), terrain))
    placed += [(x, y, "|road", "Grassland") for x, y in roads]
    return cells, placed


def great_wall() -> dict:
    run = {(-6, 6): "Grassland", (-5, 5): "Grassland", (-4, 4): "Hills", (-3, 3): "Hills",
           (-2, 2): "Mountains", (-1, 1): "Mountains", (0, 0): "Hills", (1, -1): "Grassland",
           (2, -2): "Forest", (3, -1): "Plains", (4, 0): "Desert", (5, 1): "Desert", (6, 2): "Grassland"}
    cells, tiles = wall_scene(run, gates={(1, -1)}, roads=((0, -2), (-1, -3), (2, 0), (3, 1)))
    city_run = {(-4, 2): "Grassland", (-3, 1): "Grassland", (-2, 0): "Grassland", (-1, -1): "Hills",
                (0, -2): "Hills", (1, -3): "Grassland", (2, -2): "Grassland", (3, -1): "Grassland",
                (4, 0): "Grassland"}
    city_cells, city_tiles = wall_scene(city_run)
    shots = tuple({**shot, "pieces": "half"} for shot in
                  ({"zoom": 128, "hour": 12}, {"zoom": 256, "hour": 12}, {"zoom": 256, "hour": 22}))
    return {
        "great-wall": {"title": "Great Wall across terrain", "snap": False, "cells": cells, "tiles": tiles,
                       "shots": shots, "crop": (8.0, 4.5)},
        "great-wall-city": {"title": "Great Wall beside a city", "snap": False, "cells": city_cells,
                            "tiles": city_tiles, "cities": [(0, 2, 4, 1, 1, 0, 0)], "shots": shots,
                            "crop": (6.5, 3.75)},
    }


# --- PCX-layout sheets -------------------------------------------------------
# Catalogs that complete the C3X sprite sheets cell for cell: one per culture
# file where C3X varies a district by culture. Rows are Civ III eras, columns
# the C3X building count, exactly as in the PCX files.
CULTURE_FILES = (("AMER", "american", 0), ("EURO", "european", 1), ("ROMAN", "roman", 2),
                 ("MIDEAST", "middle_eastern", 3), ("ASIAN", "asian", 4))
# (temple set token, temple, worship set token, worship hero). The marble tholos
# (Civ VI's Shrine) is the Roman temple only; the Asian temple is a Japanese
# temple compound from Civ VI's East Asian city buildings.
HOLY_BY_CULTURE = {
    "AMER": ("TEMPLE", "TEMPLE", "MEETING", "MEETING_HOUSE"),
    "EURO": ("TEMPLE", "TEMPLE", "CATHEDRAL", "CATHEDRAL"),
    "ROMAN": ("TEMPLE", "SHRINE", "SYNAGOGUE", "SYNAGOGUE"),
    "MIDEAST": ("TEMPLE", "TEMPLE", "MOSQUE", "MOSQUE"),
    "ASIAN": ("TEMPLE", "JAPANESE", "PAGODA", "PAGODA"),
}
JAPANESE_TEMPLE = "DIS_CTY_RJ_Block_WR_02"
POND = {"pond": .12, "pond_offset": .22, "trees": 14}
# The pond stays when temples are built, moved toward the front so they keep the back.
HOLY = {"density": .3, "trees": 10, "props": 4, "pond": .12, "pond_offset": .22, "hero_area": .14}


def holy_garden(code: str, era: str, pond: bool = False) -> list:
    """Temple grounds from Civ VI's City Park, used sparingly and scaled to the
    district: hedge beds and a tree planter, braziers (lamps from the
    Industrial era), and for the classical cultures a stone arch at the front
    and a pair of pillars; benches by a pond. Props: placed beside the
    buildings and never paved."""
    classical = era in ("Ancient", "Medieval")
    piece = lambda name, scale, **options: part(component(name), role="prop", scale=scale, **options)
    parts = [piece("IMP_City_Park_BushBase", .55), piece("IMP_City_Park_BushBase", .55),
             piece("IMP_City_Park_TreeBase", .65)]
    parts += [piece("IMP_City_Park_Fire" if classical else "IMP_City_Park_Lamp", 1.0) for _ in range(2)]
    if pond:
        parts += [piece("IMP_City_Park_StoneBench" if classical else "IMP_City_Park_WoodBench", 1.0) for _ in range(2)]
    elif code in ("EURO", "ROMAN", "MIDEAST"):
        parts += [piece("IMP_City_Park_Arch", .8, near="front"), piece("IMP_City_Park_Pillar", 1.0),
                  piece("IMP_City_Park_PillarB", 1.0)]
    return parts


def holy(code: str, era: str, col: int) -> list:
    """Holy Site parts, as C3X counts them: column 0 a pond ringed by trees
    (no buildings), 1 the culture's temple, 2 temple + worship building."""
    if col == 0:
        return trees() + holy_garden(code, era, pond=True)
    temple_set, temple, worship_set, worship = HOLY_BY_CULTURE[code]
    parts = [part(base("DISTRICT_HOLY_SITE", f"SHRINE, {temple_set}" + (f", {worship_set}" if col == 2 else ""), era))]
    parts.append(part(city(JAPANESE_TEMPLE)) if temple == "JAPANESE" else
                 part(hero("DISTRICT_HOLY_SITE", B + temple, era)))
    if col == 2:
        parts.append(part(hero("DISTRICT_HOLY_SITE", B + worship, era)))
    if code == "ASIAN":
        parts += houses("ASIAN", "Medieval", 4)          # Japanese-style halls round the temple
    return parts + holy_garden(code, era) + trees(("Tree_D_Lg", "Tree_D_Sm", "Tree_B_Lg") if code == "ASIAN" else TREES)


def holy_cell(name: str, label: str, row: int, col: int, code: str, era: str, **options) -> dict:
    # The pond keeps its size and place in every column and era.
    extra = {"draft": "force", "layout": POND} if col == 0 else {"layout": HOLY}
    extra["layout_key"] = f"{name.split()[0]}{col}"
    return cell(name, label, row, col, holy(code, era, col), era=era, **{**extra, **options})


def commercial(code: str, era: str, col: int) -> list:
    """Commercial Hub parts: Civ VI's market buildings among the city pack's
    houses for the culture and era, plus row houses or glass towers late."""
    bank = ("BAZAAR", "GRAND_BAZAAR") if code == "MIDEAST" else ("BANK", "BANK")
    stages = (("EMPTY", ()), ("MARKET", ("MARKET",)), (f"MARKET, {bank[0]}", ("MARKET", bank[1])),
              (f"MARKET, {bank[0]}, STOCK_EXCHANGE", ("MARKET", bank[1], "STOCK_EXCHANGE")))
    buildings, heroes = stages[col]
    return kit("DISTRICT_COMMERCIAL_HUB", buildings, era, heroes,
               extra=era_dressing("DISTRICT_COMMERCIAL_HUB", era, col) + houses(code, era, 5 + col, col))


# A hub grows as its buildings come: a wider footprint and denser blocks per column.
HUB_GROWTH = tuple({"density": density, "limit": limit, "props": 4}
                   for density, limit in ((.6, .33), (.66, .38), (.72, .43), (.78, .48)))


# Four Neighborhood looks per era (C3X picks one by tile hash): reserved squares
# and streets stay paved and open; each variant draws its own house selection.
NEIGHBORHOODS = (("dense block", {"density": .74, "trees": 2}),
                 ("round a square", {"density": .62, "trees": 4, "reserve": [(-.06, -.06, .06, .06)]}),
                 ("along a street", {"density": .62, "trees": 6, "reserve": [(-.5, -.025, .5, .025)]}),
                 ("garden houses", {"density": .4, "trees": 14}))


RESIDENTIAL = ("DIS_GEN_IND_Filler_RES_BldA", "DIS_GEN_IND_Filler_RES_BldC", "DIS_GEN_IND_Filler_RES_BldD",
               "DIS_GEN_IND_Filler_RES_BldG", "DIS_GEN_IND_Filler_RES_BldH")
SUBURBAN = ("DIS_NBH_HouseA", "DIS_NBH_HouseB", "DIS_NBH_HouseC", "DIS_NBH_HouseD", "DIS_NBH_HouseE",
            "DIS_NBH_HouseF", "DIS_NBH_HouseG")


# Civ VI recolours its houses' masked panels (their roof planks) per building;
# each house type takes a roof colour through that mask and a pastel wall
# colour on its white weatherboard: (roof, wall).
HOUSE_TINTS = (((.58, .64, .74), (1.0, .90, .74)), ((.56, .66, .54), (1.0, .90, .62)),
               ((.84, .56, .46), (.82, .90, .76)), ((.50, .50, .53), (.78, .86, .96)),
               ((.64, .52, .42), (1.0, .82, .70)), ((.52, .66, .66), (.84, .84, .82)),
               ((.74, .46, .42), (.96, .80, .78)))


def neighborhood(code: str, era: str, variant: int) -> list:
    """Houses, not towers: the culture's low city houses, joined by Civ VI's
    residential fillers in the Industrial era and its suburban Neighborhood
    houses in the Modern era, each house type in its own wall colour."""
    parts = houses(code, era, 6, 10 + variant, max_top=.1)
    late = {"Industrial": RESIDENTIAL, "Modern": SUBURBAN}.get(era, ())
    parts += [part(component(name), role="filler", tint=HOUSE_TINTS[index % len(HOUSE_TINTS)])
              for index, name in enumerate(late)]
    return parts + trees()


def sheets() -> dict:
    from Renderer.lab.studies.districts import kits
    c = {}
    for code, _, _ in CULTURE_FILES:
        labels = ("no buildings (pond and trees)", "Temple", "Temple + Cathedral")
        c[f"holy-{code.lower()}"] = {"title": f"Holy Site {code}", "draft": True, "cells": [
            holy_cell(f"H{code[:2]} {AB[era]}{col}", f"{era} · {label}", row, col, code, era)
            for row, era in enumerate(ERAS) for col, label in enumerate(labels)]}
        c[f"com-{code.lower()}"] = {"title": f"Commercial Hub {code}", "draft": True, "cells": [
            cell(f"C{code[:2]} {AB[era]}{col}", f"{era} · {label}", row, col, commercial(code, era, col), era=era,
                 layout=HUB_GROWTH[col], layout_key=f"C{code[:2]}{col}")
            for row, era in enumerate(ERAS)
            for col, label in enumerate(("no buildings", "Marketplace", "+ Bank", "+ Stock Exchange"))]}
        c[f"nbh-{code.lower()}"] = {"title": f"Neighborhood {code}", "draft": "force", "cells": [
            cell(f"N{code[:2]} {AB[era]}{col}", f"{era} · variant {col + 1}: {label}", row, col,
                 neighborhood(code, era, col), era=era, layout=layout, layout_key=f"N{code[:2]}{col}")
            for row, era in enumerate(ERAS) for col, (label, layout) in enumerate(NEIGHBORHOODS)]}
    c["ent-amer"] = {"title": "Entertainment Complex AMER", "draft": True, "layout": FAIR, "cells": [
        cell(f"EA {AB[era]}{col}", f"{era} · {label}", row, col,
             kit("DISTRICT_ENTERTAINMENT_COMPLEX", sets, era,
                 era_heroes("DISTRICT_ENTERTAINMENT_COMPLEX", era, heroes),
                 extra=ferris_wheel("DISTRICT_ENTERTAINMENT_COMPLEX", era)), era=era, layout_key=f"EA{col}")
        for row, era in enumerate(ERAS)
        for col, (label, sets, heroes) in enumerate((("no buildings", "EMPTY", ()),
                                                     ("Colosseum (Stadium in Modern)", "ARENA", ("COLOSSEUM",))))]}
    c["ent-roman"] = {"title": "Entertainment Complex ROMAN", "draft": True, "layout": FAIR, "cells": [
        cell(f"ER {AB[era]}{col}", f"{era} · {label}", row, col,
             kit("DISTRICT_HIPPODROME", sets, era, era_heroes("DISTRICT_HIPPODROME", era, heroes),
                 extra=ferris_wheel("DISTRICT_HIPPODROME", era)), era=era, layout_key=f"ER{col}")
        for row, era in enumerate(ERAS)
        for col, (label, sets, heroes) in enumerate((("no buildings", "Empty", ()),
                                                     ("Colosseum on the Hippodrome kit (Stadium in Modern)", "Arena", ("COLOSSEUM",))))]}
    # Great Wall pieces as C3X's sheet draws them: the pillar, each arm, and joined tiles.
    masks = (("alone", ()), ("NE arm", ("NE",)), ("SE arm", ("SE",)), ("SW arm", ("SW",)), ("NW arm", ("NW",)),
             ("NE–SW run", ("NE", "SW")), ("NW–SE run", ("SE", "NW")), ("corner NE–SE", ("NE", "SE")),
             ("corner SW–NW", ("SW", "NW")), ("crossing", ("NE", "SE", "SW", "NW")))
    c["great-wall-pieces"] = {"title": "Great Wall tile pieces", "snap": False, "cells": [
        cell(f"GWP {index}", label, index // 5, index % 5, wall_tile(mask), facing="tile")
        for index, (label, mask) in enumerate(masks)]}
    return c
