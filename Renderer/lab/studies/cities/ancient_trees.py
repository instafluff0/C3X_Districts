"""Place sparse existing farm trees around authored ancient building plots."""

from pathlib import Path

from Renderer.lab.shared.cities.assets import component
from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import footprint, inside_wall


TREE = "city/prop/source_farm_tree"
SITES = ((-.35, -.37), (.35, -.37), (-.39, .05), (.39, .05),
         (-.31, .42), (.31, .42), (0, -.45), (0, .46),
         (-.50, -.14), (.50, -.14), (-.51, .29), (.51, .29),
         (-.27, -.53), (.27, -.53), (-.29, .57), (.29, .57),
         (-.60, -.35), (.60, -.35), (-.62, .35), (.62, .35))


def plant(tiers, tree_pack: Path, counts=(2, 3, 5), sites=SITES,
          min_spacing=0.0):
    tree = component(TREE, tree_pack)
    for size, tier in enumerate(tiers):
        for capital in (False, True):
            buildings = tier.get("capital_houses", tier["houses"]) if capital else tier["houses"]
            buildings = buildings + [tier["palace"] if capital else tier["base_centerpiece"]]
            occupied = []
            for item in buildings:
                body = component(item["asset"], Path(item["pack"]))
                occupied.append(footprint({"low": body["lo"], "high": body["hi"]}, item))
            planted = []
            for x, y in sites:
                if any((x - previous["offset"][0]) ** 2 +
                       (y - previous["offset"][1]) ** 2 < min_spacing ** 2
                       for previous in planted):
                    continue
                item = {"asset": TREE, "pack": tree_pack.as_posix(), "scale": 2.35,
                        "rotation": 0.0, "offset": [x, y], "surface": False}
                box = footprint({"low": tree["lo"], "high": tree["hi"]}, item)
                if not inside_wall(box, size, clearance=.015):
                    continue
                if size == 0 and any(abs(value) > .5 for value in box):
                    continue
                if any(overlaps(box, prior) for prior in occupied):
                    continue
                planted.append(item)
                occupied.append(box)
                if len(planted) == counts[size]:
                    break
            tier["capital_decorations" if capital else "decorations"] = planted
