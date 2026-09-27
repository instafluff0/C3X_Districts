"""Keep authored capital layouts as full as their ordinary city counterparts."""

from Renderer.lab.shared.cities.growth import overlaps
from Renderer.lab.studies.cities.build_layouts import inside_wall


TARGETS = ((-.23, .12), (.23, .12), (0, -.18), (0, .34),
           (-.20, -.12), (.20, -.12))
OFFSETS = [(dx * .025, dy * .025) for dx in range(-12, 13)
           for dy in range(-12, 13)]
OFFSETS.sort(key=lambda item: (item[0] ** 2 + item[1] ** 2,
                               abs(item[0]), abs(item[1])))


def fill(tiers, name, scale, counts, part, box):
    """Find distinct legal plots near a palace for each population tier."""
    for size, tier in enumerate(tiers):
        houses = list(tier["houses"])
        occupied = [box(tier["palace"])] + [box(item) for item in houses]
        for index in range(counts[size]):
            target_x, target_y = TARGETS[index]
            for dx, dy in OFFSETS:
                item = part(name, scale, round(target_x + dx, 3),
                            round(target_y + dy, 3))
                bounds = box(item)
                if not inside_wall(bounds, size, clearance=.015):
                    continue
                if size == 0 and any(abs(value) > .5 for value in bounds):
                    continue
                if any(overlaps(bounds, prior) for prior in occupied):
                    continue
                houses.append(item)
                occupied.append(bounds)
                break
            else:
                raise ValueError(f"No capital infill plot for tier {size}, slot {index}")
        tier["capital_houses"] = houses
