"""Deterministic, building-aware infill for the medieval Lab auditions."""

from Renderer.lab.shared.cities.growth import overlaps


TARGETS = {
    0: ((0, -.38), (-.30, .31), (.30, .31)),
    1: ((-.26, -.44), (.26, -.44), (-.48, .02), (.48, .02),
        (-.26, .43), (.26, .43), (0, -.47), (0, .48)),
    2: ((-.34, -.57), (.34, -.57), (-.60, -.20), (.60, -.20),
        (-.55, .30), (.55, .30), (-.25, .58), (.25, .58),
        (0, -.60), (0, .60), (-.45, .51), (.45, .51)),
}
OFFSETS = [(dx * .025, dy * .025) for dx in range(-14, 15)
           for dy in range(-14, 15)]
OFFSETS.sort(key=lambda point: (point[0] ** 2 + point[1] ** 2,
                                abs(point[0]), abs(point[1])))


def fill(houses, centerpieces, size, count, palette, make, box,
         radii=(.43, .58, .70)):
    """Add full-size source buildings at legal free plots near selected gaps."""
    result = list(houses)
    occupied = [box(item) for item in result + list(centerpieces)]
    radius = radii[size] - .015
    for slot in range(count):
        tx, ty = TARGETS[size][slot]
        # A different source building leads each slot, while alternatives let
        # narrow plots retain actual architecture instead of scaling it down.
        models = palette[slot % len(palette):] + palette[:slot % len(palette)]
        for dx, dy in OFFSETS:
            x, y = round(tx + dx, 3), round(ty + dy, 3)
            for model, scale in models:
                item = make(model, scale, x, y)
                bounds = box(item)
                if size == 0 and any(abs(value) > .5 for value in bounds):
                    continue
                if any((abs(px) / radius) ** 6 + (abs(py) / radius) ** 6 > 1
                       for px in (bounds[0], bounds[2])
                       for py in (bounds[1], bounds[3])):
                    continue
                if any(overlaps(bounds, previous) for previous in occupied):
                    continue
                result.append(item)
                occupied.append(bounds)
                break
            else:
                continue
            break
        else:
            raise ValueError(f"No medieval infill plot in tier {size}, slot {slot}")
    return result
