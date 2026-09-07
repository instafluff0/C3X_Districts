"""Clip tessellated ground triangles to a sampled signed land boundary."""


def clip_ground_triangle(vertices, shore_distances, boundary=-.02):
    """Retain land (negative distance), interpolating every vertex attribute.

    Input triangles have already been projected onto terrain. The small
    tessellation cells make this a local linear shore approximation, not a
    replacement for the renderer's authoritative terrain/water geometry.
    """
    polygon = []
    previous, old_distance = vertices[-1], shore_distances[-1]
    for current, distance in zip(vertices, shore_distances):
        if (old_distance <= boundary) != (distance <= boundary):
            t = (boundary - old_distance) / (distance - old_distance)
            polygon.append([a + t * (b - a) for a, b in zip(previous, current)])
        if distance <= boundary:
            polygon.append(current)
        previous, old_distance = current, distance
    return [v for i in range(1, len(polygon) - 1)
            for v in (polygon[0], polygon[i], polygon[i + 1])]
