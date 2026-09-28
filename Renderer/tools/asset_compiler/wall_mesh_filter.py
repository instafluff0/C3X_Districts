"""Offline mesh edits for source-backed wall auditions."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt


def remove_uv_island_components(mesh: dict, uv_rect: tuple[float, float, float, float]) -> tuple[dict, int]:
    """Remove complete disconnected triangle islands contained in one UV rectangle.

    Positions are welded only to identify islands. Surviving source vertices,
    normals and UVs are copied unchanged; topology is compacted afterward.
    """
    vertices = mesh["vertices"]
    indices = mesh["topology"]["indices"]
    if mesh["topology"]["primitive"] != "triangles" or len(indices) % 3:
        raise ValueError("Wall mesh requires triangle topology")
    parent = list(range(len(vertices)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    by_position: dict[tuple[float, float, float], int] = {}
    for index, vertex in enumerate(vertices):
        position = tuple(round(value, 5) for value in vertex["position"])
        if position in by_position:
            union(index, by_position[position])
        else:
            by_position[position] = index
    for start in range(0, len(indices), 3):
        union(indices[start], indices[start + 1])
        union(indices[start], indices[start + 2])

    components: dict[int, list[int]] = defaultdict(list)
    for start in range(0, len(indices), 3):
        components[find(indices[start])].append(start)
    u0, u1, v0, v1 = uv_rect
    removed = set()
    for starts in components.values():
        used = {indices[start + offset] for start in starts for offset in range(3)}
        if all(u0 <= vertices[index]["uv0"][0] <= u1 and
               v0 <= vertices[index]["uv0"][1] <= v1 for index in used):
            removed.update(starts)

    surviving = [index for start in range(0, len(indices), 3)
                 if start not in removed for index in indices[start:start + 3]]
    remap = {old: new for new, old in enumerate(sorted(set(surviving)))}
    result = {
        **mesh,
        "vertices": [vertices[old] for old in sorted(remap)],
        "topology": {**mesh["topology"], "indices": [remap[index] for index in surviving]},
    }
    return result, len(removed)


def trim_skirt_and_ground(mesh: dict, floor_z: float) -> tuple[dict, dict[str, int]]:
    """Clip the lower skirt at a horizontal seam, then seat the cut on ground.

    This preserves the original horizontal scale and the texture's UV density.
    Only triangles crossing the seam receive interpolated edge vertices.
    """
    source = mesh["vertices"]
    indices = mesh["topology"]["indices"]
    if mesh["topology"]["primitive"] != "triangles" or len(indices) % 3:
        raise ValueError("Wall mesh requires triangle topology")
    if mesh.get("skin") is not None:
        raise ValueError("Skirt trim requires rigid wall geometry")
    vertices = [
        {**vertex, "position": [vertex["position"][0], vertex["position"][1],
                                 vertex["position"][2] - floor_z]}
        for vertex in source
    ]
    edge_vertices: dict[tuple[int, int], int] = {}

    def intersection(left: int, right: int) -> int:
        key = (min(left, right), max(left, right))
        if key in edge_vertices:
            return edge_vertices[key]
        a, b = source[left], source[right]
        az, bz = a["position"][2], b["position"][2]
        t = (floor_z - az) / (bz - az)
        normal = [a["normal"][axis] * (1 - t) + b["normal"][axis] * t
                  for axis in range(3)]
        magnitude = sqrt(sum(value * value for value in normal))
        if magnitude:
            normal = [value / magnitude for value in normal]
        vertex = {
            "position": [a["position"][axis] * (1 - t) + b["position"][axis] * t
                         for axis in (0, 1)] + [0.0],
            "normal": normal,
            "uv0": [a["uv0"][axis] * (1 - t) + b["uv0"][axis] * t
                    for axis in (0, 1)],
        }
        for channel in ('uv1', 'uv2'):
            if channel in a and channel in b:
                vertex[channel] = [a[channel][axis] * (1 - t) + b[channel][axis] * t
                                   for axis in (0, 1)]
        edge_vertices[key] = len(vertices)
        vertices.append(vertex)
        return edge_vertices[key]

    surviving = []
    removed = split = 0
    for start in range(0, len(indices), 3):
        triangle = indices[start:start + 3]
        polygon = []
        for left, right in zip(triangle, triangle[1:] + triangle[:1]):
            left_inside = source[left]["position"][2] >= floor_z
            right_inside = source[right]["position"][2] >= floor_z
            if left_inside and right_inside:
                polygon.append(right)
            elif left_inside and not right_inside:
                polygon.append(intersection(left, right))
            elif not left_inside and right_inside:
                polygon.extend((intersection(left, right), right))
        if len(polygon) < 3:
            removed += 1
            continue
        if len(polygon) == 4:
            split += 1
        for index in range(1, len(polygon) - 1):
            surviving.extend((polygon[0], polygon[index], polygon[index + 1]))
    used = sorted(set(surviving))
    remap = {old: new for new, old in enumerate(used)}
    result = {
        **mesh,
        "vertices": [vertices[old] for old in used],
        "topology": {**mesh["topology"], "indices": [remap[index] for index in surviving]},
    }
    if any(vertex["position"][2] < -1e-7 for vertex in result["vertices"]):
        raise ValueError("Trimmed wall still extends below ground")
    return result, {"discarded_triangles": removed, "split_triangles": split}
