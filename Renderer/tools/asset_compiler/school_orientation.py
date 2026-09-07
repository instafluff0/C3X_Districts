"""Offline, whole-body heading calibration for disconnected skinned schools.

Each body keeps its authored deformation and formation position. Duplicating
shared palettes per body avoids tearing vertices influenced by a neighbor's rig.
The DLL still consumes only ordinary generic skin palettes.
"""
from __future__ import annotations

import math
import struct

from Renderer.tools.asset_compiler import normalized_skin


def body_components(mesh: dict) -> list[list[int]]:
    vertices = mesh["vertices"]
    parents = list(range(len(vertices)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def join(a, b):
        parents[root(a)] = root(b)

    indices = mesh["topology"]["indices"]
    for i in range(0, len(indices), 3):
        join(indices[i], indices[i+1])
        join(indices[i], indices[i+2])
    # UV/normal seams split vertices without splitting the actual animal.
    positions = {}
    for i, vertex in enumerate(vertices):
        key = tuple(round(v, 7) for v in vertex["position"])
        if key in positions:
            join(i, positions[key])
        positions[key] = i
    components = {}
    for i in range(len(vertices)):
        components.setdefault(root(i), []).append(i)
    return list(components.values())


def align_school_payload(payload: bytes, mesh: dict, skeleton: dict,
                         heading_pairs: list[tuple[str, str]]) -> tuple[bytes, dict]:
    """Align every sampled head-minus-tail vector to +X (Civ III southeast).

    Heading names are supplied by the offline source adapter, never guessed by
    the runtime. All influences on a connected body receive the SAME rigid
    post-transform, including tiny cross-body weights in the source assets.
    """
    magic, version, count, index_count, bone_count, frames, duration = struct.unpack_from("<8s5If", payload)
    if (magic != b"C3XANM1\0" or version != 1 or count != len(mesh["vertices"]) or
            bone_count != len(skeleton["bones"]) or index_count != len(mesh["topology"]["indices"])):
        raise ValueError("school payload/source mismatch")
    palette_offset = 32 + count*64 + index_count*4
    if len(payload) != palette_offset + frames*bone_count*64:
        raise ValueError("school payload length mismatch")
    names = [b["name"] for b in skeleton["bones"]]
    pairs = [(names.index(head), names.index(tail)) for head, tail in heading_pairs]
    owners = {}
    for i, bone in enumerate(skeleton["bones"]):
        owners[i] = i if i in {head for head, _ in pairs} else owners.get(bone["parent"])
    worlds = normalized_skin.world_matrices(skeleton)
    components = body_components(mesh)
    if len(components) != len(pairs):
        raise ValueError("school body count does not match authored heading pairs")
    vertices = bytearray(payload[32:32+count*64])
    palettes = []
    records = []
    body_weights = []
    assigned = set()
    for component in components:
        weights = {}
        used = set()
        for i in component:
            vertex = mesh["vertices"][i]
            for joint, weight in zip(vertex["joints"], vertex["weights"]):
                if weight > 0:
                    used.add(joint)
                    owner = owners[joint]
                    weights[owner] = weights.get(owner, 0) + weight
        head = max(weights, key=weights.get)
        if head is None or head in assigned or weights[head] < .98*len(component):
            raise ValueError("ambiguous school body ownership")
        assigned.add(head)
        tail = next(tail for candidate, tail in pairs if candidate == head)
        dx, dy = (worlds[head][12+a]-worlds[tail][12+a] for a in range(2))
        if math.hypot(dx, dy) < 1e-6:
            raise ValueError("degenerate school heading")
        yaw = -math.atan2(dy, dx)
        pivot = [sum(mesh["vertices"][i]["position"][a] for i in component)/len(component) for a in range(3)]
        # Weighted homogeneous sums produce each posed body centroid without
        # reskinning every vertex for every frame during compilation.
        aggregates = {joint: [0., 0., 0., 0.] for joint in used}
        for i in component:
            vertex = mesh["vertices"][i]
            for joint, weight in zip(vertex["joints"], vertex["weights"]):
                if weight > 0:
                    for axis, value in enumerate([*vertex["position"], 1.]):
                        aggregates[joint][axis] += weight*value/len(component)
        body_weights.append(aggregates)
        mapping = {}
        for joint in sorted(used):
            mapping[joint] = len(palettes)
            palettes.append((joint, len(records)))
        for i in component:
            vertex = mesh["vertices"][i]
            remapped = [mapping[j] if w > 0 else 0 for j, w in zip(vertex["joints"], vertex["weights"])]
            struct.pack_into("<4I", vertices, i*64+32, *remapped)
        records.append({"head": names[head], "tail": names[tail], "vertices": component,
                        "pivot": pivot, "yaw": yaw, "source_forward": [dx, dy]})
    new_size = palette_offset + frames*len(palettes)*64
    if len(palettes) > 256 or new_size > 64*1024*1024:
        raise ValueError("aligned school exceeds DLL palette budget")
    result = bytearray(struct.pack("<8s5If", magic, version, count, index_count, len(palettes), frames, duration))
    result.extend(vertices)
    result.extend(payload[32+count*64:palette_offset])
    for frame in range(frames):
        originals = [struct.unpack_from("<16f", payload, palette_offset+(frame*bone_count+joint)*64)
                     for joint in range(bone_count)]
        transforms = []
        for record, aggregates in zip(records, body_weights):
            h, t = (names.index(record[key]) for key in ("head", "tail"))
            head = normalized_skin._multiply(worlds[h], originals[h])
            tail = normalized_skin._multiply(worlds[t], originals[t])
            dx, dy = head[12]-tail[12], head[13]-tail[13]
            if math.hypot(dx, dy) < 1e-6:
                raise ValueError("degenerate animated school heading")
            yaw = -math.atan2(dy, dx)
            c, s = math.cos(yaw), math.sin(yaw)
            px, py = (sum(sum(values[b]*originals[joint][b*4+a] for b in range(4))
                          for joint, values in aggregates.items()) for a in range(2))
            transforms.append((c, s, 0, 0, -s, c, 0, 0, 0, 0, 1, 0,
                               px-px*c+py*s, py-px*s-py*c, 0, 1))
        for joint, body in palettes:
            result.extend(struct.pack("<16f", *normalized_skin._multiply(originals[joint], transforms[body])))
    return bytes(result), {"policy": "whole_body_per_sample_centroid", "forward": "+X/SE",
                           "bodies": records, "palette_count": len(palettes)}
