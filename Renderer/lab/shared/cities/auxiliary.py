"""Attach fingerprinted, normalized auxiliary UVs without changing city geometry."""
from .fingerprint import geometry_digest
import math


def restore(mesh, records):
    record=records.get(geometry_digest(mesh))
    if record is None:return mesh
    vertices=[dict(v) for v in mesh['vertices']]
    for channel in ('uv1','uv2'):
        values=record[channel]
        if len(values)!=len(vertices) or any(len(v)!=2 or not all(math.isfinite(x) for x in v) for v in values):
            raise ValueError('Invalid normalized auxiliary UVs: '+mesh['asset_id'])
        for vertex,uv in zip(vertices,values):
            if channel in vertex and vertex[channel]!=list(uv):
                raise ValueError('Auxiliary recovery conflicts with an existing coordinate')
            vertex[channel]=list(uv)
    return {**mesh,'vertices':vertices}
