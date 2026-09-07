"""Clip tessellated ground triangles to a sampled signed land boundary."""
import math


def clip_polygon(polygon, distance):
    if not polygon:
        return []
    result=[]
    previous=polygon[-1];old_distance=distance(previous)
    for current in polygon:
        value=distance(current)
        if (old_distance<=0)!=(value<=0):
            t=old_distance/(old_distance-value)
            result.append([a+t*(b-a) for a,b in zip(previous,current)])
        if value<=0:result.append(current)
        previous=current;old_distance=value
    return result


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


def clip_ground_to_land_cells(vertices, shore_distances, cell_bases, world_axes=(9,10), boundary=-.02, excluded_cells=()):
    """Keep only dry tile cells, then the smoothed shore inside those cells.

    A water-classified tile may contain a protruding smoothed land edge. Tile
    classification and signed shore distance are separate restrictions, just
    as in the building foundation gate. Interpolate attributes at exact cell
    boundaries rather than inventing distances for categorical tile values.
    Optional excluded cells support vegetation clearance without changing the
    terrain's actual base classification or signed shore samples.
    """
    ax,ay=world_axes;result=[]
    for row in range(math.floor(min(v[ay] for v in vertices)),math.floor(max(v[ay] for v in vertices))+1):
        for col in range(math.floor(min(v[ax] for v in vertices)),math.floor(max(v[ax] for v in vertices))+1):
            polygon=[list(v)+[d] for v,d in zip(vertices,shore_distances)]
            for axis,edge,sign in ((ax,col,-1),(ax,col+1,1),(ay,row,-1),(ay,row+1,1)):
                polygon=clip_polygon(polygon,lambda v,axis=axis,edge=edge,sign=sign:sign*(v[axis]-edge))
            if len(polygon)<3:continue
            area=abs(sum(a[ax]*b[ay]-b[ax]*a[ay] for a,b in zip(polygon,polygon[1:]+polygon[:1])))
            if area<1e-12:continue
            if (col,row) not in cell_bases:raise ValueError('ground triangle entered an unqueried terrain cell')
            if cell_bases[col,row]>=11 or (col,row) in excluded_cells:continue
            polygon=clip_polygon(polygon,lambda v:v[-1]-boundary)
            result.extend(v[:-1] for i in range(1,len(polygon)-1) for v in (polygon[0],polygon[i],polygon[i+1]))
    return result
