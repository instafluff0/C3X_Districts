"""Q5 final-geometry clearance provider v1; no object placement or draw-order hacks.

Union of capsules (route ribbons and junctions) and rigid bridge polygons.
Consumers test transformed footprints/crowns, not just object origins. World
units match Q5's raw Civ III x/y * 64 plane, with Z upwards in screen pixels.
"""
from network import *

def point_segment(p,a,b):
    ab=sub(b,a);t=max(0,min(1,dot(sub(p,a),ab)/max(dot(ab,ab),1e-20)))
    return length(sub(p,add(a,mul(ab,t))))
def segments_intersect(a,b,c,d):
    def cross(a,b,c):u=sub(b,a);v=sub(c,a);return u[0]*v[1]-u[1]*v[0]
    ab,ac,cd,ce=cross(a,b,c),cross(a,b,d),cross(c,d,a),cross(c,d,b)
    if ab*ac<0 and cd*ce<0:return True
    return any(point_segment(p,x,y)<1e-8 for p,x,y in ((a,c,d),(b,c,d),(c,a,b),(d,a,b)))
def inside(p,poly):
    hit=False
    for a,b in zip(poly,poly[1:]+poly[:1]):
        if point_segment(p,a,b)<1e-8:return True
        if (a[1]>p[1])!=(b[1]>p[1]) and p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:hit=not hit
    return hit

def publish(graph,surface,bridge_evidence=(),margin=4,world_wrap=(0,0),halo_complete=False):
    entries=[]
    for e in graph.edges:
        points=[p for p,_ in graph.centerline(e)]
        heights=[surface(*p) for p in points]
        heights.extend(c.deck_z for c in graph.crossings(e))
        entries.append({'id':f'route:{e.id}','edge_id':e.id,'kind':'road_and_rail' if e.rail else 'road','shape':'capsule_chain','points':points,'occupied_radius':6.,'clearance_radius':6.+margin,'height_range':[min(heights),max(heights)+.65],'wrap_translation':e.wrap})
    for node in sorted(graph.nodes.values(),key=lambda n:n.id):
        if len(graph.incident[node.id])>1:
            z=surface(*node.xy)
            entries.append({'id':f'junction:{node.id}','kind':'junction','shape':'capsule_chain','points':[node.xy,node.xy],'occupied_radius':6.,'clearance_radius':6.+margin,'height_range':[z,z+.65]})
    for info in bridge_evidence:
        e=next(e for e in graph.edges if any(c.id==info['crossing_id'] for c in graph.crossings(e)));c=next(c for c in graph.crossings(e) if c.id==info['crossing_id']);t=unit(c.tangent);n=(-t[1],t[0]);lo,hi=info['source_bounds'];scale=info['uniform_scale']
        hx=(hi[0]-lo[0])*scale/2;hy=(hi[1]-lo[1])*scale/2
        poly=[add(c.xy,add(mul(t,x),mul(n,y))) for x,y in ((-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy))]
        entries.append({'id':f'bridge:{c.id}','kind':'bridge','shape':'polygon','points':poly,'occupied_radius':0.,'clearance_radius':margin,'height_range':[c.deck_z+lo[2]*scale,c.deck_z+hi[2]*scale],'source_asset':info['asset_id']})
    return {'schema':'c3x.q5.route_clearance.v1','coordinate_basis':'world_x=64*raw_x_delta; world_y=64*raw_y_delta; screen=(x,y/2-z)','units':'normal_zoom_pixels','margin':margin,'world_wrap':world_wrap,'halo_complete':halo_complete,'consumer_rule':'Union all envelopes; test full transformed crown/footprint. Height ranges are information, not permission for foliage to hide a corridor. Require halo_complete for cropped final placement.','entries':entries}

def footprint_intersects(envelope,polygon,clearance=True):
    """Conservative exact planar polygon vs swept segments, with periodic aliases."""
    if len(polygon)<3:raise ValueError('footprint needs at least three vertices')
    wx,wy=envelope['world_wrap']
    aliases=[(x,y) for x in ((-wx,0,wx) if wx else (0,)) for y in ((-wy,0,wy) if wy else (0,))]
    for shift in aliases:
        poly=[add(tuple(p),shift) for p in polygon];sides=list(zip(poly,poly[1:]+poly[:1]))
        for e in envelope['entries']:
            pts=e['points'];radius=e['clearance_radius' if clearance else 'occupied_radius']
            pairs=list(zip(pts,pts[1:]))
            if e['shape']=='polygon':
                pairs+= [(pts[-1],pts[0])]
                if inside(poly[0],pts):return True
            if inside(pts[0],poly):return True
            for a,b in pairs:
                for c,d in sides:
                    if segments_intersect(a,b,c,d) or min(point_segment(a,c,d),point_segment(b,c,d),point_segment(c,a,b),point_segment(d,a,b))<=radius+1e-8:return True
    return False
