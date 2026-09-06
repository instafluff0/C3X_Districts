"""Conservative transformed source bounds versus swept world-space corridors.
No tree deformation, draw-order workaround, or global placement reshuffle.
"""
import math

def dist_segment(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1];n=dx*dx+dy*dy
    t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n)) if n else 0
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)
def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
def intersects(a,b,c,d):
    return cross(a,b,c)*cross(a,b,d)<=0 and cross(c,d,a)*cross(c,d,b)<=0 and max(min(a[0],b[0]),min(c[0],d[0]))<=min(max(a[0],b[0]),max(c[0],d[0])) and max(min(a[1],b[1]),min(c[1],d[1]))<=min(max(a[1],b[1]),max(c[1],d[1]))
def source_footprint(bounds,x,y,scale,yaw):
    lo,hi=bounds['minimum'],bounds['maximum'];c,s=math.cos(yaw),math.sin(yaw)
    return [(x+scale*(a*c-b*s),y+scale*(a*s+b*c)) for a,b in [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])]]
def inside_convex(p,polygon):
    signs=[cross(a,b,p) for a,b in zip(polygon,polygon[1:]+polygon[:1])]
    return not (any(v>1e-10 for v in signs) and any(v< -1e-10 for v in signs))

def city_envelopes(witness,region_metadata):
    out=[]
    for region in witness['regions']:
        if region['origin_raw']!=region_metadata['region']['origin']:continue
        if region['source_sha256']!=region_metadata['source_sha256']:raise ValueError('Q7 city dataset mismatch')
        if region['coordinate_space']!='civ3_raw_delta_pixels_v1':raise ValueError('Q7 city coordinate drift')
        for component in region['polygons']:
            # Exact raw-delta-pixel -> Q4 column/row basis; the Q4 receiver has
            # a 1.5 tile origin offset to include neighboring real terrain.
            polygon=[((x+y)/128+1.5,(x-y)/128+1.5) for x,y in component['raw_delta_pixel_polygon']]
            out.append({'id':f"city:{component['slot']}",'shape':'polygon','points':polygon+[polygon[0]],'half_width':.045,'provider':'Q7 city footprint and overhang','asset':component['asset'],'raw_city_anchor':region['raw_city_anchor']})
    return out

def clear(footprint,envelopes,margin=.035):
    for e in envelopes:
        radius=e['half_width']+e.get('clearance',0)+margin
        if e.get('shape')=='polygon' and any(inside_convex(p,e['points'][:-1]) for p in footprint):return False
        for a,b in zip(e['points'],e['points'][1:]):
            inside=lambda p:all(cross(c,d,p)>=-1e-10 for c,d in zip(footprint,footprint[1:]+footprint[:1]))
            if inside(a) or inside(b):return False
            for c,d in zip(footprint,footprint[1:]+footprint[:1]):
                if intersects(a,b,c,d):return False
                if min(dist_segment(c,a,b),dist_segment(d,a,b),dist_segment(a,c,d),dist_segment(b,c,d))<=radius:return False
    return True


def q5_clear(footprint,envelope):
    if envelope is None:return True
    import importlib.util,sys
    from pathlib import Path
    directory=Path(__file__).resolve().parents[1]/'networks'
    # Consume the provider's polygon/capsule/wrap logic without copying it.
    key='_q4_readonly_q5_clearance'
    if key not in sys.modules:
        sys.path.insert(0,str(directory))
        try:
            spec=importlib.util.spec_from_file_location(key,directory/'clearance.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);sys.modules[key]=module
        finally:sys.path.pop(0)
    polygon=[(64*(x-y),64*(x+y)) for x,y in footprint]
    return not sys.modules[key].footprint_intersects(envelope,polygon)
