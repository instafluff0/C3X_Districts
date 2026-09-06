"""Shared scene sidecar validation and planar clearance predicates."""
import math

def finite_vector(v,n):
    return isinstance(v,list) and len(v)==n and all(isinstance(x,(float,int)) and math.isfinite(x) for x in v)

def point_segment(p,a,b):
    dx=b[0]-a[0];dy=b[1]-a[1];n=dx*dx+dy*dy
    t=max(0,min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n)) if n else 0
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)

def inside(p,poly):
    odd=False
    for a,b in zip(poly,poly[1:]+poly[:1]):
        if (a[1]>p[1])!=(b[1]>p[1]) and p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]:odd=not odd
    return odd

def intersects(a,b,margin=0):
    """Closed footprint/envelope intersection, including declared clearance."""
    if margin<0 or not math.isfinite(margin):raise ValueError('invalid clearance margin')
    if any(inside(p,b) for p in a) or any(inside(p,a) for p in b):return True
    for p,q in zip(a,a[1:]+a[:1]):
        for r,s in zip(b,b[1:]+b[:1]):
            cross=lambda x,y,z:(y[0]-x[0])*(z[1]-x[1])-(y[1]-x[1])*(z[0]-x[0])
            c1,c2,c3,c4=cross(p,q,r),cross(p,q,s),cross(r,s,p),cross(r,s,q)
            if c1*c2<0 and c3*c4<0:return True
            if min(point_segment(p,r,s),point_segment(q,r,s),point_segment(r,p,q),point_segment(s,p,q))<=margin+1e-9:return True
    return False

def validate(data):
    if not data.get('coordinate_space') or not data.get('terrain_sha256') or not data.get('region_id'):
        raise ValueError('scene exchange requires coordinate and source identity')
    schema=data.get('schema')
    if schema=='c3x.lab_v2.corridors.v1':
        if not finite_vector(data.get('wrap_period'),2) or min(data['wrap_period'])<0:raise ValueError('invalid wrap period')
        for e in data['envelopes']:
            if not e.get('id') or e.get('kind') not in ('road','rail','river','bank','bridge','city','building','wall') or not e.get('source_geometry_sha256'):
                raise ValueError('corridor missing rendered geometry provenance')
            if len(e['polygon'])<3 or not all(finite_vector(p,2) for p in e['polygon']):raise ValueError('invalid corridor polygon')
            if not finite_vector(e['height_range'],2) or e['height_range'][0]>e['height_range'][1] or not math.isfinite(e['clearance']) or e['clearance']<0:raise ValueError('invalid corridor bounds')
    elif schema=='c3x.lab_v2.world_scene.v1':
        mesh_ids=set();material_ids=set();instance_ids=set()
        for m in data['meshes']:
            if m['id'] in mesh_ids:raise ValueError('duplicate mesh identity')
            mesh_ids.add(m['id'])
            if len(m['positions'])!=len(m['normals']) or len(m['positions'])!=len(m['uv']) or len(m['indices'])%3:raise ValueError('world mesh layout mismatch')
            for key,n in [('positions',3),('normals',3),('uv',2)]:
                if not all(finite_vector(v,n) for v in m[key]):raise ValueError('nonfinite mesh attributes')
            if any(type(i)!=int or i<0 or i>=len(m['positions']) for i in m['indices']):raise ValueError('world mesh index outside vertices')
        for m in data['materials']:
            if m['id'] in material_ids or m['alpha_mode'] not in ('opaque','cutout','translucent'):raise ValueError('invalid material identity/alpha')
            material_ids.add(m['id'])
            if m['alpha_mode']=='cutout' and m['caster'] and not m.get('alpha_texture'):raise ValueError('cutout caster requires source alpha')
        for i in data['instances']:
            if i['id'] in instance_ids or i['mesh'] not in mesh_ids or i['material'] not in material_ids:raise ValueError('instance identity/reference mismatch')
            instance_ids.add(i['id'])
            if not math.isfinite(i['scale']) or i['scale']<=0 or not finite_vector(i['translation'],3) or not finite_vector(i['rotation'],9):raise ValueError('invalid uniform instance transform')
            r=i['rotation']
            for a in range(3):
                for b in range(3):
                    if abs(sum(r[a*3+k]*r[b*3+k] for k in range(3))-(a==b))>1e-4:raise ValueError('instance rotation contains source deformation')
            det=r[0]*(r[4]*r[8]-r[5]*r[7])-r[1]*(r[3]*r[8]-r[5]*r[6])+r[2]*(r[3]*r[7]-r[4]*r[6])
            if abs(det-1)>1e-4:raise ValueError('instance transform reflects source geometry')
    else:raise ValueError('unsupported scene exchange schema')
    return data
