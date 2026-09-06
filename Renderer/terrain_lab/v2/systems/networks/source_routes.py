"""Adapt normalized authored route atlases to Q5 centerline topology.

Generated ribbons are source_adaptation. Ground/city/bridge primitive witnesses
remain diagnostic_proxy, never a selected original-art fallback.
"""
from network import *

def source_strip(mesh,points,width,stage,pillaged,deck=None,ends=(False,False)):
    distance=0;sides=[]
    total=sum(length(sub(b,a)) for a,b in zip(points,points[1:]))
    for i,p in enumerate(points):
        if i:distance+=length(sub(p,points[i-1]))
        t=unit(sub(points[min(i+1,len(points)-1)],points[max(0,i-1)]));n=(-t[1],t[0]);pair=[]
        for side in (-1,1):
            q=add(p,mul(n,side*width/2));z=mesh.surface(*q) if deck is None else deck(q)
            if z<mesh.surface(*q)-1e-5:raise ValueError('bridge grade tunnels into the final surface; fit deck grade first')
            eps=.1;dx=(mesh.surface(q[0]+eps,q[1])-mesh.surface(q[0]-eps,q[1]))/(2*eps)
            dy=(mesh.surface(q[0],q[1]+eps)-mesh.surface(q[0],q[1]-eps))/(2*eps)
            normal=(-dx,-dy,1);normal=mul(normal,1/math.sqrt(dot(normal,normal)))
            fade=min(1,distance/6 if ends[0] else 1,(total-distance)/6 if ends[1] else 1)
            pair.append((*q,z+.35,*normal,fade,1,1,distance/96,side,stage,int(pillaged)))
        sides.append(pair)
    for a,b in zip(sides,sides[1:]):
        mesh.vertices.extend((a[0],a[1],b[1],a[0],b[1],b[0]))

def build_source_routes(graph,mesh,curved=True,rail_width=9):
    for e in graph.edges:
        pts=[p for p,_ in graph.centerline(e,curved=curved)];cs=graph.crossings(e);c=cs[0] if cs else None
        for crossing in cs:
            t=unit(crossing.tangent);n=(-t[1],t[0]);span=crossing.width/2+6
            for i in range(33):
                for side in (-1,1):
                    q=add(crossing.xy,add(mul(t,-span+2*span*i/32),mul(n,side*6)))
                    if mesh.surface(*q)>crossing.deck_z+1e-5:raise ValueError('bridge grade tunnels into final surface')
        def deck(p):
            z=mesh.surface(*p)
            value=z
            for crossing in cs:
                along=abs(dot(sub(p,crossing.xy),unit(crossing.tangent)));span=crossing.width/2+6
                if along<=span:value=max(value,crossing.deck_z)
                else:value=max(value,z+(crossing.deck_z-z)*max(0,1-(along-span)/14))
            return value
        source_strip(mesh,pts,12,e.stage,e.pillaged,deck if c else None,(len(graph.incident[e.a])==1,len(graph.incident[e.b])==1))
        if e.rail:source_strip(mesh,pts,rail_width,4,e.pillaged,(lambda p:deck(p)+.3),(len(graph.incident[e.a])==1,len(graph.incident[e.b])==1))
    return mesh
