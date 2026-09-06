"""Source-independent route graph geometry; coordinates are pixel-defined world units.

The caller owns surface sampling and hydrology anchors. This module never edits
terrain, infers river topology, or infers gameplay connections from proximity.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

DIRS = ((0,-2),(1,-1),(2,0),(1,1),(0,2),(-1,1),(-2,0),(-1,-1))

def add(a,b): return tuple(x+y for x,y in zip(a,b))
def mul(a,s): return tuple(x*s for x in a)
def sub(a,b): return add(a,mul(b,-1))
def length(a): return math.hypot(*a)
def unit(a):
    n=length(a)
    if n<1e-9: raise ValueError('zero direction')
    return mul(a,1/n)
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def lerp(a,b,t): return add(mul(a,1-t),mul(b,t))

@dataclass(frozen=True)
class Node:
    id: int
    raw: tuple
    xy: tuple

@dataclass(frozen=True)
class Crossing:
    id: int
    hydrology_edge: int
    xy: tuple
    tangent: tuple  # Route travel direction, perpendicular to water edge.
    width: float
    deck_z: float

@dataclass(frozen=True)
class Edge:
    id: int
    a: int
    b: int
    rail: bool=False
    stage: int=1
    pillaged: bool=False
    wrap: tuple=(0,0) # Explicit translated image of b in world pixels.
    crossing: Crossing|None=None
    additional_crossings: tuple=()

def fit_crossing_grade(crossing,surface,half_width=6,bank_clearance=1.5):
    """Q5 owns deck grade; keep hydrology identity, XY and tangent unchanged."""
    from dataclasses import replace
    t=unit(crossing.tangent);n=(-t[1],t[0]);span=crossing.width/2+6
    samples=[surface(*add(crossing.xy,add(mul(t,-span+2*span*i/32),mul(n,side*half_width)))) for i in range(33) for side in (-1,0,1)]
    return replace(crossing,deck_z=max(crossing.deck_z,max(samples)+bank_clearance))

class Graph:
    def __init__(self,nodes,edges,wrap_raw=(0,0)):
        self.nodes={n.id:n for n in nodes}; self.edges=sorted(edges,key=lambda e:e.id)
        if len(self.nodes)!=len(nodes): raise ValueError('duplicate node id')
        self.incident={n.id:[] for n in nodes}; ids=set(); links=set()
        for e in self.edges:
            if e.id in ids: raise ValueError('duplicate edge id')
            ids.add(e.id)
            if e.a not in self.nodes or e.b not in self.nodes or e.a==e.b: raise ValueError('invalid endpoint')
            key=tuple(sorted((e.a,e.b)))
            if key in links: raise ValueError('duplicate reciprocal edge')
            links.add(key)
            if e.stage not in range(4): raise ValueError('invalid route stage')
            a,b=self.nodes[e.a],self.nodes[e.b]
            delta=sub(b.raw,a.raw)
            if e.wrap!=(0,0):
                if not any(wrap_raw): raise ValueError('undeclared wrap')
                delta=tuple(d+round(w/64) for d,w in zip(delta,e.wrap))
                if any(w and abs(round(w/64))!=abs(r) for w,r in zip(e.wrap,wrap_raw)): raise ValueError('wrong wrap extent')
            if delta not in DIRS: raise ValueError('non-neighbor route')
            for c in self.crossings(e):
                ab=sub(add(b.xy,e.wrap),a.xy)
                if c.width<=0 or abs(dot(unit(ab),unit(c.tangent)))<.7: raise ValueError('invalid crossing direction')
                t=dot(sub(c.xy,a.xy),ab)/dot(ab,ab)
                if not .15<t<.85 or length(sub(c.xy,add(a.xy,mul(ab,t))))>1e-5: raise ValueError('crossing outside authoritative edge')
                if c.width+12>length(ab)*.7: raise ValueError('crossing exceeds edge')
            self.incident[e.a].append(e);self.incident[e.b].append(e)

    def crossings(self,e):
        a=self.nodes[e.a].xy;b=add(self.nodes[e.b].xy,e.wrap);t=unit(sub(b,a))
        cs=sorted(([e.crossing] if e.crossing else [])+list(e.additional_crossings),key=lambda c:dot(sub(c.xy,a),t))
        previous=-float('inf')
        for c in cs:
            center=dot(sub(c.xy,a),t);span=c.width/2+6
            if center-span<=previous:raise ValueError('overlapping bridge spans need one hydrology crossing envelope')
            previous=center+span
        return cs

    def vector(self,e,node):
        a,b=self.nodes[e.a].xy,add(self.nodes[e.b].xy,e.wrap)
        return sub(b,a) if node==e.a else sub(a,b)

    def tangent(self,e,node):
        own=unit(self.vector(e,node))
        others=[unit(self.vector(o,node)) for o in self.incident[node] if o.id!=e.id]
        if not others: return own
        opposite=min(others,key=lambda d:dot(own,d))
        if len(others)>1 and dot(own,opposite)>-.05: return own
        return unit(sub(own,opposite))

    def centerline(self,e,step=2.0,curved=True):
        a=self.nodes[e.a].xy;b=add(self.nodes[e.b].xy,e.wrap); d=length(sub(b,a))
        ta=self.tangent(e,e.a) if curved else unit(sub(b,a))
        tb=self.tangent(e,e.b) if curved else unit(sub(a,b))
        segments=[]
        cs=self.crossings(e)
        if cs:
            previous=a;incoming=ta
            for c in cs:
                t=unit(c.tangent)
                if dot(t,sub(b,a))<0:t=mul(t,-1)
                span=c.width/2+6;l=sub(c.xy,mul(t,span));r=add(c.xy,mul(t,span))
                segments.extend([(previous,incoming,l,mul(t,-1),False),(l,t,r,mul(t,-1),True)])
                previous=r;incoming=t
            segments.append((previous,incoming,b,tb,False))
        else: segments=[(a,ta,b,tb,False)]
        out=[]
        for p,t,q,u,bridge in segments:
            n=max(4,math.ceil(length(sub(q,p))/step)); arm=length(sub(q,p))*.30
            c1=add(p,mul(t,arm));c2=add(q,mul(u,arm))
            for i in range(n+1):
                if out and i==0:continue
                v=i/n
                xy=add(add(mul(p,(1-v)**3),mul(c1,3*v*(1-v)**2)),add(mul(c2,3*v*v*(1-v)),mul(q,v**3)))
                if bridge: xy=lerp(p,q,v)
                out.append((xy,bridge))
        out[0]=(a,False);out[-1]=(b,False)
        return out

class Mesh:
    def __init__(self,surface): self.surface=surface;self.vertices=[];self.route_samples=[]
    def tri(self,a,b,c,color):
        ab=sub(b,a);ac=sub(c,a)
        n=(ab[1]*ac[2]-ab[2]*ac[1],ab[2]*ac[0]-ab[0]*ac[2],ab[0]*ac[1]-ab[1]*ac[0])
        ln=math.sqrt(dot(n,n));n=mul(n,1/ln) if ln>1e-9 else (0,0,1)
        if n[2]<0:n=mul(n,-1)
        for p in (a,b,c):self.vertices.append((*p,*n,*color))
    def quad(self,a,b,c,d,color):self.tri(a,b,c,color);self.tri(a,c,d,color)
    def disk(self,p,r,color,z=None):
        def pt(xy):return (*xy,self.surface(*xy)+.20 if z is None else z)
        for k in range(24):
            x=add(p,(r*math.cos(k*math.tau/24),r*math.sin(k*math.tau/24)))
            y=add(p,(r*math.cos((k+1)*math.tau/24),r*math.sin((k+1)*math.tau/24)))
            self.tri(pt(p),pt(x),pt(y),color)
    def strip(self,points,width,color,offset=0,lift=.25,z_override=None):
        sides=[]
        for i,xy in enumerate(points):
            tangent=unit(sub(points[min(i+1,len(points)-1)],points[max(0,i-1)])); normal=(-tangent[1],tangent[0])
            pair=[]
            for side in (-1,1):
                p=add(xy,mul(normal,offset+side*width/2));z=self.surface(*p) if z_override is None else z_override(p,i)
                pair.append((*p,z+lift))
            sides.append(pair)
        for i in range(len(points)-1):
            p=points[i];variation=1+.035*math.sin(p[0]*.31+p[1]*.77)+.025*math.sin(p[0]*1.11-p[1]*.53)
            self.quad(sides[i][0],sides[i][1],sides[i+1][1],sides[i+1][0],tuple(c*variation for c in color))

ROAD_COLORS=((.44,.31,.16),(.51,.39,.24),(.40,.38,.32),(.22,.24,.24))
def build_routes(graph,mesh,curved=True,rail_width=5.4):
    """Tessellate terrain-draped ribbons with exact nodes and supplied bridges."""
    paths=[]
    for e in graph.edges:
        path=graph.centerline(e,curved=curved);pts=[p for p,_ in path]
        c=e.crossing
        def deck(p,i):
            z=mesh.surface(*p)
            if not c:return z
            axis=unit(c.tangent); along=abs(dot(sub(p,c.xy),axis));span=c.width/2+6
            if along<=span:return c.deck_z
            # Short continuous approach, bounded to the route edge.
            blend=max(0,1-(along-span)/14)
            return max(z,z+(c.deck_z-z)*blend)
        paths.append((e,pts,deck if c else None))
        road=ROAD_COLORS[e.stage]
        mesh.strip(pts,9.0,tuple(x*.73 for x in road),lift=.12,z_override=deck if c else None)
        mesh.strip(pts,6.2,road,lift=.24,z_override=deck if c else None)
        if not e.rail and e.stage<3:
            for offset in (-1.7,1.7):
                mesh.strip(pts,.8,tuple(x*.83 for x in road),offset,.28,deck if c else None)
        for p in pts:mesh.route_samples.append((e.id,*p,deck(p,0)+.24 if c else mesh.surface(*p)+.24,bool(c)))
    for node in graph.nodes.values():
        inc=graph.incident[node.id]
        if inc:mesh.disk(node.xy,3.5,ROAD_COLORS[min(inc,key=lambda e:e.id).stage],None)
    for e,pts,deck in paths:
        if e.rail:
            mesh.strip(pts,rail_width+1.4,(.25,.235,.20),lift=.38,z_override=deck)
            # Arc-length sleepers; suppress near multiway nodes for readable switches.
            dist=0;next_tie=3.5
            for i in range(1,len(pts)):
                a,b=pts[i-1],pts[i];seg=length(sub(b,a));t=unit(sub(b,a));n=(-t[1],t[0])
                while next_tie<dist+seg:
                    p=lerp(a,b,(next_tie-dist)/seg)
                    near_junction=(len(graph.incident[e.a])>2 and length(sub(p,pts[0]))<10) or (len(graph.incident[e.b])>2 and length(sub(p,pts[-1]))<10)
                    if not near_junction and not(e.pillaged and .35<length(sub(p,pts[0]))/max(length(sub(pts[-1],pts[0])),1)<.65):
                        ends=[add(p,mul(n,s*(rail_width/2+.7))) for s in (-1,1)]
                        mesh.strip(ends,1.35,(.28,.20,.12),lift=.49,z_override=(lambda q,j:deck(q,i)) if deck else None)
                    next_tie+=6
                dist+=seg
            for offset in (-rail_width/2,rail_width/2):
                if e.pillaged:
                    for low,high in ((0,.39),(.61,1)):
                        part=pts[int(low*(len(pts)-1)):int(high*(len(pts)-1))+1]
                        mesh.strip(part,.85,(.30,.32,.31),offset,.64,deck)
                else:mesh.strip(pts,.85,(.30,.32,.31),offset,.64,deck)
        if e.pillaged and not e.rail:
            for i in range(len(pts)//3,2*len(pts)//3,3):
                mesh.disk(pts[i],2.8,(.22,.19,.125), (deck(pts[i],i) if deck else mesh.surface(*pts[i]))+.7)
        if e.crossing:
            c=e.crossing;t=unit(c.tangent);span=c.width/2+6
            ends=[sub(c.xy,mul(t,span)),add(c.xy,mul(t,span))]
            # The rigid body is fitted to the supplied water width and deck plane.
            mesh.strip(ends,11,(.35,.30,.22),lift=-.12,z_override=lambda p,i:c.deck_z)
            for off in (-5.3,5.3):mesh.strip(ends,.8,(.47,.41,.30),off,2.6,lambda p,i:c.deck_z)
    return mesh
