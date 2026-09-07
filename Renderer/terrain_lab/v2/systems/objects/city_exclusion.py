"""Generic convex ground polygons that city footprints must leave unobstructed."""
import math


def intersects(box, polygon):
    """Positive-area rectangle/convex-polygon intersection (touching is legal)."""
    rectangle=[(box[0],box[1]),(box[2],box[1]),(box[2],box[3]),(box[0],box[3])]
    axes=[(1,0),(0,1)]+[(polygon[(i+1)%len(polygon)][1]-p[1],
                         p[0]-polygon[(i+1)%len(polygon)][0]) for i,p in enumerate(polygon)]
    for x,y in axes:
        if x*x+y*y<1e-24:continue
        a=[p[0]*x+p[1]*y for p in rectangle];b=[p[0]*x+p[1]*y for p in polygon]
        if max(a)<=min(b) or max(b)<=min(a):return False
    return True


class Exclusion:
    """Spatially bucketed immutable convex polygons in city-local XY."""
    def __init__(self, polygons):
        self.polygons=[];self.cells={}
        for p in polygons:
            if len(p)<3 or any(len(v)!=2 or any(not math.isfinite(x) for x in v) for v in p):
                raise ValueError('invalid city exclusion polygon')
            area=sum(v[0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*v[1] for i,v in enumerate(p))
            if abs(area)<1e-12:continue
            turns=[(p[(i+1)%len(p)][0]-v[0])*(p[(i+2)%len(p)][1]-p[(i+1)%len(p)][1])-
                   (p[(i+1)%len(p)][1]-v[1])*(p[(i+2)%len(p)][0]-p[(i+1)%len(p)][0]) for i,v in enumerate(p)]
            if any(v>1e-12 for v in turns) and any(v<-1e-12 for v in turns):
                raise ValueError('city exclusion polygon must be convex')
            box=[min(v[0] for v in p),min(v[1] for v in p),max(v[0] for v in p),max(v[1] for v in p)]
            index=len(self.polygons);self.polygons.append(p)
            for cell in self.buckets(box):self.cells.setdefault(cell,set()).add(index)

    @staticmethod
    def buckets(box):
        if any(not math.isfinite(v) or abs(v)>4 for v in box):raise ValueError('city exclusion outside local envelope')
        return ((x,y) for x in range(math.floor(box[0]*8),math.floor(box[2]*8)+1)
                for y in range(math.floor(box[1]*8),math.floor(box[3]*8)+1))

    def blocks(self, box):
        candidates=set()
        for cell in self.buckets(box):candidates.update(self.cells.get(cell,()))
        return any(intersects(box,self.polygons[i]) for i in candidates)
