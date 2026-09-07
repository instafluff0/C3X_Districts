"""Authored city ground coverage; independent of source game and source atlases."""
import math


def rectangle_distance(x,y,box):
    """Signed Euclidean distance, including rounded dilation at the corners."""
    dx=max(box[0]-x,x-box[2]);dy=max(box[1]-y,y-box[3])
    return math.hypot(max(dx,0),max(dy,0))+min(max(dx,dy),0)


def convex_hull(points):
    """Counterclockwise footprint, retaining the mesh's authored orientation."""
    points=sorted(set(tuple(p) for p in points))
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def half(sequence):
        result=[]
        for p in sequence:
            while len(result)>1 and cross(result[-2],result[-1],p)<=0:result.pop()
            result.append(p)
        return result
    hull=half(points)[:-1]+half(reversed(points))[:-1]
    if len(hull)<3:raise ValueError('ground footprint must have positive area')
    return hull


def polygon_distance(x,y,polygon):
    inside=True;distance=math.inf
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        dx=b[0]-a[0];dy=b[1]-a[1]
        inside=inside and dx*(y-a[1])-dy*(x-a[0])>=0
        t=max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)))
        distance=min(distance,math.hypot(x-a[0]-t*dx,y-a[1]-t*dy))
    return -distance if inside else distance


def footprint_alignment(points):
    """Infer the nearest grid alignment from a minimum-area footprint rectangle."""
    hull=convex_hull(points);candidates=[]
    for a,b in zip(hull,hull[1:]+hull[:1]):
        angle=math.atan2(b[1]-a[1],b[0]-a[0]);c=math.cos(angle);s=math.sin(angle)
        rotated=[(x*c+y*s,-x*s+y*c) for x,y in hull]
        area=(max(p[0] for p in rotated)-min(p[0] for p in rotated))*(max(p[1] for p in rotated)-min(p[1] for p in rotated))
        rotation=-((angle+math.pi/4)%(math.pi/2)-math.pi/4)
        candidates.append((area,abs(rotation),rotation))
    rotation=min(candidates)[2]
    # Compressed source coordinates perturb otherwise integral-degree axes.
    degrees=math.degrees(rotation)
    if abs(degrees-round(degrees))<.01:rotation=math.radians(round(degrees))
    return rotation


def coverage(x,y,boxes,margin,feather,polygons=()):
    if (not boxes and not polygons) or margin<=0:return 0.
    distance=min([rectangle_distance(x,y,b) for b in boxes]+[polygon_distance(x,y,p) for p in polygons])
    value=max(0.,min(1.,(margin-distance)/feather))
    return value*value*(3-2*value)


def grid(boxes,margin,step=.025):
    low=[math.floor((min(b[i] for b in boxes)-margin)/step) for i in (0,1)]
    high=[math.ceil((max(b[i] for b in boxes)+margin)/step) for i in (2,3)]
    width=high[0]-low[0]+1;height=high[1]-low[1]+1
    if width*height>30000:raise ValueError('settlement grid exceeds bounded city footprint')
    points=[((low[0]+x)*step,(low[1]+y)*step) for y in range(height) for x in range(width)]
    triangles=[]
    for y in range(height-1):
        for x in range(width-1):
            i=y*width+x
            triangles.extend([(i,i+1,i+width),(i+1,i+width+1,i+width)])
    return points,triangles
