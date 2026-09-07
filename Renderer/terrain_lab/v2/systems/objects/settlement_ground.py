"""Authored city ground coverage; independent of source game and source atlases."""
import math


def rectangle_distance(x,y,box):
    """Signed Euclidean distance, including rounded dilation at the corners."""
    dx=max(box[0]-x,x-box[2]);dy=max(box[1]-y,y-box[3])
    return math.hypot(max(dx,0),max(dy,0))+min(max(dx,dy),0)


def coverage(x,y,boxes,margin,feather):
    if not boxes or margin<=0:return 0.
    distance=min(rectangle_distance(x,y,b) for b in boxes)
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
