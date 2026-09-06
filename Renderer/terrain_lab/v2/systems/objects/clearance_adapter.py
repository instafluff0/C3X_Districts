"""Q7 normalized tile footprint adapter to Q5's published intersection query."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'networks'))
from clearance import footprint_intersects

def projected_plane(p):return [(p[0]-p[1])*64,(p[0]+p[1])*64]

def local_witness(kind='road_and_rail'):
    # Explicit synthetic through-route. Not captured state or an actual Q5 graph.
    # Radius includes both rendered corridor and setback in normal-zoom pixels.
    return dict(schema='c3x.q5.route_clearance.v1',world_wrap=[0,0],halo_complete=True,
        classification='diagnostic_proxy',entries=[dict(id='q7-through-route',kind=kind,
        shape='capsule_chain',points=[projected_plane([0,-.8]),projected_plane([0,.8])],
        occupied_radius=4,clearance_radius=6,height_range=[0,1])])

def intersects(bounds,envelope):
    x0,y0,x1,y1=bounds
    polygon=[projected_plane(p) for p in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]]
    return footprint_intersects(envelope,polygon)
