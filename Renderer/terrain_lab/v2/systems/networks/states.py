"""Source-fidelity stress sheet: rows ancient/medieval/industrial/modern/rail;
columns normal/pillaged. This is not a gameplay or real-map fixture.
"""
from pathlib import Path
import json,struct
from network import Node,Edge,Graph,Mesh
from source_routes import build_source_routes
from fixtures import ROOT,BASE,OUT,rel,save_json
p=OUT/'source-states';p.mkdir(exist_ok=True)
m=Mesh(lambda x,y:0)
m.quad((-300,-180,0),(300,-180,0),(300,560,0),(-300,560,0),(.18,.24,.08))
for stage in range(5):
    for state in range(2):
        x=-230+state*280;y=stage*96
        g=Graph([Node(0,(0,0),(x,y)),Node(1,(2,0),(x+128,y))],[Edge(0,0,1,stage==4,min(stage,3),bool(state))])
        build_source_routes(g,m,rail_width=7)
with (p/'scene.bin').open('wb') as f:
    f.write(struct.pack('<II4f',0x354e4555,len(m.vertices),0,96,1,0))
    for v in m.vertices:
        if len(v)==9:v=(*v,0,0,-1,0)
        f.write(struct.pack('<13f',*v))
f=json.loads((OUT/'source-08/fixture.json').read_text());f['id']='source-states';f['viewport']=[560,320]
f['scenarios']['network_mesh']=rel(p/'scene.bin')
save_json(p/'fixture.json',f)
