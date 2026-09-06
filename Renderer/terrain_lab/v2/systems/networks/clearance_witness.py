"""Generate an envelope/footprint diagnostic from the exact source candidate mesh."""
import json,struct
from fixtures import ROOT,OUT,rel,save_json,height
from network import Mesh
from clearance import footprint_intersects
p=OUT/'source-clearance';env=json.loads((p/'clearance.json').read_text());f=json.loads((p/'fixture.json').read_text())
m=Mesh(height);out=OUT/'clearance-overlay';out.mkdir(exist_ok=True)
for e in env['entries']:
    pts=[tuple(p) for p in e['points']]
    if e['shape']=='capsule_chain' and pts[0]!=pts[-1]:
        for offset in (-e['clearance_radius'],e['clearance_radius']):m.strip(pts,.6,(.5,.4,.05),offset,1.)
    elif e['shape']=='polygon':m.strip(pts+[pts[0]],.8,(.5,.4,.05),lift=1.)
cases=[]
for i,(x,y,w,d) in enumerate([(-120,290,20,24),(-120,332,20,24),(105,440,30,20),(120,505,24,28),(0,600,20,20),(60,300,25,25)]):
    poly=[(x-w/2,y-d/2),(x+w/2,y-d/2),(x+w/2,y+d/2),(x-w/2,y+d/2)]
    hit=footprint_intersects(env,poly);m.strip(poly+[poly[0]],1.2,(.6,.04,.03) if hit else (.05,.5,.15),lift=1.2)
    cases.append({'id':i,'footprint':poly,'intersects':hit,'color':'red' if hit else 'green'})
b=(p/'scene.bin').read_bytes();magic,n,*camera=struct.unpack_from('<II4f',b)
with (out/'scene.bin').open('wb') as stream:
    stream.write(struct.pack('<II4f',magic,n+len(m.vertices),*camera));stream.write(b[24:])
    for v in m.vertices:stream.write(struct.pack('<13f',*v,0,0,-1,0))
f['id']='clearance-overlay';f['scenarios']['network_mesh']=rel(out/'scene.bin');save_json(out/'fixture.json',f)
save_json(out/'footprint_results.json',{'classification':'diagnostic_proxy','cases':cases,'envelope':rel(p/'clearance.json'),'rule':'source candidate unchanged; yellow clearance outline; red rejected footprint; green allowed footprint'})
