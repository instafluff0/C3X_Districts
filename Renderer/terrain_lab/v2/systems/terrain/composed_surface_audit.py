"""Query both incident source triangles; report remaining shared geometry seams."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
def main():
 p=argparse.ArgumentParser();p.add_argument('region',choices=['wet','dry','cold']);a=p.parse_args()
 points=[];edges=[]
 for axis in ['column','row']:
  for y in range(4 if axis=='column' else 3):
   for x in range(3 if axis=='column' else 4):
    for i in range(17):
     t=i/16
     pair=[(x,y,1,1-t),(x+1,y,0,1-t)] if axis=='column' else [(x,y,t,0),(x,y+1,t,1)]
     points.extend(pair);edges.append(dict(axis=axis,column=x,row=y,along=t))
 path=V2/f'fixtures/terrain/shared-edge-points.csv'
 path.write_text(''.join(','.join(map(str,q))+'\n' for q in points))
 out=V2/f'audits/terrain/out/Q1/Q2-terrain/{a.region}-surface-query.json'
 subprocess.run([sys.executable,str(V2/'app/surface_query.py'),'--fixture',str(V2/f'fixtures/terrain/composed-hydro-{a.region}-on.fixture.json'),'--points',str(path),'--output',str(out)],cwd=ROOT,check=True)
 data=json.loads(out.read_text());maximum={'height':0,'normal':0,'shore':0};failures=[]
 for i,edge in enumerate(edges):
  x,y=data['samples'][i*2:i*2+2]
  errors={'height':abs(x['height']-y['height']),'normal':max(abs(x['normal_'+c]-y['normal_'+c]) for c in 'xyz'),'shore':abs(x['shore_distance']-y['shore_distance'])}
  for k,v in errors.items():maximum[k]=max(maximum[k],v)
  if errors['height']>1e-4 or errors['normal']>1e-3 or errors['shore']>1e-5:failures.append(dict(edge,errors=errors,terrain=[[x['base'],x['real']],[y['base'],y['real']]]))
 record=dict(schema='c3x.q2.composed_surface_audit.v1',region=a.region,source_query=out.relative_to(ROOT).as_posix(),sample_pairs=len(edges),max_delta=maximum,remaining_failures=failures,scope='Exact source builder incident edges; geometry failures belong to shared relief/shore composition, not flat Q2 base field.')
 (V2/f'audits/terrain/{a.region}_surface_seams.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k!='remaining_failures'}));print('Remaining incident-edge failures:',len(failures))
if __name__=='__main__':main()
