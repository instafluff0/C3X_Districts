"""Executable real-source regression: shared edge normals and unchanged heights.

Uses cached licensed packs locally; never copies runtime/cache trees. Historical
Q2 probes remain intact. This is geometry evidence, not visual acceptance.
"""
import json
import subprocess
import sys
from pathlib import Path
from prepare_checkpoint import ROOT, V2, save, rel


def main():
    out = V2 / 'fixtures/beauty/normal-regression-v1'
    out.mkdir(exist_ok=True)
    points = []
    for axis in (0, 1):
        for y in range(4 if axis == 0 else 3):
            for x in range(3 if axis == 0 else 4):
                for i in range(17):
                    t = i / 16
                    points.extend([(x,y,1,1-t),(x+1,y,0,1-t)] if axis == 0
                                  else [(x,y,t,0),(x,y+1,t,1)])
    (out/'points.csv').write_text(''.join(','.join(map(str,p))+'\n' for p in points))
    results=[]
    for region in ('wet','dry','cold'):
        source=V2/f'fixtures/terrain/composed-hydro-{region}-on.fixture.json'
        f=json.loads(source.read_text());f['track']='Q8-beauty'
        m=json.loads((ROOT/f['modules'][0]).read_text())
        # The source manifest retains its Q2 namespace; only the new wrapper
        # and regression outputs belong to the combined lead's Q8 namespace.
        m['shader_owner']=m['owner']
        for key in ('terrain_hooks','hydrology_hooks'):
            if key in m:m[key].setdefault('owner',m['owner'])
        m['owner']='Q8-beauty'
        samples=[]
        for mode in ('before','after'):
            m.pop('continuous_normals',None)
            if mode=='after':m['continuous_normals']=1
            mp=out/f'{region}-{mode}.module.json';save(mp,m)
            f['modules']=[rel(mp)];fp=out/f'{region}-{mode}.fixture.json';save(fp,f)
            target=V2/f'audits/beauty/out/normal-regression-v1/{region}-{mode}.json'
            subprocess.run([sys.executable,str(V2/'app/surface_query.py'),'--fixture',rel(fp),
                '--points',rel(out/'points.csv'),'--output',rel(target)],cwd=ROOT,check=True)
            samples.append(json.loads(target.read_text())['samples'])
        before,after=samples
        assert all(a['height']==b['height'] and a['depth']==b['depth']
                   and a['screen_x']==b['screen_x'] and a['screen_y']==b['screen_y']
                   for a,b in zip(before,after)), 'geometry/projection changed'
        def seams(rows):
            errors=[max(abs(a['normal_'+c]-b['normal_'+c]) for c in 'xyz')
                    for a,b in zip(rows[::2],rows[1::2])]
            return dict(max_delta=max(errors),failures=sum(e>1e-3 for e in errors))
        result=dict(region=region,pairs=len(points)//2,before=seams(before),after=seams(after),
                    heights_depth_and_anchors_unchanged=True)
        assert result['after']['failures']==0,result
        results.append(result)
    save(V2/'audits/beauty/NORMAL_REGRESSION.json',dict(accepted=False,results=results))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
