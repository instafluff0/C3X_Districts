"""Bounded Q7 candidate controls; excludes the coordinator-paused full matrix."""
import copy,json,subprocess,sys
from pathlib import Path
import presentation as p

def main():
    jobs=[]
    for name in ['objects-02','context-small','context-large','mine-relief-01']:
        jobs.append(('quick',p.FIX/'generated'/name/'fixture.json',name))
    for region in ['mixed','mixed-holdout']:
        source=p.FIX/'generated'/('registered-'+region+'-v2')
        jobs.append(('quick',source/'before.fixture.json','registered-'+region+'-before-v2'))
    f=p.read(p.FIX/'generated/modern-01/fixture.json')
    for control in ['component-id','component-depth','shadows-off','emissive-only']:
        path=p.FIX/'generated'/('modern-'+control);(p.ROOT/path).mkdir(parents=True,exist_ok=True)
        variant=copy.deepcopy(f);variant['id']='q7-modern-'+control
        (p.ROOT/path/'control.txt').write_text(control+'\n')
        variant['scenarios']['control']=str(path/'control.txt')
        p.write(path/'fixture.json',variant)
        jobs.append(('check' if control=='emissive-only' else 'quick',path/'fixture.json','modern-'+control))
    for tier,fixture,name in jobs:
        command=[sys.executable,'Renderer/terrain_lab/v2/app/runner.py',tier,'--fixture',str(fixture),'--candidate',name,'--output',str(p.AUD/'out/Q1/Q7-presentation'/name)]
        subprocess.run(command,cwd=p.ROOT,check=True)
    p.write(p.AUD/'BOUNDED_CHECKS.json',dict(schema='c3x.q7.bounded_checks.v1',jobs=[dict(tier=t,fixture=str(f),candidate=n) for t,f,n in jobs],full_pool_matrix='deferred_by_coordinator_storage_schedule'))
if __name__=='__main__':main()
