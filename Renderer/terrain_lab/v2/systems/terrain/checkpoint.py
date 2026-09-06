"""Q2 checkpoint driver. Uses only the shared Mac runner; no alternate backend."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from fixture_matrix import ROOT, OWNED, cases, csv_fixture
V2=ROOT/'Renderer/terrain_lab/v2'
RUNNER=V2/'app/runner.py'
def run(fixture,candidate,tier='check'):
 subprocess.run([sys.executable,str(RUNNER),tier,'--fixture',str(fixture),'--candidate',candidate],cwd=ROOT,check=True)
def main():
 p=argparse.ArgumentParser();p.add_argument('group',choices=['controls','real','regions','pairs']);p.add_argument('--revision',default='r06');args=p.parse_args()
 if args.group=='controls':
  for mode in ['detail','off','baseline','normal','height','roughness','albedo','weights']:
   run(OWNED/f'context-{mode}.fixture.json',f'context-{mode}-{args.revision}', 'check' if mode in ['detail','off','baseline'] else 'compose')
 elif args.group in ['real','regions']:
  names=['mixed','holdout','wrap'] if args.group=='real' else ['q2-dry','q2-cold','q2-wet','q2-dry-holdout','q2-cold-holdout','q2-wet-holdout']
  for name in names:
   run(OWNED/f'real-{name}/fixture.json',f'real-{name}-base-{args.revision}')
 else:
  # Five actual base material channels; all 14 terrain identities separately
  # exercise their mapping/transition policy in the portable matrix. Water and
  # raised relief composition need external owners, so this is not universal acceptance.
  families={0,1,2,3,9}; template=json.loads((OWNED/'micro.fixture.json').read_text())
  for case in cases():
   if len(case['families'])!=2 or not set(case['families'])<=families or case['origin']!=98 or 'base_override' in case:continue
   target=OWNED/'material-pairs';target.mkdir(exist_ok=True)
   csv=target/(case['id']+'.csv');csv.write_text(csv_fixture(case))
   f=dict(template,id='q2-'+case['id'],terrain=csv.relative_to(ROOT).as_posix(),viewport=[256,128])
   path=target/(case['id']+'.fixture.json');path.write_text(json.dumps(f,indent=2)+'\n')
   run(path,f'{case["id"]}-{args.revision}','check')
if __name__=='__main__':main()
