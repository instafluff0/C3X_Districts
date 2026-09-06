"""Q3 checkpoint only; namespaced, cached Mac renders; no VM or promotion."""
import argparse
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
parser=argparse.ArgumentParser();parser.add_argument('--candidate',default='static-05');parser.add_argument('--group',choices=['real','synthetic'],required=True);args=parser.parse_args()
base=Path('Renderer/terrain_lab/v2/fixtures/hydrology')
fixtures=([base/x/'context.fixture.json' for x in ['real-mixed','real-holdout','real-wrap','real-wrap-holdout','real-mouth','real-mouth-holdout']] if args.group=='real' else [base/(x+'.fixture.json') for x in ['coves','islands','channel','rivers','north','east','south','west','lowland','beach-only','bed-only','classification','clearance','context-before']])
for fixture in fixtures:
 name=fixture.parent.name if args.group=='real' else fixture.name.split('.')[0]
 out=Path('Renderer/terrain_lab/v2/audits/hydrology/out/Q1/Q3-hydrology')/(name+'-'+args.candidate)
 subprocess.run([sys.executable,'Renderer/terrain_lab/v2/app/runner.py','check','--fixture',str(fixture),'--candidate',args.candidate,'--output',str(out)],cwd=ROOT,check=True)
 subprocess.run([sys.executable,'Renderer/terrain_lab/v2/audits/hydrology/review.py',str(out)],cwd=ROOT,check=True)
