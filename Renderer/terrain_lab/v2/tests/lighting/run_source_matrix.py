#!/usr/bin/env python3
"""Q6 source-backed shared-light/contact checkpoint, no VM or backend mutation."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5];V2=ROOT/'Renderer/terrain_lab/v2'
for system in ['q7_city','trees','rocks','units','improvements']:
    for control in ['', '_contact_off','_shadows_off']:
        fixture=V2/f'fixtures/lighting/{system}{control}.fixture.json'
        subprocess.run([sys.executable,str(V2/'app/runner.py'),'check','--fixture',str(fixture),'--candidate',system+control+'-09'],cwd=ROOT,check=True)
for system in ['trees','rocks','units','improvements','q7_city']:
    subprocess.run([sys.executable,str(V2/'app/runner.py'),'check','--fixture',str(V2/f'fixtures/lighting/{system}.fixture.json'),'--settings',str(V2/'fixtures/lighting/scroll.settings.json'),'--candidate',system+'-scroll-09'],cwd=ROOT,check=True)

for system in ['q7_city_emissive_only','q7_city_reverse']:
    subprocess.run([sys.executable,str(V2/'app/runner.py'),'check','--fixture',str(V2/f'fixtures/lighting/{system}.fixture.json'),'--candidate',system+'-09'],cwd=ROOT,check=True)
