#!/usr/bin/env python3
"""Focused Q6 checkpoint; uses the shared runner, no platform mutation or VM."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
for system in ['city','city_shadows_off','city_contact_off','city_emissive_only','city_legacy','city_reverse','proxy_legacy']:
 subprocess.run([sys.executable,str(V2/'app/runner.py'),'check','--fixture',str(V2/f'fixtures/lighting/{system}.fixture.json'),'--candidate',system+'-04'],cwd=ROOT,check=True)
for region in ['real-mixed','real-holdout']:
 subprocess.run([sys.executable,str(V2/'app/runner.py'),'check','--fixture',str(V2/f'fixtures/lighting/{region}/response.fixture.json'),'--settings',str(V2/'fixtures/lighting/scroll.settings.json'),'--candidate',region+'-04'],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(V2/'app/runner.py'),'check','--fixture',str(V2/'fixtures/lighting/city.fixture.json'),'--settings',str(V2/'fixtures/lighting/scroll.settings.json'),'--candidate','city-scroll-04'],cwd=ROOT,check=True)
