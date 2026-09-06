#!/usr/bin/env python3
"""Q4-only GPU checkpoint. Does not invoke Windows or shared promotion gates."""
import subprocess,sys
cases=[('check','real-holdout',10),('check','biomes-clearance',10)]
cases += [('quick',n,9) for n in ['source-rock','dunes','volcano','biomes','coast-r0','coast-r1','coast-r2','coast-r3','island','cove']]
for tier,name,revision in cases:
 subprocess.run([sys.executable,'Renderer/terrain_lab/v2/app/runner.py',tier,'--fixture',f'Renderer/terrain_lab/v2/fixtures/relief/{name}-r{revision}/fixture.json','--candidate',f'{name}-{revision:02}-{tier}'],check=True)
