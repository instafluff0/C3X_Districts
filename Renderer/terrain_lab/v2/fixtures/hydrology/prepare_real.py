"""Replay Q0 verified cached regions with Q3's module; preserve terrain bytes."""
import json
import subprocess
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
for region,name in [('mixed','real-mixed'),('mixed-holdout','real-holdout'),('wrap','real-wrap'),('wrap-holdout','real-wrap-holdout'),('q3-mouth','real-mouth'),('q3-mouth-holdout','real-mouth-holdout')]:
 dest=HERE/name
 subprocess.run([sys.executable,'Renderer/terrain_lab/v2/app/real_map.py','export',region,'--owner','Q3-hydrology','--halo','2','--output',str(dest.relative_to(ROOT))],cwd=ROOT,check=True)
 path=dest/'fixture.json';f=json.loads(path.read_text());f['modules']=['Renderer/terrain_lab/v2/systems/hydrology/static.module.json'];f['references']=['civ6.sea_and_shore','civ6.river','civ6.rocky_hill_coast'];f['scenarios']={'controls':str((dest/'controls.json').relative_to(ROOT))}
 # Keep registered region camera for overview, plus a fixed 128x64 contextual view.
 path.write_text(json.dumps(f,indent=2)+'\n')
 (dest/'controls.json').write_text(json.dumps({'material_root':'Renderer/packs/Civ5EnvironmentSkin/provenance/terrain_textures','mode':0,'wrap_x':1},indent=2)+'\n')
 f['id']+='-context';f['viewport']=[384,192];f['scenarios']['controls']=str((dest/'context.controls.json').relative_to(ROOT));(dest/'context.fixture.json').write_text(json.dumps(f,indent=2)+'\n')
 (dest/'context.controls.json').write_text(json.dumps({'material_root':'Renderer/packs/Civ5EnvironmentSkin/provenance/terrain_textures','mode':0,'wrap_x':1,'tile_halfwidth':64,'padding':1},indent=2)+'\n')
