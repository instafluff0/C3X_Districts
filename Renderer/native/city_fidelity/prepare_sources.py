"""Apply the proven Lab city normal/material intake to every normalized pool."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];LAB=ROOT/'Renderer/terrain_lab/v2'
OUT=ROOT/'Renderer/packs/CityFidelitySources'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((ROOT/'Renderer/packs/CityStudyAuxiliaryUV/city_catalog.json').read_text())
    records=[]
    for pool in sorted(catalog['pools']):
        key=pool.removeprefix('city/pool/');folder=OUT/key.replace('/','-');folder.mkdir(exist_ok=True)
        for script,destination in [('prepare_city_source_normals.py',folder/'normals.json'),('prepare_city_extra_materials.py',folder/'extra')]:
            ready=destination/'mapping.json' if destination.suffix=='' else destination
            if not ready.exists():
                args=[sys.executable,str(LAB/'qa'/script),'--pool',key,'--output',str(destination)]
                if script=='prepare_city_source_normals.py':args+=['--include-frame']
                elif destination.exists():args+=['--resume']
                subprocess.run(args,cwd=ROOT,check=True)
        records.append({'pool':pool,'normals':str((folder/'normals.json').relative_to(ROOT)),'materials':str((folder/'extra/mapping.json').relative_to(ROOT))})
        print('READY',key,flush=True)
    (OUT/'manifest.json').write_text(json.dumps({'schema':'c3x.city_source_fidelity.v1','pools':records},indent=2)+'\n')
if __name__=='__main__':main()
