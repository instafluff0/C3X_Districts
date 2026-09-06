"""Compile checked provider polygons for the Q4 C++ source-vertex callback."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
V2=ROOT/'Renderer/terrain_lab/v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--name',default='city-clearance');a=p.parse_args()
 if not a.name.replace('-','').isalnum():raise ValueError('invalid owned fixture name')
 f=json.loads(a.fixture.read_text());rows=[];sources={a.fixture.as_posix():sha(a.fixture)}
 q7=V2/'audits/objects/metadata'/('registered-'+f['real_map']['region_id']+'-v2')/'corridors.json'
 if q7.is_file():f.setdefault('sidecars',[]).append(dict(path=q7.relative_to(ROOT).as_posix(),sha256=sha(q7),schema='c3x.lab_v2.corridors.v1',owner='Q7-presentation'))
 for sidecar in f.get('sidecars',[]):
  if sidecar['schema']!='c3x.lab_v2.corridors.v1':continue
  path=ROOT/sidecar['path']
  if sha(path)!=sidecar['sha256']:raise ValueError('corridor hash drift')
  c=json.loads(path.read_text())
  if c['coordinate_space']!='civ3_raw_delta_pixels_v1':raise ValueError('corridor coordinate mismatch')
  if c['terrain_sha256']!=sha(ROOT/f['terrain']):raise ValueError('corridor terrain mismatch')
  sources[sidecar['path']]=sha(path)
  for e in c['envelopes']:rows.append((e['clearance'],*c['wrap_period'],e['polygon']))
 witness=V2/'audits/objects/CITY_VEGETATION_WITNESS.json';city=json.loads(witness.read_text());sources[witness.relative_to(ROOT).as_posix()]=sha(witness)
 for region in ([] if q7.is_file() else city['regions']):
  if region['origin_raw']!=f['real_map']['region']['origin']:continue
  if region['source_sha256']!=f['real_map']['source_sha256']:raise ValueError('city source mismatch')
  if region['coordinate_space']!='civ3_raw_delta_pixels_v1':raise ValueError('city coordinate mismatch')
  # No camera/screen positions: preserve the published raw-city anchor and polygons.
  for e in region['polygons']:rows.append((4.,0.,0.,e['raw_delta_pixel_polygon']))
 if not rows:raise ValueError('no provider clearance polygons for this region')
 out=V2/'fixtures/relief'/a.name;out.mkdir(parents=True,exist_ok=True)
 # Runner namespaces fixture inputs by owner. Copy exact bytes, preserving the
 # provider's terrain/scenario hashes and augmentation semantics.
 for key,value in list(f['scenarios'].items())+[('terrain',f['terrain'])]:
  source=ROOT/value;target=out/(key+source.suffix);shutil.copyfile(source,target)
  if key=='terrain':f['terrain']=target.relative_to(ROOT).as_posix()
  else:f['scenarios'][key]=target.relative_to(ROOT).as_posix()
 if f.get('real_map',{}).get('overlay'):
  source=ROOT/f['real_map']['overlay'];target=out/'augmentation.json';shutil.copyfile(source,target)
  f['real_map']['overlay']=target.relative_to(ROOT).as_posix()
 csv=out/'clearance.csv';csv.write_text('C3X_Q4_CLEARANCE_V1\n'+''.join(','.join(map(str,[margin,wx,wy,len(poly),*(v for point in poly for v in point)]))+'\n' for margin,wx,wy,poly in rows))
 module=json.loads((ROOT/f['modules'][0]).read_text());module.update(id='q4-source-vertex-clearance',owner='Q4-relief')
 module['placement_hooks']=dict(header='Renderer/terrain_lab/v2/systems/relief/placement_adapter.h',owner='Q4-relief',initialize='q4_placement::initialize',accept_vegetation='q4_placement::accept_vegetation')
 (out/'module.json').write_text(json.dumps(module,indent=2)+'\n')
 f.update(id='q4-'+a.name,track='Q4-relief',modules=[(out/'module.json').relative_to(ROOT).as_posix()])
 compiled=dict(schema='c3x.lab_v2.corridors.v1',coordinate_space='civ3_raw_delta_pixels_v1',terrain_sha256=sha(ROOT/f['terrain']),region_id=f['real_map']['region_id'],provider='Q4-relief compiled provider cache',revision=1,wrap_period=[0,0],envelopes=[],relief_clearance=dict(path=csv.relative_to(ROOT).as_posix(),sha256=sha(csv)),source_hashes=sources)
 index=out/'q4-clearance.json';index.write_text(json.dumps(compiled,indent=2)+'\n')
 f.setdefault('sidecars',[]).append(dict(path=index.relative_to(ROOT).as_posix(),sha256=sha(index),schema=compiled['schema'],owner='Q4-relief'))
 (out/'fixture.json').write_text(json.dumps(f,indent=2)+'\n')
 (out/'provenance.json').write_text(json.dumps(dict(schema='c3x.q4.placement_input.v1',sources=sources,polygon_count=len(rows),clearance_sha256=sha(csv),scope='Actual source vertex hull versus published polygons; city-only when no corridor sidecars are attached. No claim of complete route/wrap coverage.'),indent=2)+'\n')
 print((out/'fixture.json').relative_to(ROOT))
if __name__=='__main__':main()
