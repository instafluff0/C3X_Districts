"""Verify Q2 r06 evidence artifacts; report pending composition separately."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
OUT=ROOT/'Renderer/terrain_lab/v2/audits/terrain/out/Q1/Q2-terrain'
reports=sorted(OUT.glob('*-r06/report.json'));images=repeats=0
for path in reports:
 report=json.loads(path.read_text())
 for row in report['outputs']:
  image=ROOT/row['image'];blob=image.read_bytes()
  assert hashlib.sha256(blob).hexdigest()==row['sha256'],str(image.relative_to(ROOT))
  images+=1
  if report['tier']=='check':
   assert Path(str(image)+'.repeat1.bmp').read_bytes()==blob,'deterministic repeat failed'
   repeats+=1
pairs=[p for p in reports if p.parent.name.startswith('pair_')]
assert len(pairs)==60,len(pairs)
assert images==832,images
assert repeats==792,repeats
print(json.dumps({'reports':len(reports),'images':images,'byte_identical_repeats':repeats,'base_material_pair_cases':len(pairs),'scope':'isolated base candidate; composed 14-family, Q1 calibration and D3D parity pending'}))
