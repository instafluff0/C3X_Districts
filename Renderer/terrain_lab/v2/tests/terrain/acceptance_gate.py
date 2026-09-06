"""Evidence gate: a clean isolated render cannot hide composed normal failures."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
AUDIT=ROOT/'Renderer/terrain_lab/v2/audits/terrain'
metrics=json.loads((AUDIT/'composed_metrics_v1.json').read_text())
failures=[];images=0
for checkpoint in metrics['checkpoints']:
 report=json.loads((ROOT/checkpoint['report']).read_text())
 assert report['tier']=='check' and len(report['outputs'])==8
 for row in report['outputs']:
  path=ROOT/row['image'];blob=path.read_bytes()
  assert hashlib.sha256(blob).hexdigest()==row['sha256']
  assert Path(str(path)+'.repeat1.bmp').read_bytes()==blob
  images+=1
assert images==32
for zoom,scroll in metrics['scroll'].items():
 assert scroll['return_to_origin_identical']
for region in ['wet','cold','dry']:
 report=json.loads((AUDIT/f'{region}_surface_seams.json').read_text())
 assert report['sample_pairs']==408
 if report['remaining_failures']:
  failures.append(dict(region=region,incident_edge_failures=len(report['remaining_failures']),max_delta=report['max_delta']))
candidate=json.loads((AUDIT/'CANDIDATE.json').read_text())
pending=[x for x in candidate['acceptance'] if x['state']!='passed']
print(json.dumps(dict(verified_composed_images=images,geometry_failures=failures,pending_criteria=pending),indent=2))
raise SystemExit(1 if failures or pending else 0)
