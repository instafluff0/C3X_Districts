"""Recompute compact evidence from rendered pixels and pinned source atlas data."""
from pathlib import Path
import sys,json,hashlib,struct
from fixtures import ROOT,height
from review import OUT,compare,read
sys.path.insert(0,str(ROOT/'Renderer'))
from preview.render_textured_patch import DdsBc3Texture
out=ROOT/'Renderer/terrain_lab/v2/audits/networks'
t=DdsBc3Texture.from_file(ROOT/'Renderer/packs/RouteStylesNormalized/textures/routes/base_color_c3ae5ee0b5879164.dds')
coverage=lambda y:sum(t.sample_rgba((i+.5)/256,y)[3]>127 for i in range(256))/256
metrics={'atlas_coverage':{'inherited_short_steel_row':coverage(.095),'full_length_rail_a':coverage(88.5/256),'full_length_rail_b':coverage(104.5/256)},'matched_uv_correction':compare(OUT/'source-06/h12-z1-pan00.bmp',OUT/'source-07/h12-z1-pan00.bmp'),'rail_width_change':compare(OUT/'source-09/h12-z1-pan00.bmp',OUT/'source-10/h12-z1-pan00.bmp')}
b=(ROOT/'Renderer/terrain_lab/v2/fixtures/networks/source-10/scene.bin').read_bytes();n=struct.unpack_from('<I',b,4)[0];clearance=[]
for i in range(n):
 v=struct.unpack_from('<13f',b,24+i*52)
 if 0<=v[11]<5:clearance.append(v[2]-height(v[0],v[1]))
metrics['route_grade']={'sampled_vertices':len(clearance),'min_offset_pixels':min(clearance),'max_offset_pixels':max(clearance),'below_ground':sum(v<0 for v in clearance)}
p=OUT/'final-10';report=json.loads((p/'report.json').read_text());returns=[]
for h in (12,18,0,6):
 for z in (1,2):
  a=p/f'h{h:02d}-z{z}-pan00.bmp';b=p/f'h{h:02d}-z{z}-pan03.bmp';returns.append(a.read_bytes()==b.read_bytes())
metrics['return_to_origin']={'cases':len(returns),'all_byte_identical':all(returns)}
metrics['final_frame_hashes']={o['image']:o['sha256'] for o in report['outputs']}
(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps({k:v for k,v in metrics.items() if k!='final_frame_hashes'},indent=2))
