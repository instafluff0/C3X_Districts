"""Measure normalized source channels without copying licensed image payloads."""
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'Renderer'))
from preview.render_textured_patch import decode_bc3_alpha, decode_bc3_color
PACK=ROOT/'Renderer/packs/Civ5EnvironmentSkin'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def measure(p):
 data=p.read_bytes();h,w=struct.unpack_from('<II',data,12);fmt=struct.unpack_from('<I',data,128)[0];mips=struct.unpack_from('<I',data,28)[0]
 block=16 if fmt in [77,78] else 8
 if fmt not in [77,78,80]:raise ValueError('unmeasured format')
 values=[]
 for sy in range(64):
  for sx in range(64):
   x=min(w-1,int((sx+.5)*w/64));y=min(h-1,int((sy+.5)*h/64));off=148+((y//4)*((w+3)//4)+x//4)*block;b=data[off:off+block]
   values.append(sum(decode_bc3_color(b,x%4,y%4))/3/255 if block==16 else decode_bc3_alpha(b+b'\0'*8,x%4,y%4)/255)
 mean=sum(values)/len(values)
 return {'dimensions':[w,h],'mips':mips,'dxgi_format':fmt,'sample_grid':[64,64],'measurement':'encoded RGB channel mean for base color; linear scalar for height/specular','min':min(values),'max':max(values),'mean':mean,'stddev':math.sqrt(sum((v-mean)**2 for v in values)/len(values))}
def main():
 components=[]
 for name in ['grassland','plains','desert','tundra','marsh','flood_plain']:
  descriptor=PACK/f'materials/{name}.json';m=json.loads(descriptor.read_text());roles={}
  for role in ['base_color','height','specular']:
   p=PACK/m[role]['texture'];roles[role]={'path':p.relative_to(ROOT).as_posix(),'sha256':digest(p),'declared_color_space':m[role]['color_space'],'statistics':measure(p)}
  components.append({'classification':'source_adaptation','asset_id':f'terrain/{name}/base','material':descriptor.relative_to(ROOT).as_posix(),'material_sha256':digest(descriptor),'roles':roles,'geometry':'generated flat continuous base grid, not source-authored relief','uv_adaptation':'integer periodic repeats on map extent; same source height sampled at 1x/3x/8x','engine_claim':'source channels confirmed by normalized descriptor and prior source audits; detail amplitudes and inverse-specular roughness are C3X-authored interpretation'})
 out={'schema':'c3x.q2.source_audit.v1','selected_pack':'Civ5EnvironmentSkin','components':components,'no_original_art_fallback':True,'absent_features':'This candidate adds no dominant rock, hill, dune, mountain, vegetation, water, or object body.'}
 (ROOT/'Renderer/terrain_lab/v2/audits/terrain/source_materials_v1.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
