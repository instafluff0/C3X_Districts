#!/usr/bin/env python3
"""Consume Q7's complete worked components and stable layout without editing it."""
import hashlib,importlib.util,json
from pathlib import Path
from prepare_city import ROOT,V2
from packet_geometry import write_packet

def prepare():
 source=V2/'systems/objects/presentation.py';spec=importlib.util.spec_from_file_location('q7_presentation',source);q7=importlib.util.module_from_spec(spec);spec.loader.exec_module(q7)
 pool='city/pool/american/modern';assets=[q7.component(a) for a in q7.read(q7.PACK/'city_catalog.json')['pools'][pool]['components']]
 # Read the Q7 producer's existing complete recipe, never create a second layout policy.
 layout=q7.layout(assets,2,recipe='stable');vertices=[];pairs=[];placements=[];source_inputs={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()}
 for instance in layout:
  a=instance['asset'];placements.append({k:v for k,v in instance.items() if k!='asset'}|{'asset':a['id']})
  manifest=q7.read(q7.PACK/'manifest.json');landmark=q7.PACK/manifest['assets'][a['id']]['landmark'];desc=q7.read(landmark)
  for p in [landmark,*[q7.PACK/x for x in desc['components']['geometry']],*[q7.PACK/x for x in desc['components']['materials']]]:source_inputs[str(p)]=q7.sha(p)
  for mesh,mat in a['parts']:
   channels=mat['channels'];pair=(channels['base_color']['texture'],channels.get('emissive',{}).get('texture',''))
   if pair not in pairs:pairs.append(pair)
   material=pairs.index(pair);transformed=[]
   for v in mesh['vertices']:
    pos=[v['position'][0]-(a['lo'][0]+a['hi'][0])/2,v['position'][1]-(a['lo'][1]+a['hi'][1])/2,v['position'][2]-a['lo'][2]]
    p=[x*instance['scale'] for x in q7.rotate(pos,instance['rotation'])];p[0]+=instance['x'];p[1]+=instance['y']
    transformed.append([0,0,0,*v['uv0'],*q7.rotate(v['normal'],instance['rotation']),*p,float(material)])
   for i in mesh['topology']['indices']:vertices.extend(transformed[i])
 for x,y in [(-2,-2),(-2,2),(2,-2),(2,-2),(-2,2),(2,2)]:vertices.extend((0,0,0,0,0,0,0,1,x,y,-.002,8))
 out=V2/'fixtures/lighting/generated/q7-city';texhashes,geometry_hash=write_packet(out/'scene.packet',vertices,pairs)
 provenance={'classification':'source_reuse','placement':'source_adaptation by Q7 stable recipe','diagnostic_receiver':'diagnostic_proxy; flat neutral; not real terrain','pool':pool,'size':2,'source_inputs':source_inputs,'texture_hashes':texhashes,'pairs':pairs,'placements':placements,'geometry_sha256':geometry_hash,'triangles':len(vertices)//36,'geometry_and_uv':'all Q7 worked parts, unchanged UV, whole-compound rigid rotation/uniform scale','lights':'only source emissive bindings; no unresolved analytic sockets enabled'}
 (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 f=json.loads((V2/'fixtures/lighting/city_linear.fixture.json').read_text());f.update(id='q6-q7-complete-city',viewport=[384,256],scenarios={'lighting_geometry':str((out/'scene.packet').relative_to(ROOT))})
 module=json.loads((V2/'systems/lighting/city_linear.module.json').read_text());module.update(id='lighting-q7-city',source='Renderer/terrain_lab/v2/systems/lighting/city_native.cpp');mp=V2/'systems/lighting/q7_city.module.json';mp.write_text(json.dumps(module,indent=2)+'\n');f['modules']=[str(mp.relative_to(ROOT))]
 (V2/'fixtures/lighting/q7_city.fixture.json').write_text(json.dumps(f,indent=2)+'\n')
 for control,macro in [('shadows_off','Q6_SHADOWS 0'),('contact_off','Q6_CONTACT 0'),('emissive_only','Q6_EMISSIVE_ONLY 1')]:
  sh=V2/f'shaders/lighting/q7_city_{control}.hlsl';sh.write_text(f'#define Q6_LINEAR 1\n#define {macro}\n#include "city.hlsl"\n')
  m=dict(module,id='lighting-q7-'+control,shader=str(sh.relative_to(ROOT)));mp=V2/f'systems/lighting/q7_city_{control}.module.json';mp.write_text(json.dumps(m,indent=2)+'\n')
  cf=dict(f,id='q6-q7-'+control,modules=[str(mp.relative_to(ROOT))]);(V2/f'fixtures/lighting/q7_city_{control}.fixture.json').write_text(json.dumps(cf,indent=2)+'\n')
 print('Q7 complete city:',len(pairs),'materials,',len(vertices)//36,'triangles')
if __name__=='__main__':prepare()
