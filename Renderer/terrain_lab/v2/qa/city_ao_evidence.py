"""Verify the preserved source AO coordinate and combined city material probes."""
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'
FIX=V2/'fixtures/beauty'
MEDIEVAL='european-medieval-s1'
ANCIENT='american-ancient-s1-capital'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def augmentation(revision,name):return read(FIX/f'city-scene-r{revision}'/name/'augmentation.json')
def placement(a):return [{k:i[k] for k in ('asset','slot','scale','rotation','offset','local_bounds')} for i in a['instances']]


def wire_prefix(path):
    data=path.read_bytes();magic,count=struct.unpack_from('<2I',data);offset=8;result=[]
    assert magic in (0x38514353,0x39514353)
    stride=60 if magic==0x39514353 else 52
    for _ in range(count):
        textures=[]
        for _ in range(4):
            n=struct.unpack_from('<I',data,offset)[0];offset+=4
            textures.append(data[offset:offset+n].decode());offset+=n
        n=struct.unpack_from('<I',data,offset)[0];offset+=4
        vertices=np.frombuffer(data[offset:offset+n*stride],dtype=np.uint8).reshape(n,stride);offset+=n*stride
        result.append({'textures':textures,'vertices':n,'original_52_byte_attributes_sha256':hashlib.sha256(vertices[:,:52].tobytes()).hexdigest()})
    assert offset==len(data)
    return result


def compare(before,after,name,zooms=(1,2),disabled=False):
    rows=[]
    for hour in (12,0):
        for zoom in zooms:
            filename=f'h{hour:02}-z{zoom}-pan00.png'
            p=OUT/f'city-scene-r{before}'/name/'combined'/filename
            q=OUT/f'city-scene-r{after}'/name/'combined'/filename
            a=np.asarray(Image.open(p).convert('RGB')).astype(int)
            b=np.asarray(Image.open(q).convert('RGB')).astype(int)
            assert a.shape==b.shape
            delta=np.abs(a-b).max(2);outside=delta.copy()
            outside[270//zoom:530//zoom,270//zoom:600//zoom]=0
            assert outside.max()<=1,'unrelated scene change'
            if disabled:assert delta.max()<=1 and (delta>0).sum()<=1
            ys,xs=np.where(delta>2)
            rows.append({'before':before,'after':after,'hour':hour,'zoom':zoom,
                         'changed_pixels_gt_2':len(xs),'any_changed_pixels':int((delta>0).sum()),
                         'max_channel_delta':int(delta.max()),'outside_city_roi_max':int(outside.max()),
                         'bounds':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
                         'image_sha256':sha(q)})
    return rows


def main():
    baseline=augmentation(8,MEDIEVAL)
    for revision in (24,25,26,28):
        a=augmentation(revision,MEDIEVAL)
        assert placement(a)==placement(baseline)
        for key in ('source_biq_sha256','anchor_tile','projection','pool','size','emissive_gain','emissive_uv','hdr_glow'):
            assert a[key]==baseline[key],key
    original=wire_prefix(FIX/f'city-scene-r8/{MEDIEVAL}/city.bin')
    extended=wire_prefix(FIX/f'city-scene-r25/{MEDIEVAL}/city.bin')
    assert original==extended,'auxiliary attribute changed original geometry/material bindings'
    assert placement(augmentation(23,ANCIENT))==placement(augmentation(27,ANCIENT))
    parity=[read(OUT/folder/'evidence.json') for folder in
            ('city-scene-r23/windows-capital','city-scene-r26/windows-combined-ao',
             'city-scene-r27/windows-inland-ao','city-scene-r28/windows-final-material')]
    assert [len(p['results']) for p in parity]==[4,4,2,4]
    assert all(r['metrics']['pass'] for p in parity for r in p['results'])
    holdout=augmentation(27,'european-medieval-s1-inland-at7-4')
    assert holdout['benchmark_region']=='inland' and holdout['anchor_tile']==[7,4] and holdout['ao_uv']==1
    evidence={'classification':'Partial city material/composition improvement; no overall city acceptance or promotion',
              'coordinate_roles':{'uv0':'diffuse','uv1':'source AO atlas alignment supported for tested medieval bodies',
                                  'uv2':'retained previously verified light atlas'},
              'legacy_geometry_prefix_identical':original,
              'disabled_control':compare(8,25,MEDIEVAL,disabled=True),
              'ao_only':compare(25,24,MEDIEVAL),'ground_on_ao':compare(24,26,MEDIEVAL),
              'source_addressing':compare(26,28,MEDIEVAL),'combined_vs_previous_ground':compare(18,28,MEDIEVAL),
              'ancient_composition':compare(13,23,ANCIENT),'ancient_ao':compare(23,27,ANCIENT,zooms=(1,)),
              'holdout':{'name':'inland','site':[7,4],'selection':'fixtures/beauty/city-scene-foundation/inland-site-survey/selection.json',
                         'scope':'New to city/material tuning, existing fixed terrain benchmark; now a regression witness.'},
              'standalone_windows_parity':parity,'free_disk_gib':round(shutil.disk_usage(V2).free/1024**3,2),
              'remaining':['Exact source LEAN/tangent/gloss response','Broad ground coverage and grounding-material state selection',
                           'All cultures/eras/sizes and full geometry clearance','Local night light transport and full city-reflection coverage',
                           'Human visual checkpoint and all existing milestone gates']}
    target=V2/'audits/beauty/CITY_AO_r28_EVIDENCE.json';target.write_text(json.dumps(evidence,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
