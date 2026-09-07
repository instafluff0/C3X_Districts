"""Recheck the isolated, unpromoted packed-source-normal city experiment."""
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'
FIX=V2/'fixtures/beauty'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def wire_without_normals(path):
    data=path.read_bytes();magic,count=struct.unpack_from('<2I',data);offset=8;result=[]
    assert magic==0x39514353
    for _ in range(count):
        textures=[]
        for _ in range(4):
            n=struct.unpack_from('<I',data,offset)[0];offset+=4
            textures.append(data[offset:offset+n].decode());offset+=n
        n=struct.unpack_from('<I',data,offset)[0];offset+=4
        vertices=np.frombuffer(data[offset:offset+n*60],dtype=np.uint8).reshape(n,60);offset+=n*60
        # Position, all coordinate sets, material selector and world data must be identical.
        stable=np.concatenate((vertices[:,:20],vertices[:,32:]),axis=1)
        result.append({'textures':textures,'vertices':n,'non_normal_sha256':hashlib.sha256(stable.tobytes()).hexdigest()})
    assert offset==len(data)
    return result


def case(before,after,name,zooms,roi):
    p=FIX/f'city-scene-r{before}'/name;q=FIX/f'city-scene-r{after}'/name
    a=read(p/'augmentation.json');b=read(q/'augmentation.json')
    assert all(a[k]==v for k,v in b.items() if k!='source_normals'), 'non-normal scene setting changed'
    assert not b['source_normals']['unmapped']
    assert sha(ROOT/b['source_normals']['mapping'])==b['source_normals']['sha256']
    assert wire_without_normals(p/'city.bin')==wire_without_normals(q/'city.bin')
    ar=read(OUT/f'city-scene-r{before}'/name/'combined/report.json')
    br=read(OUT/f'city-scene-r{after}'/name/'combined/report.json')
    assert ar['shader_closure_sha256']==br['shader_closure_sha256']
    assert ar['postprocess']==br['postprocess'] and ar['reflection']==br['reflection']
    rows=[]
    for hour in (12,0):
        for zoom in zooms:
            filename=f'h{hour:02}-z{zoom}-pan00.png'
            p=OUT/f'city-scene-r{before}'/name/'combined'/filename
            q=OUT/f'city-scene-r{after}'/name/'combined'/filename
            x=np.asarray(Image.open(p).convert('RGB')).astype(int)
            y=np.asarray(Image.open(q).convert('RGB')).astype(int)
            assert x.shape==y.shape
            delta=np.abs(x-y).max(2);outside=delta.copy()
            left,top,right,bottom=(v//zoom for v in roi);outside[top:bottom,left:right]=0
            assert outside.max()<=1,'unrelated scene changed'
            ys,xs=np.where(delta>2)
            rows.append({'hour':hour,'zoom':zoom,'changed_pixels_gt_2':len(xs),
                         'max_channel_delta':int(delta.max()),'outside_city_roi_max':int(outside.max()),
                         'bounds':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
                         'image_sha256':sha(q)})
    return {'before':before,'after':after,'name':name,'frames':rows,
            'non_normal_wire_data_unchanged':True,'shader_and_lighting_unchanged':True}


def main():
    source=read(FIX/'city-tangent-source-r1/medieval-normals.json')
    assert all((r['profile'],r['stride'],r['negative_geometric_dots'])==(0x315CFCD9,24,0) for r in source['evidence'])
    parity=[read(OUT/f'city-scene-r{revision}'/folder/'evidence.json') for revision,folder in
            ((29,'windows-source-normals'),(30,'windows-inland-normals'))]
    assert [len(p['results']) for p in parity]==[4,2]
    assert all(r['metrics']['pass'] for p in parity for r in p['results'])
    evidence={'classification':'Diagnostic candidate; subtle pixel changes, no new visual acceptance or promotion',
              'source_meshes':len(source['meshes']),'source_primitive_records':len(source['evidence']),
              'minimum_geometric_dot':min(r['minimum_geometric_dot'] for r in source['evidence']),
              'minimum_primitive_mean_dot':min(r['mean_geometric_dot'] for r in source['evidence']),
              'negative_geometric_dots':0,'verified_profile':'0x315CFCD9 / stride 24',
              'coastal':case(28,29,'european-medieval-s1',(1,2),(270,270,600,530)),
              'inland':case(27,30,'european-medieval-s1-inland-at7-4',(1,),(710,410,910,530)),
              'standalone_windows_parity':parity,
              'remaining':['Tangent/bitangent interpretation','LEAN normal textures and gloss response',
                           'Other profiles/cultures/eras and fixed wilderness city witness',
                           'Broad ground coverage and local night light transport','Human visual checkpoint and existing milestone gates']}
    target=V2/'audits/beauty/CITY_SOURCE_NORMAL_r29_r30_EVIDENCE.json'
    target.write_text(json.dumps(evidence,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
