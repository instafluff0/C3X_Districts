"""Verify composed city surface probes and controls; visual acceptance stays separate."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'
FIX=V2/'fixtures/beauty'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(before,after,zooms=(1,2),roi=(270,270,600,530),exact=False):
    rows=[]
    for hour in (12,0):
        for zoom in zooms:
            name=f'h{hour:02}-z{zoom}-pan00.png';p=before/name;q=after/name
            a=np.asarray(Image.open(p).convert('RGB')).astype(int)
            b=np.asarray(Image.open(q).convert('RGB')).astype(int)
            assert a.shape==b.shape
            delta=np.abs(a-b).max(2);outside=delta.copy()
            left,top,right,bottom=(v//zoom for v in roi);outside[top:bottom,left:right]=0
            assert outside.max()<=1,'unrelated terrain or object pixels changed'
            if exact:assert delta.max()==0,'disabled material changed prior image'
            ys,xs=np.where(delta>2)
            rows.append({'hour':hour,'zoom':zoom,'changed_pixels_gt_2':len(xs),
                         'max_channel_delta':int(delta.max()),'outside_city_roi_max':int(outside.max()),
                         'bounds':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None,
                         'image_sha256':sha(q)})
    return rows


def main():
    medieval='european-medieval-s1';modern='american-modern-s1-at7-5';inland=medieval+'-inland-at7-4'
    for before,after,name in ((29,31,medieval),(18,32,modern),(30,33,inland)):
        a=read(FIX/f'city-scene-r{before}'/name/'augmentation.json')
        b=read(FIX/f'city-scene-r{after}'/name/'augmentation.json')
        # r18 predates the ground-clipping audit field; body placement is fixed.
        def body_records(value):return [{k:v for k,v in i.items() if k!='ground_clipping'} for i in value['instances']]
        assert body_records(a)==body_records(b),'city placement changed'
        if before!=18:assert a['instances']==b['instances']
        # The older modern augmentation has no benchmark_region label. Compare
        # its actual sampled region metadata instead of inventing that label.
        assert read(FIX/f'city-scene-r{before}'/name/'surface.json')['region']==read(FIX/f'city-scene-r{after}'/name/'surface.json')['region']
        for key in ('source_biq_sha256','anchor_tile','projection','pool','size',
                    'uniform_scale_factor','emissive_gain','emissive_uv','hdr_glow','source_addressing','grounding'):
            assert a[key]==b[key],key
        assert all(b['capital'][k]==v for k,v in a['capital'].items())
        assert not b['capital']['requested'] and not b['capital']['drawn']
        assert not b['source_normals']['unmapped'] and b['source_surface']=='lit'
        assert sha(ROOT/b['source_normals']['mapping'])==b['source_normals']['sha256']
    source=OUT/f'city-scene-r31/{medieval}'
    full=source/'combined';off=source/'surface-control-off/render';normal=source/'surface-control-normal/render'
    baseline=OUT/f'city-scene-r29/{medieval}/combined'
    report=read(full/'report.json')
    for control in (off,normal):
        r=read(control/'report.json')
        assert r['packets']==report['packets'] and r['postprocess']==report['postprocess']
    parity=[read(OUT/f'city-scene-r{rev}'/folder/'evidence.json') for rev,folder in
            ((31,'windows-source-surface'),(32,'windows-source-surface'),(33,'windows-inland-surface'))]
    assert [len(p['results']) for p in parity]==[4,4,2]
    assert all(r['metrics']['pass'] for p in parity for r in p['results'])
    evidence={'classification':'Partial source material restoration with modest pixel gains; no complete city acceptance or milestone promotion',
              'source_shader_evidence':'CITY_SHADER_MATERIAL_SOURCE.json',
              'disabled_control':compare(baseline,off,exact=True),
              'normal_texture_only':compare(off,normal),'direct_specular_only':compare(normal,full),
              'combined_medieval':compare(baseline,full),
              'combined_modern':compare(OUT/f'city-scene-r18/{modern}/combined',OUT/f'city-scene-r32/{modern}/combined',roi=(650,350,1050,610)),
              'inland':compare(OUT/f'city-scene-r30/{inland}/combined',OUT/f'city-scene-r33/{inland}/combined',zooms=(1,),roi=(710,410,910,530)),
              'standalone_windows_parity':parity,
              'remaining':['Exact active source shader permutation/constants','Scalar variance scale and metalness intake',
                           'Filtered environment reflection and local city lights','Coherent ground coverage',
                           'Complete culture/era/size and fixed wilderness city coverage','All existing human and milestone gates']}
    target=V2/'audits/beauty/CITY_SOURCE_SURFACE_r31_r33_EVIDENCE.json';target.write_text(json.dumps(evidence,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
