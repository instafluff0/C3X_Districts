"""Verify local city-light composition using matched images and unchanged packets."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
from city_source_surface_evidence import ROOT,V2,OUT,FIX,read,sha

CASES={
 'wilderness':('city-growth-r1/r46-fixed-shadow-frame/render','city-scene-r46/american-modern-s1-wilderness-at6-6',(790,310,925,420),(1,)),
 'medieval':('city-scene-r31/european-medieval-s1/combined','city-scene-r31/european-medieval-s1',(340,365,520,475),(1,2)),
 'inland':('city-scene-r48/american-modern-s2-inland-at7-4/combined','city-scene-r48/american-modern-s2-inland-at7-4',(710,390,890,500),(1,)),
 'freshshadow':('city-scene-r49/american-modern-s1-freshshadow-at5-3/combined','city-scene-r49/american-modern-s1-freshshadow-at5-3',(565,385,720,495),(1,)),
 'capital':('city-scene-r22/american-modern-s1-capital-at7-5/combined','city-scene-r22/american-modern-s1-capital-at7-5',(765,380,975,510),(1,2)),
}
R3=OUT/'city-facade-light-r3'


def rgb(path):return np.array(Image.open(path).convert('RGB')).astype(int)


def delta(before,after,box=None):
    difference=abs(rgb(before)-rgb(after)).max(2)
    if box is not None:
        x0,y0,x1,y1=box;difference=difference[y0:y1,x0:x1]
    ys,xs=np.where(difference>2)
    return {'changed_pixels_any':int((difference>0).sum()),'changed_pixels_gt_2':len(xs),
            'max_channel_delta':int(difference.max()),
            'bounds':([int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())] if len(xs) else None)}


def main():
    cases={};parity=[]
    for name,(baseline,fixture,roi,zooms) in CASES.items():
        before=OUT/baseline;after=R3/name/'render'
        a=read(before/'report.json');b=read(after/'report.json')
        assert a['packets']==b['packets'] and a['postprocess']==b['postprocess']
        data=read(R3/name/'lights.json');augmentation=read(FIX/fixture/'augmentation.json')
        assert data['augmentation_sha256']==sha(FIX/fixture/'augmentation.json')
        assert data['gain']==4 and len(data['blockers'])==len(augmentation['instances'])
        assert 0<len(data['lights'])<=48
        for path,digest in data['texture_sha256'].items():assert sha(ROOT/path)==digest
        for light in data['lights']:
            assert .32<=light['range']<=.55 and light['sample_count']>0
            assert np.isfinite(light['position']+light['color_linear']+[light['intensity']]).all()
            owner=data['blockers'][light['owner']];axis=int(np.argmax(abs(np.array(light['direction']))));sign=light['direction'][axis]
            boundary=owner['high' if sign>0 else 'low'][axis]
            assert abs(light['position'][axis]-(boundary+sign*.012))<1e-7
        frames=[]
        for zoom in zooms:
            for hour in (12,0):
                file=f'h{hour:02}-z{zoom}-pan00.png';d=delta(before/file,after/file)
                difference=abs(rgb(before/file)-rgb(after/file)).max(2)
                x0,y0,x1,y1=[v//zoom for v in roi];difference[y0:y1,x0:x1]=0
                assert difference.max()<=1
                if hour==12:assert d['max_channel_delta']<=1 and d['changed_pixels_any']<=1
                else:assert d['changed_pixels_gt_2']>100
                frames.append({'hour':hour,'zoom':zoom,**d,'outside_local_roi_max':int(difference.max()),'sha256':sha(after/file)})
        windows=read(R3/f'windows-{name}/evidence.json');assert len(windows['results'])==2*len(zooms)
        assert all(row['metrics']['pass'] for row in windows['results']);parity.append(windows)
        cases[name]={'frames':frames,'lights':len(data['lights']),'blockers':len(data['blockers']),
                     'light_data_sha256':sha(R3/name/'lights.json'),'same_geometry_material_shadow_packets':True}
    disabled=[];culling=[]
    for hour in (12,0):
        file=f'h{hour:02}-z1-pan00.png'
        d=delta(OUT/CASES['wilderness'][0]/file,OUT/'city-facade-light-r2/disabled/render'/file)
        assert d['max_channel_delta']==0;disabled.append({'hour':hour,**d})
        d=delta(OUT/'city-facade-light-r2/wilderness/render'/file,R3/'wilderness/render'/file)
        assert d['max_channel_delta']==0;culling.append({'hour':hour,**d})
    blocked=delta(R3/'wilderness/render/h00-z1-pan00.png',R3/'wilderness-no-blockers/render/h00-z1-pan00.png')
    assert blocked['changed_pixels_gt_2']>0
    lake=(858,491,885,501)
    reflection=delta(R3/'capital/render/h00-z1-pan00.png',R3/'capital-no-reflection/render/h00-z1-pan00.png',lake)
    spill=delta(OUT/CASES['capital'][0]/'h00-z1-pan00.png',R3/'capital/render/h00-z1-pan00.png',lake)
    assert reflection['changed_pixels_gt_2']>0 and reflection['max_channel_delta']>10
    source=read(OUT/'city-source-expanded-r1/build.json')
    socket_evidence=[{'asset':row['asset_id'],'attachments':row['attachments']['points']} for row in source['assets']
                     if row['asset_id'] in ('city/component/201c181c2d3d95c2','city/component/81e6bc964c4f7c5a')]
    result={'classification':'Provisional local night-light spill improvement; source-derived proxy inputs with authored transport/calibration, not recovered source light binding',
            'cases':cases,'disabled_control':disabled,'envelope_culling_control':culling,
            'building_occlusion_control':blocked,'capital_lake_roi':lake,'capital_object_reflection_control':reflection,
            'capital_spill_delta_in_lake':{'metrics':spill,'interpretation':'Combined water illumination and reflection response; not isolated reflected spill'},
            'source_rooftop_socket_evidence':socket_evidence,'standalone_windows_parity':parity,
            'holdout':'Freshshadow had no local-light tuning; same gain/range/occlusion policy as the other cases',
            'limitations':['Bounded quadrature and cardinal facade aggregation are approximations','City AABB occluders, not exact mesh or terrain local-light shadows',
                           'Environment specular and coherent urban ground still incomplete','Full culture/era/size/capital coverage and native performance/gates remain open']}
    target=V2/'audits/beauty/CITY_FACADE_LIGHT_r1_r3_EVIDENCE.json';target.write_text(json.dumps(result,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
