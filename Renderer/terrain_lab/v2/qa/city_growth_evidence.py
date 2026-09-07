"""Verify preserved city growth, terrain clearance and the fixed shadow comparison."""
import json
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image,ImageDraw
from city_source_surface_evidence import ROOT,V2,OUT,FIX,read,sha

WILD='american-modern-s1-wilderness-at6-6'
CASES=[(46,WILD),(47,WILD.replace('-s1-','-s0-')),
       (48,'american-modern-s2-inland-at7-4'),(49,'american-modern-s1-freshshadow-at5-3')]


def placement(i):return {k:i[k] for k in ('asset','slot','offset','rotation','scale','local_bounds')}


def clearance(a):
    site=a['benchmark_region']+'-'+'-'.join(map(str,a['anchor_tile']))
    grid=read(FIX/'city-scene-foundation'/site/'surface.json')['samples']
    boxes=[];rows=[]
    for i in a['instances']:
        raw=[v+i['offset'][j%2] for j,v in enumerate(i['local_bounds'])]
        padded=[v+(-.012 if j<2 else .012) for j,v in enumerate(raw)]
        assert max(abs(v) for v in padded)<=a['footprint_half_extent_tiles']+1e-8
        for other in boxes:
            assert not (padded[0]<other[2] and padded[2]>other[0] and padded[1]<other[3] and padded[3]>other[1])
        boxes.append(raw)
        low=[math.floor((padded[j]-.12+1)/.04) for j in range(2)]
        high=[math.ceil((padded[j+2]+.12+1)/.04) for j in range(2)]
        assert min(low)>=0 and max(high)<=50
        hits=sum(grid[y*51+x]['real'] in (7,8) for y in range(low[1],high[1]+1) for x in range(low[0],high[0]+1))
        assert hits==0
        assert i['minimum_shore_distance']<=-.02 and max(i['ground_height_range'])-min(i['ground_height_range'])<=3
        rows.append({'slot':i['slot'],'vegetation_samples_in_margin':hits,'no_body_overlap':True})
    return rows


def pixels(before,after):
    result=[]
    for hour in (12,0):
        name=f'h{hour:02}-z1-pan00.png'
        a=np.asarray(Image.open(before/name).convert('RGB')).astype(int)
        b=np.asarray(Image.open(after/name).convert('RGB')).astype(int)
        assert a.shape==b.shape==(800,1360,3)
        delta=np.abs(a-b).max(2);ys,xs=np.where(delta>2)
        outside=delta.copy();outside[245:450,755:1000]=0
        assert outside.max()==0,'unrelated pixels changed after fixed-frame comparison'
        result.append({'hour':hour,'changed_pixels_gt_2':len(xs),
                       'bounds':[int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())],
                       'outside_city_roi_max':int(outside.max()),'image_sha256':sha(after/name)})
    return result


def main():
    cases={rev:read(FIX/f'city-scene-r{rev}'/name/'augmentation.json') for rev,name in CASES}
    before=read(FIX/f'city-scene-r37/{WILD}/augmentation.json')
    current=cases[46]
    for key in ('pool','size','uniform_scale_factor','source_biq_sha256','anchor_tile','projection',
                'emissive_gain','emissive_uv','source_normals','source_surface','extra_materials','grounding'):
        assert before[key]==current[key],key
    assert [(i['asset'],i['scale']) for i in before['instances']]==[(i['asset'],i['scale']) for i in current['instances']]
    assert [placement(i) for i in cases[47]['instances']]==[placement(i) for i in current['instances'][:4]]
    assert current['layout_attempts'][0]['planned_instances']==cases[47]['layout_attempts'][0]['planned_instances']
    checks={}
    for rev,name in CASES:
        a=cases[rev];s=read(FIX/f'city-scene-r{rev}'/name/'surface.json')
        assert s['region']['region']['extent']==[10,10]
        checks[str(rev)]={'instances':len(a['instances']),'clearance':clearance(a),
                          'region':a['benchmark_region'],'terrain_sha256':s['terrain_sha256'],
                          'search':a['layout_attempts'],'augmentation_sha256':sha(FIX/f'city-scene-r{rev}'/name/'augmentation.json')}
    controls=OUT/'city-growth-r1/r46-fixed-shadow-frame'
    parity=[read(OUT/f'city-scene-r{rev}/windows-growth/evidence.json') for rev,_ in CASES]
    parity.append(read(OUT/'city-growth-r1/windows-fixed-shadow-frame/evidence.json'))
    assert [len(p['results']) for p in parity]==[2]*5
    assert all(row['metrics']['pass'] for p in parity for row in p['results'])
    packets=[]
    with tempfile.TemporaryDirectory(prefix='city-frame-check-') as temporary:
        binary=Path(temporary)/'contract'
        subprocess.run(['clang++','-std=c++17','-O2',str(V2/'qa/city_shadow_frame_contract.cpp'),'-o',str(binary)],check=True)
        for index in range(2):
            args=[OUT/f'city-scene-r46/{WILD}/combined-{index}.packet',
                  OUT/f'city-scene-r37/{WILD}/combined-{index}.packet',controls/f'combined-{index}.packet']
            row=json.loads(subprocess.check_output([str(binary),*map(str,args)],text=True))
            row['packet_sha256']=[sha(p) for p in args];packets.append(row)
    rejected={str(rev):read(next((FIX/f'city-scene-r{rev}').glob('*/growth-search.json'))) for rev in (40,41,42,44,45)}
    noop=read(V2/'audits/beauty/CITY_GROWTH_SHADOW_NOOP.json')
    assert noop['source_sha256']==sha(V2/'systems/lighting/scene_shadow.cpp')
    assert all(noop[mode]['pass'] and noop[mode]['identical_frame_texture_check'] for mode in ('default','reference'))
    evidence={'classification':'Provisional wilderness layout/readability improvement; broader city quality and promotion remain open',
              'fixed_frame_pixels':pixels(OUT/f'city-scene-r37/{WILD}/combined',controls/'render'),
              'stable_growth_prefix':True,'cases':checks,'packet_frame_contracts':packets,'shadow_default_and_reference_noop':noop,
              'standalone_windows_parity':parity,'failed_searches':rejected,
              'untuned_case':{'revision':49,'region':'freshshadow','anchor':[5,3],
                              'selection':'Selected before city rendering from raw terrain cells; no city tuning on this region previously',
                              'result':'Clearance passes, skyline remains crowded; not general visual acceptance',
                              'terrain_baseline':'shadow-receiver-r1; existing distinct natural benchmark retained'},
              'remaining':['Wilderness eleven-body fit','General skyline visibility and urban ground composition',
                           'Facade environment specular and local night light pools','Full culture/era/size matrix and capital coverage',
                           'Native stable shadow envelope strategy and all milestone/human gates']}
    target=V2/'audits/beauty/CITY_GROWTH_r40_r49_EVIDENCE.json';target.write_text(json.dumps(evidence,indent=2)+'\n')
    # Native-size city crops; no source-derived reference images are duplicated.
    canvas=Image.new('RGB',(700,490),(30,32,34));draw=ImageDraw.Draw(canvas)
    for column,(folder,label) in enumerate([(OUT/f'city-scene-r37/{WILD}/combined','Previous: forest overlap'),(controls/'render','Candidate: separated skyline, fixed shadow grid')]):
        for row,hour in enumerate((12,0)):
            im=Image.open(folder/f'h{hour:02}-z1-pan00.png').convert('RGB').crop((710,240,1060,460))
            canvas.paste(im,(column*350,row*245+23));draw.text((column*350+5,row*245+5),f'{label} | {hour:02}:00',fill='white')
    canvas.save(OUT/'city-growth-r1/selected-native-comparison.png')
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
