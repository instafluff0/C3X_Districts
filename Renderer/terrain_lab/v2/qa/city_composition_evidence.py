"""Recheck r21 capital growth/visibility and the composed r22 paving result."""
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from city_ground_capital_evidence import pixels

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'
FIX=V2/'fixtures/beauty'
NAME='american-modern-s1-capital-at7-5'
CONTROL=NAME.replace('-capital','-capital-control')


def read(path):return json.loads(path.read_text())
def augmentation(revision,name=NAME):return read(FIX/f'city-scene-r{revision}'/name/'augmentation.json')
def image(revision,name,hour):
    return np.asarray(Image.open(OUT/f'city-scene-r{revision}'/name/'combined'/f'h{hour:02}-z1-pan00.png').convert('RGB')).astype(int)
def identity(instance):return {k:instance[k] for k in ('asset','slot','scale','rotation')}
def placement(instance):return {k:instance[k] for k in ('asset','slot','scale','rotation','offset','local_bounds')}


def main():
    prior,current,combined=[augmentation(r) for r in (19,21,22)]
    for candidate in (current,combined):
        assert [identity(i) for i in candidate['instances']]==[identity(i) for i in prior['instances']]
        for key in ('source_biq_sha256','anchor_tile','projection','pool','size','emissive_gain','emissive_uv','hdr_glow'):
            assert candidate[key]==prior[key],key
    assert [placement(i) for i in combined['instances']]==[placement(i) for i in current['instances']]
    visibility=[]
    for revision in (19,21):
        a=augmentation(revision);control=augmentation(revision,CONTROL)
        assert [i for i in a['instances'] if i['slot']!='capital']==control['instances']
        for hour in (12,0):
            delta=np.abs(image(revision,NAME,hour)-image(revision,CONTROL,hour)).max(2)
            visibility.append({'revision':revision,'hour':hour,'roi':[850,405,907,456],
                               'palace_area_effect_pixels_gt_2':int((delta[405:456,850:907]>2).sum()),
                               'whole_scene_effect_pixels_gt_2':int((delta>2).sum()),
                               'interior_lake_max_channel_delta':int(delta[491:501,858:885].max()),
                               'interpretation':'Localized palace/control image contribution, not a geometric visible-surface percentage.'})
    growth=[];previous=[];palace=None
    for size in (0,1,2):
        a=augmentation(21,f'american-modern-s{size}-capital-at7-5')
        houses=[placement(i) for i in a['instances'] if i['slot']!='capital']
        focal=next(placement(i) for i in a['instances'] if i['slot']=='capital')
        assert houses[:len(previous)]==previous
        assert palace is None or focal==palace
        assert len(houses)==(4,7,11)[size]
        growth.append({'size':size,'houses':len(houses),'stable_prefix':True,'fixed_palace':True})
        previous=houses;palace=focal
    holdout_name='american-modern-s1-capital-freshcanopy-at5-4'
    holdout=augmentation(21,holdout_name)
    assert holdout['benchmark_region']=='freshcanopy' and holdout['anchor_tile']==[5,4]
    assert [identity(i) for i in holdout['instances']]==[identity(i) for i in current['instances']]
    parity=[read(OUT/folder/'evidence.json') for folder in ('city-scene-r21/windows-capital','city-scene-r21/windows-freshcanopy','city-scene-r22/windows-combined')]
    assert [len(p['results']) for p in parity]==[4,2,4]
    assert all(row['metrics']['pass'] for p in parity for row in p['results'])
    result={'classification':'Provisional American modern capital composition improvement; full city goal and all gates remain open',
            'single_era':True,'matched_source_bodies':8,'palace_visibility':visibility,'growth':growth,
            'layout_pixels':pixels(OUT/'city-scene-r19'/NAME/'combined',OUT/'city-scene-r21'/NAME/'combined',(725,290,1085,580)),
            'composed_paving_pixels':pixels(OUT/'city-scene-r21'/NAME/'combined',OUT/'city-scene-r22'/NAME/'combined',(780,400,960,490)),
            'paving_ground_clipping':[i.get('ground_clipping',[]) for i in combined['instances']],
            'holdout':{'name':holdout_name,'scope':'Previously unused for city tuning; existing natural-scene fixture.',
                       'site_selection':'fixtures/beauty/city-scene-foundation/freshcanopy-site-survey/selection.json',
                       'same_asset_scale_order_recipe':True,'full_clearance_acceptance':False},
            'standalone_windows_parity':parity,'free_disk_gib':round(shutil.disk_usage(V2).free/1024**3,2),
            'remaining':['Other eras/cultures and capital mappings','Full fixed-region and clearance coverage',
                         'Broad ground coverage and complete material appearance','Local night light transport',
                         'Human visual checkpoint and all existing milestone gates']}
    target=V2/'audits/beauty/CITY_COMPOSITION_r21_r22_EVIDENCE.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
