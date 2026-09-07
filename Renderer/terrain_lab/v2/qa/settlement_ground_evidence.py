"""Validate selected settlement-ground composition and retain rejected evidence."""
import json
import math
import struct
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from city_facade_light_evidence import CASES,OUT,V2,ROOT,read,sha,rgb,delta
from city_scene_pass import executable,Cache

BASE=OUT/'city-settlement-ground-r2'
SELECTED=['inland','wilderness','freshshadow','capital','small']


def main():
    cases={};native=[];isolation={}
    exe=executable(V2/'qa/settlement_ground_contract.cpp',Cache(V2/'app/.cache'))
    for name in SELECTED+['medieval']:
        data=read(BASE/name/'settlement.json');surface=read(BASE/name/'surface.json')
        for key in ('augmentation','ground_parts'):
            assert sha(ROOT/data[key])==data[key+'_sha256']
        assert sha(ROOT/data['atlas']['texture'])==data['atlas']['sha256']
        assert sha(BASE/name/'ground.bin')==data['ground_sha256']
        assert sha(BASE/name/'surface.json')==data['surface_sha256']
        assert surface['region']['region']['extent']==[10,10]
        wire=(BASE/name/'ground.bin').read_bytes();magic,count=struct.unpack_from('<II',wire)
        assert magic==0x31524753 and len(wire)==8+count*52
        vertices=np.frombuffer(wire,dtype='<f4',offset=8).reshape(-1,13).astype(float)
        assert np.max(abs(vertices[:,3:5]-vertices[:,9:11]/np.array(data['tile_period'])))<1e-5
        cells={(int(s['column']),int(s['row'])):(s['base'],s['real']) for s in surface['samples']}
        for triangle in vertices.reshape(-1,3,13):
            point=triangle[:,9:11].mean(0);cell=tuple(math.floor(v) for v in point)
            # Degenerate clipped-edge triangles have no area to classify.
            a,b,c=triangle[:,9:11];area=abs(np.linalg.det(np.array([b-a,c-a])))
            if area<1e-10:continue
            assert cells[cell][0]<11 and cells[cell][1] not in (7,8)
        checks=[]
        for packet in data['packets']:
            for key in ('original','output'):assert sha(ROOT/packet[key])==packet[key+'_sha256']
            checks.append(json.loads(subprocess.check_output([str(exe),str(ROOT/packet['original']),str(ROOT/packet['output']),str(packet['insertion_draw'])],text=True)))
        isolation[name]=checks
        before=OUT/(f'city-ground-binding-r1/{name}/render' if name!='small' else 'city-settlement-ground-r2/small-binding-baseline/render')
        after=BASE/name/'render';roi=CASES['wilderness' if name=='small' else name][2]
        zooms=(1,2) if name in ('capital','medieval') else (1,)
        assert read(before/'report.json')['postprocess']==read(after/'report.json')['postprocess']
        frames=[]
        for zoom in zooms:
            for hour in (12,0):
                file=f'h{hour:02}-z{zoom}-pan00.png';d=delta(before/file,after/file)
                difference=abs(rgb(before/file)-rgb(after/file)).max(2)
                x0,y0,x1,y1=[v//zoom for v in roi];difference[y0:y1,x0:x1]=0
                assert difference.max()<=1 and (difference>0).sum()<=1
                assert d['changed_pixels_gt_2']>10
                frames.append({'hour':hour,'zoom':zoom,**d,'outside_city_roi_max':int(difference.max()),
                               'outside_city_roi_changed_pixels':int((difference>0).sum()),'sha256':sha(after/file)})
        if name in SELECTED:
            windows=read(BASE/f'windows-{name}/evidence.json');assert len(windows['results'])==len(frames)
            for row in windows['results']:
                assert row['metrics']['pass']
                for key,path in [('d3d11_sha256',BASE/f'windows-{name}'/row['frame']),('shader_sha256',after/'shaders/source.hlsl'),
                                 ('reflection_sha256',after/'shaders/reflection/source.hlsl'),('post_sha256',after/'postprocess/source.hlsl')]:
                    assert row[key]==sha(path)
            native.append(windows)
        cases[name]={'selected_local_candidate':name in SELECTED,'frames':frames,'geometry':data,
                     'interpretation':('Connected modern paving; city quality remains open' if name in SELECTED else 'Rejected: softer edge still reads as a flat orange platform')}
    small=cases['small']['geometry'];medium=cases['wilderness']['geometry']
    assert small['tile_period']==medium['tile_period'] and small['boxes']==medium['boxes'][:4]
    disabled=[]
    assert read(BASE/'disabled/settlement.json')['ground_sha256']==cases['inland']['geometry']['ground_sha256']
    for hour in (12,0):
        file=f'h{hour:02}-z1-pan00.png';d=delta(OUT/'city-ground-binding-r1/inland/render'/file,BASE/'disabled/render'/file)
        assert d['max_channel_delta']==0;disabled.append({'hour':hour,**d})
    lake=delta(OUT/'city-ground-binding-r1/capital/render/h00-z1-pan00.png',BASE/'capital/render/h00-z1-pan00.png',(858,491,885,501))
    assert lake['max_channel_delta']==0
    result={'classification':'Provisional modern settlement-ground improvement; medieval extension rejected; no overall city or gate approval',
            'cases':cases,'packet_isolation':isolation,'standalone_windows_parity':native,'disabled_control':disabled,
            'stable_small_medium_ground_coordinates':True,'capital_lake_roi_unchanged':lake,
            'source_height_followup':read(BASE/'height-intake.json'),
            'limitations':['Authored footprint union, not recovered city-generator ground geometry','Ground height/blend and source state semantics unresolved',
                           'Full route/river mesh clearance not established','Crowded holdout and wilderness eleven-body fit unresolved',
                           'Broader culture/era/size/capital quality and native delivery remain open']}
    target=V2/'audits/beauty/CITY_SETTLEMENT_GROUND_r1_r2_EVIDENCE.json';target.write_text(json.dumps(result,indent=2)+'\n')
    sheet=Image.new('RGB',(760,480),(25,25,25));draw=ImageDraw.Draw(sheet)
    for row,(name,roi) in enumerate([('inland',(650,310,1030,530)),('capital',(735,280,1115,500))]):
        for col,(series,label) in enumerate([('city-ground-binding-r1','Previous'),('city-settlement-ground-r2','Connected paving')]):
            sheet.paste(Image.open(OUT/series/name/'render/h12-z1-pan00.png').convert('RGB').crop(roi),(col*380,row*240+20))
            draw.text((col*380+5,row*240+4),name+' | '+label,fill='white')
    sheet.save(BASE/'selected-native-comparison.png');print(target.relative_to(ROOT))


if __name__=='__main__':main()
