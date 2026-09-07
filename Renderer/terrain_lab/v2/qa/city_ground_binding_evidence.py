"""Check matched city-ground pixels, frozen shaders and standalone Windows parity."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from city_facade_light_evidence import CASES,OUT,V2,ROOT,read,sha,rgb,delta

BASE=OUT/'city-ground-binding-r1'


def main():
    cases={};windows=[]
    for name,(_,_,roi,zooms) in CASES.items():
        before=OUT/f'city-facade-light-r3/{name}/render';after=BASE/name/'render'
        old=read(before/'report.json');new=read(after/'report.json')
        for key in ('shader_closure_sha256','reflection','postprocess'):
            assert old[key]==new[key],key
        bindings=read(BASE/name/'binding.json')
        assert sha(ROOT/bindings['mapping'])==bindings['mapping_sha256']
        for packet in bindings['packets']:
            assert packet['pass'] and packet['all_buffers_unchanged']
            assert packet['matched_ground_draws']>0 and packet['ground_vertices']>0
            for key in ('original','output'):assert sha(ROOT/packet[key])==packet[key+'_sha256']
        frames=[]
        for zoom in zooms:
            for hour in (12,0):
                file=f'h{hour:02}-z{zoom}-pan00.png';d=delta(before/file,after/file)
                difference=abs(rgb(before/file)-rgb(after/file)).max(2)
                x0,y0,x1,y1=[v//zoom for v in roi];difference[y0:y1,x0:x1]=0
                assert difference.max()<=1 and (difference>0).sum()<=1,'nonlocal ground-binding difference'
                assert d['changed_pixels_gt_2']>0
                frames.append({'hour':hour,'zoom':zoom,**d,'outside_city_roi_max':int(difference.max()),
                               'outside_city_roi_changed_pixels':int((difference>0).sum()),'sha256':sha(after/file)})
        native=read(BASE/f'windows-{name}/evidence.json')
        assert len(native['results'])==2*len(zooms)
        for frame in native['results']:
            assert frame['metrics']['pass']
            for key,path in [('shader_sha256',after/'shaders/source.hlsl'),('reflection_sha256',after/'shaders/reflection/source.hlsl'),
                             ('post_sha256',after/'postprocess/source.hlsl'),('d3d11_sha256',BASE/f'windows-{name}'/frame['frame'])]:
                assert frame[key]==sha(path)
        windows.append(native)
        cases[name]={'frames':frames,'binding_report_sha256':sha(BASE/name/'binding.json'),
                     'same_main_reflection_post_shaders':True,'geometry_and_constants_unchanged':True}
    disabled=[]
    for hour in (12,0):
        file=f'h{hour:02}-z1-pan00.png'
        d=delta(OUT/'city-facade-light-r3/wilderness/render'/file,BASE/'disabled-exact/render'/file)
        assert d['max_channel_delta']==0;disabled.append({'hour':hour,**d})
    identity=read(BASE/'disabled-exact/binding.json')
    for packet in identity['packets']:assert packet['original_sha256']==packet['output_sha256']
    alias_control=[]
    for hour in (12,0):
        file=f'h{hour:02}-z1-pan00.png'
        direct=delta(OUT/'city-facade-light-r3/wilderness/render'/file,BASE/'direct-replay'/file)
        assert direct['max_channel_delta']==0
        d=delta(BASE/'direct-replay'/file,BASE/'disabled/render'/file)
        assert d['max_channel_delta']<=1 and d['changed_pixels_any']<=1
        alias_control.append({'hour':hour,'original_direct_replay':direct,'equal_texture_different_resource_id':d})
    lake=delta(OUT/'city-facade-light-r3/capital/render/h00-z1-pan00.png',BASE/'capital/render/h00-z1-pan00.png',(858,491,885,501))
    assert lake['max_channel_delta']==0
    result={'classification':'Provisional small ground-material improvement; no full urban ground or city-quality acceptance',
            'cases':cases,'disabled_control':disabled,'disabled_packets_byte_identical':True,
            'superseded_equal_texture_rebinding_control':alias_control,
            'capital_lake_roi_unchanged':lake,'standalone_windows_parity':windows,
            'source_bindings':[read(V2/f'fixtures/beauty/city-ground-binding-r1/{era}.json') for era in ('modern','medieval')],
            'holdout':'Freshshadow uses the same modern atlas replacement without local tuning; its crowded layout remains unaccepted',
            'limitations':['Source HeightRange application and state selection unproven','Ground height channel absent',
                           'Pads remain mostly under bodies; coherent inter-building ground and skyline arrangement are unfinished',
                           'Native delivery, broad culture/era/size coverage and all approval gates remain open']}
    target=V2/'audits/beauty/CITY_GROUND_BINDING_r1_EVIDENCE.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    sheet=Image.new('RGB',(760,480),(25,25,25));draw=ImageDraw.Draw(sheet)
    for row,(name,roi) in enumerate([('inland',(650,310,1030,530)),('capital',(735,280,1115,500))]):
        for col,(series,label) in enumerate([('city-facade-light-r3','Previous'),('city-ground-binding-r1','Era-specific paving')]):
            image=Image.open(OUT/series/name/'render/h12-z1-pan00.png').convert('RGB')
            sheet.paste(image.crop(roi),(col*380,row*240+20));draw.text((col*380+5,row*240+4),name+' | '+label,fill='white')
    sheet.save(BASE/'selected-native-comparison.png')
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
