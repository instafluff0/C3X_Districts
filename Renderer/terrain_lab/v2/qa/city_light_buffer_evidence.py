"""Verify buffered-light transport, backend parity and full-spill visual controls."""
import json
import subprocess
from city_growth_evidence import ROOT, V2, OUT, read, sha
from city_growth_hierarchy_evidence import difference
from city_scene_pass import executable, Cache
from city_light_buffer_probe import payload

BASE=OUT/'city-light-buffer-r1'
CASES=['asian-medium','ancient-medium','asian-large','asian-holdout',
       'asian-medium-full','asian-holdout-full']


def main():
    exe=executable(V2/'qa/frame_data_contract.cpp',Cache(V2/'app/.cache'))
    cases={}
    for name in CASES:
        folder=BASE/name;binding=read(folder/'binding.json');render=folder/'render'
        data=read(ROOT/binding['lights']);assert sha(ROOT/binding['lights'])==binding['lights_sha256']
        assert (folder/'lights.bin').read_bytes()==payload(data)
        assert sha(folder/'lights.bin')==binding['payload_sha256']
        report=read(render/'report.json');old=read(ROOT/binding['source_render']/'report.json')
        assert report['postprocess']==old['postprocess']
        checks=[]
        for p in binding['packets']:
            for key in ('original','output'):assert sha(ROOT/p[key])==p[key+'_sha256']
            checks.append(json.loads(subprocess.check_output([str(exe),str(ROOT/p['original']),
                       str(ROOT/p['output']),str(folder/'lights.bin')],text=True)))
        windows=read(BASE/f'windows-{name}/evidence.json');assert len(windows['results'])==2
        for index,row in enumerate(windows['results']):
            assert row['metrics']['pass']
            for key,path in [('d3d11_sha256',BASE/f'windows-{name}'/row['frame']),
                             ('packet_sha256',ROOT/report['packets'][index]['path']),
                             ('shader_sha256',render/'shaders/source.hlsl'),
                             ('reflection_sha256',render/'shaders/reflection/source.hlsl'),
                             ('post_sha256',render/'postprocess/source.hlsl')]:
                assert row[key]==sha(path)
        controls=[]
        if not name.endswith('-full'):
            for hour in (12,0):
                file=f'h{hour:02}-z1-pan00.bmp'
                before=OUT/f'city-culture-r2/windows-{name}'/file;after=BASE/f'windows-{name}'/file
                assert sha(before)==sha(after),'buffer transport changed the saved Windows appearance'
                controls.append({'hour':hour,'exact_bmp_sha256':sha(after)})
        costs=[read(render/f'h{hour:02}-z1-pan00.cost.json') for hour in (12,0)]
        cases[name]={'lights':len(data['lights']),'blockers':len(data['blockers']),
                     'frame_checks':checks,'windows':windows,'same_backend_transport_controls':controls,
                     'elapsed':read(folder/'elapsed.json'),'gpu_cost':costs,
                     'shader_sha256':sha(render/'shaders/source.hlsl'),
                     'reflection_sha256':sha(render/'shaders/reflection/source.hlsl')}
    for name,roi in [('asian-medium',(630,280,1030,565)),('asian-holdout',(500,320,820,570))]:
        previous=BASE/name/'render';full=BASE/(name+'-full')/'render'
        assert cases[name]['shader_sha256']==cases[name+'-full']['shader_sha256']
        assert cases[name]['reflection_sha256']==cases[name+'-full']['reflection_sha256']
        frames=[difference(previous/f'h{hour:02}-z1-pan00.png',full/f'h{hour:02}-z1-pan00.png',roi) for hour in (12,0)]
        assert frames[0]['max_channel_delta']==0
        assert frames[1]['changed_pixels_gt_2']>100
        assert all(f['outside_city_roi_max']==0 for f in frames)
        old=read(ROOT/read(BASE/name/'binding.json')['lights'])
        complete=read(ROOT/read(BASE/(name+'-full')/'binding.json')['lights'])
        assert old['blockers']==complete['blockers']
        assert all(light in complete['lights'] for light in old['lights'])
        cases[name+'-full']['matched_full_spill_pixels']=frames
    result={'classification':'Buffered generic lighting resolves the dense-city compile blocker; full spill improves local night readability provisionally',
            'cases':cases,'replaces_pending_static_trials':'Eight culture-density comparisons superseded by eight passing buffered equivalents with exact old Windows images',
            'verification':'Twelve new Metal/D3D comparisons and twelve independent frame/packet checks; two focused payload tests',
            'visual_selection':['asian-medium-full','asian-holdout-full'],
            'remaining':['General connected large-city placement; scattered large case remains unselected',
                         'Broader culture/era/size and palace composition',
                         'New shoreline reflection coverage; existing capital-lake witness preserved',
                         'Source local-light model remains an approximation; no native or milestone promotion']}
    target=V2/'audits/beauty/CITY_LIGHT_BUFFER_EVIDENCE.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
