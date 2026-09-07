"""Verify matched GPU-reflection captures and generate native review images."""
import json
from pathlib import Path
import sys
import numpy as np
from verify_natural_water import sha,compare,rgb,sheet,ROOT,V2,OUT
sys.path.insert(0,str(V2/'app'))
from parity import compare as compare_backends
from runner import CONTRACT

def d3d_evidence():
    rows=[];effects=[]
    for region in ['freshwater','longcoast']:
        folder=OUT/'water-reflection-d3d-r5'/region
        evidence=json.loads((folder/'evidence.json').read_text())
        assert len(evidence['results'])==8
        for row in evidence['results']:
            metal=ROOT/row['metal'];target=ROOT/row['d3d11']
            assert sha(target)==row['d3d11_sha256']
            repeat=folder/(row['mode']+'-repeat-'+row['frame'])
            assert sha(target)==sha(repeat)
            metrics=compare_backends(metal,target)
            assert metrics['pass']
            rows.append({'region':region,**row,'metrics':metrics,'repeat_identical':True})
        for hour in [12,0]:
            for zoom in [1,2]:
                frame=f'h{hour:02d}-z{zoom}-pan00.bmp'
                image=lambda p:np.asarray(rgb(p),dtype=np.int16)
                metal=(image(OUT/'water-reflection-r5'/region/'combined'/frame)-
                       image(OUT/'water-natural-r6'/region/'phase-0'/frame))
                d3d=image(folder/('reflected-'+frame))-image(folder/('sky-only-'+frame))
                error=np.abs(metal-d3d)
                effects.append({'region':region,'frame':frame,
                                'd3d_reflection_changed_pixels':int(np.any(d3d!=0,axis=2).sum()),
                                'effect_delta_max_channel':int(error.max()),
                                'effect_delta_mean_channel':float(error.mean())})
    return {'classification':'standalone Windows Lab; not native integration',
            'backend_source_sha256':sha(V2/'backends/d3d11.cpp'),
            'backend_binary_sha256':sha(V2/'backends/build/d3d11.exe'),
            'unchanged_tolerances':json.loads(CONTRACT.read_text())['parity_v1'],
            'results':rows,'reflection_effect_comparisons':effects,
            'shared_resource_read_incident':json.loads((OUT/'water-reflection-d3d-r5/longcoast/shared-read-failure.json').read_text()),
            'limits':['two coastal regions, not full promotion matrix','no native-game performance measurement']}

def crop_evidence():
    probe=OUT/'water-reflection-crops-r5/freshwater/y-120'
    rows=[]
    for hour in [12,0]:
        for zoom in [1,2]:
            frame=f'h{hour:02d}-z{zoom}-pan00.bmp';shift=120//zoom
            image=lambda p:np.asarray(rgb(p),dtype=np.int16)
            original=(image(OUT/'water-reflection-r5/freshwater/combined'/frame)-
                      image(OUT/'water-natural-r6/freshwater/phase-0'/frame))[shift:]
            cropped=(image(probe/'reflected'/frame)-image(probe/'sky-only'/frame))[:-shift]
            error=np.abs(original-cropped)
            assert error.max()<=1
            rows.append({'frame':frame,'viewport_shift_internal_pixels':[0,-120],
                         'reflected_sha256':sha(probe/'reflected'/frame),
                         'sky_only_sha256':sha(probe/'sky-only'/frame),
                         'reflection_effect_changed_pixels':int(np.any(cropped!=0,axis=2).sum()),
                         'effect_crop_max_channel_error':int(error.max()),
                         'effect_crop_mean_channel_error':float(error.mean()),
                         'top_three_rows_max_channel_error':int(error[:3].max())})
    return {'classification':'matched viewport shift; already captured geometry only',
            'results':rows,'limit':'does not prove provider culling or scene halo coverage'}

def main():
    campaign=OUT/'water-reflection-r5';review=campaign/'review';review.mkdir(exist_ok=True)
    rows=[];parity=[];controls=[];cost=[]
    for region in ['coastal','inland','wilderness','longcoast','freshwater']:
        base='shadow-receiver-r1' if region=='longcoast' else ('water-natural-foundation' if region=='freshwater' else 'river-corridor-r3')
        source=json.loads((OUT/base/region/'report.json').read_text())
        current=json.loads((campaign/region/'combined/report.json').read_text())
        assert len(source['outputs'])==len(current['outputs'])==len(current['packets'])==4
        assert 'same command buffer' in current['reflection']['execution']
        for src,pkt,dst in zip(source['outputs'],current['packets'],current['outputs']):
            assert pkt['sha256']==sha(ROOT/src['packet'])
            after=ROOT/dst['image'];before=OUT/'water-natural-r6'/region/'phase-0'/after.name
            assert np.isfinite(np.fromfile(str(after)+'.linear.rgba16f',dtype='<f2')).all()
            assert Path(str(before)+'.validity.r8').read_bytes()==Path(str(after)+'.validity.r8').read_bytes()
            metrics=json.loads(after.with_suffix('.cost.json').read_text())
            assert metrics['reflection_passes']==1 and metrics['draw_count']%2==0
            rows.append({'region':region,'frame':after.name,'packet_sha256':pkt['sha256'],
                         'before_sha256':sha(before),'after_sha256':sha(after),
                         'finite_linear':True,'validity_identical':True,**compare(before,after)})
            if '-z2-' in after.name:
                sheet([rgb(before),rgb(after)],['Current water | '+region,'GPU object reflections | same camera'],review/(region+'-'+after.stem+'.png'))
            if region in ['longcoast','freshwater']:
                reference=OUT/'water-reflection-r2'/region/'combined'/after.name
                delta=compare(reference,after)
                assert delta['max_channel_error']<=1 and delta['changed_pixels']<=1
                parity.append({'region':region,'frame':after.name,**delta})
    for hour in [12,0]:
        for zoom in [1,2]:
            frame=f'h{hour:02d}-z{zoom}-pan00.bmp'
            baseline=OUT/'water-reflection-r4/cost/baseline'/frame
            previous=OUT/'water-natural-r6/longcoast/phase-0'/frame
            delta=compare(baseline,previous)
            assert delta['max_channel_error']<=1 and delta['changed_pixels']<=1
            controls.append({'frame':frame,**delta})
            a=json.loads(baseline.with_suffix('.cost.json').read_text())
            b=json.loads((campaign/'cost/reflected'/frame).with_suffix('.cost.json').read_text())
            assert a['batch_frames']==b['batch_frames']==8
            assert a['reflection_passes']==0 and b['reflection_passes']==1
            cost.append({'frame':frame,'baseline_gpu_ms':a['gpu_ms_mean'],'reflected_gpu_ms':b['gpu_ms_mean'],
                         'extra_gpu_ms':b['gpu_ms_mean']-a['gpu_ms_mean'],
                         'baseline_draws':a['draw_count'],'reflected_draws':b['draw_count'],
                         'extra_allocation_bytes':b['allocation_high_water_sampled_bytes']-a['allocation_high_water_sampled_bytes']})
    box=(560,95,800,245)
    before=rgb(OUT/'water-natural-r6/longcoast/phase-0/h12-z1-pan00.bmp').crop(box)
    after=rgb(campaign/'longcoast/combined/h12-z1-pan00.bmp').crop(box)
    sheet([before,after],['Current water | native pixels','With reflected rocks | native pixels'],review/'rocks-native.png')
    sheet([before.resize((720,450)),after.resize((720,450))],
          ['Current water | 3x detail view','GPU object reflections | 3x detail view'],review/'rocks-detail.png')
    treebox=(595,55,790,170)
    before=rgb(OUT/'water-natural-r6/freshwater/phase-0/h12-z1-pan00.bmp').crop(treebox)
    after=rgb(campaign/'freshwater/combined/h12-z1-pan00.bmp').crop(treebox)
    sheet([before.resize((585,345)),after.resize((585,345))],
          ['Current water | 3x detail view','Reflected canopy at shoreline | 3x detail view'],review/'trees-detail.png')
    evidence={'classification':'GPU planar-reflection Lab candidate; no Windows integration or visual promotion',
              'frames':rows,'offline_to_gpu_parity':parity,'reflection_disabled_controls':controls,
              'eight_frame_cost_samples':cost,
              'offscreen_reflector_probe':crop_evidence(),
              'windows_d3d_probe':d3d_evidence(),
              'constraints':['one scene-linear shader namespace','render scale 1','common water plane z=0',
                             't121 overridden only for main/water draws; feature material alias preserved'],
              'remaining':['full promotion matrix and integration delivery','provider halo/culling coverage for off-viewport reflectors',
                           'roughness/fade calibration','river elevations and multiple water planes','human visual review']}
    (V2/'audits/beauty/WATER_REFLECTION_r5_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'matched_frames':len(rows),'offline_gpu_comparisons':len(parity),
                      'disabled_controls':controls,'cost':cost},indent=2))

if __name__=='__main__':main()
