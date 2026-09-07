"""Verify mountain diagnostic inputs, actual bindings and matched pixels."""
import hashlib
import json
from pathlib import Path
import sys
from PIL import Image,ImageChops
from ground_layer_audit import packet_textures

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frames=[f'h{h:02d}-z{z}-pan00' for h in (12,0) for z in (1,2)]
    controls=[];results=[];binding_checks=[]
    for frame in frames:
        a=OUT/'river-corridor-r3/inland'/(frame+'.bmp')
        b=OUT/'rock-channels-control/inland'/(frame+'.bmp')
        assert a.read_bytes()==b.read_bytes()
        controls.append({'frame':frame,'byte_identical':True,'bmp_sha256':sha(a)})
    for region in ('coastal','inland','wilderness','freshground'):
        base='surface-decals-foundation-v2' if region=='freshground' else 'river-corridor-r3'
        source=OUT/base/region;inputs=OUT/'rock-channels-r2-input'/region
        old_jobs=json.loads((source/'batch.json').read_text())
        jobs=json.loads((inputs/'batch.json').read_text())
        bindings=json.loads((inputs/'bindings.json').read_text())
        for i,frame in enumerate(frames):
            e=bindings[i]
            assert sha(Path(old_jobs[i][0]))==e['source_packet_sha256']
            assert sha(Path(jobs[i][0]))==e['output_packet_sha256']
            assert len(e['added_channels'])==8 and e['all_non_binding_tail_bytes_identical']
            if i==0:
                actual=packet_textures(Path(jobs[i][0]))
                for channel in e['added_channels']:
                    texture=actual['textures'][channel['texture_index']-1]
                    assert texture['payload_sha256']==channel['payload_sha256']
                    assert len(texture['bindings'])>0
                    assert all(b['slot']==channel['slot'] and not b['feature_shader'] for b in texture['bindings'])
                binding_checks.append({'region':region,'packet_sha256':actual['sha256'],
                    'eight_material_channels_verified':True,'feature_bindings_preserved':True})
            a=source/(frame+'.bmp');b=OUT/'rock-channels-r2'/region/(frame+'.bmp')
            ia=Image.open(a).convert('RGB');ib=Image.open(b).convert('RGB');assert ia.size==ib.size
            diff=ImageChops.difference(ia,ib)
            results.append({'region':region,'frame':frame,'size':list(ia.size),
                'baseline_bmp_sha256':sha(a),'candidate_bmp_sha256':sha(b),
                'changed_pixels':sum(p!=(0,0,0) for p in diff.getdata()),'changed_bounds':diff.getbbox(),
                'source_packet_sha256':e['source_packet_sha256'],'candidate_packet_sha256':e['output_packet_sha256'],
                'all_non_binding_tail_bytes_identical':True,
                'baseline_gpu_ms':json.loads(a.with_suffix('.cost.json').read_text())['gpu_ms_mean'],
                'diagnostic_gpu_ms':json.loads(b.with_suffix('.cost.json').read_text())['gpu_ms_mean']})
    synthetic=[]
    for frame in frames:
        a=OUT/'rock-desert-witness-baseline/inland'/(frame+'.bmp')
        b=OUT/'rock-desert-witness-r2/inland'/(frame+'.bmp')
        diff=ImageChops.difference(Image.open(a).convert('RGB'),Image.open(b).convert('RGB'))
        synthetic.append({'frame':frame,'baseline_bmp_sha256':sha(a),'candidate_bmp_sha256':sha(b),
            'changed_pixels':sum(p!=(0,0,0) for p in diff.getdata()),'changed_bounds':diff.getbbox()})
    sys.path.insert(0,str(V2/'app'));import real_map
    reg,data=real_map.load_registry();mountains=[t for t in data['tiles'] if t['real']==6]
    source_counts={'source_biq_sha256':reg['source']['sha256'],'mountain_tiles':len(mountains),
        'desert_base_mountains':sum(t['base']==0 for t in mountains),'grass_base_mountains':sum(t['base']==2 for t in mountains)}
    result={'classification':'combined_shader_and_binding_diagnostic','promotion':False,
        'default_off_controls':controls,'matched_frames':results,'actual_binding_checks':binding_checks,
        'synthetic_desert_material_witness':synthetic,'biq_material_coverage':source_counts,
        'source_selector_evidence':'GROUND_LAYER_SELECTORS.json',
        'remaining':['source blend thresholds and physical unit mapping','world crop/wrap material stability',
            'production material binding adapter and shader cost','high-ground geometry and material graph',
            'combined visual review and all existing milestone gates']}
    (V2/'audits/beauty/ROCK_CHANNEL_r2_EVIDENCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'control_frames_exact':len(controls),'matched_frames':len(results),
        'synthetic_material_frames':len(synthetic),'actual_packet_binding_checks':len(binding_checks),
        'biq_material_coverage':source_counts}))


if __name__=='__main__':main()
