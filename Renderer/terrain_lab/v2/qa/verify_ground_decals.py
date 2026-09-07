"""Record matched decal diagnostic pixels and immutable input checks."""
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    frames=[f'h{h:02d}-z{z}-pan00' for h in (12,0) for z in (1,2)]
    controls=[];results=[]
    for frame in frames:
        a=OUT/'river-corridor-r3/inland'/(frame+'.bmp')
        b=OUT/'surface-decals-control/inland'/(frame+'.bmp')
        assert a.read_bytes()==b.read_bytes(),'disabled candidate changed baseline'
        controls.append({'frame':frame,'bmp_sha256':digest(a),'byte_identical':True})
    for region in ('coastal','inland','wilderness','freshcanopy','freshground'):
        base='surface-decals-foundation-v2' if region=='freshground' else 'river-corridor-r3'
        inputs='surface-decals-r3-input' if region=='inland' else 'surface-decals-r4-input'
        source_jobs=json.loads((OUT/base/region/'batch.json').read_text())
        copied_jobs=json.loads((OUT/inputs/region/'batch.json').read_text())
        bindings=json.loads((OUT/inputs/region/'rebinding.json').read_text())
        for i,frame in enumerate(frames):
            evidence=bindings[i]
            assert digest(Path(source_jobs[i][0]))==evidence['source_packet_sha256']
            assert digest(Path(copied_jobs[i][0]))==evidence['output_packet_sha256']
            assert len(evidence['changed_textures'])==4
            assert evidence['all_geometry_buffers_draws_bindings_unchanged']
            a=OUT/base/region/(frame+'.bmp');b=OUT/'surface-decals-r4'/region/(frame+'.bmp')
            ia=Image.open(a).convert('RGB');ib=Image.open(b).convert('RGB')
            assert ia.size==ib.size
            diff=ImageChops.difference(ia,ib)
            changed=sum(p!=(0,0,0) for p in diff.getdata())
            cost_a=json.loads(a.with_suffix('.cost.json').read_text())
            cost_b=json.loads(b.with_suffix('.cost.json').read_text())
            results.append({'region':region,'frame':frame,'output_size':list(ia.size),
                'baseline_bmp_sha256':digest(a),'candidate_bmp_sha256':digest(b),
                'changed_pixels':changed,'changed_bounds':diff.getbbox(),
                'baseline_gpu_ms':cost_a['gpu_ms_mean'],'diagnostic_gpu_ms':cost_b['gpu_ms_mean'],
                'geometry_draw_binding_tail_sha256':evidence['unchanged_packet_tail_sha256'],
                'source_packet_still_matches':True})
            if region=='freshground' and frame=='h12-z1-pan00':
                crop=(360,220,1000,540);sheet=Image.new('RGB',(640,664),(24,24,24))
                sheet.paste(ia.crop(crop),(0,0));sheet.paste(ib.crop(crop),(0,344))
                review=OUT/'surface-decals-r4/review';review.mkdir(exist_ok=True)
                sheet.save(review/'freshground-noon-native.png')
    record={'classification':'diagnostic_only','promotion':False,'default_off_controls':controls,
        'matched_frames':results,'untuned_region':'beauty-freshground-100-v1',
        'invalid_run_excluded':'surface-decals-foundation/freshground: rendered intermediate export recipe; corrected baseline is foundation-v2',
        'visual_verdict':'Subtle patch variation; insufficient reference-richness gain and excessive GPU cost. Preserve prior best.',
        'cost_caveat':'Local render timing samples, not a controlled production performance benchmark.'}
    (V2/'audits/beauty/GROUND_DECAL_r4_EVIDENCE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'control_frames_exact':len(controls),'matched_frames':len(results),'promotion':False}))


if __name__=='__main__':main()
