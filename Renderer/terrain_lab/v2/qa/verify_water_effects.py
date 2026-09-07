"""Matched-packet water exploration evidence and native-size comparisons."""
import hashlib
import json
from pathlib import Path
from PIL import Image,ImageChops,ImageDraw

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    rows=[];controls=[]
    for region in ['coastal','inland','wilderness','longcoast']:
        baseline='shadow-receiver-r1' if region=='longcoast' else 'river-corridor-r3'
        old=OUT/baseline/region
        old_report=json.loads((old/'report.json').read_text())
        for phase in (['0','1p5'] if region=='longcoast' else ['0']):
            folder=OUT/'water-effects-r2'/region/('phase-'+phase)
            report=json.loads((folder/'report.json').read_text())
            assert len(report['packets'])==4 and len(report['outputs'])==4
            for source,packet,result in zip(old_report['outputs'],report['packets'],report['outputs']):
                assert packet['sha256']==sha(ROOT/source['packet'])
                a,b=ROOT/source['image'],ROOT/result['image'];assert a.name==b.name
                ia,ib=Image.open(a).convert('RGB'),Image.open(b).convert('RGB');assert ia.size==ib.size
                diff=ImageChops.difference(ia,ib)
                rows.append({'region':region,'phase':phase,'frame':a.name,'size':list(ia.size),
                    'packet_sha256':packet['sha256'],'baseline_bmp_sha256':sha(a),'candidate_bmp_sha256':sha(b),
                    'changed_pixels':sum(p!=(0,0,0) for p in diff.get_flattened_data()),'changed_bounds':diff.getbbox()})
                if region=='longcoast' and phase=='0':
                    control=OUT/'water-effects-control/longcoast'/a.name
                    assert sha(control)==sha(a)
                    controls.append({'frame':a.name,'byte_identical':True,'sha256':sha(a)})
        # Full zoom-2 frames at their rendered size, no rescaling.
        for hour in [12,0]:
            name=f'h{hour:02d}-z2-pan00.bmp';a=Image.open(old/name).convert('RGB')
            b=Image.open(OUT/'water-effects-r2'/region/'phase-0'/name).convert('RGB')
            sheet=Image.new('RGB',(a.width*2,a.height+24),(22,24,26))
            sheet.paste(a,(0,24));sheet.paste(b,(a.width,24));draw=ImageDraw.Draw(sheet)
            draw.text((8,6),'Previous best | '+region,fill='white')
            draw.text((a.width+8,6),'Water exploration | '+('noon' if hour==12 else 'midnight'),fill='white')
            review=OUT/'water-effects-r2/review';review.mkdir(exist_ok=True)
            sheet.save(review/f'{region}-h{hour:02d}-z2.png')
    # Full-size crop of the long coast, selected to show both surf and open water.
    a=Image.open(OUT/'shadow-receiver-r1/longcoast/h12-z1-pan00.bmp').convert('RGB')
    b=Image.open(OUT/'water-effects-r2/longcoast/phase-0/h12-z1-pan00.bmp').convert('RGB')
    box=(340,130,980,510);sheet=Image.new('RGB',(1280,404),(22,24,26))
    sheet.paste(a.crop(box),(0,24));sheet.paste(b.crop(box),(640,24))
    draw=ImageDraw.Draw(sheet);draw.text((8,6),'Previous best | native gameplay pixels',fill='white')
    draw.text((648,6),'Water effects prototype | same camera and scene',fill='white')
    sheet.save(OUT/'water-effects-r2/review/native-water-comparison.png')
    phases=[]
    for hour in (12,0):
        for zoom in (1,2):
            name=f'h{hour:02d}-z{zoom}-pan00.bmp'
            a=Image.open(OUT/'water-effects-r2/longcoast/phase-0'/name).convert('RGB')
            b=Image.open(OUT/'water-effects-r2/longcoast/phase-1p5'/name).convert('RGB')
            d=ImageChops.difference(a,b)
            phases.append({'frame':name,'changed_pixels':sum(p!=(0,0,0) for p in d.get_flattened_data())})
    assert all(p['changed_pixels']>0 for p in phases)
    evidence={'classification':'water shader exploration, no visual promotion or runtime animation claim',
        'frames':rows,'disabled_controls':controls,'phase_changes':phases,
        'canonical_references':{name:sha(ROOT/'Renderer/canonical'/name) for name in ['sea_and_shore.png','river.png']},
        'remaining':['lake/sea distinction','continuous runtime time input and redraw policy','refraction scene buffers',
            'new untuned holdout','crop/wrap GPU comparisons','D3D parity','human visual checkpoint']}
    (V2/'audits/beauty/WATER_EFFECTS_r2_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'matched_frames':len(rows),'exact_disabled_controls':len(controls),'phase_pairs':len(phases)}))

if __name__=='__main__':main()
