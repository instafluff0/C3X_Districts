"""Matched static-water evidence; pixel counts are not visual acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def rgb(path):return Image.open(path).convert('RGB')

def compare(a,b):
    aa,bb=np.asarray(rgb(a),dtype=np.int16),np.asarray(rgb(b),dtype=np.int16)
    assert aa.shape==bb.shape
    d=np.abs(aa-bb)
    return {'changed_pixels':int(np.count_nonzero(d.max(2))),
            'max_channel_error':int(d.max()),'mean_channel_error':float(d.mean())}

def sheet(images,labels,path):
    w,h=images[0].size
    result=Image.new('RGB',(w*len(images),h+24),(22,24,26));draw=ImageDraw.Draw(result)
    for i,(im,label) in enumerate(zip(images,labels)):
        assert im.size==(w,h)
        result.paste(im,(i*w,24));draw.text((i*w+8,6),label,fill='white')
    result.save(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',type=int,default=6);a=parser.parse_args()
    campaign=f'water-natural-r{a.revision}';review=OUT/campaign/'review'
    review.mkdir(parents=True,exist_ok=True);rows=[]
    regions=['coastal','inland','wilderness','longcoast','freshwater']
    for region in regions:
        base='shadow-receiver-r1' if region=='longcoast' else ('water-natural-foundation' if region=='freshwater' else 'river-corridor-r3')
        old=OUT/base/region;new=OUT/campaign/region/'phase-0'
        previous=json.loads((old/'report.json').read_text())
        current=json.loads((new/'report.json').read_text())
        assert len(previous['outputs'])==len(current['outputs'])==len(current['packets'])==4
        for src,pkt,dst in zip(previous['outputs'],current['packets'],current['outputs']):
            assert pkt['sha256']==sha(ROOT/src['packet'])
            before,after=ROOT/src['image'],ROOT/dst['image'];assert before.name==after.name
            linear=np.fromfile(str(after)+'.linear.rgba16f',dtype='<f2')
            assert np.isfinite(linear).all()
            assert Path(str(before)+'.validity.r8').read_bytes()==Path(str(after)+'.validity.r8').read_bytes()
            row={'region':region,'frame':before.name,'size':list(rgb(before).size),
                 'packet_sha256':pkt['sha256'],'before_sha256':sha(before),'after_sha256':sha(after),
                 'finite_linear':True,'validity_identical':True,**compare(before,after)}
            rows.append(row)
            if '-z2-' in before.name:
                sheet([rgb(before),rgb(after)],['Preserved baseline | '+region,'Static natural water | same pixels/camera'],
                      review/(region+'-'+before.stem+'.png'))
    # Direct native gameplay crop; no resampling.
    box=(340,130,980,510);frame='h12-z1-pan00.bmp'
    baseline=rgb(OUT/'shadow-receiver-r1/longcoast'/frame).crop(box)
    early=rgb(OUT/'water-effects-r2/longcoast/phase-0'/frame).crop(box)
    final=rgb(OUT/campaign/'longcoast/phase-0'/frame).crop(box)
    sheet([baseline,final],['Preserved baseline | native gameplay pixels','Static natural water | native gameplay pixels'],review/'native-water-comparison.png')
    sheet([early,final],['Earlier wave/surf prototype','Static natural water | surf deferred'],review/'previous-prototype-comparison.png')
    # Canonical pixels are unscaled too, but its source camera/zoom differs.
    canonical=rgb(ROOT/'Renderer/canonical/sea_and_shore.png').crop((2600,650,3240,1030))
    sheet([canonical,final],['Canonical Civ VI | original pixels, different zoom','Lab | actual gameplay pixels'],review/'canonical-water-comparison.png')
    controls=[];wrap=[]
    for hour in (12,0):
        for zoom in (1,2):
            frame=f'h{hour:02d}-z{zoom}-pan00.bmp'
            base=OUT/'shadow-receiver-r1/longcoast'/frame
            disabled=OUT/campaign/'longcoast/phase-0-disabled'/frame
            delta=compare(base,disabled)
            assert delta['max_channel_error']<=1 and delta['changed_pixels']<=1
            controls.append({'frame':frame,'byte_identical':sha(base)==sha(disabled),**delta})
            current=OUT/campaign/'longcoast/phase-0'/frame
            shifted=OUT/campaign/'longcoast/phase-0-shift-50'/frame
            delta=compare(current,shifted)
            # Floating arithmetic can cross final 8-bit rounding boundaries.
            assert delta['max_channel_error']<=1 and delta['mean_channel_error']<.001
            wrap.append({'frame':frame,**delta})
    shaders=[]
    for path in (OUT/'water-effects-control/longcoast/shaders').glob('*.msl'):
        other=OUT/campaign/'longcoast/phase-0-disabled/shaders'/path.name
        assert path.read_bytes()==other.read_bytes()
        shaders.append({'name':path.name,'byte_identical':True,'sha256':sha(path)})
    repeats=[]
    for hour in (12,0):
        for zoom in (1,2):
            frame=f'h{hour:02d}-z{zoom}-pan00.bmp'
            first=OUT/campaign/'longcoast/phase-0-disabled'/frame
            repeated=OUT/campaign/'longcoast/disabled-repeat'/frame
            delta=compare(first,repeated)
            assert delta['max_channel_error']<=1 and delta['changed_pixels']<=1
            repeats.append({'frame':frame,**delta})
    benchmark=json.loads((V2/'fixtures/beauty/water-natural-foundation/freshwater/BENCHMARKS.json').read_text())
    evidence={'classification':'static water visual candidate, no promotion or Civ VI-equivalence claim',
              'campaign':campaign,'frames':rows,'disabled_controls':controls,
              'compiled_disabled_shaders':shaders,'disabled_repeats':repeats,
              'control_limit':'Three first-pass controls are exact; midnight zoom1 differs in one channel of one pixel by 1/255, with byte-identical compiled shaders. No existing parity gate is relaxed.',
              'water_only_wrap_coordinate_checks':wrap,'holdout':benchmark,
              'canonical_sha256':sha(ROOT/'Renderer/canonical/sea_and_shore.png'),
              'deferred_by_user':['animation','coastal surf'],
              'remaining':['shallows/bed detail','lake/sea response classification','full crop/edge geometry parity',
                           'D3D parity','controlled runtime cost','human visual checkpoint']}
    (V2/f'audits/beauty/WATER_NATURAL_r{a.revision}_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'matched_frames':len(rows),'disabled_controls':len(controls),'water_coordinate_checks':wrap}))

if __name__=='__main__':main()
