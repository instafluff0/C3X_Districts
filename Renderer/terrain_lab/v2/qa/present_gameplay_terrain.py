"""Make unscaled gameplay crops and matched review sheets (requires Pillow)."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='candidate-v5')
    a=p.parse_args()
    bench=json.loads((V2/'fixtures/beauty/gameplay-100-v1/BENCHMARKS.json').read_text())
    box=tuple(bench['gameplay_crop']);w=box[2]-box[0];h=box[3]-box[1]
    out=V2/'audits/beauty/out'/('gameplay-100-'+a.revision)/'review'
    out.mkdir(exist_ok=True)
    font=ImageFont.load_default()
    records=[];references=[]
    for region,reference in [('coastal','sea_and_shore.png'),('inland','mountain.png'),('wilderness','hills.png')]:
        for hour,phase,before in [(12,'day','baseline-v1'),(0,'night','candidate-v2')]:
            sheet=Image.new('RGB',(w*2+24,h+48),(23,28,33))
            draw=ImageDraw.Draw(sheet)
            sources=[]
            for i,revision in enumerate([before,a.revision]):
                source=V2/'audits/beauty/out'/('gameplay-100-'+revision)/region/f'h{hour:02}-z1-pan00.png'
                im=Image.open(source).convert('RGB').crop(box)
                sheet.paste(im,(i*(w+24),32))
                draw.text((i*(w+24)+8,10),f'{region} {phase} | {revision} | actual gameplay pixels',font=font,fill='white')
                if i:im.save(out/f'{region}-{phase}.png')
                sources.append({'path':source.relative_to(ROOT).as_posix(),
                    'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
            sheet.save(out/f'{region}-{phase}-comparison.png')
            records.append({'region':region,'phase':phase,'crop':box,'resampling':False,'sources':sources})
        # References are a separate, labeled sheet; the benchmark crop stays
        # unscaled. Source screenshot downscaling here is for navigation only.
        reference_path=ROOT/'Renderer/canonical'/reference
        ref=Image.open(reference_path).convert('RGB')
        references.append({'path':reference_path.relative_to(ROOT).as_posix(),
            'sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            'presentation':'scaled navigation overview; inspect source image for fidelity'})
        ref.thumbnail((w,h),Image.Resampling.LANCZOS)
        sheet=Image.new('RGB',(w*2+24,h+48),(23,28,33));draw=ImageDraw.Draw(sheet)
        sheet.paste(Image.open(out/f'{region}-day.png'),(0,32))
        sheet.paste(ref,(w+24+(w-ref.width)//2,32+(h-ref.height)//2))
        draw.text((8,10),f'{region} | current gameplay crop, unscaled',font=font,fill='white')
        draw.text((w+32,10),f'Canonical {reference} | scaled overview',font=font,fill='white')
        sheet.save(out/f'{region}-reference.png')
    (out/'crops.json').write_text(json.dumps({'revision':a.revision,'records':records,
        'references':references},indent=2)+'\n')
    print(out.relative_to(ROOT))

if __name__=='__main__':main()
