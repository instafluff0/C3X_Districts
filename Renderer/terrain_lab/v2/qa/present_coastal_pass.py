"""Present preserved coastal and volcano renders at unchanged gameplay size."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='rocks-r4')
    p.add_argument('--previous',help='Preserved coastal revision for matched before/after')
    p.add_argument('--regions',nargs='+',default=['coastal','inland','wilderness','longcoast'])
    a=p.parse_args();dest=OUT/('coast-pass-'+a.revision)/'review';dest.mkdir(parents=True,exist_ok=True)
    records=[];font=ImageFont.load_default()
    def comparison(name,sources,box):
        w,h=box[2]-box[0],box[3]-box[1]
        sheet=Image.new('RGB',(w*2+24,h+48),(23,28,33));draw=ImageDraw.Draw(sheet)
        record={'name':name,'crop':box,'resampling':False,'sources':[]}
        for i,(path,label) in enumerate(sources):
            im=Image.open(path).convert('RGB').crop(box)
            sheet.paste(im,(i*(w+24),32));draw.text((i*(w+24)+8,10),label,font=font,fill='white')
            if i:im.save(dest/(name+'.png'))
            record['sources'].append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        sheet.save(dest/(name+'-comparison.png'));records.append(record)
    fixed=json.loads((V2/'fixtures/beauty/gameplay-100-v1/BENCHMARKS.json').read_text())['gameplay_crop']
    for region in a.regions:
        box=json.loads((V2/'fixtures/beauty/coast-pass-foundation/BENCHMARKS.json').read_text())['gameplay_crop'] if region=='longcoast' else fixed
        before=OUT/('coast-pass-baseline-v5' if region=='longcoast' else 'gameplay-100-candidate-v5')/region
        if a.previous:before=OUT/('coast-pass-'+a.previous)/region
        after=OUT/('coast-pass-'+a.revision)/region
        for hour,phase in [(12,'day'),(0,'night')]:
            filename=f'h{hour:02}-z1-pan00.png'
            previous=before/filename
            if not a.previous and region=='longcoast' and hour==0:
                previous=OUT/'coast-pass-baseline-v5-night/longcoast'/filename
            if not previous.exists():continue
            comparison(region+'-'+phase,[(previous,region+' | '+(a.previous or 'previous best')+' | unscaled'),
                (after/filename,region+' | '+a.revision+' | unscaled')],box)
            if region=='freshcoast':
                z2=f'h{hour:02}-z2-pan00.png'
                comparison(region+'-z2-'+phase,[(before/z2,region+' | '+(a.previous or 'previous best')+' | full native zoom 2'),
                    (after/z2,region+' | '+a.revision+' | full native zoom 2')],[0,0,680,400])
    for hour,phase in [(12,'day'),(0,'night')]:
        comparison('synthetic-volcano-'+phase,[(OUT/'volcano-witness-r1/inland'/f'h{hour:02}-z1-pan00.png','Synthetic volcano | inherited mapping'),
            (OUT/'volcano-witness-r2/inland'/f'h{hour:02}-z1-pan00.png','Synthetic volcano | aligned source mapping')],fixed)
    # Navigation only: canonical reference scaling is explicitly labeled.
    sheet=Image.new('RGB',(1304,368),(23,28,33));draw=ImageDraw.Draw(sheet)
    for i,(name,label) in enumerate([('sea_and_shore.png','Canonical Civ VI | shore/rock detail'),('civ3_real_example.jpg','Canonical Civ III | coastline shape')]):
        im=Image.open(ROOT/'Renderer/canonical'/name).convert('RGB');im.thumbnail((640,320),Image.Resampling.LANCZOS)
        sheet.paste(im,(i*664+(640-im.width)//2,32));draw.text((i*664+8,10),label+' | scaled overview',font=font,fill='white')
    sheet.save(dest/'canonical-shore-references.png')
    if set(a.regions)>={'coastal','inland','wilderness','longcoast'}:
        for hour,phase in [(12,'day'),(0,'night')]:
            sheet=Image.new('RGB',(1520,960),(23,28,33));draw=ImageDraw.Draw(sheet)
            for i,region in enumerate(['longcoast','coastal','inland','wilderness']):
                x=(i%2)*824;y=(i//2)*480
                im=Image.open(OUT/('coast-pass-'+a.revision)/region/f'h{hour:02}-z2-pan00.png')
                sheet.paste(im,(x,y+26));draw.text((x+8,y+8),region+' | zoom 2, native pixels',font=font,fill='white')
            sheet.save(dest/('all-regions-z2-'+phase+'.png'))
    (dest/'crops.json').write_text(json.dumps({'records':records,'visual_accepted':False},indent=2)+'\n')
    print(dest.relative_to(ROOT))

if __name__=='__main__':main()
