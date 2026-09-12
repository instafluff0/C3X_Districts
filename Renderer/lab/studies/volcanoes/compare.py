#!/usr/bin/env python3
"""Compare volcano changes with the accepted mountain-context appearance."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from Renderer import renderer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',default='skin-lava-shadow-probe',
                        choices=['skin-shadow-probe','skin-lava-shadow-probe'])
    args=parser.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    out=ROOT/'Renderer/lab/out/volcanoes'
    study=out/'material-study'
    receipt=renderer.read(study/'render-224.json')
    sheet=Image.new('RGB',(1280,520),(28,31,36))
    draw=ImageDraw.Draw(sheet);font=ImageFont.load_default(size=20)
    hashes={}
    for x,variant,title in ((0,'skin-probe','Approved appearance'),
                            (640,args.proposal,'Lab: static lava and cast shadow' if args.proposal=='skin-lava-shadow-probe' else 'Lab: corrected cast shadow')):
        record=next(r for r in receipt['renders'] if r['variant']==variant and r['case']=='gameplay' and r['zoom']==224)
        path=ROOT/record['image']
        if renderer.checksum(path)!=record['sha256'] or record['dll_sha256']!=receipt['candidate_sha256']:
            raise ValueError('Comparison image/candidate no longer matches its receipt')
        draw.text((x+12,12),title,fill='white',font=font)
        sheet.paste(Image.open(path).convert('RGB'),(x,40))
        hashes[variant]=record['sha256']
    target=out/'lava-shadow-comparison.png';sheet.save(target)
    renderer.write(out/'lava-shadow-comparison.json',dict(diagnostic_only=True,
        candidate_sha256=receipt['candidate_sha256'],images=hashes,image=renderer.relative(target)))


if __name__=='__main__':
    main()
