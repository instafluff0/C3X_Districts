"""Quantify full-feature light-index captures and write lossless review sheets."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'Renderer/.cache/city-light-index-step')
    args=parser.parse_args();out=args.out.resolve();results={}
    for path in sorted(out.glob('capture-*-baseline/frame.bmp')):
        label=path.parent.name[:-len('-baseline')];candidate=out/(label+'-candidate')/'frame.bmp'
        if not candidate.is_file():raise RuntimeError('Unpaired capture: '+label)
        images=[Image.open(p).convert('RGB') for p in [path,candidate]]
        if images[0].size!=images[1].size:raise ValueError('Capture dimensions differ')
        arrays=[np.array(im).astype(np.int16) for im in images];delta=np.abs(arrays[0]-arrays[1]);changed=np.any(delta!=0,axis=2)
        results[label]={'size':list(images[0].size),'max_channel_difference':int(delta.max()),
                        'mean_channel_difference':float(delta.mean()),'rms_channel_difference':float(np.sqrt(np.mean(delta.astype(float)**2))),
                        'changed_pixels':int(changed.sum()),'changed_fraction':float(changed.mean()),
                        'baseline_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest()}
        heat=Image.fromarray(np.clip(np.max(delta,axis=2)*32,0,255).astype('uint8')).convert('RGB')
        size=(896,504);sheet=Image.new('RGB',(size[0]*3,size[1]+48),'#20252b');draw=ImageDraw.Draw(sheet)
        for column,(im,title) in enumerate(zip(images+[heat],['Full scan','Indexed','Absolute difference ×32'])):
            sheet.paste(im.resize(size), (column*size[0],48));draw.text((column*size[0]+12,12),title,fill='white')
        draw.text((12,32),label+'  '+str(results[label]['changed_pixels'])+' changed pixels; max '+str(results[label]['max_channel_difference']),fill='white')
        sheet.save(out/(label+'-comparison.png'))
        # Preserve full-size lossless pixels in a convenient format too.
        for im,arm in zip(images,['baseline','candidate']):im.save(out/(label+'-'+arm+'.png'))
    if not results:raise ValueError('No paired captures')
    (out/'image-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
