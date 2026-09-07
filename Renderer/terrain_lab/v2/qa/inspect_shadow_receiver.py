"""Pixel comparisons and immutable geometry checks; never grants visual approval."""
import argparse
import json
from pathlib import Path
import sys
V2=Path(__file__).resolve().parents[1];ROOT=V2.parents[2]
sys.path.insert(0,str(V2/'app'));sys.path.insert(0,str(V2/'qa'))
import real_map
from shadow_receiver_pass import REGIONS
from verify_gameplay_terrain import load,sha,difference

def main():
    from PIL import Image,ImageDraw,ImageFont
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions',nargs='+',choices=REGIONS,default=REGIONS)
    args=parser.parse_args()
    base=V2/'audits/beauty/out';dest=base/'shadow-receiver-r1/review';dest.mkdir(parents=True,exist_ok=True)
    evidence={'schema':'c3x.shadow_receiver_evidence.v1','approval':None,'visual_accepted':False,'results':[]}
    for region in args.regions:
        old=base/('shadow-receiver-baseline' if region=='freshshadow' else 'relief-size-r3')/region
        new=base/'shadow-receiver-r1'/region
        before=load(old/'report.json');after=load(new/'report.json')
        bf=before['effective']['fixture'];af=after['effective']['fixture']
        for key in ('real_map','terrain','scenarios','viewport','tile_count','packs','settings'):
            assert bf.get(key)==af.get(key),(region,key)
        assert before['effective']['pack_hash']==after['effective']['pack_hash']
        if 'real_map' in af:real_map.validate_provenance(af)
        elif region!='combinedvolcano':raise AssertionError('Unexpected synthetic fixture')
        bm=dict(before['effective']['module']);am=dict(after['effective']['module'])
        for m in (bm,am):m.pop('id');m.pop('shader')
        assert bm==am,(region,'module geometry settings')
        oldjobs=load(old/'batch.json');newjobs=load(new/'batch.json')
        record={'region':region,'synthetic':region=='combinedvolcano','frames':[]}
        for index,frame in enumerate(after['outputs']):
            previous=before['outputs'][index]
            for k in ('hour','zoom','offset'):assert frame[k]==previous[k]
            assert sha(ROOT/frame['image'])==frame['sha256']
            assert sha(ROOT/previous['image'])==previous['sha256']
            assert sha(Path(oldjobs[index][0]))==sha(Path(newjobs[index][0])),(region,'packet changed')
            assert frame['source_metadata']['sha256']==previous['source_metadata']['sha256'],(region,'source or placement changed')
            rect=[520,390,1160,710] if region=='longcoast' else [360,220,1000,540]
            if frame['zoom']==2:rect=[0,0,808 if region=='longcoast' else 680,444 if region=='longcoast' else 400]
            w,h=rect[2]-rect[0],rect[3]-rect[1]
            sheet=Image.new('RGB',(w*2+24,h+48),(23,28,33));draw=ImageDraw.Draw(sheet)
            for i,(folder,label) in enumerate([(old,'previous best'),(new,'receiver correction')]):
                source=folder/f"h{frame['hour']:02}-z{frame['zoom']}-pan00.bmp"
                sheet.paste(Image.open(source).convert('RGB').crop(rect),(i*(w+24),32))
                draw.text((i*(w+24)+8,10),f"{region} | {label} | native zoom {frame['zoom']}",fill='white',font=ImageFont.load_default())
            filename=f"{region}-h{frame['hour']:02}-z{frame['zoom']}-comparison.png";sheet.save(dest/filename)
            record['frames'].append({'hour':frame['hour'],'zoom':frame['zoom'],
                'before_sha256':previous['sha256'],'after_sha256':frame['sha256'],
                'packet_sha256':sha(Path(newjobs[index][0])),'source_metadata_sha256':frame['source_metadata']['sha256'],
                'geometry_materials_placement_shadow_field_identical':True,
                'crop':rect,'resampled':False,'comparison':(dest/filename).relative_to(ROOT).as_posix(),
                'difference':difference(ROOT/previous['image'],ROOT/frame['image'])})
        evidence['results'].append(record)
    p=V2/'audits/beauty/SHADOW_RECEIVER_r1_EVIDENCE.json';p.write_text(json.dumps(evidence,indent=2)+'\n')
    print(f'PASS {sum(len(r["frames"]) for r in evidence["results"])} matched frames; packet geometry, source materials, placement and shadow fields identical')

if __name__=='__main__':main()
