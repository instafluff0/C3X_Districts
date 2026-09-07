"""Check combined river frames and report pixel changes without granting approval."""
import json
from pathlib import Path
import sys

V2=Path(__file__).resolve().parents[1]; ROOT=V2.parents[2]
sys.path.insert(0,str(V2/'qa'))
from verify_gameplay_terrain import load,sha,difference


def main():
    from PIL import Image,ImageDraw
    base=V2/'audits/beauty/out';dest=base/'river-corridor-r2/review'
    dest.mkdir(parents=True,exist_ok=True)
    evidence={'schema':'c3x.river_corridor_evidence.v1','approval':None,
              'visual_accepted':False,'regions':[]}
    for region in ('coastal','inland','wilderness','freshcanopy'):
        old=base/'canopy-variation-r1'/region;new=base/'river-corridor-r2'/region
        before,after=load(old/'report.json'),load(new/'report.json')
        for key in ('real_map','terrain','scenarios','viewport','tile_count','packs','settings'):
            assert before['effective']['fixture'][key]==after['effective']['fixture'][key],(region,key)
        assert before['effective']['pack_hash']==after['effective']['pack_hash']
        bm,am=dict(before['effective']['module']),dict(after['effective']['module'])
        for m in (bm,am):m.pop('id');m.pop('shader')
        assert am.pop('river_corridor')==1 and bm==am
        record={'region':region,'frames':[]}
        for previous,frame in zip(before['outputs'],after['outputs']):
            for key in ('hour','zoom','offset'):assert previous[key]==frame[key]
            for f in (previous,frame):
                assert sha(ROOT/f['image'])==f['sha256']
                assert sha(ROOT/f['source_metadata']['path'])==f['source_metadata']['sha256']
            a,b=(load(ROOT/f['source_metadata']['path']) for f in (previous,frame))
            assert a['textures']==b['textures']
            assert all(mesh in a['meshes'] for mesh in b['meshes'])
            assert len(b['instances'])<=len(a['instances'])
            rect=[360,220,1000,540] if frame['zoom']==1 else [0,0,680,400]
            w,h=rect[2]-rect[0],rect[3]-rect[1]
            sheet=Image.new('RGB',(w*2+20,h+30),(23,27,30));draw=ImageDraw.Draw(sheet)
            for i,(f,label) in enumerate(((previous,'previous best'),(frame,'continuous river candidate'))):
                sheet.paste(Image.open(ROOT/f['image']).convert('RGB').crop(rect),(i*(w+20),30))
                draw.text((i*(w+20)+10,8),region+' | '+label,fill='white')
            output=dest/f"{region}-h{frame['hour']:02}-z{frame['zoom']}.png";sheet.save(output)
            record['frames'].append({'hour':frame['hour'],'zoom':frame['zoom'],
                'before_sha256':previous['sha256'],'after_sha256':frame['sha256'],
                'source_textures_unchanged':True,'remaining_meshes_unchanged':True,
                'instances_before':len(a['instances']),'instances_after':len(b['instances']),
                'crop':rect,'resampled':False,'comparison':output.relative_to(ROOT).as_posix(),
                'difference':difference(ROOT/previous['image'],ROOT/frame['image'])})
        evidence['regions'].append(record)
    # The shader-only mouth diagnosis must survive full composition unchanged.
    diagnostic=load(base/'river-material-r2/freshcanopy/report.json')
    composed=load(base/'river-corridor-r2/freshcanopy/report.json')
    assert [x['sha256'] for x in diagnostic['outputs']]==[x['sha256'] for x in composed['outputs']]
    evidence['mouth_material_replay_matches_composition']=True
    (V2/'audits/beauty/RIVER_CORRIDOR_r2_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('PASS 16 fixed combined frames; source art preserved; shader-only mouth improvement survives composition')


if __name__=='__main__':main()
