"""Isolate visible bank-rock changes and verify that other source objects stay fixed."""
import json
from pathlib import Path
import sys
V2=Path(__file__).resolve().parents[1];ROOT=V2.parents[2]
sys.path.insert(0,str(V2/'qa'))
from verify_gameplay_terrain import load,sha,difference

def objects(metadata):
    rocks={m.get('vertices_sha256') for m in metadata['meshes'] if m.get('id','').startswith('terrain/river/rock/')}
    banks=[];others=[]
    for source in metadata['instances']:
        instance=dict(source);instance.pop('id',None)
        (banks if instance.get('mesh_sha256') in rocks else others).append(instance)
    return banks,others

def main():
    from PIL import Image,ImageDraw
    base=V2/'audits/beauty/out';dest=base/'river-corridor-r3/review';dest.mkdir(parents=True,exist_ok=True)
    evidence={'schema':'c3x.bank_rock_evidence.v1','approval':None,'regions':[]}
    for region in ('coastal','inland','wilderness','freshcanopy'):
        before=load(base/'river-corridor-r2'/region/'report.json')
        after=load(base/'river-corridor-r3'/region/'report.json')
        for key in ('real_map','terrain','scenarios','viewport','tile_count','packs','settings'):
            assert before['effective']['fixture'][key]==after['effective']['fixture'][key]
        assert before['effective']['pack_hash']==after['effective']['pack_hash']
        assert before['effective']['shader_hashes']==after['effective']['shader_hashes']
        bm,am=dict(before['effective']['module']),dict(after['effective']['module'])
        for m in (bm,am):m.pop('id');m.pop('shader')
        assert am.pop('river_bank_rocks')==1 and bm==am
        record={'region':region,'frames':[]}
        for old,new in zip(before['outputs'],after['outputs']):
            for k in ('hour','zoom','offset'):assert old[k]==new[k]
            for frame in (old,new):assert sha(ROOT/frame['image'])==frame['sha256']
            a,b=(load(ROOT/f['source_metadata']['path']) for f in (old,new))
            assert a['textures']==b['textures']
            ar,ao=objects(a);br,bo=objects(b);assert ao==bo
            # Existing instances keep source choice, size and rotation. Unsafe
            # candidates can be omitted, never replaced with arbitrary new art.
            keep=('mesh_sha256','source_uniform_scale','yaw_radians')
            for rock in br:assert any(all(rock[k]==previous[k] for k in keep) for previous in ar)
            rect=[360,220,1000,540] if new['zoom']==1 else [0,0,680,400]
            w,h=rect[2]-rect[0],rect[3]-rect[1]
            sheet=Image.new('RGB',(w*2+20,h+30),(25,28,31));draw=ImageDraw.Draw(sheet)
            for i,(frame,label) in enumerate(((old,'old edge placement'),(new,'actual river bank'))):
                sheet.paste(Image.open(ROOT/frame['image']).convert('RGB').crop(rect),(i*(w+20),30))
                draw.text((i*(w+20)+10,8),region+' | '+label,fill='white')
            output=dest/f"{region}-h{new['hour']:02}-z{new['zoom']}.png";sheet.save(output)
            record['frames'].append({'hour':new['hour'],'zoom':new['zoom'],
                'before_sha256':old['sha256'],'after_sha256':new['sha256'],
                'rocks_before':ar,'rocks_after':br,'all_other_source_instances_identical':True,
                'difference':difference(ROOT/old['image'],ROOT/new['image']),
                'comparison':output.relative_to(ROOT).as_posix(),'resampled':False})
        evidence['regions'].append(record)
    (V2/'audits/beauty/RIVER_BANK_ROCK_r3_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('PASS 16 bank-rock pairs; other source objects, art, material and camera settings unchanged')

if __name__=='__main__':main()
