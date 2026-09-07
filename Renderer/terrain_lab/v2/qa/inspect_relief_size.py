"""Matched visual evidence for relief size; checks never grant art approval."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'Renderer/terrain_lab/v2';OUT=V2/'audits/beauty/out'
sys.path.insert(0,str(V2/'app'));sys.path.insert(0,str(V2/'qa'))
import real_map
from verify_gameplay_terrain import load,sha,difference

def previous(region):
    return OUT/('relief-size-baseline' if region in ('freshrelief','combinedvolcano') else 'coast-pass-rocks-r8')/region

def verify(region,revision):
    before=load(previous(region)/'report.json');after=load(OUT/('relief-size-'+revision)/region/'report.json')
    bf=before['effective']['fixture'];af=after['effective']['fixture']
    for key in ('real_map','terrain','scenarios','viewport','tile_count','packs','settings'):
        assert bf.get(key)==af.get(key),(region,key)
    assert before['effective']['module']['projection']==after['effective']['module']['projection']
    assert before['effective']['pack_hash']==after['effective']['pack_hash']
    if 'real_map' in af:real_map.validate_provenance(af)
    else:
        assert region=='combinedvolcano'
        provenance=load(V2/'fixtures/beauty/relief-size-foundation/combinedvolcano/provenance.json')
        assert sha(ROOT/af['terrain'])==provenance['terrain_sha256'] and len(provenance['changes'])==1
    records=[]
    for frame in after['outputs']:
        old=next(f for f in before['outputs'] if all(f[k]==frame[k] for k in ('hour','zoom','offset')))
        assert sha(ROOT/frame['image'])==frame['sha256']
        assert sha(ROOT/frame['source_metadata']['path'])==frame['source_metadata']['sha256']
        oldmeta=load(ROOT/old['source_metadata']['path']);meta=load(ROOT/frame['source_metadata']['path'])
        oldplants=[i for i in oldmeta['instances'] if 'source_xy_anchor' in i]
        plants=[i for i in meta['instances'] if 'source_xy_anchor' in i]
        assert len(oldplants)==len(plants)
        for a,b in zip(oldplants,plants):
            for key in ('mesh_sha256','id','source_xy_anchor','source_uniform_scale','yaw_radians'):
                assert a[key]==b[key],(region,key)
        textures={t['path']:t['sha256'] for t in meta['textures']}
        assert textures=={t['path']:t['sha256'] for t in oldmeta['textures']}
        records.append({'hour':frame['hour'],'zoom':frame['zoom'],
            'before_sha256':old['sha256'],'after_sha256':frame['sha256'],
            'source_vegetation_instances':len(plants),'vegetation_xy_scale_yaw_preserved':True,
            'grounding_heights_changed':sum(a['ground_authoring_height']!=b['ground_authoring_height'] for a,b in zip(oldplants,plants)),
            'difference':difference(ROOT/old['image'],ROOT/frame['image'])})
    return {'region':region,'synthetic':region=='combinedvolcano',
        'before_identity':before['render_identity'],'after_identity':after['render_identity'],'frames':records}

def present(regions,revision):
    from PIL import Image,ImageDraw,ImageFont
    dest=OUT/('relief-size-'+revision)/'review';dest.mkdir(parents=True,exist_ok=True)
    crop=[360,220,1000,540];records=[]
    for region in regions:
        box=[520,390,1160,710] if region=='longcoast' else crop
        after=OUT/('relief-size-'+revision)/region
        for hour,phase in [(12,'day'),(0,'night')]:
            for zoom,rect in [(1,box),(2,[0,0,808 if region=='longcoast' else 680,444 if region=='longcoast' else 400])]:
                w,h=rect[2]-rect[0],rect[3]-rect[1]
                sheet=Image.new('RGB',(w*2+24,h+48),(23,28,33));draw=ImageDraw.Draw(sheet)
                for i,(folder,label) in enumerate([(previous(region),'previous best'),(after,'larger source bodies')]):
                    image=folder/f'h{hour:02}-z{zoom}-pan00.png'
                    sheet.paste(Image.open(image).convert('RGB').crop(rect),(i*(w+24),32))
                    draw.text((i*(w+24)+8,10),f'{region} | {label} | native zoom {zoom}',fill='white',font=ImageFont.load_default())
                filename=f'{region}-{phase}-z{zoom}-comparison.png';sheet.save(dest/filename)
                records.append({'file':filename,'crop':rect,'resampled':False,'hour':hour,'zoom':zoom})
    (dest/'crops.json').write_text(json.dumps({'visual_accepted':False,'records':records},indent=2)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--revision',default='r3')
    p.add_argument('--regions',nargs='+',default=['coastal','inland','wilderness','longcoast','freshcoast','freshrelief','combinedvolcano'])
    p.add_argument('--present',action='store_true')
    a=p.parse_args()
    evidence={'schema':'c3x.relief_size_evidence.v1','revision':a.revision,
        'approval':None,'visual_accepted':False,'results':[verify(r,a.revision) for r in a.regions]}
    path=V2/'audits/beauty'/('RELIEF_SIZE_'+a.revision+'_EVIDENCE.json')
    path.write_text(json.dumps(evidence,indent=2)+'\n')
    if a.present:present(a.regions,a.revision)
    print(path.relative_to(ROOT))

if __name__=='__main__':main()
