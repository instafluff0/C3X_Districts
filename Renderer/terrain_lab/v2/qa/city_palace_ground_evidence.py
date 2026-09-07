"""Check palace paving correction against the preserved combined city scenes."""
import json
import struct
import subprocess
from PIL import Image, ImageDraw
from city_growth_evidence import ROOT, V2, OUT, read, sha
from city_growth_hierarchy_evidence import difference
from city_scene_pass import executable, Cache

OLD=OUT/'city-capital-materials-r1'
NEW=OUT/'city-palace-ground-alignment-r1'


def wire(path):
    data=path.read_bytes();magic,count=struct.unpack_from('<II',data)
    assert magic==0x31524753 and len(data)==8+count*52
    return list(struct.iter_unpack('<13f',data[8:]))


def main():
    cache=Cache(V2/'app/.cache')
    ground=executable(V2/'qa/settlement_ground_contract.cpp',cache)
    metal=executable(V2/'qa/city_metalness_contract.cpp',cache)
    cases={}
    for prefix in ('inland-','','holdout-'):
        before=OLD/(prefix+'ground');after=NEW/(prefix+'ground')
        original=read(before/'settlement.json');current=read(after/'settlement.json')
        for key in ('augmentation_sha256','atlas','atlas_uv','texels_per_tile','tile_period','margin','feather','boxes','surface_sha256'):
            assert original[key]==current[key],key
        a=read(ROOT/current['augmentation']);s=read((ROOT/current['augmentation']).parent/'surface.json')
        assert s['region']['region']['extent']==[10,10]
        assert a['generator_profile']['era_policy']=='single_current_era_user_preference'
        assert current['capital_footprint']=='source-hull' and len(current['coverage_polygons'])==1
        for source in current['footprint_sources']:
            for path,digest in source['files'].items():assert sha(ROOT/path)==digest
        # Geometry at every retained underlay vertex is unchanged, including UVs.
        # Only coverage can decrease; no added paving may occupy previously dry ground.
        old_vertices=wire(before/'ground.bin');new_vertices=wire(after/'ground.bin')
        key=lambda v:v[:8]+v[9:]
        old_lookup={key(v):v[8] for v in old_vertices}
        assert all(key(v) in old_lookup and v[8]<=old_lookup[key(v)]+1e-6 for v in new_vertices)
        palace_index=next(i for i,instance in enumerate(a['instances']) if instance['slot']=='capital')
        palace_box=current['boxes'][palace_index]
        assert current['coverage_boxes']==[b for i,b in enumerate(current['boxes']) if i!=palace_index]
        hull=current['coverage_polygons'][0]
        area=abs(sum(p[0]*q[1]-q[0]*p[1] for p,q in zip(hull,hull[1:]+hull[:1])))/2
        box_area=(palace_box[2]-palace_box[0])*(palace_box[3]-palace_box[1])
        assert area<box_area*.7
        controls=[]
        experiment=read(NEW/(prefix+'environment')/'experiment.json')
        for index,r in enumerate(current['packets']):
            for name in ('original','output'):assert sha(ROOT/r[name])==r[name+'_sha256']
            assert r['original_sha256']==original['packets'][index]['original_sha256']
            controls.append(json.loads(subprocess.check_output([str(ground),str(ROOT/r['original']),str(ROOT/r['output']),str(r['insertion_draw'])],text=True)))
            r=experiment['packets'][index]
            controls.append(json.loads(subprocess.check_output([str(metal),str(ROOT/r['original']),str(ROOT/r['output'])],text=True)))
        old_render=OLD/(prefix+'environment')/'render';render=NEW/(prefix+'environment')/'render'
        for shader in ('shaders/source.hlsl','shaders/reflection/source.hlsl','postprocess/source.hlsl'):
            assert sha(old_render/shader)==sha(render/shader)
        site=s['samples'][a['instances'][palace_index]['sample_start']]
        x,y=round(site['screen_x']),round(site['screen_y'])
        pixels=[difference(old_render/f'h{h:02}-z1-pan00.png',render/f'h{h:02}-z1-pan00.png',(x-70,y-55,x+70,y+65)) for h in (12,0)]
        windows=read(NEW/('windows-'+prefix+'environment')/'evidence.json')
        report=read(render/'report.json');assert len(windows['results'])==2
        for index,r in enumerate(windows['results']):
            assert r['metrics']['pass']
            for k,p in [('packet_sha256',ROOT/report['packets'][index]['path']),('shader_sha256',render/'shaders/source.hlsl'),
                        ('reflection_sha256',render/'shaders/reflection/source.hlsl'),('post_sha256',render/'postprocess/source.hlsl'),
                        ('d3d11_sha256',NEW/('windows-'+prefix+'environment')/r['frame'])]:assert r[k]==sha(p)
        lake=None
        if not prefix:
            import numpy as np
            lake=[]
            for h in (12,0):
                crop=lambda folder:np.asarray(Image.open(folder/f'h{h:02}-z1-pan00.png').crop((825,490,925,535))).astype(int)
                maximum=int(abs(crop(old_render)-crop(render)).max());assert maximum<=1;lake.append(maximum)
        cases[prefix or 'coastal']={'augmentation':current['augmentation'],'source_footprint_area':area,'bounding_box_area':box_area,
            'removed_bounding_area_fraction':1-area/box_area,'old_ground_vertices':len(old_vertices),'new_ground_vertices':len(new_vertices),
            'original_city_geometry_uvs_materials_lighting_and_shadows_preserved':True,'packet_checks':controls,
            'pixels':pixels,'lake_roi_max_day_night':lake,'windows':windows}
        canvas=Image.new('RGB',(640,480));draw=ImageDraw.Draw(canvas)
        for col,(folder,label) in enumerate([(old_render,'Before'),(render,'Aligned palace paving')]):
            for row,h in enumerate((12,0)):
                canvas.paste(Image.open(folder/f'h{h:02}-z1-pan00.png').crop((x-140,y-180,x+180,y+40)),(col*320,row*240+20))
                draw.text((col*320+4,row*240+4),label+(' | day' if h else ' | night'),fill='white')
        canvas.save(NEW/(prefix+'native.png'))
    target=V2/'audits/beauty/CITY_PALACE_GROUND_ALIGNMENT_EVIDENCE.json'
    target.write_text(json.dumps({'classification':'Matched correction of authored palace underlay, not recovered source city generator',
        'cause':'Axis-aligned footprint box discarded the source foundation orientation',
        'correction':'Use the convex footprint of the transformed normalized palace body; keep previous atlas coordinates and margin',
        'cases':cases,'gates_advanced':False,'visual_approval':None},indent=2)+'\n')
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
