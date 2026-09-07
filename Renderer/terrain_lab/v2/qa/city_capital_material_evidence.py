"""Verify complete capital material composition and isolated lake reflections."""
import json
import subprocess
from PIL import Image, ImageDraw
from city_growth_evidence import ROOT, V2, OUT, FIX, read, sha, placement, clearance
from city_growth_hierarchy_evidence import difference
from city_facade_light_evidence import delta
from city_connected_growth_evidence import boxes, clipped_area
from city_growth_layout import connected
from city_scene_pass import executable, Cache
from city_light_buffer_probe import payload
from mesh_fingerprint import geometry_digest

BASE=OUT/'city-capital-materials-r1'


def augmentation(revision):
    p=next((FIX/f'city-scene-r{revision}').glob('*/augmentation.json'))
    return p,read(p)


def main():
    cache=Cache(V2/'app/.cache')
    executables={k:executable(V2/'qa'/n,cache) for k,n in {
        'terrain':'city_terrain_contract.cpp','geometry':'city_geometry_material_contract.cpp',
        'frame':'city_shadow_frame_contract.cpp','light':'frame_data_contract.cpp',
        'ground':'settlement_ground_contract.cpp','metal':'city_metalness_contract.cpp'}.items()}
    def check(name,*args):
        return json.loads(subprocess.check_output([str(executables[name]),*map(str,args)],text=True))
    old_path,old=augmentation(22);path,current=augmentation(101)
    assert [placement(i) for i in old['instances']]==[placement(i) for i in current['instances']]
    for key in ('pool','source_biq_sha256','anchor_tile','projection','emissive_gain','emissive_uv'):
        assert old[key]==current[key]
    normals=read(FIX/'city-capital-materials-r1/palace-normals.json')
    pack=ROOT/'Renderer/packs/CityPalacesNormalized';asset='city/palace/root/0d0c35f4a4c9651a'
    manifest=read(pack/'manifest.json');landmark=read(pack/manifest['assets'][asset]['landmark'])
    for p in landmark['components']['geometry']:
        mesh=read(pack/p);assert geometry_digest(mesh)==normals['meshes'][mesh['asset_id']]['geometry_digest']
    extra=read(FIX/'city-capital-materials-r1/palace-extra/source-evidence.json')
    for r in extra:assert sha(ROOT/r['texture'])==r['dds_sha256']
    old_lights=read(OUT/'city-facade-light-r3/capital/lights.json')
    fixed_lights=read(FIX/'city-capital-materials-r1/frozen-lights.json')
    for k in ('lights','blockers','gain','z_metric','texture_sha256'):assert old_lights[k]==fixed_lights[k]
    assert (BASE/'ground/ground.bin').read_bytes()==(OUT/'city-settlement-ground-r2/capital/ground.bin').read_bytes()
    cases={}
    for revision,prefix in [(101,''),(102,'inland-'),(103,'holdout-')]:
        a_path,a=augmentation(revision);surface=read(a_path.parent/'surface.json')
        assert surface['region']['region']['extent']==[10,10]
        assert a['generator_profile']['era_policy']=='single_current_era_user_preference'
        assert not a['source_normals']['unmapped']
        for k in ('source_normals','extra_materials'):
            assert sha(ROOT/a[k]['mapping'])==a[k]['sha256']
        raw=OUT/f'city-scene-r{revision}'/a_path.parent.name;raw_report=read(raw/'report.json')
        natural=read(OUT/'river-corridor-r3'/a['benchmark_region']/'report.json')
        lighting=BASE/(prefix+'lights');binding=read(lighting/'binding.json');lights=read(ROOT/binding['lights'])
        assert (lighting/'lights.bin').read_bytes()==payload(lights)
        assert lights['augmentation_sha256']==sha(a_path)
        ground=BASE/(prefix+'ground');settlement=read(ground/'settlement.json')
        environment=read(BASE/(prefix+'environment')/'experiment.json')
        checks=[]
        for index,row in enumerate(raw_report['outputs']):
            natural_row=next(r for r in natural['outputs'] if (r['hour'],r['zoom'])==(row['hour'],row['zoom']))
            checks.append(check('terrain',ROOT/natural_row['packet'],ROOT/row['packet']))
            if revision==101:
                before=next(r for r in read(OUT/'city-scene-r22'/old_path.parent.name/'report.json')['outputs'] if (r['hour'],r['zoom'])==(row['hour'],row['zoom']))
                checks.append(check('geometry',ROOT/before['packet'],ROOT/row['packet']))
            if a['shadow_frame_report']:
                reference=next(r for r in read(ROOT/a['shadow_frame_report']['path'])['outputs'] if (r['hour'],r['zoom'])==(row['hour'],row['zoom']))
                checks.append(check('frame',ROOT/row['packet'],ROOT/reference['packet'],ROOT/row['packet']))
            for record in (binding['packets'][index],settlement['packets'][index],environment['packets'][index]):
                for k in ('original','output'):assert sha(ROOT/record[k])==record[k+'_sha256']
            r=binding['packets'][index];checks.append(check('light',ROOT/r['original'],ROOT/r['output'],lighting/'lights.bin'))
            r=settlement['packets'][index];checks.append(check('ground',ROOT/r['original'],ROOT/r['output'],r['insertion_draw']))
            r=environment['packets'][index];checks.append(check('metal',ROOT/r['original'],ROOT/r['output']))
        coverage={}
        if revision!=101:
            b=boxes(a);houses=[box for i,box in zip(a['instances'],b) if i['slot']!='capital']
            assert connected(houses[:4],.08) and connected(houses,.08)
            ex=read(ROOT/a['river_exclusion']['path'])
            hits=[i['slot'] for i,box in zip(a['instances'],b) if any(clipped_area(p,[v+(-.024 if j<2 else .024) for j,v in enumerate(box)])>1e-12 for p in ex['polygons'])]
            assert not hits
            coverage={'clearance':clearance(a),'connected_house_prefixes':[4,7],'river_overlap_slots':hits,'search':a['layout_attempts']}
        cases[str(revision)]={'augmentation_sha256':sha(a_path),'region':a['benchmark_region'],'anchor':a['anchor_tile'],
            'tile_count':100,'body_count':len(a['instances']),'lights':len(lights['lights']),
            'normal_meshes_applied':len(a['source_normals']['applied']),'checks':checks,'coverage':coverage}
    windows={}
    for name in ('ground','environment','reflection-off','inland-environment','holdout-environment'):
        render=BASE/name/'render';report=read(render/'report.json');w=read(BASE/('windows-'+name)/'evidence.json')
        assert len(w['results'])==2
        for index,r in enumerate(w['results']):
            assert r['metrics']['pass']
            for k,p in [('packet_sha256',ROOT/report['packets'][index]['path']),('shader_sha256',render/'shaders/source.hlsl'),
                        ('reflection_sha256',render/'shaders/reflection/source.hlsl'),('post_sha256',render/'postprocess/source.hlsl'),
                        ('d3d11_sha256',BASE/('windows-'+name)/r['frame'])]:assert r[k]==sha(p)
        windows[name]=w
    pixels={}
    previous=OUT/'city-settlement-ground-r2/capital/render'
    for name in ('ground','environment'):
        pixels[name]=[difference(previous/f'h{h:02}-z1-pan00.png',BASE/name/'render'/f'h{h:02}-z1-pan00.png',(700,260,1100,550)) for h in (12,0)]
    on=BASE/'environment/render';off=BASE/'reflection-off/render'
    assert read(on/'report.json')['packets']==read(off/'report.json')['packets']
    for shader in ('shaders/source.hlsl','postprocess/source.hlsl'):assert sha(on/shader)==sha(off/shader)
    water=[difference(off/f'h{h:02}-z1-pan00.png',on/f'h{h:02}-z1-pan00.png',(800,470,930,540)) for h in (12,0)]
    assert water[1]['changed_pixels_gt_2']>500 and water[1]['max_channel_delta']>60
    legacy=delta(previous/'h00-z1-pan00.png',on/'h00-z1-pan00.png',(858,491,885,501));assert legacy['max_channel_delta']<=1
    evidence={'classification':'Provisional capital material restoration and composed coverage; no general city or milestone approval',
        'exact_preserved_capital_instances':8,'source_palace_meshes':len(normals['meshes']),
        'source_palace_primitives':len(normals['evidence']),'source_palace_extra_channels':extra,
        'frozen_prior_light_proxies':31,'settlement_ground_wire_exact':True,'cases':cases,'windows':windows,
        'matched_pixels':pixels,'isolated_city_water_reflection':water,'prior_night_lake_roi':legacy,
        'untuned_region':{'revision':103,'region':'freshcanopy','terrain_origin':[76,58],'anchor':[5,2],
            'selection':'Flat nonriver grass tile selected from raw terrain cells before any city rendering; unchanged material/placement recipe, no local retries'},
        'remaining':['Exact source environment calibration and LEAN1 variance; hemisphere reflection remains authored approximation',
            'The prior Asian dielectric environment trial stays unselected',
            'Broader single-era culture/era/size matrix and more varied architectural/open-ground composition',
            'Older capital preserves legacy clearance policy; new inland/holdout additionally check river/forest envelopes',
            'Unresolved separate coastal-capital and medium/large river placement; no native or manual gate advancement']}
    target=V2/'audits/beauty/CITY_CAPITAL_MATERIAL_EVIDENCE.json';target.write_text(json.dumps(evidence,indent=2)+'\n')
    # Keep gameplay pixel scale; the lake control is separate diagnostic evidence.
    canvas=Image.new('RGB',(760,560));draw=ImageDraw.Draw(canvas)
    for col,(folder,label) in enumerate([(previous,'Previous capital'),(on,'Restored capital materials')]):
        for row,h in enumerate((12,0)):
            canvas.paste(Image.open(folder/f'h{h:02}-z1-pan00.png').crop((710,270,1090,530)),(380*col,280*row+20))
            draw.text((380*col+5,280*row+4),label,fill='white')
    canvas.save(BASE/'selected-capital-native.png');print(target.relative_to(ROOT))


if __name__=='__main__':main()
