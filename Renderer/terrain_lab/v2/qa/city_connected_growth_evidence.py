"""Recheck connected-city pixels, preserved growth, river clearance and composition."""
import json
import subprocess
from PIL import Image,ImageDraw
from city_growth_evidence import ROOT,V2,OUT,FIX,read,sha,placement,clearance
from city_growth_hierarchy_evidence import difference
from city_scene_pass import executable,Cache
from city_growth_layout import connected
from city_light_buffer_probe import payload

BASE=OUT/'city-connected-growth-r1'
CASES={'asian-large-before':62,'asian-large':65,'ancient-large':70,
       'holdout-small-before':69,'holdout-small':72,'untuned-coast':73}


def augmentation(revision):
    path=next((FIX/f'city-scene-r{revision}').glob('*/augmentation.json'))
    return path,read(path)


def boxes(a):
    return [[v+i['offset'][j%2] for j,v in enumerate(i['local_bounds'])] for i in a['instances']]


def area(boxes):
    return (max(b[2] for b in boxes)-min(b[0] for b in boxes))*(max(b[3] for b in boxes)-min(b[1] for b in boxes))


def clipped_area(polygon,box):
    # Independent rectangle clipping, rather than the planner's separating axes.
    p=polygon
    for axis,bound,sign in [(0,box[0],1),(0,box[2],-1),(1,box[1],1),(1,box[3],-1)]:
        result=[]
        for i,a in enumerate(p):
            b=p[(i+1)%len(p)];da=(a[axis]-bound)*sign;db=(b[axis]-bound)*sign
            if da>=0:result.append(a)
            if (da>=0)!=(db>=0):
                t=da/(da-db);result.append([a[j]+(b[j]-a[j])*t for j in range(2)])
        p=result
    return abs(sum(v[0]*p[(i+1)%len(p)][1]-p[(i+1)%len(p)][0]*v[1] for i,v in enumerate(p)))*.5


def main():
    cache=Cache(V2/'app/.cache')
    terrain=executable(V2/'qa/city_terrain_contract.cpp',cache)
    frame=executable(V2/'qa/city_shadow_frame_contract.cpp',cache)
    light=executable(V2/'qa/frame_data_contract.cpp',cache)
    export=executable(V2/'qa/river_city_exclusion.cpp',cache)
    cases={};all_a={};exclusions={}
    for name,revision in CASES.items():
        path,a=augmentation(revision);all_a[name]=a
        folder=BASE/name;binding=read(folder/'binding.json');render=folder/'render';report=read(render/'report.json')
        lights=read(ROOT/binding['lights']);assert sha(ROOT/binding['lights'])==binding['lights_sha256']
        assert lights['augmentation_sha256']==sha(path)
        assert (folder/'lights.bin').read_bytes()==payload(lights)
        assert sha(folder/'lights.bin')==binding['payload_sha256']
        assert a['generator_profile']['era_policy']=='single_current_era_user_preference'
        assert a['graduated_growth'] and a['uniform_scale_factor']==1.5
        assert read(path.parent/'surface.json')['region']['region']['extent']==[10,10]
        b=boxes(a);counts=a['stage_component_counts'];prefixes={str(n):connected(b[:n],.08) for n in counts if n<=len(b)}
        if name!='asian-large-before':assert all(prefixes.values())
        series={'freshshadow':'shadow-receiver-r1','freshwater':'water-natural-foundation'}.get(a['benchmark_region'],'river-corridor-r3')
        natural=read(OUT/series/a['benchmark_region']/'report.json')
        source_rows=[r for r in natural['outputs'] if r['zoom']==1]
        key=a['benchmark_region']
        if key not in exclusions:
            target=BASE/(key+'-bank-exclusion.json')
            subprocess.run([str(export),str(ROOT/source_rows[0]['packet']),*map(str,a['anchor_tile']),'12.4',str(target)],check=True)
            exclusions[key]=read(target)
        ex=exclusions[key];hits=[]
        for i,box in enumerate(b):
            padded=[v+(-.012 if j<2 else .012) for j,v in enumerate(box)]
            if any(clipped_area(p,padded)>1e-12 for p in ex['polygons']):hits.append(i)
        if name!='holdout-small-before':assert not hits,(name,hits)
        checks=[]
        for index,record in enumerate(binding['packets']):
            for field in ('original','output'):assert sha(ROOT/record[field])==record[field+'_sha256']
            checks.append(json.loads(subprocess.check_output([str(terrain),str(ROOT/source_rows[index]['packet']),str(ROOT/record['original'])],text=True)))
            checks.append(json.loads(subprocess.check_output([str(light),str(ROOT/record['original']),str(ROOT/record['output']),str(folder/'lights.bin')],text=True)))
            if a['shadow_frame_report']:
                ref=read(ROOT/a['shadow_frame_report']['path'])['outputs'][index]['packet']
                checks.append(json.loads(subprocess.check_output([str(frame),str(ROOT/record['original']),str(ROOT/ref),str(ROOT/record['original'])],text=True)))
        windows=read(BASE/f'windows-{name}/evidence.json');assert len(windows['results'])==2
        for index,row in enumerate(windows['results']):
            assert row['metrics']['pass']
            for field,p in [('d3d11_sha256',BASE/f'windows-{name}'/row['frame']),('packet_sha256',ROOT/report['packets'][index]['path']),
                            ('shader_sha256',render/'shaders/source.hlsl'),('reflection_sha256',render/'shaders/reflection/source.hlsl'),
                            ('post_sha256',render/'postprocess/source.hlsl')]:assert row[field]==sha(p)
        cases[name]={'revision':revision,'augmentation_sha256':sha(path),'body_count':len(b),'bounding_area_tiles':area(b),
                     'connected_prefixes_at_gap_0_08':prefixes,'river_bank_overlap_slots':hits,'exclusion_polygon_count':ex['polygon_count'],
                     'clearance':clearance(a),'lights':len(lights['lights']),'blockers':len(lights['blockers']),
                     'packet_checks':checks,'windows':windows,'search':a['layout_attempts'][0]}
    for name,previous in [('asian-large',60),('ancient-large',61)]:
        _,a=augmentation(previous);b=all_a[name]
        assert [placement(i) for i in a['instances']]==[placement(i) for i in b['instances'][:16]]
        cases[name]['exact_preserved_medium_prefix']=16
    for before,after,roi in [('asian-large-before','asian-large',(630,280,1050,570)),
                              ('holdout-small-before','holdout-small',(480,310,880,570))]:
        a,b=all_a[before],all_a[after]
        for key in ('pool','size','uniform_scale_factor','source_biq_sha256','anchor_tile','projection','textures',
                    'emissive_gain','emissive_uv','source_normals','source_surface','extra_materials','grounding','compound_ground'):
            assert a[key]==b[key],key
        assert [(i['asset'],i['scale']) for i in a['instances']]==[(i['asset'],i['scale']) for i in b['instances']]
        assert cases[before]['lights']==cases[after]['lights']
        cases[after]['matched_pixels']=[difference(BASE/before/'render'/f'h{h:02}-z1-pan00.png',BASE/after/'render'/f'h{h:02}-z1-pan00.png',roi) for h in (12,0)]
    reduction=1-cases['asian-large']['bounding_area_tiles']/cases['asian-large-before']['bounding_area_tiles']
    assert reduction>.2
    rejected={str(r):read(next((FIX/f'city-scene-r{r}').glob('*/growth-search.json'))) for r in (66,67,71)}
    assert all(v['status']=='budget_exhausted' for v in rejected.values())
    river_regressions={}
    for r in (64,68,69):
        _,a=augmentation(r)
        river_regressions[str(r)]=[i for i,b in enumerate(boxes(a)) if any(clipped_area(p,b)>1e-12 for p in exclusions['freshshadow']['polygons'])]
        assert river_regressions[str(r)]
    evidence={'classification':'Provisional connected-city and river-clearance gains; no human acceptance or native promotion',
              'cases':cases,'asian_large_bounding_area_reduction_fraction':reduction,'preserved_failed_searches':rejected,
              'superseded_river_overlap_candidates':river_regressions,
              'untuned_region':{'revision':73,'region':'freshwater','anchor':[8,3],'terrain_origin':[52,30],'tile_count':100,
                                'selection':'Flat coastal land chosen from existing terrain cells before city rendering; same recipe, no local retries',
                                'interpretation':'Generalization witness, no matched prior city improvement or new reflection-quality claim'},
              'remaining':['Medium/large city fit beside the holdout river; budget exhaustion is not infeasibility',
                           'Broader single-era capitals and landmark hierarchy from the 47-root palace pack',
                           'Source material/environment response and less repetitive horizontal neighborhoods',
                           'Existing capital-lake reflection witness preserved; no new water-reflection quality claim',
                           'All human visual and native milestone gates remain open']}
    target=V2/'audits/beauty/CITY_CONNECTED_GROWTH_EVIDENCE.json';target.write_text(json.dumps(evidence,indent=2)+'\n')
    image=Image.new('RGB',(760,260));draw=ImageDraw.Draw(image)
    for i,h in enumerate((12,0)):
        image.paste(Image.open(BASE/'untuned-coast/render'/f'h{h:02}-z1-pan00.png').crop((600,400,980,640)),(380*i,20))
        draw.text((380*i+5,4),'Untuned 100-tile coast | '+('day' if h else 'night'),fill='white')
    image.save(BASE/'untuned-coast-native.png')
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
