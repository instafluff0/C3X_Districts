"""Verify selected central, orthogonal capital compositions without advancing gates."""
import json
import math
import subprocess
from PIL import Image,ImageDraw
from city_growth_evidence import ROOT,V2,OUT,FIX,read,sha,clearance
from city_growth_hierarchy_evidence import difference
from city_scene_pass import executable,Cache,city
from city_connected_growth_evidence import boxes,clipped_area
from city_growth_layout import connected,shares_frontage,surround_sector
from settlement_ground import convex_hull,polygon_distance,footprint_alignment

BASE=OUT/'city-central-capital-r2'


def main():
    cache=Cache(V2/'app/.cache')
    names={'terrain':'city_terrain_contract.cpp','frame':'city_shadow_frame_contract.cpp',
           'light':'frame_data_contract.cpp','ground':'settlement_ground_contract.cpp','metal':'city_metalness_contract.cpp'}
    exes={k:executable(V2/'qa'/n,cache) for k,n in names.items()};cases={}
    def check(name,*args):return json.loads(subprocess.check_output([str(exes[name]),*map(str,args)],text=True))
    for name,revision,old_revision,prefix in [('inland',111,102,'inland-'),('holdout',112,103,'holdout-')]:
        path=next((FIX/f'city-scene-r{revision}').glob('*/augmentation.json'));a=read(path)
        old_path=next((FIX/f'city-scene-r{old_revision}').glob('*/augmentation.json'));old=read(old_path)
        s=read(path.parent/'surface.json');assert s['region']['region']['extent']==[10,10]
        assert a['projection']==old['projection'] and a['source_biq_sha256']==old['source_biq_sha256']
        assert a['generator_profile']['era_policy']=='single_current_era_user_preference'
        assert a['capital']['composition']=='central_surrounded' and a['grid_alignment']['enabled']
        palace=next(i for i in a['instances'] if i['slot']=='capital');center=palace['offset']
        assert center==a['capital']['center_offset']==[0,0]
        assert abs(palace['rotation']-math.pi/6)<1e-10
        b=boxes(a);houses=[i for i in a['instances'] if i['slot']!='capital'];pb=b[-1]
        assert [i['asset'] for i in a['instances']]==[i['asset'] for i in old['instances']]
        assert [i['scale'] for i in a['instances']]==[i['scale'] for i in old['instances']]
        for i in houses:assert surround_sector(i['slot'],*i['offset'],center)
        for count in (4,7):
            assert polygon_distance(*center,convex_hull([i['offset'] for i in houses[:count]]))<-.01
            assert connected([pb]+b[:count],.08)
        assert all(shares_frontage(pb,box,.08) for box in b[:4])
        axes=[]
        for i in a['instances']:
            pack=a['capital']['mapping']['pack'] if i['slot']=='capital' else a['pack']
            body=city.component(i['asset'],ROOT/pack)
            points=[city.rotate(v['position'],i['rotation'])[:2] for mesh,mat in body['parts'] if mat['alpha_mode']!='blend' for v in mesh['vertices']]
            residual=abs(math.degrees(footprint_alignment(points)));assert residual<.02
            axes.append({'slot':i['slot'],'residual_degrees':residual})
        exclusion=read(ROOT/a['river_exclusion']['path'])
        assert not any(clipped_area(poly,[v+(-.024 if j<2 else .024) for j,v in enumerate(box)])>1e-12 for poly in exclusion['polygons'] for box in b)
        coverage=clearance(a);checks=[]
        raw=read(OUT/f'city-scene-r{revision}'/path.parent.name/'report.json')
        natural=read(OUT/'river-corridor-r3'/a['benchmark_region']/'report.json')
        fixed=read(ROOT/a['shadow_frame_report']['path'])
        light=read(BASE/name/'lights/binding.json');ground=read(BASE/name/'ground/settlement.json');env=read(BASE/name/'environment/experiment.json')
        assert ground['capital_footprint']=='source-hull'
        for index,row in enumerate(raw['outputs']):
            match=lambda report:next(r for r in report['outputs'] if (r['hour'],r['zoom'])==(row['hour'],row['zoom']))
            checks.append(check('terrain',ROOT/match(natural)['packet'],ROOT/row['packet']))
            checks.append(check('frame',ROOT/row['packet'],ROOT/match(fixed)['packet'],ROOT/row['packet']))
            r=light['packets'][index];checks.append(check('light',ROOT/r['original'],ROOT/r['output'],BASE/name/'lights/lights.bin'))
            r=ground['packets'][index];checks.append(check('ground',ROOT/r['original'],ROOT/r['output'],r['insertion_draw']))
            r=env['packets'][index];checks.append(check('metal',ROOT/r['original'],ROOT/r['output']))
        render=BASE/name/'environment/render';previous=OUT/'city-palace-facade-alignment-r1'/(prefix+'environment')/'render'
        site=s['samples'][palace['sample_start']];x,y=round(site['screen_x']),round(site['screen_y'])
        pixels=[difference(previous/f'h{h:02}-z1-pan00.png',render/f'h{h:02}-z1-pan00.png',(x-170,y-190,x+180,y+90)) for h in (12,0)]
        w=read(BASE/('windows-'+name)/'evidence.json');report=read(render/'report.json');assert len(w['results'])==2
        for index,r in enumerate(w['results']):
            assert r['metrics']['pass']
            for k,p in [('packet_sha256',ROOT/report['packets'][index]['path']),('shader_sha256',render/'shaders/source.hlsl'),
                        ('reflection_sha256',render/'shaders/reflection/source.hlsl'),('post_sha256',render/'postprocess/source.hlsl'),
                        ('d3d11_sha256',BASE/('windows-'+name)/r['frame'])]:assert r[k]==sha(p)
        cases[name]={'augmentation':str(path.relative_to(ROOT)),'augmentation_sha256':sha(path),'tile_count':100,
            'center':center,'axis_alignment':axes,'surrounded_connected_growth_prefixes':[4,7],
            'clearance':coverage,'packet_checks':checks,'pixels':pixels,'windows':w}
        canvas=Image.new('RGB',(720,520));draw=ImageDraw.Draw(canvas)
        for col,(folder,label) in enumerate([(previous,'Previous city'),(render,'Central palace, aligned grid')]):
            for row,h in enumerate((12,0)):
                canvas.paste(Image.open(folder/f'h{h:02}-z1-pan00.png').crop((x-170,y-160,x+190,y+80)),(col*360,row*260+20))
                draw.text((col*360+4,row*260+4),label+(' | day' if h else ' | night'),fill='white')
        canvas.save(BASE/(name+'-native.png'))
    failure=next((FIX/'city-scene-r115').glob('*/capital-growth-search.json'))
    failures=read(failure);assert len(failures['attempts'])==25 and all(r['status']=='no_grid_solution' for r in failures['attempts'])
    result={'classification':'Provisional central orthogonal capital; no general city or milestone approval','cases':cases,
        'coastal_remaining':{'evidence':str(failure.relative_to(ROOT)),'sha256':sha(failure),'centers_tested':25,
            'finding':'Even alternate side assignments lack legal placements for the required third house. Preserve prior coastal scene; no clearance relaxation or native fallback change.'},
        'remaining':['Resolve coastal surrounding layout through footprint/foundation design rather than light or texture tuning',
                     'Broader single-era culture/era/size coverage and source material/environment reconstruction'],
        'gates_advanced':False,'approval':None}
    target=V2/'audits/beauty/CITY_CENTRAL_CAPITAL_EVIDENCE.json';target.write_text(json.dumps(result,indent=2)+'\n');print(target.relative_to(ROOT))


if __name__=='__main__':main()
