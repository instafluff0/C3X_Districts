"""Normalize the complete selected city composition into a generic runtime library.

Source adapters stay offline. Runtime sees mesh channels, immutable growth
placements, transformed facade lights and footprint metadata, never source names.
"""
from pathlib import Path
from collections import defaultdict
import hashlib,json,math,struct,sys
ROOT=Path(__file__).resolve().parents[3];LAB=ROOT/'Renderer/terrain_lab/v2';OUT=ROOT/'Renderer/packs/CityCompositionRuntime'
sys.path[:0]=[str(LAB/'qa'),str(LAB/'systems/objects')]
import presentation as city
from city_growth_layout import solve,bounds,expanded,overlaps
from settlement_ground import footprint_alignment,convex_hull,grid,coverage
from mesh_fingerprint import geometry_digest
from city_facade_light_probe import derive

def read(p):return json.loads((ROOT/p).read_text())
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=read('Renderer/packs/CityFidelitySources/manifest.json');catalog=read('Renderer/packs/CityStudyAuxiliaryUV/city_catalog.json')
    styles=['american','european','mediterranean','middle_eastern','asian'];eras=['ancient','medieval','industrial','modern']
    materials=[];material_ids={};models=[];model_ids={};templates=[];pins={};gaps=[]
    frames={};extra={}
    for record in source['pools']:
        for key in ['normals','materials']:pins[record[key]]=sha(record[key])
        frames.update(read(record['normals'])['meshes']);extra.update(read(record['materials'])['materials'])
    # The selected palace is already fingerprinted by the pickup gates.
    for name in ['combined-normals.json','combined-extra.json']:
        p='Renderer/terrain_lab/v2/fixtures/beauty/city-capital-materials-r1/'+name;pins[p]=sha(p)
        if 'normals' in name:frames.update(read(p)['meshes'])
        else:extra.update(read(p)['materials'])
    ground_parts=read('Renderer/terrain_lab/v2/fixtures/beauty/city-ground-binding-r1/modern-ground-parts.json')['parts']
    ground_binding=read('Renderer/terrain_lab/v2/fixtures/beauty/city-ground-binding-r1/modern.json')
    ground_reference=read('Renderer/terrain_lab/v2/audits/beauty/out/city-central-capital-r2/inland/ground/settlement.json')
    body_cache={}
    def body(asset,pack=Path('Renderer/packs/CityStudyAuxiliaryUV')):
        key=str(pack)+'/'+asset
        if key not in body_cache:body_cache[key]=city.component(asset,pack)
        return body_cache[key]
    def material(mat,asset,ground=False):
        channels={**mat['channels']};overlay=extra.get(asset+':'+mat.get('name',''))
        if overlay:channels.update(overlay['channels'])
        paths=[channels.get(k,{}).get('texture','') for k in ['base_color','emissive','ambient_occlusion','normal_0','gloss','metalness','opacity']]
        if ground and asset in ground_parts and paths[0]==ground_binding['expected']['texture']:
            if sha(paths[0])!=ground_binding['expected']['sha256']:raise ValueError('ground binding source changed')
            paths[0]=ground_binding['replacement']['texture']
            if sha(paths[0])!=ground_binding['replacement']['sha256']:raise ValueError('ground binding replacement changed')
        mode=sum((1<<i) for i,a in enumerate(['u','v']) if channels['base_color'].get('address_'+a)=='clamp')
        if mode not in (0,3):raise ValueError('selected city material expects matching address axes')
        bits=(bool(paths[2])*1+bool(paths[3])*2+(mode==0)*4+bool(paths[4])*8+bool(paths[5])*16+bool(paths[6])*32)
        # Keep Asian/ancient dielectric evidence; bound metalness/environment
        # is enabled only by a template's selected environment flag.
        key=tuple(paths)+(mode,bits,int(ground))
        if key not in material_ids:
            material_ids[key]=len(materials);materials.append({'textures':paths,'address':mode,'bits':bits,'ground':ground})
            for p in paths:
                if p:pins[p]=sha(p)
        return material_ids[key]
    def model(asset,pack=Path('Renderer/packs/CityStudyAuxiliaryUV')):
        key=str(pack)+'/'+asset
        if key in model_ids:return model_ids[key]
        b=body(asset,pack);parts=[]
        source_parts=b['parts']+[(x['mesh'],x['material']) for x in ground_parts.get(asset,[])]
        for mesh,mat in source_parts:
            ground=mat['alpha_mode']=='blend';frame=frames.get(mesh.get('asset_id'))
            if not ground and frame is None:raise ValueError('missing source frame '+mesh['asset_id'])
            if frame and geometry_digest(mesh)!=frame['geometry_digest']:raise ValueError('source frame fingerprint mismatch')
            vertices=[]
            for i,v in enumerate(mesh['vertices']):
                position=[v['position'][j]-(b['lo'][j]+b['hi'][j])/2 for j in (0,1)]+[v['position'][2]]
                normal=frame['normals'][i] if frame else v['normal']
                tangent=frame['tangents'][i] if frame else [1,0,0];bitangent=frame['bitangents'][i] if frame else [0,1,0]
                vertices.append(position+v['uv0']+normal+v.get('uv1',[0,0])+tangent+bitangent+v.get('uv2',[0,0]))
            parts.append({'material':material(mat,asset,ground),'vertices':vertices,'indices':mesh['topology']['indices']})
        mid=len(models);model_ids[key]=mid
        hull=convex_hull([(v['position'][0]-(b['lo'][0]+b['hi'][0])/2,v['position'][1]-(b['lo'][1]+b['hi'][1])/2) for mesh,mat in b['parts'] if mat['alpha_mode']!='blend' for v in mesh['vertices']])
        models.append({'asset':asset,'pack':str(pack),'low':b['lo'],'high':b['hi'],'hull':hull,'parts':parts})
        return mid
    selected={}
    for r in [60,61,64,111,112,101]:
        p=next((LAB/f'fixtures/beauty/city-scene-r{r}').glob('*/augmentation.json'));a=read(p);pins[str(p.relative_to(ROOT))]=sha(p)
        selected[r]=a
    # Use the actual selected placements first. The coastal capital retains its
    # preceding legal composition instead of inventing a central placement.
    light_cache={}
    def template(pool,size,instances,capital=False,authority=None,environment=False,clearance=None):
        culture,era=pool.removeprefix('city/pool/').split('/')
        if culture not in styles:raise ValueError('unknown normalized culture '+culture)
        out={'culture':styles.index(culture),'era':eras.index(era),'size':size,'capital':capital,'environment':environment,'authority':authority,
            'clearance':clearance or [.05,2.5,.12,12.4],'instances':[]}
        for inst in instances:
            asset=inst['asset'];pack=Path(inst.get('pack','Renderer/packs/CityStudyAuxiliaryUV'));b=body(asset,pack)
            mid=model(asset,pack);rot=inst['rotation'];scale=inst['scale'];offset=inst['offset'];box=bounds(b,rot,scale)
            # The original quadrature and facade-plane rule run once offline;
            # all resulting positions are transformed with the same instance.
            key=(str(pack),asset,scale,rot,inst['slot']=='capital')
            if key not in light_cache:
                entry={'asset':asset,'slot':inst['slot'],'scale':scale,'rotation':rot,'offset':[0,0],'local_bounds':box,'sample_start':0}
                aug={'emissive_uv':2,'grounding':'source_z_zero','scene_world_z_per_source_unit':1/0.648266978876,'pack':str(pack),'source_normals':None,'emissive_gain':8,'capital':{'mapping':{'pack':str(pack)}},'instances':[entry]}
                # Supply exact recovered frames to the existing derivation.
                mapping=OUT/'frames.json'
                aug['source_normals']={'mapping':str(mapping.relative_to(ROOT))}
                surface={'samples':[{'column':0,'row':0,'u':0,'v':1,'height':0}]}
                try:derived=derive(aug,surface,128,source_facade_slots=('capital',))
                except ValueError as e:
                    if 'no emitting facade samples' not in str(e):raise
                    derived={'lights':[],'blockers':[]}
                light_cache[key]=derived
            out['instances'].append({'model':mid,'slot':inst['slot'],'scale':scale,'rotation':rot,'offset':offset,'bounds':box,'lights':light_cache[key]['lights']})
        # The selected connected paving is the same footprint union used by
        # Lab. Store topology/coverage offline; runtime only conforms/clips it
        # against captured terrain and supplies a world-stable UV origin.
        out['paving']=None
        if era=='modern':
            ordinary=[i for i in out['instances'] if i['slot']!='capital']
            scale=ordinary[0]['scale']
            if any(abs(i['scale']-scale)>1e-8 for i in ordinary):raise ValueError('nonuniform city scale')
            boxes=[];coverage_boxes=[];polygons=[]
            for i in out['instances']:
                x,y=i['offset'];b=i['bounds'];box=[x+b[0],-y-b[3],x+b[2],-y-b[1]];boxes.append(box)
                if i['slot']=='capital':
                    polygons.append(convex_hull([(x+p[0]*i['scale'],-y-p[1]*i['scale']) for h in models[i['model']]['hull'] for p in [city.rotate([*h,0],i['rotation'])]]))
                else:coverage_boxes.append(box)
            xy,triangles=grid(boxes,.1)
            alpha=[coverage(x,y,coverage_boxes,.1,.025,polygons) for x,y in xy]
            indices=[i for tri in triangles if max(alpha[i] for i in tri)>0 for i in tri]
            paving_mat=material({'channels':{'base_color':{'texture':ground_reference['atlas']['texture'],'address_u':'clamp','address_v':'clamp'}}},'',True)
            out['paving']={'material':paving_mat,'period':[v*scale/ground_reference['uniform_ordinary_city_scale'] for v in ground_reference['tile_period']],
                'atlas_uv':ground_reference['atlas_uv'],'margin':.1,'feather':.025,'grid_step':.025,
                'vertices':[[x,y,a] for (x,y),a in zip(xy,alpha)],'indices':indices,'coverage_boxes':coverage_boxes,'coverage_polygons':polygons}
        templates.append(out);return out
    (OUT/'frames.json').write_text(json.dumps({'meshes':frames},separators=(',',':'))+'\n')
    for revision,a in selected.items():
        inst=[{**i,**({'pack':a['capital']['mapping']['pack']} if i['slot']=='capital' else {})} for i in a['instances']]
        counts=a.get('stage_component_counts') or [4,7,11]
        for size in [0,1]:
            subset=[i for i in inst if i['slot']=='capital' or i['slot']<counts[size]]
            template(a['pool'],size,subset,a['capital']['drawn'],f'selected-r{revision}',revision in [111,112,101],
                [.05,2.5,a.get('vegetation_clearance') or 0,(a.get('river_exclusion') or {}).get('threshold_pixels',0)])
    # The same bounded Lab growth solver and source-scale rule cover other
    # normalized pools. These are production adaptations, not new Lab witnesses.
    for pool,record in sorted(catalog['pools'].items()):
        culture,era=pool.removeprefix('city/pool/').split('/')
        assets=[body(a) for a in record['components']]
        reference=[city.component(a) for a in read(city.PACK/'city_catalog.json')['pools'][pool]['components']]
        scale=city.layout(reference,0,factor=1.5)[0]['scale']
        for a in assets:a['grid_rotation']=footprint_alignment([v['position'][:2] for mesh,m in a['parts'] if m['alpha_mode']!='blend' for v in mesh['vertices']])
        # Source neighborhoods use the smaller body budget; individual houses
        # use the selected dense growth budget. This is offline pack policy.
        report=read('Renderer/terrain_lab/v2/audits/beauty/out/city-source-expanded-r1/build.json')
        entries=next(x['selected'] for x in report['pools'] if x['pool']==pool)
        blocks={x['asset_id'] for x in entries if '_Block_' in x['entry']}
        order=sorted([a for a in assets if a['id'] not in blocks],key=lambda a:(a['hi'][2]-a['lo'][2],a['id']))
        if not order:order=assets
        counts=[4,7,11] if blocks else [8,16,24]
        preserved=[]
        for size,count in enumerate(counts):
            try:
                plan,stats=solve(order,count,scale,[.65,.8,.95][size],lambda box:True,preserved=preserved,node_limit=20000,grid_step=.07,neighbor_gap=.08,connected_prefixes=tuple(counts[:size+1]))
            except ValueError as e:
                gaps.append({'pool':pool,'size':size,'reason':str(e)});break
            if plan is None:
                gaps.append({'pool':pool,'size':size,'reason':stats});break
            instances=[{'asset':i['asset']['id'],'slot':i['slot'],'scale':i['scale'],'rotation':i['rotation'],'offset':[i['x'],i['y']],'local_bounds':bounds(i['asset'],i['rotation'],i['scale'])} for i in plan]
            preserved=instances
            template(pool,size,instances,False,'generic-source-growth',era=='modern')
        print('COMPOSED',pool,flush=True)
    meta={'schema':'c3x.city_composition.v1','materials':materials,'models':[{k:v for k,v in m.items() if k!='parts'} for m in models],'templates':templates,'gaps':gaps,'source_sha256':pins}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    # Generic binary: all paths are relative to the mod root; no source package
    # parser or Python runtime is needed by the game.
    wire=bytearray(b'C3XCITY2')
    def u(n):wire.extend(struct.pack('<I',n))
    def f(values):wire.extend(struct.pack('<'+'f'*len(values),*values))
    def string(s):b=s.encode();u(len(b));wire.extend(b)
    u(len(materials));u(len(models));u(len(templates))
    for m in materials:
        u(m['address']);u(m['bits']);u(int(m['ground']))
        for p in m['textures']:string(p)
    for m in models:
        u(len(m['parts']));f(m['low']+m['high']);u(len(m['hull']))
        for p in m['hull']:f(p)
        for part in m['parts']:
            u(part['material']);u(len(part['vertices']));u(len(part['indices']))
            for v in part['vertices']:f(v)
            for i in part['indices']:u(i)
    for t in templates:
        for k in ['culture','era','size','capital','environment']:u(int(t[k]))
        string(t['authority']);f(t['clearance']);u(len(t['instances']))
        for i in t['instances']:
            u(i['model']);u(1 if i['slot']=='capital' else 0);f([i['scale'],i['rotation']]+i['offset']+i['bounds']);u(len(i['lights']))
            for l in i['lights']:f(l['position']+[l['range']]+l['color_linear']+[l['intensity']]+l['direction']+[0])
        p=t['paving'];u(1 if p else 0)
        if p:
            u(p['material']);f(p['period']+p['atlas_uv']);u(len(p['vertices']));u(len(p['indices']))
            for v in p['vertices']:f(v)
            for i in p['indices']:u(i)
    (OUT/'city.bin').write_bytes(wire)
    print('PASS',len(models),'models',len(materials),'materials',len(templates),'growth templates; bytes',len(wire),'gaps',len(gaps))
if __name__=='__main__':main()
