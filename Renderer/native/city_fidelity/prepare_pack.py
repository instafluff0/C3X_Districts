"""Normalize the complete selected city composition into a generic runtime library.

Source adapters stay offline. Runtime sees mesh channels, immutable growth
placements, transformed facade lights and footprint metadata, never source names.
"""
from pathlib import Path
from collections import defaultdict
import argparse,hashlib,json,math,struct,sys
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'Renderer/packs/CityCompositionRuntime'
INPUT=ROOT/'Renderer/packs/CityFidelitySources/current'
sys.path.insert(0,str(ROOT))
from Renderer.lab.shared.cities import assets as city
from Renderer.lab.shared.cities.growth import solve,bounds,expanded,overlaps
from Renderer.lab.shared.cities.ground import footprint_alignment,convex_hull,grid,coverage
from Renderer.lab.shared.cities.fingerprint import geometry_digest
from Renderer.lab.shared.cities.facades import derive

def read(p):return city.read(p)
def sha(p):return hashlib.sha256(city.input_bytes(p)).hexdigest()
def build_pack(output=OUT, lab_layouts=None, lab_focus=None, lab_frames=None):
    output=Path(output).resolve()
    output.relative_to(ROOT/'Renderer')
    for source in (INPUT.parent,ROOT/city.PACK,ROOT/'Renderer/packs/CityStudyAuxiliaryUV',
                   ROOT/'Renderer/packs/CityPalacesNormalized',ROOT/'Renderer/packs/CityAdjunctsNormalized'):
        source=source.resolve()
        if output==source or output in source.parents or source in output.parents:
            raise ValueError('City output must not overlap preserved source inputs')
    # Cached parsed meshes must not hide source edits or omit dependencies when
    # multiple category builds run in the same Python process.
    city.component.cache_clear()
    with city.track_inputs(generated=(output/'frames.json',)) as consumed:
        meta=_build(output, lab_layouts, lab_focus, lab_frames)
    return meta,consumed

def _build(OUT, lab_layouts=None, lab_focus=None, lab_frames=None):
    OUT.mkdir(parents=True,exist_ok=True)
    source=read('Renderer/packs/CityFidelitySources/manifest.json');catalog=read('Renderer/packs/CityStudyAuxiliaryUV/city_catalog.json')
    styles=['american','european','mediterranean','middle_eastern','asian'];eras=['ancient','medieval','industrial','modern']
    materials=[];material_ids={};models=[];model_ids={};templates=[];pins={};gaps=[]
    frames={};extra={}
    current=read(INPUT/'recipes.json')
    pins[str((INPUT/'recipes.json').relative_to(ROOT))]=sha(INPUT/'recipes.json')
    for record in source['pools']:
        for key in ['normals','materials']:pins[record[key]]=sha(record[key])
        frames.update(read(record['normals'])['meshes']);extra.update(read(record['materials'])['materials'])
    # The selected palace is already fingerprinted by the pickup gates.
    for name in ['palace-normals.json','palace-materials.json',
                 'asian-ancient-palace-normals.json',
                 'asian-ancient-palace-materials.json/mapping.json']:
        p=str((INPUT/name).relative_to(ROOT));pins[p]=sha(p)
        if 'normals' in name:frames.update(read(p)['meshes'])
        else:extra.update(read(p)['materials'])
    if lab_frames is not None:
        frame_path=Path(lab_frames).resolve()
        frame_path.relative_to(ROOT/'Renderer/lab')
        if lab_layouts is None:raise ValueError('Lab frames require Lab layouts')
        frames.update(read(frame_path)['meshes'])
        pins[str(frame_path.relative_to(ROOT))]=sha(frame_path)
    ground_parts=read(INPUT/'ground-parts.json')['parts']
    pins[str((INPUT/'ground-parts.json').relative_to(ROOT))]=sha(INPUT/'ground-parts.json')
    ground_binding=current['ground_binding']
    ground_reference=current['ground_reference']
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
    def model(asset,pack=Path('Renderer/packs/CityStudyAuxiliaryUV'),source_z_factor=1.0):
        key=(str(pack),asset,source_z_factor)
        if key in model_ids:return model_ids[key]
        b=body(asset,pack);parts=[]
        def direction(values,normal=False):
            # A Lab layout may retain the authored vertical proportion while
            # the runtime keeps its established source-to-world projection.
            if source_z_factor==1.0:return list(values)
            v=list(values)
            v[2]*=1/source_z_factor if normal else source_z_factor
            length=math.sqrt(sum(n*n for n in v))
            return [n/length for n in v] if length else v
        source_parts=b['parts']+[(x['mesh'],x['material']) for x in ground_parts.get(asset,[])]
        for mesh,mat in source_parts:
            ground=mat['alpha_mode']=='blend';frame=frames.get(mesh.get('asset_id'))
            if not ground and frame is None:raise ValueError('missing source frame '+mesh['asset_id'])
            if frame and geometry_digest(mesh)!=frame['geometry_digest']:raise ValueError('source frame fingerprint mismatch')
            vertices=[]
            for i,v in enumerate(mesh['vertices']):
                position=[v['position'][j]-(b['lo'][j]+b['hi'][j])/2 for j in (0,1)]+[v['position'][2]*source_z_factor]
                normal=frame['normals'][i] if frame else v['normal']
                tangent=frame['tangents'][i] if frame else [1,0,0];bitangent=frame['bitangents'][i] if frame else [0,1,0]
                vertices.append(position+v['uv0']+direction(normal,True)+v.get('uv1',[0,0])+direction(tangent)+direction(bitangent)+v.get('uv2',[0,0]))
            parts.append({'material':material(mat,asset,ground),'vertices':vertices,'indices':mesh['topology']['indices']})
        mid=len(models);model_ids[key]=mid
        hull=convex_hull([(v['position'][0]-(b['lo'][0]+b['hi'][0])/2,v['position'][1]-(b['lo'][1]+b['hi'][1])/2) for mesh,mat in b['parts'] if mat['alpha_mode']!='blend' for v in mesh['vertices']])
        models.append({'asset':asset,'pack':str(pack),
                       'low':b['lo'][:2]+[b['lo'][2]*source_z_factor],
                       'high':b['hi'][:2]+[b['hi'][2]*source_z_factor],
                       'hull':hull,'parts':parts})
        return mid
    selected={}
    for item in current['selected']:
        selected[int(item['runtime_authority'].removeprefix('selected-r'))]=item['recipe']
    # Use the actual selected placements first. The coastal capital retains its
    # preceding legal composition instead of inventing a central placement.
    light_cache={}
    foundation_source=None
    def foundation():
        nonlocal foundation_source
        if foundation_source is None:
            asset='city/walls/medieval/segment_01'
            wall=body(asset,Path('Renderer/packs/CityAdjunctsNormalized'))
            # The masonry side of this normalized wall supplies a proven
            # textured patch for Lab retaining faces. No source format reaches
            # the runtime; its generic pack stores only a material and UV box.
            mesh,mat=wall['parts'][0]
            side=[v for v in mesh['vertices'] if v['normal'][0]>.9 and
                  -.014<=v['position'][2]<=.03 and
                  .65<=v['uv0'][0]<=.70 and .55<=v['uv0'][1]<=.72]
            if len(side)<4:raise ValueError('foundation masonry UV patch missing')
            uv=[min(v['uv0'][0] for v in side),min(v['uv0'][1] for v in side),
                max(v['uv0'][0] for v in side),max(v['uv0'][1] for v in side)]
            positions=sorted({round(v['position'][1],6) for v in side})
            module_width=min(b-a for a,b in zip(positions,positions[1:]) if b-a>.005)
            module_height=max(v['position'][2] for v in side)-min(v['position'][2] for v in side)
            # Use a uniform 2x masonry module on the retaining faces. This
            # keeps the source aspect ratio while avoiding a tiny tiled grid.
            step=[module_width*2.3*2,module_height*2.3*2/.648266978876*112]
            foundation_source={'material':material(mat,asset),'uv':uv,'step':step}
        return foundation_source
    def template(pool,size,instances,capital=False,authority=None,environment=False,clearance=None,
                 source_z_factor=1.0):
        culture,era=pool.removeprefix('city/pool/').split('/')
        if culture not in styles:raise ValueError('unknown normalized culture '+culture)
        out={'culture':styles.index(culture),'era':eras.index(era),'size':size,'capital':capital,'environment':environment,'authority':authority,
            'clearance':clearance or [.05,2.5,.12,12.4],'instances':[]}
        if authority and authority.startswith('lab-fixed-'):
            out['foundation']=foundation()
        for inst in instances:
            asset=inst['asset'];pack=Path(inst.get('pack','Renderer/packs/CityStudyAuxiliaryUV'));b=body(asset,pack)
            mid=model(asset,pack,source_z_factor);rot=inst['rotation'];scale=inst['scale'];offset=inst['offset'];box=bounds(b,rot,scale)
            # The original quadrature and facade-plane rule run once offline;
            # all resulting positions are transformed with the same instance.
            key=(str(pack),asset,scale,rot,inst['slot']=='capital',source_z_factor)
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
                if source_z_factor!=1.0:
                    for light in derived['lights']:
                        light['position'][2]*=source_z_factor
                light_cache[key]=derived
            out['instances'].append({'model':mid,'slot':inst['slot'],'scale':scale,'rotation':rot,'offset':offset,'bounds':box,'lights':light_cache[key]['lights']})
        # The selected connected paving is the same footprint union used by
        # Lab. Store topology/coverage offline; runtime only conforms/clips it
        # against captured terrain and supplies a world-stable UV origin.
        out['paving']=None
        if era=='modern':
            ordinary=[i for i in out['instances'] if i['slot']!='capital']
            scale=ordinary[0]['scale']
            if any(abs(i['scale']-scale)>1e-8 for i in ordinary):
                if not authority or not authority.startswith('lab-fixed-'):
                    raise ValueError('nonuniform city scale')
                scale=sum(i['scale'] for i in ordinary)/len(ordinary)
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
    if lab_layouts is not None:
        layout_path=Path(lab_layouts).resolve()
        layout_path.relative_to(ROOT/'Renderer/lab')
        layouts=read(layout_path)
        if layouts.get('schema')!='c3x.lab.city_design.v1':raise ValueError('unsupported Lab city design')
        for design in layouts['designs']:
            culture=design['culture_name'].lower().replace(' ','_')
            era=design['era_name'].lower()
            if lab_focus is not None and (culture,era)!=lab_focus:continue
            pool=f'city/pool/{culture}/{era}'
            if pool not in catalog['pools']:raise ValueError('Lab city design has no source pool')
            for size,count in enumerate(design['population_counts']):
                if not 0<count<=len(design['houses']):raise ValueError('invalid Lab population tier')
                tier=design['tier_designs'][size]
                if len(tier['houses'])!=count:raise ValueError('invalid Lab size design')
                houses=[{**item,'slot':slot} for slot,item in enumerate(tier['houses'])]
                civic={**tier['base_centerpiece'],'slot':count}
                for capital in (False,True):
                    instances=houses+([] if capital and design.get('capital_replaces_centerpiece') else [civic])
                    if capital:instances.append({**tier['palace'],'slot':'capital'})
                    vertical_metric=design.get('vertical_metric',.648266978876)
                    template(pool,size,instances,capital,
                             f'lab-fixed-{culture}-{era}',era=='modern',
                             [.04,18.0,0,4.0],.648266978876/vertical_metric)
    # The same bounded Lab growth solver and source-scale rule cover other
    # normalized pools. These are production adaptations, not new Lab witnesses.
    for pool,record in sorted(catalog['pools'].items()):
        culture,era=pool.removeprefix('city/pool/').split('/')
        assets=[body(a) for a in record['components']]
        reference=[city.component(a) for a in read(city.PACK/'city_catalog.json')['pools'][pool]['components']]
        scale=city.source_scale(reference,1.5)
        for a in assets:a['grid_rotation']=footprint_alignment([v['position'][:2] for mesh,m in a['parts'] if m['alpha_mode']!='blend' for v in mesh['vertices']])
        # Preserve the documented Civ III population hierarchy at gameplay
        # scale. Individual source houses need negative space just as much as
        # precomposed neighborhood blocks; doubling their count collapses roof
        # and facade silhouettes into one dark mass after reconstruction.
        blocks=set(current['blocks_by_pool'][pool])
        order=sorted([a for a in assets if a['id'] not in blocks],key=lambda a:(a['hi'][2]-a['lo'][2],a['id']))
        if not order:order=assets
        counts=[4,7,11]
        preserved=[]
        for size,count in enumerate(counts):
            layout={};buildable=lambda box:True
            if not blocks:
                # Keep a stable civic court in the middle of individual-house
                # pools. The first four growth slots occupy its four sides;
                # later stages extend that connected neighborhood instead of
                # minimizing into one unreadable row.
                court=[-.17,-.17,.17,.17]
                buildable=lambda box,court=court:not overlaps(box,court)
                layout={'fixed_neighbors':[court],
                    'surround_center':[0,0]}
            try:
                plan,stats=solve(order,count,scale,[.65,.8,.95][size],buildable,preserved=preserved,node_limit=20000,grid_step=.07,neighbor_gap=.1,connected_prefixes=tuple(counts[:size+1]),**layout)
            except ValueError as e:
                gaps.append({'pool':pool,'size':size,'reason':str(e)});break
            if plan is None:
                gaps.append({'pool':pool,'size':size,'reason':stats});break
            instances=[{'asset':i['asset']['id'],'slot':i['slot'],'scale':i['scale'],'rotation':i['rotation'],'offset':[i['x'],i['y']],'local_bounds':bounds(i['asset'],i['rotation'],i['scale'])} for i in plan]
            preserved=instances
            template(pool,size,instances,False,'generic-source-growth',era=='modern')
            if pool=='city/pool/asian/ancient':
                # AncientWood is the checked source fallback for Civ III's
                # Asian ancient group. Its matching source palace gives a
                # capital a readable center without inventing Japanese art.
                palace='city/palace/root/57520845b3095674'
                template(pool,size,instances+[{'asset':palace,'slot':'capital',
                    'pack':'Renderer/packs/CityPalacesNormalized','scale':8.5,
                    'rotation':0,'offset':[0,0]}],True,
                    'asian-ancientwood-capital',True)
        print('COMPOSED',pool,flush=True)
    meta={'schema':'c3x.city_composition.v1','materials':materials,'models':[{k:v for k,v in m.items() if k!='parts'} for m in models],'templates':templates,'gaps':gaps,'source_sha256':pins}
    (OUT/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    # Generic binary: all paths are relative to the mod root; no source package
    # parser or Python runtime is needed by the game.
    wire=bytearray(b'C3XCITY3' if lab_layouts is not None else b'C3XCITY2')
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
        if lab_layouts is not None:
            foundation_data=t.get('foundation')
            u(1 if foundation_data else 0)
            if foundation_data:
                u(foundation_data['material']);f(foundation_data['uv']+foundation_data['step'])
    (OUT/'city.bin').write_bytes(wire)
    print('PASS',len(models),'models',len(materials),'materials',len(templates),'growth templates; bytes',len(wire),'gaps',len(gaps))
    return meta

def main():
    return build_pack(OUT)[0]
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,help='Build a disposable candidate without changing the runtime pack')
    parser.add_argument('--lab-layouts',type=Path,help='Optional Lab-only city design recipes')
    parser.add_argument('--lab-focus',help='Compile one culture,era from --lab-layouts')
    parser.add_argument('--lab-frames',type=Path,help='Optional Lab-only source frame evidence')
    args=parser.parse_args()
    if args.output:
        OUT=(ROOT/args.output).resolve()
        if not OUT.is_relative_to(ROOT/'Renderer'):
            parser.error('output must stay within Renderer')
    if args.lab_layouts:
        if not args.output:parser.error('--lab-layouts requires an isolated --output')
        focus=tuple(args.lab_focus.split(',')) if args.lab_focus else None
        if focus is not None and (len(focus)!=2 or
                                  focus[0] not in ('american','european','mediterranean','middle_eastern','asian') or
                                  focus[1] not in ('ancient','medieval','industrial','modern')):
            parser.error('invalid --lab-focus')
        build_pack(OUT,args.lab_layouts,focus,args.lab_frames)
    else:
        if args.lab_focus:parser.error('--lab-focus requires --lab-layouts')
        if args.lab_frames:parser.error('--lab-frames requires --lab-layouts')
        main()
