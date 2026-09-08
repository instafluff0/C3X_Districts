"""Recover authored normals for every imported runtime component, offline.

Replay fingerprinted importer evidence through the same proven primitive decoder
and local skin-palette remap. Geometry, UVs, indices, and active skin influences
must match the existing normalized mesh before publishing any replacement.
"""
from pathlib import Path
import hashlib,json,sys,re,collections
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from Renderer.tools.asset_compiler.indexed_static_package import IndexedStaticPackage
from Renderer.tools.asset_compiler.compound_landmark_importer import _decode_states,_decode_primitive,_normalize_geometry,TYPE_PRIM_GROUP,TYPE_MATERIAL
from Renderer.tools.asset_compiler.unit_model_extractor import _remap_skin_palette,_normalized_mesh,SOURCE_UNITS_PER_TILE
from Renderer.tools.asset_compiler.unit_family_asset_importer import _initial_entry
PACKS=ROOT/'Renderer/packs';OUT=PACKS/'UnitNormalFidelity'
REPORTS={'UnitEarlyLab':'early','UnitFamilyLab':'family','UnitRosterLab':'roster','UnitRosterExpansionLab':'roster_expansion','CompoundUnitRosterLab':'compound_roster'}
def read(p):return json.loads(p.read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    runtime=read(PACKS/'UnitAnimationRuntime/manifest.json');jobs=[];pins={};records={};anchors={}
    for name,report_name in REPORTS.items():
        path=ROOT/f'Renderer/preview/out/units/{report_name}_build.json';report=read(path);pins[str(path.relative_to(ROOT))]=sha(path.read_bytes())
        if report.get('strategy'):
            strategy=read(Path(report['strategy']['path']))
            for key,values in strategy.get('package_anchors',{}).items():
                if key in report['packages']:anchors.setdefault(report['packages'][key]['path'],[]).extend(values)
        manifest=read(PACKS/name/'manifest.json')
        for uid,unit in runtime['units'].items():
            if unit['source_pack']!=name:continue
            recipe=read(PACKS/name/unit['source_recipe']);slug=uid.split('/',1)[1]
            if 'nodes' in recipe:
                groups=[(node['components'],[c for c in report['components'] if c['composition']==slug and c['node']==node_id]) for node_id,node in recipe['nodes'].items()]
            else:
                source_unit=next(x['source_artdef'] for x in report['units'] if x['slug']==slug)
                groups=[(recipe['components'],[c for c in report['components'] if c['unit']==source_unit])]
            for components,evidence in groups:
                by_key={};counts=collections.Counter()
                for ev in evidence:
                    role=re.sub(r'[^a-z0-9]+','_',ev['role'].lower()).strip('_');counts[role]+=1
                    by_key[role if counts[role]==1 else role+'_'+str(counts[role])]=ev
                for comp in components:
                    ev=by_key[comp['asset'].rsplit('/',1)[1]]
                    doc=read(PACKS/name/manifest['assets'][comp['asset']]['component'])
                    if doc['role']!=ev['role']:raise ValueError('component identity mismatch '+comp['asset']+' '+str((doc['role'],ev['role'],doc['attachment_point'],ev['attachment_point'])))
                    mesh_paths=doc.get('meshes',[doc.get('mesh')])
                    if len(mesh_paths)!=len(ev['geometry']['parts']):raise ValueError('primitive inventory changed')
                    # Reports pin the effective physical package. An overlay can
                    # inherit its model from Base; actual vertex hashes disambiguate.
                    packages=[p for key,p in report['packages'].items() if key==ev['source_package'] or key.endswith('/'+ev['source_package'])]
                    if not packages:raise ValueError('source package evidence missing')
                    for relative,part in zip(mesh_paths,ev['geometry']['parts']):
                        jobs.append((name,relative,doc,ev,part,packages))
    # Group by package so its indexed allocation table is resident only once.
    packages={p['path']:p for job in jobs for p in job[-1]}
    unresolved=list(jobs);done=set();mesh_count=0;skin_count=0
    for physical,pin in packages.items():
        candidates=[j for j in unresolved if any(p['path']==physical for p in j[-1])]
        if not candidates:continue
        path=Path(physical)
        if sha(path.read_bytes())!=pin['sha256']:raise ValueError('source package changed')
        package=IndexedStaticPackage(path,_initial_entry(path,[j[3]['source_entry'] for j in candidates]+anchors.get(physical,[])))
        print('Reading source package',path.name,'components',len(candidates),flush=True)
        for job in candidates:
            name,relative,doc,ev,part,_=job;key=name+'/'+relative
            if key in done:continue
            vb=part['source_vertex_buffer'];ib=part['source_index_buffer']
            try:
                vertices=package.big_data(vb['offset'],vb['bytes']);indices=package.big_data(ib['offset'],ib['bytes'])
            except ValueError:continue
            if sha(vertices)!=part['vertex_sha256'] or sha(indices)!=part['index_sha256']:continue
            base=ev['pointer_chain']['base_model'];user=ev['pointer_chain']['user_data']
            states,_state=_decode_states(package,user)
            prim=package.pointer_fields(base,TYPE_PRIM_GROUP);mat=package.pointer_fields(base,TYPE_MATERIAL)
            if len(prim)!=1 or len(mat)!=1:raise ValueError('source primitive table changed')
            primitive=_decode_primitive(package,package.array_element(prim[0][1],part['primitive_index']),states,package.allocations[mat[0][1]-1]['element_count'])
            mesh,proof=_normalize_geometry(vertices,indices,vb,ib,primitive,SOURCE_UNITS_PER_TILE,ev['bones'] if part['skinned'] else None,normalize_skin_weights=True,use_authored_normals=True)
            if part['skinned']:_remap_skin_palette(mesh,part['skin_palette'],ev['bones']);skin_count+=1
            mesh=_normalized_mesh(mesh,doc['asset_id']);old=read(PACKS/name/relative)
            if mesh['topology']!=old['topology'] or len(mesh['vertices'])!=len(old['vertices']):raise ValueError('normalized topology changed '+key)
            for a,b in zip(mesh['vertices'],old['vertices']):
                for field in ['position','uv0']+(['joints','weights'] if part['skinned'] else []):
                    if a[field]!=b[field]:raise ValueError('normalized '+field+' changed '+key)
            result={'normal_source':proof['normal_source'],'normals':[v['normal'] for v in mesh['vertices']],
                    'address_mode':proof['uv_address'],'normalized_mesh_sha256':sha((PACKS/name/relative).read_bytes()),
                    'source_vertex_sha256':part['vertex_sha256'],'skin_palette':part['skin_palette']}
            destination=OUT/name/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(json.dumps(result,separators=(',',':'))+'\n')
            records[key]=str(destination.relative_to(ROOT));done.add(key);mesh_count+=1
        unresolved=[j for j in unresolved if j[0]+'/'+j[1] not in done]
        del package
    if unresolved:raise ValueError('unresolved source primitives: '+str([(j[0],j[1]) for j in unresolved]))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'manifest.json').write_text(json.dumps({'schema':'c3x.authored_normal_refresh.v1','meshes':records,'source_reports':pins},indent=2)+'\n')
    print('PASS',mesh_count,'meshes;',skin_count,'local skin palettes preserved')
if __name__=='__main__':main()
