"""Freeze once / verify the implementation-preparation package; never approves it."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
V2=ROOT/'Renderer/terrain_lab/v2'
REGIONS=['coastal','inland','wilderness','longcoast','freshcoast','freshrelief','combinedvolcano','freshshadow']

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def local(path):
    p=Path(path)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Nonportable package path: '+str(path))
    resolved=(ROOT/p).resolve();resolved.relative_to(ROOT)
    return resolved

def rel(path):return path.relative_to(ROOT).as_posix()
def load(path):return json.loads(path.read_text())

def freeze():
    manifest=HERE/'manifest.json'
    if manifest.exists():raise ValueError('Frozen package exists. Create a new revision; do not silently repin.')
    sources=set()
    for directory in ['app','contracts','shared','systems','shaders','qa','tests']:
        for base,dirs,files in os.walk(V2/directory):
            dirs[:]=[d for d in dirs if not d.startswith('.') and d not in ('out','__pycache__')]
            for name in files:
                p=Path(base)/name
                if p.suffix in ('.py','.cpp','.h','.hlsl'):sources.add(p)
    sources.update([HERE/'README.md',HERE/'IMPLEMENTATION.md',HERE/'package.py',HERE/'.gitignore'])
    for name in ['SHADOW_RECEIVER_PASS.md','SHADOW_RECEIVER_r1_EVIDENCE.json','SHADOW_RECEIVER_DIAGNOSTICS.json','RELIEF_SIZE_PASS.md','COAST_SOURCE_JOIN_PASS.md']:
        sources.add(V2/'audits/beauty'/name)
    references=[];assets={};packs={}
    for region in REGIONS:
        report=V2/'audits/beauty/out/shadow-receiver-r1'/region/'report.json';r=load(report)
        f=r['effective']['fixture'];m=r['effective']['module']
        fixture=V2/'fixtures/beauty/shadow-receiver-r1'/region/'fixture.json'
        sources.add(fixture)
        for path in f['modules']+[f['terrain']]+list(f['scenarios'].values()):sources.add(local(path))
        sources.add(local(m['shader']))
        for hook in ('terrain_hooks','hydrology_hooks','placement_hooks'):
            if hook in m:sources.add(local(m[hook]['header']))
        for key in ('hill_source','coastal_rocks'):
            a=m[key];assets[a['path']]={'path':a['path'],'sha256':a['sha256'],'kind':key}
        provenance=local(m['coastal_rocks']['path']).parent/'provenance.json'
        sources.add(provenance)
        qualified_sources=list(load(provenance)['sources'])+[m['hill_source']['path']]
        for name,path in f['packs'].items():packs[name]=path
        frames=[]
        for frame in r['outputs']:
            image=local(frame['image'])
            if sha(image)!=frame['sha256']:raise ValueError('Reference image drift: '+frame['image'])
            meta_path=local(frame['source_metadata']['path'])
            if sha(meta_path)!=frame['source_metadata']['sha256']:raise ValueError('Metadata drift')
            meta=load(meta_path)
            for t in meta['textures']:
                resolved=(ROOT/t['path']).resolve()
                if not resolved.is_file():
                    matches=[local(p) for p in qualified_sources if Path(p).name==Path(t['path']).name and sha(local(p))==t['sha256']]
                    if len(matches)!=1:raise ValueError('Unresolved or ambiguous source texture: '+t['path'])
                    resolved=matches[0]
                asset_path=rel(resolved)
                entry=assets.setdefault(asset_path,{'path':asset_path,'sha256':t['sha256'],'kind':'loaded_texture','view_formats':[]})
                if entry['sha256']!=t['sha256']:raise ValueError('Conflicting source texture')
                entry.setdefault('view_formats',[])
                if t['view_format'] not in entry['view_formats']:entry['view_formats'].append(t['view_format'])
            frames.append({k:frame[k] for k in ('hour','zoom','offset','image','sha256')})
        references.append({'region':region,'fixture':rel(fixture),'synthetic':region=='combinedvolcano',
            'tile_count':f['tile_count'],'viewport':f['viewport'],'projection':m['projection'],
            'terrain':f['terrain'],'terrain_sha256':sha(local(f['terrain'])),
            'source_region':f.get('real_map',{}).get('region'),
            'report':rel(report),'report_sha256':sha(report),'render_identity':r['render_identity'],
            'module':m,'frames':frames})
    # Include every referenced C/C++/shader header under Renderer and fixture
    # text under the selected recipes, but never follow data into licensed packs.
    import re
    pending=list(sources)
    while pending:
        path=pending.pop()
        if path.suffix not in ('.cpp','.h','.hlsl'):continue
        for name in re.findall(r'^\s*#include\s+"([^"]+)"',path.read_text(),re.M):
            child=(path.parent/name).resolve()
            if child.is_file() and child.is_relative_to(ROOT/'Renderer') and not child.is_relative_to(ROOT/'Renderer/native') and child not in sources:
                sources.add(child);pending.append(child)
    for path in packs.values():
        p=local(path)/'manifest.json'
        if p.is_file():assets[rel(p)]={'path':rel(p),'sha256':sha(p),'kind':'pack_manifest'}
    pinned=[{'path':rel(p),'sha256':sha(p)} for p in sorted(sources)]
    archive=HERE/'source_snapshot.tar.gz'
    with archive.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w|') as tar:
        for row in pinned:
            data=local(row['path']).read_bytes();entry=tarfile.TarInfo(row['path']);entry.size=len(data);entry.mode=0o644;entry.mtime=0
            tar.addfile(entry,io.BytesIO(data))
    native_files=['c3x_renderer.cpp','integrated_terrain.hlsl','terrain_rendering.hlsl','terrain_scene_runtime.h','c3x_renderer_api.h']
    d={'schema':'c3x.renderer_integration_preparation.v1','id':HERE.name,'prepared_on':'2026-09-06',
        'status':'candidate_preparation_not_promotion','approval':None,'visual_acceptance':False,
        'retained_candidate':'shadow-receiver-r1','historical_handoffs_unchanged':True,
        'required_gates':['existing LQ0/LQ1/LQ2 requirements','D3D11 parity','explicit visual approval','deliberate Integration refresh','native integration and live strategic checkpoint'],
        'required_user_action':[],'new_patch_symbols':[],
        'source_files':pinned,'source_archive':{'path':rel(archive),'sha256':sha(archive),'contains':'implementation and fixture text only; no pack art or native source'},
        'references':references,'pack_roots':packs,'local_assets':sorted(assets.values(),key=lambda x:x['path']),
        'asset_inventory_note':'Loaded dependencies include dormant object bindings; this does not establish object coverage. Licensed payloads are not bundled.',
        'native_advisory_snapshot':[{'path':'Renderer/native/'+name,'sha256':sha(ROOT/'Renderer/native'/name)} for name in native_files],
        'native_snapshot_policy':'Advisory only; integration files may evolve. Never restore them from this package.',
        'baseline_handoffs':[{'path':rel(p),'sha256':sha(p)} for p in sorted((ROOT/'Renderer/handoffs').glob('L*.json'))]}
    manifest.write_text(json.dumps(d,indent=2)+'\n')
    print(f'Frozen {len(pinned)} source files, {len(assets)} local dependencies, 32 reference frames')

def verify(evidence=False,assets=False):
    d=load(HERE/'manifest.json');errors=[]
    if d['approval'] is not None or d['visual_acceptance'] or d['status']!='candidate_preparation_not_promotion':
        errors.append('Preparation/approval state changed')
    entries=d['source_files']+d['baseline_handoffs']+[d['source_archive']]
    if assets:entries+=d['local_assets']
    if evidence:
        for r in d['references']:
            entries.append({'path':r['report'],'sha256':r['report_sha256']})
            entries += [{'path':f['image'],'sha256':f['sha256']} for f in r['frames']]
    for row in entries:
        p=local(row['path'])
        if not p.is_file():errors.append('Missing: '+row['path'])
        elif sha(p)!=row['sha256']:errors.append('Hash drift: '+row['path'])
    archive=local(d['source_archive']['path'])
    if archive.is_file():
        expected={r['path']:r['sha256'] for r in d['source_files']};seen=set()
        with tarfile.open(archive,'r:gz') as tar:
            for entry in tar:
                if not entry.isfile() or entry.name not in expected or entry.name in seen:raise ValueError('Unexpected archive entry')
                seen.add(entry.name)
                if hashlib.sha256(tar.extractfile(entry).read()).hexdigest()!=expected[entry.name]:errors.append('Archive content drift: '+entry.name)
        if seen!=set(expected):errors.append('Archive source coverage mismatch')
    if errors:raise ValueError('\n'.join(errors))
    print('PASS preparation package; source snapshot and immutable v1 handoffs verified; approval remains pending')
    if evidence:print('PASS 32 retained frame hashes and eight report hashes')
    if assets:print('PASS locally installed normalized asset hashes')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['freeze','verify'])
    p.add_argument('--evidence',action='store_true');p.add_argument('--assets',action='store_true');a=p.parse_args()
    if a.command=='freeze':freeze()
    else:verify(a.evidence,a.assets)

if __name__=='__main__':main()
