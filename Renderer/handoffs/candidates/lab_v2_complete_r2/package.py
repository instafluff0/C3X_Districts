"""One entry point for the consolidated pickup. Pins in place; copies no art/cache."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from catalog import AUDITS, BEAUTY, CASES, ENTRIES, EXCLUDED, SYSTEMS, V2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def local(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('Nonportable path: '+str(name))
    q = (ROOT/p).resolve()
    q.relative_to(ROOT/'Renderer')
    return q

def relative(p):
    return p.resolve().relative_to(ROOT).as_posix()

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(p.read_text())
def write(p, d): p.write_text(json.dumps(d, indent=2)+'\n')
def pin(p):
    if not p.is_file(): raise ValueError('Missing required input: '+relative(p))
    return {'path':relative(p), 'sha256':sha(p), 'bytes':p.stat().st_size}

def shadow_probe():
    compiler = shutil.which('c++')
    if not compiler: raise ValueError('C++17 compiler is required')
    with tempfile.TemporaryDirectory(prefix='renderer-shadow-contract-') as tmp:
        exe = Path(tmp)/'probe'
        subprocess.run([compiler,'-std=c++17','-O2','-I',str(ROOT),str(HERE/'shadow_probe.cpp'),
                        str(ROOT/'Renderer/native/environment_runtime.cpp'),'-o',str(exe)],check=True)
        result = json.loads(subprocess.check_output([str(exe)],text=True))
    # This reports an observed production gap rather than concealing it in a pass count.
    return result

def freeze():
    target = HERE/'manifest.json'
    if target.exists(): raise ValueError('Immutable manifest exists. Create a new revision; do not repin silently.')
    baseline = 'a0683e0d2a5cb0694961d1993b4c56a8aa1d1323'
    tracked = subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'Renderer'],cwd=ROOT,text=True).splitlines()
    source = set()
    for name in tracked:
        # Preserve the complete common implementation/tool library, including tests.
        # These are dependency-library pins, NOT a license to enable every diagnostic.
        common = any(name.startswith(V2+d+'/') for d in ['app','contracts','shared','systems','shaders','qa','tests'])
        common |= name.startswith('Renderer/tools/asset_compiler/')
        legacy = name.startswith('Renderer/terrain_lab/') and '/v2/' not in name
        if (common or legacy) and Path(name).suffix in {'.py','.h','.cpp','.hlsl','.json','.csv','.bat'}:
            source.add(local(name))
    source.update(p for p in HERE.iterdir() if p.suffix in {'.py','.h','.cpp','.md'})
    source.update(local(BEAUTY+n) for n in AUDITS)
    for paths in ENTRIES.values(): source.update(local(V2+n) for n in paths)
    for n in ['renderer_workstreams.md','environment_lighting_and_ambient_effects.md',
              'city_palace_asset_import.md','barbarian_camp_import.md','goody_huts_and_colonies.md',
              'i20_native_unit_animation_handoff.md','animation_integration_checkpoint.md']:
        source.add(local('Renderer/docs/'+n))
    legacy = []
    gates = {}
    assets = set()
    for p in sorted((ROOT/'Renderer/handoffs').glob('L*.json')):
        d=read(p); gates[d['lab_gate']]=relative(p); legacy.append(pin(p))
        c=d['source_contract']
        roots=c.get('pack_roots',[])+[c[k] for k in ['pack_root','ordinary_pack_root','compound_pack_root'] if k in c]
        for root in roots:
            manifest=local('Renderer/'+root+'/manifest.json')
            if manifest.is_file(): assets.add(manifest)
        for k in ['runtime_bundle','runtime_bundles']:
            names=c.get(k,[]); names=[names] if isinstance(names,str) else names
            assets.update(local('Renderer/'+n) for n in names)
        audit=d.get('reference',{}).get('audit')
        if audit: source.add(local('Renderer/'+audit))
    old=read(ROOT/'Renderer/handoffs/candidates/lab_v2_terrain_lighting_r1/manifest.json')
    assets.update(local(x['path']) for x in old['local_assets'])
    palace=local('Renderer/packs/CityPalacesNormalized/manifest.json')
    if palace.is_file(): assets.add(palace)
    cases=[]; evidence=set(); packets=set(); chain=set()
    def report_chain(p):
        if p in chain:return
        chain.add(p); evidence.add(p); d=read(p)
        if 'source_report' in d: report_chain(local(d['source_report']))
        def inputs(v):
            if isinstance(v,dict):
                for x in v.values(): inputs(x)
            elif isinstance(v,list):
                for x in v:inputs(x)
            elif isinstance(v,str) and v.startswith('Renderer/'):
                q=local(v)
                if q.is_file() and q.suffix in {'.json','.csv','.h','.cpp','.hlsl'}:source.add(q)
        inputs(d.get('effective',{}))
    for case_id, folder, note in CASES:
        p=local(BEAUTY+'out/'+folder+'/report.json'); d=read(p); report_chain(p)
        frames=[]
        for f in d['outputs']:
            q=local(f['image']); row=pin(q)
            if row['sha256'] != f['sha256']:raise ValueError('Reference image drift: '+f['image'])
            evidence.add(q);frames.append(row)
        recipe=[]
        jobs=read(p.parent/'batch.json')
        for job in jobs:
            portable=[]
            for value in job:
                value=str(value)
                if value.startswith(str(ROOT)+ '/'):
                    q=Path(value); value=relative(q)
                    if q.is_file():
                        if q.suffix=='.packet' or q==Path(job[0]):packets.add(q)
                        elif q.suffix not in {'.bmp','.png','.rgba16f','.rgba32f'}:source.add(q)
                    elif q.is_dir():
                        source.update(x for x in q.rglob('*') if x.suffix in {'.hlsl','.msl'})
                elif Path(value).is_absolute():raise ValueError('External batch dependency must be normalized')
                portable.append(value)
            recipe.append(portable)
        cases.append({'id':case_id,'selection':note,'report':relative(p),'frames':frames,'portable_batch':recipe})
    # Resolve the C/C++/HLSL include closure for selected generated wrappers too.
    pending=list(source)
    while pending:
        p=pending.pop()
        if p.suffix not in {'.cpp','.h','.hlsl'}:continue
        for name in re.findall(r'^\s*#include\s+"([^"]+)"',p.read_text(),re.M):
            q=(p.parent/name).resolve()
            if not q.is_file():q=(ROOT/name).resolve()
            if q.is_file() and q.is_relative_to(ROOT/'Renderer') and '/native/' not in str(q) and q not in source:
                source.add(q);pending.append(q)
    systems=[dict(id=i,legacy_handoffs=[gates[g] for g in gs],native_state=n,selected=s,
                  remaining=r,entry_points=[V2+x for x in ENTRIES.get(i,[])]) for i,gs,n,s,r in SYSTEMS]
    native_files=['c3x_renderer.cpp','unit_body_renderer.h','unit_shadow.h','environment_runtime.cpp',
                  'environment_runtime.h','profile_v2/source_shadow.h','profile_v2/integrated_v2.hlsl']
    pinned_source=[]
    for p in sorted(source):
        name=relative(p)
        if name.startswith('Renderer/tools/asset_compiler/') or name in {
            'Renderer/docs/animation_integration_checkpoint.md',
            'Renderer/docs/i20_native_unit_animation_handoff.md'}:
            blob=subprocess.check_output(['git','rev-parse',baseline+':'+name],cwd=ROOT,text=True).strip()
            data=subprocess.check_output(['git','cat-file','blob',blob],cwd=ROOT)
            pinned_source.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'git_blob':blob})
        else:pinned_source.append(pin(p))
    d={'schema':'c3x.complete_lab_pickup.v1','id':HERE.name,'status':'prepared_not_promoted',
       'baseline_commit':baseline,
       'approval':None,'visual_acceptance':False,'required_user_action':[],'new_patch_symbols':[],
       'gates':['LQ0/LQ1/LQ2 unchanged','formal outstanding Integration gates unchanged','M9/M10/M11 deferred',
                'combined Windows visual comparison','explicit meaningful user checkpoint'],
       'systems':systems,'excluded':EXCLUDED,'cases':cases,
       'source_files':pinned_source,'historical_handoffs':legacy,
       'local_assets':[pin(p) for p in sorted(assets)],'evidence_files':[pin(p) for p in sorted(evidence)],
       'packet_files':[pin(p) for p in sorted(packets)],
       'native_advisory':[pin(ROOT/'Renderer/native'/n) for n in native_files],
       'native_policy':'Read-only baseline; never restore native files from this catalog. Coordinate new native revision before port.',
       'storage_policy':'Pins in existing checkout, no art archive or packet copies. Optional local assets and evidence require existing licensed packs/cache.',
       'asset_scope':'Pins legacy runtime bundles, terrain loaded channels, and pack manifests; not every transitive art blob in every city/animation pack. Use the pack compiler and replay validation for payload completeness.',
       'shadow_probe':shadow_probe()}
    write(target,d)
    print('Frozen',len(systems),'systems,',len(cases),'cases,',len(source),'source/library files; no payload copies')

def verify(with_evidence=False, with_assets=False, with_packets=False):
    d=read(HERE/'manifest.json');errors=[]
    if d['approval'] is not None or d['visual_acceptance'] or d['status']!='prepared_not_promoted':
        errors.append('Preparation/approval contract changed')
    actual={relative(p) for p in (ROOT/'Renderer/handoffs').glob('L*.json')}
    if actual!={r['path'] for r in d['historical_handoffs']}:errors.append('Historical handoff inventory changed')
    rows=d['source_files']+d['historical_handoffs']
    if with_evidence:rows+=d['evidence_files']
    if with_assets:rows+=d['local_assets']
    if with_packets:rows+=d['packet_files']
    for row in rows:
        p=local(row['path'])
        if 'git_blob' in row:
            result=subprocess.run(['git','cat-file','blob',row['git_blob']],cwd=ROOT,capture_output=True)
            if result.returncode or hashlib.sha256(result.stdout).hexdigest()!=row['sha256']:
                errors.append('Missing/drifted committed source: '+row['path'])
            elif not p.is_file() or sha(p)!=row['sha256']:
                print('ADVISORY use pinned Git blob; active importer differs:',row['path'])
        elif not p.is_file():errors.append('Missing: '+row['path'])
        elif sha(p)!=row['sha256']:errors.append('Drift: '+row['path'])
    for row in d['native_advisory']:
        p=local(row['path'])
        if not p.is_file() or sha(p)!=row['sha256']:print('ADVISORY native baseline changed:',row['path'])
    if errors:raise ValueError('\n'.join(errors))
    print('PASS',len(rows),'pins; preparation only, visual and integration gates remain open')

def replay(case_id, output):
    d=read(HERE/'manifest.json');case=next(c for c in d['cases'] if c['id']==case_id)
    out=output.resolve();out.relative_to(ROOT/'Renderer/terrain_lab/v2/audits/beauty/out')
    if out.exists():raise ValueError('Replay output must be new; preserve previous best')
    if shutil.disk_usage(out.parent).free<8*1024**3:raise ValueError('8 GiB storage floor')
    expected={r['path']:r['sha256'] for r in d['packet_files']+d['source_files']}
    jobs=[]
    for job in case['portable_batch']:
        result=[]
        for i,value in enumerate(job):
            if value.startswith('Renderer/'):
                p=local(value)
                if i in (2,3):p=out/p.name
                elif p.is_file() and value in expected and sha(p)!=expected[value]:raise ValueError('Replay input drift: '+value)
                elif p.is_dir():
                    for name,digest in expected.items():
                        if name.startswith(value+'/') and sha(local(name))!=digest:raise ValueError('Shader drift: '+name)
                result.append(str(p))
            else:result.append(value)
        jobs.append(result)
    import sys
    sys.path.insert(0,str(ROOT/V2/'app'))
    import runner
    from cache import Cache
    _,metal=runner.executables(Cache(ROOT/V2/'app/.cache'))
    out.mkdir();batch=out/'batch.json';write(batch,jobs)
    subprocess.run([str(metal),'--batch',str(batch)],check=True,cwd=ROOT)
    actual=[pin(Path(j[2])) for j in jobs]
    write(out/'replay.json',{'case':case_id,'outputs':actual,'approval':None,
                           'reference_hashes_match':[a['sha256']==b['sha256'] for a,b in zip(actual,case['frames'])]})
    print('Replayed',case_id,'into',relative(out),'; inspect before claiming visual acceptance')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['freeze','verify','list','shadows','replay','extract-source'])
    p.add_argument('--evidence',action='store_true');p.add_argument('--assets',action='store_true')
    p.add_argument('--packets',action='store_true');p.add_argument('--case');p.add_argument('--output',type=Path)
    p.add_argument('--source')
    a=p.parse_args()
    if a.command=='freeze':freeze()
    elif a.command=='verify':verify(a.evidence,a.assets,a.packets)
    elif a.command=='shadows':print(json.dumps(shadow_probe(),indent=2))
    elif a.command=='extract-source':
        if not a.source or not a.output:p.error('extract-source needs --source and --output')
        row=next(r for r in read(HERE/'manifest.json')['source_files'] if r['path']==a.source)
        out=a.output.resolve();out.relative_to(ROOT/'Renderer')
        if out.exists():raise ValueError('Never overwrite active source')
        data=(subprocess.check_output(['git','cat-file','blob',row['git_blob']],cwd=ROOT)
              if 'git_blob' in row else local(row['path']).read_bytes())
        if hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('Source drift')
        out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
        print('Extracted pinned source:',relative(out))
    elif a.command=='replay':
        if not a.case or not a.output:p.error('replay needs --case and --output')
        replay(a.case,a.output)
    else:
        for row in read(HERE/'manifest.json')['systems']:print(row['id']+': '+row['native_state']+' | '+row['selected'])

if __name__=='__main__':main()
