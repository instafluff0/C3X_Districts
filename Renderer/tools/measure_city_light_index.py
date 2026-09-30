"""Bounded paired FRESH lighting comparison; no installation or reference edits.

Requires isolated baseline/candidate DLLs and shader bundles in --out. Uses the
existing client camera/time trace; it does not qualify production scrolling.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Renderer.lab.platform import windows_root,native_command_result


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def fingerprint(out,label):
    # A superset of runtime pack payloads closes implicit provider dependencies,
    # in addition to explicitly selected definitions. Excludes disposable caches.
    suffixes={'.bin','.dds','.json','.txt','.hlsl'}
    paths=[p for p in (ROOT/'Renderer/packs').rglob('*') if p.is_file() and p.suffix.lower() in suffixes]
    paths+=[ROOT/'Renderer/default.custom_rendering.txt',ROOT/'Renderer/custom.custom_rendering.txt',
            ROOT/'Renderer/native/build/performance-review-current/developed-scene.csv']
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    record={'files':hashes,'bytes':sum(p.stat().st_size for p in set(paths)),
            'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
            'scope':'complete pack payload superset, definitions and scene; excludes compiled caches'}
    (out/('inputs-'+label+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    print('INPUTS',label,len(hashes),record['bytes'],record['sha256'],flush=True)
    if label=='after':
        before=json.loads((out/'inputs-before.json').read_text())
        if before['files']!=hashes:raise RuntimeError('Inputs changed during comparison')


def summary(log):
    text=log.read_text(errors='replace')
    raw=[tuple(map(float,m)) for m in re.findall(r'CLIENT_FRAME_TIMING frame=\d+ total_ms=([\d.]+) draw_ms=([\d.]+) present_ms=([\d.]+)',text)]
    stats={}
    for column,name in enumerate(['total','draw','present']):
        values=sorted(row[column] for row in raw)
        if values:
            def at(f):return values[min(len(values)-1,int(f*(len(values)-1)))]
            stats[name]={'samples':len(values),'mean_ms':sum(values)/len(values),'trace_ms':sum(values),'p50_ms':at(.5),'p95_ms':at(.95),'p99_ms':at(.99),'worst_ms':at(1),
                         'missed_16_67ms':sum(v>1000/60 for v in values)}
    stats['transitions']=re.findall(r'CLIENT_TRANSITION[^\n]+',text)
    stats['preparation']=re.findall(r'(?:CLIENT_PREPARE|CLIENT_PREWARM|CLIENT_PRIME|TIMING|PREPARE_TIMING)[^\n]+',text)
    stats['lights']=re.findall(r'CITY_LIGHT_(?:INDEX|REUSE)[^\n]+',text)
    stats['phases']=re.findall(r'CLIENT_CYCLE[^\n]+',text)
    return stats


def run(out,label,arm,*,hour=0,dense=True,capture=False,zoom=None,scene=None,frames=90,city_case='0,3,2,0'):
    directory=out/label
    if directory.exists():raise ValueError('Preserve existing receipts; choose a new run label: '+label)
    directory.mkdir()
    winroot=windows_root()
    def win(path):return str(winroot/path.relative_to(ROOT).as_posix())
    options={
        'C3X_RENDERER_VISUAL_PROFILE':'city-fidelity','C3X_RENDERER_TRACE':'0',
        'C3X_RENDERER_SHARED_SCENE_SURFACE':'1','C3X_RENDERER_WATER_MOTION':'1','C3X_RENDERER_WAVES':'1',
        'C3X_SANDBOX_SHADOW_PATCHES':'1','C3X_SANDBOX_WHOLE_WORLD':'1','C3X_SANDBOX_UNITS':'1',
        'C3X_RENDERER_PREVIEW_UNITS':'1','C3X_RENDERER_CITY_PACK':'Renderer/packs/CityCompositionRuntime',
        'C3X_RENDERER_SHADER_SOURCE_ROOT':win(out/arm/'shaders'),
        'C3X_SANDBOX_MOVE_START_TILE':'24,56','C3X_RENDERER_SCENE_SAMPLES':'1','C3X_RENDERER_SCENE_FILMIC':'0.5',
        'C3X_SANDBOX_REPLAY_CLIP':'1','C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':win(ROOT/'Renderer/custom.custom_rendering.txt'),
        'C3X_SANDBOX_CAPTURE':win(directory/'frame.bmp') if capture else '',
        'C3X_SANDBOX_CAPTURE_SEQUENCE':'','C3X_SANDBOX_PROFILE_COMPLETION':'','C3X_SANDBOX_PRESENT':'',
        'C3X_SANDBOX_PRESENT_MODE':'',
        'C3X_RENDERER_PREVIEW_DENSE_SCENE':'1' if dense else '',
        'C3X_RENDERER_PREVIEW_DENSE_CITY_CASE':city_case if dense else '',
        'C3X_SANDBOX_STUDY_START_MS':'29500','C3X_SANDBOX_CLIP_FRAMES':str(1 if capture else frames),
        'C3X_SANDBOX_STUDY_ZOOM':'' if zoom is None else str(zoom),'C3X_SANDBOX_STUDY_ZOOM_PEAK':'1.25',
        'C3X_SANDBOX_DAY_NIGHT':'','C3X_SANDBOX_FRAME_TIMINGS':'0' if capture else '1',
        'C3X_RENDERER_CITY_LIGHT_FULL_SCAN':'','C3X_RENDERER_CITY_LIGHT_DIAGNOSTICS':'1' if arm=='candidate' else '',
        'C3X_RENDERER_CITY_BORDER_FIXTURE':''}
    scene=scene or ROOT/'Renderer/native/build/performance-review-current/developed-scene.csv'
    paths=[out/'candidate/client_x64.exe',out/arm/'C3XRenderer_x64.dll',ROOT,
           ROOT/'Renderer/default.custom_rendering.txt',scene,directory/'initial.bmp']
    command=' '.join('"'+win(p)+'"' for p in paths)+f' 2240 1260 24 56 128 {hour}'
    batch=directory/'run.bat'
    batch.write_text('@echo off\nsetlocal\n'+'\n'.join(f'set "{k}={v}"' for k,v in options.items())+'\n'+
                     command+' >"'+win(directory/'run.log')+'" 2>&1\nexit /b %errorlevel%\n')
    print('BEGIN',label,flush=True)
    wake=out/'keep-awake.ps1'
    dispatch=(f'powershell -NoProfile -ExecutionPolicy Bypass -File "{win(wake)}" -BatchPath "{win(batch)}"' if wake.is_file() else f'call "{win(batch)}"')
    result=native_command_result('Renderer/native',dispatch,timeout_seconds=600)
    result.update(arm=arm,options=options,scene_sha256=sha(scene),dll_sha256=sha(paths[1]),client_sha256=sha(paths[0]),
                  shader_sha256={p.relative_to(out/arm/'shaders').as_posix():sha(p) for p in (out/arm/'shaders').rglob('*.hlsl')})
    if (directory/'run.log').exists():result['measurements']=summary(directory/'run.log')
    (directory/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass' or (capture and not (directory/'frame.bmp').is_file()):
        raise RuntimeError('Failed run; inspect existing process before retry: '+label)
    print('END',label,json.dumps(result.get('measurements',{}).get('total',{})),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'Renderer/.cache/city-light-index-step')
    parser.add_argument('phase',choices=['before','after','captures','timings','rich-timings'])
    args=parser.parse_args();out=args.out.resolve()
    if args.phase in ['before','after']:fingerprint(out,args.phase);return
    if args.phase=='captures':
        for density,dense in [('developed',True),('no-city',False),('rich',True)]:
            for hour in ([12,18,0] if density=='developed' else [0]):
                for zoom in [1,1.25]:
                    for arm in ['baseline','candidate']:
                        run(out,f'capture-{density}-h{hour}-z{zoom}-{arm}',arm,hour=hour,dense=dense,capture=True,zoom=zoom,
                            city_case='4,3,2,1' if density=='rich' else '0,3,2,0')
    elif args.phase=='rich-timings':
        for arm in ['baseline','candidate']:
            run(out,f'timing-rich-1-{arm}',arm,city_case='4,3,2,1')
    else:
        for workload,hour,dense,repeats in [('night',0,True,2),('noon',12,True,2),('no-city',0,False,2)]:
            for pair in range(repeats):
                for arm in (['baseline','candidate'] if pair%2==0 else ['candidate','baseline']):
                    run(out,f'timing-{workload}-{pair+1}-{arm}',arm,hour=hour,dense=dense)


if __name__=='__main__':main()
