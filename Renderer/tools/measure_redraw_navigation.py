"""Isolated complete-path redraw and authoritative-camera witness receipts."""
from pathlib import Path
import argparse
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Renderer.lab.platform import windows_root,native_command_result
from Renderer.tools.measure_city_light_index import sha,summary

OUT=ROOT/'Renderer/.cache/redraw-navigation-step'
LIGHT=ROOT/'Renderer/.cache/city-light-index-step'
PASS_NAMES=('selection','shadow','main_scene','reflection_scene','water','main_material',
            'reflection_material','relight','units','reflected_units','unit_shadow','reconstruction','publication')


def distribution(values):
    ordered=sorted(values)
    if not ordered:return {}
    def at(f):return ordered[min(len(ordered)-1,int(f*(len(ordered)-1)))]
    return dict(samples=len(values),trace_ms=sum(values),mean_ms=sum(values)/len(values),
                p50_ms=at(.5),p95_ms=at(.95),p99_ms=at(.99),worst_ms=at(1),
                missed_16_67ms=sum(v>1000/60 for v in values))


def navigation_summary(text):
    frames={}
    for view,frame,total,draw,present in re.findall(
            r'CAMERA_FRAME view=(\w+) frame=(\d+) total_ms=([\d.]+) draw_ms=([\d.]+) present_ms=([\d.]+)',text):
        frames.setdefault(view,[]).append((int(frame),float(total),float(draw),float(present)))
    result={}
    for view,rows in frames.items():
        result[view]={scope:{name:distribution([row[column] for row in rows if row[0]>=first])
                     for column,name in enumerate(('total','draw','present'),1)}
                     for scope,first in (('complete_steady',0),('warmed_steady',3))}
    return result


def workload_summary(text):
    views={}
    for line in text.splitlines():
        if not line.startswith('CLIENT_PASS '):continue
        fields=dict(piece.split('=',1) for piece in line.split()[1:])
        frame=int(fields['frame'])
        if frame<3:continue
        view=views.setdefault(fields['view'],{'frames':set(),'layers':{}})
        view['frames'].add(frame)
        key=f"{PASS_NAMES[int(fields['pass'])]}/{fields['layer']}"
        total=view['layers'].setdefault(key,{})
        for name,value in fields.items():
            if name not in ('view','frame','pass','layer'):total[name]=total.get(name,0)+int(value)
    for view in views.values():
        count=len(view['frames']);view['frames']=count
        view['passes']={}
        for key,total in view['layers'].items():
            for name,value in total.items():total[name]=value/count
            combined=view['passes'].setdefault(key.split('/')[0],{})
            for name,value in total.items():combined[name]=combined.get(name,0)+value
        view['total']={}
        for total in view['passes'].values():
            for name,value in total.items():view['total'][name]=view['total'].get(name,0)+value
    return views


def run(label,arm='diagnostic',scenario='zoom',hour=12,frames=90,controls=None,captures=False,display='windowed',client_arm='diagnostic'):
    directory=OUT/label
    if directory.exists():raise ValueError('Preserve existing run: '+label)
    directory.mkdir()
    winroot=windows_root()
    def win(p):return str(winroot/p.relative_to(ROOT).as_posix())
    options=json.loads((LIGHT/'timing-night-3-candidate/receipt.json').read_text())['options']
    options.update(C3X_RENDERER_SHADER_SOURCE_ROOT=win(OUT/'accepted/shaders'),
        C3X_SANDBOX_STUDY_START_MS='29500',C3X_SANDBOX_CLIP_FRAMES=str(frames),
        C3X_SANDBOX_FRAME_TIMINGS='1',C3X_SANDBOX_PASS_COUNTS='1',
        C3X_SANDBOX_DISPLAY='full_guest' if display=='full_guest' else '',
        C3X_RENDERER_CITY_LIGHT_DIAGNOSTICS='',C3X_SANDBOX_PRESENT_MODE='',
        C3X_SANDBOX_STUDY_ZOOM='1' if scenario=='stationary' else '',
        C3X_SANDBOX_CAMERA_WITNESS='1' if scenario=='navigation' else '',
        C3X_SANDBOX_WITNESS_FRAMES=str(frames),C3X_SANDBOX_WITNESS_ENDPOINT='',
        C3X_SANDBOX_WITNESS_CAPTURE_DIR=win(directory) if captures else '',
        C3X_SANDBOX_CAPTURE='',C3X_SANDBOX_CAPTURE_MOVED='',C3X_SANDBOX_CAPTURE_SCROLL='',
        C3X_SANDBOX_CAPTURE_JUMP='',C3X_SANDBOX_CAPTURE_RETURN='',C3X_SANDBOX_CAPTURE_WRAP='',
        C3X_SANDBOX_CAPTURE_SEQUENCE='',C3X_SANDBOX_SKIP_VEGETATION='',
        C3X_SANDBOX_SKIP_REFLECTION='',C3X_SANDBOX_SKIP_WATER_PASS='',
        C3X_SANDBOX_SKIP_VEGETATION_DEPTH='',C3X_SANDBOX_VEGETATION_CULL_OFF='',
        C3X_SANDBOX_RESOLVE_COPY_REFERENCE='')
    options.update(controls or {})
    scene=ROOT/'Renderer/native/build/performance-review-current/developed-scene.csv'
    dll=OUT/arm/'C3XRenderer_x64.dll';client=OUT/client_arm/'client_x64.exe'
    paths=[client,dll,ROOT,ROOT/'Renderer/default.custom_rendering.txt',scene,directory/'initial.bmp']
    command=' '.join('"'+win(p)+'"' for p in paths)+f' 2240 1260 24 56 128 {hour}'
    batch=directory/'run.bat'
    batch.write_text('@echo off\nsetlocal\n'+'\n'.join(f'set "{k}={v}"' for k,v in options.items())+'\n'+command+' >"'+win(directory/'run.log')+'" 2>&1\nexit /b %errorlevel%\n')
    print('BEGIN',label,flush=True)
    wake=OUT/'keep-awake.ps1'
    result=native_command_result('Renderer/native',f'powershell -NoProfile -ExecutionPolicy Bypass -File "{win(wake)}" -BatchPath "{win(batch)}"',timeout_seconds=600)
    log=directory/'run.log'
    result.update(arm=arm,client_arm=client_arm,scenario=scenario,hour=hour,options=options,dll_sha256=sha(dll),client_sha256=sha(client),scene_sha256=sha(scene),
        shader_sha256={p.relative_to(OUT/'accepted/shaders').as_posix():sha(p) for p in (OUT/'accepted/shaders').rglob('*.hlsl')})
    if log.exists():
        raw=log.read_text(errors='replace')
        result['measurements']=summary(log)
        result['display_records']=re.findall(r'(?:CLIENT_DISPLAY|SANDBOX_SWAPCHAIN)[^\n]+',raw)
        result['camera_adoptions']=re.findall(r'CAMERA_ADOPTION[^\n]+',raw)
        result['camera_samples']=re.findall(r'CAMERA_FRAME[^\n]+',raw)
        result['navigation_measurements']=navigation_summary(raw)
        result['workload']=workload_summary(raw)
    (directory/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise RuntimeError('Inspect existing process before retry: '+label)
    if scenario=='navigation' and 'CAMERA_WITNESS pass' not in log.read_text():raise RuntimeError('Missing executed witness marker')
    print('END',label,result.get('measurements',{}).get('total',{}),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('label');p.add_argument('--arm',default='diagnostic')
    p.add_argument('--scenario',choices=['zoom','stationary','navigation'],default='zoom')
    p.add_argument('--hour',type=int,default=12);p.add_argument('--frames',type=int,default=90)
    p.add_argument('--display',choices=['windowed','full_guest'],default='windowed')
    p.add_argument('--control',action='append',default=[]);p.add_argument('--captures',action='store_true')
    a=p.parse_args();run(a.label,a.arm,a.scenario,a.hour,a.frames,dict(x.split('=',1) for x in a.control),a.captures,a.display)
