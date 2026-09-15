"""Validate resident-map publication and composition on the existing GPU worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from Renderer.lab.platform import ROOT, windows_root
from Renderer.native.record_renderer_build import unit_inputs, DLL_UNITS


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--dll',type=Path,default=Path('Renderer/native/build/candidate/C3XRenderer.dll'))
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480)
    args=parser.parse_args()
    scene=args.scene.resolve();dll=args.dll.resolve()
    for path in (scene,dll):
        if any(c in path.relative_to(ROOT).as_posix() for c in '\r\n"%&|<>^!'):parser.error('unsupported input path')
    if not(64<=args.width<=2240 and 64<=args.height<=1192):parser.error('unsupported extent')
    header=scene.read_text().splitlines()[0].split(',');cx=int(header[1])//2;cy=int(header[2])//2
    invocation=uuid.uuid4().hex;out=ROOT/'Renderer/native/build/gpu-composition'/invocation;out.mkdir()
    inputs={}
    for unit in DLL_UNITS:inputs.update(unit_inputs(unit))
    for path in (scene,dll,*[ROOT/'Renderer/native'/n for n in ('gpu_frame_preview.h','test_gpu_frame_api.c','biq_preview.cpp','BUILD.bat','record_gpu_frame.py')]):inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    win=windows_root();target=win/out.relative_to(ROOT)
    settings={'C3X_RENDERER_VISUAL_PROFILE':'city-fidelity','C3X_RENDERER_SHARED_SCENE_SURFACE':'1',
        'C3X_RENDERER_REFLECTION_CONTROL':'1','C3X_RENDERER_WAVES':'0','C3X_RENDERER_GPU_FRAME_TEST':'1',
        'C3X_RENDERER_TRACE':'1','C3X_RENDERER_TRACE_FILE':str(target/'renderer.log'),
        'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':r'..\..\Renderer\custom.custom_rendering.txt',
        'C3X_RENDERER_PREVIEW_OBJECTS':'1','C3X_RENDERER_PREVIEW_CITY':'0,3,1,1',
        'C3X_RENDERER_PREVIEW_SESSION':'','C3X_RENDERER_PREVIEW_REPLAY':'','C3X_RENDERER_PREVIEW_ANIMATION':''}
    command=f'build\\gpu-composition\\test_gpu_frame.exe "{win/dll.relative_to(ROOT)}" ..\\.. ..\\default.custom_rendering.txt "{win/scene.relative_to(ROOT)}" "{target/"control.bmp"}" {args.width} {args.height} {cx} {cy} 128 12'
    (out/'run.cmd').write_text('@echo off\nsetlocal\n'+f'pushd "{win/"Renderer/native"}"\n'
        +f'call BUILD.bat gpu-frame >"{target/"build.log"}" 2>&1\nif errorlevel 1 goto failed\n'
        +''.join(f'set "{k}={v}"\n' for k,v in settings.items())
        +command+f' >"{target/"test.log"}" 2>&1\nif not "%errorlevel%"=="0" goto failed\n'
        +f'>"{target/"completion.txt"}" echo {invocation} 0\nexit /b 0\n:failed\n>"{target/"completion.txt"}" echo {invocation} 1\nexit /b 1\n')
    print(out.relative_to(ROOT),flush=True)
    process=subprocess.run(['prlctl','exec',os.environ.get('C3X_RENDERER_VM','Windows 11'),'--current-user','cmd','/d','/s','/c',f'call "{target/"run.cmd"}"'],capture_output=True,text=True,timeout=240)
    complete=(out/'completion.txt').read_text().split() if (out/'completion.txt').exists() else []
    log=(out/'test.log').read_text(errors='replace') if (out/'test.log').exists() else ''
    unchanged=all(digest(ROOT/p)==h for p,h in inputs.items())
    passed=complete==[invocation,'0'] and unchanged and 'PASS resident map GPU worker:' in log
    receipt={'status':'pass' if passed else 'fail' if complete else 'unconfirmed','inputs':inputs,'inputs_unchanged':unchanged,'transport_returncode':process.returncode,'transport_output':process.stdout+process.stderr,'scope':'production captured renderer map -> existing GPU worker -> packed composition; oracle readback explicit; no native final presentation or game speedup claim'}
    if passed:
        shutil.copy2(ROOT/'Renderer/native/build/gpu-composition/test_gpu_frame.exe',out/'test_gpu_frame.exe')
        shutil.copy2(dll,out/'C3XRenderer.dll')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if (out/'build.log').exists():print((out/'build.log').read_text(errors='replace')[-3000:])
    print(log[-4500:]);print(receipt['status'])
    if not complete:print('Native dispatch has no completion receipt; check the VM before another run. '+process.stdout+process.stderr)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
