"""Execute extracted production hooks against local JGL, without launching Civ III."""
import argparse
import hashlib
import json
import re
import os
from pathlib import Path
import shutil
import subprocess
import uuid
from Renderer.lab.platform import ROOT, windows_root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jgl',type=Path,required=True)
    parser.add_argument('--gpu',action='store_true',help='Validate the packed GPU executor against actual JGL images')
    parser.add_argument('--adapter',action='store_true',help='Execute actual hooked JGL operations on the GPU with CPU ownership barriers')
    parser.add_argument('--lifetimes',action='store_true',help='Execute startup tracking and native lifetime/escape contracts')
    parser.add_argument('--observer-dll',type=Path,help='Also execute the staged observation export with the current hooks')
    args=parser.parse_args()
    if sum([args.gpu,args.adapter,args.lifetimes])>1:parser.error('choose one backend contract')
    if args.lifetimes and not args.observer_dll:parser.error('lifetime contract requires --observer-dll with the candidate renderer')
    if args.observer_dll and (args.gpu or args.adapter):parser.error('observer DLL requires pass-through contract')
    observer=args.observer_dll.resolve() if args.observer_dll else None
    if observer and any(c in str(observer.relative_to(ROOT)) for c in '\r\n"%&|<>^!'):parser.error('unsupported observer path')
    jgl=args.jgl.resolve();relative=jgl.relative_to(ROOT)
    if any(c in str(relative) for c in '\r\n"%&|<>^!'):
        parser.error('unsupported path characters')
    if digest(jgl)!='0b0cd514de0d95b93d20655f4b5194173fe257325af82e558a152305ff0dbdf2':
        parser.error('unrecognized JGL binary')
    native=ROOT/'Renderer/native';build=native/'build'
    text=(ROOT/'injected_code.c').read_text()
    start=text.index('int\ninitialize_native_line_backend')
    # C's native `this` parameter is a reserved token in the C++ harness.
    (build/'native_line_hooks.h').write_text(re.sub(r'\bthis\b','context',text[start:text.index('int __fastcall\npatch_Tile_check_water',start)]))
    if args.lifetimes:
        start=text.index('void\nstart_custom_renderer_native_tracking ()')
        (build/'native_tracking_bootstrap.h').write_text(text[start:text.index('void\npatch_init_floating_point ()',start)])
    (build/'native_probe_hooks.h').write_text(text[text.index('// JGL observation hooks:'):text.index('// End JGL observation hooks.')])
    text=(ROOT/'C3X.h').read_text();start=text.index('\tc3x_renderer_native_observe_fn')
    (build/'native_probe_state.h').write_text(text[start:text.index('\tc3x_renderer_unit_draw_background_fn',start)])
    invocation=uuid.uuid4().hex;out=build/'gpu-composition'/invocation;out.mkdir()
    inputs=[ROOT/'injected_code.c',ROOT/'C3X.h',ROOT/'civ_prog_objects.csv',jgl,*[native/n for n in
        ('c3x_renderer_api.h','native_observation.h','test_native_observation.cpp','test_native_line_bridge.h','record_native_observation.py','BUILD.bat','gpu_image_compositor.h','gpu_image_commands.h','test_local_image_backend.h','test_gpu_image_compositor.cpp','native_image_adapter.h','native_sprite_diagnostics.h','test_native_image_adapter.cpp','test_native_ui_assets.h','native_ui_fixture.py','native_lifetime_registry.h','test_native_bootstrap.h','test_native_lifetimes.cpp')]]
    if observer:inputs.append(observer)
    before={p.relative_to(ROOT).as_posix():digest(p) for p in inputs}
    if args.adapter:
        from Renderer.native.native_ui_fixture import prepare
        prepare(ROOT,out/'native-ui.pack',before)
    win=windows_root();winout=win/out.relative_to(ROOT)
    mode='gpu-image-operations' if args.gpu else 'native-observation'
    executable='test_gpu_image_compositor.exe' if args.gpu else 'test_native_observation.exe'
    marker='PASS GPU native operations:' if args.gpu else 'PASS actual injected JGL hooks:'
    if args.adapter:
        mode='native-image-adapter';executable='test_native_image_adapter.exe';marker='PASS hooked native GPU adapter:'
    if args.lifetimes:
        mode='native-lifetimes';executable='test_native_lifetimes.exe';marker='PASS native startup lifetimes:'
    observer_argument=f' "{win/observer.relative_to(ROOT)}"' if observer else ''
    run=out/'run.cmd';run.write_text('@echo off\nsetlocal\n'+(f'set \"C3X_RENDERER_NATIVE_UI_PACK={winout/"native-ui.pack"}\"\n' if args.adapter else '')+f'pushd "{win/"Renderer/native"}"\n'
        +f'call BUILD.bat {mode} >"{winout/"build.log"}" 2>&1\nif errorlevel 1 goto failed\n'
        +f'build\\gpu-composition\\{executable} "{win/relative}"{observer_argument} >"{winout/"test.log"}" 2>&1\n'
        +'if not "%errorlevel%"=="0" goto failed\n'+f'>"{winout/"completion.txt"}" echo {invocation} 0\nexit /b 0\n:failed\n'
        +f'>"{winout/"completion.txt"}" echo {invocation} 1\nexit /b 1\n')
    command=['prlctl','exec',os.environ.get('C3X_RENDERER_VM','Windows 11'),'--current-user','cmd','/d','/s','/c',f'call "{winout/"run.cmd"}"']
    if os.name=='nt':command=['cmd','/d','/c',str(run)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    completion=(out/'completion.txt').read_text().split() if (out/'completion.txt').exists() else []
    unchanged=all(digest(ROOT/path)==value for path,value in before.items())
    passed=completion==[invocation,'0'] and unchanged and (out/'test.log').is_file() and marker in (out/'test.log').read_text(errors='replace')
    exe=build/'gpu-composition'/executable
    if passed:shutil.copy2(exe,out/exe.name)
    receipt={'status':'pass' if passed else 'fail' if completion else 'unconfirmed','inputs':before,'inputs_unchanged':unchanged,
             'transport_returncode':result.returncode,'transport_output':result.stdout+result.stderr,
             'executable_sha256':digest(exe) if passed else None,
             'scope':('GPU packed image executor against actual JGL; no game integration claim' if args.gpu else 'actual injected hooks against isolated JGL; native final wrapper uses an ordered stub; no game/UI coverage claim')}
    if args.adapter:receipt['scope']='actual injected hooks substitute GPU copy/fill; isolated native images and CPU fallback; no live map or final-transfer replacement'
    if args.lifetimes:receipt['scope']='actual startup bootstrap and JGL hooks with candidate lifetime service; read-only ownership evidence, no map/GPU admission or game launch'
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name in ('build.log','test.log'):
        if (out/name).exists():print((out/name).read_text(errors='replace')[-7000:])
    if not completion:print('No native completion receipt; inspect VM before retry. '+result.stdout+result.stderr)
    print(out.relative_to(ROOT),receipt['status']);return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
