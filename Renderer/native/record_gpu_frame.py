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
    parser.add_argument('--jgl',type=Path,default=Path('Renderer/native/build/gpu-composition/audit/jgl.dll'))
    parser.add_argument('--benchmark',action='store_true',help='Compare complete native CPU/GPU frame requests and desktop completion')
    parser.add_argument("--visual-only",action="store_true",help="Use production rendering settings and validate independent visual frames without the 384-request comparison")
    args=parser.parse_args()
    jgl=args.jgl.resolve()
    if digest(jgl)!='0b0cd514de0d95b93d20655f4b5194173fe257325af82e558a152305ff0dbdf2':parser.error('unrecognized JGL binary')
    native=ROOT/'Renderer/native';build=native/'build'
    unit_source=(native/'unit_body_renderer.h').read_text()
    storage=unit_source[unit_source.index('    HDC dc=nullptr;'):unit_source.index('    bool ensure(ID3D11Device*')]
    blitter=unit_source[unit_source.index('    bool blit_pixels('):unit_source.index('    unsigned keyed_pixels=')]
    (build/'native_unit_blitter.h').write_text('struct NativeUnitBlitter {\n'+storage+blitter+'\n~NativeUnitBlitter(){reset_blit();}\n};\n')
    text=(ROOT/'injected_code.c').read_text()
    (build/'native_probe_hooks.h').write_text(text[text.index('// JGL observation hooks:'):text.index('// End JGL observation hooks.')])
    start=text.index('void\nstart_custom_renderer_native_tracking ()')
    (build/'native_tracking_bootstrap.h').write_text(text[start:text.index('void\npatch_init_floating_point ()',start)])
    text=(ROOT/'C3X.h').read_text();start=text.index('\tc3x_renderer_native_observe_fn')
    (build/'native_probe_state.h').write_text(text[start:text.index('\tc3x_renderer_unit_draw_background_fn',start)])
    scene=args.scene.resolve();dll=args.dll.resolve()
    for path in (scene,dll,jgl):
        if any(c in path.relative_to(ROOT).as_posix() for c in '\r\n"%&|<>^!'):parser.error('unsupported input path')
    if not(64<=args.width<=2240 and 64<=args.height<=1192):parser.error('unsupported extent')
    header=scene.read_text().splitlines()[0].split(',');cx=int(header[1])//2;cy=int(header[2])//2
    invocation=uuid.uuid4().hex;out=ROOT/'Renderer/native/build/gpu-composition'/invocation;out.mkdir()
    inputs={}
    for unit in DLL_UNITS:inputs.update(unit_inputs(unit))
    for path in (scene,dll,jgl,ROOT/'injected_code.c',ROOT/'C3X.h',ROOT/'civ_prog_objects.csv',*[ROOT/'Renderer/native'/n for n in ('gpu_frame_preview.h','native_frame_workload.h','native_frame_benchmark.h','test_native_screen.h','test_native_bootstrap.h','test_native_worker.cpp','test_gpu_unit_composition.h','test_native_image_adapter.cpp','test_native_observation.cpp','native_image_adapter.h','native_sprite_diagnostics.h','native_composition_owner.h','native_observation.h','gpu_image_worker_client.h','gpu_image_commands.h','color_quantization.h','test_gpu_frame_api.c','biq_preview.cpp','BUILD.bat','record_gpu_frame.py')]):inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    win=windows_root();target=win/out.relative_to(ROOT)
    settings={'C3X_RENDERER_GPU_JGL_TEST':str(win/jgl.relative_to(ROOT)),'C3X_RENDERER_VISUAL_PROFILE':'city-fidelity','C3X_RENDERER_SHARED_SCENE_SURFACE':'1',
        'C3X_RENDERER_REFLECTION_CONTROL':'1','C3X_RENDERER_WAVES':'0','C3X_RENDERER_GPU_FRAME_TEST':'1','C3X_RENDERER_NATIVE_FRAME_BENCHMARK':'1' if args.benchmark else '',
        'C3X_RENDERER_TRACE':'2','C3X_RENDERER_TRACE_BUFFERED':'1' if args.benchmark or args.visual_only else '', 'C3X_RENDERER_TRACE_FILE':str(target/'renderer.log'),
        'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':r'..\..\Renderer\custom.custom_rendering.txt',
        'C3X_RENDERER_PREVIEW_OBJECTS':'1','C3X_RENDERER_PREVIEW_CITY':'0,3,1,1',
        'C3X_RENDERER_PREVIEW_SESSION':'','C3X_RENDERER_PREVIEW_REPLAY':'','C3X_RENDERER_PREVIEW_ANIMATION':''}
    if args.benchmark or args.visual_only:
        # Match configure_custom_renderer_effects with the shipped cache enabled.
        # These are harness settings, never a setup requirement for the player.
        settings.update({
            'C3X_RENDERER_BLOCK_CLIP':'1','C3X_RENDERER_RASTER_REUSE_CONTROL':'1',
            'C3X_RENDERER_WORLD_RASTER_GRID':'1','C3X_RENDERER_WORLD_REGIONS':'1',
            'C3X_RENDERER_REGION_RECEIVER_SHADOWS':'1','C3X_RENDERER_TIGHT_NATURAL_BOUNDS':'1',
            'C3X_RENDERER_REGION_INPUT_RING':'4','C3X_RENDERER_REGION_SIZE':'128',
            'C3X_RENDERER_REGION_METADATA_MIB':'96','C3X_RENDERER_WORLD_REGIONS_CONTROL':'0',
            'C3X_RENDERER_BOUNDED_POST':'0'})
    command=f'build\\gpu-composition\\test_gpu_frame.exe "{win/dll.relative_to(ROOT)}" ..\\.. ..\\default.custom_rendering.txt "{win/scene.relative_to(ROOT)}" "{target/"control.bmp"}" {args.width} {args.height} {cx} {cy} 128 12'
    (out/'run.cmd').write_text('@echo off\nsetlocal\n'+f'pushd "{win/"Renderer/native"}"\n'
        +f'call BUILD.bat gpu-frame >"{target/"build.log"}" 2>&1\nif errorlevel 1 goto failed\n'
        +''.join(f'set "{k}={v}"\n' for k,v in settings.items())
        +command+f' >"{target/"test.log"}" 2>&1\nif not "%errorlevel%"=="0" goto failed\n'
        +f'>"{target/"completion.txt"}" echo {invocation} 0\nexit /b 0\n:failed\n>"{target/"completion.txt"}" echo {invocation} 1\nexit /b 1\n')
    print(out.relative_to(ROOT),flush=True)
    process=subprocess.run(['prlctl','exec',os.environ.get('C3X_RENDERER_VM','Windows 11'),'--current-user','cmd','/d','/s','/c',f'call "{target/"run.cmd"}"'],capture_output=True,text=True,timeout=900 if args.benchmark else 240)
    complete=(out/'completion.txt').read_text().split() if (out/'completion.txt').exists() else []
    log=(out/'test.log').read_text(errors='replace') if (out/'test.log').exists() else ''
    unchanged=all(digest(ROOT/p)==h for p,h in inputs.items())
    passed=complete==[invocation,'0'] and unchanged and 'PASS resident map GPU worker:' in log and 'PASS native GPU worker transport:' in log and 'PASS native screen transfer:' in log and 'PASS live native screen:' in log and 'PASS production native map owner:' in log and 'PASS prepared GPU map adoption:' in log
    receipt={'status':'pass' if passed else 'fail' if complete else 'unconfirmed','inputs':inputs,'inputs_unchanged':unchanged,'transport_returncode':process.returncode,'transport_output':process.stdout+process.stderr,'settings':settings,'scope':'production captured renderer map -> existing GPU worker -> packed composition; oracle readback explicit; actual native final presentation including CPU compatibility callback; no game speedup claim'}
    trace=(out/'renderer.log').read_text(errors='replace') if (out/'renderer.log').exists() else ''
    resident_units=[line for line in trace.splitlines() if 'resident_pose=1' in line]
    resident_proof=bool(resident_units) and any('cache_hit=0' in line for line in resident_units) and any('cache_hit=1' in line for line in resident_units) and all('body_readbacks=0 composition_uploads=0' in line for line in resident_units)
    receipt['resident_unit_proof']={'requests':len(resident_units),'cold_and_warm_without_body_readback_or_composition_upload':resident_proof}
    # The connected producer must finish real cold poses without the old CPU
    # round trip, not merely avoid reading its destination/background canvas.
    passed=passed and resident_proof
    receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
    if args.benchmark:
        import re,statistics
        samples=[];parse_errors=[]
        for line in log.splitlines():
            if line.startswith('FRAME_SAMPLE '):
                record=dict(re.findall(r'(\w+)=([^ ]+)',line))
                try:
                    for key in ('workload','block','step','request_ms','desktop_ms','map_ms','units_ms','UI_present_prepare_ms','sample_age_ms'):float(record[key])
                    if record['route'] not in ('CPU','GPU'):raise ValueError('unknown route')
                except (ValueError,KeyError):parse_errors.append(line)
                else:samples.append(record)
        groups=[]
        for workload in range(3):
            for route in ('CPU','GPU'):
                selected=[r for r in samples if r['workload']==str(workload) and r['route']==route]
                summary={'workload':workload,'route':route,'samples':len(selected)}
                for key in ('request_ms','desktop_ms','map_ms','units_ms','UI_present_prepare_ms','sample_age_ms','geometry_ms','draw_ms','readback_ms'):
                    values=sorted(float(r[key]) for r in selected if key in r)
                    if values:summary[key]={'mean':statistics.mean(values),'median':statistics.median(values),'p95':values[min(len(values)-1,int(len(values)*.95))],'max':max(values)}
                groups.append(summary)
        receipt['whole_frame_comparison']={'groups':groups,'samples':samples,'parse_errors':parse_errors,'control':'same candidate DLL using native CPU publication, blit, units, JGL UI and native GDI final transfer','capture_outside_timing':True,'desktop_completion_is_not_physical_scanout':True}
        passed=passed and not parse_errors and len(samples)==384 and 'PASS whole native frame comparison:' in log
        receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
    if args.visual_only:
        passed=passed and 'PASS independent resident frames:' in log and 'PASS visual timer transport:' in log
        receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
        receipt['visual_frames']=[line for line in log.splitlines() if line.startswith(('VISUAL_SAMPLE ','PASS independent resident frames:','PASS visual timer transport:'))]
    if passed:
        shutil.copy2(ROOT/'Renderer/native/build/gpu-composition/test_gpu_frame.exe',out/'test_gpu_frame.exe')
        shutil.copy2(dll,out/'C3XRenderer.dll')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if (out/'build.log').exists():print((out/'build.log').read_text(errors='replace')[-3000:])
    print(log[-4500:]);print(receipt['status'])
    if not complete:print('Native dispatch has no completion receipt; check the VM before another run. '+process.stdout+process.stderr)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
