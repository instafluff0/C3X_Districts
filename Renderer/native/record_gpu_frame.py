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


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,help="Explicit disposable output directory for category dispatch")
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--dll',type=Path,default=Path('Renderer/native/build/candidate/C3XRenderer.dll'))
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--tile-width',type=int,choices=(64,128,160,192),default=128)
    parser.add_argument('--jgl',type=Path,default=Path('Renderer/native/build/gpu-composition/audit/jgl.dll'))
    parser.add_argument('--waves',choices=('0','1'),default='0',help='Enable the existing shoreline effect with identical controls in both comparison arms')
    parser.add_argument('--benchmark',action='store_true',help='Compare complete native CPU/GPU frame requests and desktop completion')
    parser.add_argument('--profile',action='store_true',help='Enable existing phase and address-space samples; match this setting in both comparison arms')
    parser.add_argument('--dense-scene',action='store_true',help='Use the existing world-fixed dense city/infrastructure/resource fixture in both comparison arms')
    parser.add_argument('--dense-city-case',default='',help='Dense city culture,era,size,capital, for example 0,3,1,1; requires --dense-scene')
    parser.add_argument('--object-workers',choices=('0','1'),default='1',help='Use the identical object compiler on the foreground (0) or bounded worker (1)')
    parser.add_argument('--ground-workers',choices=('0','1'),default='1',help='Run the production ground compiler serially (0) or on its bounded worker (1)')
    parser.add_argument("--unit-count",type=int,choices=(1,8,16,32),default=8,help="Unit count in the complete native-frame workload")
    parser.add_argument("--unit-scene",choices=("0","1"),default="1",help="Ordered direct unit scene draws (1) or preserved resident-pose control (0)")
    parser.add_argument("--visibility",action="store_true",help="Capture world-fixed visible, explored and unseen regions")
    parser.add_argument("--tactical",action="store_true",help="Exercise native tactical capture and save actual connected display previews")
    parser.add_argument("--visual-only",action="store_true",help="Use production rendering settings and validate independent visual frames without the 384-request comparison")
    parser.add_argument("--scroll-coverage",action="store_true",help="Exercise fine scrolling and guard coverage against missing map pixels")
    args=parser.parse_args(argv)
    if args.dense_city_case:
        try:
            fields=tuple(int(value) for value in args.dense_city_case.split(','))
        except ValueError:
            parser.error('dense city case requires four integer fields')
        if not args.dense_scene or len(fields)!=4 or any(value<0 or value>limit for value,limit in zip(fields,(4,3,2,1))):
            parser.error('dense city case requires --dense-scene and valid culture,era,size,capital')
        args.dense_city_case=','.join(str(value) for value in fields)
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
    invocation=uuid.uuid4().hex;out=args.out.resolve() if args.out else ROOT/'Renderer/native/build/gpu-composition'/invocation;out.mkdir(parents=True,exist_ok=True)
    inputs={}
    for unit in DLL_UNITS:inputs.update(unit_inputs(unit))
    for path in (scene,dll,jgl,ROOT/'injected_code.c',ROOT/'C3X.h',ROOT/'civ_prog_objects.csv',*[ROOT/'Renderer/native'/n for n in ('gpu_frame_preview.h','native_frame_workload.h','native_frame_benchmark.h','test_native_screen.h','test_native_bootstrap.h','test_native_worker.cpp','test_gpu_unit_composition.h','test_native_image_adapter.cpp','test_native_observation.cpp','native_image_adapter.h','native_sprite_diagnostics.h','native_composition_owner.h','native_observation.h','gpu_image_worker_client.h','gpu_image_commands.h','color_quantization.h','test_gpu_frame_api.c','biq_preview.cpp','BUILD.bat','record_gpu_frame.py')]):inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    win=windows_root();target=win/out.relative_to(ROOT)
    settings={'C3X_RENDERER_GPU_JGL_TEST':str(win/jgl.relative_to(ROOT)),'C3X_RENDERER_VISUAL_PROFILE':'city-fidelity','C3X_RENDERER_SHARED_SCENE_SURFACE':'',
        'C3X_RENDERER_REFLECTION_CONTROL':'1','C3X_RENDERER_WAVES':args.waves,'C3X_RENDERER_GPU_FRAME_TEST':'1','C3X_RENDERER_SCROLL_COVERAGE_TEST':'1' if args.scroll_coverage else '','C3X_RENDERER_NATIVE_FRAME_BENCHMARK':'1' if args.benchmark else '',
        'C3X_RENDERER_PROFILE':'1' if args.profile else '0','C3X_RENDERER_GROUND_WORKERS':args.ground_workers,'C3X_RENDERER_OBJECT_WORKERS':args.object_workers,
        'C3X_RENDERER_TRACE':'2','C3X_RENDERER_TRACE_MIB':'32','C3X_RENDERER_TRACE_BUFFERED':'1' if args.benchmark or args.visual_only else '', 'C3X_RENDERER_TRACE_FILE':str(target/'renderer.log'),
        'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':r'..\..\Renderer\custom.custom_rendering.txt',
        'C3X_RENDERER_PREVIEW_OBJECTS':'1','C3X_RENDERER_PREVIEW_CITY':'0,3,1,1',
        'C3X_RENDERER_PREVIEW_DENSE_SCENE':'1' if args.dense_scene else '',
        'C3X_RENDERER_PREVIEW_DENSE_CITY_CASE':args.dense_city_case,
        'C3X_RENDERER_PREVIEW_VISIBILITY':'1' if args.visibility else '',
        'C3X_RENDERER_UNIT_SCENE_CONTROL':'1' if args.unit_scene=='0' else '',
        'C3X_RENDERER_BENCHMARK_UNITS':str(args.unit_count),
        'C3X_RENDERER_TACTICAL_PREVIEW':str(target/'tactical') if args.tactical else '',
        'C3X_RENDERER_PREVIEW_SESSION':'','C3X_RENDERER_PREVIEW_REPLAY':'','C3X_RENDERER_PREVIEW_ANIMATION':''}
    if args.benchmark or args.visual_only or args.scroll_coverage:
        # Match configure_custom_renderer_effects with the shipped cache enabled.
        # These are harness settings, never a setup requirement for the player.
        settings.update({
            'C3X_RENDERER_BLOCK_CLIP':'1','C3X_RENDERER_RASTER_REUSE_CONTROL':'1',
            'C3X_RENDERER_WORLD_RASTER_GRID':'1','C3X_RENDERER_WORLD_REGIONS':'1',
            'C3X_RENDERER_REGION_RECEIVER_SHADOWS':'1','C3X_RENDERER_TIGHT_NATURAL_BOUNDS':'1',
            'C3X_RENDERER_REGION_INPUT_RING':'4','C3X_RENDERER_REGION_SIZE':'128',
            'C3X_RENDERER_REGION_METADATA_MIB':'96','C3X_RENDERER_WORLD_REGIONS_CONTROL':'0',
            'C3X_RENDERER_BOUNDED_POST':'0'})
    command=f'build\\gpu-composition\\test_gpu_frame.exe "{win/dll.relative_to(ROOT)}" ..\\.. ..\\default.custom_rendering.txt "{win/scene.relative_to(ROOT)}" "{target/"control.bmp"}" {args.width} {args.height} {cx} {cy} {args.tile_width} 12'
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
    passed=complete==[invocation,'0'] and unchanged and 'PASS resident map GPU worker:' in log and 'PASS native GPU worker transport:' in log and 'PASS native screen transfer:' in log and 'PASS live native screen:' in log and 'PASS production native map owner:' in log and ('PASS prepared GPU map adoption:' in log or (args.width>2224 and args.height>1176 and 'PASS bounded GPU map demand:' in log))
    receipt={'status':'pass' if passed else 'fail' if complete else 'unconfirmed','inputs':inputs,'inputs_unchanged':unchanged,'transport_returncode':process.returncode,'transport_output':process.stdout+process.stderr,'settings':settings,'scope':'production captured renderer map -> existing GPU worker -> packed composition; oracle readback explicit; actual native final presentation including CPU compatibility callback; no game speedup claim'}
    trace=(out/'renderer.log').read_text(errors='replace') if (out/'renderer.log').exists() else ''
    import re
    dropped=sum(int(value) for value in re.findall(r'TRACE_BUFFER dropped=(\d+)',trace))
    receipt['trace_coverage']={'dropped_lines':dropped,'complete':dropped==0}
    resident_units=[line for line in trace.splitlines() if 'resident_pose=1' in line or 'direct_scene=1' in line]
    resident_proof=bool(resident_units) and any('cache_hit=0' in line for line in resident_units) and any('cache_hit=1' in line for line in resident_units) and all('body_readbacks=0 composition_uploads=0' in line for line in resident_units)
    receipt['resident_unit_proof']={'requests':len(resident_units),'cold_and_warm_without_body_readback_or_composition_upload':resident_proof}
    # The connected producer must finish real cold poses without the old CPU
    # round trip, not merely avoid reading its destination/background canvas.
    passed=passed and resident_proof
    direct=[line for line in trace.splitlines() if 'stage=unit-scene ' in line]
    if direct:
        direct_proof=all('body_readbacks=0 composition_uploads=0' in line for line in direct)
        receipt['direct_unit_scene_proof']={'requests':len(direct),'no_CPU_round_trip':direct_proof}
        passed=passed and direct_proof
        import re
        map_draws=max((int(m.group(1)) for line in direct if (m:=re.search(r'\bmap_draws=(\d+)',line))),default=0)
        region_peak=max((int(m.group(1)) for line in direct if (m:=re.search(r'\bregion_bytes=(\d+)',line))),default=0)
        work_peak=max((int(m.group(1)) for line in direct if (m:=re.search(r'\bwork_bytes=(\d+)',line))),default=0)
        content_peak=max((int(m.group(1)) for line in direct if (m:=re.search(r'\bgpu_content_bytes=(\d+)',line))),default=0)
        passed=passed and content_peak<=192*1024*1024
        receipt['direct_unit_scene_proof'].update(maximum_reported_gpu_content_bytes=content_peak,direct_body_draws=map_draws,maximum_reported_region_bytes=region_peak,maximum_reported_work_bytes=work_peak)
        passed=passed and work_peak<=96*1024*1024
        if args.unit_scene=='1' and not args.visibility and (args.benchmark or args.visual_only):
            passed=passed and map_draws>0 and region_peak<=64*1024*1024
    receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
    if args.scroll_coverage:
        passed=complete==[invocation,'0'] and unchanged and 'PASS scroll coverage:' in log
        receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
        receipt['scope']='fine-pan/zoom map coverage with idle guard preparation; missing pixels compared against a cold no-guard render; no native UI or performance claim'
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
        if direct:
            import bisect
            fields=('map_draws','compatibility_builds','compatibility_hits','region_captures','region_evictions','region_reuses','gpu_content_builds','gpu_content_hits','gpu_content_reuses')
            rows=[]
            for line in direct:
                values=dict(re.findall(r'(\w+)=(\d+)',line))
                if 'qpc' in values:rows.append({key:int(value) for key,value in values.items()})
            times=[row['qpc'] for row in rows];totals=dict.fromkeys(fields,0)
            for sample in samples:
                if sample['route']!='GPU':continue
                a=bisect.bisect_left(times,int(sample['begin_qpc']))-1
                b=bisect.bisect_right(times,int(sample['end_qpc']))-1
                if b<0:continue
                for field in fields:totals[field]+=rows[b].get(field,0)-(rows[a].get(field,0) if a>=0 else 0)
            receipt['direct_unit_scene_proof']['timed_GPU_operations']=totals
            receipt['direct_unit_scene_proof']['counter_totals_are_lower_bounds']=dropped>0
            receipt['direct_unit_scene_proof']['timed_GPU_samples_with_counter_coverage']=sum(1 for sample in samples if sample['route']=='GPU' and times and (not dropped or int(sample['end_qpc'])<=times[-1]))
            if args.unit_scene=='1' and not args.visibility:passed=passed and totals['map_draws']>0
        passed=passed and not parse_errors and len(samples)==384 and 'PASS whole native frame comparison:' in log
        receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
    if args.visual_only or args.benchmark:
        passed=passed and 'PASS independent resident frames:' in log and 'PASS visual timer transport:' in log
        receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
        receipt['visual_frames']=[line for line in log.splitlines() if line.startswith(('VISUAL_SAMPLE ','PASS independent resident frames:','PASS visual timer transport:'))]
    if args.tactical:
        passed=passed and 'PASS tactical native composition:' in log
        receipt['tactical']={'capture':'actual native JGL line/text seams','previews':['tactical-route.bmp','tactical-grid.bmp'],'passed':passed}
        receipt['status']='pass' if passed else 'fail'
    if passed:
        shutil.copy2(ROOT/'Renderer/native/build/gpu-composition/test_gpu_frame.exe',out/'test_gpu_frame.exe')
        shutil.copy2(dll,out/'C3XRenderer.dll')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if (out/'build.log').exists():print((out/'build.log').read_text(errors='replace')[-3000:])
    print(log[-4500:]);print(receipt['status'])
    if not complete:print('Native dispatch has no completion receipt; check the VM before another run. '+process.stdout+process.stderr)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
