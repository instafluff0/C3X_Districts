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
    import re
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,help="Explicit disposable output directory for category dispatch")
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--dll',type=Path,default=Path('Renderer/native/build/candidate/C3XRenderer.dll'))
    parser.add_argument('--reserve-address-mib',type=int,choices=range(0,1537,64),default=0,help='Harness-only reservation simulating co-resident process address-space pressure')
    parser.add_argument('--input-soak-seconds', type=int, choices=(30, 600), help='Real wall-clock native recorder endurance; capture-on/off use identical input owners')
    parser.add_argument('--record-inputs',action='store_true',help='Development renderer input capture; not yet gameplay qualification')
    parser.add_argument('--window-witness-seconds',type=int,choices=range(5,901),help='External sampled window evidence alongside this fixture; not a performance baseline')
    parser.add_argument('--record-composition',action='store_true',help='Record exact native GPU image traffic; adds diagnostic readbacks and invalidates performance-baseline claims')
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--tile-width',type=int,choices=(64,128,160,192),default=128)
    parser.add_argument('--jgl',type=Path,default=Path('Renderer/native/build/gpu-composition/audit/jgl.dll'))
    parser.add_argument('--waves',choices=('0','1'),default='1',help='Enable the existing shoreline effect with identical controls in both comparison arms')
    parser.add_argument('--reflections',choices=('0','1'),default='1',help='Keep object reflections enabled; off is an explicit diagnostic control')
    parser.add_argument('--water-motion',choices=('0','1'),default='1',help='Advance open-water and river material normals on the retained scene')
    parser.add_argument('--native-recovery',action='store_true',help='Exercise pending/ready cancellation and reset/recreation through production native exports')
    parser.add_argument('--native-navigation',action='store_true',help='Exercise the live navigation coordinator and fresh-capture commit through the real native owner')
    parser.add_argument('--native-camera-requests',action='store_true',help='Use nonblocking production native map transactions; the test driver measures completion separately')
    parser.add_argument('--atomic-camera-views',action='store_true',help='Require the complete GPU publication identity transition and cold-pixel matrix')
    parser.add_argument('--camera-requests',action='store_true',help='Exercise replaceable GPU camera exports and retained front lifetime')
    parser.add_argument('--benchmark',action='store_true',help='Compare complete native CPU/GPU frame requests and desktop completion')
    parser.add_argument('--profile',action='store_true',help='Enable existing phase and address-space samples; match this setting in both comparison arms')
    parser.add_argument('--completion-probe',action='store_true',help='Serialize GPU completion at existing scene/resolve/finish boundaries for attribution only; requires an oracle build and is not performance acceptance')
    parser.add_argument('--half-pixels',action='store_true',help='Diagnostic-only half geometry coverage; retains full finishing and is never visual/performance acceptance')
    parser.add_argument('--world-readiness-only',action='store_true',help='Run only the complete world-navigation workload and independent cold-pixel oracles')
    parser.add_argument('--world-readiness',action='store_true',help='Page the complete world through the production caller callback, prepare it independently of the route, and measure 100 distributed GPU requests')
    parser.add_argument('--world-geometry-mib',type=int,choices=(384,512,640,768),help='Isolated Huge-world residency budget control; leaves other budgets and detail unchanged')
    parser.add_argument('--rigid-sources',choices=('0','1'),default='1',help='Shared rigid infrastructure geometry (1) or exact expanded compiler control (0)')
    parser.add_argument('--dense-scene',action='store_true',help='Use the existing world-fixed dense city/infrastructure/resource fixture in both comparison arms')
    parser.add_argument('--dense-city-case',default='',help='Dense city culture,era,size,capital, for example 0,3,1,1; requires --dense-scene')
    parser.add_argument('--object-workers',choices=('0','1'),default='1',help='Use the identical object compiler on the foreground (0) or bounded worker (1)')
    parser.add_argument('--ground-workers',choices=('0','1'),default='1',help='Run the production ground compiler serially (0) or on its bounded worker (1)')
    parser.add_argument("--unit-count",type=int,choices=(1,8,16,32),default=8,help="Unit count in the complete native-frame workload")
    parser.add_argument("--visual-units",type=int,choices=(1,8,16,32),default=1,help="Actual retained units in independent visual frames")
    parser.add_argument("--visual-frames",type=int,choices=range(30,1201),default=30,help="Independent visual opportunities; use at least 100 for percentile acceptance")
    parser.add_argument("--visual-unit-case",choices=("selected","work","mixed"),default="selected",help="One selected idle unit plus frozen idle, authored workers, or mixed frozen/work/native-action units")
    parser.add_argument("--unit-scene",choices=("0","1"),default="1",help="Ordered direct unit scene draws (1) or preserved resident-pose control (0)")
    parser.add_argument("--visibility",action="store_true",help="Capture world-fixed visible, explored and unseen regions")
    parser.add_argument("--tactical",action="store_true",help="Exercise native tactical capture and save actual connected display previews")
    parser.add_argument("--visual-only",action="store_true",help="Use production rendering settings and validate independent visual frames without the 384-request comparison")
    parser.add_argument("--scroll-coverage",action="store_true",help="Exercise fine scrolling and guard coverage against missing map pixels")
    args=parser.parse_args(argv)
    if args.window_witness_seconds and args.benchmark:
        parser.error('Window evidence competes for GPU/CPU; use a separate witness run from the benchmark')
    if args.window_witness_seconds and (not args.input_soak_seconds or args.window_witness_seconds>args.input_soak_seconds-5):
        parser.error('Window evidence requires an input soak at least five seconds longer than capture')
    if args.world_readiness_only:args.world_readiness=True
    if args.native_recovery:args.native_navigation=True
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
    start=text.index('int\ninitialize_native_line_backend')
    # C's native `this` parameter is a reserved token in the C++ harness.
    (build/'native_line_hooks.h').write_text(re.sub(r'\bthis\b','context',text[start:text.index('int __fastcall\npatch_Tile_check_water',start)]))
    (build/'native_probe_hooks.h').write_text(text[text.index('// JGL observation hooks:'):text.index('// End JGL observation hooks.')])
    start=text.index('void\nstart_custom_renderer_native_tracking ()')
    (build/'native_tracking_bootstrap.h').write_text(text[start:text.index('void\npatch_init_floating_point ()',start)])
    text=(ROOT/'C3X.h').read_text();start=text.index('\tc3x_renderer_native_observe_fn')
    (build/'native_probe_state.h').write_text(text[start:text.index('\tc3x_renderer_unit_draw_background_fn',start)])
    scene=args.scene.resolve();dll=args.dll.resolve()
    for path in (scene,dll,jgl):
        if any(c in path.relative_to(ROOT).as_posix() for c in '\r\n"%&|<>^!'):parser.error('unsupported input path')
    if not(64<=args.width<=2240 and 64<=args.height<=1260):parser.error('unsupported extent')
    header=scene.read_text().splitlines()[0].split(',');cx=int(header[1])//2;cy=int(header[2])//2
    invocation=uuid.uuid4().hex;out=args.out.resolve() if args.out else ROOT/'Renderer/native/build/gpu-composition'/invocation;out.mkdir(parents=True,exist_ok=True)
    inputs={}
    for unit in DLL_UNITS:inputs.update(unit_inputs(unit))
    build_path=dll.parent/'build-evidence.json'
    build_record=json.loads(build_path.read_text()) if build_path.exists() else {}
    build_sources={path:value for closure in build_record.get('unit_inputs',{}).values() for path,value in closure.items()}
    if not build_record and dll==ROOT/'Renderer/native/build/candidate/C3XRenderer.dll':
        build_path=ROOT/'Renderer/lab/.cache/native-build.json'
        if build_path.exists():
            build_record=json.loads(build_path.read_text())
            if build_record.get('dll_sha256')==digest(dll):build_sources=build_record.get('inputs',{})
    if (args.completion_probe or args.half_pixels) and (build_record.get('preview_only') or 'C3X_RENDERER_BENCHMARK_ORACLE' not in build_record.get('flags','')):
        parser.error('diagnostic controls require a verified benchmark-oracle DLL')
    binary_provenance={'build_receipt':build_path.relative_to(ROOT).as_posix() if build_record else None,
        'build_receipt_sha256':digest(build_path) if build_record else None,
        'current_runtime_matches_build':bool(build_sources) and all(build_sources.get(path)==value for path,value in inputs.items()),
        'purpose':'Candidate or explicitly selected historical binary; harness inputs are recorded separately.'}
    for path in (scene,dll,jgl,ROOT/'injected_code.c',ROOT/'C3X.h',ROOT/'civ_prog_objects.csv',*[ROOT/'Renderer/native'/n for n in ('gpu_frame_preview.h','input_recording_soak.h','world_readiness_preview.h','gpu_camera_identity_preview.h','native_frame_workload.h','native_frame_benchmark.h','test_native_screen.h','test_native_bootstrap.h','test_native_worker.cpp','test_gpu_unit_composition.h','test_native_image_adapter.cpp','test_native_ui_assets.h','native_ui_fixture.py','test_native_observation.cpp','test_native_line_bridge.h','native_image_adapter.h','native_sprite_diagnostics.h','native_composition_owner.h','native_observation.h','gpu_image_worker_client.h','gpu_image_commands.h','color_quantization.h','test_gpu_frame_api.c','biq_preview.cpp','BUILD.bat','record_gpu_frame.py')]):inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    if args.window_witness_seconds:
        for name in ('window_witness.cpp','BUILD_WINDOW_WITNESS.bat'):
            path=ROOT/'Renderer/tools'/name;inputs[path.relative_to(ROOT).as_posix()]=digest(path)
    # Runtime HLSL is part of the result identity even when the DLL is unchanged.
    from Renderer.lab.preparation import require_current, receipt as shader_receipt
    require_current(ROOT)
    shader_record=shader_receipt(ROOT)
    inputs.update(shader_record['inputs']);inputs.update(shader_record['outputs'])
    from Renderer.native.native_ui_fixture import prepare
    prepare(ROOT,out/"native-ui.pack",inputs)
    win=windows_root();target=win/out.relative_to(ROOT)
    settings={'C3X_RENDERER_MANUAL_VISUAL':'1','C3X_RENDERER_GPU_JGL_TEST':str(win/jgl.relative_to(ROOT)),'C3X_RENDERER_VISUAL_PROFILE':'city-fidelity','C3X_RENDERER_SHARED_SCENE_SURFACE':'',
        'C3X_RENDERER_WATER_COVERAGE':'','C3X_RENDERER_REFLECTION_CONTROL':'0' if args.reflections=='1' else '1','C3X_RENDERER_WAVES':args.waves,'C3X_RENDERER_WATER_MOTION':args.water_motion,'C3X_RENDERER_GPU_FRAME_TEST':'1','C3X_RENDERER_NATIVE_CAMERA_TEST':'1' if args.native_camera_requests or args.native_navigation else '', 'C3X_RENDERER_NATIVE_NAVIGATION_TEST':'1' if args.native_navigation else '', 'C3X_RENDERER_NATIVE_RECOVERY_TEST':'1' if args.native_recovery else '', 'C3X_RENDERER_GPU_CAMERA_IDENTITY_TEST':'1' if args.atomic_camera_views else '', 'C3X_RENDERER_GPU_CAMERA_TEST':'1' if args.camera_requests else '','C3X_RENDERER_SCROLL_COVERAGE_TEST':'1' if args.scroll_coverage else '','C3X_RENDERER_NATIVE_FRAME_BENCHMARK':'1' if args.benchmark else '',
        'C3X_RENDERER_OUTPUT_COMPLETION_PROBE':'1' if args.completion_probe else '0',
        'C3X_RENDERER_DIAGNOSTIC_HALF_PIXELS':'1' if args.half_pixels else '0',
        'C3X_RENDERER_PROFILE':'1' if args.profile else '0','C3X_RENDERER_GROUND_WORKERS':args.ground_workers,'C3X_RENDERER_OBJECT_WORKERS':args.object_workers,
        'C3X_RENDERER_TRACE':'2','C3X_RENDERER_TRACE_MIB':'32','C3X_RENDERER_TRACE_BUFFERED':'1' if args.benchmark or args.visual_only or args.world_readiness else '', 'C3X_RENDERER_TRACE_FILE':str(target/'renderer.log'),
        'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':r'..\..\Renderer\custom.custom_rendering.txt',
        'C3X_RENDERER_PREVIEW_OBJECTS':'1','C3X_RENDERER_PREVIEW_CITY':'0,3,1,1',
        'C3X_RENDERER_PREVIEW_DENSE_SCENE':'1' if args.dense_scene else '',
        'C3X_RENDERER_PREVIEW_DENSE_CITY_CASE':args.dense_city_case,
        'C3X_RENDERER_PREVIEW_VISIBILITY':'1' if args.visibility else '',
        'C3X_RENDERER_UNIT_SCENE_CONTROL':'1' if args.unit_scene=='0' else '',
        'C3X_RENDERER_BENCHMARK_UNITS':str(args.unit_count),
        'C3X_RENDERER_VISUAL_FRAMES':str(args.visual_frames),
        'C3X_RENDERER_VISUAL_UNITS':str(args.visual_units),'C3X_RENDERER_VISUAL_UNIT_CASE':args.visual_unit_case,
        'C3X_RENDERER_TACTICAL_PREVIEW':str(target/'tactical') if args.tactical else '',
        'C3X_RENDERER_PREVIEW_SESSION':'','C3X_RENDERER_PREVIEW_REPLAY':'','C3X_RENDERER_PREVIEW_ANIMATION':''}
    settings['C3X_RENDERER_NATIVE_UI_PACK']=str(target/'native-ui.pack')
    settings['C3X_RENDERER_TEST_RESERVE_MIB']=str(args.reserve_address_mib)
    settings['C3X_RENDERER_INPUT_SOAK_SECONDS']=str(args.input_soak_seconds) if args.input_soak_seconds else ''
    witness_event='Local\\C3XRendererWitness-'+invocation if args.window_witness_seconds else ''
    settings['C3X_RENDERER_INPUT_WINDOW_EVENT']=witness_event
    settings['C3X_RENDERER_INPUT_RECORD_DIR']=str(target/'inputs') if args.record_inputs else ''
    settings['C3X_RENDERER_RECORD_FILE']=str(target/'composition.c3xr') if args.record_composition else ''
    settings['C3X_RENDERER_WORLD_READINESS_ONLY']='1' if args.world_readiness_only else ''
    settings['C3X_RENDERER_WORLD_READINESS_TEST']='1' if args.world_readiness else ''
    settings['C3X_RENDERER_WORLD_GEOMETRY_MIB']=str(args.world_geometry_mib) if args.world_geometry_mib else ''
    settings['C3X_RENDERER_RIGID_SOURCES']=args.rigid_sources
    if args.benchmark or args.visual_only or args.scroll_coverage or args.world_readiness:
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
    witness_build=''
    if args.window_witness_seconds:
        # The fixture owns its HWND; the helper observes that exact process.
        # Start-Process preserves the PID without guessing a name or touching a
        # user's game. No helper timing is treated as renderer performance.
        def ps(value):return "'"+str(value).replace("'","''")+"'"
        executable,arguments=command.split(' ',1)
        witness_arguments=f' "{target/"window-witness"}" {args.window_witness_seconds} 5 sampled-window-evidence "{witness_event}"'
        script=("$ErrorActionPreference = 'Stop'\n"
            +f'$fixture = Start-Process -FilePath {ps(executable)} -ArgumentList {ps(arguments)} -PassThru -RedirectStandardOutput {ps(target/"test.log")} -RedirectStandardError {ps(target/"test.stderr.log")}\n'
            +'$null = $fixture.Handle\n'
            +'$capture = $null\ntry {\n'
            +f'  $arguments = [string]$fixture.Id + {ps(witness_arguments)}\n'
            +f'  $capture = Start-Process -FilePath {ps(target/"window_witness.exe")} -ArgumentList $arguments -PassThru -RedirectStandardOutput {ps(target/"witness.log")} -RedirectStandardError {ps(target/"witness.stderr.log")}\n'
            +'  $null = $capture.Handle\n'
            +'  $fixture.WaitForExit()\n'
            +f'  if (!$capture.WaitForExit({(args.window_witness_seconds+60)*1000})) {{ throw "Window witness did not terminate" }}\n'
            +'  $capture.WaitForExit()\n'
            +f'  @{{fixture = $fixture.ExitCode; observer = $capture.ExitCode}} | ConvertTo-Json | Set-Content -Encoding UTF8 {ps(target/"window-process-exits.json")}\n'
            +'  if ($null -eq $fixture.ExitCode -or $null -eq $capture.ExitCode -or $fixture.ExitCode -ne 0 -or $capture.ExitCode -ne 0) { exit 1 }\n'
            +'} finally {\n  if ($capture -and !$capture.HasExited) { $capture.Kill() }\n  if (!$fixture.HasExited) { $fixture.Kill() }\n}\nexit 0\n')
        (out/'window-fixture.ps1').write_text(script)
        witness_build=(f'call ..\\tools\\BUILD_WINDOW_WITNESS.bat >"{target/"witness-build.log"}" 2>&1\nif errorlevel 1 goto failed\n'
            +f'copy /y build\\window-witness\\window_witness.exe "{target/"window_witness.exe"}" >nul\nif errorlevel 1 goto failed\n')
        command=f'powershell -NoProfile -ExecutionPolicy Bypass -File "{target/"window-fixture.ps1"}"'
    invocation_log='witness-run.log' if args.window_witness_seconds else 'test.log'
    (out/'run.cmd').write_text('@echo off\nsetlocal\n'+f'pushd "{win/"Renderer/native"}"\n'
        +witness_build
        +f'call BUILD.bat gpu-frame >"{target/"build.log"}" 2>&1\nif errorlevel 1 goto failed\n'
        +''.join(f'set "{k}={v}"\n' for k,v in settings.items())
        +command+f' >"{target/invocation_log}" 2>&1\nif not "%errorlevel%"=="0" goto failed\n'
        +f'>"{target/"completion.txt"}" echo {invocation} 0\nexit /b 0\n:failed\n>"{target/"completion.txt"}" echo {invocation} 1\nexit /b 1\n')
    print(out.relative_to(ROOT),flush=True)
    process=subprocess.run(['prlctl','exec',os.environ.get('C3X_RENDERER_VM','Windows 11'),'--current-user','cmd','/d','/s','/c',f'call "{target/"run.cmd"}"'],capture_output=True,text=True,timeout=max(900 if args.benchmark else 480, (args.input_soak_seconds or 0)+480, (args.window_witness_seconds or 0)+480))
    complete=(out/'completion.txt').read_text().split() if (out/'completion.txt').exists() else []
    log=(out/'test.log').read_text(errors='replace') if (out/'test.log').exists() else ''
    unchanged=all(digest(ROOT/p)==h for p,h in inputs.items())
    passed=complete==[invocation,'0'] and unchanged and 'PASS resident map GPU worker:' in log and 'PASS native GPU worker transport:' in log and 'PASS native screen transfer:' in log and 'PASS live native screen:' in log and 'PASS production native map owner:' in log and ('PASS prepared GPU map adoption:' in log or (args.width>2224 and args.height>1244 and 'PASS bounded GPU map demand:' in log))
    if args.native_recovery:passed=passed and 'PASS native async recovery: cases=4 ' in log
    if args.native_navigation:passed=passed and 'PASS native navigation:' in log
    if args.native_camera_requests or args.native_navigation:passed=passed and 'PASS nonblocking native camera:' in log
    if args.camera_requests:passed=passed and 'PASS replaceable GPU camera:' in log
    if args.atomic_camera_views:passed=passed and 'PASS atomic GPU identity transitions: cases=16 ' in log
    receipt={'status':'pass' if passed else 'fail' if complete else 'unconfirmed','inputs':inputs,'inputs_unchanged':unchanged,'transport_returncode':process.returncode,'transport_output':process.stdout+process.stderr,'settings':settings,'scope':'production captured renderer map -> existing GPU worker -> packed composition; oracle readback explicit; actual native final presentation including CPU compatibility callback; no game speedup claim'}
    trace=(out/'renderer.log').read_text(errors='replace') if (out/'renderer.log').exists() else ''
    import re
    if args.native_camera_requests or args.native_navigation:
        receipt['native_camera_samples']=[{key:float(value) for key,value in re.findall(r'(\w+)=([0-9.]+)',line)}
            for line in log.splitlines() if line.startswith('NATIVE_CAMERA_SAMPLE ')]
        receipt['native_camera_scope']='Pending polls exclude render waits; ready polls include session import/admission. Completion includes test-driver scheduling. The native fixture services completion hints; the live bridge retries through the unchanged native Animator cadence. This is not live-game latency.'
    dropped=sum(int(value) for value in re.findall(r'TRACE_BUFFER dropped=(\d+)',trace))
    receipt['binary_provenance']=binary_provenance
    receipt['diagnostic_only']=bool(args.completion_probe or args.half_pixels or args.record_composition or args.record_inputs or args.window_witness_seconds)
    if args.record_composition:
        recording=out/'composition.c3xr'
        passed=passed and recording.is_file() and recording.stat().st_size>16
        receipt['composition_recording']={'path':'composition.c3xr','present':recording.is_file(),
            'scope':'native GPU composition; external map/pose snapshots; native ownership and visual observations',
            'performance_baseline':False}
    receipt['trace_coverage']={'dropped_lines':dropped,'complete':bool(trace) and dropped==0}
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
        receipt['scope']='fine-pan/zoom map coverage with cancelled requests and independent visual work; cold comparison only on missing-pixel failure; no native UI or performance claim'
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
        passed=passed and 'PASS independent resident frames:' in log and 'PASS blocked UI visual delivery:' in log
        receipt['status']='pass' if passed else 'fail' if complete else 'unconfirmed'
        receipt['visual_workload']={'units':args.visual_units,'case':args.visual_unit_case,'effects':{'waves':args.waves,'water_motion':args.water_motion,'reflections':args.reflections}}
        receipt['visual_frames']=[line for line in log.splitlines() if line.startswith(('VISUAL_SAMPLE ','PASS independent resident frames:','PASS blocked UI visual delivery:'))]
    if args.tactical:
        passed=passed and 'PASS tactical native composition:' in log
        receipt['tactical']={'capture':'actual native JGL line/text seams','previews':['tactical-route.bmp','tactical-grid.bmp'],'passed':passed}
        receipt['status']='pass' if passed else 'fail'
    if args.world_readiness:
        import re,statistics
        rows=[{key:float(value) for key,value in re.findall(r'(\w+)=([0-9.]+)',line)}
              for line in log.splitlines() if line.startswith('WORLD_JUMP ')]
        readiness=[{key:float(value) for key,value in re.findall(r'(\w+)=([0-9.]+)',line)}
                   for line in log.splitlines() if line.startswith('WORLD_READINESS ')]
        values=sorted(row['request_ms'] for row in rows)
        desktop=sorted(row['desktop_ms'] for row in rows)
        oracles=[line for line in log.splitlines() if line.startswith('WORLD_ORACLE ')]
        if args.world_readiness_only:passed=complete==[invocation,'0'] and unchanged and bool(trace) and dropped==0
        backing=[{key:int(value) for key,value in re.findall(r'(\w+)=(\d+)',line)}
                 for line in trace.splitlines() if 'stage=world-backing ' in line]
        for row in rows:
            operations=[event for event in backing if row.get('begin_qpc',0)<=event.get('qpc',-1)<=row.get('end_qpc',-2)]
            row['world_compiler_calls']=sum(event['compiler_calls'] for event in operations) if operations else None
            row['world_restore_calls']=sum(event['restore_calls'] for event in operations) if operations else None
        receipt['world_readiness']={'preparation':readiness,'samples':rows,
            'scope':'GPU request submission and request through actual desktop completion; CSV capture, native overlay composition and live trigger-to-display are excluded',
            'zero_world_compiler_calls':bool(rows) and all(row['world_compiler_calls']==0 for row in rows),
            'zero_geometry_adoption_or_upload':bool(rows) and all(row['built']==0 and row['uploads']==0 for row in rows),
            'latency':{'mean':statistics.mean(values),'p95':values[int(len(values)*.95)],'max':max(values)} if values else None,
            'desktop_latency':{'mean':statistics.mean(desktop),'p95':desktop[int(len(desktop)*.95)],'max':max(desktop)} if desktop else None,
            'minimum_largest_free':min((row['largest_free'] for row in rows),default=0),'cold_oracles':oracles,
            'over_100_ms':sum(value>100 for value in desktop),'distinct_destinations':len({(row['x'],row['y']) for row in rows})}
        passed=passed and len(rows)==100 and len(oracles)==6 and 'PASS world readiness workload:' in log
        receipt['status']='pass' if passed else 'fail'
    if args.input_soak_seconds:
        soak_samples=[{key:float(value) for key,value in re.findall(r'(\w+)=([0-9.]+)', line)}
                      for line in log.splitlines() if line.startswith('INPUT_SOAK_SAMPLE ')]
        soak_memory=[{key:int(value) for key,value in re.findall(r'(\w+)=(\d+)', line)}
                     for line in log.splitlines() if line.startswith('INPUT_SOAK_MEMORY ')]
        passed=passed and 'PASS input native soak:' in log and bool(soak_samples) and bool(soak_memory)
        receipt['input_soak']={'seconds':args.input_soak_seconds,'samples':soak_samples,'memory':soak_memory,
                               'scope':'Native fixture wall-clock endurance; manual ambient offers; not gameplay or scanout acceptance'}
    if args.record_inputs:
        finished = out / 'inputs/finished.json'
        storage = json.loads(finished.read_text()) if finished.is_file() else None
        closed = bool(storage and storage.get('stop_reason') in (0, 1))
        receipt['input_recording'] = {'path': 'inputs', 'storage': storage, 'closed': closed,
                                      'qualified_for_gameplay': False, 'replayed': False,
                                      'scope': 'Development production renderer inputs; replay validation is separate'}
        passed = passed and closed
    if args.window_witness_seconds:
        from Renderer.tools.inspect_window_witness import inspect
        try:
            witness,_,_=inspect(out/'window-witness')
            exits=json.loads((out/'window-process-exits.json').read_text(encoding='utf-8-sig'))
            receipt['window_witness']={'inspection':witness,'helper_sha256':digest(out/'window_witness.exe'),
                'process_exits':exits,
                'scope':'Sampled external production test window; recording overhead not qualified'}
            passed=passed and witness['complete'] and exits=={'fixture':0,'observer':0}
        except Exception as error:
            receipt['window_witness']={'error':str(error)};passed=False
    receipt['status'] = 'pass' if passed else 'fail' if complete else 'unconfirmed'
    if passed:
        shutil.copy2(ROOT/'Renderer/native/build/gpu-composition/test_gpu_frame.exe',out/'test_gpu_frame.exe')
        shutil.copy2(dll,out/'C3XRenderer.dll')
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if (out/'build.log').exists():print((out/'build.log').read_text(errors='replace')[-3000:])
    print(log[-4500:]);print(receipt['status'])
    if not complete:print('Native dispatch has no completion receipt; check the VM before another run. '+process.stdout+process.stderr)
    return 0 if passed else 1


if __name__=='__main__':raise SystemExit(main())
