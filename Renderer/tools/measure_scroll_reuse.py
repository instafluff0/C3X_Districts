"""Prepare an isolated two-state static-raster pilot; VM dispatch requires an explicit slot.

Example after copying this tool into the worktree (all commands are host-only):
  python3 Renderer/tools/measure_scroll_reuse.py prepare --asset-root "$C3X_PRIMARY" \
    --binaries-root Renderer/.cache/static-raster-step --label candidate125 \
    --arm candidate --zoom 1.25 --trace static --diagnostic
  python3 Renderer/tools/measure_scroll_reuse.py analyze --label candidate125

The primary checkout supplies ignored packs, developed scene and accepted shader
inputs read-only. It is never an output destination. Timing includes preparation,
adoption, actual draw and Present; diagnostic/quality runs do not qualify FPS.
"""
from pathlib import Path, PureWindowsPath
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
PASS_NAMES = ('selection', 'shadow', 'main_scene', 'reflection_scene', 'water',
              'main_material', 'reflection_material', 'relight', 'units',
              'reflected_units', 'unit_shadow', 'reconstruction', 'publication')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def windows_path(path):
    path = path.resolve()
    if os.name == 'nt':
        return str(path)
    try:
        return str(PureWindowsPath(r'\\Mac\Home') / path.relative_to(Path.home()).as_posix())
    except ValueError as error:
        raise ValueError('Inputs must be shared through Mac Home, or prepare on Windows') from error


def batch_value(value):
    value = str(value)
    if any(c in value for c in '"%\r\n'):
        raise ValueError('Unsupported batch argument')
    return value


def distribution(values):
    values = sorted(values)
    if not values:
        return {}
    def q(f):
        return values[min(len(values)-1, int((len(values)-1)*f))]
    result = dict(samples=len(values), mean_ms=sum(values)/len(values), p50_ms=q(.5),
                  worst_ms=max(values), over_16_67ms=sum(v > 1000/60 for v in values))
    # Small pilots do not support defensible p95/p99 tail claims.
    if len(values) >= 20:
        result['p95_ms'] = q(.95)
    if len(values) >= 100:
        result['p99_ms'] = q(.99)
    return result


def output_root(args):
    path = (ROOT / args.out).resolve()
    if ROOT not in path.parents or not path.relative_to(ROOT).as_posix().startswith('Renderer/'):
        raise ValueError('Outputs must stay under this worktree Renderer directory')
    return path



def source_manifest(path):
    value = json.loads(path.read_text())
    files = value.get('sources', value)
    if not files or not isinstance(files, dict):
        raise ValueError('Expected an explicit source SHA256 manifest: ' + str(path))
    for name, digest in files.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or not name.startswith('Renderer/'):
            raise ValueError('Source manifest paths must be repository-relative Renderer files')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Expected a SHA256 source identity: ' + name)
    return value, files


def reviewed_revision(revision):
    result = subprocess.run(['git', 'rev-parse', '--verify', revision+'^{commit}'],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def freeze_build_sources(directory, arm, source_root, manifest_path, baseline, overrides):
    manifest, files = source_manifest(manifest_path)
    frozen = directory/'sources'/arm
    frozen_manifest = directory/(arm+'-build-source-manifest.json')
    shutil.copy2(manifest_path, frozen_manifest)
    differences = {}
    for name, digest in files.items():
        source = source_root/name
        if not source.is_file() or sha(source) != digest:
            raise ValueError('Frozen '+arm+' build source does not match manifest: '+name)
        if arm == 'control':
            result = subprocess.run(['git', 'show', baseline+':'+name], cwd=ROOT,
                                    capture_output=True)
            reviewed = hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None
            if reviewed != digest:
                override = overrides.get(name, {})
                if override.get('sha256') != digest or not override.get('reason'):
                    raise ValueError('Control differs from reviewed baseline without an explicit diagnostic override: '+name)
                differences[name] = {'reviewed_sha256': reviewed, 'build_sha256': digest,
                                     'reason': override['reason']}
        target = frozen/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    unexpected = set(overrides)-set(differences)
    if arm == 'control' and unexpected:
        raise ValueError('Unused control overrides: '+', '.join(sorted(unexpected)))
    return {'sources': files, 'source_count': len(files), 'manifest_sha256': sha(manifest_path),
            'manifest': manifest, 'reviewed_source_differences': differences}


def freeze_harness(directory):
    # Capture the current shared client and all quoted include dependencies.
    pending = ['Renderer/sandbox/client_x64.cpp', 'Renderer/sandbox/scroll_witness.h',
               'Renderer/tools/measure_scroll_reuse.py']
    files = {}
    while pending:
        name = pending.pop()
        if name in files:
            continue
        original = ROOT/name
        if not original.is_file():
            raise ValueError('Missing shared pilot source: '+name)
        target = directory/'sources'/'harness'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        files[name] = sha(original)
        if original.suffix in ('.cpp', '.c', '.h'):
            for quoted in re.findall(r'#\s*include\s*"([^"]+)"', original.read_text(errors='replace')):
                for choice in (original.parent/quoted, ROOT/'Renderer/native'/quoted, ROOT/quoted):
                    choice = choice.resolve()
                    if choice.is_file() and ROOT in choice.parents:
                        pending.append(choice.relative_to(ROOT).as_posix())
                        break
    return files


def prepare(args):
    out = output_root(args)
    directory = out / args.label
    if directory.exists():
        raise ValueError('Preserve existing invocation: ' + args.label)
    asset = args.asset_root.resolve()
    if out == asset or asset in out.parents:
        raise ValueError('The asset checkout is read-only; select an isolated worktree output')
    if args.execute_vm and not args.vm_slot:
        raise ValueError('--execute-vm requires --vm-slot with the explicit auditor grant identifier')
    baseline = reviewed_revision(args.baseline_revision)
    binaries = (ROOT / args.binaries_root).resolve()
    candidate_manifest = (ROOT/args.candidate_manifest).resolve() if args.candidate_manifest else binaries/'candidate-source-manifest.json'
    control_manifest = (ROOT/args.control_manifest).resolve() if args.control_manifest else binaries/'control-source-manifest.json'
    candidate_sources = (ROOT/args.candidate_sources).resolve() if args.candidate_sources else binaries/'candidate-sources'
    control_sources = (ROOT/args.control_sources).resolve() if args.control_sources else binaries/'control-sources'
    overrides = json.loads(args.control_overrides.read_text()) if args.control_overrides else {}
    directory.mkdir(parents=True)
    builds = {}
    for arm, source_root, manifest_path in (('control', control_sources, control_manifest),
                                           ('candidate', candidate_sources, candidate_manifest)):
        builds[arm] = freeze_build_sources(directory, arm, source_root, manifest_path, baseline, overrides)
    identities = {'reviewed_baseline_revision': baseline,
                  'integrated_baseline_revision': args.integrated_baseline_revision,
                  'builds': builds, 'harness': freeze_harness(directory), 'binaries': {},
                  'binary_build_sources': builds[args.arm]['manifest'],
                  'reason_counters_available': True,
                  'static_state_metrics_export': 'optional_observed_in_run'}
    # Preserve both compared DLLs beside the exact source snapshots, even when
    # this invocation runs only one arm. The common client is copied below.
    identities['compared_dlls'] = {}
    for arm in ('control', 'candidate'):
        path = binaries/arm/'C3XRenderer_x64.dll'
        if not path.is_file():
            raise ValueError('Missing compared '+arm+' DLL: '+str(path))
        digest = sha(path)
        declared = builds[arm]['manifest'].get('binaries', {}).get('dll')
        if isinstance(declared, dict):
            declared = declared.get('sha256')
        if declared and declared != digest:
            raise ValueError('Compared DLL does not match its build receipt: '+arm)
        destination = directory/'binaries'/arm/path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        identities['compared_dlls'][arm] = {'filename': path.name, 'sha256': digest,
                                          'build_receipt_hash_confirmed': bool(declared)}
    paths = {'dll': binaries/args.arm/'C3XRenderer_x64.dll',
             'client': binaries/'client'/'client_x64.exe'}
    for kind, path in paths.items():
        if not path.is_file():
            raise ValueError('Build the isolated pilot first; missing '+str(path))
        digest = sha(path)
        declared = builds[args.arm]['manifest'].get('binaries', {}).get(kind)
        if isinstance(declared, dict):
            declared = declared.get('sha256')
        if declared and declared != digest:
            raise ValueError('Binary does not match its build receipt: '+kind)
        identities['binaries'][kind] = {'filename': path.name, 'sha256': digest,
                                       'build_receipt_hash_confirmed': bool(declared)}
        destination = directory/path.name
        shutil.copy2(path, destination)
        paths[kind] = destination
    scene = asset / 'Renderer/native/build/performance-review-current/developed-scene.csv'
    shader_root = asset / 'Renderer/.cache/redraw-underlay-step/accepted/shaders'
    if not scene.is_file() or not shader_root.is_dir():
        raise ValueError('Missing read-only developed scene or accepted shader inputs')
    shutil.copy2(scene, directory / 'developed-scene.csv')
    shutil.copytree(shader_root, directory / 'shaders')
    identities['scene_sha256'] = sha(scene)
    identities['shaders'] = {p.relative_to(directory/'shaders').as_posix(): sha(p)
                            for p in (directory/'shaders').rglob('*') if p.is_file()}
    seed = json.loads((asset/'Renderer/.cache/city-light-index-step/timing-night-3-candidate/receipt.json').read_text())['options']
    options = {k: str(v) for k, v in seed.items()}
    options.update(C3X_RENDERER_SHADER_SOURCE_ROOT=windows_path(directory/'shaders'),
        C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS=windows_path(asset/'Renderer/custom.custom_rendering.txt'),
        C3X_RENDERER_TRACE='0', C3X_RENDERER_PROFILE='0',
        C3X_RENDERER_CITY_LIGHT_DIAGNOSTICS='0', C3X_SANDBOX_DISPLAY='full_guest',
        C3X_SANDBOX_CAMERA_WITNESS='1', C3X_SANDBOX_SCROLL_WITNESS='1',
        C3X_SANDBOX_SCROLL_ZOOM=str(args.zoom), C3X_SANDBOX_SCROLL_TRACE=args.trace,
        C3X_SANDBOX_SCROLL_FORCE_FULL='1' if args.force_full else '0',
        C3X_SANDBOX_SCROLL_QUALITY='1' if args.quality else '0',
        C3X_SANDBOX_SCROLL_SCRIPTED_TIME='1' if args.scripted_time or args.quality else '0',
        C3X_SANDBOX_SCROLL_STEPS=str(args.steps),
        C3X_SANDBOX_SCROLL_OUTPUT=windows_path(directory),
        C3X_SANDBOX_SCROLL_CALLS='1' if args.diagnostic else '0',
        C3X_SANDBOX_PASS_COUNTS='1' if args.diagnostic else '0',
        C3X_SANDBOX_FRAME_TIMINGS='0', C3X_SANDBOX_GPU_TIMESTAMPS='0',
        C3X_SANDBOX_CAPTURE='', C3X_SANDBOX_CAPTURE_SEQUENCE='',
        C3X_SANDBOX_WHOLE_WORLD='1', C3X_SANDBOX_UNITS='1',
        C3X_RENDERER_PREVIEW_UNITS='1', C3X_RENDERER_WATER_MOTION='1', C3X_RENDERER_WAVES='1')
    run_id = uuid.uuid4().hex
    arguments = [windows_path(paths['client']), windows_path(paths['dll']), windows_path(asset),
        windows_path(asset/'Renderer/default.custom_rendering.txt'), windows_path(directory/'developed-scene.csv'),
        windows_path(directory/'initial.bmp')]
    command = ' '.join('"'+batch_value(p)+'"' for p in arguments) + f' 2240 1260 24 56 128 {args.hour}'
    # The exact child PID/exit receipt is collected by PowerShell. Timeout never
    # authorizes a second run or broad image-name cleanup.
    run_batch = directory/'run.bat'
    run_batch.write_text('@echo off\nsetlocal\n' + '\n'.join(f'set "{batch_value(k)}={batch_value(v)}"' for k, v in options.items())
        + '\n' + command + '\nexit /b %errorlevel%\n')
    launcher = directory/'launch.ps1'
    launcher.write_text('param()\n$ErrorActionPreference="Stop"\n'
        + 'Add-Type -TypeDefinition \'using System.Runtime.InteropServices; public class ScrollPilotAwake { [DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags); }\'\n'
        + 'if([ScrollPilotAwake]::SetThreadExecutionState([uint32]2147483651) -eq 0){throw "Cannot establish bounded verification wake request"}\ntry {\n'
        + '$root=$PSScriptRoot\n'
        + '$process=Start-Process -FilePath $env:ComSpec -ArgumentList @("/d","/c",("`""+(Join-Path $root "run.bat")+"`"")) -PassThru -RedirectStandardOutput (Join-Path $root "run.log") -RedirectStandardError (Join-Path $root "stderr.log")\n'
        + f'"{run_id} $($process.Id)" | Set-Content (Join-Path $root "process.txt")\n'
        + '$processHandle=$process.Handle\n'
        + '$process.WaitForExit(); $process.Refresh(); $exitCode=$process.ExitCode\n'
        + 'if($null -eq $exitCode){throw "Child exit code is unconfirmed; inspect this invocation PID"}\n'
        + f'"{run_id} $exitCode" | Set-Content (Join-Path $root "completion.txt")\n'
        + '} finally {[void][ScrollPilotAwake]::SetThreadExecutionState([uint32]2147483648)}\n'
        + 'exit $exitCode\n')
    plan = {'run_id': run_id, 'arm': args.arm, 'hour': args.hour, 'zoom': args.zoom,
            'diagnostic': args.diagnostic, 'quality': args.quality, 'force_full': args.force_full,
            'trace': args.trace, 'options': options,
            'native_input': {'slow_arrow_step': [32, 16], 'edge_formula': '2*(step/(edge_coordinate+1))',
                             'edge15_step': [4, 2], 'edge7_step': [8, 4], 'native_edge_timer_ms': 66,
                             'motion_schedule_ms': 17, 'motion_schedule_scope': 'diagnostic paced stress'}, 'source_and_binary_identities': identities,
            'vm_command': 'powershell -NoProfile -ExecutionPolicy Bypass -File "'+windows_path(launcher)+'"',
            'scope': 'standalone copied-native pilot; not installed-game or busy-unit qualification'}
    (directory/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print('Prepared:', directory.relative_to(ROOT))
    print(plan['vm_command'])
    if args.execute_vm:
        if not args.vm_slot:
            raise ValueError('--execute-vm requires --vm-slot with the explicit auditor grant identifier')
        sys.path.insert(0, str(ROOT))
        from Renderer.lab.platform import native_command_result
        result = native_command_result('', plan['vm_command'], timeout_seconds=600)
        (directory/'transport.json').write_text(json.dumps(result, indent=2)+'\n')
        if result['status'] != 'pass':
            raise RuntimeError('Inspect exact process.txt PID before another run; VM slot remains held')
        analyze(args)


def analyze(args):
    directory = output_root(args)/args.label
    plan = json.loads((directory/'plan.json').read_text())
    completion = (directory/'completion.txt').read_text().strip().split()
    if completion != [plan['run_id'], '0']:
        raise ValueError('No matching successful child completion; inspect the exact PID')
    text = (directory/'run.log').read_text(errors='replace')
    if 'SCROLL_WITNESS pass' not in text:
        raise ValueError('Missing executed scroll witness')
    groups = {}
    for line in text.splitlines():
        if not line.startswith(('SCROLL_', 'CLIENT_PASS ', 'CLIENT_DISPLAY ', 'SANDBOX_SWAPCHAIN ')):
            continue
        tag, _, rest = line.partition(' ')
        pairs = re.findall(r'(\w+)=(\S+)', rest)
        row = dict(pairs)
        # Earlier diagnostic records used projection for both zoom and reason.
        projection_values = [value for key, value in pairs if key == 'projection']
        if tag == 'SCROLL_STATIC' and len(projection_values) == 2:
            row['projection_reason'] = projection_values[1]
            row['projection'] = projection_values[0]
        for key, value in row.items():
            try:
                row[key] = float(value) if '.' in value else int(value)
            except ValueError:
                pass
        groups.setdefault(tag, []).append(row)
    frames = groups.get('SCROLL_FRAME', [])
    summary = {'analysis_tool_sha256': sha(Path(__file__)), 'scope': plan['scope'], 'diagnostic': plan['diagnostic'], 'quality': plan['quality'],
               'records': groups, 'source_and_binary_identities': plan['source_and_binary_identities'],
               'reason_counters_available': plan['source_and_binary_identities'].get('reason_counters_available', plan['arm'] != 'control'),
               'static_state_counters_available': any(row.get('available') == 1 for row in groups.get('SCROLL_STATIC', [])),
               'quality_oracle_invalidates_retained_states': bool(plan['quality']),
               'reasons_are_overlapping_not_redraw_counts': True,
               'latency': {name: distribution([row[name] for row in frames])
                           for name in ('first_correct_ms', 'capture_ms', 'submission_ms', 'preparation_ms', 'adoption_ms', 'draw_ms', 'present_ms')},
               'changing_view_latency': {name: distribution([row[name] for row in frames[1:]])
                           for name in ('first_correct_ms', 'capture_ms', 'submission_ms', 'preparation_ms', 'adoption_ms', 'draw_ms', 'present_ms')},
               'intervals': distribution([row['interval_ms'] for row in frames[1:]]),
               'cold_destination': frames[:1], 'changing_views': frames[1:]}
    state_rows = [row for row in groups.get('SCROLL_STATIC', []) if row.get('available') == 1]
    summary['static_state_stage_totals'] = {}
    for row in state_rows:
        key = str(row['state'])+'/'+str(row['stage'])
        totals = summary['static_state_stage_totals'].setdefault(key, {})
        for counter in ('full_draws', 'restores', 'reuses', 'strip_fills'):
            totals[counter] = totals.get(counter, 0)+row[counter]
    summary['observed_memory_peaks_bytes'] = {name: max((row[name] for row in state_rows), default=None)
        for name in ('total_region_bytes', 'shared_viewport_bytes', 'total_gpu_bytes')}
    summary['memory_metrics_available'] = bool(state_rows)
    summary['observed_capture_unit_records'] = [dict(frame=row['frame'], view=row['view'], unit_records=row.get('unit_records', 0))
                                              for row in frames]
    summary['animated_proof_scope'] = 'Use existing focused live-unit/effects/composition regression; this pilot adds no actors'
    if plan['quality']:
        import numpy as np
        def pixels(path):
            raw = Path(path).read_bytes()
            offset = struct.unpack_from('<I', raw, 10)[0]
            width, height, planes, bits, compression = struct.unpack_from('<iiHHI', raw, 18)
            if raw[:2] != b'BM' or planes != 1 or bits != 32 or compression != 0 or width <= 0 or height == 0:
                raise ValueError('Expected the witness uncompressed32-bit BGRA BMP')
            data = np.frombuffer(raw, dtype=np.uint8, offset=offset)
            if data.size != width*abs(height)*4:
                raise ValueError('BMP extent mismatch')
            data = data.reshape(abs(height), width, 4)
            return data[::-1] if height > 0 else data
        comparisons = []
        for row in groups.get('SCROLL_ORACLE', []):
            prefix = directory/f"frame_{row['frame']:03d}"
            a = pixels(str(prefix)+'_retained.bmp').astype(np.int16)
            b = pixels(str(prefix)+'_full.bmp').astype(np.int16)
            difference = np.abs(a-b)
            da = np.fromfile(str(prefix)+'_retained.depth', dtype='<u4')
            db = np.fromfile(str(prefix)+'_full.depth', dtype='<u4')
            if not np.array_equal(da[:3], db[:3]) or da.size != db.size:
                raise ValueError('Depth format or extent mismatch')
            depth_difference = np.abs((da[3:]&0xffffff).astype(np.int64)-(db[3:]&0xffffff).astype(np.int64))
            comparisons.append(dict(frame=row['frame'], view=row['view'], color_changed_pixels=int(np.any(difference!=0, axis=2).sum()),
                color_max=int(difference.max()), color_mean=float(difference.mean()), alpha_changed_pixels=int((difference[:, :, 3]!=0).sum()), depth_changed_pixels=int((depth_difference!=0).sum()),
                depth_max=int(depth_difference.max()), depth_pixels_above_1=int((depth_difference>1).sum()),
                color_pixels_above_1=int(np.any(difference[:, :, :3]>1, axis=2).sum()),
                color_pixels_above_3=int(np.any(difference[:, :, :3]>3, axis=2).sum()),
                color_pixels_above_10=int(np.any(difference[:, :, :3]>10, axis=2).sum()),
                stencil_changed_pixels=int(((da[3:]>>24)!=(db[3:]>>24)).sum())))
        summary['independent_full_redraw_comparisons'] = comparisons
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k in ('latency', 'intervals', 'independent_full_redraw_comparisons')}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'analyze'))
    p.add_argument('--asset-root', type=Path)
    p.add_argument('--binaries-root', type=Path, default=Path('Renderer/.cache/static-raster-step'))
    p.add_argument('--baseline-revision', default='4d820048', help='Reviewed control commit; resolved in this worktree')
    p.add_argument('--integrated-baseline-revision', default='dbf65855', help='Auditor-recorded primary integration provenance')
    p.add_argument('--control-sources', type=Path, help='Exact frozen reviewed control build source directory')
    p.add_argument('--candidate-sources', type=Path, help='Exact frozen candidate build source directory')
    p.add_argument('--control-manifest', type=Path, help='Explicit control source SHA256 build manifest')
    p.add_argument('--candidate-manifest', type=Path, help='Explicit candidate source SHA256 build manifest')
    p.add_argument('--control-overrides', type=Path, help='JSON file mapping diagnostic-only control differences to sha256 and reason')
    p.add_argument('--out', default='Renderer/.cache/static-raster-step/runs')
    p.add_argument('--label', required=True)
    p.add_argument('--arm', choices=('candidate', 'control'), default='candidate')
    p.add_argument('--zoom', type=float, default=1.25)
    p.add_argument('--hour', type=int, default=12)
    p.add_argument('--trace', choices=('pilot', 'phase', 'static', 'motion', 'regression'), default='pilot')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--quality', action='store_true')
    p.add_argument('--force-full', action='store_true')
    p.add_argument('--scripted-time', action='store_true')
    p.add_argument('--execute-vm', action='store_true')
    p.add_argument('--vm-slot', help='Explicit auditor VM reservation grant identifier')
    args = p.parse_args()
    if args.action == 'prepare' and args.asset_root is None:
        p.error('prepare requires the read-only --asset-root checkout')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.label):
        p.error('label must contain only letters, numbers, underscore or hyphen')
    if args.action == 'prepare':
        prepare(args)
    else:
        analyze(args)


if __name__ == '__main__':
    main()
