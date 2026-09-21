"""Short production input replay and missing-input controls (not gameplay qualification)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from Renderer.lab.platform import ROOT, windows_root
from Renderer.native.record_renderer_build import DLL_UNITS, unit_inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_call(out, name, command, timeout=300):
    win = windows_root()
    target = win / out.relative_to(ROOT)
    nonce = uuid.uuid4().hex
    batch = out / (name + '.cmd')
    claim = name + '-' + nonce + '-lock'
    batch.write_text('@echo off\nsetlocal\n' + f'mkdir "{target / claim}" >nul 2>&1\nif errorlevel 1 exit /b 0\n' + f'pushd "{win / "Renderer/native"}"\n'
                     + f'call :work >"{target / (name + ".log")}" 2>&1\n'
                     + f'>"{target / (name + ".done")}" echo {nonce} %errorlevel%\nexit /b 0\n:work\n'
                     + command + '\nexit /b %errorlevel%\n')
    done = out / (name + '.done')
    result = None
    for attempt in range(3):
        # The atomic directory claim makes an uncertain transport retry safe:
        # only one Windows child can execute this invocation's work.
        result = subprocess.run(['prlctl', 'exec', os.environ.get('C3X_RENDERER_VM', 'Windows 11'),
                                 '--current-user', 'cmd', '/d', '/s', '/c', f'call "{target / batch.name}"'],
                                capture_output=True, text=True, timeout=timeout)
        if done.exists():
            break
        if (out / claim).exists():
            deadline = time.monotonic() + timeout
            while not done.exists() and time.monotonic() < deadline:
                time.sleep(.5)
            break
        time.sleep(1)
    if not done.exists():
        raise RuntimeError(f'Unconfirmed native invocation: {batch.relative_to(ROOT)}; transport={result.returncode}')
    identity, code = done.read_text().split()
    if identity != nonce:
        raise RuntimeError('Unmatched native completion receipt')
    log = (out / (name + '.log')).read_text(errors='replace')
    print(name, code, log[-2500:], flush=True)
    return {'returncode': int(code), 'log': name + '.log', 'transport_returncode': result.returncode}


def check_time_and_prefix(out, cases):
    """Exercise time seeking and a killed writer's final partial record."""
    import shutil
    target = windows_root() / out.relative_to(ROOT)
    entries = [json.loads(line) for line in (out / 'control-1.timeline.jsonl').read_text().splitlines()]
    start = entries[1]['seconds']
    end = entries[2]['seconds']
    # Timeline decimal rounding must not select the neighboring boundary.
    start = (entries[0]['seconds'] + start) / 2
    end = (entries[1]['seconds'] + end) / 2
    cases['seconds'] = native_call(out, 'seconds',
        f'build\\input-recording\\replay_inputs.exe --development "{target / "C3XRenderer.dll"}" "{target / "capture"}" --seconds "{target / "seconds"}" {start:.9f} {end:.9f}')
    selected = list((out / 'seconds').glob('*.bmp'))
    if cases['seconds']['returncode'] or selected != [out / 'seconds/frame-000002.bmp'] or digest(selected[0]) != digest(out / 'control-1/frame-000002.bmp'):
        raise RuntimeError('Selected-time replay differs from complete replay')
    prefix = out / 'torn-prefix'
    shutil.copytree(out / 'capture', prefix)
    final = sorted(prefix.glob('segment-*.c3xi'))[-1]
    with final.open('r+b') as stream:
        stream.truncate(final.stat().st_size - 9)
    for name in ('finished.json', 'index.jsonl'):
        (prefix / name).unlink(missing_ok=True)
    command = f'build\\input-recording\\replay_inputs.exe --development "{target / "C3XRenderer.dll"}" "{target / "torn-prefix"}"'
    cases['prefix-rejected'] = native_call(out, 'prefix-rejected', command)
    if not cases['prefix-rejected']['returncode']:
        raise RuntimeError('Truncated capture accepted as complete')
    cases['prefix-allowed'] = native_call(out, 'prefix-allowed', command + ' --allow-prefix')
    reports = [json.loads(line) for line in (out / 'prefix-allowed.log').read_text().splitlines() if line.startswith('{')]
    if cases['prefix-allowed']['returncode'] or len(reports) != 1 or reports[0]['complete'] or reports[0]['qualified'] or reports[0]['accepted_presentations'] != len(entries):
        raise RuntimeError('Verified prefix was lost or incorrectly certified complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3, choices=range(3, 1201))
    parser.add_argument('--width', type=int, default=640, choices=range(320, 2241))
    parser.add_argument('--height', type=int, default=480, choices=range(240, 1261))
    parser.add_argument('--reuse-candidate', action='store_true', help='Reuse only a candidate whose source closure and build receipt match exactly')
    args = parser.parse_args()
    scene = args.scene.resolve()
    scene.relative_to(ROOT)
    if any(char in str(scene) for char in '\r\n"%&|<>^!'):
        parser.error('unsupported scene path')
    out = ROOT / 'Renderer/native/build/input-recording' / ('contract-' + uuid.uuid4().hex)
    out.mkdir(parents=True)
    print(out.relative_to(ROOT), flush=True)
    win = windows_root()
    target = win / out.relative_to(ROOT)
    cases = {}
    closures = {unit: unit_inputs(unit) for unit in DLL_UNITS}
    recipe = digest(ROOT / 'Renderer/native/BUILD.bat')
    if args.reuse_candidate:
        candidate = ROOT / 'Renderer/native/build/candidate'
        recorded = json.loads((candidate / 'build-evidence.json').read_text())
        if (recorded.get('unit_inputs') != closures or recorded.get('build_recipe') != recipe
                or recorded.get('dll_sha256') != digest(candidate / 'C3XRenderer.dll')
                or recorded.get('returncode') != 0 or not recorded.get('sources_unchanged')):
            raise RuntimeError('Candidate is not verified against current source inputs')
    for mode in (('input-recording', 'input-replay', 'gpu-frame') if args.reuse_candidate else ('candidate-compile', 'input-recording', 'input-replay', 'gpu-frame')):
        cases[mode] = native_call(out, 'build-' + mode, 'call BUILD.bat ' + mode)
        if cases[mode]['returncode']:
            raise RuntimeError('Native build failed')
    cases['storage'] = native_call(out, 'storage', f'build\\input-recording\\test.exe "{target / "storage"}"')
    if cases['storage']['returncode']:
        raise RuntimeError('Native storage contract failed')
    cases['canvas'] = native_call(out, 'canvas', f'build\\input-recording\\test_input_canvas.exe "{target / "canvas"}"')
    if cases['canvas']['returncode']:
        raise RuntimeError('Native CPU canvas capture contract failed')
    # Freeze the candidate used by both arms. Rebuilding another candidate cannot
    # invalidate this local control or silently change its replay identity.
    import shutil
    dll = ROOT / 'Renderer/native/build/candidate/C3XRenderer.dll'
    shutil.copy2(dll, out / 'C3XRenderer.dll')
    identity = digest(out / 'C3XRenderer.dll')
    if closures != {unit: unit_inputs(unit) for unit in DLL_UNITS} or recipe != digest(ROOT / 'Renderer/native/BUILD.bat'):
        raise RuntimeError('Recorder sources changed during build')
    build_record = {'unit_inputs': closures, 'dll_sha256': identity, 'build_recipe': recipe,
                    'sources_unchanged': True, 'command': 'BUILD.bat candidate-compile', 'returncode': 0}
    (out / 'build-evidence.json').write_text(json.dumps(build_record, indent=2) + '\n')
    (dll.parent / 'build-evidence.json').write_text(json.dumps(build_record, indent=2) + '\n')
    settings = {'C3X_RENDERER_INPUT_RECORD_DIR': str(target / 'capture'),
                'C3X_RENDERER_INPUT_CONTRACT': str(args.steps), 'C3X_RENDERER_MANUAL_VISUAL': '1',
                'C3X_RENDERER_GPU_FRAME_TEST': '1', 'C3X_RENDERER_VISUAL_PROFILE': 'city-fidelity',
                'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS': r'..\..\Renderer\custom.custom_rendering.txt',
                'C3X_RENDERER_PREVIEW_OBJECTS': '1', 'C3X_RENDERER_WAVES': '1', 'C3X_RENDERER_WATER_MOTION': '1',
                'C3X_RENDERER_REFLECTION_CONTROL': '0', 'C3X_RENDERER_TRACE': '0', 'C3X_RENDERER_RECORD_FILE': ''}
    header = scene.read_text().splitlines()[0].split(',')
    capture = ''.join(f'set "{key}={value}"\n' for key, value in settings.items())
    capture += (f'build\\gpu-composition\\test_gpu_frame.exe "{target / "C3XRenderer.dll"}" ..\\.. '
                f'..\\..\\Renderer\\default.custom_rendering.txt "{win / scene.relative_to(ROOT)}" '
                f'"{target / "control.bmp"}" {args.width} {args.height} {int(header[1]) // 2} {int(header[2]) // 2} 128 12')
    cases['capture'] = native_call(out, 'capture', capture)
    if cases['capture']['returncode']:
        raise RuntimeError('Production input capture failed')
    cases['inspect'] = native_call(out, 'inspect',
        f'build\\input-recording\\inspect_inputs.exe "{target / "capture"}" "{target / "inspection"}"')
    if cases['inspect']['returncode']:
        raise RuntimeError('Input inspection failed')
    inspection = json.loads((out / 'inspection/report.json').read_text())
    frame_count = inspection['accepted_presentations']
    if frame_count < args.steps:
        raise RuntimeError('Input timeline lost accepted presentations')
    for name in ('control-1', 'control-2'):
        command = (f'build\\input-recording\\replay_inputs.exe --development "{target / "C3XRenderer.dll"}" '
                   f'"{target / "capture"}" --frames "{target / name}" {frame_count} --timeline "{target / (name + ".timeline.jsonl")}"'
                   f' --fingerprints "{target / (name + ".frames.jsonl")}"')
        cases[name] = native_call(out, name, command)
        if cases[name]['returncode']:
            raise RuntimeError('Production input replay failed')
    frames = sorted((out / 'control-1').glob('*.bmp'))
    if len(frames) != frame_count or any(digest(path) != digest(out / 'control-2' / path.name) for path in frames):
        raise RuntimeError('Repeated replay presentation sources differ')
    fingerprints = [[json.loads(line) for line in (out / (name + '.frames.jsonl')).read_text().splitlines()]
                    for name in ('control-1', 'control-2')]
    if fingerprints[0] != fingerprints[1] or len(fingerprints[0]) != frame_count or any(
            row['frame'] != n or [row['width'], row['height']] != [args.width, args.height]
            for n, row in enumerate(fingerprints[0], 1)):
        raise RuntimeError('Repeated display fingerprints differ or omit frames')
    cases['range'] = native_call(out, 'range',
        f'build\\input-recording\\replay_inputs.exe --development "{target / "C3XRenderer.dll"}" "{target / "capture"}" --range "{target / "range"}" 2 2')
    if cases['range']['returncode'] or list((out / 'range').glob('*.bmp')) != [out / 'range/frame-000002.bmp'] or digest(out / 'range/frame-000002.bmp') != digest(out / 'control-1/frame-000002.bmp'):
        raise RuntimeError('Selected-frame replay differs from complete replay')
    check_time_and_prefix(out, cases)
    rejection = {
        'missing-clock': 'replay missing consumed clock input',
        'missing-asset': 'consumed asset missing from capture',
        'missing-unit': 'replay pixel witness differs',
        'missing-reset': 'replay missing consumed clock input',
        'alter-configuration': 'replay map output differs',
        'alter-visibility': 'replay map output differs',
        'alter-action': 'consumed asset missing from capture',
        'alter-cpu-write': 'replay CPU unit pixels differ',
    }
    for mutation in ('missing-clock', 'missing-asset', 'missing-unit', 'missing-reset',
                     'alter-configuration', 'alter-visibility', 'alter-action', 'alter-cpu-write'):
        cases[mutation + '-make'] = native_call(out, mutation + '-make',
            f'build\\input-recording\\mutate_inputs.exe "{target / "capture"}" "{target / mutation}" {mutation}')
        if cases[mutation + '-make']['returncode']:
            raise RuntimeError('Negative-control construction failed')
        cases[mutation] = native_call(out, mutation,
            f'build\\input-recording\\replay_inputs.exe --development "{target / "C3XRenderer.dll"}" "{target / mutation}"')
        if cases[mutation]['returncode'] != 1:
            raise RuntimeError('Missing input must reject cleanly with exit code 1: ' + mutation)
        if rejection[mutation] not in (out / (mutation + '.log')).read_text():
            raise RuntimeError('Negative control failed for an unrelated reason: ' + mutation)
    report = {'status': 'pass', 'qualified_for_gameplay': False, 'dll_sha256': identity,
              'extent': [args.width, args.height], 'steps': args.steps, 'frames': frame_count,
              'frames_identical_across_replays': True, 'range_matches_full_prefix': True,
              'fingerprints_identical_across_replays': True,
              'seconds_match_full_prefix': True, 'torn_prefix_explicitly_incomplete': True,
              'scope': 'Short renderer input control; forensic readiness, no physical scanout or performance claim',
              'storage': json.loads((out / 'capture/finished.json').read_text()), 'cases': cases}
    (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
