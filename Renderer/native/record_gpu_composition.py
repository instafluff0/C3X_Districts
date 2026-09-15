"""Bounded standalone composition probe; never stages or launches Civ III."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from Renderer.lab.platform import ROOT, windows_root


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True, help='Preserved production renderer BMP')
    parser.add_argument('--width', type=int, default=1440)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--present', action='store_true', help='Include DXGI presentation in timing; otherwise present once outside timing')
    parser.add_argument('--jgl', type=Path, help='Optional local audited JGL DLL; executes its image operations without the game')
    args = parser.parse_args()
    source = args.map.resolve()
    relative = source.relative_to(ROOT)
    if source.suffix.lower() != '.bmp' or not source.is_file():
        parser.error('map must be an existing BMP in the checkout')
    if not (640 <= args.width <= 2240 and 480 <= args.height <= 1192):
        parser.error('viewport exceeds probe budget')
    # All command-tail values are controlled numbers/UUIDs or checked paths.
    if any(c in str(relative) for c in '\r\n"%&|<>^!'):
        parser.error('map path contains cmd metacharacters')
    jgl = args.jgl.resolve() if args.jgl else None
    if jgl:
        jgl_relative = jgl.relative_to(ROOT)
        if any(c in str(jgl_relative) for c in '\r\n"%&|<>^!'):
            parser.error('JGL path contains cmd metacharacters')
        if digest(jgl) != '0b0cd514de0d95b93d20655f4b5194173fe257325af82e558a152305ff0dbdf2':
            parser.error('JGL ABI witness only supports the audited binary')
    invocation = uuid.uuid4().hex
    directory = ROOT / 'Renderer/native/build/gpu-composition' / invocation
    directory.mkdir(parents=True)
    closure = [ROOT / 'Renderer/native' / name for name in
               ('benchmark_gpu_composition.cpp', 'test_jgl_image_operations.cpp', 'gpu_image_compositor.h', 'record_gpu_composition.py', 'BUILD.bat')]
    inputs = [source, *closure] + ([jgl] if jgl else [])
    before = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    windows_directory = windows_root() / directory.relative_to(ROOT)
    executable = ROOT / 'Renderer/native/build/gpu-composition/benchmark_gpu_composition.exe'
    run = directory / 'run.cmd'
    jgl_commands = ''
    if jgl:
        jgl_commands = (f'call BUILD.bat jgl-image-operations >"{windows_directory / "jgl-build.log"}" 2>&1\n'
                        'if errorlevel 1 goto failed\n'
                        f'build\\gpu-composition\\test_jgl_image_operations.exe "{windows_root() / jgl_relative}" '
                        f'>"{windows_directory / "jgl.log"}" 2>&1\n'
                        'if errorlevel 1 goto failed\n')
    run.write_text('@echo off\nsetlocal\n'
                   f'pushd "{windows_root() / "Renderer/native"}"\n'
                   + jgl_commands +
                   f'call BUILD.bat gpu-composition >"{windows_directory / "build.log"}" 2>&1\n'
                   'if errorlevel 1 goto failed\n'
                   f'build\\gpu-composition\\benchmark_gpu_composition.exe "{windows_root() / relative}" '
                   f'"{windows_directory / "results.json"}" {args.width} {args.height} {int(args.present)} '
                   f'>"{windows_directory / "native.log"}" 2>&1\n'
                   'set "C3X_PROBE_EXIT=%errorlevel%"\n'
                   f'>"{windows_directory / "completion.txt"}" echo {invocation} %C3X_PROBE_EXIT%\n'
                   'exit /b %C3X_PROBE_EXIT%\n:failed\n'
                   f'>"{windows_directory / "completion.txt"}" echo {invocation} 1\nexit /b 1\n')
    print(f'INVOCATION {directory.relative_to(ROOT)}', flush=True)
    start = time.monotonic()
    command = ['prlctl', 'exec', os.environ.get('C3X_RENDERER_VM', 'Windows 11'), '--current-user',
               'cmd', '/d', '/s', '/c', f'call "{windows_directory / "run.cmd"}"']
    if os.name == 'nt':
        command = ['cmd', '/d', '/c', str(run)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        transport = {'returncode': result.returncode, 'output': result.stdout + result.stderr}
    except subprocess.TimeoutExpired:
        # No automatic retry: a native process may still own the window/device.
        transport = {'returncode': None, 'output': 'Transport timed out; inspect this invocation before retrying.'}
    complete = (directory / 'completion.txt').read_text().split() if (directory / 'completion.txt').exists() else []
    unchanged = before == {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    passed = complete == [invocation, '0'] and unchanged and (directory / 'results.json').is_file()
    receipt = {'invocation': invocation, 'status': 'pass' if passed else 'incomplete_or_fail',
               'elapsed_seconds': time.monotonic() - start, 'inputs_sha256': before,
               'inputs_unchanged': unchanged, 'transport': transport,
               'executable_sha256': digest(executable) if executable.exists() else None,
               'scope': 'resident-map downstream composition; excludes renderer work and actual native presentation',
               'control_limitation': 'CPU arm includes a final upload to the common DXGI presenter; not the installed GDI presentation path',
               'completion': 'equal 1-pixel diagnostic staging barrier; not physical display latency',
               'gpu_timestamps': 'not used; unreliable on Parallels'}
    (directory / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if passed:
        shutil.copy2(executable, directory / executable.name)
    for name in ('jgl-build.log', 'jgl.log', 'build.log', 'native.log'):
        if (directory / name).exists():
            print((directory / name).read_text(errors='replace')[-6000:])
    print(json.dumps({'status': receipt['status'], 'directory': str(directory.relative_to(ROOT))}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
