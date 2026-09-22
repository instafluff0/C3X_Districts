"""Pin verified short-diagnostic evidence; optionally stage its exact DLL.

This never grants ten-minute or live-performance qualification and never starts
Civ III or INSTALL.bat. Evidence directories are local, ignored build artifacts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from Renderer.lab.platform import ROOT
from Renderer.native.record_renderer_build import DLL_UNITS, unit_inputs


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def validate_build(receipt, identity, current_inputs):
    # Both supported builders pin the DLL: the input-contract builder uses a
    # scalar field, while record_renderer_build records all output binaries.
    identities = [receipt.get('dll_sha256'),
                  receipt.get('binaries', {}).get('C3XRenderer.dll')]
    identities = [value for value in identities if value is not None]
    # The standard builder also records the standalone preview executable.
    recorded = receipt.get('unit_inputs', {})
    runtime = {unit: recorded.get(unit) for unit in current_inputs}
    if (receipt.get('returncode') != 0 or not receipt.get('sources_unchanged')
            or receipt.get('preview_only', False) or not identities
            or any(value != identity for value in identities)
            or runtime != current_inputs):
        raise ValueError('Build does not match current runtime sources and DLL')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--observer', type=Path, required=True)
    parser.add_argument('--stage', action='store_true')
    args = parser.parse_args()
    build, campaign, controls, observer = [p.resolve() for p in
                                         (args.build, args.campaign, args.controls, args.observer)]
    for path in (build, campaign, controls, observer):
        path.relative_to(ROOT / 'Renderer/native/build')
    binary = build / 'C3XRenderer.dll'
    identity = digest(binary)
    receipt = read(build / 'build-evidence.json')
    validate_build(receipt, identity, {u: unit_inputs(u) for u in DLL_UNITS})
    policy = read(campaign / 'policy.json')
    result = read(campaign / 'receipt.json')
    if result['status'] != 'pass' or policy['order'] != ['off-1', 'on-1', 'on-2', 'off-2']:
        raise ValueError('Missing passing reverse-order overhead campaign')
    for group in result['groups'].values():
        for key in ('median', 'p95'):
            limit = max(policy['max_added_' + key + '_ms'],
                        policy['max_added_' + key + '_fraction'] * group['off'][key])
            if group['on'][key] - group['off'][key] > limit:
                raise ValueError('Capture overhead exceeds its declared limit')
    for name in policy['order']:
        arm = read(campaign / name / 'receipt.json')
        if (arm['status'] != 'pass' or not arm['inputs_unchanged']
                or not arm['binary_provenance']['current_runtime_matches_build']
                or arm['inputs'].get(binary.relative_to(ROOT).as_posix()) != identity):
            raise ValueError('Campaign arm identity or correctness differs: ' + name)
        if name.startswith('on') and not arm['input_recording']['closed']:
            raise ValueError('Recorded campaign arm is incomplete')
        settings = arm['settings']
        if (settings['C3X_RENDERER_TEST_RESERVE_MIB'] != '1024'
                or settings['C3X_RENDERER_PROFILE'] != '0'
                or settings['C3X_RENDERER_WAVES'] != '1'
                or settings['C3X_RENDERER_WATER_MOTION'] != '1'
                or settings['C3X_RENDERER_REFLECTION_CONTROL'] != '0'):
            raise ValueError('Campaign pressure/effect controls differ')
        if arm['input_soak']['seconds'] != 30 or not arm['input_soak']['memory']:
            raise ValueError('Missing measured workload duration or memory')
    checks = read(controls / 'receipt.json')
    if checks['status'] != 'pass' or checks['dll_sha256'] != identity or checks['frames'] < 1 or any(
            value['returncode'] for value in checks['cases'].values()):
        raise ValueError('Requested-stop/window/exact replay controls failed')
    if (controls / 'control-1.jsonl').read_bytes() != (controls / 'control-2.jsonl').read_bytes():
        raise ValueError('Repeated replay fingerprints differ')
    window = read(observer / 'receipt.json')
    if (window['status'] != 'pass' or not window['input_recording']['closed']
            or not window['inputs_unchanged']
            or window['inputs'].get(binary.relative_to(ROOT).as_posix()) != identity
            or window['window_witness']['helper_sha256'] != checks['window_witness_sha256']):
        raise ValueError('Combined fullscreen input/window capture failed')
    if checks['launcher_sha256'] != digest(ROOT / 'Renderer/tools/capture_game.ps1'):
        raise ValueError('Launcher changed after its tests')
    if checks['window_witness_sha256'] != digest(ROOT / 'Renderer/native/build/window-witness/window_witness.exe'):
        raise ValueError('Window collector changed after its tests')
    target = ROOT / 'Renderer/native/build/input-recording/short-capture-ready.json'
    tools = target.parent
    record = dict(status='pass', scope='short-diagnostic-capture',
                  ten_minute_qualified=False, live_fps_qualified=False,
                  dll_sha256=identity,
                  window_witness_sha256=digest(ROOT / 'Renderer/native/build/window-witness/window_witness.exe'),
                  inspector_sha256=digest(tools / 'inspect_inputs.exe'),
                  replay_sha256=digest(tools / 'replay_inputs.exe'),
                  launcher_sha256=digest(ROOT / 'Renderer/tools/capture_game.ps1'),
                  evidence={name: str(path.relative_to(ROOT)) for name, path in
                            [('build', build), ('campaign', campaign), ('controls', controls), ('observer', observer)]},
                  low_address_space_stop_bytes=128 * 1024 * 1024,
                  limitations=['Capacity reservation does not reproduce game heap fragmentation',
                               'Recorded FPS includes capture and sampled-window observer overhead',
                               'Earlier native consumption points remain constrained by captured CPU inputs'])
    if args.stage:
        staged = ROOT / 'Renderer/bin/C3XRenderer.dll'
        rollback = controls / 'previous-staged-C3XRenderer.dll'
        if rollback.exists():
            raise ValueError('Rollback already exists; inspect prior staging before repeating')
        record['previous_dll_sha256'] = digest(staged)
        shutil.copy2(staged, rollback)
        shutil.copy2(binary, staged)
        if digest(staged) != identity:
            raise ValueError('Staged DLL differs')
        record['rollback'] = str(rollback.relative_to(ROOT))
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
