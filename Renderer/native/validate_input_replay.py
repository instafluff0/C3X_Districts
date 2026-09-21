"""Compare two complete production input replays using bounded display readbacks.

Requires BUILD.bat input-replay and a DLL with the optional fingerprint export.
This is forensic repeatability, not performance or complete gameplay qualification.
"""
import argparse
import json
import shutil
from pathlib import Path
from Renderer.lab.platform import ROOT, windows_root
from Renderer.native.test_input_replay import digest, native_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--dll', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--compare-candidate', action='store_true')
    parser.add_argument('--tools', type=Path, default=Path('Renderer/native/build/input-recording'),
                        help='Matching protocol tools; frozen historical tool directory is supported')
    args = parser.parse_args()
    capture, dll, out = (path.resolve() for path in (args.capture, args.dll, args.out))
    win = windows_root()
    tool_source=args.tools.resolve()
    for path in (capture, dll, out, tool_source):
        relative = path.relative_to(ROOT).as_posix()
        if any(char in relative for char in '\r\n"%&|<>^!'):
            parser.error('unsupported input path')
    if not capture.is_dir() or not dll.is_file():
        parser.error('capture and DLL must exist')
    out.mkdir(parents=True, exist_ok=False)
    target = win / out.relative_to(ROOT)
    source = win / capture.relative_to(ROOT)
    shutil.copy2(dll,out/'C3XRenderer.dll')
    dll=out/'C3XRenderer.dll'
    binary = target/'C3XRenderer.dll'
    frozen_tools=out/'replay-tools';frozen_tools.mkdir()
    for name in ('replay_inputs.exe','inspect_inputs.exe'):
        shutil.copy2(tool_source/name,frozen_tools/name)
    # The DLL and assets are reverified against the recorded manifest by each
    # run. Pin the tool bytes as well so this comparison remains inspectable.
    tool = frozen_tools/'replay_inputs.exe'
    replay_tool=target/'replay-tools/replay_inputs.exe'
    inspect_tool=target/'replay-tools/inspect_inputs.exe'
    report = {'status': 'fail', 'qualified_for_gameplay': False,
              'scope': 'Recorded output witnesses and repeated composed display fingerprints; forensic, not performance',
              'dll_sha256': digest(dll), 'tool_sha256': digest(tool),
              'candidate_comparison': args.compare_candidate, 'cases': {}}
    try:
        cases = report['cases']
        cases['inspect'] = native_call(out, 'inspect',
            f'"{inspect_tool}" "{source}" "{target / "inspection"}"')
        if cases['inspect']['returncode']:
            raise RuntimeError('Input inspection failed')
        inspection = json.loads((out / 'inspection/report.json').read_text())
        expected_frames = inspection['accepted_presentations']
        if not inspection['complete'] or not expected_frames:
            raise RuntimeError('Validation requires a complete capture with presentations')
        controls = []
        for name in ('control-1', 'control-2'):
            command = (f'"{replay_tool}" --development "{binary}" "{source}"'
                       f' --fingerprints "{target / (name + ".frames.jsonl")}"'
                       f' --timeline "{target / (name + ".timeline.jsonl")}"')
            if args.compare_candidate:
                command += ' --compare-candidate'
            cases[name] = native_call(out, name, command, timeout=1800)
            if cases[name]['returncode']:
                raise RuntimeError('Production input replay failed')
            records = [json.loads(line) for line in (out / (name + '.frames.jsonl')).read_text().splitlines()]
            if len(records) != expected_frames or [row['frame'] for row in records] != list(range(1, expected_frames + 1)):
                raise RuntimeError('Display fingerprint coverage incomplete')
            controls.append(records)
        differences = [a['frame'] for a, b in zip(*controls) if a != b]
        report.update(frames=expected_frames, differing_frames=differences,
                      repeatable=not differences,
                      fingerprint_bytes=sum((out / (name + '.frames.jsonl')).stat().st_size
                                            for name in ('control-1', 'control-2')))
        if differences:
            raise RuntimeError('Repeated display sources differ')
        if digest(dll) != report['dll_sha256'] or digest(tool) != report['tool_sha256']:
            raise RuntimeError('Validation binary changed during comparison')
        report['status'] = 'pass'
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        (out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
