"""Unpaced production-boundary measurements for a complete recorded workload.

No frame readback, forensic timing enforcement, or live-FPS claim. Compare the
same capture and declared reservation across builds, after forensic validation.
"""
import argparse
import json
import math
import shutil
from pathlib import Path
from Renderer.lab.platform import ROOT, windows_root
from Renderer.native.test_input_replay import digest, native_call


def summarize(path):
    groups, memory = {}, []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if 'private_bytes' in row:
            memory.append(row)
        # These owners have explicit timing around the production call, after
        # reconstruction of captured external data. Leaf scaffolding does not.
        if row['family'] not in (19, 12) or row['reused_adoption']:
            continue
        if row['family'] == 12 and row['subtype'] != 1:
            continue
        key = f"{row['family']}/{row['subtype']}/result-{row['actual_result']}"
        groups.setdefault(key, []).append(row['production_service_ms'])
    def stats(values):
        values = sorted(values)
        return {'count': len(values), 'mean_ms': sum(values) / len(values),
                'median_ms': values[len(values) // 2],
                'p95_ms': values[max(0, math.ceil(len(values) * .95) - 1)],
                'max_ms': values[-1]}
    return {'production_boundary_groups': {k: stats(v) for k, v in groups.items()},
            'memory_samples': len(memory),
            'peak_private_bytes': max((x['private_bytes'] for x in memory), default=0),
            'minimum_free_va_bytes': min((x['free_va_bytes'] for x in memory), default=0),
            'minimum_largest_free_va_bytes': min((x['largest_free_va_bytes'] for x in memory), default=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--dll', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reserve-mib', type=int, default=0, choices=range(0, 2049))
    parser.add_argument('--runs', type=int, default=2, choices=range(1, 5))
    parser.add_argument('--compare-candidate', action='store_true')
    args = parser.parse_args()
    capture, dll, out = [x.resolve() for x in (args.capture, args.dll, args.out)]
    for path in (capture, dll, out):
        if any(c in path.relative_to(ROOT).as_posix() for c in '\r\n"%&|<>^!'):
            parser.error('unsupported input path')
    out.mkdir(parents=True, exist_ok=False)
    win = windows_root()
    target = win / out.relative_to(ROOT)
    source = win / capture.relative_to(ROOT)
    shutil.copy2(dll, out / 'C3XRenderer.dll')
    tools = ROOT / 'Renderer/native/build/input-recording'
    for name in ('replay_inputs.exe', 'inspect_inputs.exe'):
        shutil.copy2(tools / name, out / name)
    receipt = {'status': 'fail', 'scope': 'unpaced_native_service_not_live_fps',
               'qualified_for_gameplay': False, 'reserved_va_mib': args.reserve_mib,
               'dll_sha256': digest(out / 'C3XRenderer.dll'),
               'tool_sha256': digest(out / 'replay_inputs.exe'),
               'candidate_comparison': args.compare_candidate, 'runs': [],
               'limitations': ['External native input reconstruction and journal I/O excluded from service time; they still contend for CPU.',
                               'Serial completion-order execution does not reproduce original overlap or game-thread think time.',
                               'Camera publication retains recorded consumption points because external CPU inputs are unavailable earlier; actual readiness is awaited there.',
                               'Reservation models capacity, not live heaps, fragmentation or co-resident GPU/CPU activity.',
                               'Memory samples include the replay harness and native dependency cache.',
                               'Changed native input requirements fail rather than inventing data.']}
    try:
        result = native_call(out, 'inspect', f'"{target / "inspect_inputs.exe"}" "{source}" "{target / "inspection"}"')
        if result['returncode'] or not json.loads((out / 'inspection/report.json').read_text())['complete']:
            raise RuntimeError('performance comparison needs a complete input corpus')
        for index in range(args.runs):
            name = f'run-{index + 1}'
            command = (f'"{target / "replay_inputs.exe"}" --development "{target / "C3XRenderer.dll"}" "{source}"'
                       f' --performance "{target / (name + ".jsonl")}"')
            if args.reserve_mib:
                command += f' --reserve-mib {args.reserve_mib}'
            if args.compare_candidate:
                command += ' --compare-candidate'
            result = native_call(out, name, command, timeout=1800)
            if result['returncode']:
                raise RuntimeError('production measurement replay failed')
            receipt['runs'].append(result | summarize(out / (name + '.jsonl')))
        receipt['status'] = 'pass'
    finally:
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
