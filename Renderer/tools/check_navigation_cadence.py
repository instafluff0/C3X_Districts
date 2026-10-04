"""Check sampled zoom completion and presentation stalls; never report scanout FPS."""
import argparse
import json
from pathlib import Path
import re
import statistics


def check(samples, frequency, targets, max_settle=3., max_stall=1., max_sample_gap=.25):
    if frequency <= 0 or not samples or not targets:
        raise ValueError('Cadence and zoom commands are required')
    rows = sorted(samples, key=lambda row: row['qpc'])
    start = targets[0][0]
    rows = [row for row in rows if row['qpc'] / frequency >= start]
    failures, unknown, completions = [], [], []
    last_progress = None
    largest_stall = 0.
    for previous, row in zip(rows, rows[1:]):
        time, prior = row['qpc'] / frequency, previous['qpc'] / frequency
        if time - prior > max_sample_gap or row['frames'] < previous['frames']:
            unknown.append([prior, time])
            last_progress = None
        elif row['frames'] > previous['frames']:
            last_progress = time
        else:
            if last_progress is None:
                last_progress = prior
            largest_stall = max(largest_stall, time - last_progress)
    if largest_stall > max_stall:
        failures.append('Presentation progress stopped beyond the allowed duration')
    for index, (time, target) in enumerate(targets):
        end = targets[index + 1][0] if index + 1 < len(targets) else float('inf')
        # Rapid reversals are intentionally superseded, not failed completion.
        if end - time < .35:
            completions.append({'target_q16': target, 'superseded': True})
            continue
        matching = [row['qpc'] / frequency for row in rows
                    if time <= row['qpc'] / frequency < end and row['frames'] > 0
                    and abs(row['zoom_q16'] - target) <= 64]
        latency = matching[0] - time if matching else None
        completions.append({'target_q16': target, 'seconds': latency})
        if latency is None or latency > max_settle:
            failures.append('Zoom did not reach its target within the allowed duration')
    return {'status': 'fail' if failures else 'incomplete' if unknown else 'pass',
            'failures': failures, 'zoom_completions': completions,
            'max_observed_no_progress_seconds': largest_stall,
            'unobserved_intervals': unknown,
            'scope': 'Sampled successful helper presentations; not physical scanout or proof of visible fidelity'}


def analyze(capture, **limits):
    cadence = json.loads((capture / 'cadence.json').read_text(encoding='utf-8-sig'))
    frequency = cadence['qpc_frequency']
    offsets, commands = [], []
    for line in (capture / 'renderer.log').read_text(errors='replace').splitlines():
        columns = line.split('\t')
        if len(columns) < 3:
            continue
        time = float(columns[1])
        qpc = re.search(r'\bqpc=(\d+)', line)
        if qpc:
            offsets.append(int(qpc[1]) / frequency - time)
        zoom = re.search(r'stage=zoom-target .*new_width=(\d+)', line)
        if zoom:
            commands.append((time, int(zoom[1]) * 512))
    if not offsets:
        raise ValueError('No shared clock markers in the native trace')
    origin = statistics.median(offsets)
    return check(cadence['samples'], frequency, [(t + origin, z) for t, z in commands], **limits)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--max-settle', type=float, default=3.)
    parser.add_argument('--max-stall', type=float, default=1.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.capture, max_settle=args.max_settle, max_stall=args.max_stall)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'pass' else 1)
