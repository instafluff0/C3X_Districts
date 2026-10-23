"""Day/night clock cost from a scripted Civ III capture (renderer.log at trace
level 1, optional cadence.json from -MeasureCadence).

Reports each Renderer64 lighting change (lighting-adopt: how long the new time
waited; lighting-settled: draws, static refinement ms and wall time until it
was fully refined), the Civ III-side capture cost (composite capture/render/
blit ms) and presentation cadence/CPU per second, split into the scroll and
idle segments of the run.

    python3 Renderer/tools/day_night_cost_report.py <capture-directory> [...]
"""
import json
import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(r'qpc=(-?\d+).*?stage=([a-z0-9-]+) (.*)')
FIELD = re.compile(r'([a-z_]+)=(-?[0-9.]+)')


def records(log):
    for line in log.splitlines():
        match = LINE.search(line)
        if match:
            fields = {k: float(v) for k, v in FIELD.findall(match.group(3))}
            yield int(match.group(1)), match.group(2), fields


def summary(values):
    if not values:
        return 'n=0'
    ordered = sorted(values)
    return (f'n={len(values)} mean={statistics.fmean(values):.1f} p50={ordered[len(ordered) // 2]:.1f} '
            f'max={ordered[-1]:.1f}')


def report(directory):
    directory = Path(directory)
    log = (directory / 'renderer.log').read_text(errors='replace')
    rows = list(records(log))
    frequency = 1e7
    cadence = []
    if (directory / 'cadence.json').is_file():
        data = json.loads((directory / 'cadence.json').read_text(encoding='utf-8-sig'))
        frequency = float(data.get('qpc_frequency') or frequency)
        cadence = data.get('samples') or []
    # Scroll window: Civ III's scripted camera steps (composite lines carry qpc;
    # scroll steps do not, so the composite lines between the first and last
    # step's log positions bound it).
    lines = log.splitlines()
    step_lines = [i for i, line in enumerate(lines) if 'stage=scripted-game-scroll step=' in line]
    window = None
    if step_lines:
        def qpc_near(index, direction):
            while 0 <= index < len(lines):
                match = LINE.search(lines[index])
                if match:
                    return int(match.group(1))
                index += direction
            return None
        window = (qpc_near(step_lines[0], -1), qpc_near(step_lines[-1], 1))
    def segment(qpc):
        if not window or window[0] is None or window[1] is None:
            return 'all'
        return 'scroll' if window[0] <= qpc <= window[1] else ('idle' if qpc > window[1] else 'before')
    out = {'capture': directory.name, 'scroll_steps': len(step_lines)}
    adopts = [(q, f) for q, stage, f in rows if stage == 'lighting-adopt']
    settled = [(q, f) for q, stage, f in rows if stage == 'lighting-settled']
    composites = [(q, f) for q, stage, f in rows if stage == 'composite']
    for name in ('scroll', 'idle', 'before', 'all'):
        a = [f for q, f in adopts if segment(q) == name]
        s = [f for q, f in settled if segment(q) == name]
        c = [f for q, f in composites if segment(q) == name]
        if not (a or s or c):
            continue
        out[name] = {
            'lighting_changes': len(a),
            'waited_ms': summary([f['waited_ms'] for f in a]),
            'settle_draws': summary([f['draws'] for f in s]),
            'settle_static_ms': summary([f['static_ms'] for f in s]),
            'settle_elapsed_ms': summary([f['elapsed_ms'] for f in s]),
            'captures': len(c),
            'capture_ms': summary([f['capture_ms'] for f in c if 'capture_ms' in f]),
            'render_wait_ms': summary([f['render_wait_ms'] for f in c if 'render_wait_ms' in f]),
        }
    if len(cadence) > 1:
        per = {}
        for previous, current in zip(cadence, cadence[1:]):
            seconds = (current['qpc'] - previous['qpc']) / frequency
            if seconds <= 0:
                continue
            name = segment(current['qpc'])
            bucket = per.setdefault(name, {'fps': [], 'game_cpu': [], 'helper_cpu': []})
            bucket['fps'].append((current['frames'] - previous['frames']) / seconds)
            bucket['game_cpu'].append(100 * (current['game_cpu_seconds'] - previous['game_cpu_seconds']) / seconds)
            bucket['helper_cpu'].append(100 * (current['helper_cpu_seconds'] - previous['helper_cpu_seconds']) / seconds)
        for name, bucket in per.items():
            out.setdefault(name, {})['presents_per_s'] = summary(bucket['fps'])
            out[name]['game_cpu_percent'] = summary(bucket['game_cpu'])
            out[name]['helper_cpu_percent'] = summary(bucket['helper_cpu'])
    return out


if __name__ == '__main__':
    for path in sys.argv[1:]:
        print(json.dumps(report(path), indent=1))
