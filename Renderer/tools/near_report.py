"""Report the `near` scenario's segments from a scripted capture.

Usage: near_report.py CAPTURE_DIR [CAPTURE_DIR ...]

CAPTURE_DIR is a run of tools/run_scripted_game_test.ps1 with -Scenario near
and -MeasureCadence. It needs cadence.json, mouse-events.json and renderer.log;
input tracing (which `near` sets) supplies the edge-scroll and handoff lines.

The near scenario's 25 inputs (scripted_game_test.ps1) bound the segments:
1x idle, 1x scroll on x then y, two minimap jumps, four notches in, a 2x
scroll, one double notch, a 3x scroll and idle, six notches out, 1x idle.

Per segment it reports presented frames per second (the helper's visual frame
counter, sampled every 20 ms) and, for scrolls, the camera steps Civ III
adopted: native px/s, screen px/s at the presented zoom, steps, the interval
between adoptions and the latency from each step's first request to its adoption.
For jumps it reports the time from the minimap click to the adopted camera.
Compare runs only at the same trace level (performance review 4u).
"""
import json
import pathlib
import re
import statistics
import sys

# (name, start event, end event, start offset s, end offset s); events are the
# near scenario's mouse inputs in order. A negative start event means "end
# event minus the offset".
SEGMENTS = [
    ('1x idle', 0, 0, -8.0, 0.0, 'idle'),
    ('1x scroll x', 0, 1, 0.0, 0.0, 'scroll'),
    ('1x scroll y', 2, 3, 0.0, 0.0, 'scroll'),
    ('zoom in to 2x', 9, 13, 0.0, 0.0, 'zoom'),
    ('2x scroll', 13, 14, 0.0, 0.0, 'scroll'),
    ('zoom to 3x', 15, 16, 0.0, 0.0, 'zoom'),
    ('3x scroll', 16, 17, 0.0, 0.0, 'scroll'),
    ('3x idle', 17, 18, 0.0, 0.0, 'idle'),
    ('zoom out to 1x', 18, 23, 0.0, 3.0, 'zoom'),
    ('1x idle end', 23, 24, 4.0, 0.0, 'idle'),
]
JUMPS = [('jump 1', 5), ('jump 2', 7)]
LINE = re.compile(r'^\s*\d+\t([\d.]+)\t\[\d+\]\t(.*)$')


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(fraction * (len(values) - 1) + 0.5))]


def load(capture):
    capture = pathlib.Path(capture)
    cadence = json.loads((capture / 'cadence.json').read_text())
    frequency = float(cadence['qpc_frequency'])
    samples = [(s['qpc'], s['frames'], s.get('zoom_q16') or 65536) for s in cadence['samples']]
    raw = json.loads((capture / 'mouse-events.json').read_text())
    events = [e['qpc'] for e in (raw['events'] if isinstance(raw, dict) else raw)]
    pairs, handoffs, requests = [], [], []
    for line in (capture / 'renderer.log').read_text(errors='replace').splitlines():
        match = LINE.match(line)
        if not match:
            continue
        seconds, text = float(match[1]), match[2]
        qpc = re.search(r'qpc=(\d+)', text)
        if qpc:
            pairs.append((seconds, int(qpc[1])))
        if 'stage=native-handoff ' in text:
            camera = re.search(r'requested=(-?\d+),(-?\d+)', text)
            handoffs.append((seconds, int(qpc[1]) if qpc else None, int(camera[1]), int(camera[2])))
        elif 'stage=edge-scroll ' in text and qpc:
            camera = re.search(r'camera=(-?\d+),(-?\d+)', text)
            requests.append((int(qpc[1]), int(camera[1]), int(camera[2])))
    # Handoffs carry their QPC since 2026-10-07. Older captures map DebugView
    # seconds to QPC by least squares, which can lag by several hundred ms.
    if len(pairs) < 2:
        raise SystemExit(f'{capture}: no qpc-stamped lines to align the log')
    mean_s = statistics.fmean(p[0] for p in pairs)
    mean_q = statistics.fmean(p[1] for p in pairs)
    slope = (sum((s - mean_s) * (q - mean_q) for s, q in pairs) /
             max(1e-12, sum((s - mean_s) ** 2 for s, _ in pairs)))
    to_qpc = lambda s: mean_q + (s - mean_s) * slope
    handoffs = [(q if q is not None else to_qpc(s), x, y) for s, q, x, y in handoffs]
    return frequency, samples, events, handoffs, requests


def frames_between(samples, t0, t1):
    inside = [s for s in samples if t0 <= s[0] <= t1]
    if len(inside) < 2:
        return None, None
    seconds = (inside[-1][0] - inside[0][0])
    zoom = statistics.median(s[2] for s in inside) / 65536.
    return (inside[-1][1] - inside[0][1]), seconds, zoom


def report(capture):
    frequency, samples, events, handoffs, requests = load(capture)
    if len(events) < 25:
        raise SystemExit(f'{capture}: expected the near scenario\'s 25 inputs, found {len(events)}')
    rows = []
    for name, first, last, start, end, kind in SEGMENTS:
        t0 = events[first] + start * frequency
        t1 = events[last] + end * frequency
        if first == last:
            t0, t1 = events[last] + start * frequency, events[last]
        counted = frames_between(samples, t0, t1)
        row = {'segment': name, 'seconds': round((t1 - t0) / frequency, 2)}
        if counted[0] is not None:
            frames, seconds, zoom = counted
            row['fps'] = round(frames / (seconds / frequency), 1)
            row['zoom'] = round(zoom, 2)
        if kind == 'scroll':
            before = [h for h in handoffs if h[0] < t0]
            steps = [h for h in handoffs if t0 <= h[0] <= t1]
            previous = before[-1] if before else None
            distance, intervals, latencies = 0, [], []
            previous_adoption = before[-1][0] if before else None
            for h in steps:
                if previous:
                    distance += abs(h[1] - previous[1]) + abs(h[2] - previous[2])
                    if previous[0] >= t0:
                        intervals.append((h[0] - previous[0]) / frequency * 1000)
                previous = h
                # Civ III repeats a step's request every tick until it is
                # adopted; time the adoption from the first request for it.
                matching = [r for r in requests if r[0] <= h[0] and (r[1], r[2]) == (h[1], h[2])
                            and (not previous_adoption or r[0] > previous_adoption)]
                if matching:
                    latencies.append((h[0] - matching[0][0]) / frequency * 1000)
                previous_adoption = h[0]
            elapsed = (t1 - t0) / frequency
            row.update({'steps': len(steps), 'native_px_s': round(distance / elapsed, 1),
                        'screen_px_s': round(distance / elapsed * row.get('zoom', 1.0), 1),
                        'step_ms_p50': round(percentile(intervals, .5), 1) if intervals else None,
                        'step_ms_max': round(max(intervals), 1) if intervals else None,
                        'adopt_ms_p50': round(percentile(latencies, .5), 1) if latencies else None})
        rows.append(row)
    for name, event in JUMPS:
        # Civ III can move on the button press, before the release (JUMPS
        # names the release); time from the press.
        click = events[event - 1]
        prior = [h for h in handoffs if h[0] < click]
        start = prior[-1] if prior else None
        # A click Civ III ignored has no move before the next scripted input;
        # a later scroll handoff must not be reported as a slow jump.
        until = events[event + 1] if event + 1 < len(events) else float('inf')
        landed = next((h for h in handoffs if click < h[0] < until and start and
                       abs(h[1] - start[1]) + abs(h[2] - start[2]) > 1000), None)
        rows.append({'segment': name, 'jump_ms': round((landed[0] - click) / frequency * 1000, 1) if landed else None})
    return rows


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    for capture in argv[1:]:
        print(f'== {capture}')
        for row in report(capture):
            print('  ' + '  '.join(f'{k}={v}' for k, v in row.items() if v is not None))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
