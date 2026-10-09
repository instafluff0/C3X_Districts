"""Gaps between presented frames per `near` segment of a scripted capture.

Usage: frame_gap_report.py CAPTURE_DIR [CAPTURE_DIR ...]

CAPTURE_DIR needs a helper log with route-presented traces (-ProfileRenderer
with C3X_RENDERER_ROUTE_WITNESS=1; renderer-core.log.x64) and the near
scenario's mouse-events.json. Each presented frame's present_qpc is a helper
presentation (not physical scanout). Per segment it reports the gap p50, p90
and maximum in ms and how many gaps exceeded 30 ms: smooth motion needs even
pacing, not only a frame rate (performance review, section 17).
"""
import json
import pathlib
import re
import sys

PRESENT = re.compile(r'stage=route-presented .*?present_qpc=(\d+)')


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(fraction * (len(values) - 1) + 0.5))]


def gaps(presents, t0, t1, frequency):
    inside = sorted(q for q in presents if t0 <= q <= t1)
    return [(b - a) / frequency * 1000 for a, b in zip(inside, inside[1:])]


def report(capture):
    from Renderer.tools.near_report import SEGMENTS
    capture = pathlib.Path(capture)
    log = capture / 'renderer-core.log.x64'
    presents = [int(m[1]) for m in PRESENT.finditer(log.read_text(errors='replace'))]
    frequency = 24e6
    match = re.search(r'stage=route-presented .*?frequency=(\d+)', log.read_text(errors='replace'))
    if match:
        frequency = float(match[1])
    raw = json.loads((capture / 'mouse-events.json').read_text())
    events = [e['qpc'] for e in (raw['events'] if isinstance(raw, dict) else raw)]
    rows = []
    for name, first, last, start, end, _ in SEGMENTS:
        t0, t1 = events[first] + start * frequency, events[last] + end * frequency
        if first == last:
            t0, t1 = events[last] + start * frequency, events[last]
        values = gaps(presents, t0, t1, frequency)
        rows.append({'segment': name, 'frames': len(values) + 1 if values else 0,
                     'gap_p50': round(percentile(values, .5), 1) if values else None,
                     'gap_p90': round(percentile(values, .9), 1) if values else None,
                     'gap_max': round(max(values), 1) if values else None,
                     'over_30ms': sum(1 for v in values if v > 30)})
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
