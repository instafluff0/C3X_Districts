"""Summarize a domestic-slider capture: UI-thread pings and slider display lag.

Usage: python3 Renderer/tools/domestic_slider_lag.py CAPTURE [CAPTURE...]

Pings (pings.json) are WM_NULL round trips to the game window: Civ III's
message latency, measured the same way with or without Renderer64. The
science flask's x position in each window frame of the drag, compared with
the scripted cursor's x some time earlier, gives the lag at which the slider
follows the cursor (a parabola through the best fitting lags, 5 ms steps).
Needs ffmpeg and numpy.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BAND = (296, 328)     # client rows of the science flask
TRACK = (1164, 1362)  # client columns of its track (even width for ffmpeg)
PHASES = (('map idle', 30, 39), ('science drag', 45.5, 65), ('luxury drag', 67.5, 77), ('advisor idle', 79, 91))


def flask_x(frame):
    width, height = TRACK[1] - TRACK[0], BAND[1] - BAND[0]
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(frame), '-vf',
                          f'crop={width}:{height}:{TRACK[0]}:{BAND[0]}', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True, check=True).stdout
    pixels = np.frombuffer(raw, np.uint8).reshape(height, width, 3).astype(int)
    # Flask glass and pink body; the yellow track has little blue.
    columns = np.nonzero(((pixels[..., 2] > 170) & (pixels[..., 0] > 150)).sum(0) >= 3)[0]
    return None if len(columns) < 4 else TRACK[0] + float(np.median(columns))


def best_lag(rows, times, xs):
    lags = np.arange(40, 260, 5)
    error = np.array([np.mean([abs(x - np.interp(t - lag / 1000, times, xs)) for t, x in rows]) for lag in lags])
    i = int(error.argmin())
    window = slice(max(0, i - 6), i + 7)
    a, b, _ = np.polyfit(lags[window], error[window], 2)
    return -b / (2 * a), float(error.min())


def summarize(capture):
    capture = Path(capture)
    start = json.load(open(capture / 'window/started.json', encoding='utf-8-sig'))['started_qpc']
    result = json.load(open(capture / 'result.json', encoding='utf-8-sig'))
    mouse = json.load(open(capture / 'mouse-events.json', encoding='utf-8-sig'))
    frequency = mouse['qpc_frequency']
    times = np.array([(e['qpc'] - start) / frequency for e in mouse['events']])
    xs = np.array([e['client_x'] for e in mouse['events']])
    origin = times[0] - 44.0  # the first scripted move is at 44 s
    print(f"{capture.name}: native_rendering={result.get('native_rendering')} mouse={result['mouse_commands']} "
          f"interactions={result['interaction_commands']} failures={len(result['native_failures'])}")
    pings = json.load(open(capture / 'pings.json', encoding='utf-8-sig'))
    trips = [((sent - start) / frequency - origin, (received - sent) * 1000 / frequency) for sent, received, _ in pings['samples']]
    for name, low, high in PHASES:
        values = sorted(ms for t, ms in trips if low <= t < high)
        if values:
            at = lambda q: values[min(len(values) - 1, int(q * len(values)))]
            print(f"  {name:13} pings={len(values):4} median={np.median(values):5.1f} p90={at(.9):5.1f} max={values[-1]:5.1f} ms")
    rows = []
    for line in open(capture / 'window/timeline.jsonl'):
        event = json.loads(line)
        if 'frame' not in event or 'arrival_qpc' not in event:
            continue
        t = (event['arrival_qpc'] - start) / frequency
        if 45.6 <= t - origin < 65:
            x = flask_x(capture / 'window' / f"window-{event['frame']:06d}.jpg")
            if x is not None:
                rows.append((t, x))
    if len(rows) < 20:
        print('  slider: too few frames with the flask (is the Domestic Advisor open?)')
        return
    lag, error = best_lag(rows, times, xs)
    halves = [best_lag(part, times, xs)[0] for part in (rows[:len(rows) // 2], rows[len(rows) // 2:])]
    print(f"  slider lag {lag:.0f} ms (halves {halves[0]:.0f}/{halves[1]:.0f}) residual {error:.1f} px over {len(rows)} frames")


if __name__ == '__main__':
    for path in sys.argv[1:]:
        summarize(path)
