"""Summarize a sound_service_probe log and, optionally, a --record capture.

Civ III's sound.dll mixes ~270 ms ahead of the DirectSound play cursor, so a
service interval of about 280 ms or more replays stale ring audio (the audible
"blip"). The report lists long intervals, whether the probe itself was paused
for them (guest-wide) and, for a recording, the windows that repeat the audio
one 44.1 kHz ring (65536 frames) earlier.

    python3 Renderer/tools/sound_probe_report.py PROBE.txt [--record REC.txt]
Exit status 1 when any interval reaches --audible-ms (default 280).
"""
import argparse
import re
import sys

import numpy as np


def stalls(text):
    return [(float(m.group(1)), int(m.group(2)), m.group(3)) for m in
            re.finditer(r'^stall t=([\d.]+) .*?interval_ms=(\d+) .*?(guest-wide|game-only)', text, re.M)]


def replay_windows(record_path, window_s=0.02):
    """Times (s) of windows whose audio repeats the sound.dll ring exactly."""
    header = open(record_path).read()
    fmt = dict(re.findall(r'(\w+)=(\d+)', header.split('\n', 1)[0]))
    rate, channels = int(fmt['rate']), int(fmt['channels'])
    if int(fmt['bits']) != 32:
        raise SystemExit('expected a float32 shared-mode mix format')
    x = np.fromfile(record_path + '.raw', dtype=np.float32).reshape(-1, channels).mean(axis=1)
    win = int(rate * window_s)
    lag = int(round(65536 / 44100 * rate))
    hits = []
    for i in range(lag // win + 2, len(x) // win):
        a = x[i * win:(i + 1) * win]
        if float((a * a).sum()) < 1e-6:
            continue
        best = 0.0
        for shift in range(lag - 30, lag + 31):
            b = x[i * win - shift:i * win - shift + win]
            d = float(np.sqrt((a * a).sum() * (b * b).sum()))
            if d > 1e-9:
                best = max(best, float((a * b).sum()) / d)
        if best > 0.99:
            hits.append(i * window_s)
    return hits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('probe')
    parser.add_argument('--record')
    parser.add_argument('--audible-ms', type=int, default=280)
    args = parser.parse_args()
    text = open(args.probe).read()
    summary = re.search(r'^summary .*$', text, re.M)
    holds = re.findall(r'^hold .*$', text, re.M)
    print(summary.group(0) if summary else ('hold test' if holds else 'summary missing (probe still running or stopped early)'))
    for line in holds:
        print('  ' + line)
    found = stalls(text)
    for t, interval, kind in found:
        print(f'  t={t:8.3f}s interval={interval:4d} ms {kind}')
    audible = [s for s in found if s[1] >= args.audible_ms]
    print(f'intervals >= {args.audible_ms} ms: {len(audible)}')
    if args.record:
        hits = replay_windows(args.record)
        runs, start, last = [], None, None
        for t in hits:
            if start is None or t - last > 0.05:
                if start is not None:
                    runs.append((start, last))
                start = t
            last = t
        if start is not None:
            runs.append((start, last))
        # Only runs longer than a few windows are ring replays; music repeats briefly.
        replays = [(a, b) for a, b in runs if b - a >= 0.08]
        for a, b in replays:
            print(f'  ring replay at {a:7.2f}-{b:7.2f}s of the recording')
        print(f'ring replays: {len(replays)}')
        audible += replays
    sys.exit(1 if audible else 0)


if __name__ == '__main__':
    main()
