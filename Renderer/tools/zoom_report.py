"""Per-notch zoom responsiveness from a scripted capture.

Usage: zoom_report.py CAPTURE_DIR

Needs cadence.json (-MeasureCadence: presented frame counts and the presented
zoom, sampled about every 20 ms) and mouse-events.json (the scripted wheel
events). For each wheel event it reports:
  start_ms   wheel event to the first sample whose presented zoom moved
  settle_ms  wheel event to the first sample within 1% of the final zoom
  fps        presented frames per second from the event to settle
  gap_ms     the longest interval between presented frames in that span
  after_fps  presented frames per second in the second after settle
Consecutive notches closer together than a transition share their samples.
"""
import json
import pathlib
import sys


def main(argv):
    capture = pathlib.Path(argv[1])
    cadence = json.loads((capture / 'cadence.json').read_text())
    f = float(cadence['qpc_frequency'])
    samples = [(s['qpc'], s['frames'], (s.get('zoom_q16') or 65536) / 65536.) for s in cadence['samples']]
    raw = json.loads((capture / 'mouse-events.json').read_text())
    wheels = [e for e in (raw['events'] if isinstance(raw, dict) else raw) if e.get('wheel_delta')]
    for n, event in enumerate(wheels):
        start = event['qpc']
        end = wheels[n + 1]['qpc'] if n + 1 < len(wheels) else start + int(3 * f)
        span = [s for s in samples if start <= s[0] < end]
        if len(span) < 3:
            continue
        before = [s for s in samples if s[0] < start]
        initial = before[-1][2] if before else span[0][2]
        final = span[-1][2]
        moved = next((s for s in span if abs(s[2] - initial) > 1e-3), None)
        settled = next((s for s in span if abs(s[2] - final) <= .01 * final), span[-1])
        transition = [s for s in span if s[0] <= settled[0]]
        frames = transition[-1][1] - transition[0][1]
        seconds = max(1e-6, (transition[-1][0] - transition[0][0]) / f)
        gap, last_change = 0., None
        for a, b in zip(transition, transition[1:]):
            if b[1] != a[1]:
                if last_change is not None:
                    gap = max(gap, (b[0] - last_change) / f * 1000)
                last_change = b[0]
        after = [s for s in span if settled[0] <= s[0] <= settled[0] + f]
        after_fps = (after[-1][1] - after[0][1]) / max(1e-6, (after[-1][0] - after[0][0]) / f) if len(after) > 1 else 0.
        print(f"wheel {n + 1} delta {event['wheel_delta']:+d} zoom {initial:.2f}->{final:.2f} "
              f"start_ms {((moved[0] - start) / f * 1000) if moved else float('nan'):.0f} "
              f"settle_ms {(settled[0] - start) / f * 1000:.0f} fps {frames / seconds:.1f} "
              f"gap_ms {gap:.0f} after_fps {after_fps:.1f}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
