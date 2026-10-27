"""Per-screen report of a screen-sweep capture.

Usage: python3 Renderer/tools/screen_sweep_report.py CAPTURE [CAPTURE...]

For each screen in sweep-screens.json: GPU canvas fallbacks the adapter
logged (`native-cpu-fallback`, by operation), CPU-ownership readbacks and their
cost, native operation failures, and Civ III's worst message round trip
(pings.json). Operations 6-8 (pixel/bits/DC access) and 100-101 (drain,
reinit) are CPU ownership by design; any other fallback is a refused GPU draw
and fails the sweep, as do native failures and second-long pings.
"""
import json
import re
import sys
from pathlib import Path

CPU_ACCESS = {6, 7, 8, 100, 101}
NAMES = {6: 'pixel', 7: 'bits', 8: 'dc', 9: 'copy', 10: 'fill', 11: 'image', 12: 'sprite', 100: 'drain',
         101: 'reinit', 107: 'text', 108: 'sprite-blend', 109: 'tint', 110: 'line', 111: 'lookup',
         112: 'sprite-lookup', 113: 'sprite-lookup-over', 114: 'sprite-lookup-scaled', 115: 'sprite-style'}


def load(capture):
    capture = Path(capture)
    schedule = json.load(open(capture / 'sweep-screens.json', encoding='utf-8-sig'))
    mouse = json.load(open(capture / 'mouse-events.json', encoding='utf-8-sig'))
    frequency = mouse['qpc_frequency']
    lines = (capture / 'renderer.log').read_text(errors='replace').splitlines()
    # DebugView seconds -> QPC, from lines that carry both.
    pairs = []
    for line in lines:
        parts = line.split('\t')
        found = re.search(r'qpc=(\d+)', line)
        if len(parts) > 2 and found:
            try:
                pairs.append((float(parts[1]), int(found.group(1))))
            except ValueError:
                pass
    if len(pairs) < 2:
        raise SystemExit(f'{capture.name}: no timed trace lines')
    (t0, q0), (t1, q1) = pairs[0], pairs[-1]
    slope = (q1 - q0) / (t1 - t0) if t1 != t0 else frequency

    def qpc_at(seconds):
        return q0 + (seconds - t0) * slope
    # Harness seconds -> QPC: the first key after the load Enters opens the
    # first screen at its scheduled time.
    keys = json.load(open(capture / 'key-events.json', encoding='utf-8-sig'))['events']
    opened = next(k['qpc'] for k in keys if k['key'] != 13)
    origin = opened - frequency * schedule[0]['open']
    events = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) < 3:
            continue
        try:
            when = qpc_at(float(parts[1]))
        except ValueError:
            continue
        fallback = re.search(r'stage=native-cpu-fallback operation=(\d+) count=(\d+)', line)
        barrier = re.search(r'stage=native-cpu-barrier[^\n]*total_ms=([0-9.]+)', line)
        if fallback:
            events.append((when, 'fallback', int(fallback.group(1)), int(fallback.group(2))))
        elif barrier:
            events.append((when, 'barrier', float(barrier.group(1)), 0))
        elif 'stage=native-cpu-ownership-request' in line:
            events.append((when, 'readback', 0, 0))
        elif 'stage=native-operation-failed' in line:
            events.append((when, 'failed', 0, 0))
    pings = json.load(open(capture / 'pings.json', encoding='utf-8-sig'))['samples'] if (capture / 'pings.json').exists() else []
    return capture, schedule, frequency, origin, events, pings


def summarize(path):
    capture, schedule, frequency, origin, events, pings = load(path)
    result = json.load(open(capture / 'result.json', encoding='utf-8-sig'))
    print(f"{capture.name}: mouse={result['mouse_commands']} interactions={result['interaction_commands']} "
          f"exited_early={result['game_exited_early']} ui_ping_max_ms={result.get('ui_ping_max_ms')} "
          f"ui_pings_over_1s={result.get('ui_pings_over_1s')} native_failures={len(result['native_failures'])}")
    failed_any = False
    for item in schedule:
        low = origin + frequency * item['open']
        high = origin + frequency * (item['close'] + 1.5)
        inside = [e for e in events if low <= e[0] < high]
        refused = sorted({e[2] for e in inside if e[1] == 'fallback' and e[2] not in CPU_ACCESS})
        access = sorted({e[2] for e in inside if e[1] == 'fallback' and e[2] in CPU_ACCESS})
        readbacks = sum(1 for e in inside if e[1] == 'readback')
        barrier = max([e[2] for e in inside if e[1] == 'barrier'], default=0.0)
        failures = sum(1 for e in inside if e[1] == 'failed')
        # A ping belongs to a screen when its round trip overlaps the screen:
        # one sent just before a screen's key waits out the whole opening.
        trips = [(b - a) * 1000 / frequency for a, b, ok in pings if b >= low and a < high]
        unanswered = sum(1 for a, b, ok in pings if b >= low and a < high and not ok)
        worst = max(trips, default=0.0)
        bad = bool(refused or failures or unanswered or worst > 1000)
        failed_any |= bad
        print(f"  {'FAIL' if bad else 'ok  '} {item['screen']:13} refused={[NAMES.get(o, o) for o in refused]} "
              f"cpu_access={[NAMES.get(o, o) for o in access]} readbacks={readbacks} barrier_max_ms={barrier:.0f} "
              f"failures={failures} ping_max_ms={worst:.0f}{' unanswered=' + str(unanswered) if unanswered else ''}")
    return not failed_any


if __name__ == '__main__':
    ok = all([summarize(path) for path in sys.argv[1:]])
    sys.exit(0 if ok else 1)
