"""Fail when the retained composition builds up anywhere in a capture.

Usage: check_retained_growth.py CAPTURE_DIR [--max-depth N] [--max-growth N] [--hover]
CAPTURE_DIR holds renderer.log (and renderer-core.log.x64, if present) from a
profiled run at trace level 2: any CAPTURE_GAMEPLAY.bat session, or a scripted
scenario run with -ProfileRenderer.

Build-up means each new version of a canvas reads the previous one, so the
graph that the helper re-evaluates every frame keeps deepening. Civ III's
Civilopedia cursor did this until October 21, 2026 (review 90). Each move
restored the saved screen from a copy that read the previous restore. In the
user's diplomacy hover, nodes grew from 130 to 1,071, frames took up to 218 ms,
and native presents blocked 1.1-1.6 s of every 2 s of the game thread.

With every eighth direct visual frame, the helper traces the longest input path
of its composition plan as `plan_depth`. A screen's depth is set when it opens
and then stays put: 5-11 on the busy 1498 AD map, 34 in the city screen
(review 91). A chain gains a level with every version, the cursor about one per
mouse move, and past 256 the helper refuses the composition. The check fails on
every stretch deeper than --max-depth (default 64), whatever caused it. It lists
each stretch with its time, depth, node count, slowest frame and the number of
cursor-sized presents, which show Civ III's software cursor was up.

Captures from builds before plan_depth carry no depth. For them the check lists
the largest rises of the node count. Opening a screen or loading a busy map also
raises it by several hundred nodes, so these rises are leads, not failures. The
check then fails only on the cursor case: a hover of continuous cursor-sized
presents during which nodes grow by more than --max-growth (default 128).

--hover also requires that the software cursor was active (the scripted
`pedia-hover` scenario).
"""
import argparse
import pathlib
import re
import statistics
import sys

PRESENT = re.compile(r'stage=native-ui-present result=1 .*area=(-?\d+),(-?\d+),(-?\d+),(-?\d+)')
VISUAL = re.compile(r'stage=direct-visual drawn=\d+ .*?sample_ms=([\d.]+) .*?nodes=(\d+) .*?copied_pixels=(\d+)')
PLAN = re.compile(r'stage=retained-execution-counts .*?plan_nodes=(\d+)(?: plan_depth=(\d+))?')
PROFILE = re.compile(r'stage=native-entry-profile window_ms=(\d+) .*top=(.*)')
QPC = re.compile(r'qpc=(\d+)')
FREQUENCY = 24e6
HOVER_GAP = 3.0  # seconds without cursor-sized presents that end a hover


def records(capture):
    """(seconds, text) for DebugView lines and helper trace-file lines."""
    rows, offsets = [], []
    for line in (capture / 'renderer.log').read_text(errors='replace').splitlines():
        fields = line.split('\t')
        if len(fields) < 4:
            continue
        try:
            seconds = float(fields[1])
        except ValueError:
            continue
        rows.append((seconds, fields[3]))
        qpc = QPC.search(fields[3])
        if qpc:
            offsets.append(int(qpc[1]) / FREQUENCY - seconds)
    core = capture / 'renderer-core.log.x64'
    if core.exists() and offsets:
        offset = statistics.median(offsets)
        for line in core.read_text(errors='replace').splitlines():
            qpc = QPC.search(line)
            if qpc and ('stage=direct-visual' in line or 'stage=retained-execution-counts' in line):
                rows.append((int(qpc[1]) / FREQUENCY - offset, line))
    rows.sort(key=lambda row: row[0])
    return rows


def parse(rows):
    visual, plans, cursor, profiles = [], [], [], []
    for t, text in rows:
        if m := VISUAL.search(text):
            visual.append((t, int(m[2]), float(m[1]), int(m[3])))
        elif m := PLAN.search(text):
            plans.append((t, int(m[1]), None if m[2] is None else int(m[2])))
        elif m := PRESENT.search(text):
            if 16 <= int(m[3]) - int(m[1]) <= 64 and 16 <= int(m[4]) - int(m[2]) <= 64:
                cursor.append(t)
        elif m := PROFILE.search(text):
            # Only the game thread's profile names native operations (opN).
            if re.search(r'(?:^|,)op\d+:', m[2]):
                present = re.search(r'op103:([\d.]+)/', m[2])
                profiles.append((t, int(m[1]), float(present[1]) if present else 0.))
    return visual, plans, cursor, profiles


def within(samples, begin, end):
    return [s for s in samples if begin <= s[0] <= end]


def hovers(cursor):
    """Continuous runs of cursor-sized presents: (begin, end, count)."""
    result = []
    for t in cursor:
        if result and t - result[-1][1] <= HOVER_GAP:
            result[-1] = (result[-1][0], t, result[-1][2] + 1)
        else:
            result.append((t, t, 1))
    return result


def stretch_summary(begin, end, visual, cursor, extra):
    frames = within(visual, begin, end)
    summary = dict(extra, begin=begin, end=end,
                   max_nodes=max((v[1] for v in frames), default=0),
                   slowest_ms=max((v[2] for v in frames), default=0.),
                   cursor_presents=sum(begin <= t <= end for t in cursor))
    return summary


def deep_stretches(plans, visual, cursor, limit):
    result, current = [], None
    for t, _, depth in plans:
        if depth is None:
            continue
        if depth > limit:
            current = [t, t, depth] if current is None else [current[0], t, max(current[2], depth)]
        elif current is not None:
            result.append(current)
            current = None
    if current is not None:
        result.append(current)
    return [stretch_summary(b, e, visual, cursor, {'max_depth': d}) for b, e, d in result]


def node_rises(visual, count=3):
    """Largest climbs of the 3 s minimum node count, each until it falls."""
    bins = {}
    for t, nodes, _, _ in visual:
        bins.setdefault(int(t // 3), []).append(nodes)
    keys = sorted(bins)
    rises, start = [], 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or min(bins[keys[i]]) < min(bins[keys[i - 1]]):
            low, high = min(bins[keys[start]]), min(bins[keys[i - 1]])
            if high > low:
                rises.append((high - low, keys[start] * 3., keys[i - 1] * 3. + 3., low, high))
            start = i
    return sorted(rises, reverse=True)[:count]


def hover_growth(visual, cursor, profiles):
    result = []
    for begin, end, presents in hovers(cursor):
        frames = within(visual, begin, end)
        window = sum(w for t, w, _ in profiles if begin <= t <= end + 2)
        blocked = sum(b for t, _, b in profiles if begin <= t <= end + 2)
        result.append({'begin': begin, 'end': end, 'cursor_presents': presents,
                       'first_nodes': frames[0][1] if frames else 0,
                       'max_nodes': max((v[1] for v in frames), default=0),
                       'slowest_ms': max((v[2] for v in frames), default=0.),
                       'present_blocked_ms_per_s': 1000. * blocked / window if window else 0.})
    return result


def check(capture, max_depth=64, max_growth=128, require_hover=False):
    """Returns (ok, report lines)."""
    visual, plans, cursor, profiles = parse(records(capture))
    lines, ok = [], True
    if not visual:
        return False, ['no direct-visual samples (trace level 2 is required)']
    depths = [d for _, _, d in plans if d is not None]
    lines.append(f'frames sampled={len(visual)} nodes={min(v[1] for v in visual)}-{max(v[1] for v in visual)} '
                 f'plan_depth={"-" if not depths else f"{min(depths)}-{max(depths)}"}')
    if depths:
        deepest = max(plans, key=lambda p: -1 if p[2] is None else p[2])
        lines.append(f'deepest plan {deepest[2]} at {deepest[0]:.1f}s ({deepest[1]} plan nodes)')
        for s in deep_stretches(plans, visual, cursor, max_depth):
            ok = False
            lines.append(f"build-up {s['begin']:.1f}-{s['end']:.1f}s: plan_depth {s['max_depth']} > {max_depth}, "
                         f"nodes up to {s['max_nodes']}, slowest frame {s['slowest_ms']:.0f} ms, "
                         f"cursor-sized presents {s['cursor_presents']}")
    else:
        lines.append('no plan_depth in this capture (older build): node rises are leads, cursor hovers decide')
        for rise, begin, end, low, high in node_rises(visual):
            lines.append(f'node rise {begin:.0f}-{end:.0f}s: {low} -> {high} (+{rise})')
    active = False
    for h in hover_growth(visual, cursor, profiles):
        grown = h['max_nodes'] - h['first_nodes']
        active |= h['cursor_presents'] >= 50
        lines.append(f"cursor hover {h['begin']:.1f}-{h['end']:.1f}s: presents {h['cursor_presents']}, "
                     f"nodes {h['first_nodes']} -> {h['max_nodes']} (+{grown}), slowest frame {h['slowest_ms']:.0f} ms, "
                     f"presents blocked {h['present_blocked_ms_per_s']:.0f} ms/s")
        if not depths and h['cursor_presents'] >= 50 and grown > max_growth:
            ok = False
            lines.append(f'  build-up: grew more than {max_growth} nodes while the cursor moved')
    if require_hover and not active:
        ok = False
        lines.append('software cursor was not active (no hover with 50 cursor-sized presents)')
    return ok, lines


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('capture', type=pathlib.Path)
    parser.add_argument('--max-depth', type=int, default=64)
    parser.add_argument('--max-growth', type=int, default=128)
    parser.add_argument('--hover', action='store_true')
    args = parser.parse_args(argv[1:])
    ok, lines = check(args.capture, args.max_depth, args.max_growth, args.hover)
    for line in lines:
        print(line)
    print(f"{'PASS' if ok else 'FAIL'} retained growth")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
