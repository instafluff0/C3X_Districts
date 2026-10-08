"""Summarize Civ III's thread time inside bridge entries, per near segment.

Usage: entry_profile_report.py CAPTURE_DIR [--top N]

CAPTURE_DIR is a scripted `near` capture with input tracing (renderer.log with
`stage=native-entry-profile`, c3x_renderer.cpp `c3x_entry_profile`), plus
mouse-events.json. Each 2 s report lists name:ms/calls/max_ms for the
costliest keys: opN = native image operation (c3x_renderer_api.h), observeN and
lifetimeN = JGL observation/lifetime operations, and named entries (unit facts,
map-view, navigation, camera-request/poll, world-timer, policy-read).

Reports are assigned to the segment containing their end. Windows that span a
segment boundary are counted in the later segment.
"""
import collections
import json
import pathlib
import re
import sys

ENTRY = re.compile(r'([a-z\-]+\d*):([\d.]+)/(\d+)(?:/([\d.]+))?')


def segments(capture):
    raw = json.loads((capture / 'mouse-events.json').read_text())
    q = [e['qpc'] for e in (raw['events'] if isinstance(raw, dict) else raw)]
    if len(q) < 25:
        return [('all', 0, 1 << 62)]
    return [('1x scroll', q[0], q[3]), ('2x scroll', q[13], q[14]), ('3x scroll', q[16], q[17]),
            ('3x idle', q[17], q[18]), ('1x idle end', q[19], q[24])]


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    capture = pathlib.Path(argv[1])
    top = int(argv[argv.index('--top') + 1]) if '--top' in argv else 8
    spans = segments(capture)
    ms = collections.defaultdict(collections.Counter)
    calls = collections.defaultdict(collections.Counter)
    longest = collections.defaultdict(dict)
    window = collections.Counter()
    total = collections.Counter()
    for line in (capture / 'renderer.log').read_text(errors='replace').splitlines():
        if 'native-entry-profile' not in line:
            continue
        qpc = int(re.search(r'qpc=(\d+)', line)[1])
        name = next((n for n, a, b in spans if a <= qpc <= b), None)
        if not name:
            continue
        window[name] += float(re.search(r'window_ms=([\d.]+)', line)[1])
        total[name] += float(re.search(r'total_ms=([\d.]+)', line)[1])
        for key, spent, count, peak in ENTRY.findall(line.split('top=', 1)[1]):
            ms[name][key] += float(spent)
            calls[name][key] += int(count)
            if peak:
                longest[name][key] = max(longest[name].get(key, 0.0), float(peak))
    for name, _, _ in spans:
        if not window[name]:
            continue
        print(f'== {name}: window_ms={window[name]:.0f} bridge_ms={total[name]:.0f} '
              f'({100 * total[name] / window[name]:.0f}% of the window)')
        for key, spent in ms[name].most_common(top):
            n = calls[name][key]
            peak = longest[name].get(key)
            print(f'   {key:16s} ms={spent:8.1f} calls={n:7d} us/call={1000 * spent / max(1, n):8.1f}'
                  + (f' max_ms={peak:.1f}' if peak is not None else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
