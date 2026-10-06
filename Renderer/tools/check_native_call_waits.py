"""Fail when Civ III's thread waits on the input-coverage worker again.

Usage: check_native_call_waits.py CAPTURE_DIR [max_window_ms]
CAPTURE_DIR holds renderer.log from a scripted run with input tracing
(C3X_RENDERER_TRACE_INPUT=1, which the `near`, `zoom` and `reveal-scroll`
scenarios set). The bridge writes `native-call-waits` every 2 s.

While scrolling, Civ III issues thousands of native UI commands per second. In
October 2026 the coverage worker fell behind, and the game thread spent
350-940 ms of every 2 s window blocked on its backlog (performance review 4t).
After the fix the worst busy window was about 200 ms. The default limit,
300 ms per window, separates the two with margin for VM noise.
"""
import pathlib
import re
import sys

WAITS = re.compile(r'stage=native-call-waits calls=(\d+) .*?backlog_ms=([\d.]+) .*?window_ms=([\d.]+)')


def windows(lines):
    result = []
    for line in lines:
        match = WAITS.search(line)
        if match:
            result.append((int(match[1]), float(match[2]), float(match[3])))
    return result


def check(rows, limit=300.):
    # Every window counts: staged UI commands reach the worker even when few
    # helper calls happen, and those windows had the longest waits.
    worst = max((row[1] for row in rows), default=0.)
    return worst <= limit, worst, len(rows)


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__.strip())
        return 2
    limit = float(argv[2]) if len(argv) == 3 else 300.
    rows = windows((pathlib.Path(argv[1]) / 'renderer.log').read_text(errors='replace').splitlines())
    if not rows:
        print('FAIL no native-call-waits windows (input tracing and a current build are required)')
        return 1
    ok, worst, busy = check(rows, limit)
    print(f"{'PASS' if ok else 'FAIL'} native call waits: windows={busy} worst_backlog_ms={worst:.1f} limit_ms={limit:.0f}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
