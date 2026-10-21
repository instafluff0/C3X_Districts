"""Fail when hovering with Civ III's software cursor grows the retained UI.

Usage: check_cursor_hover.py CAPTURE_DIR [max_node_growth]
CAPTURE_DIR holds renderer.log from a profiled run (trace level 2): the
scripted `pedia-hover` scenario or a CAPTURE_GAMEPLAY session that hovers
Civilopedia links in diplomacy or the city screen.

Over such links Civ III hides the Windows cursor and draws a sprite cursor
itself: each move restores the saved screen under the old position, saves
the new one and presents both small rectangles. Until October 21 2026 every
restore chained to the previous save, so the retained composition grew by
about three animated nodes per move (130 -> 1070) and each frame re-ran the
chain (up to 141 M copied pixels, 218 ms). Native presents waited for those
frames: 1.1-1.6 s of every 2 s of the game thread.

The hover is the span of cursor-sized native presents. The check requires
that the software cursor was active and that the retained node count
(`direct-visual nodes=`) grows by at most max_node_growth (default 128)
within it. Present blocking and the slowest visual frame are reported.
"""
import pathlib
import re
import statistics
import sys

PRESENT = re.compile(r'stage=native-ui-present result=1 .*area=(-?\d+),(-?\d+),(-?\d+),(-?\d+)')
VISUAL = re.compile(r'stage=direct-visual drawn=\d+ .*?sample_ms=([\d.]+) .*?nodes=(\d+) .*?copied_pixels=(\d+)')
PROFILE = re.compile(r'stage=native-entry-profile window_ms=(\d+) .*top=(.*)')
QPC = re.compile(r'qpc=(\d+)')
FREQUENCY = 24e6


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
            if qpc and 'stage=direct-visual' in line:
                rows.append((int(qpc[1]) / FREQUENCY - offset, line))
    rows.sort(key=lambda row: row[0])
    return rows


def measure(rows):
    cursor = [t for t, text in rows for m in [PRESENT.search(text)] if m and
              16 <= int(m[3]) - int(m[1]) <= 64 and 16 <= int(m[4]) - int(m[2]) <= 64]
    if not cursor:
        return None
    begin, end = cursor[0], cursor[-1]
    visual = [(float(m[1]), int(m[2]), int(m[3])) for t, text in rows if begin <= t <= end
              for m in [VISUAL.search(text)] if m]
    blocked, window = 0., 0.
    for t, text in rows:
        m = PROFILE.search(text) if begin <= t <= end + 2 else None
        # Only the game thread's profile names native operations (opN).
        if m and re.search(r'(?:^|,)op\d+:', m[2]):
            window += int(m[1])
            present = re.search(r'op103:([\d.]+)/', m[2])
            blocked += float(present[1]) if present else 0.
    return {
        'cursor_presents': len(cursor), 'seconds': end - begin,
        'first_nodes': visual[0][1] if visual else 0, 'max_nodes': max((v[1] for v in visual), default=0),
        'slowest_visual_ms': max((v[0] for v in visual), default=0.),
        'max_copied_pixels': max((v[2] for v in visual), default=0),
        'present_blocked_ms_per_s': 1000. * blocked / window if window else 0.,
        'visual_samples': len(visual)}


def check(result, growth=128):
    if result is None or result['cursor_presents'] < 50:
        return False, 'software cursor was not active (fewer than 50 cursor-sized presents)'
    if not result['visual_samples']:
        return False, 'no direct-visual samples during the hover (trace level 2 is required)'
    grown = result['max_nodes'] - result['first_nodes']
    return grown <= growth, f'node growth {grown} (limit {growth})'


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__.strip())
        return 2
    result = measure(records(pathlib.Path(argv[1])))
    ok, reason = check(result, int(argv[2]) if len(argv) == 3 else 128)
    detail = ' '.join(f'{k}={v:.1f}' if isinstance(v, float) else f'{k}={v}' for k, v in (result or {}).items())
    print(f"{'PASS' if ok else 'FAIL'} cursor hover: {reason}; {detail}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv))
