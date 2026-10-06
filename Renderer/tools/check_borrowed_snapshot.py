"""Fail when ambient frames redo snapshot work during camera jobs.

Usage: check_borrowed_snapshot.py CAPTURE_DIR
CAPTURE_DIR holds a trace-level-2 helper log (renderer-core.log.x64), for
example a `unit-turn` run of tools/run_scripted_game_test.ps1 with
-ProfileRenderer.

While a camera job runs, ambient frames draw the borrowed completed view: an
immutable snapshot whose shadow atlas and static raster were drawn from that
same view. Only the live topology the job has already applied differs. Judging
the snapshot by it redrew the whole shadow atlas (reuse_failures=128) and
repaired the same static region on every frame, so water and units skipped
after a unit move (performance review 4v).

A frame counts as borrowed when its trace says `borrowed=1`. Older shadow
traces without that field fall back to the camera-job window between
`render-begin` and `camera-complete`.
"""
import pathlib
import re
import sys

STAGE = re.compile(r'qpc=(\d+) .*?stage=(fresh-shadow-build|static-compose|render-begin|camera-complete)\b(.*)')


def violations(lines):
    jobs, builds, statics = [], [], []
    for line in lines:
        match = STAGE.search(line)
        if not match:
            continue
        qpc, stage = int(match[1]), match[2]
        fields = dict(re.findall(r'(\w+)=([\w.,-]+)', match[3]))
        if stage in ('render-begin', 'camera-complete'):
            jobs.append((qpc, stage))
        elif stage == 'fresh-shadow-build':
            builds.append((qpc, fields))
        else:
            statics.append((qpc, fields))
    windows, start = [], None
    for qpc, stage in sorted(jobs):
        if stage == 'render-begin':
            start = qpc
        elif start is not None:
            windows.append((start, qpc))
            start = None
    if start is not None:
        windows.append((start, float('inf')))
    found = []
    for qpc, fields in builds:
        borrowed = fields['borrowed'] == '1' if 'borrowed' in fields else any(a <= qpc <= b for a, b in windows)
        if borrowed and fields.get('reuse_failures') == '128' and int(fields.get('pages', '0')) > 0:
            found.append((qpc, 'shadow atlas redraw', fields))
    for qpc, fields in statics:
        # Entry bit 16: the displayed slot was repaired this frame.
        if fields.get('borrowed') == '1' and int(fields.get('entry', '0')) & 16:
            found.append((qpc, 'static raster repair', fields))
    diagnosed = sum('reuse_failures' in fields for _, fields in builds)
    return sorted(found, key=lambda item: item[0]), len(builds), len(windows), diagnosed


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    log = pathlib.Path(argv[1]) / 'renderer-core.log.x64'
    found, builds, windows, diagnosed = violations(log.read_text(errors='replace').splitlines())
    if not builds or diagnosed < builds:
        print(f'FAIL {log}: needs fresh-shadow-build traces with reuse_failures (trace level 2, current build)')
        return 1
    for qpc, kind, fields in found:
        print(f"borrowed {kind} qpc={qpc} pages={fields.get('pages', '-')} changed={fields.get('changed', '-')}")
    status = 'FAIL' if found else 'PASS'
    print(f'{status} borrowed snapshot reuse: shadow_builds={builds} camera_jobs={windows} redone={len(found)}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
