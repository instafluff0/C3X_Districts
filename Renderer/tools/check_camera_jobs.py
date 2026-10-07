"""Fail when a camera job fails, which leaves the map black or stale.

Usage: check_camera_jobs.py CAPTURE_DIR
CAPTURE_DIR holds a trace-level-2 helper log (renderer-core.log.x64) from a
scripted run with -ProfileRenderer.

On the 1498 AD save, the first camera job needed 88 KB more geometry than the
budget left after loading. Every resident entry belonged to that job, so the
admission was refused, the job failed (`camera-complete ... result=0`), and
the map stayed black for the session (performance review 4x). Under ordinary
memory conditions every camera job must succeed. Refusals are listed with
their `tile-cache-budget` detail; `tile-cache-overflow` lines show jobs that
needed the required-geometry ceiling.
"""
import pathlib
import re
import sys

COMPLETE = re.compile(r'stage=camera-complete ticket=(\d+) .*?result=(\d+)')
BUDGET = re.compile(r'stage=tile-cache-budget (.*)')
OVERFLOW = re.compile(r'stage=tile-cache-overflow (.*)')


def jobs(lines):
    results, refusals, overflows = [], [], []
    for line in lines:
        if match := COMPLETE.search(line):
            results.append((int(match[1]), match[2] == '1'))
        elif match := BUDGET.search(line):
            refusals.append(match[1].strip())
        elif match := OVERFLOW.search(line):
            overflows.append(match[1].strip())
    return results, refusals, overflows


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    log = pathlib.Path(argv[1]) / 'renderer-core.log.x64'
    results, refusals, overflows = jobs(log.read_text(errors='replace').splitlines())
    if not results:
        print(f'FAIL {log}: no camera-complete traces (trace level 2 is required)')
        return 1
    failed = [ticket for ticket, ok in results if not ok]
    for detail in refusals:
        print(f'refused {detail}')
    status = 'FAIL' if failed else 'PASS'
    print(f'{status} camera jobs: completed={len(results)} failed={len(failed)} '
          f'overflows={len(overflows)} first_failed={failed[0] if failed else "-"}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
