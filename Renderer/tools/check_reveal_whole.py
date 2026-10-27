"""Fail when one reveal was shown in pieces.

Usage: check_reveal_whole.py CAPTURE_DIR
CAPTURE_DIR holds a helper log (renderer-core.log.x64) from
tools/run_scripted_game_test.ps1 with -ProfileRenderer.

A unit's step reveals tiles. The fog lifted at once on the view already
drawn for the tiles it drew (the hidden ring), and the rest waited for the
next view: tiles two away, tiles with a city or unit. Tiles then appeared
before their contents and neighbours, with sharp edges (the user, October
27). Now a reveal lifts on the drawn view only when that view draws all of
it; otherwise all of it waits for its view.

`reveal-shown` reports the cells a displayed sample newly showed and the age
of the native capture whose visibility they show. Two samples showing cells
from the same capture are one reveal shown in pieces.
"""
import pathlib
import re
import sys

SHOWN = re.compile(r'qpc=(\d+) .*?stage=reveal-shown cells=(\d+) capture_age_ms=([\d.]+)')
FREQUENCY = re.compile(r'qpc_frequency=(\d+)')


def pieces(lines):
    frequency, reveals = 0, []
    for line in lines:
        if not frequency:
            found = FREQUENCY.search(line)
            frequency = int(found[1]) if found else 0
        shown = SHOWN.search(line)
        if not shown or not frequency:
            continue
        at = int(shown[1]) * 1000 / frequency
        capture = at - float(shown[3])
        if reveals and abs(reveals[-1]['capture'] - capture) < .3:
            reveals[-1]['samples'].append((at, int(shown[2])))
        else:
            reveals.append({'capture': capture, 'samples': [(at, int(shown[2]))]})
    return reveals


def main(argv):
    if len(argv) != 2:
        print(__doc__.strip())
        return 2
    log = pathlib.Path(argv[1]) / 'renderer-core.log.x64'
    reveals = pieces(log.read_text(errors='replace').splitlines())
    if not reveals:
        print(f'FAIL {log}: no reveal-shown traces (needs -ProfileRenderer and a reveal)')
        return 1
    split = [reveal for reveal in reveals if len(reveal['samples']) > 1]
    for reveal in split[:5]:
        first = reveal['samples'][0][0]
        parts = ', '.join(f'+{at - first:.0f} ms {cells} cells' for at, cells in reveal['samples'])
        print(f'split reveal capture_ms={reveal["capture"]:.1f}: {parts}')
    status = 'FAIL' if split else 'PASS'
    print(f'{status} whole reveals: reveals={len(reveals)} split={len(split)}')
    return 1 if split else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
