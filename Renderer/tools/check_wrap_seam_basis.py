"""Fail when waves and resource poses were drawn a world width off screen.

Usage: check_wrap_seam_basis.py CAPTURE_DIR [WRAP_PIXELS]
CAPTURE_DIR holds a trace-level-2 helper log (renderer-core.log.x64) from
tools/run_scripted_game_test.ps1 with -ProfileRenderer. WRAP_PIXELS is the
world width at 1x (map width x 64); a -Route run's renderer.log supplies it.

Coastal waves, resource poses and the winter flurry are window records drawn
once, without wrap copies, at the capture's camera (Renderer64, store mode).
The window basis is kept per window (it changes when a scene render's
canonical membership is not covered). Across the wrap seam Civ III re-bases
the camera a world width while the window keeps its frame: a basis kept from
before then drew those layers off screen until the window next changed, so
waves and animals vanished until a turn ended (the user, October 27; review
92). The fix follows the re-base and traces `window-basis-rebase`.

Each draw's own basis is its fresh-scene-phases translation minus camera. A
draw fails when the basis in effect (the window's, or the last traced
re-base) is half a world width or more away from it.

Reproduce on a wrapping map (the user's 100x100 2950 BC save): debug-scroll
with a -Route that holds x=0 and steps rows, then steps to x=64.
"""
import pathlib
import re
import sys

DRAW = re.compile(r'qpc=(\d+) .*?stage=fresh-scene-phases .*?camera=(-?\d+),(-?\d+) translation=(-?[\d.]+),(-?[\d.]+)')
REBASE = re.compile(r'stage=window-basis-rebase basis=(-?[\d.]+)')


def misplaced(lines, wrap):
    basis, pending, draws, found = None, True, 0, []
    for line in lines:
        if 'stage=canonical-membership ' in line:
            pending = pending or ' covered=0 ' in line
            continue
        rebase = REBASE.search(line)
        if rebase:
            basis, pending = float(rebase[1]), False
            continue
        draw = DRAW.search(line)
        if not draw:
            continue
        own = float(draw[4]) - int(draw[2])
        if pending or basis is None:
            basis, pending = own, False
        draws += 1
        if abs(own - basis) >= wrap / 2:
            found.append((int(draw[1]), int(draw[2]), int(draw[3]), basis, own))
    return draws, found


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__.strip())
        return 2
    capture = pathlib.Path(argv[1])
    wrap = int(argv[2]) if len(argv) == 3 else 0
    if not wrap and (capture / 'renderer.log').is_file():
        width = re.search(r'map_width=(\d+)', (capture / 'renderer.log').read_text(errors='replace'))
        wrap = int(width[1]) * 64 if width else 0
    if not wrap:
        print('FAIL: pass WRAP_PIXELS (map width x 64); this capture has no route map_width')
        return 1
    log = capture / 'renderer-core.log.x64'
    draws, found = misplaced(log.read_text(errors='replace').splitlines(), wrap)
    if not draws:
        print(f'FAIL {log}: needs fresh-scene-phases traces with translation (trace level 2)')
        return 1
    for qpc, x, y, basis, own in found[:5]:
        print(f'off screen qpc={qpc} camera={x},{y} basis={basis:.0f} own={own:.0f}')
    status = 'FAIL' if found else 'PASS'
    print(f'{status} wrap seam basis: draws={draws} off_screen={len(found)} wrap={wrap}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
