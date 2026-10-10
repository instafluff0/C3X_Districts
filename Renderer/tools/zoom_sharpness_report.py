"""Summarize presented-frame sharpness by zoom from a scripted capture.

Needs a capture run with C3X_RENDERER_ROUTE_WITNESS=1 (route-presented and
route-frame-budget lines in renderer-core.log.x64). A frame is "soft" when its
world was drawn at a lower scale than presented (stretch > 1). Steady segments
at one zoom are reported with frame rate and soft share; review section 49.

    python3 Renderer/tools/zoom_sharpness_report.py <capture-folder>
"""
import re
import sys
from pathlib import Path


def frames(log):
    zoom, stretch = {}, {}
    for line in log.read_text(errors='replace').splitlines():
        if 'route-presented' in line:
            m = re.search(r'present_index=(\d+) zoom_q16=(\d+).*present_qpc=(\d+)', line)
            if m:
                zoom[int(m.group(1))] = (int(m.group(2)) / 65536, int(m.group(3)))
        elif 'route-frame-budget' in line:
            p = re.search(r'present_index=(\d+)', line)
            s = re.search(r'stretch=([0-9.]+)', line)
            if p and s:
                stretch[int(p.group(1))] = float(s.group(1))
    rows = sorted((q, z, stretch.get(i, 1.0)) for i, (z, q) in zoom.items())
    if not rows:
        return []
    start = rows[0][0]
    return [((q - start) / 24e6, z, s) for q, z, s in rows]


def segments(rows, gap=1.0):
    """Runs of frames at one zoom with no pause longer than `gap` seconds."""
    out = []
    for t, z, s in rows:
        if out and out[-1]['zoom'] == round(z, 3) and t - out[-1]['end'] <= gap:
            seg = out[-1]
        else:
            seg = {'zoom': round(z, 3), 'begin': t, 'end': t, 'frames': 0, 'soft': 0}
            out.append(seg)
        seg['end'] = t
        seg['frames'] += 1
        seg['soft'] += s > 1.01
    return out


def main(folder):
    rows = frames(Path(folder) / 'renderer-core.log.x64')
    if not rows:
        print('no route-presented frames (run with C3X_RENDERER_ROUTE_WITNESS=1)')
        return 1
    for seg in segments(rows):
        span = seg['end'] - seg['begin']
        if seg['frames'] < 8 or span < 1.0:
            continue
        print('zoom=%.2f t=%.1f-%.1fs frames=%d fps=%.1f soft=%d (%.0f%%)' % (
            seg['zoom'], seg['begin'], seg['end'], seg['frames'], seg['frames'] / span,
            seg['soft'], 100.0 * seg['soft'] / seg['frames']))
        if seg['zoom'] != 1.0:
            # Per-second detail: scrolling seconds show as low fps or soft frames.
            second = {}
            for t, z, s in rows:
                if seg['begin'] <= t <= seg['end'] and round(z, 3) == seg['zoom']:
                    n, soft = second.get(int(t), (0, 0))
                    second[int(t)] = (n + 1, soft + (s > 1.01))
            print('  per second (frames/soft): ' + ' '.join('%d:%d/%d' % (k, n, soft) for k, (n, soft) in sorted(second.items())))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
