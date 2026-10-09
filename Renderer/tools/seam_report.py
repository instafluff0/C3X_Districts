"""Find straight seams in sampled window frames of a scripted capture.

Usage: seam_report.py CAPTURE_DIR [CAPTURE_DIR ...] [--band TOP,BOTTOM] [--list]

A seam is a full-length discontinuity: a column (or row) whose step to its
neighbour is much larger than the local gradient on most rows (or columns) of
the map band. Map content (coasts, borders, roads, labels) does not run
straight across most of the screen; stale strips, repeated slices and uncovered
edges do (performance review, section 24). Only the map band is examined;
TOP and BOTTOM are fractions of the frame height (the default excludes Civ
III's top buttons and the bottom panel and minimap).

CAPTURE_DIR is a run of tools/run_scripted_game_test.ps1 with -SampleHz 10
(window/timeline.jsonl and window-*.jpg). When mouse-events.json holds the
`near` scenario's inputs, frames are grouped into near_report's segments by
their arrival QPC. JPEG frames are decoded with Pillow when present, otherwise
with macOS `sips`.
"""
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

BAND = (0.04, 0.78)
# A line votes when its step exceeds RATIO times the mean local gradient and
# FLOOR (summed over RGB, 0..765); a seam needs VOTES of the band's lines.
# Calibrated October 8 on busy-save 10 Hz frames: three real stale-strip seams
# during 1x scroll voted 0.48-0.57; 87 clean map frames had none.
RATIO, FLOOR, VOTES = 2.5, 20.0, 0.4
NEIGHBOURS = (-6, -5, -4, -3, -2, 2, 3, 4, 5, 6)


def seams(image, band=BAND):
    """Seams in one HxWx3 frame: [(axis, position, vote fraction)].

    axis 'x' is a vertical seam between columns position and position+1;
    axis 'y' a horizontal seam between rows position and position+1."""
    height = image.shape[0]
    top, bottom = int(band[0] * height), int(band[1] * height)
    region = image[top:bottom].astype(np.int16)
    found = []
    for axis, plane in (('x', region), ('y', region.transpose(1, 0, 2))):
        # step[line, i]: change between positions i and i+1 along the line.
        step = np.abs(np.diff(plane, axis=1)).sum(axis=2).astype(np.float32)
        count = step.shape[1]
        if count <= 2 * max(NEIGHBOURS) + 1:
            continue
        inner = slice(max(NEIGHBOURS), count - max(NEIGHBOURS))
        local = sum(step[:, max(NEIGHBOURS) + k:count - max(NEIGHBOURS) + k] for k in NEIGHBOURS) / len(NEIGHBOURS)
        here = step[:, inner]
        votes = ((here > RATIO * local) & (here > FLOOR)).mean(axis=0)
        for index in np.flatnonzero(votes >= VOTES):
            position = int(index) + max(NEIGHBOURS)
            if axis == 'y':
                position += top
            found.append((axis, position, round(float(votes[index]), 2)))
    return found


def read_bmp(path):
    data = path.read_bytes()
    offset = int.from_bytes(data[10:14], 'little')
    width = int.from_bytes(data[18:22], 'little', signed=True)
    height = int.from_bytes(data[22:26], 'little', signed=True)
    depth = int.from_bytes(data[28:30], 'little') // 8
    if depth not in (3, 4):
        raise ValueError(f'{path}: unsupported BMP depth {depth * 8}')
    stride = (width * depth + 3) & ~3
    rows = np.frombuffer(data, np.uint8, stride * abs(height), offset).reshape(abs(height), stride)
    pixels = rows[:, :width * depth].reshape(abs(height), width, depth)[:, :, 2::-1]
    return pixels[::-1] if height > 0 else pixels


def frames(window, keep=lambda qpc: True):
    """(arrival_qpc, decoded frame) in capture order, for arrivals `keep` accepts."""
    timeline = [json.loads(line) for line in (window / 'timeline.jsonl').read_text().splitlines() if line.strip()]
    entries = [(row['arrival_qpc'], window / f"window-{row['frame'] + 1:06d}.jpg") for row in timeline if 'frame' in row]
    entries = [(qpc, path) for qpc, path in entries if path.exists() and keep(qpc)]
    try:
        from PIL import Image
        for qpc, path in entries:
            with Image.open(path) as image:
                yield qpc, np.asarray(image.convert('RGB'))
        return
    except ImportError:
        pass
    if not shutil.which('sips'):
        raise SystemExit('seam_report needs Pillow or macOS sips to decode JPEG frames')
    with tempfile.TemporaryDirectory() as scratch:
        scratch = pathlib.Path(scratch)
        for start in range(0, len(entries), 200):
            chunk = entries[start:start + 200]
            subprocess.run(['sips', '-s', 'format', 'bmp', *[str(path) for _, path in chunk], '--out', str(scratch)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for qpc, path in chunk:
                converted = scratch / (path.stem + '.bmp')
                yield qpc, read_bmp(converted)
                converted.unlink()


def segments(capture):
    """near_report segments as (name, t0, t1) in QPC, or [] for other scenarios."""
    events_file = capture / 'mouse-events.json'
    cadence_file = capture / 'cadence.json'
    if not events_file.exists() or not cadence_file.exists():
        return []
    from Renderer.tools.near_report import SEGMENTS, JUMPS
    raw = json.loads(events_file.read_text())
    events = [e['qpc'] for e in (raw['events'] if isinstance(raw, dict) else raw)]
    if len(events) < 25:
        return []
    frequency = float(json.loads(cadence_file.read_text())['qpc_frequency'])
    rows = []
    for name, first, last, start, end, _ in SEGMENTS:
        t0, t1 = events[first] + start * frequency, events[last] + end * frequency
        if first == last:
            t0, t1 = events[last] + start * frequency, events[last]
        rows.append((name, t0, t1))
    for name, event in JUMPS:
        rows.append((name, events[event - 1], events[event + 1] if event + 1 < len(events) else events[event]))
    return rows


def report(capture, band=BAND):
    capture = pathlib.Path(capture)
    spans = segments(capture)
    totals = {name: {'frames': 0, 'seam_frames': 0, 'worst': 0.0} for name, _, _ in spans}
    totals['other'] = {'frames': 0, 'seam_frames': 0, 'worst': 0.0}
    flagged = []
    # Loading screens and menus outside the near segments have real straight
    # edges; with segments known, only frames inside them are decoded.
    keep = (lambda qpc: any(t0 <= qpc <= t1 for _, t0, t1 in spans)) if spans else (lambda qpc: True)
    for qpc, image in frames(capture / 'window', keep):
        name = next((n for n, t0, t1 in spans if t0 <= qpc <= t1), 'other')
        found = seams(image, band)
        row = totals[name]
        row['frames'] += 1
        if found:
            row['seam_frames'] += 1
            row['worst'] = max(row['worst'], max(v for _, _, v in found))
            flagged.append({'qpc': qpc, 'segment': name, 'seams': found})
    return totals, flagged


def main(argv):
    args = [a for a in argv[1:] if not a.startswith('--')]
    band = BAND
    for a in argv[1:]:
        if a.startswith('--band='):
            band = tuple(float(v) for v in a.split('=', 1)[1].split(','))
    if not args:
        print(__doc__.strip())
        return 2
    for capture in args:
        totals, flagged = report(capture, band)
        print(f'== {capture}')
        for name, row in totals.items():
            if row['frames']:
                print(f"  {name}: frames={row['frames']} seam_frames={row['seam_frames']} worst={row['worst']}")
        if '--list' in argv:
            for row in flagged:
                print(f"    qpc={row['qpc']} {row['segment']} {row['seams'][:4]}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
