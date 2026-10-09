"""Summarize the input-coverage worker's applied operations.

Usage: hit_trace_report.py CAPTURE_DIR [--window START_QPC END_QPC]

CAPTURE_DIR holds renderer-core.log.hit, written by a scripted run with
-ProfileRenderer and -RendererOptions "C3X_RENDERER_HIT_TRACE=1"
(gpu_image_worker_client.h, HitWorker::record). Each record is one operation
the worker applied to its coverage model, with its steady-clock start and its
apply time in microseconds. Kind 5 records a canvas the form hit test never
reads (an exempt canvas); its later draws are skipped before the worker.

Civ III's game thread waits whenever the worker falls more than 512 operations
behind (`native-call-waits backlog_ms`). This report shows where the worker's
time goes: per operation kind, per command kind and destination, and the
largest single applications.
"""
import collections
import pathlib
import struct
import sys

KINDS = {0: 'create', 1: 'destroy', 2: 'upload', 3: 'command', 4: 'query', 5: 'exempt'}


def records(path):
    data = pathlib.Path(path).read_bytes()
    at = 0
    while at + 16 <= len(data):
        kind, began, micros = struct.unpack_from('<IQI', data, at)
        at += 16
        if kind == 0:
            ident, width, height, fmt = struct.unpack_from('<QIII', data, at)
            at += 20
            yield kind, began, micros, {'id': ident, 'width': width, 'height': height}
        elif kind in (1, 5):
            (ident,) = struct.unpack_from('<Q', data, at)
            at += 8
            yield kind, began, micros, {'id': ident}
        elif kind == 2:
            ident, count = struct.unpack_from('<QI', data, at)
            at += 12 + 4 * count
            yield kind, began, micros, {'id': ident, 'pixels': count}
        elif kind == 3:
            if at + struct.calcsize('<IQQ10iIQQQiiQ') > len(data):
                return  # the trace ends inside a record (the process exited mid-write)
            values = struct.unpack_from('<IQQ10iIQQQiiQ', data, at)
            at += struct.calcsize('<IQQ10iIQQQiiQ')
            (command, destination, source, left, top, right, bottom, cl, ct, cr, cb, sx, sy,
             color, background, detail, background_detail, sw, sh, program) = values
            yield kind, began, micros, {'command': command, 'destination': destination, 'source': source,
                                        'area': (right - left) * (bottom - top), 'clip': (cr - cl) * (cb - ct)}
        elif kind == 4:
            if at + 24 > len(data):
                return
            ident, x, y, found, value = struct.unpack_from('<QiiII', data, at)
            at += 24
            yield kind, began, micros, {'id': ident, 'x': x, 'y': y, 'found': found, 'value': value}
        else:
            raise SystemExit(f'unknown record kind {kind} at byte {at - 16}')


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    capture = pathlib.Path(argv[1])
    path = capture / 'renderer-core.log.hit' if capture.is_dir() else capture
    window = None
    if '--window' in argv:
        i = argv.index('--window')
        window = (int(argv[i + 1]), int(argv[i + 2]))
    by_kind = collections.Counter()
    time_kind = collections.Counter()
    by_command = collections.Counter()
    time_command = collections.Counter()
    time_destination = collections.Counter()
    largest = []
    queried = collections.Counter()
    total, count = 0, 0
    for kind, began, micros, fields in records(path):
        if window and not (window[0] <= began <= window[1]):
            continue
        total += micros
        count += 1
        by_kind[KINDS[kind]] += 1
        time_kind[KINDS[kind]] += micros
        if kind == 3:
            key = fields['command']
            by_command[key] += 1
            time_command[key] += micros
            time_destination[fields['destination']] += micros
        if kind == 4:
            queried[fields['id']] += 1
        largest.append((micros, KINDS[kind], fields))
    print(f'operations={count} apply_ms={total / 1000:.1f}')
    for name, micros in time_kind.most_common():
        print(f'  {name:8s} n={by_kind[name]:8d} ms={micros / 1000:9.1f} mean_us={micros / max(1, by_kind[name]):8.1f}')
    print('commands by kind:')
    for key, micros in time_command.most_common(12):
        print(f'  kind={key:3d} n={by_command[key]:8d} ms={micros / 1000:9.1f} mean_us={micros / by_command[key]:8.1f}')
    print('commands by destination image:')
    for key, micros in time_destination.most_common(8):
        print(f'  destination={key} ms={micros / 1000:9.1f}')
    print('queried images:', dict(queried))
    print('largest applications:')
    for micros, name, fields in sorted(largest, key=lambda r: -r[0])[:12]:
        print(f'  {micros:8d} us {name} {fields}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
