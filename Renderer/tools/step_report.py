"""Break each camera job of a scripted capture into its phases.

Usage: step_report.py CAPTURE_DIR [--all]

CAPTURE_DIR needs a trace-level-2 helper log (renderer-core.log.x64, from
tools/run_scripted_game_test.ps1 -ProfileRenderer) and renderer.log. With input
tracing (the `near` scenario), each Civ III edge-scroll request is matched to
the job that served it and to its adoption.

One row per camera job (render-begin to camera-complete), all times in ms:
  wait     request to render-begin (the latest edge-scroll before the job)
  queued   the camera-begin record's wait in the bridge publication queue
  send     that record's synchronous service (delivery to the helper)
  job      render-begin to camera-complete
  turns    native UI service turns inside the job (camera-service-turn ms)
  mesh     ground + features + cliffs + terrain prep + upload (mesh-phases)
  shadow   fresh-shadow-build refresh + body + proofs + casters + draw
  prepare  fresh-scene-phases prepare (includes city lights and shadows)
  static   fresh-scene-phases static (strip rasters)
  reflect  fresh-scene-phases reflection
  topo     navigation-phases topology
  built    tiles built for the job
  native   the next native map pass (map-complete total_ms) after the job
  adopt    request to native-handoff
By default only jobs during the near scenario's scroll segments are listed.
"""
import json
import pathlib
import re
import statistics
import sys

STAGES = ('render-begin', 'camera-complete', 'camera-service-turn', 'mesh-phases', 'fresh-shadow-build',
          'fresh-scene-phases', 'navigation-phases')
FIELD = re.compile(r'(\w+)=([-\d.]+)')


def number(fields, key):
    try:
        return float(fields.get(key, 0))
    except ValueError:
        return 0.0


def jobs(capture):
    capture = pathlib.Path(capture)
    frequency = 24e6
    cadence = capture / 'cadence.json'
    if cadence.exists():
        frequency = float(json.loads(cadence.read_text()).get('qpc_frequency', frequency))
    helper = []
    for line in (capture / 'renderer-core.log.x64').read_text(errors='replace').splitlines():
        match = re.search(r'qpc=(\d+) .*?stage=([a-z0-9-]+) ?(.*)', line)
        if match and match[2] in STAGES:
            helper.append((int(match[1]), match[2], dict(FIELD.findall(match[3]))))
    requests, natives, handoffs, begins = [], [], [], []
    for line in (capture / 'renderer.log').read_text(errors='replace').splitlines():
        qpc = re.search(r'qpc=(\d+)', line)
        if not qpc:
            continue
        if 'stage=edge-scroll ' in line:
            requests.append(int(qpc[1]))
        elif 'stage=map-complete ' in line:
            natives.append((int(qpc[1]), float(re.search(r'total_ms=([\d.]+)', line)[1])))
        elif 'stage=native-handoff ' in line:
            handoffs.append(int(qpc[1]))
        elif 'operation=camera-begin ' in line:
            timing = re.search(r'queue_ms=([\d.]+) service_ms=([\d.]+)', line)
            begins.append((int(qpc[1]), float(timing[1]), float(timing[2])))
    rows, current = [], None
    for qpc, stage, fields in helper:
        if stage == 'render-begin':
            current = {'begin': qpc, 'turns': 0.0, 'mesh': 0.0, 'shadow': 0.0, 'prepare': 0.0, 'static': 0.0,
                       'reflect': 0.0, 'topo': 0.0, 'built': 0}
        elif current is None:
            continue
        elif stage == 'camera-service-turn':
            current['turns'] += number(fields, 'ms')
            current['built'] = max(current['built'], int(number(fields, 'built')))
        elif stage == 'mesh-phases':
            current['mesh'] += sum(number(fields, k) for k in ('ground_ms', 'features_ms', 'cliffs_ms', 'terrain_prep_ms', 'upload_ms'))
        elif stage == 'fresh-shadow-build':
            current['shadow'] += sum(number(fields, k) for k in ('refresh_ms', 'body_ms', 'proofs_ms', 'casters_ms', 'draw_ms'))
        elif stage == 'fresh-scene-phases':
            current['prepare'] += number(fields, 'prepare')
            current['static'] += number(fields, 'static')
            current['reflect'] += number(fields, 'reflection')
        elif stage == 'navigation-phases':
            current['topo'] += number(fields, 'topology_ms')
        elif stage == 'camera-complete':
            current['end'] = qpc
            asked = [r for r in requests if r <= current['begin']]
            current['request'] = asked[-1] if asked else None
            sent = [b for b in begins if b[0] <= current['begin']]
            current['queued'], current['send'] = (sent[-1][1], sent[-1][2]) if sent else (None, None)
            later = [n for n in natives if n[0] > qpc]
            current['native'] = later[0][1] if later else None
            adopted = [h for h in handoffs if h > qpc]
            current['adopted'] = adopted[0] if adopted else None
            rows.append(current)
            current = None
    return frequency, rows


def windows(capture, frequency):
    path = pathlib.Path(capture) / 'mouse-events.json'
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    events = [e['qpc'] for e in (raw['events'] if isinstance(raw, dict) else raw)]
    if len(events) < 25:
        return None
    return [(events[0], events[1]), (events[2], events[3]), (events[13], events[14]), (events[16], events[17])]


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip())
        return 2
    capture = argv[1]
    frequency, rows = jobs(capture)
    spans = None if '--all' in argv else windows(capture, frequency)
    ms = lambda ticks: ticks / frequency * 1000
    keys = ('wait', 'queued', 'send', 'job', 'turns', 'mesh', 'shadow', 'prepare', 'static', 'reflect', 'topo', 'native', 'adopt')
    print('  '.join(f'{k:>7}' for k in keys) + '    built')
    summary = {k: [] for k in keys}
    for row in rows:
        if spans and not any(a <= row['begin'] <= b for a, b in spans):
            continue
        values = {
            'wait': ms(row['begin'] - row['request']) if row['request'] else None,
            'queued': row['queued'], 'send': row['send'],
            'job': ms(row['end'] - row['begin']),
            'turns': row['turns'], 'mesh': row['mesh'], 'shadow': row['shadow'], 'prepare': row['prepare'],
            'static': row['static'], 'reflect': row['reflect'], 'topo': row['topo'], 'native': row['native'],
            'adopt': ms(row['adopted'] - row['request']) if row['adopted'] and row['request'] else None,
        }
        for k, v in values.items():
            if v is not None:
                summary[k].append(v)
        print('  '.join(f'{v:7.1f}' if v is not None else '      -' for v in values.values()) + f'  {row["built"]:7d}')
    print('p50: ' + '  '.join(f'{k}={statistics.median(v):.1f}' for k, v in summary.items() if v))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
