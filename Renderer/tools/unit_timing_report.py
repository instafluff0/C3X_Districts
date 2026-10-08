"""Unit move timing from a scripted capture (unit-turn, unit-motion).

Pairs each posted move key (key-events.json, QPC) with the next unit-arrival
trace from the helper (trace level 1+). Reports, per step:
  key_to_move_ms   key posted -> Civ III's move event (FLC move target)
  shown_lag_ms     move event -> first scene sample holding the step
  travel_ms        displayed travel
  end_vs_commit_ms displayed arrival minus Civ III's confirmation
and each reveal-shown line (Civ III visibility capture -> first shown frame).

Usage: unit_timing_report.py CAPTURE_DIR
"""
import json
import pathlib
import re
import sys

MOVE_KEYS = {0x61, 0x62, 0x63, 0x64, 0x66, 0x67, 0x68, 0x69}
FIELD = re.compile(r'(\w+)=(-?[\d.]+)')


def main(argv):
    capture = pathlib.Path(argv[1])
    keys_path = capture / 'key-events.json'
    keys = []
    if keys_path.exists():
        raw = json.loads(keys_path.read_text())
        keys = [e for e in raw['events'] if int(e['key']) in MOVE_KEYS]
    log = (capture / 'renderer-core.log.x64').read_text(errors='replace').splitlines()
    arrivals, reveals = [], []
    for line in log:
        if 'stage=unit-arrival ' in line:
            arrivals.append(dict(FIELD.findall(line.split('stage=unit-arrival', 1)[1])))
        elif 'stage=reveal-shown ' in line:
            reveals.append(dict(FIELD.findall(line.split('stage=reveal-shown', 1)[1])))
    # QPC runs at 24 MHz in the VM; derive it from two stamped lines.
    stamped = [(int(m[1]), float(m[2])) for m in
               (re.search(r'qpc=(\d+) ms=([\d.]+)', l) for l in log) if m]
    rate = (stamped[-1][0] - stamped[0][0]) / ((stamped[-1][1] - stamped[0][1]) / 1000.)
    used = set()
    for a in arrivals:
        event = int(a.get('event_qpc', 0))
        key = max((k for k in keys if int(k['qpc']) <= event and id(k) not in used),
                  key=lambda k: int(k['qpc']), default=None) if event else None
        if key is not None:
            used.add(id(key))
        key_ms = f"{1000. * (event - int(key['qpc'])) / rate:.1f}" if key else '-'
        print(f"unit {a.get('id')}: key_to_move_ms={key_ms} shown_lag_ms={a.get('shown_lag_ms', '-')} "
              f"travel_ms={a.get('travel_ms')} end_vs_commit_ms={a.get('end_vs_commit_ms')}")
    for r in reveals:
        print(f"reveal: cells={r.get('cells')} capture_age_ms={r.get('capture_age_ms')}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
