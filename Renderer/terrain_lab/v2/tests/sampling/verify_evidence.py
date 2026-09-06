#!/usr/bin/env python3
"""Check preserved Q1 evidence. --accept also enforces unresolved owner gates."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[5]
AUDIT=ROOT/'Renderer/terrain_lab/v2/audits/sampling'
OUT=AUDIT/'out/Q1/Q1-sampling'
REPORTS=['c006mixedab','c007pan','c011real-final','c012holdout-final',
         'c013off-mixed','c014off-holdout','c015animation0','c016animation1',
         'c017animation2','c018animation3','c019real-pan','c020linear-mixed',
         'c021linear-matrix','c022linear-holdout','c023linear-pan','c024linear-animation1',
         'c025linear-off','c026linear-animation0','c027linear-animation2','c028linear-animation3',
         'c030q2-linear-on','c031q2-linear-off','c032linear-off-holdout','c033validity-fixed']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--accept',action='store_true');a=ap.parse_args()
    dedup=json.loads((OUT/'repeat-deduplication.json').read_text())
    removed={r['removed']:r for r in dedup['files']};count=0
    for name in REPORTS:
        report=json.loads((OUT/name/'report.json').read_text())
        if report.get('source_changed_during_run'):raise ValueError('source changed during render: '+name)
        fixture=ROOT/report['fixture']
        if digest(fixture)!=report['fixture_sha256']:raise ValueError('fixture drift: '+name)
        for e in report['outputs']:
            image=ROOT/e['image'];repeat=Path(str(image)+'.repeat1.bmp')
            if digest(image)!=e['sha256'] or not e['repeat_identical']:raise ValueError('image/repeat evidence drift')
            if repeat.exists():
                if digest(repeat)!=e['sha256']:raise ValueError('repeat mismatch')
            else:
                key=repeat.relative_to(ROOT).as_posix()
                if removed[key]['sha256']!=e['sha256']:raise ValueError('missing repeat or exact dedup proof')
            if e['internal_size'][0]//e['output_size'][0] not in (1,2,4):raise ValueError('invalid scale')
            count+=1
    for name in ['c007pan','c019real-pan','c023linear-pan']:
        report=json.loads((OUT/name/'report.json').read_text())
        for variant in {r['variant'] for r in report['outputs']}:
            for zoom in [1,2]:
                rows=[r for r in report['outputs'] if r['variant']==variant and r['zoom']==zoom]
                if rows[0]['sha256']!=rows[-1]['sha256']:raise ValueError('pan origin drift')
    print('PASS preserved Q1 images/repeats/fixtures and pan return:',count,'variants')
    if a.accept:
        handoff=json.loads((AUDIT/'candidate_v1.json').read_text())
        pending=[r['id'] for r in handoff['acceptance_gates'] if r['state']!='passed']
        if pending:raise ValueError('acceptance remains pending: '+', '.join(pending))

if __name__=='__main__':main()
