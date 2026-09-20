"""Join completed native visual requests to their exact renderer trace intervals."""
import argparse
import json
import statistics
from pathlib import Path
from Renderer.native.analyze_navigation_run import fields, distribution


def analyze(log, trace):
    samples=[fields(line) for line in log.splitlines() if line.startswith('VISUAL_SAMPLE ')]
    rows=[fields(line) for line in trace.splitlines() if 'qpc=' in line and 'stage=' in line]
    stages=('frame','shared-scene-surface','scene-reflection','scene-waves','scene-water',
            'material-submission','world-view-submission','submission-casters','visual-unit-scene')
    intervals=[]
    for sample in samples:
        begin,end=int(sample['begin_qpc']),int(sample['end_qpc'])
        if begin>end:raise ValueError('Reversed visual request endpoints')
        selected=[row for row in rows if begin<=int(row['qpc'])<=end]
        intervals.append({stage:[row for row in selected if row['stage']==stage] for stage in stages})
    def all_zero(stage, keys):
        return present(stage) and all(all(int(row.get(key,-1))==0 for key in keys)
                   for interval in intervals for row in interval[stage])
    def present(stage):
        return bool(intervals) and all(interval[stage] for interval in intervals)
    seen={stage:sum(len(interval[stage]) for interval in intervals) for stage in stages}
    proof={
        'static_world_builds_zero':seen['frame']>0 and all_zero('frame',('built','upload_bytes')),
        'static_scene_draws_zero':seen['shared-scene-surface']>0 and all_zero('shared-scene-surface',('static_selected','readbacks')),
        'reflection_builds_zero':seen['scene-reflection']>0 and all_zero('scene-reflection',('built',)),
        'wave_uploads_and_builds_zero':seen['scene-waves']>0 and all_zero('scene-waves',('upload_bytes','cells_built')),
        'water_uploads_zero':seen['scene-water']>0 and all_zero('scene-water',('geometry_upload_bytes',)),
        'material_selection_reused':present('material-submission') and all(int(row.get('reused',0))==1
            for interval in intervals for row in interval['material-submission']),
        'static_caster_collection_zero':present('shared-scene-surface') and seen['submission-casters']==0,
    }
    def timing(key):
        values=[float(row[key]) for row in samples]
        return {**distribution(values),'mean_ms':statistics.mean(values) if values else None}
    return {'samples':len(samples),'request':timing('request_ms'),
            'desktop':timing('desktop_ms'),
            'proof':proof,'stage_counts':seen,
            'unit_totals':{key:sum(int(row.get(key,0)) for interval in intervals for row in interval['visual-unit-scene'])
                           for key in ('body_draws','body_reuses','body_copies')},
            'intervals':intervals,
            'scope':'Completed visual API requests; desktop completion is not physical scanout. Missing stages do not prove zero work.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=analyze((args.directory/'test.log').read_text(),(args.directory/'renderer.log').read_text())
    receipt=json.loads((args.directory/'receipt.json').read_text())
    result['receipt_status']=receipt['status']
    result['complete_verified_run']=receipt['status']=='pass' and receipt.get('inputs_unchanged',False) and receipt.get('trace_coverage',{}).get('complete',False)
    result['expected_samples']=int(receipt['settings'].get('C3X_RENDERER_VISUAL_FRAMES',30))
    result['complete_verified_run']&=result['samples']==result['expected_samples']
    (args.directory/'visual-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='intervals'},indent=2))


if __name__=='__main__':main()
