"""Compare capture-on/off native endurance receipts without claiming game FPS.

Both arms must use the same DLL, scene, extent, controls and declared reservation.
Wall-clock pacing permits different completion counts; distributions are compared
by operation class, and are not a substitute for fixed-input performance replay.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    values=sorted(values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError('Missing or invalid endurance samples')
    return {'count':len(values),'median_ms':statistics.median(values),
            'p95_ms':values[max(0,math.ceil(.95*len(values))-1)],'max_ms':max(values)}


def arm(root, discard_seconds=0):
    receipt=json.loads((root/'receipt.json').read_text())
    if receipt['status']!='pass' or not receipt['inputs_unchanged'] or not receipt['binary_provenance']['current_runtime_matches_build']:
        raise ValueError('Endurance arm lacks current successful provenance')
    soak=receipt['input_soak']
    if soak['seconds']!=600:
        raise ValueError('Comparison requires real 600-second arms')
    groups={'idle':[],'native_action':[],'native_camera_phase':[],'camera_change':[]}
    previous_second=-1
    for sample in soak['samples']:
        second=int(sample['elapsed_ms']//1000)
        camera=sample['step']==0 or (sample['phase']==2 and second!=previous_second)
        kind='camera_change' if camera else ('idle','native_action','native_camera_phase')[int(sample['phase'])]
        if sample['elapsed_ms']>=discard_seconds*1000:
            groups[kind].append(sample['call_ms'])
        previous_second=second
    memory=soak['memory']
    if not memory or max(sample['elapsed_ms'] for sample in soak['samples'])<599000:
        raise ValueError('Endurance timeline ends before the final second')
    summary={'groups':{key:distribution(values) for key,values in groups.items()},
             'minimum_available_virtual_bytes':min(row['available_virtual'] for row in memory),
             'minimum_largest_free_bytes':min(row['largest_free'] for row in memory),
             'maximum_retained_bytes':max(row['retained_bytes'] for row in memory),
             'first_available_virtual_bytes':memory[0]['available_virtual'],
             'last_available_virtual_bytes':memory[-1]['available_virtual'],
             'memory_samples':len(memory)}
    return receipt,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-on',required=True,type=Path)
    parser.add_argument('--capture-off',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--discard-initial-seconds',type=int,default=0,choices=range(0,301),
                        help='Exclude the same declared warm-up/interference interval in both arms')
    args=parser.parse_args()
    on,on_summary=arm(args.capture_on,args.discard_initial_seconds);off,off_summary=arm(args.capture_off,args.discard_initial_seconds)
    if not on.get('input_recording',{}).get('closed') or 'input_recording' in off:
        raise ValueError('Need a closed recording arm and an unrecorded control')
    if on['input_recording']['storage']['stop_reason']!=1:
        raise ValueError('Recording did not close at its real duration boundary')
    # Output destinations are allowed to differ. Their inputs and settings must
    # still match; the native UI pack contents are verified independently.
    destinations={'C3X_RENDERER_INPUT_RECORD_DIR','C3X_RENDERER_TRACE_FILE',
                  'C3X_RENDERER_NATIVE_UI_PACK','C3X_RENDERER_TACTICAL_PREVIEW'}
    settings=lambda value:{key:data for key,data in value['settings'].items() if key not in destinations}
    if settings(on)!=settings(off):
        raise ValueError('Endurance controls differ')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    for filename in ('C3XRenderer.dll','native-ui.pack'):
        if digest(args.capture_on/filename)!=digest(args.capture_off/filename):
            raise ValueError('Endurance binaries/assets differ: '+filename)
    shared=set(on['inputs'])&set(off['inputs'])
    if any(on['inputs'][key]!=off['inputs'][key] for key in shared):
        raise ValueError('Endurance source or asset identity changed')
    overhead={}
    for kind in on_summary['groups']:
        a=on_summary['groups'][kind];b=off_summary['groups'][kind]
        overhead[kind]={key:a[key]-b[key] for key in ('median_ms','p95_ms')}
    report={'status':'measured','qualified_for_gameplay':False,
            'scope':'Paced native fixture capture overhead and capacity; not game FPS or fixed-input performance replay',
            'dll_sha256':digest(args.capture_on/'C3XRenderer.dll'),
            'discarded_initial_seconds':args.discard_initial_seconds,
            'known_interference':[json.loads(path.read_text()) for path in
                (args.capture_on/'comparison-notes.json',args.capture_off/'comparison-notes.json') if path.exists()],
            'capture_on':on_summary,'capture_off':off_summary,'capture_added_ms':overhead,
            'storage':on['input_recording']['storage'],
            'automatic_acceptance_threshold':None,
            'limitations':['Different wall-clock completion counts and sampled animation times',
                           'Single ordered pair; VM scheduling is not controlled',
                           'Reserved address space is a capacity probe, not Civ III heap fragmentation']}
    with args.out.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
