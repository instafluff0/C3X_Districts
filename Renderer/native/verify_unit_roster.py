"""Exercise the expanded unit candidate headlessly through the actual DLL."""
import argparse
import json
import re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from Renderer.tools.renderer_dev import windows_command_result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hour',type=int,choices=range(24),default=12)
    parser.add_argument('--unit',help='Verify only this existing unit key after an isolated data correction.')
    args=parser.parse_args()
    if args.unit and not re.fullmatch(r'PRTO_[A-Za-z0-9_-]+',args.unit):raise ValueError('unsafe unit key')
    pack=Path('Renderer/packs/UnitRosterRuntimeCandidate')
    bindings=json.loads((pack/'bindings.json').read_text())
    numbers={'idle':1,'move':2,'attack':3,'death':6,'fortify':7,'fidget':8,'victory':9,
             'capture':10,'fortress':11,'build':12,'road':13,'mine':14,'irrigate':15,'jungle':16,'forest':17,'plant':18}
    rows=[]
    for i in range(bindings['unit_count']):
        u=bindings[f'unit{i}']
        if u['complete']!=1:raise ValueError(f'incomplete candidate unit {i}')
        mask=sum(1<<n for name,n in numbers.items() if name in u)
        if 'attack' in u:mask |= (1<<4)|(1<<5)
        for key in range(u['key_count']):
            if not args.unit or u[f'key{key}']==args.unit:rows.append(f"{u[f'key{key}']} {mask}")
    if not rows:raise ValueError('requested key is absent from candidate bindings')
    out=Path('Renderer/verification/animation/roster');out.mkdir(parents=True,exist_ok=True)
    cases=f'cases-{args.unit}.txt' if args.unit else 'cases.txt'
    (out/cases).write_text('\n'.join(rows)+'\n')
    hour=args.hour;name=f'roster-{args.unit}-h{hour}' if args.unit else f'roster-h{hour}'
    settings={'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':'..\\..\\Renderer\\custom.custom_rendering.txt',
        'C3X_RENDERER_VISUAL_PROFILE':'pickup-r1','C3X_RENDERER_PREVIEW_ANIMATION':'1',
        'C3X_RENDERER_PREVIEW_UNITS':'1','C3X_RENDERER_TRACE':'2',
        'C3X_RENDERER_UNIT_PACK':'UnitRosterRuntimeCandidate',
        'C3X_RENDERER_UNIT_CASES':f'..\\verification\\animation\\roster\\{cases}',
        'C3X_RENDERER_TRACE_FILE':f'..\\verification\\animation\\roster\\{name}.log'}
    command=' && '.join(f'set "{k}={v}"' for k,v in settings.items())
    command+=' && build\\biq_preview.exe build\\candidate\\C3XRenderer.dll ..\\.. ..\\..\\Renderer\\default.custom_rendering.txt ..\\verification\\pickup\\world.csv '
    command+=f'..\\verification\\animation\\roster\\{name}.bmp 960 640 75 39 128 {hour}'
    result=windows_command_result('Renderer/native',command);result.pop('cwd',None)
    text=result.get('output_tail','')
    if 'ROSTER complete' not in text or 'failures=0 status=pass' not in text or 'UNIT post-draw terrain parity: pass' not in text or 'FAIL' in text:
        result['status']='fail'
    trace=(out/f'{name}.log').read_text()
    residency=[int(x) for x in re.findall(r'stage=unit-payload .*?resident_bytes=(\d+)',trace)]
    sprite_bytes=[int(x) for x in re.findall(r'stage=unit-body .*?cache_bytes=(\d+)',trace)]
    result['memory']={'payload_peak_bytes':max(residency,default=0),'payload_budget_bytes':96*1024*1024,
        'sprite_peak_bytes':max(sprite_bytes,default=0),'sprite_budget_bytes':8*1024*1024,
        'payload_load_batches':len(residency),'sprite_cache_hits':trace.count('cache_hit=1')}
    if not residency or max(residency)>96*1024*1024 or not sprite_bytes or max(sprite_bytes)>8*1024*1024 or not trace.count('cache_hit=1'):
        result['status']='fail'
    result['unit_keys']=len(rows)
    (out/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],text[-2500:])
    return result['status']!='pass'


if __name__=='__main__':raise SystemExit(main())
