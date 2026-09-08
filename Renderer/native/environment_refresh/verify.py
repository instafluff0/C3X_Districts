#!/usr/bin/env python3
"""Headless actual-DLL checks. Never runs INSTALL.bat or starts Civ III."""
from pathlib import Path
import argparse,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Renderer.tools.renderer_dev import windows_command_result
OUT=ROOT/'Renderer/lab/out/verification/environment_refresh'
def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['first','default','coast','coast-control','matrix','objects','animation','replay','edits','minimap','control']);p.add_argument('--resume',action='store_true');p.add_argument('--sky-only',action='store_true');a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cases=[('inland',69,50,128,12)]
    if a.mode in ['coast','coast-control']:cases=[('coast',85,38,128,12)]
    if a.mode=='matrix':cases=[(region,x,y,z,h) for region,x,y in [('inland',69,50),('coast',85,38),('wrap',0,50)] for z in [128,64] for h in [12,18,0,6]]
    if a.sky_only:a.mode='coast-sky'
    records=[]
    old=json.loads((OUT/(a.mode+'.json')).read_text()) if a.resume and (OUT/(a.mode+'.json')).exists() else []
    dll_hash=hashlib.sha256((ROOT/'Renderer/native/build/candidate/C3XRenderer.dll').read_bytes()).hexdigest()
    for region,x,y,z,h in cases:
        name=f'{a.mode}-{region}-z{z}-h{h}'
        previous=next((r for r in old if r['name']==name and r['status']=='pass' and r.get('dll_sha256')==dll_hash),None)
        if previous:
            records.append(previous);continue
        env={'C3X_RENDERER_VISUAL_PROFILE':'' if a.mode=='default' else 'environment-refresh','C3X_RENDERER_TRACE':'2',
             'C3X_RENDERER_TRACE_FILE':f'..\\lab\\out\\verification\\environment_refresh\\{name}.log',
             'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':'..\\..\\Renderer\\custom.custom_rendering.txt',
             'C3X_RENDERER_FIDELITY_SHADOW_CONTROL':'1' if a.mode in ['control','coast-control'] else '',
             'C3X_RENDERER_REFLECTION_CONTROL':'1' if a.sky_only else '',
             'C3X_RENDERER_PREVIEW_REPLAY':'1' if a.mode in ['replay','minimap'] else '',
             'C3X_RENDERER_PREVIEW_MINIMAP':'1' if a.mode=='minimap' else '',
             'C3X_RENDERER_PREVIEW_EDITS':'1' if a.mode=='edits' else ''}
        env['C3X_RENDERER_PREVIEW_OBJECTS']='1' if a.mode=='objects' else ''
        # Animation removal intentionally clears resources; it cannot share
        # the static-object check that requires resource ownership at the end.
        env['C3X_RENDERER_PREVIEW_ANIMATION']='1' if a.mode=='animation' else ''
        cmd=' && '.join(f'set "{k}={v}"' for k,v in env.items())
        cmd+=f' && build\\biq_preview.exe build\\candidate\\C3XRenderer.dll ..\\.. ..\\default.custom_rendering.txt ..\\lab\\.local\\verification\\world.csv ..\\lab\\out\\verification\\environment_refresh\\{name}.bmp 960 640 {x} {y} {z} {h}'
        print('Checking '+name,flush=True);r=windows_command_result('Renderer/native',cmd);r.pop('cwd',None);r.pop('host',None)
        output=r['output_tail'];r['name']=name;r['dll_sha256']=dll_hash
        if '0 fallback, output=' not in output or 'FAIL' in output:r['status']='fail'
        if a.mode in ['replay','minimap','edits']:
            m=re.search(r'PICKUP (?:edit )?pixel parity: changed=(\d+) error=(\d+) bytes=(\d+)',output)
            if m:
                changed,error,size=map(int,m.groups());r['parity']={'changed':changed,'error':error,'bytes':size}
                if changed>size//4000 or error>size//100:r['status']='fail'
            else:r['status']='fail'
        if (OUT/(name+'.bmp')).exists():r['raw_sha256']=hashlib.sha256((OUT/(name+'.bmp')).read_bytes()).hexdigest()
        # Remove machine-specific compiler/transport diagnostics from retained
        # metadata. The complete opt-in debugger log remains local.
        r['output_tail']='\n'.join(line for line in output.splitlines() if not re.search(r'[A-Z]:\\|/Users/|\\\\Mac\\',line))
        records.append(r);(OUT/(a.mode+'.json')).write_text(json.dumps(records,indent=2)+'\n')
        if r['status']!='pass':return 1
    return 0
if __name__=='__main__':raise SystemExit(main())
