"""Headless actual-DLL city composition checks; never installs or runs Civ III."""
from pathlib import Path
import argparse,hashlib,json,sys,re
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from Renderer.tools.renderer_dev import windows_command_result
OUT=ROOT/'Renderer/lab/out/verification/city_fidelity'
def main():
    p=argparse.ArgumentParser();p.add_argument('--hour',type=int,default=12);p.add_argument('--zoom',type=int,choices=[64,128],default=128)
    p.add_argument('--city',default='0,3,1,1');p.add_argument('--profile',choices=['default','city-fidelity','source-fidelity-r13'],default='city-fidelity')
    p.add_argument('--replay',action='store_true');p.add_argument('--synthetic',action='store_true')
    p.add_argument('--lights-off',action='store_true');p.add_argument('--glow-off',action='store_true');a=p.parse_args()
    if not re.fullmatch('[0-4],[0-3],[0-2],[01]',a.city) or not 0<=a.hour<24:raise ValueError('invalid case')
    OUT.mkdir(parents=True,exist_ok=True)
    name=f'{a.profile}-{a.city.replace(",","-")}-h{a.hour}-z{a.zoom}'+('-replay' if a.replay else '')
    if a.synthetic:name+='-synthetic'
    if a.lights_off:name+='-lights-off'
    if a.glow_off:name+='-glow-off'
    env={'C3X_RENDERER_VISUAL_PROFILE':'' if a.profile=='default' else a.profile,'C3X_RENDERER_TRACE':'2',
        'C3X_RENDERER_TRACE_FILE':f'..\\lab\\out\\verification\\city_fidelity\\{name}.log',
        'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':'..\\..\\Renderer\\custom.custom_rendering.txt',
        'C3X_RENDERER_PREVIEW_OBJECTS':'1','C3X_RENDERER_PREVIEW_CITY':a.city,
        'C3X_RENDERER_PREVIEW_REPLAY':'1' if a.replay else '',
        'C3X_RENDERER_PREVIEW_MINIMAP':'','C3X_RENDERER_PREVIEW_EDITS':'','C3X_RENDERER_PREVIEW_ANIMATION':'',
        'C3X_RENDERER_FIDELITY_SHADOW_CONTROL':'','C3X_RENDERER_REFLECTION_CONTROL':'',
        'C3X_RENDERER_CITY_LIGHT_CONTROL':'1' if a.lights_off else '',
        'C3X_RENDERER_CITY_GLOW_CONTROL':'1' if a.glow_off else ''}
    scene='..\\lab\\.local\\verification\\world.csv';cx,cy=69,50
    if a.synthetic:
        rows=[f'{x},{y},2,2,0,0,0' for y in range(20) for x in range(y%2,20,2)]
        (OUT/'dry-city.csv').write_text('C3X_BIQ_TERRAIN_V3,20,20,200\n'+'\n'.join(rows)+'\n')
        scene='..\\lab\\out\\verification\\city_fidelity\\dry-city.csv';cx=cy=10
    cmd=' && '.join(f'set "{k}={v}"' for k,v in env.items())
    cmd+=f' && build\\biq_preview.exe build\\candidate\\C3XRenderer.dll ..\\.. ..\\default.custom_rendering.txt {scene} ..\\lab\\out\\verification\\city_fidelity\\{name}.bmp 640 480 {cx} {cy} {a.zoom} {a.hour}'
    pins={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [ROOT/'Renderer/native/build/candidate/C3XRenderer.dll',ROOT/'Renderer/packs/CityCompositionRuntime/city.bin',*sorted((ROOT/'Renderer/native/city_fidelity').glob('*.hlsl'))]}
    r=windows_command_result('Renderer/native',cmd);r.pop('cwd',None);r.pop('host',None)
    output=r['output_tail'];r['inputs_sha256']=pins
    if '0 fallback, output=' not in output or 'FAIL' in output:r['status']='fail'
    if a.replay:
        parity=re.search(r'PICKUP pixel parity: changed=(\d+) error=(\d+) bytes=(\d+)',output)
        if not parity:r['status']='fail'
        else:
            changed,error,size=map(int,parity.groups());r['parity']={'changed':changed,'error':error,'bytes':size}
            if changed>size//4000 or error>size//100:r['status']='fail'
    log=OUT/(name+'.log')
    r['composition_diagnostics']=list(dict.fromkeys(line.split('stage=city-composition',1)[1] for line in log.read_text(errors='replace').splitlines() if 'stage=city-composition ' in line)) if log.exists() else []
    if a.profile!='source-fidelity-r13' and not r['composition_diagnostics']:r['status']='fail'
    r['output_tail']='\n'.join(line for line in output.splitlines() if not re.search(r'[A-Z]:\\|/Users/|\\\\Mac\\',line))
    (OUT/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
    print(name,r['status'],r['composition_diagnostics'])
    return 0 if r['status']=='pass' else 1
if __name__=='__main__':raise SystemExit(main())
