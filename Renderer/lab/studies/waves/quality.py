"""Freeze a before image and evaluate the current wave material/candidate in isolation."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from Renderer import renderer
from Renderer.lab.platform import run_native_fixture

ROOT=renderer.ROOT
OUT=ROOT/'Renderer/lab/out/waves-quality'
SNAPSHOT=OUT/'snapshot'


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('city_fidelity','environment_refresh','source_fidelity','render_core'):
        for path in (ROOT/'Renderer/native'/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.hlsl','.h','.cso'):
                target=SNAPSHOT/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
    for path in (ROOT/'Renderer/native').iterdir():
        if path.is_file() and path.suffix in ('.hlsl','.h'):
            shutil.copy2(path,SNAPSHOT/path.relative_to(ROOT))
    names={'NaturalFidelityRuntime','CityCompositionRuntime','TerrainProfileR1','CoastalWavesRuntime'}
    for name in ('default.custom_rendering.txt','custom.custom_rendering.txt'):
        path=ROOT/'Renderer'/name
        names.update(re.findall(r'path = mod:Renderer\\packs\\([^\s]+)',path.read_text()))
        shutil.copy2(path,SNAPSHOT/'Renderer'/name)
    names.update(name.decode() for name in re.findall(rb'Renderer/packs/([A-Za-z0-9_]+)/',
        (ROOT/'Renderer/packs/CityCompositionRuntime/city.bin').read_bytes()))
    for name in sorted(names):
        target=SNAPSHOT/'Renderer/packs'/name
        if not target.exists():shutil.copytree(ROOT/'Renderer/packs'/name,target)
    shutil.copy2(ROOT/'Renderer/bin/C3XRenderer.dll',OUT/'C3XRenderer.dll')
    shutil.copy2(ROOT/'Renderer/lab/.cache/native_preview.exe',OUT/'native_preview.exe')
    shader=(ROOT/'Renderer/native/city_fidelity/hydrology.hlsl').read_bytes()
    (OUT/'baseline.hlsl').write_bytes(shader)
    (OUT/'snapshot.json').write_text(json.dumps({'dll_sha256':renderer.checksum(OUT/'C3XRenderer.dll'),
        'preview_sha256':renderer.checksum(OUT/'native_preview.exe'),
        'shader_sha256':hashlib.sha256(shader).hexdigest(),'packs':sorted(names)},indent=2)+'\n')


def render(variant,case='mixed',hour=12,zoom=128,dll=None,fragment=None,sequence=0):
    dll=dll or OUT/'C3XRenderer.dll'
    output=OUT/variant/f'{case}-h{hour}-z{zoom}';output.mkdir(parents=True,exist_ok=True)
    renderer.scene('ocean-waves',case,output/'scene.csv')
    shader=(OUT/'baseline.hlsl').read_text()
    if variant!='baseline':
        marker='// Generic optional coastal breaker material.'
        assert shader.count(marker)==1
        shader=shader[:shader.index(marker)]+(fragment or ROOT/'Renderer/lab/shared/shaders/hydrology/coastal_waves.hlsl').read_text()
    (SNAPSHOT/'Renderer/native/city_fidelity/hydrology.hlsl').write_text(shader)
    (output/'hydrology.hlsl').write_text(shader)
    def win(path):return '..\\..\\'+path.relative_to(ROOT).as_posix().replace('/','\\')
    run_id=uuid.uuid4().hex
    env={'C3X_RENDERER_VISUAL_PROFILE':'','C3X_RENDERER_WAVES':'',
        'C3X_RENDERER_TRACE':'2','C3X_RENDERER_TRACE_FILE':win(output/'renderer.log'),
        'C3X_RENDERER_PREVIEW_OBJECTS':'','C3X_RENDERER_PREVIEW_REPLAY':'',
        'C3X_RENDERER_PREVIEW_EDITS':'','C3X_RENDERER_PREVIEW_ANIMATION':'','C3X_RENDERER_PREVIEW_UNITS':'',
        'C3X_LAB_UNIT_STUDY':'','C3X_LAB_OBJECT_STUDY':'','C3X_LAB_WATER_STUDY':'',
        'C3X_LAB_WAVE_STUDY':case,'C3X_LAB_WAVE_SEQUENCE':str(sequence),'C3X_LAB_PID_FILE':win(output/'process.txt'),'C3X_LAB_RUN_ID':run_id,
        'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':win(SNAPSHOT/'Renderer/custom.custom_rendering.txt')}
    command='\n'.join(f'set "{key}={value}"' for key,value in env.items())
    command+=f'\n"{win(OUT/"native_preview.exe")}" "{win(dll)}" "{win(SNAPSHOT)}" "{win(SNAPSHOT/"Renderer/default.custom_rendering.txt")}" "{win(output/"scene.csv")}" "{win(output/"frame.bmp")}" 960 640 13 17 {zoom} {hour}'
    batch=output/'render.bat'
    batch.write_text('@echo off\nsetlocal\n'+command+f' > "{win(output/"native.log")}" 2>&1\nset "C3X_LAB_EXIT=%errorlevel%"\n> "{win(output/"completion.txt")}" echo {run_id} %C3X_LAB_EXIT%\nexit /b %C3X_LAB_EXIT%\n')
    result=run_native_fixture(output,f'call "{win(batch)}"',run_id)
    if result['status']!='pass' or 'PASS coastal wave lifecycle' not in result['output_tail']:
        raise ValueError(str(result))
    (output/'result.json').write_text(json.dumps(dict(result,variant=variant,
        shader_sha256=hashlib.sha256(shader.encode()).hexdigest(),dll_sha256=renderer.checksum(dll)),indent=2)+'\n')
    return output



def review(candidate='refined'):
    from PIL import Image, ImageChops, ImageDraw
    baseline=OUT/'baseline/mixed-h12-z128'
    refined=OUT/candidate/'mixed-h12-z128'
    control=ImageChops.difference(Image.open(baseline/'frame.bmp.wave-off.bmp'),
                                  Image.open(refined/'frame.bmp.wave-off.bmp'))
    if control.getbbox():raise ValueError('Wave-off controls differ; comparison is not isolated')
    context=Image.new('RGB',(1440,664),(20,28,31))
    detail=Image.new('RGB',(960,592),(20,28,31))
    for i,(folder,label) in enumerate(((baseline,'Previous production'),(refined,'Candidate: '+candidate.replace('-',' ')))):
        frame=Image.open(folder/'frame.bmp.wave-9.bmp')
        context.paste(frame.crop((200,0,920,640)),(720*i,24))
        detail.paste(frame.crop((340,260,580,540)).resize((480,560)),(480*i,24))
        ImageDraw.Draw(context).text((720*i+12,7),label,fill='white')
        ImageDraw.Draw(detail).text((480*i+12,7),label,fill='white')
    context.save(OUT/'comparison.png');detail.save(OUT/'detail.png')
    print(OUT/'comparison.png');print(OUT/'detail.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','baseline','refined','review'))
    parser.add_argument('--case',default='mixed',choices=('mixed','beach','rocky-control'))
    parser.add_argument('--hour',type=int,default=12)
    parser.add_argument('--zoom',type=int,default=128)
    parser.add_argument('--dll',type=Path,help='Optional frozen candidate DLL under the checkout')
    parser.add_argument('--output',type=Path,help='Separate study directory under the checkout')
    parser.add_argument('--shader',type=Path,help='Optional experimental wave fragment')
    parser.add_argument('--label',help='Experiment name for the output folder')
    parser.add_argument('--sequence',type=int,default=0,help='Quarter-second motion samples, maximum 240')
    args=parser.parse_args()
    if args.output:OUT=args.output.resolve();SNAPSHOT=OUT/'snapshot'
    if args.action=='prepare':prepare()
    elif args.action=='review':review(args.label or 'refined')
    else:print(render(args.label or args.action,args.case,args.hour,args.zoom,args.dll.resolve() if args.dll else None,args.shader.resolve() if args.shader else None,args.sequence))
