"""Render bounded city/border examples with the actual Renderer64 DLL."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.lab.platform import windows_root, native_command_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', default='Renderer/packs/CityCompositionRuntime')
    parser.add_argument('--dll', default='Renderer/bin/renderer64/C3XRenderer_x64.dll')
    parser.add_argument('--tile-width', type=int, choices=[128,192,256,384,512])
    parser.add_argument('--hour', type=float, default=12)
    parser.add_argument('--shaders', default='Renderer/packs/Renderer64CutoverControl')
    parser.add_argument('--case', choices=['american-town', 'roman-city', 'asian-metropolis', 'borders-relief'])
    args = parser.parse_args()
    out = ROOT/'Renderer/native/build/cities-borders/examples'
    out.mkdir(parents=True, exist_ok=True)
    scene = out/'world.csv'
    rows = []
    for y in range(24):
        for x in range(y%2,24,2):
            relief = 6 if (x,y) in [(11,13),(12,12),(13,13)] else 7 if (x,y) in [(8,6),(9,7),(10,6)] else 2
            rows.append(f'{x},{y},2,{relief},0,0,0')
    scene.write_text('C3X_BIQ_TERRAIN_V3,24,24,288\n'+'\n'.join(rows)+'\n')
    cases = {'american-town':(0,0,0,0,1,0,256), 'roman-city':(2,1,1,1,0,1,256),
             'asian-metropolis':(4,3,2,1,0,2,256), 'borders-relief':(1,0,0,0,1,1,192)}
    root = windows_root()
    def win(path): return str(root/path.relative_to(ROOT).as_posix())
    for label, values in cases.items():
        if args.case and args.case != label: continue
        name=label if args.hour==12 else f'{label}-h{args.hour:g}'
        if args.tile_width:name+=f'-z{args.tile_width}'
        case=out/name
        case.mkdir(exist_ok=True)
        culture,era,size,capital,wall,seed,tile = values
        tile=args.tile_width or tile
        options = {'C3X_RENDERER_VISUAL_PROFILE':'city-fidelity', 'C3X_RENDERER_TRACE':'2',
                   'C3X_RENDERER_TRACE_FILE':win(case/'renderer.log'),
                   'C3X_RENDERER_SHARED_SCENE_SURFACE':'1', 'C3X_RENDERER_WATER_MOTION':'1',
                   'C3X_RENDERER_WAVES':'1', 'C3X_SANDBOX_SHADOW_PATCHES':'1',
                   'C3X_SANDBOX_WHOLE_WORLD':'1', 'C3X_SANDBOX_UNITS':'1',
                   'C3X_RENDERER_PREVIEW_UNITS':'1',
                   'C3X_RENDERER_CITY_PACK':args.pack,
                   'C3X_RENDERER_SHADER_SOURCE_ROOT':win(ROOT/args.shaders),
                   'C3X_RENDERER_CITY_BORDER_FIXTURE':f'10,10,{culture},{era},{size},{capital},{wall},{seed}',
                   'C3X_SANDBOX_REPLAY_CLIP':'1','C3X_SANDBOX_CLIP_FRAMES':'1',
                   'C3X_SANDBOX_CAPTURE':win(case/'frame.bmp'),
                   'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS':win(ROOT/'Renderer/custom.custom_rendering.txt')}
        paths = [ROOT/'Renderer/sandbox/out/client_x64.exe',ROOT/args.dll,ROOT,
                 ROOT/'Renderer/default.custom_rendering.txt',scene,case/'initial.bmp']
        command = ' '.join('"'+win(p)+'"' for p in paths)+f' 1280 800 10 10 {tile} {args.hour}'
        (case/'run.bat').write_text('@echo off\nsetlocal\n'+'\n'.join(f'set "{k}={v}"' for k,v in options.items())+'\n'+command+'\nexit /b %errorlevel%\n')
        result = native_command_result('Renderer/native',f'call "{win(case/"run.bat")}"',timeout_seconds=300)
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        result.update(client_sha256=sha(paths[0]),dll_sha256=sha(ROOT/args.dll),pack_sha256=sha(ROOT/args.pack/'city.bin'),scene_sha256=sha(scene),shader_sha256={p.relative_to(ROOT/args.shaders).as_posix():sha(p) for p in (ROOT/args.shaders).rglob("*.hlsl")})
        (case/'capture.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['status'] != 'pass' or 'fallback=0' not in result['output_tail'] or not (case/'frame.bmp').exists():
            raise RuntimeError('Capture failed: '+label)
        print('CAPTURE',label,flush=True)


if __name__ == '__main__':
    main()
