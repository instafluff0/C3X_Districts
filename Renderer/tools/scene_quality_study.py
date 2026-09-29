"""Small, repeatable Renderer64 material/sampling study; no game or asset copies."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from Renderer.lab.platform import native_command_result, windows_root

OUT = ROOT / 'Renderer/native/build/scene-quality'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True)
    p.add_argument('--dll', default='Renderer/bin/renderer64/C3XRenderer_x64.dll')
    p.add_argument('--client', default='Renderer/sandbox/out/client_x64.exe')
    p.add_argument('--shaders', default='Renderer/packs/Renderer64CutoverControl')
    p.add_argument('--samples', type=int, choices=[1, 2, 4], default=1)
    p.add_argument('--hour', type=int, default=12)
    p.add_argument('--land', choices=['grassland', 'plains'], default='grassland')
    p.add_argument('--biq', action='store_true', help='Use the unchanged local test.biq map')
    p.add_argument('--view-only', action='store_true',
                   help='Prepare the bounded camera neighborhood for a focused still')
    p.add_argument('--city-site', type=int, nargs=2, metavar=('X', 'Y'),
                   help='Add a disclosed synthetic city/border fixture at this map tile')
    p.add_argument('--tile', type=int, default=256)
    p.add_argument('--frames', type=int, default=1)
    p.add_argument('--hdr', action='store_true')
    p.add_argument('--zoom', type=float)
    p.add_argument('--zoom-peak', type=float, default=1.2)
    p.add_argument('--start-ms', type=int, default=0)
    p.add_argument('--filmic', type=float, default=.5)
    p.add_argument('--center', type=int, nargs=2, default=[24, 56])
    args = p.parse_args()
    if not args.name.replace('-', '').replace('_', '').isalnum():
        p.error('Use a simple case name')
    if not (0 <= args.hour <= 23 and 64 <= args.tile <= 512 and
            1 <= args.frames <= 300 and 0 <= args.start_ms <= 34000 and
            math.isfinite(args.filmic) and 0 <= args.filmic <= 1 and
            math.isfinite(args.zoom_peak) and 1 <= args.zoom_peak <= 3 and
            (args.zoom is None or math.isfinite(args.zoom) and 1 <= args.zoom <= 3)):
        p.error('Invalid study bounds: hour 0..23, tile 64..512, frames 1..300, zoom 1..3, filmic 0..1')
    for value in (args.dll, args.client, args.shaders):
        path = ROOT / value
        if path.is_absolute() and (not path.resolve().is_relative_to(ROOT) or
                any(c in str(path) for c in '\r\n"%&|<>^!')):
            p.error('Study inputs must be safe paths within the checkout')
        if not path.exists():
            p.error('Study input does not exist: ' + value)
    OUT.mkdir(parents=True, exist_ok=True)
    used = sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())
    if used > 900 * 1024**2:
        raise RuntimeError('Clean disposable study outputs before exceeding the 1 GiB budget')
    case = OUT / args.name
    case.mkdir(exist_ok=True)
    scene = case / 'scene.csv'
    if args.biq:
        biq = ROOT / 'Renderer/packs/RendererSourceStudies/maps/test.biq'
        subprocess.run(['node', str(ROOT / 'Renderer/sandbox/export_biq.js'),
                        str(biq), str(scene)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    else:
        rows = []
        for y in range(64):
            for x in range(y % 2, 48, 2):
                terrain = 13 if x >= 32 else 12 if x >= 30 else 11 if x >= 28 else 2 if args.land == 'grassland' else 1
                feature = 6 if (x, y) in [(26, 50), (27, 49)] else 7 if (x, y) in [(20, 54), (21, 55), (20, 56)] else terrain
                rows.append(f'{x},{y},{terrain},{feature},0,0,0')
        scene.write_text('C3X_BIQ_TERRAIN_V3,48,64,1536\n' + '\n'.join(rows) + '\n')
    win = lambda path: str(windows_root() / path.relative_to(ROOT).as_posix())
    options = {
        'C3X_RENDERER_VISUAL_PROFILE': 'city-fidelity',
        'C3X_RENDERER_TRACE': '0',
        'C3X_RENDERER_SHARED_SCENE_SURFACE': '1',
        'C3X_RENDERER_WATER_MOTION': '1', 'C3X_RENDERER_WAVES': '1',
        'C3X_SANDBOX_SHADOW_PATCHES': '1',
        'C3X_SANDBOX_WHOLE_WORLD': '0' if args.view_only else '1',
        'C3X_SANDBOX_UNITS': '1', 'C3X_RENDERER_PREVIEW_UNITS': '1',
        'C3X_RENDERER_CITY_PACK': 'Renderer/packs/CityCompositionRuntime',
        'C3X_RENDERER_SHADER_SOURCE_ROOT': win(ROOT / args.shaders),
        'C3X_SANDBOX_MOVE_START_TILE':
            f'{args.center[0] + (args.center[0]-args.center[1])%2},{args.center[1]}'
            if args.biq else '22,56',
        'C3X_RENDERER_SCENE_SAMPLES': str(args.samples),
        'C3X_RENDERER_SCENE_FILMIC': str(args.filmic),
        'C3X_SANDBOX_STUDY_ZOOM_PEAK': str(args.zoom_peak),
        'C3X_SANDBOX_REPLAY_CLIP': '1',
        'C3X_SANDBOX_CLIP_FRAMES': str(args.frames),
        'C3X_SANDBOX_CAPTURE': win(case / 'frame.bmp'),
        'C3X_RENDERER_PREVIEW_CUSTOM_DEFINITIONS': win(ROOT / 'Renderer/custom.custom_rendering.txt'),
    }
    if args.city_site:
        x, y = args.city_site
        if x < 0 or y < 0 or x >= (100 if args.biq else 48) or y >= (100 if args.biq else 64) or (x-y)%2:
            p.error('City site must be an existing tile of the selected map')
        options['C3X_RENDERER_CITY_BORDER_FIXTURE'] = f'{x},{y},2,1,1,1,0,1'
    elif not args.biq:
        options['C3X_RENDERER_CITY_BORDER_FIXTURE'] = '24,54,2,1,1,1,0,1'
    if args.hdr:
        options['C3X_SANDBOX_HDR_CAPTURE'] = win(case / 'scene.crh')
    if args.zoom:
        options['C3X_SANDBOX_STUDY_ZOOM'] = str(args.zoom)
    options['C3X_SANDBOX_STUDY_START_MS'] = str(args.start_ms)
    paths = [ROOT / args.client, ROOT / args.dll, ROOT,
             ROOT / 'Renderer/default.custom_rendering.txt', scene, case / 'initial.bmp']
    command = ' '.join('"' + win(f) + '"' for f in paths) + f' 1280 800 {args.center[0]} {args.center[1]} {args.tile} {args.hour}'
    script = case / 'run.bat'
    script.write_text('@echo off\nsetlocal\n' + '\n'.join(f'set "{k}={v}"' for k, v in options.items()) + '\n' + command + '\nexit /b %errorlevel%\n')
    for stale in ('frame.bmp', 'initial.bmp', 'frame.png', 'scene.crh', 'scene.crh.gz'):
        (case / stale).unlink(missing_ok=True)
    start = time.monotonic()
    result = native_command_result('Renderer/native', f'call "{win(script)}"', timeout_seconds=240)
    result['seconds'] = time.monotonic() - start
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    result.update(options=vars(args), client_sha256=sha(paths[0]), dll_sha256=sha(paths[1]),
                  scene_sha256=sha(scene), pack_sha256=sha(ROOT / 'Renderer/packs/CityCompositionRuntime/city.bin'),
                  biq_sha256=sha(biq) if args.biq else None,
                  natural_sha256=sha(ROOT / 'Renderer/packs/NaturalFidelityRuntime/natural.bin'),
                  low_relief_sha256=sha(ROOT / 'Renderer/packs/NaturalFidelityRuntime/low-relief.bin')
                    if (ROOT / 'Renderer/packs/NaturalFidelityRuntime/low-relief.bin').exists() else None,
                  shader_sha256={f.relative_to(ROOT / args.shaders).as_posix(): sha(f) for f in (ROOT / args.shaders).rglob('*.hlsl')})
    (case / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'pass' or 'fallback=0' not in result['output_tail'] or not (case / 'frame.bmp').exists():
        raise RuntimeError('Study capture failed')
    from PIL import Image
    with Image.open(case / 'frame.bmp') as im:
        im.save(case / 'frame.png')
    # PNG is lossless; retain a pixel witness before removing this disposable BMP.
    with Image.open(case / 'frame.bmp') as a, Image.open(case / 'frame.png') as b:
        assert a.convert('RGBA').tobytes() == b.convert('RGBA').tobytes()
    (case / 'frame.bmp').unlink()
    (case / 'initial.bmp').unlink(missing_ok=True)
    if args.hdr:
        capture = case / 'scene.crh'
        raw = capture.read_bytes()
        packed = case / 'scene.crh.gz'
        with gzip.open(packed, 'wb', compresslevel=6) as stream:
            stream.write(raw)
        with gzip.open(packed, 'rb') as stream:
            if stream.read() != raw:
                raise RuntimeError('HDR compression verification failed')
        capture.unlink()
    print('STUDY', args.name, round(result['seconds'], 2), 'seconds', flush=True)


if __name__ == '__main__':
    main()
