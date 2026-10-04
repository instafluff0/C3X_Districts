"""Sampled stationary-forest witness for the fixed 4000 BC reveal-scroll save.

Requires Pillow only when reading captures. This complements exact GPU oracles;
it cannot certify unsampled frames, arbitrary saves, or a moving camera.
"""
import argparse
import json
from pathlib import Path
import re
import statistics

ROI = (650, 562, 740, 621)


def excursion(samples):
    """Largest RGB deviation from both stable endpoints (8-bit channel units)."""
    before, after = samples[0], samples[-1]
    def error(a, b):
        return sum(abs(x-y) for x, y in zip(a, b)) / len(a)
    return [min(error(pixels, before), error(pixels, after)) for pixels in samples]


def assess(times, errors, *, camera_stable=True):
    if errors and max(errors) > 12:
        return 'fail'
    if (not camera_stable or len(times) < 30 or times[0] > 94.3 or times[-1] < 109.7
            or max((b-a for a, b in zip(times, times[1:])), default=99) > .3):
        return 'incomplete'
    return 'pass'


def analyze(capture):
    from PIL import Image
    lines=(capture/'renderer.log').read_text(errors='replace').splitlines()
    core=capture/'renderer-core.log.x64'
    offsets=[]
    frequency=24000000
    for line in core.open(errors='replace'):
        match=re.search(r'qpc_frequency=(\d+)',line)
        if match:
            frequency=int(match[1])
            break
    for line in lines:
        match=re.search(r'qpc=(\d+)',line)
        if match:
            offsets.append(int(match[1])/frequency-float(line.split('\t')[1]))
    if not offsets:
        return {'status':'incomplete','reason':'Missing capture clock correlation'}
    offset=statistics.median(offsets)
    cameras=set()
    for line in core.open(errors='replace'):
        if 'stage=fresh-scene-phases' not in line:
            continue
        qpc=re.search(r'qpc=(\d+)',line)
        camera=re.search(r'camera=(-?\d+,-?\d+)',line)
        if qpc and camera and 94 <= int(qpc[1])/frequency-offset <= 110:
            cameras.add(camera[1])
    times=[]; samples=[]; frames=[]
    for line in (capture/'window/timeline.jsonl').open():
        frame=json.loads(line)
        if 'frame' not in frame:
            continue
        time=frame['arrival_qpc']/frequency-offset
        if not 94 <= time <= 110:
            continue
        file=capture/'window'/f'window-{frame["frame"]:06d}.jpg'
        if not file.is_file():
            continue
        with Image.open(file) as image:
            if image.size != (2240,1192):
                return {'status':'incomplete','reason':'Fixture viewport differs from 2240x1192'}
            samples.append(image.crop(ROI).convert('RGB').tobytes())
        times.append(time);frames.append(frame['frame'])
    if not samples:
        return {'status':'incomplete','reason':'No reveal interval samples'}
    errors=excursion(samples)
    stable=len(cameras)==1
    status=assess(times,errors,camera_stable=stable) if stable else 'incomplete'
    return {'status':status,'scope':'Sampled stationary forest during three moves at 1.75x',
            'roi':ROI,'camera_stable':stable,'threshold':12,'max_excursion':max(errors),
            'frames':[{'frame':f,'seconds':t,'excursion':e} for f,t,e in zip(frames,times,errors)]}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=analyze(args.capture)
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'frames'},indent=2))
    raise SystemExit(0 if result['status']=='pass' else 1)
