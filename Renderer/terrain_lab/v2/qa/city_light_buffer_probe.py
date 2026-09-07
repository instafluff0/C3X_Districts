"""Move frozen facade proxies into the existing generic shared frame buffer."""
import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from city_scene_pass import ROOT, V2, executable, Cache, compact_packet, rel, save, file_hash

LIGHTS = 128
BLOCKERS = 32
DECLARATIONS = '''
 float4 Q8LocalParameters;
 float4 Q8LocalEnvelopeLowData;
 float4 Q8LocalEnvelopeHighData;
 float4 Q8LocalPositionRange[128];
 float4 Q8LocalColorIntensity[128];
 float4 Q8LocalDirectionOwner[128];
 float4 Q8LocalBoxLow[32];
 float4 Q8LocalBoxHigh[32];
'''
MACROS = '''
#define Q8_LOCAL_LIGHT_COUNT int(Q8LocalParameters.x)
#define Q8_LOCAL_BLOCKER_COUNT int(Q8LocalParameters.y)
#define Q8_LOCAL_LIGHT_GAIN Q8LocalParameters.z
#define Q8_LOCAL_Z_METRIC Q8LocalParameters.w
#define Q8LocalEnvelopeLow Q8LocalEnvelopeLowData.xyz
#define Q8LocalEnvelopeHigh Q8LocalEnvelopeHighData.xyz
'''


def payload(data):
    lights, boxes = data['lights'], data['blockers']
    if not 1<=len(lights)<=LIGHTS or not 1<=len(boxes)<=BLOCKERS:
        raise ValueError('frame light capacity exceeded')
    def vector(value):
        return len(value)==3 and all(math.isfinite(v) for v in value)
    if not math.isfinite(data['gain']) or not 0<=data['gain']<=4 or not math.isfinite(data['z_metric']) or data['z_metric']<=0:
        raise ValueError('invalid light gain or coordinate scale')
    for light in lights:
        if (not all(vector(light[k]) for k in ('position','color_linear','direction')) or
            not math.isfinite(light['range']) or light['range']<=0 or
            not math.isfinite(light['intensity']) or light['intensity']<0 or
            any(v<0 for v in light['color_linear']) or
            not isinstance(light['owner'],int) or not 0<=light['owner']<len(boxes)):
            raise ValueError('invalid light proxy')
    for box in boxes:
        if not vector(box['low']) or not vector(box['high']) or any(x>y for x,y in zip(box['low'],box['high'])):
            raise ValueError('invalid light blocker')
    # Match the decimal literals used by the frozen static-array shader before
    # packing float32. This is a transport change, not a precision reduction.
    number = lambda x: float(f'{float(x):.10f}')
    rows = [[len(lights), len(boxes), float(f"{data['gain']:.8f}"), number(data['z_metric'])]]
    rows += [[number(min(l['position'][i]-l['range'] for l in lights)) for i in range(3)]+[0],
             [number(max(l['position'][i]+l['range'] for l in lights)) for i in range(3)]+[0]]
    for values, capacity in [([l['position']+[l['range']] for l in lights], LIGHTS),
                             ([l['color_linear']+[l['intensity']] for l in lights], LIGHTS),
                             ([l['direction']+[l['owner']] for l in lights], LIGHTS),
                             ([b['low']+[0] for b in boxes], BLOCKERS),
                             ([b['high']+[0] for b in boxes], BLOCKERS)]:
        rows += [[number(v) for v in row] for row in values]
        rows += [[0,0,0,0] for _ in range(capacity-len(values))]
    return b''.join(struct.pack('<4f', *row) for row in rows)


def shader(source, data):
    marker=' float4 Q6ShadowFlags; // enabled, tighter contact, reserved, reserved\n};'
    if source.count(marker)!=1:raise ValueError('shared frame declaration changed')
    source=source.replace(marker,marker[:-2]+DECLARATIONS+'};')
    marker='float q6_receiver_visibility(PixelInput input,float3 normal,float legacy_shadow) {'
    if source.count(marker)!=1:raise ValueError('receiver entry changed')
    code=f'#define Q8_LOCAL_OCCLUSION {int(data["local_building_occlusion"])}\n'+MACROS+(V2/'shaders/lighting/local_facade_lights.hlsl').read_text()+'\n'
    source=source.replace(marker,code+marker)
    marker='return frame_illumination(normal,'
    if source.count(marker)!=3:raise ValueError('receiver wrappers changed')
    return source.replace(marker,'return q8_local_irradiance(input.q6_world,normal,ambient_visibility)+frame_illumination(normal,')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-render', type=Path, required=True)
    p.add_argument('--lights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a=p.parse_args();out=a.output.resolve();out.relative_to(V2/'audits/beauty/out')
    if out.exists():raise ValueError('preserve existing frame-light trial')
    if shutil.disk_usage(V2).free<8*2**30:raise ValueError('preserve at least 8 GiB free')
    source=a.source_render.resolve();data=json.loads(a.lights.read_text())
    if not data['object_reflections']:raise ValueError('this control requires object reflections enabled')
    out.mkdir(parents=True);(out/'lights.bin').write_bytes(payload(data))
    current=json.loads((source/'report.json').read_text());input_path=ROOT/current['source_report']
    report=json.loads(input_path.read_text());jobs=json.loads((input_path.parent/'batch.json').read_text())
    exe=executable(V2/'qa/append_frame_data.cpp',Cache(V2/'app/.cache'));records=[]
    for index,(row,job) in enumerate(zip(report['outputs'],jobs)):
        original=ROOT/row['packet'];target=out/f'combined-{index}.packet'
        details=json.loads(subprocess.check_output([str(exe),str(original),str(out/'lights.bin'),str(target),'80'],text=True))
        compact_packet(target,V2/'app/.cache/content')
        records.append({'original':rel(original),'original_sha256':file_hash(original),
                        'output':rel(target),'output_sha256':file_hash(target),**details})
        row['packet']=rel(target);job[0]=str(target)
    for name,path in [('combined',source/'shaders/source.hlsl'),('reflection',source/'shaders/reflection/source.hlsl')]:
        (out/f'{name}.hlsl').write_text(shader(path.read_text(),data))
    save(out/'report.json',report);save(out/'batch.json',jobs)
    save(out/'binding.json',{'classification':'Generic frame-buffer transport of existing light proxies; no source geometry or lighting recipe change',
         'lights':rel(a.lights.resolve()),'lights_sha256':file_hash(a.lights),'payload_sha256':file_hash(out/'lights.bin'),
         'source_render':rel(source),'packets':records,'max_lights':LIGHTS,'max_blockers':BLOCKERS})
    started=time.monotonic()
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(out/'report.json'),
                    '--shader',str(out/'combined.hlsl'),'--reflection-shader',str(out/'reflection.hlsl'),
                    '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(out/'render')]+
                   (['--prepare-only'] if a.prepare_only else []),check=True)
    save(out/'elapsed.json',{'preparation_only':a.prepare_only,'seconds':time.monotonic()-started,
                           'scope':'shader preparation, backend pipeline creation, GPU execution and readback; not isolated shader compiler time'})


if __name__=='__main__':main()
