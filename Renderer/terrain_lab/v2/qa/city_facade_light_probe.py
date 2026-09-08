"""Derive explicitly approximate facade light proxies and render matched packets.

Requires Pillow/NumPy only offline. Source light/VFX attachment binding is not
claimed; the generic light record is derived from sampled emissive geometry.
"""
import argparse
from collections import defaultdict
import json
import io
import math
from pathlib import Path
import subprocess
import struct
import shutil
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2'
sys.path.insert(0,str(V2/'systems/objects'));import presentation as city


sys.path.insert(0, str(ROOT))
from Renderer.lab.shared.cities.facades import (file_hash, linear, emission_texture, bilinear,
    bounded_lights, facade_plane_proxy, derive)


def shader_data(data,gain):
    def array(name,values):
        rows=[','.join(f'{float(x):.10f}' for x in row) for row in values]
        return f'static const float4 {name}[{len(rows)}]={{'+','.join('float4('+r+')' for r in rows)+'};\n'
    text=f'#define Q8_LOCAL_LIGHT_GAIN {gain:.8f}\n#define Q8_LOCAL_Z_METRIC {data["z_metric"]:.10f}\n#define Q8_LOCAL_LIGHT_COUNT {len(data["lights"])}\n#define Q8_LOCAL_BLOCKER_COUNT {len(data["blockers"])}\n'
    text+=array('Q8LocalPositionRange',[l['position']+[l['range']] for l in data['lights']])
    text+=array('Q8LocalColorIntensity',[l['color_linear']+[l['intensity']] for l in data['lights']])
    text+=array('Q8LocalDirectionOwner',[l['direction']+[l['owner']] for l in data['lights']])
    text+=array('Q8LocalBoxLow',[b['low']+[0] for b in data['blockers']])
    text+=array('Q8LocalBoxHigh',[b['high']+[0] for b in data['blockers']])
    low=[min(l['position'][i]-l['range'] for l in data['lights']) for i in range(3)]
    high=[max(l['position'][i]+l['range'] for l in data['lights']) for i in range(3)]
    for name,values in [('Q8LocalEnvelopeLow',low),('Q8LocalEnvelopeHigh',high)]:
        text+=f'static const float3 {name}=float3('+','.join(f'{v:.10f}' for v in values)+');\n'
    return text+(V2/'shaders/lighting/local_facade_lights.hlsl').read_text()+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--augmentation',type=Path,required=True)
    p.add_argument('--source-render',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gain',type=float,default=1)
    p.add_argument('--light-budget',type=int,default=48,help='Bounded spill proxies; every emitting body retains at least one')
    p.add_argument('--prepare-only',action='store_true',help='Prepare independent backend inputs without invoking Metal')
    p.add_argument('--resume',action='store_true',help='Repair preparation before any rendered output exists')
    p.add_argument('--no-blockers',action='store_true',help='Unselected diagnostic: disable local building occlusion')
    p.add_argument('--no-object-reflections',action='store_true',help='Diagnostic: retain local illumination but disable the object reflection pass')
    a=p.parse_args()
    if not 0<=a.gain<=4:raise ValueError('bounded light gain required')
    source=ROOT/a.source_render;output=ROOT/a.output
    output.resolve().relative_to(V2/'audits/beauty/out')
    if shutil.disk_usage(V2).free<8*1024**3:raise ValueError('preserve at least 8 GiB free disk space')
    if output.exists() and any(output.iterdir()) and not a.resume:raise ValueError('preserve prior local light probe')
    if a.resume and ((output/'render/report.json').exists() or list((output/'render').glob('*.bmp'))):raise ValueError('preserve rendered result')
    output.mkdir(parents=True,exist_ok=True)
    augmentation=city.read(a.augmentation);surface=city.read(a.augmentation.parent/'surface.json')
    data=derive(augmentation,surface,a.light_budget);data['gain']=a.gain
    data['local_building_occlusion']=not a.no_blockers;data['object_reflections']=not a.no_object_reflections
    data['augmentation_sha256']=file_hash(ROOT/a.augmentation)
    (output/'lights.json').write_text(json.dumps(data,indent=2)+'\n')
    report=city.read(a.source_render/'report.json');input_report=ROOT/report['source_report']
    for name,path in [('combined',source/'shaders/source.hlsl'),('reflection',source/'shaders/reflection/source.hlsl')]:
        shader=path.read_text();marker='float q6_receiver_visibility(PixelInput input,float3 normal,float legacy_shadow) {'
        if shader.count(marker)!=1:raise ValueError('shared receiver entry changed')
        shader=shader.replace(marker,f'#define Q8_LOCAL_OCCLUSION {int(not a.no_blockers)}\n'+shader_data(data,a.gain)+marker)
        marker='return frame_illumination(normal,'
        if shader.count(marker)!=3:raise ValueError('shared receiver lighting wrappers changed')
        shader=shader.replace(marker,'return q8_local_irradiance(input.q6_world,normal,ambient_visibility)+frame_illumination(normal,')
        if a.no_object_reflections:
            count=shader.count('#define Q3_OBJECT_REFLECTION 1')
            if (name=='combined' and count!=1) or count>1:raise ValueError('object reflection control marker changed')
            shader=shader.replace('#define Q3_OBJECT_REFLECTION 1','// Object reflection disabled for explicit diagnostic')
        (output/f'{name}.hlsl').write_text(shader)
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(input_report),'--shader',str(output/'combined.hlsl'),
                    '--post-shader',str(source/'postprocess/source.hlsl'),'--output',str(output/'render')]+
                    ([] if a.no_object_reflections else ['--reflection-shader',str(output/'reflection.hlsl')])+
                    (['--prepare-only'] if a.prepare_only else []),check=True)


if __name__=='__main__':main()
