"""Static planar-reflection diagnostic using exact retained scene packets.

With --gpu, both passes execute in one command buffer. The optional two-render
reference proves composition independently. Targets contain frame-specific
linear radiance, not new authored object textures. Windows integration is pending.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
from extend_packet_materials import extend

ROOT=Path(__file__).resolve().parents[4];V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out'

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,indent=2)+'\n')

def render(report,shader,out):
    subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(report),
                    '--shader',str(shader),'--output',str(out)],check=True,cwd=ROOT)

def reflection_dds(raw,path,w,h):
    b=raw.read_bytes()
    if len(b)!=w*h*8:raise ValueError('linear target dimensions')
    header=bytearray(148);header[:4]=b'DDS '
    struct.pack_into('<7I',header,4,124,0x100f,h,w,w*8,0,1)
    struct.pack_into('<2I4s',header,76,32,4,b'DX10')
    struct.pack_into('<I',header,108,0x1000)
    struct.pack_into('<5I',header,128,10,3,0,1,0)
    path.write_bytes(header+b)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region',choices=['longcoast','coastal','inland','wilderness','freshwater'],required=True)
    p.add_argument('--revision',type=int,default=1)
    p.add_argument('--gpu',action='store_true',help='Render both passes on GPU with the unmodified source packet')
    a=p.parse_args();region=a.region;campaign=f'water-reflection-r{a.revision}'
    base='shadow-receiver-r1' if region=='longcoast' else ('water-natural-foundation' if region=='freshwater' else 'river-corridor-r3')
    original=OUT/base/region;folder=OUT/campaign/region
    if folder.exists():raise ValueError('preserved reflection diagnostic exists')
    report=json.loads((original/'report.json').read_text());jobs=json.loads((original/'batch.json').read_text())
    packets=[Path(j[0]) for j in jobs]
    dimensions=[struct.unpack_from('<3I',q.read_bytes(),8) for q in packets]
    # The retained packets render zoom2 by reconstruction/downsample. All four
    # internal target dimensions and projection are identical within a fixture.
    if len(set(dimensions))!=1:
        # The downsample factor may differ, but internal dimensions cannot.
        assert len({d[:2] for d in dimensions})==1
    w,h=dimensions[0][:2]
    fixture=V2/'fixtures/beauty'/campaign/region;fixture.mkdir(parents=True)
    baseline=f'../../{base}/{region}/combined.hlsl'
    # World z is authoring height /112. Projection is .82*halfwidth/112.
    source_module=ROOT/report['effective']['fixture']['modules'][0]
    half=json.loads(source_module.read_text())['projection']['half_width']
    mirrored=fixture/'reflection.hlsl'
    save(mirrored,'#define VSMain Q3OriginalVSMain\n#define VSFeature Q3OriginalVSFeature\n'
         '#define PSMain Q3OriginalPSMain\n#define PSFeature Q3OriginalPSFeature\n'+
         f'#include "{baseline}"\n'+
         '#undef VSMain\n#undef VSFeature\n#undef PSMain\n#undef PSFeature\n'+
         f'#define Q3_REFLECTION_HEIGHT_NDC {4*.82*half/h:.12f}\n'+
         '#include "../../../../shaders/hydrology/planar_reflection_pass.hlsl"\n')
    combined=fixture/'combined.hlsl'
    save(combined,'#define Q3_NATURAL_WATER 1\n#define Q3_OBJECT_REFLECTION 1\n'+
         f'#define Q3_REFLECTION_SIZE float2({w}.0,{h}.0)\n'+f'#include "{baseline}"\n')
    if a.gpu:
        subprocess.run([sys.executable,str(V2/'qa/replay_shader.py'),'--report',str(original/'report.json'),
                        '--shader',str(combined),'--reflection-shader',str(mirrored),
                        '--output',str(folder/'combined')],check=True,cwd=ROOT)
        save(folder/'reflection_provenance.json',{'kind':'GPU planar-reflection diagnostic',
             'runtime_integration':False,'source_report':(original/'report.json').relative_to(ROOT).as_posix(),
             'execution':'One command buffer, reflected geometry pass then combined scene; no intermediate image readback or packet modification',
             'world_z_to_screen_pixels':.82*half,'reflection_height_ndc':4*.82*half/h,
             'reflection_plane_world_z':0,'input_dimensions':[w,h]})
        return
    render(original/'report.json',mirrored,folder/'reflected')
    input_dir=folder/'input';input_dir.mkdir();evidence=[]
    for i,(job,src) in enumerate(zip(jobs,report['outputs'])):
        target=input_dir/f'packet-{i}';dds=input_dir/f'reflection-{i}.dds'
        raw=folder/'reflected'/(Path(src['image']).name+'.linear.rgba16f')
        reflection_dds(raw,dds,w,h)
        evidence.append(extend(Path(job[0]),target,{121:dds}))
        job[0]=str(target);src['packet']=target.relative_to(ROOT).as_posix()
    save(input_dir/'report.json',report);save(input_dir/'batch.json',jobs)
    save(input_dir/'bindings.json',evidence)
    render(input_dir/'report.json',combined,folder/'combined')
    save(folder/'reflection_provenance.json',{'kind':'static two-pass planar reflection diagnostic',
         'runtime_integration':False,'source_report':(original/'report.json').relative_to(ROOT).as_posix(),
         'world_z_to_screen_pixels':.82*half,'reflection_height_ndc':4*.82*half/h,
         'reflection_plane_world_z':0,'input_dimensions':[w,h],
         'geometry_buffers_unchanged':all(e['all_non_binding_tail_bytes_identical'] for e in evidence)})

if __name__=='__main__':main()
