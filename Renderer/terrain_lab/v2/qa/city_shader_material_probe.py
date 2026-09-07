"""Preserve bounded installed rigid-model shader evidence without exporting source code."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from Renderer.tools.asset_compiler.compound_landmark_importer import MAC_ASSETS_ROOT
from Renderer.tools.asset_compiler import compound_landmark_importer as source
from Renderer.tools.asset_compiler.clutter_blp_extractor import TYPE_TEXTURE
from Renderer.tools import renderer_dev

V2=ROOT/'Renderer/terrain_lab/v2'
OUT=V2/'audits/beauty/out/city-shader-source-r1'


def sha(data):return hashlib.sha256(data).hexdigest()


def audit_extra_slots():
    report=json.loads((V2/'audits/beauty/out/city-source-expanded-r1/build.json').read_text())
    selected=next(p['selected'] for p in report['pools'] if p['pool']=='city/pool/american/modern')
    packages={};rows=[]
    for item in selected:
        key=item['package']
        if key not in packages:packages[key]=source.IndexedStaticPackage(MAC_ASSETS_ROOT/key,item['entry'])
        package=packages[key];package.select_direct_string(item['entry'])
        _,owner,model=source.landmark_base_model(package)
        materials=package.pointer_fields(model,source.TYPE_MATERIAL)[0][1]
        textures=package.unique_allocation(TYPE_TEXTURE)
        for index in range(package.allocations[materials-1]['element_count']):
            user_data=struct.unpack_from('<Q',package.array_element(materials,index))[0]
            raw=package.bytes_for(package.pointer_fields(user_data,source.TYPE_MATERIAL_DATA)[0][1])
            for offset in (0x2c,0x30,0x38):
                value=struct.unpack_from('<I',raw,offset)[0]
                if value!=0xffffffff:
                    rows.append({'asset':item['asset_id'],'material':index,'offset':hex(offset),
                                 'texture':source.decode_texture_entry(package,textures,value)})
    target=OUT/'modern-extra-slots.json';target.write_text(json.dumps(rows,indent=2)+'\n')
    return {'records':len(rows),'roster':target.relative_to(ROOT).as_posix(),'sha256':sha(target.read_bytes()),
            'roles':sorted({r['texture']['class'] for r in rows})}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disassemble',action='store_true',help='Use the approved Windows dispatcher for local inspection')
    a=parser.parse_args()
    relative=Path('Base/Binaries/Win64Steam/ShaderAutoGen_Windows_DX11_FinalRelease.bs')
    data=(MAC_ASSETS_ROOT/relative).read_bytes();OUT.mkdir(exist_ok=True)
    rows=[]
    for name,offset,expected_size,stage in [('vertex',1784304,8536,1),('model',1835904,17224,0),('model-opacity',1870480,17360,0)]:
        if data[offset:offset+4]!=b'DXBC':raise ValueError('installed shader container changed; re-audit offsets')
        size,count=struct.unpack_from('<2I',data,offset+24)
        if size!=expected_size:raise ValueError('installed shader size changed')
        shader=data[offset:offset+size];chunks={}
        for start in struct.unpack_from('<'+'I'*count,shader,32):
            tag=shader[start:start+4];length=struct.unpack_from('<I',shader,start+4)[0]
            if start+8+length>size:raise ValueError('invalid shader chunk')
            chunks[tag]=shader[start+8:start+8+length]
        instructions=chunks.get(b'SHEX',chunks.get(b'SHDR'))
        if struct.unpack_from('<I',instructions)[0]>>16!=stage:raise ValueError('unexpected shader stage')
        target=OUT/f'{name}-{offset}.dxbc'
        if target.exists() and target.read_bytes()!=shader:raise ValueError('preserved shader changed')
        if not target.exists():target.write_bytes(shader)
        rows.append({'stage':name,'offset':offset,'bytes':size,'sha256':sha(shader),'disassembly':target.with_suffix('.asm')})
    if a.disassemble:
        command='powershell -NoProfile -ExecutionPolicy Bypass -File Renderer/terrain_lab/v2/qa/disassemble_ground_source.ps1 -Source '+OUT.relative_to(ROOT).as_posix()
        result=renderer_dev.native_command_result('.',command)
        if result['status']!='pass':raise RuntimeError('source shader inspection failed')
    vs=rows[0]['disassembly'].read_text().rstrip('\0');ps=rows[1]['disassembly'].read_text().rstrip('\0')
    # Fail when the locally inspected instruction sequence no longer supports the findings.
    for marker in ('ibfe r0.x, l(8), l(16), v0.y','ishr r0.z, v0.y, l(24)',
                   'mul o3.xyz, r0.wwww, r2.xyzx','mul o4.xyz, r0.xxxx, r1.xyzx'):
        if marker not in vs:raise ValueError('source vertex-frame witness changed')
    for marker in ('g_Roughness','g_LeanMap0','g_LeanMap1','g_Environment_Cube','FlattenedLightInfo',
                   'dp2 r0.w, r2.xyxx, r2.xyxx','sqrt r0.w, r0.w',
                   'mad r1.x, r1.x, l(0.333333), r1.z','mad r1.x, r1.y, l(0.666667), r1.x'):
        if marker not in ps:raise ValueError('source material witness changed')
    opacity=rows[2]['disassembly'].read_text().rstrip('\0')
    for marker in ('g_OpacityMap','min r0.w, r0.w, v6.w','round_ne r0.w, r0.w',
                   'ishl r0.w, l(1), r0.w','iadd oMask, r0.w, l(-1)','mov o0.w, v6.w'):
        if marker not in opacity:raise ValueError('source opacity coverage witness changed')
    for row in rows:
        row['disassembly_sha256']=sha(row['disassembly'].read_bytes())
        row['disassembly']=row['disassembly'].relative_to(ROOT).as_posix()
    result={'classification':'Installed rigid-model shader-family evidence; exact active city permutation not established',
            'source':relative.as_posix(),'source_sha256':sha(data),'shaders':rows,
            'findings':{'frame':'Position payload contains octahedral normal bytes; two SNORM8 octahedral tangent directions feed the pixel shader',
                        'normal_texture':'Normal XY is remapped from UNORM; Z is reconstructed as sqrt(max(0,1-dot(XY,XY)))',
                        'cooked_gloss':'RGB is consumed as two lobe variance parameters and a broad component, not a scalar gloss slider',
                        'variance':'Scalar LEAN1 modifies both lobe variances using g_LeanInfo and UV screen derivatives',
                        'environment':'Source uses roughness-filtered environment cubearray and spherical-harmonic irradiance',
                        'local_lighting':'Source evaluates structured point and spot light lists',
                        'opacity':'Inspected permutation samples opacity at UV0 and writes rounded sample coverage; output alpha remains instance fade. Lab single-sample cutoff 0.5 is an adaptation, not exact source MSAA'},
            'extra_material_slots':audit_extra_slots(),
            'remaining':['Exact active material permutation and constant bindings','LEAN1 variance scale','General importer intake beyond the modern Lab overlay',
                         'Source environment lighting normalization and local-light attachment conversion']}
    target=V2/'audits/beauty/CITY_SHADER_MATERIAL_SOURCE.json';target.write_text(json.dumps(result,indent=2)+'\n')
    print(target.relative_to(ROOT))


if __name__=='__main__':main()
