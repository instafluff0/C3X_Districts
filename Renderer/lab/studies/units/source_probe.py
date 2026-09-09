"""Inventory local object-material DXBC and optionally disassemble it in the VM.

No game code is executed. Source-derived bytecode and text stay in ignored output.
This identifies an object shader family, not a proven unit draw/permutation binding.
"""
from pathlib import Path
import argparse
import hashlib
import json
import struct
ROOT = Path(__file__).resolve().parents[4]


def u32(data, offset):
    return struct.unpack_from('<I',data,offset)[0]


def shader_records(data):
    offset=0
    while True:
        offset=data.find(b'DXBC',offset)
        if offset<0:return
        if offset+32>len(data):raise ValueError('Truncated DXBC header')
        length,count=u32(data,offset+24),u32(data,offset+28)
        if count>64 or length<32+4*count or offset+length>len(data):
            raise ValueError('Invalid DXBC extent')
        shader=data[offset:offset+length];chunks={}
        for index in range(count):
            start=u32(shader,32+index*4)
            if start+8>length:raise ValueError('Invalid DXBC chunk')
            size=u32(shader,start+4)
            if start+8+size>length:raise ValueError('Truncated DXBC chunk')
            chunks[shader[start:start+4]]=shader[start+8:start+8+size]
        resources=[]
        if b'RDEF' in chunks:
            r=chunks[b'RDEF']
            for i in range(u32(r,8)):
                entry=u32(r,12)+i*32;name_offset=u32(r,entry)
                name=r[name_offset:r.index(b'\0',name_offset)].decode('utf-8')
                resources.append({'name':name,'type':u32(r,entry+4),'slot':u32(r,entry+20)})
        yield offset,shader,resources
        offset+=length


def probe(archive, output, disassemble=False):
    data=archive.read_bytes();output.mkdir(parents=True,exist_ok=True)
    result={'archive_name':archive.name,'archive_sha256':hashlib.sha256(data).hexdigest(),
            'dxbc_count':0,'object_lean_shaders':[],'object_skinned_shaders':[],
            'limit':'Object material family only; exact unit permutation and tangent/constant bindings remain unproven.'}
    for offset,shader,resources in shader_records(data):
        result['dxbc_count']+=1
        names={r['name'] for r in resources}
        family='object_lean_shaders' if 'g_LeanMap0' in names else (
            'object_skinned_shaders' if 'SkinnedModelDynamics_Tint' in names else None)
        if family is None:continue
        (output/f'{offset}.dxbc').write_bytes(shader)
        result[family].append({'offset':offset,'bytes':len(shader),
            'sha256':hashlib.sha256(shader).hexdigest(),'resources':resources})
    if disassemble:
        from Renderer.lab.platform import native_command_result
        # The compiler and tool operate under the Lab root; no installed shader
        # or game directory is writable input to this command.
        relative=output.resolve().relative_to(ROOT/'Renderer/lab').as_posix().replace('/','\\')
        setup=(ROOT/'Renderer/lab/build_native_preview.bat').read_text().split('cl /nologo')[0]
        setup=setup.replace('pushd "%~dp0"','')
        exe=relative+'\\disassemble.exe'
        lines=[setup,f'cl /nologo /std:c++17 /EHsc /O2 /W4 /WX studies\\units\\disassemble.cpp '
               f'/Fo:"{relative}\\disassemble.obj" /Fe:"{exe}" /link d3dcompiler.lib',
               'if errorlevel 1 exit /b 1']
        for record in result['object_lean_shaders']+result['object_skinned_shaders']:
            base=relative+'\\'+str(record['offset'])
            lines += [f'"{exe}" "{base}.dxbc" "{base}.asm"','if errorlevel 1 exit /b 1']
        lines.append('exit /b 0');(output/'disassemble.bat').write_text('\n'.join(lines)+'\n')
        status=native_command_result('Renderer/lab',f'call "{relative}\\disassemble.bat"',timeout_seconds=60)
        if status['status']!='pass':raise ValueError('Object shader disassembly failed')
        for record in result['object_lean_shaders']+result['object_skinned_shaders']:
            asm=output/f"{record['offset']}.asm"
            if not asm.is_file():raise ValueError('Missing disassembled shader')
            record['disassembly_sha256']=hashlib.sha256(asm.read_bytes()).hexdigest()
    (output/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path.home()/
        "Library/Application Support/Steam/steamapps/common/Sid Meier's Civilization VI/Civ6.app/Contents/Assets/Base/Binaries/Win64Steam/ShaderAutoGen_Windows_DX11_FinalRelease.bs")
    parser.add_argument('--disassemble',action='store_true')
    args=parser.parse_args()
    result=probe(args.archive,ROOT/'Renderer/lab/out/units/source-shaders',args.disassemble)
    print(f"PASS {result['dxbc_count']} DXBC shaders; {len(result['object_lean_shaders'])} object LEAN variants")
