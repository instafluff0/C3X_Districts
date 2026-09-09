"""Build a private source snapshot for a controlled unit material experiment.

Patches below apply ONLY to ignored snapshot files. Shared runtime sources and
staged DLLs are read-only. This is a diagnostic, not a runtime format promotion.
"""
from pathlib import Path
import argparse, copy, hashlib, json, shutil, struct
import numpy as np
from .prepare import ROOT, SOURCE, read
from .frame_probe import OUT as FRAME_PACK, recover
HERE=Path(__file__).resolve().parent
OUT=ROOT/'Renderer/lab/out/units/material-study'
SNAP=OUT/'source/Renderer'
PACK=ROOT/'Renderer/packs/UnitMaterialStudy'


def replace(path, old, new, count=1):
    s=path.read_text()
    if s.count(old)!=count:raise ValueError('Snapshot patch context changed: '+str(path.relative_to(OUT))+' '+old[:80])
    path.write_text(s.replace(old,new))


SUBJECTS=[('warrior','UNIT_WARRIOR',None),('spearman','UNIT_SPEARMAN',0),('pikeman','UNIT_PIKEMAN',0),
          ('archer','UNIT_ARCHER',None),('settler','UNIT_SETTLER',0),('worker','UNIT_BUILDER',0)]
ROSTER=False


def prepare_pack():
    frames=(recover(SUBJECTS,FRAME_PACK) if ROSTER else recover())['components']
    manifest=read(SOURCE/'manifest.json');bindings=read(SOURCE/'bindings.json')
    if ROSTER:
        from .prepare import build as sizing_build, OUTPUT as sizing_pack
        sizing_build();sized=read(sizing_pack/'bindings.json')
    PACK.mkdir(parents=True,exist_ok=True)
    result={'unit_count':18};proof={};source_manifest=read(FRAME_PACK/'manifest.json')
    for column,(slug,_,_) in enumerate(SUBJECTS):
        uid='unit/'+(slug if ROSTER else 'warrior');unit=manifest['units'][uid]
        old=next(v for v in bindings.values() if isinstance(v,dict) and v.get('key0') in unit['civ3_ids'])
        for row,mode in enumerate(('baseline','anatomy','ssaa')):
            record=copy.deepcopy(old);record.update(key_count=1,sample_scale=2)
            for k in list(record):
                if k.startswith('key') and k!='key_count':del record[k]
            if ROSTER:
                fit=next(v for v in sized.values() if isinstance(v,dict) and v.get('key0')=='PRTO_Lab_anatomy_'+slug)
                record.update(scale=fit['scale'],offset_z=fit['offset_z'])
            for action,data in unit['actions'].items():
                for i,part in enumerate(data['parts']):
                    bound=record[action]['part'+str(i)]
                    for channel in part['material']['channels'].values():
                        src=SOURCE/channel['texture'];dst=PACK/channel['texture'];dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
                    blob=(SOURCE/part['mesh']).read_bytes()
                    if row:
                        frame=frames[part['asset']];n=struct.unpack_from('<I',blob,12)[0]
                        mesh=read(FRAME_PACK/frame['normalized_mesh'])
                        component=read(FRAME_PACK/source_manifest['assets'][part['asset']]['component'])
                        local_scale=component['model_scale'] if component['binding_mode']=='rigid_attachment' else 1
                        if n!=len(mesh['vertices']):raise ValueError('Frame vertex count mismatch')
                        new=bytearray(blob[:32]);new[:8]=b'C3XANM2\0';struct.pack_into('<I',new,8,2)
                        for j,v in enumerate(mesh['vertices']):
                            original=blob[32+j*64:32+(j+1)*64];values=struct.unpack('<8f4I4f',original)
                            expected=[x*local_scale for x in v['position']]+v['normal']+v['uv0']
                            if not np.allclose(values[:8],expected,atol=1e-6,rtol=0):
                                raise ValueError('Frame/payload vertex identity mismatch '+part['asset'])
                            new+=original+struct.pack('<6f',*(frame['tangents'][j]+frame['bitangents'][j]))
                        new+=blob[32+n*64:];blob=bytes(new)
                        normal=part['material']['channels'].get('normal_0')
                        if normal:bound['normal_texture']=normal['texture']
                    relative='clips/'+hashlib.sha256(blob).hexdigest()+'.bin';dst=PACK/relative;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(blob);bound['mesh']=relative
                    if row==2:bound['study_tint']=1
                    proof[relative]=hashlib.sha256(blob).hexdigest()
            record['key0']='PRTO_Lab_'+mode+'_'+slug
            result['unit'+str(row*6+column)]=record
    (PACK/'bindings.json').write_text(json.dumps(result,indent=2)+'\n')
    (PACK/'study.json').write_text(json.dumps({'production_enabled':False,'subject':'six-unit anatomy roster' if ROSTER else 'warrior',
        'rows':['current material, 2x sampling','authored normal detail, 2x sampling','normal detail plus source alpha tint, 2x sampling'],
        'limit':'Normal and tint equations only. Existing C3X lighting, AO, gloss and tone retained. Second LEAN/specular preprocessing not implemented.',
        'files':proof},indent=2)+'\n')


def snapshot():
    pins={}
    for directory in ('native','lab/shared'):
        for source in (ROOT/'Renderer'/directory).rglob('*'):
            rel=source.relative_to(ROOT/'Renderer')
            if not source.is_file() or 'build' in rel.parts or source.suffix not in ('.h','.cpp','.hlsl','.def'):continue
            target=SNAP/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            pins[str(rel)]=hashlib.sha256(source.read_bytes()).hexdigest()
    source=ROOT/'Renderer/lab/native_preview.cpp';target=SNAP/'lab/native_preview.cpp';shutil.copyfile(source,target)
    pins['lab/native_preview.cpp']=hashlib.sha256(source.read_bytes()).hexdigest()
    (OUT/'source-inputs.json').write_text(json.dumps(pins,indent=2)+'\n')
    native=SNAP/'native';body=native/'unit_body_renderer.h';animation=native/'animation_runtime.h';loader=native/'c3x_renderer.cpp';shader=native/'environment_refresh/unit_shader.h'
    replace(animation,'    FeatureSourceVertex source;','    FeatureSourceVertex source;\n    std::array<float,3> tangent{},bitangent{};')
    replace(animation,'std::memcmp(data.data(), "C3XANM1\\0", 8) != 0','(std::memcmp(data.data(), "C3XANM1\\0", 8) != 0 && std::memcmp(data.data(), "C3XANM2\\0", 8) != 0)')
    replace(animation,'    if (u32() != 1) return false;','    auto version=u32();\n    if ((version!=1 && version!=2) || data[6]!=char(\'0\'+version)) return false;')
    replace(animation,'vertex_count * 64ull','vertex_count * (version==2?88ull:64ull)')
    replace(animation,'        if (std::abs(sum - 1.0f) > 0.00001f) return false;','''        if (std::abs(sum - 1.0f) > 0.00001f) return false;
        if(version==2) {
            for(auto &v:vertex.tangent){v=f32();if(!std::isfinite(v))return false;}
            for(auto &v:vertex.bitangent){v=f32();if(!std::isfinite(v))return false;}
        }''')
    replace(body,'unsigned material_textures[3]={UINT32_MAX,UINT32_MAX,UINT32_MAX};','unsigned material_textures[4]={UINT32_MAX,UINT32_MAX,UINT32_MAX,UINT32_MAX}; float study_tint=0;')
    replace(body,'std::vector<std::array<float,11>> upload;','std::vector<std::array<float,17>> upload;')
    replace(body,'            upload.resize(posed.size());','''            upload.resize(posed.size());
            auto frames=study_frames(mesh.animation,pose.phase);
''')
    replace(body,'(y-shadow.dy*z-shadow.top)/shadow.height};','''(y-shadow.dy*z-shadow.top)/shadow.height};
                for(unsigned basis=0;basis<2;++basis) {
                    auto f=frames[i][basis];
                    auto direction=lighting::object_normal(f[0]*cosine-f[1]*sine,f[0]*sine+f[1]*cosine,f[2]);
                    for(unsigned axis=0;axis<3;++axis)upload[i][11+basis*3+axis]=direction[axis];
                }''')
    replace(body,'ID3D11ShaderResourceView* extra[3]={};','ID3D11ShaderResourceView* extra[4]={};')
    replace(body,'channel<3;++channel)if(part.material_textures','channel<4;++channel)if(part.material_textures')
    replace(body,'context->PSSetShaderResources(2,3,extra);','context->PSSetShaderResources(2,4,extra);\n            values[23]=part.study_tint;')
    replace(body,'UINT stride=44,offset=0;','UINT stride=68,offset=0;')
    replace(body,'empty[5]={};context->PSSetShaderResources(0,5,empty);','empty[6]={};context->PSSetShaderResources(0,6,empty);')
    replace(body,'{"TEXCOORD",1,DXGI_FORMAT_R32G32B32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0}};', '''{"TEXCOORD",1,DXGI_FORMAT_R32G32B32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"BINORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,56,D3D11_INPUT_PER_VERTEX_DATA,0}};''')
    replace(body,'CreateInputLayout(elements,4,','CreateInputLayout(elements,6,')
    helper=(HERE/'sample_frame.h').read_text()
    replace(body,'    template<class Prepare>\n',helper+'\n    template<class Prepare>\n')
    replace(loader,'"ao_texture","gloss_texture","emissive_texture"','"ao_texture","gloss_texture","emissive_texture","normal_texture"')
    replace(loader,'for(unsigned channel=0;channel<3;++channel) {\n                        std::string relative;','for(unsigned channel=0;channel<4;++channel) {\n                        std::string relative;')
    replace(loader,'                    action.parts.push_back(part);','                    json_number_after(data,"study_tint",record,part.study_tint);\n                    action.parts.push_back(part);')
    replace(loader,'part.material_textures[1],part.material_textures[2]};','part.material_textures[1],part.material_textures[2],part.material_textures[3]};')
    replace(loader,'for(unsigned channel=0;channel<4;++channel) {\n            unsigned id=material_ids','for(unsigned channel=0;channel<5;++channel) {\n            unsigned id=material_ids')
    replace(shader,'SamplerState sample_base', 'Texture2D<float4> normal_texture : register(t5);\nSamplerState sample_base')
    replace(shader,'float3 shadow:TEXCOORD1;','float3 shadow:TEXCOORD1;float3 tangent:TANGENT;float3 bitangent:BINORMAL;',count=2)
    replace(shader,'o.shadow=i.shadow;return o;','o.shadow=i.shadow;o.tangent=i.tangent;o.bitangent=i.bitangent;return o;')
    replace(shader,' float3 n=normalize(i.n);int2 cell=', ''' if(moon_color.w>.5) {
  // Confirmed tint-family equation. Owner color replaces the tint only on
  // owner-colored pieces; alpha is a positive modulation mask.
  float3 modulation=owner.w>.5?owner.rgb:tint.rgb;
  albedo=b.rgb*lerp(float3(1,1,1),modulation,b.a);
 }
 float3 n=normalize(i.n);
 if(channels.w>.5) {
  float2 xy=normal_texture.Sample(sample_base,i.uv).rg*2-1;
  float z=sqrt(max(0,1-dot(xy,xy)));
  n=normalize(i.tangent*xy.x+i.bitangent*xy.y+i.n*z);
 }
 int2 cell=''')
    preview=SNAP/'lab/native_preview.cpp'
    replace(preview,'    bool sizing=std::strncmp(enabled,"sizing",6)==0;', '    bool sizing=std::strncmp(enabled,"sizing",6)==0;')
    # Select only the isolated pack before renderer initialization, never change
    # the standard dispatcher or its production pack selection.
    s=preview.read_text();anchor='int main(';idx=s.find(anchor)
    if idx<0:raise ValueError('Missing preview main')
    brace=s.index('{',idx);s=s[:brace+1]+'\n    SetEnvironmentVariableA("C3X_RENDERER_UNIT_PACK","'+PACK.name+'");\n'+s[brace+1:]
    if not ROSTER:s=s.replace('unit.action=std::strcmp(enabled,"sizing-move")==0?2:1;','unit.direction=column+1;\n            unit.action=std::strcmp(enabled,"sizing-move")==0?2:1;')
    # Use a direction label so the existing fixture keys do not imply six units.
    if not ROSTER:s=s.replace('TextOutA(dc,70+column*200,274+row*290,study_keys[column],int(std::strlen(study_keys[column])));','char heading[32];sprintf_s(heading,"Warrior, direction %d",column+1);TextOutA(dc,50+column*200,274+row*290,heading,int(std::strlen(heading)));')
    import re
    s,n=re.subn(r'char const\* labels\[\]\s*=\s*\{[^;]+;', 'char const* labels[]={"Current material, 2x sampling","Authored normal detail, 2x sampling","Normal detail + source alpha tint, 2x sampling"};',s)
    if n!=1:raise ValueError('Missing fixture labels')
    preview.write_text(s)
    setup=(ROOT/'Renderer/lab/build_native_preview.bat').read_text().split('cl /nologo')[0]
    native_win=str(native.relative_to(ROOT/'Renderer/lab')).replace('/','\\')
    preview_win=str(preview.relative_to(ROOT/'Renderer/lab')).replace('/','\\')
    batch=setup.replace('pushd "%~dp0"','')+f'pushd "{native_win}"\n'
    batch+='cl /nologo /std:c++17 /EHsc /O2 /W4 /WX /LD c3x_renderer.cpp terrain_scene_runtime.cpp environment_runtime.cpp terrain_definition_runtime.cpp scene_export.cpp frame_scheduler.cpp /Fo:.\\ /Fe:C3XRenderer.dll /link /DEF:c3x_renderer.def d3d11.lib d3dcompiler.lib dxgi.lib gdi32.lib msimg32.lib user32.lib bcrypt.lib\nif errorlevel 1 exit /b 1\npopd\n'
    batch+=f'cl /nologo /std:c++17 /EHsc /O2 /W4 /WX "{preview_win}" /Fo:"{preview_win}.obj" /Fe:"{preview_win}.exe" /link /LARGEADDRESSAWARE gdi32.lib user32.lib\nexit /b %errorlevel%\n'
    (OUT/'build.bat').write_text(batch)


def main():
    global ROSTER,OUT,SNAP,PACK,FRAME_PACK
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true');parser.add_argument('--render',action='store_true');parser.add_argument('--roster',action='store_true');args=parser.parse_args()
    if args.roster:
        ROSTER=True;OUT=ROOT/'Renderer/lab/out/units/material-roster-study';SNAP=OUT/'source/Renderer'
        PACK=ROOT/'Renderer/packs/UnitMaterialRosterStudy';FRAME_PACK=ROOT/'Renderer/lab/out/units/source-roster-frame-pack'
    prepare_pack();snapshot()
    if args.build or args.render:
        from Renderer.lab.platform import native_command_result
        result=native_command_result('Renderer/lab','call '+str((OUT/'build.bat').relative_to(ROOT/'Renderer/lab')).replace('/','\\'),timeout_seconds=120)
        if result['status']!='pass':raise ValueError('Private material candidate failed: '+str(result))
    if args.render:
        from Renderer.renderer import native_render
        outputs=[]
        for case in ('sizing','sizing-gameplay','sizing-move'):
            for zoom in (128,192):
                outputs.append(native_render('units',case,12,zoom,OUT/case,candidate=SNAP/'native/C3XRenderer.dll',preview=SNAP/'lab/native_preview.cpp.exe'))
        (OUT/'renders.json').write_text(json.dumps(outputs,indent=2)+'\n')
    print('PASS isolated material study')

if __name__=='__main__':main()
