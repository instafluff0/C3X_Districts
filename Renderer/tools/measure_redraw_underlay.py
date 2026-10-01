"""Private underlay-entry cost/visible-influence probe against frozen C7."""
from pathlib import Path
import argparse,hashlib,json,shutil
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.tools.measure_redraw_submission import archive_outputs
from Renderer.lab.platform import windows_root,native_command_result

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Renderer/.cache/redraw-underlay-step'
PREVIOUS=ROOT/'Renderer/.cache/redraw-submission-step'
CONTROL_SHA='c7bc766764558a27587bc4e1cd66f4b54e3229131ba3ce36c7d1375ccd6c0931'
ENTRY='''float4 PSSandboxUnderlay(PixelInput input) : SV_Target {
    input.surface_kind = 0.5;
    return PSMain(input).color;
}'''

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def transformed(relative,text):
    if relative!='Renderer/sandbox/fresh_pipeline.h':return text
    if text.count(ENTRY)!=1:raise ValueError('Underlay entry changed')
    # Only this appended entry changes. Original PSMain and all other entries
    # keep their source, shader identity, resources and compilation definitions.
    probe='''float4 PSSandboxUnderlay(PixelInput input) : SV_Target {
    return float4(C3X_PRIVATE_UNDERLAY_COLOR, 1);
}'''
    anchor='''        if(std::strcmp(entry,"PSSandboxUnderlay")==0)
            shader.append(R"('''+ '\n'+ENTRY+'\n'+''')");'''
    if text.count(anchor)!=1:raise ValueError('Underlay append changed')
    replacement='''        if(std::strcmp(entry,"PSSandboxUnderlay")==0){
            char value[8]{};
            unsigned mode=GetEnvironmentVariableA("C3X_SANDBOX_UNDERLAY_PROBE",value,sizeof(value))?
                unsigned(std::atoi(value)):0;
            if(mode==1 || mode==2){
                shader.insert(0,mode==1?"#define C3X_PRIVATE_UNDERLAY_COLOR 1,0,1\\n":
                    "#define C3X_PRIVATE_UNDERLAY_COLOR 0,1,0\\n");
                shader.append(R"(
'''+probe+'''
)");
            }else shader.append(R"(
'''+ENTRY+'''
)");
            std::printf("UNDERLAY_ENTRY file=%s entry=%s mode=%u alpha=1 depth_output=0\\n",file,entry,mode);
        }'''
    return text.replace(anchor,replacement)

def prepare():
    if OUT.exists():raise ValueError('Preserve existing underlay evidence')
    runtime=json.loads((PREVIOUS/'compiled-input-verification.json').read_text())['runtime']
    for p,h in runtime.items():
        if sha(ROOT/p)!=h:raise ValueError('C7 source changed: '+p)
    for name in ('control','private','client'):(OUT/name).mkdir(parents=True)
    control=PREVIOUS/'candidate/C3XRenderer_x64.dll';assert sha(control)==CONTROL_SHA
    shutil.copy2(control,OUT/'control/C3XRenderer_x64.dll')
    shutil.copy2(PREVIOUS/'client/client_x64.exe',OUT/'client/client_x64.exe')
    shutil.copy2(PREVIOUS/'keep-awake.ps1',OUT/'keep-awake.ps1')
    shutil.copytree(PREVIOUS/'accepted/shaders',OUT/'accepted/shaders')
    runtime['Renderer/native/c3x_renderer.def']=sha(ROOT/'Renderer/native/c3x_renderer.def')
    manifest={}
    for p,h in runtime.items():
        path=ROOT/p
        if path.suffix not in ('.h','.cpp','.hlsl','.def'):continue
        target=OUT/'sources'/p;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(transformed(p,path.read_text()))
        manifest[p]={'control':h,'private':sha(target)}
    (OUT/'private/obj').mkdir()
    build=(PREVIOUS/'candidate/build.bat').read_text().replace(
        '..\\.cache\\redraw-submission-step\\candidate','..\\..\\..\\private')
    (OUT/'private/build.bat').write_text(build)
    (OUT/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')

def build():
    batch=windows_root()/OUT.relative_to(ROOT).as_posix()/'private/build.bat'
    result=native_command_result(str((OUT/'sources/Renderer/native').relative_to(ROOT)),
        'call "'+str(batch)+'"',timeout_seconds=600)
    (OUT/'private/build-result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise ValueError('Build failed')

def prepare_candidate():
    if (OUT/'candidate').exists():raise ValueError('Preserve existing candidate')
    shutil.copytree(OUT/'sources',OUT/'candidate-sources')
    shutil.copytree(OUT/'accepted/shaders',OUT/'candidate-shaders')
    shutil.copy2(ROOT/'Renderer/tools/redraw_underlay_candidate.h',
        OUT/'candidate-sources/Renderer/sandbox/underlay_occlusion.h')
    shutil.copy2(ROOT/'Renderer/tools/redraw_underlay_coverage.hlsl',
        OUT/'candidate-shaders/Renderer/sandbox/underlay_coverage.hlsl')
    path=OUT/'candidate-sources/Renderer/sandbox/fresh_pipeline.h';text=path.read_text()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('Candidate insertion changed: '+old[:60])
        text=text.replace(old,new)
    replace('#include "pass_workload.h"','#include "pass_workload.h"\n#include "underlay_occlusion.h"')
    replace('    ID3D11PixelShader* underlay = nullptr;',
        '    ID3D11PixelShader* underlay = nullptr;\n    ID3D11PixelShader* opaque_coverage[2] = {};')
    replace('        if(underlay) underlay->Release();',
        '        if(underlay) underlay->Release();\n        for(auto* shader:opaque_coverage)if(shader)shader->Release();')
    replace('        if(albedo_variant)shader.append(R"(', '''        if(std::strcmp(entry,"PSSandboxOpaqueCoverage")==0){
            if(std::strcmp(file,"terrain.hlsl")==0)
                shader.insert(0,"#define C3X_UNDERLAY_GROUND_COVERAGE 1\\n");
            std::ifstream coverage(renderer.shader_root+"/Renderer/sandbox/underlay_coverage.hlsl",std::ios::binary);
            if(!coverage)return fail_source("opaque-coverage-missing");
            shader.append(std::istreambuf_iterator<char>(coverage),std::istreambuf_iterator<char>());
        }
        if(albedo_variant)shader.append(R"(''')
    replace('        if(!compile("hydrology.hlsl","PSSandboxUnderlay",&underlay))return false;',
        '''        if(!compile("hydrology.hlsl","PSSandboxUnderlay",&underlay))return false;
        if(!compile("terrain.hlsl","PSSandboxOpaqueCoverage",&opaque_coverage[0]) ||
           !compile("mountain.hlsl","PSSandboxOpaqueCoverage",&opaque_coverage[1]))return false;''')
    replace('struct SandboxFreshPipeline {','''struct SandboxFreshPipeline {
    SandboxUnderlayOcclusion underlay_occlusion;
    unsigned underlay_occlusion_stage=0;''')
    replace('        return issue_records(records,layer,settings,rect,mirrored);\n    }\n    bool draw_scene(','''        if(!mirrored){
            if(underlay_occlusion_stage==1 && layer==geometry_underlay){
                context->PSSetShader(nullptr,nullptr,0);
                context->OMSetBlendState(underlay_occlusion.no_color.Get(),nullptr,0xffffffffu);
            }else if(underlay_occlusion_stage==2 &&
                    (layer==geometry_natural_terrain || layer==geometry_natural_mountain)){
                context->PSSetShader(visual.opaque_coverage[layer==geometry_natural_mountain?1:0],nullptr,0);
                context->OMSetBlendState(underlay_occlusion.no_color.Get(),nullptr,0xffffffffu);
                context->OMSetDepthStencilState(underlay_occlusion.mark.Get(),1);
            }else if(underlay_occlusion_stage==3 && layer==geometry_underlay)
                context->OMSetDepthStencilState(underlay_occlusion.uncovered.Get(),0);
        }
        return issue_records(records,layer,settings,rect,mirrored);
    }
    bool draw_scene(''')
    replace('''        if(!mirrored && ((!skip_underlay && !draw(geometry_underlay)) ||
                !draw(geometry_land)))return false;''','''        bool covered_underlay=false;
        char reference[8]{};
        bool reference_path=GetEnvironmentVariableA("C3X_SANDBOX_UNDERLAY_REFERENCE",reference,sizeof(reference)) &&
            std::strcmp(reference,"1")==0;
        // Other profiles/sample modes keep their original material path.
        // Clear only stencil, leaving all existing region depth untouched.
        if(!mirrored && !skip_underlay && !cached_terrain && !reference_path &&
           renderer.city_profile && scene_samples==1 && depth &&
           underlay_occlusion.ensure(renderer.device)){
            auto* context=renderer.context;
            context->ClearDepthStencilView(depth,D3D11_CLEAR_STENCIL,1,0);work.clear(depth);
            underlay_occlusion_stage=1;bool ok=draw(geometry_underlay);
            underlay_occlusion_stage=2;
            if(ok)ok=draw(geometry_natural_terrain) && draw(geometry_natural_mountain);
            underlay_occlusion_stage=3;
            if(ok)ok=draw(geometry_underlay);
            underlay_occlusion_stage=0;
            if(!ok)return false;
            covered_underlay=true;
        }
        if(!mirrored && ((!skip_underlay && !covered_underlay && !draw(geometry_underlay)) ||
                !draw(geometry_land)))return false;''')
    path.write_text(text)
    (OUT/'candidate/obj').mkdir(parents=True)
    batch=(OUT/'private/build.bat').read_text().replace('..\\..\\..\\private','..\\..\\..\\candidate')
    (OUT/'candidate/build.bat').write_text(batch)
    manifest={p.relative_to(OUT/'candidate-sources').as_posix():sha(p)
        for p in (OUT/'candidate-sources').rglob('*') if p.is_file()}
    (OUT/'candidate-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')

def build_candidate():
    batch=windows_root()/OUT.relative_to(ROOT).as_posix()/'candidate/build.bat'
    result=native_command_result(str((OUT/'candidate-sources/Renderer/native').relative_to(ROOT)),
        'call "'+str(batch)+'"',timeout_seconds=600)
    (OUT/'candidate/build-result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise ValueError('Candidate build failed')

def run(label,arm='normal',hour=12,frames=180,counts=False,captures=False,controls=None):
    navigation.OUT=OUT
    directory=windows_root()/OUT.relative_to(ROOT).as_posix()/label
    options={'C3X_RENDERER_PATCH_PIXELS':'0','C3X_SANDBOX_PASS_COUNTS':'1' if counts else '0',
        'C3X_RENDERER_TRACE':'0','C3X_RENDERER_PROFILE':'0','C3X_SANDBOX_PROFILE_COMPLETION':'0',
        'C3X_RENDERER_OUTPUT_COMPLETION_PROBE':'0','C3X_SANDBOX_GPU_TIMESTAMPS':'0',
        'C3X_SANDBOX_FRAME_TIMINGS':'1','C3X_SANDBOX_WITNESS_SCENARIO':'zoom',
        'C3X_SANDBOX_ZOOM_CHECKPOINTS':'0','C3X_SANDBOX_CAUSAL_MODE':'0','C3X_SANDBOX_CAUSAL_LEDGER':'0',
        'C3X_SANDBOX_WORLD_CONTENT_DIR':str(directory) if counts else '',
        'C3X_SANDBOX_UNDERLAY_PROBE':{'normal':'0','private-normal':'0','magenta':'1','green':'2','candidate':'0','candidate-reference':'0'}[arm],
        'C3X_SANDBOX_UNDERLAY_REFERENCE':'1' if arm=='candidate-reference' else '0'}
    if arm.startswith('candidate'):
        options['C3X_RENDERER_SHADER_SOURCE_ROOT']=str(windows_root()/OUT.relative_to(ROOT).as_posix()/'candidate-shaders')
    options.update(controls or {})
    dll_arm='control' if arm=='normal' else 'candidate' if arm.startswith('candidate') else 'private'
    receipt=navigation.run(label,dll_arm,'navigation',
        hour,frames,options,captures,'full_guest','client')
    receipt['underlay_arm']=arm
    shader_root=OUT/('candidate-shaders' if arm.startswith('candidate') else 'accepted/shaders')
    receipt['shader_sha256']={p.relative_to(shader_root).as_posix():sha(p)
        for p in shader_root.rglob('*.hlsl')}
    (OUT/label/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    archive_outputs(OUT/label);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','build','prepare-candidate','build-candidate','run'))
    p.add_argument('--label');p.add_argument('--arm',choices=('normal','private-normal','magenta','green','candidate','candidate-reference'),default='normal')
    p.add_argument('--hour',type=int,default=12);p.add_argument('--frames',type=int,default=180)
    p.add_argument('--counts',action='store_true');p.add_argument('--captures',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='build':build()
    elif a.action=='prepare-candidate':prepare_candidate()
    elif a.action=='build-candidate':build_candidate()
    else:run(a.label,a.arm,a.hour,a.frames,a.counts,a.captures)
