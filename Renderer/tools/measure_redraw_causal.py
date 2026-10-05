"""Private C7 raster/submission isolation; never stage or alter production code."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.tools.measure_redraw_submission import archive_outputs
from Renderer.lab.platform import windows_root, native_command_result

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Renderer/.cache/redraw-causal-step'
PREVIOUS=ROOT/'Renderer/.cache/redraw-submission-step'
CONTROL_SHA='c7bc766764558a27587bc4e1cd66f4b54e3229131ba3ce36c7d1375ccd6c0931'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old)!=1:raise ValueError('Unexpected C7 source: '+old[:100])
    return text.replace(old,new)


def restrict_function(text, start, end, scope_anchor):
    first=text.index(start);last=text.index(end,first)
    section=text[first:last]
    section=replace_once(section,scope_anchor,scope_anchor+'\n        SandboxCausalRaster causal_raster(context);')
    section,count=re.subn(r'context->(DrawIndexed(?:Instanced)?)\(([^;]+)\);',
        r'if(sandbox_causal.issue())context->\1(\2);',section)
    if not count:raise ValueError('No geometry draws found')
    return text[:first]+section+text[last:],count


def transformed(relative, text):
    counts={}
    if relative=='Renderer/sandbox/resident_scene.cpp':
        text=replace_once(text,'#include "../native/c3x_renderer.cpp"',
            '#include "redraw_causal.h"\n#include "../native/c3x_renderer.cpp"')
    if relative=='Renderer/sandbox/fresh_pipeline.h':
        for name,end,anchor in (
            ('bool issue_records(', 'void bind_common(', 'auto* context=renderer.context;'),
            ('bool draw_vegetation_instances(', 'bool draw_layer(', 'auto* context=renderer.context;')):
            text,counts[name]=restrict_function(text,name,end,anchor)
        text=replace_once(text,'work.begin();shadow.work=&work;',
            'work.begin();SandboxCausalFrame causal_frame(&work.pass,frame.presentation_time_ticks);shadow.work=&work;')
    if relative=='Renderer/sandbox/direct_units.h':
        # Synthetic bodies have no intervening self-shadow draw. Real bodies
        # enter the scope after each self-shadow pass has restored scene state.
        text=replace_once(text,'for(int index=0;index<4;++index){',
            'SandboxCausalRaster causal_raster(context);\n        for(int index=0;index<4;++index){')
        bind='if(!known||shadow_view!=last_shadow_view){context->PSSetShaderResources(1,1,&shadow_view);last_shadow_view=shadow_view;}'
        text=replace_once(text,bind,bind+'\n            SandboxCausalRaster causal_raster(context);')
        # Exactly three scene draws (legacy shadow/body plus the resident
        # body loop, which shares one site for its shadow and body layers);
        # the height-atlas draw is deliberately intact.
        text,count=re.subn(r'context->DrawIndexed\(UINT\(source->indices.size\(\)\),0,0\);',
            r'if(sandbox_causal.issue())\g<0>',text)
        if count!=3:raise ValueError('Unexpected unit draw sites')
        counts['unit_scene']=count
    return text,counts


API=re.compile(r'\b([A-Za-z_]\w*(?:(?:\.|->)[A-Za-z_]\w*)*)->('
    r'(?:IA|VS|PS|GS|HS|DS|CS|RS|OM|SO)Set\w+|UpdateSubresource\w*|Map|Unmap|'
    r'Copy\w+|Clear(?:RenderTargetView|DepthStencilView|UnorderedAccessView\w*)|'
    r'ResolveSubresource|Draw(?:Indexed|Instanced|IndexedInstanced)?)(?=\s*\()')


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'sources').exists():raise ValueError('Preserve existing diagnostic snapshot')
    runtime=json.loads((PREVIOUS/'compiled-input-verification.json').read_text())['runtime']
    mismatch={p:sha(ROOT/p) for p,h in runtime.items() if sha(ROOT/p)!=h}
    if mismatch:raise ValueError('C7 runtime changed: '+str(mismatch))
    runtime['Renderer/native/c3x_renderer.def']=sha(ROOT/'Renderer/native/c3x_renderer.def')
    for arm in ('control','private','ledger','client'):(OUT/arm).mkdir()
    control=PREVIOUS/'candidate/C3XRenderer_x64.dll'
    if sha(control)!=CONTROL_SHA:raise ValueError('Wrong C7 control')
    shutil.copy2(control,OUT/'control/C3XRenderer_x64.dll')
    shutil.copy2(PREVIOUS/'client/client_x64.exe',OUT/'client/client_x64.exe')
    shutil.copy2(PREVIOUS/'keep-awake.ps1',OUT/'keep-awake.ps1')
    # Small accepted shaders only. No asset pack or prior evidence duplication.
    shutil.copytree(PREVIOUS/'accepted/shaders',OUT/'accepted/shaders')
    manifest={};sites={};api_sites={}
    for p,h in runtime.items():
        path=ROOT/p
        if path.suffix not in ('.h','.cpp','.hlsl','.def'):continue
        original=path.read_text();text,changed=transformed(p,original)
        if changed:sites[p]=changed
        target=OUT/'sources'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        # Ledger-only builds count the real calls without changing their args.
        ledger,n=API.subn(lambda m:'(sandbox_causal.api("'+m[2]+'"),'+m[1]+')->'+m[2],text)
        if n:api_sites[p]=n
        target=OUT/'ledger-sources'/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(ledger)
        manifest[p]={'c7':h,'private':sha(OUT/'sources'/p),'ledger':sha(target)}
    header=ROOT/'Renderer/sandbox/redraw_causal.h'
    for source in ('sources','ledger-sources'):
        shutil.copy2(header,OUT/source/'Renderer/sandbox/redraw_causal.h')
    build=(PREVIOUS/'candidate/build.bat').read_text()
    for arm,source in (('private','sources'),('ledger','ledger-sources')):
        (OUT/arm/'obj').mkdir()
        altered=build.replace('..\\.cache\\redraw-submission-step\\candidate',
            '..\\..\\..\\'+arm)
        altered=altered.replace('/DC3X_HELPER_TRIAL',
            '/FI"..\\sandbox\\redraw_causal.h" /DC3X_HELPER_TRIAL')
        (OUT/arm/'build.bat').write_text(altered)
    (OUT/'sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (OUT/'transformation.json').write_text(json.dumps({'geometry_draw_sites':sites,
        'api_sites':api_sites,'header_sha256':sha(header),'runtime_count':len(runtime),
        'control_sha256':sha(control),'client_sha256':sha(OUT/'client/client_x64.exe')},indent=2)+'\n')


def build(arm):
    source='ledger-sources' if arm=='ledger' else 'sources'
    batch=windows_root()/OUT.relative_to(ROOT).as_posix()/arm/'build.bat'
    result=native_command_result(str((OUT/source/'Renderer/native').relative_to(ROOT)),
        'call "'+str(batch)+'"',timeout_seconds=600)
    (OUT/arm/'build-result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise ValueError('Build failed')


def material_probe():
    """One invalid-output probe of main ground shading; retain its coverage."""
    destination=OUT/'material-shaders-corrected'
    if destination.exists():raise ValueError('Preserve existing material probe')
    shutil.copytree(OUT/'accepted/shaders',destination)
    path=destination/'Renderer/native/city_fidelity/terrain.hlsl'
    before=path.read_text()
    entry='''Output PSFeature(P input) {
#ifndef SANDBOX_TERRAIN_MATERIAL
 if(input.material.y>=.5 && input.material.y<1.5){
  float alpha=saturate(input.coast_coverage+10);
  float2 edge_uv=input.world.xy*Detail.x*2.05+float2(.31,.17);
  float edge_height=GrassColor.Sample(Wrap,edge_uv).a;
  float edge_mean=GrassColor.SampleBias(Wrap,edge_uv,3).a;
  float edge_grain=saturate(.5+(edge_height-edge_mean)*3);
  alpha=lerp(coast_edge_coverage(alpha,edge_grain),alpha,input.coast_inland);
  clip(alpha-.001);
  Output output;output.color=float4(.2,.3,.1,1)*alpha;
  output.validity=alpha;return output;
 }
#endif
 return shade(input);
}'''
    text=replace_once(before,'Output PSFeature(P input) { return shade(input); }',entry)
    # Reflection entry points and material-capture shade() remain unchanged.
    path.write_text(text)
    (OUT/'material-probe-corrected.json').write_text(json.dumps({'file':str(path.relative_to(OUT)),
        'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':sha(path),
        'scope':'PSFeature main ground .5 <= material.y < 1.5; original alpha/clip, vertices, depth, reflected shading and other material branches'},indent=2)+'\n')


def run(label, arm='A', hour=12, frames=180, ledger=False, captures=False):
    navigation.OUT=OUT
    options={'C3X_SANDBOX_PASS_COUNTS':'1' if ledger else '0',
        'C3X_RENDERER_TRACE':'0','C3X_RENDERER_PROFILE':'0',
        'C3X_SANDBOX_WITNESS_SCENARIO':'zoom','C3X_SANDBOX_GPU_TIMESTAMPS':'0',
        'C3X_SANDBOX_FRAME_TIMINGS':'1','C3X_SANDBOX_ZOOM_CHECKPOINTS':'0',
        'C3X_SANDBOX_CAUSAL_MODE':{'A':'0','B':'1','C':'2','D':'0','E':'0','frozen':'0'}[arm],
        'C3X_SANDBOX_CAUSAL_LEDGER':'1' if ledger else '0',
        'C3X_SANDBOX_SKIP_REFLECTION':'1' if arm=='D' else '',
        'C3X_SANDBOX_WORLD_CONTENT_DIR':'','C3X_RENDERER_TRACE_FILE':''}
    dll='control' if arm=='frozen' else 'ledger' if ledger else 'private'
    shaders=OUT/'material-shaders-corrected' if arm=='E' else OUT/'accepted/shaders'
    if arm=='E':options['C3X_RENDERER_SHADER_SOURCE_ROOT']=str(windows_root()/shaders.relative_to(ROOT).as_posix())
    result=navigation.run(label,dll,'navigation',hour,frames,options,captures,'full_guest','client')
    result['experiment_arm']=arm
    result['shader_sha256']={p.relative_to(shaders).as_posix():sha(p) for p in shaders.rglob('*.hlsl')}
    (OUT/label/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    archive_outputs(OUT/label)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','build','run','material-probe'])
    parser.add_argument('--label');parser.add_argument('--arm',default='A')
    parser.add_argument('--hour',type=int,default=12);parser.add_argument('--frames',type=int,default=180)
    parser.add_argument('--ledger',action='store_true');parser.add_argument('--captures',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='build':build(args.arm)
    elif args.action=='material-probe':material_probe()
    else:run(args.label,args.arm,args.hour,args.frames,args.ledger,args.captures)
