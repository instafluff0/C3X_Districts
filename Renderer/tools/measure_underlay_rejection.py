"""Bounded actual-adapter mask/early-rejection follow-up of the same candidate."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,shutil
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.tools.measure_redraw_submission import archive_outputs
from Renderer.tools.measure_redraw_underlay import ROOT,sha
from Renderer.lab.platform import windows_root,native_command_result

BASE=ROOT/'Renderer/.cache/redraw-underlay-step'
OUT=ROOT/'Renderer/.cache/redraw-underlay-rejection-step'

def prepare():
    if OUT.exists():raise ValueError('Preserve existing rejection evidence')
    OUT.mkdir()
    # Reuse immutable accepted assets, clients and control, without new copies.
    for target,source in (('accepted/shaders','accepted/shaders'),('candidate-shaders','candidate-shaders')):
        path=OUT/target;path.parent.mkdir(parents=True,exist_ok=True)
        path.symlink_to(BASE/source,target_is_directory=True)
    # Parallels cannot launch through Mac directory symlinks. Regular binary
    # hard links preserve the frozen bytes without duplicating storage.
    for folder,name in (('control','C3XRenderer_x64.dll'),('client','client_x64.exe')):
        (OUT/folder).mkdir();os.link(BASE/folder/name,OUT/folder/name)
    shutil.copy2(BASE/'keep-awake.ps1',OUT/'keep-awake.ps1')
    shutil.copytree(BASE/'candidate-sources',OUT/'probe-sources')
    shutil.copy2(ROOT/'Renderer/tools/underlay_rejection_probe.h',
        OUT/'probe-sources/Renderer/sandbox/underlay_rejection_probe.h')
    path=OUT/'probe-sources/Renderer/sandbox/fresh_pipeline.h';text=path.read_text()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('Insertion changed: '+old[:70])
        text=text.replace(old,new)
    replace('#include "underlay_occlusion.h"','#include "underlay_occlusion.h"\n#include "underlay_rejection_probe.h"')
    replace('    SandboxUnderlayOcclusion underlay_occlusion;',
        '    SandboxUnderlayOcclusion underlay_occlusion;\n    SandboxUnderlayRejectionProbe underlay_probe;')
    replace('''            }else if(underlay_occlusion_stage==3 && layer==geometry_underlay)
                context->OMSetDepthStencilState(underlay_occlusion.uncovered.Get(),0);''','''            }else if(underlay_occlusion_stage==3 && layer==geometry_underlay){
                context->OMSetDepthStencilState(underlay_occlusion.uncovered.Get(),0);
                if(!underlay_probe.control_state(renderer.device,context,underlay_occlusion.uncovered.Get()))return false;
            }
            if(underlay_occlusion_stage)
                underlay_probe.bound(context,underlay_occlusion_stage,unsigned(layer),
                    underlay_occlusion_stage==1?nullptr:underlay_occlusion_stage==3?visual.underlay:
                    visual.opaque_coverage[layer==geometry_natural_mountain?1:0],projection_zoom);''')
    replace('''        if(!mirrored && !skip_underlay && !cached_terrain && !reference_path &&
           renderer.city_profile''','''        if(!mirrored && !skip_underlay && !cached_terrain && !reference_path &&
           !records[geometry_underlay].empty() && renderer.city_profile''')
    replace('''            underlay_occlusion_stage=1;bool ok=draw(geometry_underlay);
            underlay_occlusion_stage=2;''','''            underlay_probe.begin(target==static_region.target,projection_zoom);
            bool ok=underlay_probe.copy(renderer.device,context,depth,"before");
            underlay_occlusion_stage=1;if(ok)ok=draw(geometry_underlay);
            if(ok)ok=underlay_probe.copy(renderer.device,context,depth,"underlay");
            underlay_occlusion_stage=2;''')
    replace('''            underlay_occlusion_stage=3;
            if(ok)ok=draw(geometry_underlay);''','''            if(ok)ok=underlay_probe.copy(renderer.device,context,depth,"mask");
            underlay_occlusion_stage=3;
            if(ok)ok=draw(geometry_underlay);''')
    replace('''        if(!mirrored && !skip_wave && !draw(geometry_wave))return false;
        return true;
    }
    bool ensure_glow''','''        if(!mirrored && !skip_wave && !draw(geometry_wave))return false;
        if(underlay_probe.active){
            if(!underlay_probe.copy(renderer.device,renderer.context,target,"color"))return false;
            underlay_probe.active=false;
        }
        return true;
    }
    bool ensure_glow''')
    path.write_text(text)
    (OUT/'probe/obj').mkdir(parents=True)
    batch=(BASE/'candidate/build.bat').read_text().replace('..\\..\\..\\candidate','..\\..\\..\\probe')
    (OUT/'probe/build.bat').write_text(batch)
    manifest={p.relative_to(OUT/'probe-sources').as_posix():sha(p)
        for p in (OUT/'probe-sources').rglob('*') if p.is_file()}
    (OUT/'probe-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')

def build(arm='probe'):
    batch=windows_root()/OUT.relative_to(ROOT).as_posix()/arm/'build.bat'
    result=native_command_result(str((OUT/(arm+'-sources')/'Renderer/native').relative_to(ROOT)),
        'call "'+str(batch)+'"',timeout_seconds=600)
    (OUT/arm/'build-result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise ValueError('Build failed')

def prepare_candidate():
    if (OUT/'candidate-sources').exists():raise ValueError('Preserve existing correction')
    shutil.copytree(OUT/'probe-sources',OUT/'candidate-sources')
    path=OUT/'candidate-sources/Renderer/sandbox/fresh_pipeline.h';text=path.read_text()
    original='''float4 PSSandboxUnderlay(PixelInput input) : SV_Target {
    input.surface_kind = 0.5;
    return PSMain(input).color;
}'''
    if text.count(original)!=1:raise ValueError('Opaque underlay entry changed')
    text=text.replace(original,'[earlydepthstencil]\n'+original)
    path.write_text(text)
    (OUT/'candidate/obj').mkdir(parents=True)
    batch=(OUT/'probe/build.bat').read_text().replace('..\\..\\..\\probe','..\\..\\..\\candidate')
    (OUT/'candidate/build.bat').write_text(batch)
    manifest={p.relative_to(OUT/'candidate-sources').as_posix():sha(p)
        for p in (OUT/'candidate-sources').rglob('*') if p.is_file()}
    (OUT/'candidate-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')

def prepare_client():
    if (OUT/'qa-client-sources').exists():raise ValueError('Preserve existing QA client')
    shutil.copytree(OUT/'probe-sources',OUT/'qa-client-sources',copy_function=os.link)
    previous=ROOT/'Renderer/.cache/redraw-submission-step/client-source'
    for name in ('pass_counts.h','capture_model.h','reference_x64.cpp','client_x64.cpp','camera_witness.h'):
        relative=Path('Renderer/sandbox')/name;target=OUT/'qa-client-sources'/relative
        if target.exists():target.unlink()
        source=previous/relative
        if not source.exists():source=ROOT/relative
        shutil.copy2(source,target)
    # Reference-client fixture headers are not part of the DLL compile closure.
    # Freeze the missing existing native headers; keep all DLL headers frozen.
    for source in (ROOT/'Renderer/native').glob('*.h'):
        target=OUT/'qa-client-sources/Renderer/native'/source.name
        if not target.exists():shutil.copy2(source,target)
    path=OUT/'qa-client-sources/Renderer/sandbox/camera_witness.h';text=path.read_text()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('QA client insertion changed: '+old[:70])
        text=text.replace(old,new)
    replace('    if(zoom_sequence)views={{0,0,1,"capture_zoom"}};', '''    if(zoom_sequence)views={{0,0,1,"capture_zoom"}};
    bool pan_sequence=!std::strcmp(scenario,"pan") || !std::strcmp(scenario,"pan125");
    if(pan_sequence)views={{0,0,!std::strcmp(scenario,"pan125")?1.25f:1.f,"capture_pan"}};
    char capture_every_text[16]{};
    int capture_every=GetEnvironmentVariableA("C3X_UNDERLAY_QA_CAPTURE_EVERY",capture_every_text,sizeof(capture_every_text))?
        std::clamp(std::atoi(capture_every_text),0,180):0;''')
    replace('''            float zoom=zoom_sequence?sandbox_capture_zoom(step):view.zoom;
            LARGE_INTEGER start''','''            float zoom=zoom_sequence?sandbox_capture_zoom(step):view.zoom;
            int pan=pan_sequence?(step%90<=45?step%90:90-step%90):0;
            int camera_x=view.x+pan*2,camera_y=view.y+pan;
            LARGE_INTEGER start''')
    replace('draw(&frame,nullptr,view.x,view.y,24,56,1,47,1,zoom)',
            'draw(&frame,nullptr,camera_x,camera_y,24,56,1,47,1,zoom)')
    replace('present(window,&frame,24,56,1,47,1,view.x,view.y);QueryPerformanceCounter(&done)',
            'present(window,&frame,24,56,1,47,1,camera_x,camera_y);QueryPerformanceCounter(&done)')
    replace('''            char checkpoints[8]={};
            if(captures[0]''','''            if(captures[0] && capture_every && step%capture_every==0){
                char prefix[4*MAX_PATH]={};sprintf_s(prefix,"%s\\\\motion_%d",captures,step);
                if(!capture || capture(prefix))return 30;
            }
            char checkpoints[8]={};
            if(captures[0]''')
    replace('''int(step)*33),view.x,view.y,
                zoom_sequence''','''int(step)*33),view.x+(pan_sequence?(int(step)%90<=45?int(step)%90:90-int(step)%90)*2:0),
                view.y+(pan_sequence?(int(step)%90<=45?int(step)%90:90-int(step)%90):0),
                zoom_sequence''')
    path.write_text(text)
    (OUT/'qa-client/obj').mkdir(parents=True)
    batch=(ROOT/'Renderer/.cache/redraw-submission-step/client/build.bat').read_text().replace(
        '..\\.cache\\redraw-submission-step\\client','..\\..\\..\\qa-client')
    (OUT/'qa-client/build.bat').write_text(batch)
    manifest={p.relative_to(OUT/'qa-client-sources').as_posix():sha(p)
        for p in (OUT/'qa-client-sources').rglob('*') if p.is_file()}
    (OUT/'qa-client-sources.json').write_text(json.dumps(manifest,indent=2)+'\n')

def run(label,arm='probe',hour=12,frames=180,counts=False,captures=False,mask=False,controls=None,client_arm='client'):
    navigation.OUT=OUT
    directory=windows_root()/OUT.relative_to(ROOT).as_posix()/label
    dll_arm='control' if arm=='normal' else 'candidate' if arm=='candidate' else 'probe'
    source='accepted/shaders' if arm=='normal' else 'candidate-shaders'
    options={'C3X_RENDERER_PATCH_PIXELS':'0','C3X_SANDBOX_PASS_COUNTS':'1' if counts else '0',
        'C3X_RENDERER_TRACE':'0','C3X_RENDERER_PROFILE':'0','C3X_SANDBOX_PROFILE_COMPLETION':'0',
        'C3X_RENDERER_OUTPUT_COMPLETION_PROBE':'0','C3X_SANDBOX_GPU_TIMESTAMPS':'0',
        'C3X_SANDBOX_FRAME_TIMINGS':'1','C3X_SANDBOX_WITNESS_SCENARIO':'zoom',
        'C3X_SANDBOX_ZOOM_CHECKPOINTS':'0','C3X_SANDBOX_CAUSAL_MODE':'0','C3X_SANDBOX_CAUSAL_LEDGER':'0',
        'C3X_SANDBOX_WORLD_CONTENT_DIR':str(directory) if counts else '',
        'C3X_SANDBOX_UNDERLAY_PROBE':'0','C3X_SANDBOX_UNDERLAY_REFERENCE':'0',
        'C3X_UNDERLAY_STENCIL_CONTROL':{'fail':'2','pass':'1'}.get(arm,'0'),
        'C3X_UNDERLAY_MASK_DIR':str(directory) if mask else '',
        'C3X_UNDERLAY_QA_CAPTURE_EVERY':'0',
        'C3X_RENDERER_SHADER_SOURCE_ROOT':str(windows_root()/BASE.relative_to(ROOT).as_posix()/source)}
    options.update(controls or {})
    receipt=navigation.run(label,dll_arm,'navigation',hour,frames,options,captures,'full_guest',client_arm)
    receipt['rejection_arm']=arm
    receipt['shader_sha256']={p.relative_to(OUT/source).as_posix():sha(p) for p in (OUT/source).rglob('*.hlsl')}
    (OUT/label/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    archive_outputs(OUT/label)
    raw_receipts={}
    for path in (OUT/label).glob('*.raw'):
        data=path.read_bytes();zipped=gzip.compress(data,mtime=0);target=path.with_suffix('.raw.gz');target.write_bytes(zipped)
        assert gzip.decompress(zipped)==data
        raw_receipts[path.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'compressed_sha256':sha(target)}
        path.unlink()
    if raw_receipts:(OUT/label/'lossless-mask.json').write_text(json.dumps(raw_receipts,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','prepare-candidate','prepare-client','build','run'))
    p.add_argument('--label');p.add_argument('--arm',choices=('normal','probe','fail','pass','candidate','qa-client'),default='probe')
    p.add_argument('--hour',type=int,default=12);p.add_argument('--frames',type=int,default=180)
    p.add_argument('--counts',action='store_true');p.add_argument('--captures',action='store_true');p.add_argument('--mask',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='prepare-candidate':prepare_candidate()
    elif a.action=='prepare-client':prepare_client()
    elif a.action=='build':build(a.arm)
    else:run(a.label,a.arm,a.hour,a.frames,a.counts,a.captures,a.mask)
