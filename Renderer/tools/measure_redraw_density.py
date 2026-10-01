"""Bounded frozen-C7 density/refinement fixture; no production LOD or staging."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
from Renderer.tools import measure_redraw_navigation as navigation
from Renderer.tools.measure_redraw_submission import archive_outputs
from Renderer.lab.platform import windows_root,native_command_result

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Renderer/.cache/redraw-density-step'
PREVIOUS=ROOT/'Renderer/.cache/redraw-submission-step'
CONTROL_SHA='c7bc766764558a27587bc4e1cd66f4b54e3229131ba3ce36c7d1375ccd6c0931'
PIXELS={'normal':'0','32':'3','16':'5'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    if OUT.exists():raise ValueError('Preserve existing density evidence')
    runtime=json.loads((PREVIOUS/'compiled-input-verification.json').read_text())['runtime']
    changed={p:sha(ROOT/p) for p,h in runtime.items() if sha(ROOT/p)!=h}
    if changed:raise ValueError('Accepted C7 runtime changed: '+str(changed))
    control=PREVIOUS/'candidate/C3XRenderer_x64.dll'
    if sha(control)!=CONTROL_SHA:raise ValueError('Wrong control DLL')
    for name in ('control','client'):(OUT/name).mkdir(parents=True)
    shutil.copy2(control,OUT/'control/C3XRenderer_x64.dll')
    shutil.copy2(PREVIOUS/'client/client_x64.exe',OUT/'client/client_x64.exe')
    shutil.copy2(PREVIOUS/'keep-awake.ps1',OUT/'keep-awake.ps1')
    shutil.copytree(PREVIOUS/'accepted/shaders',OUT/'accepted/shaders')
    (OUT/'source-before.json').write_text(json.dumps(runtime,indent=2)+'\n')
    (OUT/'binary-identities.json').write_text(json.dumps({'control':sha(control),
        'client':sha(OUT/'client/client_x64.exe')},indent=2)+'\n')


def prepare_candidate():
    """One private refinement implementation; leave the accepted runtime alone."""
    if (OUT/'candidate').exists():raise ValueError('Preserve existing refinement candidate')
    (OUT/'candidate/obj').mkdir(parents=True)
    runtime=json.loads((OUT/'source-before.json').read_text())
    relative='Renderer/native/c3x_renderer.def'
    runtime[relative]=sha(ROOT/relative)
    sources={}
    for relative,h in runtime.items():
        path=ROOT/relative
        if sha(path)!=h:raise ValueError('C7 source changed: '+relative)
        if path.suffix not in ('.h','.cpp','.hlsl','.def'):continue
        text=path.read_text()
        if relative=='Renderer/lab/shared/natural/ground.h':
            text=text.replace('#include "patch.h"','#include "patch.h"\n#include "surface_refinement.h"')
            anchor='    if(edge_divisions<=divisions){\n'
            if text.count(anchor)!=1:raise ValueError('Ground insertion changed')
            text=text.replace(anchor,anchor+'        if(divisions==64 && !clip_coast)\n            return append_refined_surface_grid(out,grid,divisions,cancelled,indices);\n')
        if relative=='Renderer/lab/shared/natural/relief_mesh_body.h':
            anchor='            append_surface_grid(natural_vertices[2],grid,count-1,false,mountain_indices,&patch_layouts.get(count-1));'
            if text.count(anchor)!=1:raise ValueError('Relief insertion changed')
            text=text.replace(anchor,'            if(count-1==64){\n                if(!append_refined_surface_grid(natural_vertices[2],grid,count-1,cancelled,mountain_indices))return false;\n            }else\n'+anchor)
        target=OUT/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
        sources[relative]={'control':h,'candidate':sha(target)}
    relative='Renderer/lab/shared/natural/surface_refinement.h'
    origin=ROOT/'Renderer/tools/redraw_density_candidate.h'
    target=OUT/'sources'/relative
    text=origin.read_text().replace('#include "../lab/shared/natural/vertex.h"','#include "vertex.h"')
    target.write_text(text)
    sources[relative]={'candidate':sha(target),'fixture_source':str(origin.relative_to(ROOT)),
        'fixture_sha256':sha(origin)}
    build=(PREVIOUS/'candidate/build.bat').read_text().replace(
        '..\\.cache\\redraw-submission-step\\candidate','..\\..\\..\\candidate')
    (OUT/'candidate/build.bat').write_text(build)
    (OUT/'candidate-sources.json').write_text(json.dumps(sources,indent=2)+'\n')


def build_candidate():
    batch=windows_root()/OUT.relative_to(ROOT).as_posix()/'candidate/build.bat'
    result=native_command_result(str((OUT/'sources/Renderer/native').relative_to(ROOT)),
        'call "'+str(batch)+'"',timeout_seconds=600)
    receipt=OUT/'candidate/build-result.json';attempt=1
    while receipt.exists():
        attempt+=1;receipt=OUT/f'candidate/build-result-{attempt}.json'
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise ValueError('Build failed')


def run(label, detail='normal', hour=12, frames=180, counts=False, captures=False,arm='control'):
    navigation.OUT=OUT
    directory=windows_root()/OUT.relative_to(ROOT).as_posix()/label
    options={'C3X_RENDERER_PATCH_PIXELS':PIXELS[detail],
        'C3X_SANDBOX_PASS_COUNTS':'1' if counts else '0',
        'C3X_RENDERER_TRACE':'2' if counts else '0','C3X_RENDERER_PROFILE':'0',
        'C3X_SANDBOX_PROFILE_COMPLETION':'0','C3X_RENDERER_OUTPUT_COMPLETION_PROBE':'0',
        'C3X_SANDBOX_GPU_TIMESTAMPS':'0','C3X_SANDBOX_FRAME_TIMINGS':'1',
        'C3X_SANDBOX_WITNESS_SCENARIO':'zoom','C3X_SANDBOX_ZOOM_CHECKPOINTS':'0',
        'C3X_SANDBOX_WORLD_CONTENT_DIR':str(directory) if counts else '',
        'C3X_RENDERER_TRACE_FILE':str(directory/'renderer.log') if counts else '',
        'C3X_RENDERER_TRACE_BUFFERED':'0',
        'C3X_SANDBOX_CAUSAL_MODE':'0','C3X_SANDBOX_CAUSAL_LEDGER':'0'}
    result=navigation.run(label,arm,'navigation',hour,frames,options,captures,'full_guest','client')
    result['density_control']=detail
    (OUT/label/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    archive_outputs(OUT/label)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','prepare-candidate','build-candidate','run'));p.add_argument('--label')
    p.add_argument('--detail',choices=tuple(PIXELS),default='normal')
    p.add_argument('--hour',type=int,default=12);p.add_argument('--frames',type=int,default=180)
    p.add_argument('--counts',action='store_true');p.add_argument('--captures',action='store_true')
    p.add_argument('--arm',choices=('control','candidate'),default='control')
    a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='prepare-candidate':prepare_candidate()
    elif a.action=='build-candidate':build_candidate()
    else:run(a.label,a.detail,a.hour,a.frames,a.counts,a.captures,a.arm)
