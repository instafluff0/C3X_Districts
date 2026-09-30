"""Untimed full-scene alias versus CopyResource proof using the same GPU frame.

Builds a private diagnostic extension of current production sources. The delivered
renderer/client and timed receipts are unchanged. Requires the retained assignment
inputs under Renderer/.cache/redraw-navigation-step.
"""
from pathlib import Path
import argparse
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from Renderer.lab.platform import native_command_result,windows_root
from Renderer.tools.measure_city_light_index import sha
from Renderer.tools.measure_redraw_navigation import OUT,run


def build():
    directory=OUT/'proof'
    if directory.exists():raise ValueError('Preserve existing proof; use its build receipt')
    expected=json.loads((OUT/'candidate/sources.json').read_text())['source_sha256']
    def require_sources():
        if {name:sha(ROOT/name) for name in expected}!=expected:
            raise ValueError('Current compiled sources differ from the retained timing candidate')
    require_sources()
    (directory/'obj').mkdir(parents=True)
    program=r'''
#include "Renderer/sandbox/resident_scene.cpp"
// Copy exact immutable HDR outputs, rerun the existing bloom/display shaders,
// and retain both images/depths. No scene is regenerated between the two arms.
extern "C" __declspec(dllexport) int c3x_sandbox_copy_parity(char const* prefix){
 if(c3x_sandbox_witness_capture(prefix))return 1;
 using Microsoft::WRL::ComPtr;
 ComPtr<ID3D11Texture2D> copied[2];ComPtr<ID3D11ShaderResourceView> views[2];
 ID3D11Texture2D* sources[]={sandbox_fresh.static_cache.color,sandbox_fresh.glow.linear.color};
 for(unsigned i=0;i<2;++i){
  D3D11_TEXTURE2D_DESC desc={};sources[i]->GetDesc(&desc);
  if(desc.SampleDesc.Count!=1)return 2;
  desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  if(FAILED(renderer.device->CreateTexture2D(&desc,nullptr,&copied[i])) ||
     FAILED(renderer.device->CreateShaderResourceView(copied[i].Get(),nullptr,&views[i])))return 3;
  renderer.context->CopyResource(copied[i].Get(),sources[i]);
 }
 int result=0;
 {
  struct Restore {
   ID3D11ShaderResourceView*& a;ID3D11ShaderResourceView*& b;
   ID3D11ShaderResourceView* old_a;ID3D11ShaderResourceView* old_b;
   ~Restore(){a=old_a;b=old_b;}
  } restore{sandbox_fresh.static_cache.view,sandbox_fresh.glow.linear.view,
            sandbox_fresh.static_cache.view,sandbox_fresh.glow.linear.view};
  sandbox_fresh.static_cache.view=views[0].Get();sandbox_fresh.glow.linear.view=views[1].Get();
  if(!sandbox_fresh.bloom.draw(views[0].Get(),views[1].Get()))return 4;
  std::string copy_prefix=std::string(prefix)+".copy";
  result=c3x_sandbox_witness_capture(copy_prefix.c_str());
 }
 if(!sandbox_fresh.bloom.draw(sandbox_fresh.static_cache.view,sandbox_fresh.glow.linear.view))return 5;
 std::printf("SAME_FRAME_COPY_PROOF prefix=%s result=%d\n",prefix,result);std::fflush(stdout);
 return result;
}
'''
    (directory/'proof.cpp').write_text(program)
    header=(ROOT/'Renderer/sandbox/camera_witness.h').read_text().replace('"c3x_sandbox_witness_capture"','"c3x_sandbox_copy_parity"')
    (directory/'proof-camera-witness.h').write_text(header)
    client=ROOT/'Renderer/sandbox/client_x64.cpp'
    source=client.read_text()
    def include(match):
        path=(client.parent/match[1]).resolve()
        if path.name=='camera_witness.h':return '#include "proof-camera-witness.h"'
        return '#include "'+path.relative_to(ROOT).as_posix()+'"'
    source=re.sub(r'#include "([^"]+)"',include,source)
    (directory/'proof-client.cpp').write_text(source)
    batch=(OUT/'candidate/build.bat').read_text().replace(r'redraw-navigation-step\candidate',r'redraw-navigation-step\proof')
    batch=batch.replace(r'..\sandbox\resident_scene.cpp',r'..\.cache\redraw-navigation-step\proof\proof.cpp')
    batch=batch.replace(r'..\sandbox\client_x64.cpp',r'..\.cache\redraw-navigation-step\proof\proof-client.cpp')
    batch=batch.replace('cl /nologo',f'cl /I "{windows_root()}" /nologo')
    (directory/'build.bat').write_text(batch)
    result=native_command_result('Renderer/native','call "'+str(windows_root()/directory.relative_to(ROOT).as_posix()/'build.bat')+'"',timeout_seconds=180)
    require_sources()
    result['private_source_sha256']={p.name:sha(p) for p in directory.glob('*') if p.suffix in ('.cpp','.h','.bat')}
    result['current_sources']=expected
    if result['status']=='pass':result['binaries']={name:sha(directory/name) for name in ('C3XRenderer_x64.dll','client_x64.exe')}
    (directory/'build-result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='pass':raise RuntimeError('Private proof build failed')


def verify():
    from PIL import Image
    import numpy as np
    rows=[]
    for hour in (12,0):
        label=f'same-frame-copy-proof-h{hour}'
        run(label,arm='proof',client_arm='proof',scenario='navigation',hour=hour,
            frames=3,captures=True,display='full_guest')
        directory=OUT/label
        for original in sorted(directory.glob('*.bmp')):
            if original.stem.endswith('.copy') or original.name=='initial.bmp':continue
            copied=original.with_name(original.stem+'.copy.bmp')
            a=np.array(Image.open(original).convert('RGBA'));b=np.array(Image.open(copied).convert('RGBA'))
            delta=np.abs(a.astype('int16')-b.astype('int16'))
            row=dict(hour=hour,view=original.stem,different_color_pixels=int(np.count_nonzero(delta.any(axis=2))),
                     maximum_channel_error=int(delta.max()),exact_depth=original.with_suffix('.depth').read_bytes()==copied.with_suffix('.depth').read_bytes(),
                     alias_color_sha256=sha(original),copied_color_sha256=sha(copied),alias_depth_sha256=sha(original.with_suffix('.depth')),copied_depth_sha256=sha(copied.with_suffix('.depth')))
            rows.append(row)
    (OUT/'same-frame-copy-parity.json').write_text(json.dumps(rows,indent=2)+'\n')
    if len(rows)!=14 or any(r['different_color_pixels'] or not r['exact_depth'] for r in rows):raise RuntimeError('Same-frame parity failed; preserve evidence')
    print('PASS exact same-frame color/depth: 14 day/night camera/zoom views',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['build','verify']);args=parser.parse_args()
    build() if args.phase=='build' else verify()
