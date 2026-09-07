"""Execute local source AO bytecode offline; runtime never consumes DXBC source."""
from pathlib import Path
import os
import json
import hashlib
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'Renderer/tools'))
import renderer_dev

if __name__=='__main__':
    out=ROOT/'Renderer/terrain_lab/v2/audits/beauty/out/ground-shader-source'
    for name in ['grassland','plains','desert','marsh','tundra','grassland_shift','flat_control']:
        pack=ROOT/'Renderer/packs/Civ5EnvironmentSkin'
        material=json.loads((pack/'materials'/((name if name not in ('grassland_shift','flat_control') else 'grassland')+'.json')).read_text())
        path=pack/material['height']['texture']
        a=np.array(Image.open(path).convert('L').resize((1024,1024),Image.Resampling.BOX),dtype=np.float32)/255
        if name=='grassland_shift':a=np.roll(a,(-10,-18),(0,1))
        if name=='flat_control':a.fill(.4)
        np.pad(a,8,mode='wrap').astype('<f4').tofile(out/(name+'-ao-high.f32'))
        half=(a[::2,::2]+a[1::2,::2]+a[::2,1::2]+a[1::2,1::2])*.25
        np.pad(half,4,mode='wrap').astype('<f4').tofile(out/(name+'-ao-half.f32'))
    os.environ.setdefault('C3X_RENDERER_WINDOWS_ROOT',str(renderer_dev.windows_live_target()))
    result=renderer_dev.native_command_result('Renderer/terrain_lab/v2/qa','call bake_source_occlusion.bat')
    if result['status']=='pass':
        read=lambda name:np.fromfile(out/(name+'-ao.rgba8'),dtype=np.uint8).reshape(1024,1024,4)
        original=read('grassland');shift=read('grassland_shift')
        delta=np.abs(np.roll(original,(-10,-18),(0,1)).astype(int)-shift.astype(int))
        assert delta.max()<=1, 'AO cache depends on dispatch block or periodic crop'
        assert np.all(read('flat_control')[:,:,1]==255), 'flat ground must have no height occlusion'
        records=[]
        for name in ['grassland','plains','desert','marsh','tundra']:
            records.append({'material':name,'output_sha256':hashlib.sha256((out/(name+'-ao.rgba8')).read_bytes()).hexdigest(),
                'high_input_sha256':hashlib.sha256((out/(name+'-ao-high.f32')).read_bytes()).hexdigest(),
                'half_input_sha256':hashlib.sha256((out/(name+'-ao-half.f32')).read_bytes()).hexdigest()})
        evidence={'classification':'original source compute under explicit Lab inputs; source engine runtime constants and layer composition unproven',
            'shader_sha256':hashlib.sha256((out/'shader-00426f26.dxbc').read_bytes()).hexdigest(),
            'periodic_shift':[18,10],'periodic_shift_max_rgba8_error':int(delta.max()),
            'flat_control_ao':1,'pixels_per_control':1048576,'records':records}
        (out.parent.parent/'SOURCE_OCCLUSION_EVIDENCE.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print('PASS occlusion flat-ground and periodic-shift controls')
    raise SystemExit(0 if result['status']=='pass' else 1)
