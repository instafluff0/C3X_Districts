"""Execute local source AO bytecode offline; runtime never consumes DXBC source."""
from pathlib import Path
import os
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'Renderer/tools'))
import renderer_dev

if __name__=='__main__':
    out=ROOT/'Renderer/terrain_lab/v2/audits/beauty/out/ground-shader-source'
    for name in ['grassland','plains','desert','marsh','tundra']:
        path=ROOT/'Renderer/packs/Civ5EnvironmentSkin/textures'/(name+'_height.dds')
        a=np.array(Image.open(path).convert('L').resize((1024,1024),Image.Resampling.BOX),dtype=np.float32)/255
        np.pad(a,8,mode='wrap').astype('<f4').tofile(out/(name+'-ao-high.f32'))
        half=(a[::2,::2]+a[1::2,::2]+a[::2,1::2]+a[1::2,1::2])*.25
        np.pad(half,4,mode='wrap').astype('<f4').tofile(out/(name+'-ao-half.f32'))
    os.environ.setdefault('C3X_RENDERER_WINDOWS_ROOT',str(renderer_dev.windows_live_target()))
    result=renderer_dev.native_command_result('Renderer/terrain_lab/v2/qa','call bake_source_occlusion.bat')
    raise SystemExit(0 if result['status']=='pass' else 1)
