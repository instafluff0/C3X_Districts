"""Dispatch source compute shader on a deterministic synthetic height fixture."""
from pathlib import Path
import math
import os
import struct
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'Renderer/tools'))
import renderer_dev

if __name__=='__main__':
    out=ROOT/'Renderer/terrain_lab/v2/audits/beauty/out/ground-shader-source'
    field=[.4+.12*math.sin(x*.8)+.08*math.cos(y*.4)+.001*x*y for y in range(34) for x in range(34)]
    (out/'normal-probe-input.f32').write_bytes(struct.pack('<'+str(len(field))+'f',*field))
    os.environ.setdefault('C3X_RENDERER_WINDOWS_ROOT',str(renderer_dev.windows_live_target()))
    result=renderer_dev.native_command_result('Renderer/terrain_lab/v2/qa','call probe_source_normal.bat')
    raise SystemExit(0 if result['status']=='pass' else 1)
