"""A mirror reused while the camera moves is read at the drawn camera.

For 150 ms after a scroll step the water pass samples the previous camera's
mirror at screen + 4 + shift. With the shift's sign reversed it read the mirror
two steps to the side, and the shallows mirrored land out to a displaced copy
of the coastline until the exact mirror was redrawn (October 17).
"""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class MirrorReuseShiftTests(unittest.TestCase):
    def test_reused_mirror_samples_the_same_world_point(self):
        pipeline = (ROOT / 'Renderer/sandbox/fresh_pipeline.h').read_text()
        # The conventions the model below reproduces: screen and mirror
        # translations and guards, and where the water pass samples.
        for fact in ('settings.translation[0]+=4;settings.translation[1]+=4;',
                     'reflected.translation[0]+=8;reflected.translation[1]+=8;',
                     '4+(mirrored?0.f:reflection_shift[0])', '4+(mirrored?0.f:reflection_shift[1])',
                     'reflection_shift[0]=reused_mirror_shift(camera_x,reflection_camera_x,projection_zoom);',
                     'reflection_shift[1]=reused_mirror_shift(camera_y,reflection_camera_y,projection_zoom);',
                     'float guard=mirrored?8.f:4.f;', '.viewport(viewport,mirrored?8.f:4.f,'):
            self.assertIn(fact, pipeline)
        self.assertIn('(input.position.xy + NativeReflectionTarget.zw)',
                      (ROOT / 'Renderer/sandbox/water_surface.hlsl').read_text())
        self.assertIn('return (pixels + c3x_viewport_translation) *',
                      (ROOT / 'Renderer/native/source_fidelity/hydrology.hlsl').read_text())
        marker = '    static float reused_mirror_shift('
        method = marker + pipeline.split(marker, 1)[1].split('\n    }\n', 1)[0] + '\n    }\n'
        run_cpp(r'''
#include "Renderer/native/scene_projection.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
struct Pipeline {
''' + method + r'''
};
constexpr unsigned W=2240,H=1260;
struct Viewport {float TopLeftX=0,TopLeftY=0,Width=0,Height=0;};
// A canonical vertex pixel through translated_position and the target
// viewport, as clip_settings and target_viewport set them up.
std::array<float,2> project(float px,float py,float tx,float ty,float guard,float zoom){
 c3x_renderer::SceneProjection projection(W,H,zoom);
 float extent_x=W+2*guard,extent_y=H+2*guard;
 float translation[2]={tx,ty},inverse[2]={1/extent_x,1/extent_y};
 float margin_x=(1/inverse[0]-W-2*guard)*.5f,margin_y=(1/inverse[1]-H-2*guard)*.5f;
 projection.clip_transform(translation,inverse,guard,margin_x,margin_y);
 Viewport viewport{0,0,extent_x,extent_y};projection.viewport(viewport,guard,0,0,1.f);
 float ndc_x=(px+translation[0])*inverse[0]*2-1,ndc_y=(py+translation[1])*inverse[1]*-2+1;
 return {viewport.TopLeftX+(ndc_x+1)*.5f*viewport.Width,viewport.TopLeftY+(1-ndc_y)*.5f*viewport.Height};
}
int main(){
 int const steps[][2]={{64,0},{-64,0},{0,32},{0,-32},{64,32},{-64,-32},{128,64},{-128,96}};
 unsigned checked=0;
 for(float zoom:{.5f,.625f,.75f,1.f,1.25f,2.f,3.f})for(auto const& step:steps)
  for(int camera_x:{-2400,-64,5000})for(int camera_y:{-300,0,700}){
   int mirror_x=camera_x-step[0],mirror_y=camera_y-step[1];
   float shift_x=Pipeline::reused_mirror_shift(camera_x,mirror_x,zoom);
   float shift_y=Pipeline::reused_mirror_shift(camera_y,mirror_y,zoom);
   for(float px=2600;px<=4800;px+=550)for(float py=300;py<=1500;py+=300){
    auto screen=project(px,py,float(camera_x)+4,float(camera_y)+4,4,zoom);
    auto mirror=project(px,py,float(mirror_x)+8,float(mirror_y)+8,8,zoom);
    float read_x=screen[0]+4+shift_x,read_y=screen[1]+4+shift_y;
    if(std::fabs(read_x-mirror[0])>.01f||std::fabs(read_y-mirror[1])>.01f){
     std::printf("zoom=%.3f step=%d,%d camera=%d,%d reads %.2f,%.2f, mirror holds the point at %.2f,%.2f\n",
      zoom,step[0],step[1],camera_x,camera_y,read_x,read_y,mirror[0],mirror[1]);
     return 1;
    }
    ++checked;
   }
  }
 // The unshifted mirror (the camera it was drawn at) needs no shift.
 assert(Pipeline::reused_mirror_shift(-2400,-2400,1.75f)==0.f);
 std::printf("checked=%u\n",checked);
}
''')


if __name__ == '__main__':
    unittest.main()
