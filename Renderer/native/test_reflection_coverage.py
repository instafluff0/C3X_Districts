"""Conservative reflection coverage for camera zoom and water distortion."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ReflectionCoverageTests(unittest.TestCase):
    def test_every_visible_water_sample_remains_covered(self):
        source = (Path(__file__).resolve().parents[1] / 'sandbox/fresh_pipeline.h').read_text()
        method = '    D3D11_RECT reflected_water_bounds(' + source.split('    D3D11_RECT reflected_water_bounds(', 1)[1].split('    ViewportShaderSettings clip_settings(', 1)[0]
        run_cpp(r'''
#include "Renderer/native/scene_projection.h"
#include <array>
#include <vector>
#include <cassert>
using LONG=int;
struct D3D11_RECT {int left,top,right,bottom;};
struct ViewportShaderSettings {float translation[2];};
struct Record {D3D11_RECT area;int translation_x=0,translation_y=0;};
struct GeometryDrawReference {Record const& r;GeometryDrawReference(Record const& a):r(a){} D3D11_RECT const& bounds()const{return r.area;}};
enum {geometry_water,geometry_river};
struct Pipeline {
 struct {unsigned content_view_width=2240,content_view_height=1260;} renderer;
 float projection_zoom=1;
 std::array<std::vector<Record>,2> water_visible;
''' + method + r'''
};
int main(){
 Pipeline p;ViewportShaderSettings settings{{4,4}};
 auto empty=p.reflected_water_bounds(settings,2248,1268);assert(empty.left>=empty.right);
 for(float zoom:{.5f,.625f,.75f,.875f,1.f,1.125f,1.5f,2.f,2.75f,3.f})for(int wrap:{-6400,0,6400}){
  p.projection_zoom=zoom;
  p.water_visible[geometry_water]={{{50-wrap,80,750-wrap,360},wrap,0}};
  p.water_visible[geometry_river]={{{1000,590,1320,630},0,0}};
  auto clip=p.reflected_water_bounds(settings,2248,1268);
  c3x_renderer::SceneProjection projection(2240,1260,zoom);
  for(auto const& records:p.water_visible)for(auto const& r:records)
   for(int y=r.area.top;y<=r.area.bottom;++y)for(int x=r.area.left;x<=r.area.right;++x){
    float px=projection.x(float(x+r.translation_x))+4;
    float py=projection.y(float(y+r.translation_y))+4;
    if(px<0||px>=2248||py<0||py>=1268)continue;
    // Worst bounded normal displacement, mirror guard and bilinear footprint
    // at the default .375 raster scale. Test both ends of each axis.
    for(float dx:{-62.67f,62.67f})for(float dy:{-32.67f,32.67f}){
     float sx=std::clamp(px+4+dx,0.f,2255.f),sy=std::clamp(py+4+dy,0.f,1275.f);
     assert(sx>=clip.left&&sx<clip.right&&sy>=clip.top&&sy<clip.bottom);
    }
   }
 }
 p.water_visible[geometry_water].clear();p.water_visible[geometry_river]={{{10000,0,10100,40},0,0}};
 auto outside=p.reflected_water_bounds(settings,2248,1268);assert(outside.left>=outside.right);
}
''')


if __name__ == '__main__':
    unittest.main()
