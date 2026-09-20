"""Prove coverage rejection against uploaded samples and the current shader contract."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class WaterCoverageTests(unittest.TestCase):
    def test_actual_samples_keep_shores_water_and_unknown_values(self):
        run_cpp(r'''
#include <cassert>
#include <vector>
#include <limits>
#include "Renderer/lab/shared/natural/vertex.h"
#include "Renderer/native/render_core/water_coverage.h"
int main() {
 using c3x_renderer::render_core::water_surface_can_contribute;
 std::vector<c3x_renderer::fidelity::MapVertex> samples(289);
 for(auto& v:samples)v.shore_true_distance=.25f;
 assert(!water_surface_can_contribute(samples));
 for(float distance:{-.01f,0.f,.01f,-100.f,std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
   samples[137].shore_true_distance=distance;
   assert(water_surface_can_contribute(samples));
 }
 samples[137].shore_true_distance=.01001f;
 assert(!water_surface_can_contribute(samples));
 // A single interior wet sample must keep a lake even when all edges are dry.
 samples[144].shore_true_distance=-.0002f;
 assert(water_surface_can_contribute(samples));
 samples.clear();assert(!water_surface_can_contribute(samples));
}
''')

    def test_projected_forward_layers_keep_water_overlap(self):
        run_cpp(r'''
#include <cassert>
#include <limits>
#include "Renderer/native/render_core/water_coverage.h"
struct World {
 int wet_c=8,wet_r=0;unsigned wet=12;
 int index(int c,int r)const{return (r+100)*256+c+100;}
 unsigned at(int i)const{return i==index(wet_c,wet_r)?wet:2;}
};
struct Bounds {float low[3],high[3];};
int main(){
 using c3x_renderer::render_core::water_under_projection;
 World world;Bounds ground{{0,0,0},{1,1,0}};
 assert(!water_under_projection(world,ground));
 Bounds tower{{0,0,0},{1,1,3}};
 assert(water_under_projection(world,tower));
 world.wet_c=1;assert(water_under_projection(world,ground));
 world.wet=2|(32u<<16);assert(water_under_projection(world,ground));
 world.wet=0xffffffffu;assert(water_under_projection(world,ground));
 world.wet_c=80;Bounds broad{{0,0,0},{70,1,0}};
 assert(water_under_projection(world,broad));
 ground.high[2]=std::numeric_limits<float>::quiet_NaN();
 assert(water_under_projection(world,ground));
}
''')

    def test_shader_distance_and_discard_contract(self):
        for directory in ("city_fidelity", "environment_refresh"):
            shader=(ROOT / f"Renderer/native/{directory}/hydrology.hlsl").read_text()
            water=shader.split("float4 q3_water_material(PixelInput input)",1)[1].split("// Optical absorption",1)[0]
            self.assertIn("float sd=input.hydrology_data.x",water)
            self.assertIn("clip(-sd-.0001);",water)
            self.assertIn("if(kind<4.5)return",water)
            # VS passes the uploaded distance through without displacement.
            vertex=shader.split("float4 PSIntegrated",1)[0].rsplit("output.hydrology_data",1)[1]
            self.assertTrue(vertex.startswith(" = input.hydrology_data;"))
        cpp=(ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        coverage=cpp.index("bool water_coverage=")
        upload=cpp.index("if(!cache_geometry_layer(mesh_upload,part.vertices",coverage)
        self.assertLess(coverage,upload)
        self.assertIn("if(!water_coverage)continue;",cpp[coverage:cpp.index("bool natural_layer=", coverage)])


if __name__ == "__main__":
    unittest.main()
