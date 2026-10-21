"""Resource bodies on a mountain stand on the rock the mesh draws.

The mountain mesh (relief_mesh_body.h) eases the mountain's rise out at the
coast and beside rivers: natural ground + displacement * coast_relief * river
scale * hidden taper. Resource seating (c3x_renderer.cpp resource_height_at)
applied only the hidden taper, so gems and other minerals seated by a river or
on a shore stood on rise that is not drawn and hovered over the bank or beach.
This test compiles the seating statement against the mesh's own factors.
"""
import re
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class ResourceMountainSeatingTests(unittest.TestCase):
    def test_seating_follows_the_drawn_mountain_by_coast_and_river(self):
        mesh = (ROOT / "Renderer/lab/shared/natural/relief_mesh_body.h").read_text()
        # The mesh's factors this test reproduces below.
        self.assertIn("smooth01((float(river_at(px,py))-6.f)/16.f)", mesh)
        self.assertIn("float scale=coast_relief(float(shore.distance),float(shore.beach_width))*\n"
                      "                        mountain_river_scale(world_x,world_y)*hidden_taper_at(world_x,world_y);", mesh)
        self.assertIn("height_natural(world_x,world_y,&support)+\n                        sample.displacement*scale;", mesh)
        renderer = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        start = renderer.index("auto resource_height_at=[&](float u,float v,float* support=nullptr){")
        body = renderer[renderer.index("if(!resource_mountains.count)return height;", start):
                        renderer.index("\n            };", start)]
        run_cpp(r'''
#include "Renderer/native/source_fidelity/coast_join.h"
#include "Renderer/native/render_core/relief_query.h"
#include <cassert>
#include <cmath>
#include <initializer_list>
struct P {float x,y;};
struct Shore {double distance,beach_width;};
struct Natural {
    bool river_world=true;double river=1000;
    struct Sample {double distance;};
    Sample river_sample(P)const {return {river};}
} natural;
struct Queries {
    double distance=5,beach=.1;float taper=1;
    Shore shore(float,float){return {distance,beach};}
    float hidden_taper(float,float){return taper;}
} queries;
struct Mountains {
    unsigned count=1;float displacement=0;
    struct Sample {float displacement;};
    Sample sample(Natural const&,float,float)const {return {displacement};}
} resource_mountains;
float seat(float height,float u,float v){
''' + body + r'''
}
int main(){
    for(float d:{0.f,20.f,60.f})for(double river:{0.,8.,14.,22.,1000.})
    for(double shore:{0.,.2,.4,.7,5.})for(float taper:{1.f,.5f}){
        resource_mountains.displacement=d;natural.river=river;queries.distance=shore;queries.taper=taper;
        float drawn=2.5f+d*c3x_renderer::fidelity::coast_relief(float(shore),.1f)*
            c3x_renderer::render_core::smooth01((float(river)-6.f)/16.f)*taper;
        assert(std::abs(seat(2.5f,.3f,.4f)-drawn)<1e-4f);
    }
    natural.river_world=false;natural.river=0;resource_mountains.displacement=30;queries.distance=5;queries.taper=1;
    assert(std::abs(seat(2.5f,.3f,.4f)-32.5f)<1e-4f);   // no river data: the full rise
}
''')

    def test_routes_and_resources_share_the_mesh_factors(self):
        # Routes already followed the drawn rock; resources now use the same factors.
        preparation = (ROOT / "Renderer/native/object_preparation.h").read_text()
        renderer = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        self.assertIn("coast_relief(float(shore.distance),float(shore.beach_width))*river_scale*queries.hidden_taper(x,y)",
                      preparation)
        seating = renderer[renderer.index("auto resource_height_at=[&](float u,float v,float* support=nullptr){"):]
        seating = seating[:seating.index("\n            };")]
        self.assertTrue(re.search(r"coast_relief\(float\(shore\.distance\),\s*float\(shore\.beach_width\)\)\*river_scale\*"
                                  r"queries\.hidden_taper\(u,v\)", seating))


if __name__ == "__main__":
    unittest.main()
