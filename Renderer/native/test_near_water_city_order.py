"""Near water, city bodies stay ordered after the routes.

Records whose bounds reach water leave the static pass for the retained
near-water overlay, drawn later against the static depth. City bodies used to
be exempt (opaque, so the static color covers the water under them), so a
road beside a lake drew after the city it ran through: its strip, 6.5 units
over the ground, painted over shrubs and the feet of walls (Beijing beside its
lake, 2026-10-27). This runs the production flag statement and the fresh
pipeline's draw order.
"""
import re
import subprocess
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_fresh_shared_submission import method

FLAG_START = "chunk.water_dependent=layer==geometry_water"
FLAG_END = "chunk.tile_x=record.tile_x;"
OLD_EXEMPTION = """                if(city_profile && source_chunk.city_material!=0xffffffffu &&
                   !cities.library.materials[source_chunk.city_material].ground)
                    chunk.water_dependent=false;
"""


def flag_statement(cpp):
    start = cpp.index(FLAG_START)
    return cpp[start:cpp.index(FLAG_END, start)]


def layer_enum(cpp):
    start = cpp.index("enum GeometryLayer : std::size_t {")
    return cpp[start:cpp.index("};", start) + 2]


def scene_order(fresh):
    """Layers in draw_scene's order, up to the wall layer."""
    body = method(fresh, "    bool draw_scene(")
    return re.findall(r"geometry_(?:route|feature|site|mine|farm|city|wall)\b", body)


def program(enum, flag, layers):
    return r'''
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <vector>
namespace c3x_renderer { namespace fidelity { constexpr std::size_t max_natural_bodies=4; }
namespace render_core { bool water_under_projection(bool wet,int){return wet;} } }
''' + enum + r'''
struct Coast {bool wet;bool world()const{return wet;}};
struct Material {bool ground;};
struct Library {std::vector<Material> materials{{false},{true}};};
struct Cities {Library library;};
struct Source {int world_bounds=0;unsigned city_material=0xffffffffu;};
struct Chunk {bool water_dependent=false;};
bool flagged(std::size_t layer,unsigned city_material,bool wet){
    Coast world_coast{wet};Cities cities;bool city_profile=true;(void)cities;(void)city_profile;
    Source source_chunk;source_chunk.city_material=city_material;Chunk chunk;
    ''' + flag + r'''
    return chunk.water_dependent;
}
int main(){
    // Beside a lake the route joins the near-water pass, and so must every
    // layer the scene draws after it there: features, sites, mines and the
    // city's bodies (material 0) and plate (material 1, ground).
    assert(flagged(geometry_route,0xffffffffu,true));
    std::size_t const later[]={''' + layers + r'''};
    for(auto layer:later){
        assert(flagged(layer,0xffffffffu,true));
        if(layer==geometry_city){assert(flagged(layer,1,true));assert(flagged(layer,0,true));}
    }
    // Away from water everything stays in the static pass.
    assert(!flagged(geometry_route,0xffffffffu,false));
    assert(!flagged(geometry_city,0,false));
    std::puts("PASS near-water city bodies follow the routes");
}
'''


class NearWaterCityOrderTests(unittest.TestCase):
    def setUp(self):
        self.cpp = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        self.fresh = (ROOT / "Renderer/sandbox/fresh_pipeline.h").read_text()

    def later_layers(self):
        order = scene_order(self.fresh)
        route = order.index("geometry_route")
        self.assertLess(route, order.index("geometry_city"), "routes draw before city bodies")
        return ",".join(dict.fromkeys(order[route + 1:]))

    def test_city_bodies_beside_water_draw_in_the_routes_pass(self):
        run_cpp(program(layer_enum(self.cpp), flag_statement(self.cpp), self.later_layers()))

    def test_the_old_exemption_fails(self):
        statement = flag_statement(self.cpp)
        old = statement.rstrip(" ") + OLD_EXEMPTION + "                "
        # It compiles and then fails its assertions (not a build error).
        with self.assertRaises((subprocess.CalledProcessError, AssertionError)) as caught:
            run_cpp(program(layer_enum(self.cpp), old, self.later_layers()))
        failure = caught.exception
        if isinstance(failure, subprocess.CalledProcessError):
            self.assertNotIn("-std=c++17", failure.cmd)
        else:
            self.assertIn("Assertion failed", str(failure))

if __name__ == "__main__":
    unittest.main()
