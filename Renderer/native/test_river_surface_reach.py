"""River-surface distances have no jump on a river tile's far side.

The river corridor is exact only within its bucket reach (57.24 px per tile of
reach): a sample reads only its own cell's bucket, so one tile from a river a
point inside the river tile reads ~55-70 px and a point on its far edge, in
the next cell, reads 1000. The river layer covers the whole river tile, and
that jump inside one row of its triangles defeated the reach cull and widened
the shader's antialiasing into a thin dark line along the tile edge (seen in
game on beaches beside rivers). River surfaces clamp to the exact reach.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPILER = ROOT / "Renderer/native/source_fidelity/ground_compiler.h"

HARNESS = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include "Renderer/lab/shared/natural/queries.h"
using namespace c3x_renderer;
REACH
int main(){
 fidelity::NaturalWorld natural;
 natural.fields.resize(1);auto& f=natural.fields[0];f.width=f.height=4;
 for(unsigned i=0;i<16;++i)f.pixels.push_back(std::uint8_t((i*37+11)%256));
 f.minimum=11/255.f;f.maximum=1;natural.terrain[14]=0;
 render_core::World dims{48,64,true,true};
 std::vector<unsigned> tiles(48*64/2,2|(2<<8));
 auto at=[&](int c,int r)->unsigned&{int x=render_core::mod(c+r,48),y=render_core::mod(c-r,64);return tiles[(y*48+x)/2];};
 // A straight river on node row 12: tiles (c,11) and (c,12) both carry it.
 for(int c=10;c<24;++c){at(c,12)|=32u<<16;at(c,11)|=2u<<16;}
 render_core::WorldCoast coast;coast.update(dims,tiles.data(),tiles.size(),1);
 natural.update_rivers(coast.world(),1);
 auto surface=[&](double x,double y){return river_surface_reach(float(natural.river_sample({x,y}).distance));};
 // River-layer vertices of an owner (c,r) are world (c+u, r+1-v) on a 32 grid.
 // Owner (c,12)'s far side is world v=13; owner (c,11)'s is world v=11.
 float worst=0,raw_worst=0;unsigned sides=0;
 for(int c=12;c<22;++c)for(int k=0;k<=32;++k){
  double u=c+k/32.;
  for(auto side:{std::array<double,2>{13.,13.-1/32.},{11.,11.+1/32.}}){
   float edge=surface(u,side[0]),inside=surface(u,side[1]);
   worst=std::max(worst,std::abs(edge-inside));
   raw_worst=std::max(raw_worst,float(std::abs(natural.river_sample({u,side[0]}).distance-natural.river_sample({u,side[1]}).distance)));
   ++sides;
  }
 }
 std::printf("far-side jump: raw %.1f px, river surface %.1f px over %u samples\n",raw_worst,worst,sides);
 // The corridor itself still reports the jump; the river surface must not.
 assert(raw_worst>500);
 assert(worst<6);
 // Inside the exact reach the river surface keeps the corridor's distance.
 unsigned kept=0;
 for(double v=11.5;v<=12.5;v+=1/64.)for(double u=12;u<22;u+=1/16.){
  double d=natural.river_sample({u,v}).distance;
  if(d<50){assert(surface(u,v)==float(d));++kept;}
 }
 assert(kept>1000);
}
'''


class RiverSurfaceReachTests(unittest.TestCase):
    def test_river_tile_far_side_has_no_distance_jump(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = COMPILER.read_text()
        if "inline float river_surface_reach(" in source:
            reach = "inline float river_surface_reach(" + source.split(
                "inline float river_surface_reach(", 1)[1].split("\n}\n", 1)[0] + "\n}\n"
            reach = reach.replace("::river::", "river::")
        else:
            reach = "inline float river_surface_reach(float distance) { return distance; }\n"
        program = HARNESS.replace("REACH", reach)
        with tempfile.TemporaryDirectory(prefix="c3x-river-reach-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O1", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_river_surfaces_use_the_reach(self):
        # Every river-surface vertex distance (river layer, and the river_near
        # underlay/land/shadow attribute) passes through the clamp.
        source = COMPILER.read_text()
        vertex = source.split("auto make_ground_vertex = ", 1)[1].split("auto append_ground_layer", 1)[0]
        assignment = vertex.split("float river_surface_distance =", 1)[1].split(";", 1)[0]
        self.assertIn("river_surface_reach(river_distance(tile, u, v))", assignment)


if __name__ == "__main__":
    unittest.main()
