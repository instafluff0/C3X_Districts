"""Exercise the same bounded pose-local shadow rasterizer used by the DLL."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

class UnitShadowTests(unittest.TestCase):
    def test_self_occlusion_ground_cast_winding_and_light_direction(self):
        compiler=shutil.which('c++')
        if not compiler:self.skipTest('C++ compiler unavailable')
        code=r'''
#include "unit_shadow.h"
#include <cassert>
using namespace c3x_renderer;
int main() {
    std::vector<UnitShadow::Point> points={{-.3f,-.3f,.5f},{.3f,-.3f,.5f},{0,.3f,.5f}};
    UnitShadow shadow;shadow.fit(points,1,0);
    shadow.triangle(points[0],points[1],points[2]);
    float center=-shadow.dx*.5f;
    assert(shadow.coverage(center,0)>.9f); // cast on ground
    assert(shadow.coverage(center+shadow.dx*.25f,0,.25f)>.9f); // receiver beneath caster
    assert(shadow.coverage(0,0,.5f)==0); // no acne on the caster itself
    assert(shadow.coverage(10,10)==0);
    auto forward=shadow.heights;
    shadow.fit(points,1,0);shadow.triangle(points[2],points[1],points[0]);
    for(unsigned i=0;i<forward.size();++i)assert(std::abs(forward[i]-shadow.heights[i])<1e-5f);
    shadow.fit(points,-1,0);shadow.triangle(points[0],points[1],points[2]);
    assert(shadow.coverage(-center,0)>.9f && shadow.coverage(center,0)==0);
    for(auto& p:points)p[2]=-.5f;
    shadow.fit(points,1,0);shadow.triangle(points[0],points[1],points[2]);
    for(auto height:shadow.heights)assert(height<0);
    points[0][0]=1e30f;assert(!shadow.fit(points,1,0));
    points[0][0]=std::nanf("");assert(!shadow.fit(points,1,0));
}
'''
        with tempfile.TemporaryDirectory() as folder:
            cpp=Path(folder)/'shadow.cpp';exe=Path(folder)/'shadow';cpp.write_text(code)
            subprocess.run([compiler,'-std=c++17','-O2','-Wall','-Wextra','-Werror','-I',str(Path(__file__).parent),str(cpp),'-o',str(exe)],check=True)
            subprocess.run([str(exe)],check=True)
