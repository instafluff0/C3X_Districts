import unittest
from Renderer.native.native_cpp_test import run_cpp

class TacticalInputTests(unittest.TestCase):
    def test_copied_semantics_clipping_and_bounds(self):
        run_cpp(r'''
#include "Renderer/native/tactical_overlay.h"
#include <cassert>
int main(){using namespace c3x_renderer::tactical;Input a;a.line(-100,20,40,20);a.ring(50,60,128,true);a.label(80,40,"12*",24);
 auto b=a;a.primitives.clear();assert(b.primitives.size()==5&&b.animated);
 auto r=b.extent({0,0,100,80});assert(r[0]==0&&r[2]==100&&r[3]==80);
 bool rejected=false;try{b.line(NAN,0,1,1);}catch(...){rejected=true;}assert(rejected);
 Input c;c.ring(0,0,128,false);assert(!c.animated);
 assert(c.primitives[0].shape[2]==44.f && c.primitives[0].shape[3]==21.5f);
 Input small;small.ring(0,0,64,false);assert(small.primitives[0].shape[2]==22.f);
 Input custom;custom.ring(0,0,192,false);assert(custom.primitives[0].shape[2]==44.f);
 Input selected;selected.ring(0,0,128,true);assert(selected.primitives[0].shape[1]==-2.f);
 assert(c.primitives[0].shape[1]==0.f);
 assert(cursor_phase(0)==cursor_phase(5));
 assert(cursor_phase(1.25)<cursor_phase(2.5) && cursor_phase(3.75)<cursor_phase(2.5));
 assert(cursor_phase(2.5)>1.57f && cursor_phase(2.5)<1.58f);
 return 0;}
''')

    def test_production_gpu_coverage_and_packed_composition(self):
        from Renderer.lab.platform import native_command_result
        result = native_command_result("Renderer/native", "call BUILD.bat tactical")
        self.assertEqual(result["status"], "pass", result["output_tail"])
        self.assertIn("PASS tactical GPU:", result["output_tail"])
