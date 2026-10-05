"""Camera steps slide into place and chain into steady motion.

A published step is first shown at the previous camera's position and slides
linearly to rest; the previous world (underlay) keeps its position relative to
the new one. Consecutive steps at a steady interval carry unfinished offset
forward so motion stays continuous; a long pause restarts the timing.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class PanTransitionTests(unittest.TestCase):
    def test_slides_chain_and_underlay_stays_aligned(self):
        run_cpp(r'''
#include "Renderer/native/pan_transition.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
using c3x_renderer::PanTransition;
int main(){
 const long long f=1000; // ticks per second
 PanTransition pan;assert(!pan.sample(0,f).active);
 // First step: full step offset, underlay (previous world) at rest.
 pan.begin(160,-80,1000,f);auto o=pan.sample(1000,f);
 assert(o.active&&o.x==160&&o.y==-80&&o.under_x==0&&o.under_y==0);
 // Linear slide over 0.9 x first interval (0.6 s): half way at 270 ms.
 o=pan.sample(1270,f);assert(std::abs(o.x-80)<=1&&std::abs(o.y+40)<=1&&o.under_x==o.x-160&&o.under_y==o.y+80);
 assert(!pan.sample(1541,f).active&&!pan.moving());
 // Steady steps every 500 ms: the next step starts before the previous ends
 // only when the interval shrinks; unfinished offset carries forward.
 pan.begin(100,0,2000,f);pan.sample(2100,f);
 pan.begin(100,0,2200,f); // 200 ms later: interval EMA shrinks
 o=pan.sample(2200,f);assert(o.active&&o.x>100&&o.under_x==o.x-100);
 // A long pause restarts with the first-step duration.
 pan.begin(50,0,9000,f);assert(pan.sample(9000+520,f).active&&!pan.sample(9000+541,f).active);
 // Zero steps and cancellation leave nothing active.
 pan.begin(0,0,10000,f);assert(!pan.moving());pan.begin(40,0,11000,f);pan.cancel();assert(!pan.sample(11001,f).active);
 std::printf("PASS pan transition: slide=linear chain=carry restart=1\n");
}
''')


if __name__ == '__main__':
    unittest.main()
