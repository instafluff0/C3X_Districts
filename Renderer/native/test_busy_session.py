"""The independent input clock must cover busy phases and preserve camera identity."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class BusySessionTests(unittest.TestCase):
    def test_wall_clock_plan_boundaries_return_zoom_and_skipped_slots(self):
        run_cpp(r'''
#include <cassert>
#include <set>
#include "Renderer/native/busy_session_plan.h"
int main(){
 using namespace c3x_renderer;
 BusySessionPlan plan{75,39,100,100};
 assert(plan.at(9999999).phase==0 && plan.at(10000000).phase==1);
 assert(plan.at(19999999).phase==1 && plan.at(20000000).width==160);
 assert(plan.at(22000000).width==192 && plan.at(26000000).width==128);
 assert(plan.at(28000000).phase==3 && plan.at(40000000).phase==5);
 assert(plan.at(50000000).same_camera(plan.at(0)));
 std::set<int> phases,widths;
 for(long long us=0;us<BusySessionPlan::duration_us;us+=BusySessionPlan::slot_us){
  auto view=plan.at(us);phases.insert(view.phase);widths.insert(view.width);
  assert(view.x%2==1 && view.y%2==1 && view.x>=0 && view.x<100 && view.y<100);
 }
 assert(phases.size()==8 && widths==std::set<int>({128,160,192}));
 // A stalled renderer samples the latest independent input, without advancing
 // one scripted frame and pretending the user waited for it.
 auto before=plan.at(27000000),after=plan.at(35000000);
 assert(!before.same_camera(after) && after.phase==4);
 assert(35000000/BusySessionPlan::slot_us-27000000/BusySessionPlan::slot_us-1>200);
}
''')


if __name__=='__main__':unittest.main()
