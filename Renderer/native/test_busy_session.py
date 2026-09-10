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
 BusySessionInputs inputs{plan};
 assert(inputs.select(19000000).event==-1);
 int expected[]={160,192,160,128,128,128,128};
 for(int i=0;i<7;++i){
  assert(!inputs.finished(65000000));
  auto request=inputs.select(65000000);
  assert(request.event==i && request.view.width==expected[i] && request.requested_us<=50000000);
 }
 assert(inputs.finished(65000000));
 assert(inputs.select(66000000).event==-1);

 auto replay=fixed_busy_replay(plan,25);
 assert(replay.size()==200);
 std::set<int> replay_phases,replay_widths,replay_events;
 int counts[8]={};long long previous=-1;
 for(auto const& request:replay){
  assert(request.logical_us>previous);previous=request.logical_us;
  replay_phases.insert(request.view.phase);replay_widths.insert(request.view.width);
  if(request.event>=0)replay_events.insert(request.event);
  ++counts[request.view.phase];
 }
 assert(replay_phases.size()==8 && replay_widths==std::set<int>({128,160,192}));
 assert(replay_events.size()==7);
 for(auto count:counts)assert(count==25);
}
''')


if __name__=='__main__':unittest.main()
