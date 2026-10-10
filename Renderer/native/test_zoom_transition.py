import unittest
from Renderer.native.native_cpp_test import run_cpp


class ZoomTransitionTests(unittest.TestCase):
    def test_clock_reversals_bounds_and_presented_picking(self):
        run_cpp(r'''
#include "Renderer/native/zoom_transition.h"
#include <cassert>
#include <limits>
using c3x_renderer::ZoomTransition;
int main(){
 for(double target:{.5,.625,.75,.875,1.25,1.5,1.75,2.,2.5,3.}){
  ZoomTransition coarse,fine;coarse.target(target,1000,1000);fine.target(target,1000,1000);
  for(int t=1001;t<=1250;++t)fine.sample(t,1000);
  assert(std::abs(coarse.sample(1250,1000)-fine.current())<1.e-12);
  assert(std::abs(coarse.current()-target)<.002);
  assert(target<1.?coarse.current()>target:coarse.current()<target);
  assert(coarse.sample(1240,1000)==coarse.current()); // clock regression holds
  coarse.sample(100000,1000);assert(coarse.current()==target&&!coarse.moving());
 }
 ZoomTransition zoom;zoom.target(1.5,0,1000000);
 double before=zoom.sample(50000,1000000);
 zoom.target(1.,50000,1000000);assert(zoom.current()==before);
 assert(zoom.sample(50001,1000000)>before); // reversal decelerates without snapping
 for(int t=50002;t<400000;t+=137){double s=zoom.sample(t,1000000);assert(s>=1.&&s<=1.5);}
 zoom.sample(1000000,1000000);assert(zoom.current()==1.&&!zoom.moving());
 zoom.target(1.5,1000000,1000000);double selected=zoom.sample(1060000,1000000);
 assert(zoom.last_presented()==1.);zoom.did_present(selected);
 zoom.sample(1070000,1000000);assert(zoom.last_presented()==selected);
 for(double center:{0.,631.,1120.})for(double point:{-400.,0.,2240.,9000.}){
  auto p=zoom.project(point,center);
  assert(std::abs(zoom.unproject(p,center)-point)<1.e-9);
 }
 for(int t=1080000;t<2000000;t+=1000){zoom.target(t%3000?1.:1.5,t,1000000);assert(zoom.current()>=1.&&zoom.current()<=1.5);}
 ZoomTransition a,b;a.target(1.5,0,1000);b.target(1.5,0,1000);
 for(int t=1;t<=200;++t){a.target(1.5,t,1000);b.sample(t,1000);}
 assert(std::abs(a.current()-b.current())<1.e-12); // duplicate wheel endpoint does not restart
 zoom.reset();assert(zoom.current()==1.&&zoom.target()==1.&&zoom.last_presented()==1.&&!zoom.moving());
 for(double bad:{.49,3.01,std::numeric_limits<double>::quiet_NaN()}){
  bool rejected=false;try{zoom.target(bad,0,1000);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
 }
 bool rejected=false;try{zoom.sample(0,0);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
}
''')


    def test_a_hinted_request_moves_in_the_first_frame_after_it(self):
        # The hint is applied at the start of a display frame. Sampling at
        # that frame's time first left the frame unmoved; the visible change
        # waited for the next frame, 30-45 ms later near 1x (review, 42).
        run_cpp(r'''
#include "Renderer/native/zoom_transition.h"
#include <cassert>
using c3x_renderer::ZoomTransition;
int main(){
 ZoomTransition old_way,hinted;
 for(auto* z:{&old_way,&hinted}){z->sample(1000,1000);z->sample(1020,1000);} // frames at rest, 20 ms apart
 old_way.target(1.25,1040,1000);hinted.retarget(1.25);                         // request at the next frame's start
 assert(old_way.sample(1040,1000)==1.);                                        // the old first frame does not move
 double first=hinted.sample(1040,1000);assert(first>1.01 && first<1.25);       // the hinted one does
 assert(hinted.target()==1.25);hinted.sample(5000,1000);assert(hinted.current()==1.25&&!hinted.moving());
 bool rejected=false;try{hinted.retarget(3.5);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
}
''')

    def test_late_ordered_requests_never_undo_a_newer_hint(self):
        # The shared hint applies a wheel request at the next display frame;
        # its ordered copy arrives 150-250 ms later on busy maps (review, 42).
        run_cpp(r'''
#include "Renderer/native/zoom_transition.h"
#include <cassert>
int main(){
 c3x_renderer::ZoomRequests requests;
 assert(requests.accept(1));                       // hint for notch 1
 assert(requests.accept(2));                       // hint for notch 2
 assert(!requests.accept(1)&&!requests.accept(2)); // their ordered copies arrive late
 assert(requests.accept(3));                       // notch 3 ordered before its hint
 assert(!requests.accept(3));                      // the hint repeats it: no restart
 assert(requests.accept(0)&&requests.applied==3);  // recorded unsequenced inputs still apply
}
''')


if __name__ == "__main__":
    unittest.main()
