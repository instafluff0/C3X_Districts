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
 for(double target:{1.25,1.5}){
  ZoomTransition coarse,fine;coarse.target(target,1000,1000);fine.target(target,1000,1000);
  for(int t=1001;t<=1250;++t)fine.sample(t,1000);
  assert(std::abs(coarse.sample(1250,1000)-fine.current())<1.e-12);
  assert(coarse.current()>target-.001&&coarse.current()<target);
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
 for(double bad:{.99,1.51,std::numeric_limits<double>::quiet_NaN()}){
  bool rejected=false;try{zoom.target(bad,0,1000);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
 }
 bool rejected=false;try{zoom.sample(0,0);}catch(std::invalid_argument const&){rejected=true;}assert(rejected);
}
''')


if __name__ == "__main__":
    unittest.main()
