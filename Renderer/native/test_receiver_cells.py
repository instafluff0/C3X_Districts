"""Mirror receiver cells never reject a rectangle that meets a receiver.

The mirror keeps a record when its reflected bounds share a coarse cell with
any water receiver. Rejection must imply no intersection with every receiver;
rectangles far from water are rejected.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ReceiverCellTests(unittest.TestCase):
    def test_conservative_against_brute_force(self):
        run_cpp(r'''
#include "Renderer/native/render_core/receiver_cells.h"
#include <cassert>
#include <cstdio>
#include <random>
using c3x_renderer::render_core::ReceiverCells;
using Rect=std::array<int,4>;
bool meets(Rect const& a,Rect const& b){return !(a[2]<b[0]||a[0]>b[2]||a[3]<b[1]||a[1]>b[3]);}
int main(){
 std::mt19937 random(11);unsigned rejected=0,queries=0;
 for(int trial=0;trial<60;++trial){
  std::uniform_int_distribution<int> at(-900,2600),extent(1,160);
  Rect bounds={-300+trial,-200,2400,1500-trial};
  std::vector<Rect> receivers;
  for(int i=0;i<trial*3;++i){int x=at(random),y=at(random);receivers.push_back({x,y,x+extent(random),y+extent(random)});}
  ReceiverCells cells;cells.build(bounds,receivers,trial%2?32:48);
  for(int q=0;q<400;++q){int x=at(random),y=at(random);Rect r={x,y,x+extent(random),y+extent(random)};
   bool hit=false;for(auto const& w:receivers){Rect clipped={std::max(w[0],bounds[0]),std::max(w[1],bounds[1]),std::min(w[2],bounds[2]),std::min(w[3],bounds[3])};
    if(clipped[2]>=clipped[0]&&clipped[3]>=clipped[1]&&meets(r,clipped)){hit=true;break;}}
   bool kept=cells.any(r);++queries;
   assert(!hit||kept);
   if(!kept)++rejected;
  }
 }
 ReceiverCells empty;empty.build({0,0,100,100},{});assert(!empty.any({0,0,100,100}));
 ReceiverCells inverted;inverted.build({10,10,0,0},{{0,0,5,5}});assert(!inverted.any({0,0,5,5}));
 assert(rejected>queries/4);
 std::printf("PASS receiver cells: queries=%u rejected=%u conservative=1\n",queries,rejected);
}
''')


if __name__ == '__main__':
    unittest.main()
