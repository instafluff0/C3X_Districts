"""Portable regression for Q0's incident-edge derivative adapter."""
from pathlib import Path
import subprocess
import tempfile
import unittest

class ContinuousNormalTests(unittest.TestCase):
 def test_world_stencil_crosses_incident_edges_and_preserves_slope(self):
  header=Path(__file__).resolve().parents[2]/'systems/relief/continuous_normal.h'
  source=r'''
#include "continuous_normal.h"
#include <cassert>
int main() {
 auto h=[](double x,double y){return 128*(.4*x-.7*y+.1*x*x);};
 for(int row=0;row<4;row++)for(int col=0;col<3;col++)for(int i=0;i<=16;i++) {
  double along=i/16.;
  auto left=q4_relief::continuous_normal(col+1.,row+along,.006,128,h);
  auto right=q4_relief::continuous_normal((col+1)+0.,row+along,.006,128,h);
  assert(left==right);
  double slope=.4+.2*(col+1);double length=std::sqrt(slope*slope+.49+1);
  assert(std::abs(left[0]+slope/length)<1e-11);
  assert(std::abs(left[1]-.7/length)<1e-11);
  auto local=q4_relief::to_local_uv(left);assert(local[1]==-left[1]);
 }
 bool rejected=false;
 try {q4_relief::continuous_normal(0,0,0,128,h);}catch(std::invalid_argument const&){rejected=true;}
 assert(rejected);
}
'''
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'test.cpp').write_text(source)
   subprocess.run(['clang++','-std=c++17','-I',str(header.parent),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
   subprocess.run([str(p/'test')],check=True)

if __name__=='__main__':unittest.main()
