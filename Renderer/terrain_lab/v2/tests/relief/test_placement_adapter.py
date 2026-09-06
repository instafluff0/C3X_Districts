from pathlib import Path
import subprocess
import tempfile
import unittest

class PlacementAdapterTests(unittest.TestCase):
 def test_actual_crown_wrap_and_local_stability(self):
  header=Path(__file__).resolve().parents[2]/'systems/relief/placement_adapter.h'
  code=r'''
#include "placement_adapter.h"
#include <cassert>
int main(){
 using namespace q4_placement;
 envelopes.push_back({.1,100,0,{{0,0},{2,0},{2,2},{0,2}}});
 float crown[]={-3,0,0, 1,0,5, 1,1,5, -3,1,0};
 assert(!accept_vegetation("forest","source",1,1,crown,4));
 for(int i=0;i<4;i++)crown[i*3]+=100;
 assert(!accept_vegetation("forest","source",1,1,crown,4));
 for(int i=0;i<4;i++)crown[i*3]+=10;
 assert(accept_vegetation("forest","source",1,1,crown,4));
 float contained[]={.5f,.5f,0,1.5f,.5f,2,1.5f,1.5f,1,.5f,1.5f,0};
 assert(!accept_vegetation("forest","source",1,1,contained,4));
 assert(hull({{0,0},{1,0},{1,1},{0,1},{.5,.5},{0,0}}).size()==4);
}
'''
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);(p/'test.cpp').write_text(code)
   subprocess.run(['clang++','-std=c++17','-I',str(header.parent),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
   subprocess.run([str(p/'test')],check=True)
