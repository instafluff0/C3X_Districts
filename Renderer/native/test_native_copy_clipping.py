"""Exercise production source clipping for cursor saves at every screen edge."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class NativeCopyClippingTests(unittest.TestCase):
    def test_cursor_saves_and_restores_do_not_require_cpu_pixels(self):
        source = (ROOT / 'Renderer/native/native_image_adapter.h').read_text()
        start = source.index('        if(input&&command.kind==Kind::copy){')
        body = source[start:source.index('        if(input&&command.kind!=Kind::native_image', start)]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
struct Rect{int left,top,right,bottom;};
enum class Kind{copy};
struct Image{unsigned width,height;};
struct Command{Kind kind=Kind::copy;Rect area{},clip{};int source_x=0,source_y=0;};
struct Harness{struct{unsigned translated=0;}counters;
 int clip(Image* input,Command& command,Rect& selected){
''' + body + r'''
 return 2;
 }};
int main(){unsigned cases=0;
 for(auto size:{Image{32,32},Image{2240,1192}}){
  for(int x:{-40,-31,-1,0,1,int(size.width)-31,int(size.width)-1,int(size.width),int(size.width)+1})
   for(int y:{-40,-31,-1,0,1,int(size.height)-31,int(size.height)-1,int(size.height),int(size.height)+1})
    for(int inset:{0,3}){
     Harness h;Command c;c.area={0,0,32,32};c.clip={inset,inset,32-inset,32-inset};
     c.source_x=x;c.source_y=y;Rect selected=c.clip;int result=h.clip(&size,c,selected);bool any=false;
     for(int py=0;py<32;++py)for(int px=0;px<32;++px){
      bool expected=px>=inset&&px<32-inset&&py>=inset&&py<32-inset&&
       px+x>=0&&py+y>=0&&px+x<int(size.width)&&py+y<int(size.height);
      bool actual=result==2&&px>=c.clip.left&&px<c.clip.right&&py>=c.clip.top&&py<c.clip.bottom;
      assert(expected==actual);any|=expected;
     }
     assert((result==2)==any);assert(h.counters.translated==unsigned(!any));++cases;
    }
 }
 std::printf("PASS cursor source clipping: %u edge/empty/destination-clip cases\n",cases);
}
''')


if __name__ == '__main__':
    unittest.main()
