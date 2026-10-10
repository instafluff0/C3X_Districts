"""Solid one-pixel axis-aligned native strokes become fills (review, section 49).

The tactical native line covers exactly the half-open pixel run from its start
toward its end (checked on the GPU in test_tactical_overlay.cpp). Each Civ III
city label draws four such strokes; a fill replaces a tactical raster for each.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class NativeStrokeFillTests(unittest.TestCase):
    def test_axis_aligned_strokes_fill_their_runs(self):
        source = (ROOT / 'Renderer/native/native_composition_owner.h').read_text()
        start = source.index('        if(op==C3X_NATIVE_STROKE){')
        body = source[start:source.index('        if(op==C3X_NATIVE_TACTICAL_ROUTE_BEGIN){', start)]
        run_cpp(r'''
#include <cassert>
#include <vector>
struct RECT {long left,top,right,bottom;};
enum {C3X_NATIVE_STROKE=124,C3X_NATIVE_FILL=7,C3X_NATIVE_DC=1};
struct c3x_renderer_native_stroke {int x1,y1,x2,y2,width,dash;unsigned argb;};
struct Fill {RECT r;unsigned color;};std::vector<Fill> fills;int tacticals=0,fill_result=1;
struct Adapter {bool owns(void*){return true;}bool admit(void*){return true;}
 int operation(int op,void*,void*,void const*,void const* to,unsigned color){
  if(op==C3X_NATIVE_FILL){fills.push_back({*static_cast<RECT const*>(to),color});return fill_result;}return 0;}} adapter_value,*adapter=&adapter_value;
struct Tactical {void native_line(float,float,float,float,int,int,unsigned){}};
bool tactical=true;
int tactical_draw(void*,Tactical const&){++tacticals;return 1;}
int stroke(int x1,int y1,int x2,int y2,unsigned argb,int width=1,int dash=0){
 int op=C3X_NATIVE_STROKE;void* image=&adapter_value;c3x_renderer_native_stroke s{x1,y1,x2,y2,width,dash,argb};void const* from=&s;
''' + body + r'''
 return -1;
}
int main(){
 auto run=[&](RECT r,unsigned color){assert(fills.size()==1&&fills[0].r.left==r.left&&fills[0].r.top==r.top&&
  fills[0].r.right==r.right&&fills[0].r.bottom==r.bottom&&fills[0].color==color&&tacticals==0);fills.clear();};
 // City label edges, drawn left to right and top to bottom: the end pixel is not covered.
 assert(stroke(10,5,30,5,0xff000000u)==1);run({10,5,30,6},0x80000000u);
 assert(stroke(10,5,10,26,0xff000000u)==1);run({10,5,11,26},0x80000000u);
 // Reversed strokes cover from their start back to, but not including, their end.
 assert(stroke(30,9,10,9,0xffffffffu)==1);run({11,9,31,10},0x80007fffu);
 assert(stroke(44,20,44,2,0xffffffffu)==1);run({44,3,45,21},0x80007fffu);
 // Anything else keeps the tactical raster.
 for(auto other:{0}){(void)other;
  assert(stroke(0,0,9,9,0xff000000u)==1);assert(stroke(0,0,9,0,0xff000000u,2)==1);
  assert(stroke(0,0,9,0,0xff000000u,1,1)==1);assert(stroke(0,0,9,0,0x80000000u)==1);
  assert(stroke(0,0,9,0,0xff102030u)==1);assert(stroke(4,4,4,4,0xff000000u)==1);}
 assert(fills.empty()&&tacticals==6);tacticals=0;
 // A refused fill falls back to the tactical raster.
 fill_result=0;assert(stroke(10,5,30,5,0xff000000u)==1&&tacticals==1);
}
''')


if __name__ == '__main__':
    unittest.main()
