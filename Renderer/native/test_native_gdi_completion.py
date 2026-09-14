"""Native CPU callers must observe completed pixels without flushing our GDI batch."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
ROOT=Path(__file__).resolve().parents[2]

class GdiCompletionTests(unittest.TestCase):
    def test_map_and_unit_completion_before_native_cpu_access(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        map_body=source[source.index('class MapBlitter {'):source.index('\nRendererState renderer;')]
        source=(ROOT/'Renderer/native/unit_body_renderer.h').read_text()
        unit_body=source[source.index('    bool blit_pixels('):source.index('    unsigned keyed_pixels=')]
        storage=source[source.index('    HDC dc=nullptr;'):source.index('    bool ensure(ID3D11Device*')]
        force=ROOT/'Renderer/native/build/gdi-completion-windows.cpp'
        force.write_text('#include <windows.h>\n')
        program=r'''
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/color_quantization.h"
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"msimg32.lib")
struct Trace {int level=1;unsigned spans=0;void write(char const* stage,char const*,bool){if(!std::strcmp(stage,"map-black-span"))++spans;}};
'''+map_body+'\nstruct UnitBlitter {\n'+storage+'\n'+unit_body+'\n~UnitBlitter(){reset_blit();}\n};\n'+r'''
int main(){
 GdiSetBatchLimit(1000);
 BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
 info.bmiHeader.biWidth=64;info.bmiHeader.biHeight=-64;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
 HDC dc=CreateCompatibleDC(nullptr);void* bits=nullptr;
 HBITMAP bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);auto old=SelectObject(dc,bitmap);
 auto pixels=static_cast<std::uint32_t*>(bits);unsigned map_failures=0,unit_failures=0;Trace trace;
 {
 MapBlitter map;UnitBlitter unit;
 std::vector<std::uint32_t> terrain(64*64),body(16*16,0x80000000u);
 c3x_renderer_output_v1 output={};output.width=output.height=64;output.stride_bytes=256;
 output.clip_right=output.clip_bottom=64;output.bgra_pixels=terrain.data();
 for(unsigned i=0;i<100;++i){
  // A native CPU clear/overlay immediately follows each renderer call.
  GdiFlush();std::fill_n(pixels,4096,0xffbadbadu);std::fill(terrain.begin(),terrain.end(),0x00708090u+i);
  if(!map.blit(output,dc,0,0,trace))return 2;
  if((pixels[32*64+32]&0xffffffu)!=(terrain[0]&0xffffffu))++map_failures;
  pixels[0]=0x00010203;GdiFlush();if((pixels[0]&0xffffffu)!=0x00010203)++map_failures;
  std::fill_n(pixels,4096,0x00808080u);unsigned keyed=0;
  if(!unit.blit_pixels(body,16,16,dc,8,8,dc,keyed))return 3;
  if((pixels[8*64+8]&0xffffffu)!=0x00404040u)++unit_failures;
  // Native erase is a CPU write; an outstanding blit must not undo it.
  std::fill_n(pixels,4096,0x00808080u);GdiFlush();
  if((pixels[8*64+8]&0xffffffu)!=0x00808080u)++unit_failures;
 }
 // Diagnostic observes source pixels and never changes their output. Report
 // the appearance/disappearance once, not once per identical map request.
 std::fill(terrain.begin(),terrain.end(),0x00708090u);
 std::fill(terrain.begin()+16*64+8,terrain.begin()+16*64+56,0u);
 if(!map.blit(output,dc,0,0,trace) || trace.spans!=1 || pixels[16*64+8]!=0)return 4;
 if(!map.blit(output,dc,0,0,trace) || trace.spans!=1)return 5;
 std::fill(terrain.begin(),terrain.end(),0x00708090u);
 if(!map.blit(output,dc,0,0,trace) || trace.spans!=2)return 6;
 }
 SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);
 std::printf("native CPU handoff: map_failures=%u unit_failures=%u\n",map_failures,unit_failures);
 return map_failures||unit_failures?1:0;
}
'''
        run_cpp(program,sources=(str(force.relative_to(ROOT)),),timeout=60)

if __name__=='__main__':unittest.main()
