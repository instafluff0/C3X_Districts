"""Decode ordinary 8-bit Civ III sprites for renderer-drawn HUD parts.

Stage 4.3: the unit status LED comes from MovementLED.pcx, which
Sprite::slice_pcx stores row-trimmed. The adapter must decode both plain and
trimmed sprites, with indices 254/255 transparent and trimmed margins clear.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class OrdinarySpriteDecodeTests(unittest.TestCase):
    def test_plain_and_row_trimmed_sprites(self):
        source = (ROOT / 'Renderer/native/native_image_adapter.h').read_text()
        start = source.index('    Id ordinary_sprite(')
        body = source[start:source.index('    Counts stats()const', start)]
        run_cpp(r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
enum class Format{rgb555,rgb565,bgra32};
using Id=unsigned;
struct Sprite{int f[0x40/4]{};unsigned char* rows=nullptr;unsigned char* pixels=nullptr;};
namespace c3x_native_access{
 unsigned short table[256];int released=0;
 unsigned char* rows(void* s){return static_cast<Sprite*>(s)->rows;}
 void* pointer(void*,int){return nullptr;}
 void* palette(){return table;}
 unsigned short const* colors(void* p,bool){return static_cast<unsigned short*>(p);}
 unsigned char* sprite_bytes(void* s){return static_cast<Sprite*>(s)->pixels;}
 void release_sprite(void*){++released;}
}
struct Canvas{Format format=Format::rgb565;};
struct Harness{
 Canvas canvas;Id sprite_image=0;std::vector<std::uint32_t> uploaded;
 Canvas* find(void*){return &canvas;}
 bool native_sprite(void*){return true;}
 int field(void* s,int offset){return static_cast<Sprite*>(s)->f[offset/4];}
 bool upload_sprite(std::vector<std::uint32_t>& d,unsigned,unsigned){uploaded=d;sprite_image=7;return true;}
''' + body + r'''
};
Sprite make(int w,int h,bool trimmed){Sprite s;s.f[0x20/4]=8;s.f[0x30/4]=w;s.f[0x34/4]=h;s.f[0x2c/4]=w;s.f[0x18/4]=trimmed;return s;}
int main(){
 for(int i=0;i<256;++i)c3x_native_access::table[i]=std::uint16_t(i*3);
 Harness a;unsigned w=0,h=0;
 // Plain: every index below 254 is opaque, 254 and 255 are transparent.
 unsigned char plain[]={1,254,2,255,3,4};Sprite p=make(3,2,false);p.pixels=plain;
 assert(a.ordinary_sprite(&p,nullptr,w,h)==7&&w==3&&h==2);
 assert(a.uploaded[0]==(65536u|3)&&a.uploaded[1]==0&&a.uploaded[2]==(65536u|6)&&a.uploaded[3]==0);
 // Row-trimmed, as MovementLED's slices: per-row left, count and stream offset.
 unsigned char stream[]={10,11,20,21,22,30};
 unsigned char rows[]={1,2,0,0, 0,3,2,0, 2,1,5,0};
 Sprite t=make(4,3,true);t.rows=rows;t.pixels=stream;t.f[0x2c/4]=0;
 assert(a.ordinary_sprite(&t,nullptr,w,h)==7&&w==4&&h==3);
 std::uint32_t expected[]={0,65536u|30,65536u|33,0, 65536u|60,65536u|63,65536u|66,0, 0,0,65536u|90,0};
 for(int i=0;i<12;++i)assert(a.uploaded[i]==expected[i]);
 // A trimmed row that claims pixels past the width is refused.
 rows[1]=4;assert(a.ordinary_sprite(&t,nullptr,w,h)==0);rows[1]=2;
 // BGRA canvases and large or non-8-bit sources fall back to native.
 Sprite big=make(65,2,false);big.pixels=plain;assert(a.ordinary_sprite(&big,nullptr,w,h)==0);
 Sprite deep=make(3,2,false);deep.f[0x20/4]=16;deep.pixels=plain;assert(a.ordinary_sprite(&deep,nullptr,w,h)==0);
 a.canvas.format=Format::bgra32;assert(a.ordinary_sprite(&p,nullptr,w,h)==0);
 std::printf("PASS ordinary sprite decode: plain, row-trimmed and refusals\n");
}
''')


if __name__ == '__main__':
    unittest.main()
