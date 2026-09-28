"""Execute the catalog loader's timing read for older non-ambient run packs."""
import unittest
from Renderer.native.native_cpp_test import ROOT, run_cpp

class UnitClipTimingTests(unittest.TestCase):
    def test_actual_loader_reads_move_header_before_payload_or_map(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        start=source.index('                if(action.name=="move" && !action.ambient)')
        block=source[start:source.index('                unit.actions.push_back',start)]
        run_cpp(r'''
#include <cassert>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <limits>
using DWORD=unsigned;using HANDLE=void*;
HANDLE INVALID_HANDLE_VALUE=(void*)-1;
constexpr int GENERIC_READ=1,FILE_SHARE_READ=1,OPEN_EXISTING=1,FILE_ATTRIBUTE_NORMAL=1,FALSE=0;
unsigned char bytes[32]{};unsigned count_read=32;int opens=0,closes=0;
HANDLE CreateFileA(char const* path,int,int,void*,int,int,void*){assert(std::string(path)=="run.bin");++opens;return bytes;}
int ReadFile(HANDLE,void* out,unsigned size,DWORD* count,void*){assert(size==32);std::memcpy(out,bytes,32);*count=count_read;return 1;}
void CloseHandle(HANDLE){++closes;}
struct Action {std::string name="move";bool ambient=false;unsigned frames=0;float duration=0;struct Part{unsigned mesh;};std::vector<Part> parts{{0}};};
struct Mesh{std::string path="run.bin";};struct {std::vector<Mesh> meshes{Mesh{}};} unit_bodies;
bool load(Action& action){
'''+block+r'''
 return true;
}
int main(){
 for(unsigned version:{1u,2u}){
  std::memcpy(bytes,"C3XANM1\0",8);bytes[6]=char('0'+version);std::memcpy(bytes+8,&version,4);
  unsigned frames=18;float duration=0.5666667f;
  std::memcpy(bytes+24,&frames,4);std::memcpy(bytes+28,&duration,4);
  Action run;assert(load(run));assert(run.frames==18&&std::abs(run.duration-duration)<1e-6);
 }
 Action bad;count_read=31;assert(!load(bad));count_read=32;
 float duration=std::numeric_limits<float>::quiet_NaN();std::memcpy(bytes+28,&duration,4);assert(!load(bad));
 duration=0;std::memcpy(bytes+28,&duration,4);assert(!load(bad));
 bytes[0]='?';assert(!load(bad));assert(opens==closes);
 Action idle;idle.name="idle";idle.ambient=true;idle.frames=91;idle.duration=3;int before=opens;
 assert(load(idle)&&opens==before&&idle.frames==91&&idle.duration==3);
}
''')

if __name__=='__main__':unittest.main()
