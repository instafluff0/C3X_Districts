"""Both map output transfers compile with the time-of-day display grade.

The Lab/bridge LinearOutput (render_core/linear_target.h) and Renderer64's
backbuffer output (sandbox/resident_scene.cpp) are HLSL strings compiled at
startup; the game-only one never runs in a Lab preview. A grade local named
`y` once broke LinearOutput (FXC scopes a for-loop's `int y` to the whole
function): the shader failed, assets never became ready and every preview
drew nothing. This compiles both programs as the runtime does.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def raw_source(text, start):
    """The raw string literal opening at or after start (an index or a marker)."""
    start = text.index('R"(', start if isinstance(start, int) else text.index(start)) + 3
    return text[start:text.index(')"', start)]


def program():
    linear = raw_source((ROOT / "Renderer/native/render_core/linear_target.h").read_text(), "struct LinearOutput")
    display = raw_source((ROOT / "Renderer/native/scene_display.h").read_text(), "scene_display_shader")
    resident = (ROOT / "Renderer/sandbox/resident_scene.cpp").read_text()
    backbuffer = raw_source(resident, resident.rindex('R"(', 0, resident.index("sandbox_backbuffer_output")))
    for source in (linear, backbuffer):
        assert ",saturation));" in source, "the display grade moved; update this test"
    return r'''
#define NOMINMAX
#include <windows.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <string>
#pragma comment(lib,"d3dcompiler.lib")
bool compile(std::string const& text,char const* name,char const* entry,char const* target){
 ID3DBlob *code=nullptr,*errors=nullptr;
 HRESULT hr=D3DCompile(text.data(),text.size(),name,nullptr,nullptr,entry,target,D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
 if(errors){std::printf("%s %s: %s\n",name,entry,static_cast<char const*>(errors->GetBufferPointer()));errors->Release();}
 if(code)code->Release();
 return SUCCEEDED(hr);
}
int main(){
 std::string linear=R"HLSL(''' + linear + r''')HLSL";
 std::string backbuffer=std::string(R"HLSL(''' + display + r''')HLSL")+R"HLSL(''' + backbuffer + r''')HLSL";
 bool ok=compile(linear,"render_core_output","PSOutput","ps_4_0")&&compile(linear,"render_core_output","VSOutput","vs_4_0")&&
     compile(backbuffer,"sandbox_backbuffer_output","PSOutput","ps_4_0")&&compile(backbuffer,"sandbox_backbuffer_output","VSOutput","vs_4_0");
 std::printf(ok?"PASS output grade: LinearOutput and Renderer64 backbuffer output compile\n":"FAIL output shader compile\n");
 return ok?0:1;
}
'''


class OutputGradeTests(unittest.TestCase):
    def test_output_transfers_compile(self):
        run_cpp(program(), timeout=120)


if __name__ == "__main__":
    unittest.main()
