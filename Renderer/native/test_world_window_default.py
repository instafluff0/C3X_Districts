"""The resident world window is on by default; C3X_RENDERER_WORLD_WINDOW=0 opts out.

Stage 2 finished with the window behind an opt-in flag, so ordinary games never
deferred scroll steps (every step was refused, reason 1). Busy 1x scroll went
from about 6.7 to 10-14 fps with it (performance review, section 51). The
renderer's control and the capture margin it returns to Civ III must agree.
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class WorldWindowDefaultTests(unittest.TestCase):
    def test_control_and_capture_margin_default_on(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        control = re.search(r'world_window_control=(.*GetEnvironmentVariableA.*);\n', source).group(1)
        margin = source[source.index('static int const margin=[]{'):]
        margin = margin[:margin.index('}();') + 4]
        run_cpp(r'''
#include <cassert>
#include <cstring>
namespace c3x_renderer {namespace render_core {struct WorldWindow {static constexpr unsigned margin_x=7,margin_y=9;};}}
char const* value=nullptr;
unsigned GetEnvironmentVariableA(char const* name,char* out,unsigned size){
 assert(!std::strcmp(name,"C3X_RENDERER_WORLD_WINDOW"));if(!value)return 0;std::strncpy(out,value,size);return unsigned(std::strlen(value));}
bool control_for(char const* v){value=v;char control[8]={};return ''' + control + r''';}
int margin_for(char const* v){value=v;using Window=c3x_renderer::render_core::WorldWindow;
 ''' + margin.replace('static int const margin=', 'int const margin=') + r'''
 return margin;}
int main(){
 int on=int(7u|(9u<<16));
 assert(control_for(nullptr)&&margin_for(nullptr)==on);
 assert(control_for("1")&&margin_for("1")==on);
 assert(!control_for("0")&&margin_for("0")==0);
}
''')


if __name__ == '__main__':
    unittest.main()
