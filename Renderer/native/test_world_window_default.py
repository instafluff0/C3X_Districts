"""The resident world window is on by default; C3X_RENDERER_WORLD_WINDOW=0 opts out.

It was opt-in while it stalled adoption on the light save (performance review,
section 51); the stall was the camera delta's wrapped copies (section 52).
Civ III keeps its native capture envelope (margin 0): window tiles past it come
from complete world inputs, and the wider margin tripled Civ III's per-step
capture (section 57).
"""
import re
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class WorldWindowDefaultTests(unittest.TestCase):
    def test_control_defaults_on(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        control = re.search(r'world_window_control=(.*GetEnvironmentVariableA.*);\n', source).group(1)
        run_cpp(r'''
#include <cassert>
#include <cstring>
char const* value=nullptr;
unsigned GetEnvironmentVariableA(char const* name,char* out,unsigned size){
 assert(!std::strcmp(name,"C3X_RENDERER_WORLD_WINDOW"));if(!value)return 0;std::strncpy(out,value,size);return unsigned(std::strlen(value));}
bool control_for(char const* v){value=v;char control[8]={};return ''' + control + r''';}
int main(){
 assert(control_for(nullptr));assert(control_for("1"));assert(!control_for("0"));
}
''')

    def test_civ3_keeps_its_native_capture_envelope(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        self.assertTrue('if(operation==C3X_NATIVE_CAPTURE_MARGIN)return 0;' in source,
                        'the capture margin query must return 0')
        self.assertFalse('margin_x' in (ROOT / 'Renderer/native/render_core/world_window.h').read_text(),
                         'world_window.h must not define a capture margin')


if __name__ == '__main__':
    unittest.main()
