"""Under custom rendering, celebrating cities' fireworks are hidden from the
native Animator update and their saved effect is restored afterwards."""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def helper_source():
    source = (ROOT / "injected_code.c").read_text()
    begin = source.index("void\nhide_custom_renderer_fireworks (bool hide)")
    return source[begin:source.index("\n}\n", begin) + 3]


def patch_source():
    source = (ROOT / "injected_code.c").read_text()
    begin = source.index("patch_Animator_update_display (Animator * this, int edx)")
    return source[begin:source.index("\n}\n", begin)]


class CustomRendererFireworks(unittest.TestCase):
    def test_celebrations_are_hidden_for_the_native_call_and_restored(self):
        run_cpp(r'''
#include <cassert>
#include <cstddef>
enum { AE_Disorder = 1, AE_Fireworks = 2, AE_Plague = 0xB };
struct City_Body { int field_A4; };
struct City { City_Body Body; };
struct { int LastIndex; } cities_list, * p_cities = &cities_list;
City cities[5] = {{{AE_Fireworks}}, {{0}}, {{AE_Disorder}}, {{AE_Plague}}, {{AE_Fireworks}}};
City * get_city_ptr (int id) { return id == 1 ? nullptr : &cities[id]; }
''' + helper_source() + r'''
int main () {
    p_cities->LastIndex = 4;
    hide_custom_renderer_fireworks (true);
    // The native effect walk queues a city's FLC only while field_A4 > 0.
    assert (cities[0].Body.field_A4 <= 0 && cities[4].Body.field_A4 <= 0);
    assert (cities[2].Body.field_A4 == AE_Disorder && cities[3].Body.field_A4 == AE_Plague);
    hide_custom_renderer_fireworks (false);
    assert (cities[0].Body.field_A4 == AE_Fireworks && cities[4].Body.field_A4 == AE_Fireworks);
    assert (cities[2].Body.field_A4 == AE_Disorder && cities[3].Body.field_A4 == AE_Plague);
    // A native stop during the call (field_A4 cleared) is left alone.
    hide_custom_renderer_fireworks (true);
    cities[0].Body.field_A4 = 0;
    hide_custom_renderer_fireworks (false);
    assert (cities[0].Body.field_A4 == 0 && cities[4].Body.field_A4 == AE_Fireworks);
    return 0;
}
''')

    def test_only_the_custom_renderer_path_hides_them(self):
        patch = patch_source()
        vanilla = patch.index("Animator_update_display (this, edx);\n        return;")
        hide = patch.index("hide_custom_renderer_fireworks (true);")
        native = patch.index("Animator_update_display (this, __);")
        restore = patch.index("hide_custom_renderer_fireworks (false);")
        self.assertLess(vanilla, hide)
        self.assertLess(hide, native)
        self.assertLess(native, restore)


if __name__ == "__main__":
    unittest.main()
