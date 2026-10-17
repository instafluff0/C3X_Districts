"""The retained composition budget scales with the window.

Opening a full-screen native window (city view, diplomacy) keeps about 25
window-sized textures resident: native uploads, the layers pinned by the
composition being built, fused sources, retired pixels, the HUD pair and
replay scratch. At the user's 2240x1192 window each is 10.7 MB; a fixed
256 MiB refused the window's capture (`retained-admission-rejected`), so its
updates never reached the screen while clicks still worked: no hover
highlight, negotiation choices that seemed to do nothing (October 17,
review 80). The budget is now 40 window-sized textures, at least 256 MiB,
and never shrinks while the front is reset.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def budget_block(source):
    start = source.index('    constexpr static std::uint64_t base_resident_budget=')
    end = source.index('    }\n', source.index('    std::uint64_t resident_budget()const{', start)) + 6
    return source[start:end]


class RetainedBudgetScalingTests(unittest.TestCase):
    def setUp(self):
        self.source = (ROOT / 'Renderer/native/retained_composition.h').read_text()

    def test_budget_follows_the_window(self):
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
struct Owner {
 struct Picture {unsigned width=0,height=0;} front;
''' + budget_block(self.source) + r'''
};
int main(){
 constexpr std::uint64_t mib=1024u*1024u;
 Owner o;assert(o.resident_budget()==256*mib);
 o.front={1024,768};assert(o.resident_budget()==256*mib);   // small windows keep the floor
 o.front={2240,1192};auto window=std::uint64_t(2240)*1192*4;
 assert(o.resident_budget()==40*window);
 // The failure in the user's capture: 267 MB resident, a 21.4 MB HUD pair.
 std::uint64_t resident=267253776,pair=21360640;
 assert(resident+pair>256*mib && resident+pair<=o.resident_budget());
 // A reset front (released, or between screens) does not shrink it.
 o.front={0,0};assert(o.resident_budget()==40*window);
 o.front={3840,2160};assert(o.resident_budget()==40*std::uint64_t(3840)*2160*4);
}
''')

    def test_every_admission_uses_the_scaled_budget(self):
        body = self.source.split(budget_block(self.source), 1)[1]
        # The old code compared against a fixed constant.
        self.assertNotIn('constexpr static std::uint64_t resident_budget=', self.source)
        self.assertEqual(body.count('resident_budget()'), 19)  # 18 admission sites and the public budget()
        self.assertNotIn('resident_budget-', body.replace('resident_budget()-', ''))


if __name__ == '__main__':
    unittest.main()
