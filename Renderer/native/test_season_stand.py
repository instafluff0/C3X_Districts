"""Season stands: which chunks draw in which season, and when a flip retires the selection."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class SeasonStandTests(unittest.TestCase):
    def test_masks_and_retirement(self):
        run_cpp(r'''
#include <cassert>
#include "Renderer/native/render_core/season_stand.h"
int main(){
 using namespace c3x_renderer::render_core;
 // A chunk without a stand draws in every season.
 for(int season=0;season<4;++season)assert(season_stand_shown(0,season));
 // Forest stands: the summer stand outside Winter, the Winter stand only in it.
 assert(season_stand_shown(season_stand_outside_winter,0) && season_stand_shown(season_stand_outside_winter,1) &&
        !season_stand_shown(season_stand_outside_winter,2) && season_stand_shown(season_stand_outside_winter,3));
 assert(!season_stand_shown(season_stand_winter,0) && season_stand_shown(season_stand_winter,2));
 std::uint16_t forests=std::uint16_t((1u<<season_stand_outside_winter)|(1u<<season_stand_winter));
 // With only forest stands in use, only Winter's edges retire (as before).
 assert(season_stands_differ(forests,1,2) && season_stands_differ(forests,2,3));
 assert(!season_stands_differ(forests,0,1) && !season_stands_differ(forests,3,0));
 // No stands in use: no flip retires.
 assert(!season_stands_differ(0,1,2));
 // A stand drawn only in Autumn retires at Autumn's edges too.
 std::uint16_t autumn=std::uint16_t(forests|(1u<<0b0010u));
 assert(season_stands_differ(autumn,0,1) && season_stands_differ(autumn,1,2));
}
''')


if __name__ == "__main__":
    unittest.main()
