"""River-bank rocks keep clear of bridged edges.

A bank rock stands on about one river edge in three, partway along the edge,
pushed onto the bank. A bridge stands on a river edge where routes meet across
it. The full-size bridge decks covered those rocks; smaller pattern bridges
left small boulders beside the bridge ends. The river-rock placement now skips
an edge when both incident tiles carry a road or railroad.
This test compiles that rule from c3x_renderer.cpp.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def rule():
    source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
    block = source[source.index('if (river_rock_group != nullptr'):]
    start = block.index('if ((tile.road_mask || tile.railroad_mask)')
    return block[start:block.index('continue;', start) + len('continue;')]


class RiverRockBridgeTests(unittest.TestCase):
    def test_bridged_edges_get_no_bank_rock(self):
        run_cpp(r'''
#include <cassert>
#include <cstdio>
struct Tile {unsigned road_mask=0,railroad_mask=0;};
struct Observation {Tile occurrence;};
bool rock(Tile tile,Observation const* neighbor){
 for(int once=0;once<1;++once){
''' + rule() + r'''
  return true;
 }
 return false;
}
int main(){
 Tile road{1,0},rail{1,1},plain{};
 Observation road_across{road},rail_across{rail},plain_across{plain};
 assert(!rock(road,&road_across) && !rock(rail,&road_across) && !rock(road,&rail_across));
 assert(rock(road,&plain_across) && rock(plain,&road_across) && rock(plain,&plain_across));
 assert(rock(road,nullptr));
 std::puts("PASS river rocks skip bridged edges");
}
''')


if __name__ == '__main__':
    unittest.main()
