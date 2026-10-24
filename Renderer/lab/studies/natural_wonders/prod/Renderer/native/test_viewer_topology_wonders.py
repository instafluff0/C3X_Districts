"""Natural-wonder relief slots: Renderer64 folds an explored wonder's slot into its viewer topology."""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class ViewerTopologyWonderTests(unittest.TestCase):
    def test_explored_wonder_sets_slot_and_revision(self):
        run_cpp(r'''
#include "Renderer/native/render_core/viewer_topology.h"
#include <cassert>
#include <tuple>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int main(){
 // A 4x2 world: (0,0),(2,0),(1,1),(3,1). Grassland, no river.
 std::vector<std::uint32_t> raw(4,2u|(2u<<8));
 ViewerTopology topology;bool changed=false;
 std::vector<std::tuple<int,int,unsigned,unsigned>> inputs={{0,0,known|explored,0}};
 auto run=[&](std::int64_t revision,std::uint64_t sequence){
  return topology.update(raw.data(),raw.size(),4,2,revision,sequence,[&](auto mark){
   for(auto const& input:inputs)mark(std::get<0>(input),std::get<1>(input),std::get<2>(input),std::get<3>(input));},&changed);};
 auto plain=run(5,1);assert(plain==raw.data() && topology.revision==5 && !topology.masked);
 // An explored record naming a relief kit (slot 7): bits 28-31, a new revision.
 inputs={{0,0,known|explored,0},{2,0,known|explored,7}};
 auto folded=run(5,2);assert(changed && folded!=raw.data() && folded[1]==(raw[1]|(7u<<28)) && folded[0]==raw[0]);
 assert(topology.wonder_tiles==1 && topology.revision!=5);auto revision=topology.revision;
 // The same inputs again: unchanged.
 run(5,3);assert(!changed && topology.revision==revision);
 // An unexplored tile never takes a slot (its record names nothing in the game; a slot is ignored).
 inputs={{0,0,known,9},{2,0,known|explored,7}};
 auto hidden=run(5,4);assert(hidden[0]==raw[0] && hidden[1]>>28==7);   // real relief keeps the hidden tile's word
 // The wonder's slot leaves with its record.
 inputs={{0,0,known|explored,0}};
 auto restored=run(5,7);assert(restored==raw.data() && topology.revision==5 && !topology.masked);
}
''')


if __name__ == "__main__":
    unittest.main()
