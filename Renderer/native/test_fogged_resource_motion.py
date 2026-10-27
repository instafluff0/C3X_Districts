"""Explored resources keep moving in fog, as water does.

Resource animations (whales, fish, animals) on explored tiles that were not
currently visible were posed at time zero and did not count as moving, so
they stood still while the water around them moved (the user, October 10).
Objects beyond the drawn ground still contribute nothing; the hidden ring is
drawn ground (October 27, below).
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp

SOURCE = ROOT / 'Renderer/native/c3x_renderer.cpp'


def drawn_ground_source(source):
    start = source.index('    bool drawn_ground(int tile_x,int tile_y)const{')
    end = source.index('\n    }\n', start) + len('\n    }\n')
    return source[start:end].replace(')const{', '){', 1)


STUBS = r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <cassert>
#include <cstdint>
#include <map>
struct Coverage {std::map<std::pair<int,int>,unsigned> states;
 unsigned state(int x,int y)const{auto f=states.find({x,y});return f==states.end()?0:f->second;}} visibility_coverage;
struct Input {struct {unsigned tile_flags=0;} occurrence;};
struct Topology {std::map<std::uint64_t,Input> inputs;
 std::uint64_t key(int x,int y)const{return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);}
 Input const* world_input(std::uint64_t id)const{auto f=inputs.find(id);return f==inputs.end()?nullptr:&f->second;}} topology_cache;
bool visibility_pass=true;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED,ring=1u<<24;
'''


class FoggedResourceMotionTests(unittest.TestCase):
    def test_explored_fogged_resources_advance_and_count_as_moving(self):
        source = SOURCE.read_text()
        start = source.index('            if(!drawn_ground(anchor.tile_x,anchor.tile_y))continue;')
        end = source.index('animation.mesh.duration,anchor.seed);', start) + len('animation.mesh.duration,anchor.seed);')
        pose = source[start:end]
        self.assertEqual(source.count('++visible_resource_animations;++moving_resources;'), 2)
        run_cpp(STUBS + r'''
namespace c3x_renderer {double ambient_animation_time(std::int64_t ticks,std::int64_t,double,unsigned){return double(ticks);}}
''' + drawn_ground_source(source) + r'''
struct Anchor {int tile_x=0,tile_y=0;unsigned seed=0;} anchor;
struct Mesh {double duration=1;};struct Animation {Mesh mesh;} animation;
struct Frame {std::int64_t presentation_frequency=24000000;} frame;
std::int64_t ticks=4321;
double pose_time(unsigned state,bool& skipped){visibility_coverage.states[{0,0}]=state;skipped=true;
 for(int once=0;once<1;++once){
''' + pose + r'''
  skipped=false;return time;}
 return -1;}
int main(){
 bool skipped=false;
 assert(pose_time(2,skipped)==4321.0&&!skipped);   // visible
 assert(pose_time(1,skipped)==4321.0&&!skipped);   // explored, fogged: moves too
 pose_time(0,skipped);assert(skipped);              // never explored, not drawn: absent
 // The hidden ring is drawn under the overlay: its bodies are posed too.
 topology_cache.inputs[topology_cache.key(0,0)].occurrence.tile_flags=known|explored|ring;
 assert(pose_time(0,skipped)==4321.0&&!skipped);
}
''')

    def test_hidden_ring_draws_its_waves_bodies_and_flurry(self):
        # A reveal lifts the overlay on the view already drawn. That view
        # drew the hidden ring's ground but skipped its waves, moving bodies
        # and flurries (coverage 0), so they appeared a moment after their
        # tile (the user, October 27). drawn_ground counts the ring, and a
        # tile Civ III has since explored, whose input carries EXPLORED.
        source = SOURCE.read_text()
        self.assertEqual(source.count('if(!drawn_ground(c+r,c-r))continue;'), 3)  # two retained, one direct wave path
        self.assertEqual(source.count('if(!drawn_ground(t.tile_x,t.tile_y))continue;'), 1)  # flurry
        self.assertEqual(source.count('visibility_pass && !drawn_ground(tile.tile_x,tile.tile_y)'), 1)  # water motion
        self.assertNotIn('visibility_coverage.state(c+r,c-r)', source)
        run_cpp(STUBS + drawn_ground_source(source) + r'''
int main(){
 visibility_coverage.states[{0,0}]=2;visibility_coverage.states[{2,0}]=1;
 assert(drawn_ground(0,0) && drawn_ground(2,0));        // seen, fogged
 assert(!drawn_ground(4,0));                            // no input: black
 topology_cache.inputs[topology_cache.key(4,0)].occurrence.tile_flags=known;
 assert(!drawn_ground(4,0));                            // unexplored, beyond the ring
 topology_cache.inputs[topology_cache.key(4,0)].occurrence.tile_flags=known|explored|ring;
 assert(drawn_ground(4,0));                             // the hidden ring
 topology_cache.inputs[topology_cache.key(6,0)].occurrence.tile_flags=known|explored;
 assert(drawn_ground(6,0));                             // revealed since this view's coverage
 visibility_pass=false;assert(drawn_ground(8,0));       // no fog pass: everything drawn
}
''')


if __name__ == '__main__':
    unittest.main()
