"""Unexplored land and revealed neighbours (ViewerTopology).

Real relief (default; the user, October 17): revealed ground takes its final
shape at once. Hidden land keeps its real category and rivers and loses only
its effect and eruption bits; it carries unexplored_bit, not hidden_bit, so
there is no datum taper, only a black apron below raised edges facing it.
Previously edges dropped toward hidden hills and mountains and reshaped when
those were explored.
The flat mask (C3X_RENDERER_HIDDEN_RELIEF_CONTROL=1) is kept as a control.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class ViewerTopologyTests(unittest.TestCase):
    def test_hidden_land_keeps_biome_and_rivers_and_revisions_stay_distinct(self):
        run_cpp(r'''
#include "Renderer/native/render_core/viewer_topology.h"
#include "Renderer/native/render_core/world_coast.h"
#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <tuple>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
// Byte 0 is the biome (m49), byte 1 the visible category (m50), byte 2 the
// river code and bit 24 the effect.
std::uint32_t tile(unsigned biome,unsigned category,unsigned river=0,bool effect=false){
    return biome|(category<<8)|(river<<16)|(effect?1u<<24:0u);}
int main(){
    int const width=8,height=8;std::size_t const count=width*height/2;
    std::vector<std::uint32_t> raw(count,tile(2,2));
    auto at=[&](int x,int y){return (std::size_t(y)*width+x)/2;};
    raw[at(2,2)]=tile(1,5,7,true);  // hills on plains, river, effect
    raw[at(3,3)]=tile(2,6);         // mountains
    raw[at(4,4)]=tile(12,12);       // sea
    raw[at(5,5)]=tile(0,4);         // flood plain
    std::vector<std::tuple<int,int,unsigned>> inputs;
    ViewerTopology viewer;viewer.real_relief=false;bool changed=false;  // the flat mask (control)
    auto update=[&](std::uint64_t sequence,std::int64_t revision=3){
        return viewer.update(raw.data(),count,width,height,revision,sequence,[&](auto mark){
            for(auto const& input:inputs)mark(std::get<0>(input),std::get<1>(input),std::get<2>(input));},&changed);};
    // Lab inputs (no world inputs) and explored tiles keep the capture.
    assert(update(1)==raw.data()&&changed&&viewer.revision==3&&!viewer.masked);
    inputs={{2,2,known|explored},{3,3,0u}};
    assert(update(2)==raw.data()&&!changed&&viewer.revision==3);
    // Unexplored land loses relief and cover; water and rivers stay.
    inputs={{2,2,known},{3,3,known},{4,4,known},{5,5,known},{6,6,known}};
    auto values=update(3);
    auto hidden=ViewerTopology::hidden_bit;
    assert(values!=raw.data()&&changed&&viewer.masked&&viewer.hidden_tiles==5);
    assert(values[at(2,2)]==(tile(1,1,7)|hidden)&&values[at(3,3)]==(tile(2,2)|hidden)&&
        values[at(4,4)]==(tile(12,12)|hidden)&&values[at(5,5)]==(tile(0,0)|hidden));
    assert(ViewerTopology::hidden(values[at(4,4)])&&!ViewerTopology::hidden(values[at(1,1)])&&!ViewerTopology::hidden(0xffffffffu));
    auto first=viewer.revision;assert(first!=3 && (first&((std::int64_t(1)<<40)-1))==3);
    // Unchanged inputs reuse the revision; an unrelated input change does too.
    assert(update(3)==values&&!changed&&viewer.revision==first);
    inputs.push_back({1,1,known|explored});assert(update(4)==viewer.values.data()&&!changed&&viewer.revision==first);
    // A reveal changes the mask and the revision.
    inputs[0]={2,2,known|explored};values=update(5);
    assert(changed&&values[at(2,2)]==raw[at(2,2)]&&values[at(3,3)]==(tile(2,2)|hidden)&&viewer.revision!=first);
    auto second=viewer.revision;
    // A capture revision change always yields a new revision.
    update(5,4);assert(changed&&viewer.revision!=second&&(viewer.revision&((std::int64_t(1)<<40)-1))==4);
    // Nothing hidden: the capture revision again.
    inputs={};assert(update(6,4)==raw.data()&&changed&&viewer.revision==4);
    // A lease failure keeps the last mask instead of revealing everything.
    inputs={{3,3,known}};update(7,4);auto kept=viewer.revision;
    auto failed=viewer.update(raw.data(),count,width,height,4,8,[](auto){throw std::length_error("lease");},&changed);
    assert(failed==viewer.values.data()&&!changed&&viewer.revision==kept);
    // The masked topology is what neighbour queries observe.
    WorldCoast coast;coast.update({width,height,false,false},viewer.values.data(),count,viewer.revision);
    auto mountain=coast.world().tile((3+3)/2,(3-3)/2);
    assert(mountain.present&&mountain.real==2&&mountain.base==2&&mountain.hidden);
    assert(!coast.world().tile((2+2)/2,(2-2)/2).hidden);
    // Revealed ground meets a hidden tile at the datum and is unchanged a
    // quarter tile away (exactly 1, so explored-only scenes are bit-identical).
    auto hidden_east=[](int c,int r){return c==1&&r==0;};
    assert(hidden_taper(.5f,.5f,[](int,int){return false;})==1.f);
    assert(hidden_taper(.999f,.5f,hidden_east)<.01f&&hidden_taper(.75f,.5f,hidden_east)==1.f);
    float mid=hidden_taper(.875f,.5f,hidden_east);assert(mid>.4f&&mid<.6f);
    auto hidden_corner=[](int c,int r){return c==1&&r==1;};
    assert(hidden_taper(.99f,.99f,hidden_corner)<.01f&&hidden_taper(.99f,.5f,hidden_corner)==1.f);
    std::printf("PASS viewer topology: hidden_relief=1 hidden_flag=1 datum_taper=1 water_kept=1 rivers_kept=1 distinct_revisions=1 lease_failure=1\n");
}
''')


    def test_real_relief_never_changes_terrain_with_exploration(self):
        # The user (October 17): like Civ III, the map is drawn as it really
        # is and the fog overlay is laid over it. A reveal must not change any
        # terrain input, or it recompiles the ground around it before the
        # revealed tiles can show (slow reveals). Only an unexplored volcano's
        # live state (smoldering, erupting) is withheld: its glow could bloom
        # past the overlay, and Civ III shows nothing there.
        run_cpp(r"""
#include "Renderer/native/render_core/viewer_topology.h"
#include <cassert>
#include <cstdio>
#include <tuple>
#include <vector>
using namespace c3x_renderer::render_core;
constexpr unsigned known=C3X_RENDERER_TILE_VISIBILITY_KNOWN,explored=C3X_RENDERER_TILE_EXPLORED;
int main(){
    int const width=8,height=8;std::size_t const count=width*height/2;
    std::vector<std::uint32_t> raw(count,2u|(2u<<8));
    auto at=[&](int x,int y){return (std::size_t(y)*width+x)/2;};
    raw[at(2,2)]=1u|(5u<<8)|(7u<<16);              // hills with a river
    raw[at(5,5)]=2u|(8u<<8)|(1u<<24)|(1u<<26)|(1u<<27); // an erupting snowy volcano
    std::vector<std::tuple<int,int,unsigned>> inputs={{1,1,known|explored},{2,2,known},{5,5,known}};
    ViewerTopology viewer;bool changed=false;assert(viewer.real_relief);
    auto update=[&](std::uint64_t sequence){
        return viewer.update(raw.data(),count,width,height,3,sequence,[&](auto mark){
            for(auto const& input:inputs)mark(std::get<0>(input),std::get<1>(input),std::get<2>(input));},&changed);};
    auto values=update(1);
    // Terrain as it is; the hidden volcano keeps its snow, not its activity.
    for(std::size_t i=0;i<count;++i)assert(values[i]==(i==at(5,5)?raw[i]&~((1u<<24)|(1u<<27)):raw[i]));
    auto revision=viewer.revision;assert(viewer.hidden_tiles==1);
    // A reveal: the same values and revision, so no compiled ground goes stale.
    inputs[1]={2,2,known|explored};values=update(2);
    assert(viewer.revision==revision && !changed && values[at(2,2)]==raw[at(2,2)]);
    // Revealing the volcano shows its activity.
    inputs[2]={5,5,known|explored};values=update(3);
    assert(changed && values==raw.data() && viewer.revision==3);
    std::printf("PASS real relief: terrain_unchanged=1 reveal_unchanged=1 hidden_activity=0\n");
}
""")


if __name__ == '__main__':
    unittest.main()
