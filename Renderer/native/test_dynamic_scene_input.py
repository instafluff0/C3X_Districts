"""Immutable dynamic captures, clocks, budgets and retirement."""
import unittest
from Renderer.native.native_cpp_test import run_cpp

class DynamicSceneInputTests(unittest.TestCase):
    def test_map_records(self):
        run_cpp(r'''
#include <algorithm>
#include "Renderer/native/render_core/dynamic_scene_input.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer::render_core;
int main(){
 c3x_renderer_tile_v1 tile={};tile.resource_id=7;tile.anchor_x=192;tile.tile_flags=C3X_RENDERER_TILE_VISIBILITY_BITS|C3X_RENDERER_TILE_RENDER;
 std::strcpy(tile.resource_name,"Cattle");unsigned topology=0x030202;
 c3x_renderer_frame_v1 f={};f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);f.tiles=&tile;f.tile_count=1;
 f.world_topology=&topology;f.world_topology_count=1;f.presentation_frequency=1000;f.presentation_time_ticks=125;
 f.hour=20;f.season=3;f.tile_width=192;f.tile_height=96;
 DynamicSceneInputs owner;auto a=owner.capture(f,{1,2,3,4});assert(a && owner.bytes()==a->bytes() && a->identity().visibility_epoch==3);
 tile.resource_id=19;tile.tile_flags=0;tile.anchor_x=384;topology=9;f.hour=12;f.presentation_time_ticks=500;
 c3x_renderer_frame_v1 sample={};assert(a->sample(3000,10000,1000,sample));
 assert(sample.presentation_time_ticks==325 && sample.hour==20 && sample.season==3 && sample.tile_width==192);
 assert(sample.tiles[0].resource_id==7 && sample.tiles[0].anchor_x==192 && sample.tiles[0].tile_flags== (C3X_RENDERER_TILE_VISIBILITY_BITS|C3X_RENDERER_TILE_RENDER));
 assert(sample.tiles!=&tile && sample.world_topology!=&topology && *sample.world_topology==0x030202);
 assert(!a->sample(100,0,0,sample));assert(!a->sample(-1,1000,0,sample));
 assert(a->sample(0,1000,100,sample) && sample.presentation_time_ticks==125);
 auto b=owner.capture(f,{1,2,3,4});assert(b && a->valid()); // capture doesn't retire a displayed front
 auto retained=owner.bytes();owner.invalidate();assert(!a->sample(1,1000,0,sample) && !b->valid() && owner.bytes()==retained);
 auto c=owner.capture(f,{1,2,3,4});assert(c && c->valid());a.reset();b.reset();assert(owner.bytes()==c->bytes());c.reset();assert(!owner.bytes());
 auto after_caller=[&]{std::vector<c3x_renderer_tile_v1> temporary(1,tile);auto temporary_frame=f;
 temporary_frame.tiles=temporary.data();return owner.capture(temporary_frame,{5,6,7,8});}();
 assert(after_caller && after_caller->frame().tiles[0].resource_id==19 && after_caller->identity().viewer_epoch==6);
 after_caller.reset();
 DynamicSceneInputs bounded(owner.peak/2);a=bounded.capture(f,{1,2,3,4});assert(a);assert(!bounded.capture(f,{1,2,3,4}));a.reset();assert(bounded.capture(f,{1,2,3,4}));
 f.tile_count=8193;assert(!owner.capture(f,{1,2,3,4}));f.tile_count=1;f.world_topology=nullptr;assert(!owner.capture(f,{1,2,3,4}));
 f.world_topology=&topology;f.presentation_frequency=0;assert(!owner.capture(f,{1,2,3,4}));f.presentation_frequency=1000;f.presentation_time_ticks=INT64_MAX;
 a=owner.capture(f,{1,2,3,4});assert(a && !a->sample(1000,1000,0,sample));
}
''')
if __name__=='__main__':unittest.main()
