"""Camera requests carry only changed tiles; the helper rebuilds the frame.

Civ III re-captures its whole envelope at every step and the bridge sent all of
it, so a wider envelope (the world window) cost Civ III's thread about 30 ms
per step waiting on the transport (performance review, section 29). The
receiver must rebuild each frame exactly (order, anchors, every field), and
ask for a base whenever it lacks a referenced tile.
"""
import unittest

from Renderer.native.native_cpp_test import run_cpp


class CameraDeltaTests(unittest.TestCase):
    def test_round_trip_is_exact_and_small_after_the_base(self):
        run_cpp(r'''
#include "Renderer/native/camera_delta.h"
#include <cassert>
#include <cstdio>
using namespace c3x_remote_scene;
std::vector<c3x_renderer_tile_v1> capture(int camera_x,int seed,int wrap_copy){
 std::vector<c3x_renderer_tile_v1> tiles;
 for(int y=0;y<40;++y)for(int x=0;x<60;++x){if((x+y)&1)continue;
  c3x_renderer_tile_v1 t{};t.tile_x=(x+camera_x/64)%100;t.tile_y=y;t.anchor_x=x*64-camera_x%64;t.anchor_y=y*32;
  t.terrain_type=(t.tile_x*7+y)%11;t.tile_flags=(x>4&&x<55)?C3X_RENDERER_TILE_RENDER:C3X_RENDERER_TILE_PREFETCH;
  t.resource_id=(t.tile_x*13+y*seed)%97==0?5:-1;std::snprintf(t.city_owner,sizeof(t.city_owner),"c%d",t.tile_x%3);
  tiles.push_back(t);}
 if(wrap_copy){auto copy=tiles.front();copy.anchor_x+=6400;tiles.push_back(copy);}
 return tiles;}
c3x_renderer_frame_v1 frame_for(std::vector<c3x_renderer_tile_v1>& tiles,std::vector<c3x_renderer_u32>& topology){
 c3x_renderer_frame_v1 f{};f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);f.tile_width=128;f.tile_height=64;
 f.target_width=1000;f.target_height=600;f.world_width_tiles=100;f.world_height_tiles=40;f.world_wrap_x=1;
 f.tiles=tiles.data();f.tile_count=unsigned(tiles.size());f.world_topology=topology.data();f.world_topology_count=unsigned(topology.size());return f;}
std::vector<unsigned char> encoded(c3x_renderer_frame_v1 const& f){c3x_inputs::Writer w;c3x_inputs::frame(w,f);return w.bytes;}
int main(){
 CameraDeltaSender sender;CameraDeltaReceiver receiver;
 std::vector<c3x_renderer_u32> topology(2000);for(unsigned i=0;i<topology.size();++i)topology[i]=i*31u;
 std::size_t base_bytes=0,step_bytes=0;
 auto step=[&](int camera_x,int seed,int wrap_copy,bool expect_base_missing){
  auto tiles=capture(camera_x,seed,wrap_copy);auto f=frame_for(tiles,topology);
  for(int attempt=0;attempt<2;++attempt){
   c3x_inputs::Writer w;sender.encode(w,f);c3x_inputs::Reader r{w.bytes};c3x_inputs::Frame out;
   if(!receiver.decode(r,out)){assert(expect_base_missing&&attempt==0);sender.reset();continue;}
   r.done();sender.commit();
   assert(encoded(out.value)==encoded(f));                       // exact rebuild
   if(sender.full_tiles==f.tile_count)base_bytes=w.bytes.size();else step_bytes=w.bytes.size();
   return;}
  assert(false);};
 step(0,3,0,false);                                     // base
 step(128,3,0,false);step(256,3,0,false);               // camera steps: anchors move
 assert(sender.full_tiles<=6*40/2); // a 128 px step: two entering columns and two RENDER-edge columns on each side
 assert(step_bytes*4<base_bytes); // unchanged tiles cost 20 bytes (coordinates, anchors, flag)
 step(256,5,0,false);                                    // a content change elsewhere
 assert(sender.full_tiles>0&&sender.full_tiles<60);
 step(384,5,1,false);                                    // a wrapped duplicate occurrence
 topology[7]^=1;step(384,5,1,false);                     // topology change travels
 receiver=CameraDeltaReceiver{};step(512,5,0,true);      // a restarted receiver asks for a base
 std::printf("PASS camera delta: exact rebuild base=%zu step=%zu bytes\n",base_bytes,step_bytes);
}
''')


    def test_wrapped_copies_with_different_placement_rebuild_exactly(self):
        # A small wrapping map inside the world window's wide capture margin
        # holds one tile twice: on screen (RENDER) and as a wrapped copy past
        # the zoom envelope (TOPOLOGY_HALO | PREFETCH). The receiver keeps one
        # slot per coordinate and overwrites it while decoding, so judging
        # "unchanged" against the previous frame rebuilt one copy with the
        # other's flags. Every completion then differed from the request,
        # was superseded, and Civ III adopted no step after the first
        # (performance review, section 51: the light-save stall).
        run_cpp(r'''
#include "Renderer/native/camera_delta.h"
#include <cassert>
#include <cstdio>
using namespace c3x_remote_scene;
std::vector<unsigned char> encoded(c3x_renderer_frame_v1 const& f){c3x_inputs::Writer w;c3x_inputs::frame(w,f);return w.bytes;}
int main(){
 for(int halo_first=0;halo_first<2;++halo_first){
  CameraDeltaSender sender;CameraDeltaReceiver receiver;
  for(int step=0;step<4;++step){
   c3x_renderer_tile_v1 shown{},wrapped{},other{};
   shown.tile_x=10;shown.tile_y=4;shown.anchor_x=640+step*64;shown.anchor_y=128;shown.terrain_type=3;
   shown.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_RENDER;
   wrapped=shown;wrapped.anchor_x-=64*64;
   wrapped.tile_flags=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO;
   other=shown;other.tile_x=12;other.anchor_x+=128;
   std::vector<c3x_renderer_tile_v1> tiles;
   if(halo_first)tiles={wrapped,other,shown};else tiles={shown,other,wrapped};
   c3x_renderer_frame_v1 f{};f.api_version=C3X_RENDERER_API_VERSION;f.struct_size=sizeof(f);f.tile_width=128;f.tile_height=64;
   f.target_width=1000;f.target_height=600;f.world_width_tiles=64;f.world_height_tiles=72;f.world_wrap_x=1;
   f.tiles=tiles.data();f.tile_count=unsigned(tiles.size());
   c3x_inputs::Writer w;sender.encode(w,f);c3x_inputs::Reader r{w.bytes};c3x_inputs::Frame out;
   assert(receiver.decode(r,out));r.done();sender.commit();
   for(unsigned i=0;i<f.tile_count;++i)if(out.value.tiles[i].tile_flags!=tiles[i].tile_flags)
    std::printf("order=%d step=%d occurrence=%u flags sent=%x rebuilt=%x\n",halo_first,step,i,tiles[i].tile_flags,out.value.tiles[i].tile_flags);
   assert(encoded(out.value)==encoded(f));
   if(step)assert(sender.reused_tiles==1); // the unique tile still travels as a reference
  }
 }
 std::printf("PASS camera delta: wrapped copies rebuild exactly\n");
}
''')


if __name__ == '__main__':
    unittest.main()
