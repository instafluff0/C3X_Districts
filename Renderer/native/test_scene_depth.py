"""Shared depth basis survives raster target placement and camera reuse."""
import unittest
from Renderer.native.native_cpp_test import run_cpp

class SceneDepthTests(unittest.TestCase):
    def test_common_depth_survives_camera_and_occurrence_relocation(self):
        run_cpp(r'''
#include "Renderer/native/render_core/scene_depth.h"
#include <cassert>
#include <initializer_list>
using namespace c3x_renderer::render_core;
int main(){
 for(int tile_height:{64,80,96}){
  int row=39,screen_origin=-1000,source_anchor=screen_origin+row*tile_height/2;
  auto initial=scene_depth_basis(source_anchor,row,tile_height,1192,0);
  // Reused geometry translates with the camera; a freshly assembled occurrence
  // encodes the same delta in its anchor instead. Both must produce equal depth.
  for(int pan:{-128,-17,0,17,128}){
   auto reused=scene_depth_basis(source_anchor+pan,row,tile_height,1192,pan);
   auto rebuilt=scene_depth_basis(source_anchor+pan,row,tile_height,1192,0);
   assert(initial.world_origin==reused.world_origin);
   float source_depth=float(source_anchor)+12.5f;
   assert(source_depth+initial.translation==source_depth+reused.translation);
   assert(source_depth+initial.translation==source_depth+pan+rebuilt.translation);
   // A different reference tile must recover the same canonical screen origin.
   auto alternate=scene_depth_basis(source_anchor+pan+tile_height,row+2,tile_height,1192,pan);
   assert(alternate.world_origin==reused.world_origin && alternate.translation==reused.translation);
  }
 }
 for(int center:{-100000,-4097,-4096,-1,0,4095,4096,100000}){
  auto basis=scene_depth_basis(-center,0,64,0,0);
  assert(basis.world_origin%4096==0);
  assert(basis.translation>=-2048 && basis.translation<2048);
 }
 auto before=scene_depth_basis(0,0,64,0,0),after=scene_depth_basis(-4096,0,64,0,0);
 assert(before.world_origin!=after.world_origin); // Stored depth cannot be relabelled at an origin change.
}
''')

    def test_production_depth_is_consumable_across_raster_regions(self):
        run_cpp('',sources=('Renderer/native/test_scene_depth_windows.cpp',),timeout=60)
