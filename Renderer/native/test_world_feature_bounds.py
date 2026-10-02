"""Host regression for canonical river-rock payloads and their culling bounds."""
import os
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp


SOURCE = Path(__file__).resolve().parents[2] / "Renderer/native/c3x_renderer.cpp"


def rock_writers():
    source = SOURCE.read_text()
    start = source.index("if (river_rock_group != nullptr")
    body = source[start:source.index("int animated_resource =", start)]
    relative = [line.strip() for line in body.splitlines()
                if line.strip().startswith(("int relative_x=", "int relative_y="))]
    position = [line.strip() for line in body.splitlines()
                if line.strip().startswith(("vertex.x=", "vertex.y=", "vertex.z="))]
    if len(relative) != 2 or len(position) != 3:
        raise AssertionError("River-rock relative/position writer extraction is stale")
    return "\n".join(relative), "\n".join(position)


PREAMBLE = r'''
#include "Renderer/native/render_core/prepared_mesh.h"
#include <cassert>
#include <array>
struct Tile { int tile_x=0,tile_y=0,anchor_x=0,anchor_y=0; };
struct Frame { int tile_width=128,tile_height=64; };
using Vertex=c3x_renderer::render_core::Vertex;
using Mesh=c3x_renderer::render_core::PreparedMesh;
void same_mesh(Mesh const& a,Mesh const& b){
 assert(a.vertices==b.vertices && a.indices==b.indices && a.bounds==b.bounds);
 assert(a.world_low==b.world_low && a.world_high==b.world_high);
 assert(a.projected_bounds.extent==b.projected_bounds.extent);
 assert(a.vertex_stride==b.vertex_stride && a.index_stride==b.index_stride);
 assert(a.index_count==b.index_count && a.shared_grid==b.shared_grid);
}
'''


FUNCTION = r'''
Mesh NAME(Tile const& tile,Tile const& receiving,Frame const& frame,bool world_objects){
 auto owner=&receiving;
 RELATIVE
 float local_u=.43f,local_v=.76f;
 std::array<float,3> ground_sample={1,0,0};
 std::vector<Vertex> vertices;
 // A small non-flat authored body drives the real compact feature packer.
 for(auto p:std::array<std::array<float,3>,3>{{{-.1f,-.1f,0},{.1f,-.1f,.03f},{0,.1f,.08f}}}){
  float local_x=p[0],local_y=p[1],local_z=p[2];
  float feature_height_tiles=local_z*150.f/.82f;
  Vertex vertex={};
  vertex.x=float(relative_x)+float(frame.tile_width)*.5f+
   (local_u-local_v+local_x-local_y)*float(frame.tile_width)*.5f;
  vertex.y=float(relative_y)+(local_u+local_v+local_x+local_y)*float(frame.tile_height)*.5f;
  if(world_objects){POSITION}
  vertex.world_x=float(owner->tile_x+owner->tile_y)*.5f+local_u+local_x;
  vertex.world_y=float(owner->tile_x-owner->tile_y)*.5f+1.f-local_v-local_y;
  vertex.world_z=(ground_sample[0]+2.5f+feature_height_tiles)/112.f;
  vertex.world_valid=1.f;vertex.normal_z=1.f;vertex.base_terrain=8.f;
  vertices.push_back(vertex);
 }
 c3x_renderer::render_core::MeshFormat format;format.feature=true;format.projection_kind=2;
 Mesh mesh;
 assert(c3x_renderer::render_core::prepare_mesh(vertices,nullptr,format,mesh,[]{return false;}));
 assert(mesh.vertex_stride==48 && mesh.index_count==3);
 return mesh;
}
'''


def program(main, *, reproduce_old=False):
    relative, position = rock_writers()
    current = FUNCTION.replace("NAME", "current_writer").replace("RELATIVE", relative).replace("POSITION", position)
    old = ""
    if reproduce_old:
        # Reproduce only the faulty expressions in the current source body;
        # no cached source snapshot or alternative packer is involved.
        substitutions = {
            "float(relative_x)": "float(owner->anchor_x-tile.anchor_x)",
            "float(relative_y)": "float(owner->anchor_y-tile.anchor_y)",
        }
        for original, replacement in substitutions.items():
            if position.count(original) != 1:
                raise AssertionError("Canonical river-rock position writer changed")
            position = position.replace(original, replacement)
        old = FUNCTION.replace("NAME", "old_writer").replace("RELATIVE", relative).replace("POSITION", position)
    return PREAMBLE + current + old + main


@unittest.skipIf(os.name == "nt", "Host-only contracts never dispatch Windows or VM tools")
class WorldFeatureBoundsTests(unittest.TestCase):
    def test_canonical_payload_and_bounds_ignore_occurrence_anchors(self):
        run_cpp(program(r'''
int main(){
 for(int width:{64,128,256}){
  Frame frame{width,width/2};Tile tile{84,20,200,400};
  Tile receiving{85,19,tile.anchor_x+width/2,tile.anchor_y-width/4};
  auto expected=current_writer(tile,receiving,frame,true);
  for(auto delta:std::array<std::array<int,4>,4>{{
    {0,0,-544,-662},{300,-700,300,-700},{-12800,3200,0,0},{17,41,-93,77}}}){
   auto occurrence=tile,other=receiving;
   occurrence.anchor_x+=delta[0];occurrence.anchor_y+=delta[1];
   other.anchor_x+=delta[2];other.anchor_y+=delta[3];
   same_mesh(expected,current_writer(occurrence,other,frame,true));
  }
 }
}
'''))

    def test_receiving_and_continuous_wrap_coordinates_remain_distinct(self):
        run_cpp(program(r'''
int main(){
 Frame frame;Tile tile{84,20,200,400},receiving{85,19,264,368};
 auto base=current_writer(tile,receiving,frame,true);
 auto moved=receiving;++moved.tile_x;++moved.tile_y;
 auto other=current_writer(tile,moved,frame,true);
 assert(other.bounds!=base.bounds && other.vertices!=base.vertices);
 assert(other.world_low!=base.world_low && other.world_high!=base.world_high);
 // A different continuous receiving occurrence cannot collapse to a camera
 // anchor or a modulo-world key, even when its native anchor is unchanged.
 auto wrapped=receiving;wrapped.tile_x+=100;
 auto wrap_mesh=current_writer(tile,wrapped,frame,true);
 assert(wrap_mesh.bounds!=base.bounds && wrap_mesh.vertices!=base.vertices);
 assert(wrap_mesh.world_low!=base.world_low);
 // Shift both world coordinates together: local bounds stay reusable, while
 // absolute world channels retain the distinct continuous-world placement.
 auto source_wrap=tile;source_wrap.tile_x+=100;
 auto common_wrap=current_writer(source_wrap,wrapped,frame,true);
 assert(common_wrap.bounds==base.bounds && common_wrap.indices==base.indices);
 assert(common_wrap.vertices!=base.vertices && common_wrap.world_low!=base.world_low);
}
'''))

    def test_old_anchor_writer_reproduces_failure_and_legacy_stays_native(self):
        run_cpp(program(r'''
int main(){
 Frame frame;Tile tile{84,20,200,400},warm{85,19,264,368},cold=warm;
 cold.anchor_x-=544;cold.anchor_y-=662;
 auto old_warm=old_writer(tile,warm,frame,true),old_cold=old_writer(tile,cold,frame,true);
 assert(old_warm.bounds!=old_cold.bounds && old_warm.vertices!=old_cold.vertices);
 assert(old_warm.world_low==old_cold.world_low && old_warm.world_high==old_cold.world_high);
 // Preserve exact bytes at the normal native relation, not only a tolerance.
 same_mesh(current_writer(tile,warm,frame,true),old_warm);
 same_mesh(current_writer(tile,cold,frame,true),old_warm);
 auto legacy_warm=current_writer(tile,warm,frame,false);
 auto legacy_cold=current_writer(tile,cold,frame,false);
 assert(legacy_warm.bounds!=legacy_cold.bounds && legacy_warm.vertices!=legacy_cold.vertices);
 assert(legacy_warm.world_low==legacy_cold.world_low && legacy_warm.world_high==legacy_cold.world_high);
 same_mesh(legacy_warm,old_writer(tile,warm,frame,false));
 same_mesh(legacy_cold,old_writer(tile,cold,frame,false));
}
''', reproduce_old=True))


if __name__ == "__main__":
    unittest.main()
