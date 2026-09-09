"""Production animation-region placement and conservative static identities."""
import unittest

from Renderer.native.native_cpp_test import run_cpp
from Renderer.lab.platform import ROOT


class AnimationRetentionTests(unittest.TestCase):
    def test_bounded_post_preserves_full_dispatch_workgroup_coordinates(self):
        source=(ROOT / "Renderer/native/city_fidelity/glow.h").read_text()
        body="D3D11_RECT dispatch_rectangle("+source.split("D3D11_RECT dispatch_rectangle(",1)[1].split("    std::size_t reconstruct(",1)[0]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
#include <initializer_list>
using LONG=long;
struct D3D11_RECT {LONG left,top,right,bottom;};
''' + body + r'''
int main() {
 for(int width:{136,264,520,2248})for(int height:{136,264,272})
 for(int x=-9;x<width+9;x+=13)for(int y=-9;y<height+9;y+=11){
  D3D11_RECT dirty={x,y,x+17,y+19};
  auto r=dispatch_rectangle(width,height,&dirty);
  int left=std::clamp(x,0,width),top=std::clamp(y,0,height);
  int right=std::clamp(x+17,left,width),bottom=std::clamp(y+19,top,height);
  if(left==right || top==bottom){assert(r.right==0 && r.bottom==0);continue;}
  assert(r.left<=left && r.top<=top && r.right>=right && r.bottom>=bottom);
  assert(r.left%8==0 && r.top%8==0 && r.right%8==0 && r.bottom%8==0);
  for(int py=top;py<bottom;++py)for(int px=left;px<right;++px){
   // Offset dispatch IDs address the same pixels, workgroups and local lanes.
   int local_x=px-r.left,local_y=py-r.top;
   assert(local_x/8+r.left/8==px/8 && local_y/8+r.top/8==py/8);
   assert(local_x%8==px%8 && local_y%8==py%8);
  }
 }
 auto full=dispatch_rectangle(2248,264,nullptr);
 assert(full.left==0 && full.top==0 && full.right==2248 && full.bottom==264);
 D3D11_RECT empty={30,40,30,50};assert(dispatch_rectangle(2248,264,&empty).right==0);
}
''')

    def test_rectangular_strip_guard_clips_each_axis_independently(self):
        source=(ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        body="D3D11_RECT guarded_block_rectangle("+source.split("D3D11_RECT guarded_block_rectangle(",1)[1].split("    void collect_shadow_casters",1)[0]
        run_cpp(r'''
#include <algorithm>
#include <cassert>
using LONG=long;
struct D3D11_RECT {LONG left,top,right,bottom;};
''' + body + r'''
int main() {
 auto full=guarded_block_rectangle({0,0,2240,256},4,4,4,2248,264);
 assert(full.left==0 && full.top==0 && full.right==2248 && full.bottom==264);
 auto edge=guarded_block_rectangle({2220,250,2240,256},4,4,4,2248,264);
 assert(edge.left==2220 && edge.top==250 && edge.right==2248 && edge.bottom==264);
 auto negative=guarded_block_rectangle({0,0,30,20},-10,-5,4,2248,264);
 assert(negative.left==0 && negative.top==0 && negative.right==24 && negative.bottom==19);
 auto square=guarded_block_rectangle({0,0,128,128},4,4,4,136);
 assert(square.right==136 && square.bottom==136);
}
''')

    def test_wave_cell_owner_pressure_and_pinned_lifetime(self):
        source=(ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        owner="struct RetainedWaveCell {"+source.split("struct RetainedWaveCell {",1)[1].split("struct ResourceAnimation",1)[0]
        admission="    bool make_wave_cell_room("+source.split("    bool make_wave_cell_room(",1)[1].split("    bool prepare_retained_wave_chunks(",1)[0]
        run_cpp(r'''
#include <cassert>
#include <map>
#include <cstdint>
#include <cstddef>
#include <utility>
struct Buffer {int refs=1;void AddRef(){++refs;}void Release(){assert(refs>0);--refs;}};
struct CachedVertexChunk {Buffer* buffer=nullptr;Buffer* indices=nullptr;};
''' + owner + r'''
struct Pool {
 std::map<std::pair<int,int>,RetainedWaveCell> retained_wave_cells;
 std::uint64_t retained_wave_epoch=3;
 std::size_t wave_geometry_bytes=0;
''' + admission + r'''
};
int main() {
 constexpr std::size_t mib=1024u*1024u;
 Buffer first,first_indices,second,second_indices;
 Pool pool;
 {RetainedWaveCell cell;cell.chunk={&first,&first_indices};cell.bytes=20*mib;cell.used=3;
  pool.retained_wave_cells.emplace(std::make_pair(0,0),std::move(cell));}
 {RetainedWaveCell cell;cell.chunk={&second,&second_indices};cell.bytes=12*mib;cell.used=2;
  pool.retained_wave_cells.emplace(std::make_pair(1,0),std::move(cell));}
 pool.wave_geometry_bytes=32*mib;
 assert(!pool.make_wave_cell_room(33*mib));assert(pool.retained_wave_cells.size()==2);
 assert(!pool.make_wave_cell_room(16*mib)); // The active 20 MiB remains pinned.
 assert(pool.wave_geometry_bytes==20*mib && first.refs==1 && second.refs==0 && second_indices.refs==0);
 assert(pool.make_wave_cell_room(12*mib));
 // A frame holds its own references until the view is cleared.
 first.AddRef();first_indices.AddRef();pool.retained_wave_cells.clear();pool.wave_geometry_bytes=0;
 assert(first.refs==1 && first_indices.refs==1);first.Release();first_indices.Release();
 Buffer failed;{RetainedWaveCell incomplete;incomplete.chunk.buffer=&failed;}assert(failed.refs==0);
 for(int i=0;i<16384;++i){RetainedWaveCell empty;empty.used=3;
   pool.retained_wave_cells.emplace(std::make_pair(i,0),std::move(empty));}
 assert(!pool.make_wave_cell_room(0));
 pool.retained_wave_cells.begin()->second.used=2;
 assert(pool.make_wave_cell_room(0));assert(pool.retained_wave_cells.size()==16383);
}
''')

    def test_wave_scope_and_capture_eligibility(self):
        run_cpp(r'''
#include <cassert>
#include "Renderer/native/c3x_renderer_api.h"
#include "Renderer/native/render_core/wave_retention.h"
int main() {
 using namespace c3x_renderer::render_core;
 c3x_renderer_tile_v1 tiles[3]={};
 tiles[0].tile_flags=C3X_RENDERER_TILE_RENDER;
 tiles[1].tile_x=100;tiles[2].tile_y=100;
 c3x_renderer_frame_v1 frame={};frame.tiles=tiles;frame.tile_count=3;
 frame.tile_width=128;frame.tile_height=64;frame.target_width=2240;frame.target_height=1192;
 frame.world_width_tiles=frame.world_height_tiles=100;frame.world_wrap_x=frame.world_wrap_y=1;
 auto scope=wave_geometry_scope(frame,7,3);
 tiles[0].anchor_x=123;frame.hour=18;frame.season=3;frame.presentation_time_ticks=900;
 assert(wave_geometry_scope(frame,7,3)==scope);
 auto cells=captured_wave_cells(frame,C3X_RENDERER_TILE_RENDER);assert(cells.size()==9);
 tiles[1].tile_flags=tiles[2].tile_flags=C3X_RENDERER_TILE_RENDER;
 cells=captured_wave_cells(frame,C3X_RENDERER_TILE_RENDER);assert(cells.size()==27);
 assert(cells.count({50,50}) && cells.count({50,-50}));
 for(auto& t:tiles)t.tile_flags=0;
 assert(captured_wave_cells(frame,C3X_RENDERER_TILE_RENDER).empty());
 // No topology payload is necessary for its new revision to invalidate data.
 frame.world_topology_revision=1;assert(wave_geometry_scope(frame,7,3)!=scope);frame.world_topology_revision=0;
 frame.tile_width=64;assert(wave_geometry_scope(frame,7,3)!=scope);frame.tile_width=128;
 frame.world_wrap_y=0;assert(wave_geometry_scope(frame,7,3)!=scope);frame.world_wrap_y=1;
 assert(wave_geometry_scope(frame,8,3)!=scope && wave_geometry_scope(frame,7,4)!=scope);
}
''')

    def test_regions_cover_clipped_edges_and_keep_camera_independent_identity(self):
        run_cpp(r'''
#include <cassert>
#include <algorithm>
#include <vector>
#include "Renderer/native/render_core/raster_grid.h"
int main() {
 using namespace c3x_renderer::render_core;
 assert(static_region_identity(17,2)!=static_region_identity(17,3));
 assert(static_region_identity(17,2)!=static_region_identity(18,2));
 for(int size:{1,127,128,129,1192,2240})for(int phase=0;phase<128;++phase) {
   RasterRegionAxis grid(size,phase);
   assert(grid.count<=(size+127)/128+1);
   std::vector<int> coverage(size,0);
   for(int i=0;i<grid.count;++i) {
     int origin=grid.start(i),low=std::max(0,origin),high=std::min(size,origin+128);
     assert(low<high && low-origin>=0 && high-origin<=128);
     for(int p=low;p<high;++p){assert(grid.at(p)==i);++coverage[p];}
   }
   for(int count:coverage)assert(count==1);
 }
 for(int zoom:{64,96,128,160,192})for(int anchor=-129;anchor<130;++anchor)for(int pan=-129;pan<130;pan+=7) {
   int phase=raster_anchor_phase(anchor,17,zoom,128);
   int next=raster_anchor_phase(anchor+pan,17,zoom,128);
   for(int pixel=0;pixel<2240;pixel+=117) {
     int origin=raster_region_floor(pixel,phase,128);
     int moved=raster_region_floor(pixel+pan,next,128);
     assert(origin-anchor==moved-(anchor+pan));
   }
 }
}
''')

    def test_production_signature_reuses_only_unchanged_static_inputs(self):
        run_cpp(r'''
#include <cassert>
#include "Renderer/native/terrain_scene_runtime.h"
int main() {
 c3x_renderer_tile_v1 tiles[2]={};
 for(auto& t:tiles){t.tile_flags=C3X_RENDERER_TILE_RENDER;t.terrain_type=2;}
 tiles[1].tile_x=2;tiles[1].anchor_x=128;
 c3x_renderer_frame_v1 frame={};frame.tile_count=2;frame.tiles=tiles;
 frame.target_width=2240;frame.target_height=1192;frame.tile_width=128;frame.tile_height=64;
 frame.hour=12;frame.world_width_tiles=100;frame.world_height_tiles=100;frame.world_wrap_x=1;
 auto signature=[&](){return c3x_renderer::terrain_frame_signature(frame,7,3);};
 auto before=signature();
 for(auto& t:tiles){t.anchor_x+=9;t.anchor_y-=17;}
 auto moved=signature();assert(moved.geometry==before.geometry && moved.complete!=before.complete);
 // Environment, projection, scene, ownership, ordering, world and device
 // remain conservative invalidation boundaries for backdrop color AND depth.
 frame.hour=18;assert(signature().geometry!=moved.geometry);frame.hour=12;
 frame.season=1;assert(signature().geometry!=moved.geometry);frame.season=0;
 frame.tile_width=64;assert(signature().geometry!=moved.geometry);frame.tile_width=128;
 frame.target_height=600;assert(signature().geometry!=moved.geometry);frame.target_height=1192;
 tiles[0].terrain_type=11;assert(signature().geometry!=moved.geometry);tiles[0].terrain_type=2;
 tiles[0].tile_flags=0;assert(signature().geometry!=moved.geometry);tiles[0].tile_flags=C3X_RENDERER_TILE_RENDER;
 std::swap(tiles[0],tiles[1]);assert(signature().geometry!=moved.geometry);std::swap(tiles[0],tiles[1]);
 frame.world_wrap_y=1;assert(signature().geometry!=moved.geometry);frame.world_wrap_y=0;
 frame.world_topology_count=1;frame.world_topology_revision=4;auto topology=signature();
 frame.world_topology_revision=5;assert(signature().geometry!=topology.geometry);
 assert(c3x_renderer::terrain_frame_signature(frame,8,3).geometry!=signature().geometry);
 assert(c3x_renderer::terrain_frame_signature(frame,7,4).geometry!=signature().geometry);
}
''', sources=("Renderer/native/terrain_scene_runtime.cpp",))


if __name__ == "__main__":
    unittest.main()
