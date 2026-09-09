"""Execute production indexing and projection against independent witnesses."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ZoomMeshTests(unittest.TestCase):
    def test_retained_ground_dependencies_and_bounded_admission(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        retained = "struct CachedGroundGrid {" + source.split("struct CachedGroundGrid {", 1)[1].split("struct RiverNode", 1)[0]
        lookup = "auto ground_key=" + source.split("auto ground_key=", 1)[1].split("            auto append_ground_layer", 1)[0]
        admission = "if(!pending_ground_grids.empty()){\n" + source.split("if(!pending_ground_grids.empty()){\n", 1)[1].split("            QueryPerformanceCounter(&phase_end);ground_ticks", 1)[0]
        program = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
''' + retained + r'''
struct State {
 struct Frame {int world_width_tiles=100,world_height_tiles=100,world_wrap_x=1,world_wrap_y=0;
  int world_topology_revision=1,tile_width=128,tile_height=64,target_width=128,target_height=64;} frame;
 struct Tile {int tile_x=0,tile_y=0,anchor_x=0,anchor_y=0;} tile;
 std::uint64_t content_revision=1,tile_geometry_epoch=1;
 std::unordered_map<std::uint64_t,CachedGroundTile> ground_grid_cache;
 std::size_t ground_grid_cache_bytes=0,natural_mesh_cache_budget=8192,natural_mesh_cache_capacity=4;
 std::unordered_map<std::uint64_t,std::uint64_t> semantic_by_coordinate{{9,11}};
 struct World {unsigned value=17;unsigned at(std::size_t) const{return value;}};
 struct Coast {World data;std::uint64_t revision=23;
  World const& world() const{return data;}
  std::uint64_t node_revision(std::uint64_t) const{return revision;}} world_coast;
 auto coordinate_key(int x,int y){return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);}
 std::uint64_t tile_content_signature(Tile const&) const{return 3;}
 bool run(int x,bool record){
  tile.tile_x=x;tile.anchor_x=x*64;
  bool retain_ground_grids=true,prewarming=false;
  std::unordered_map<std::uint64_t,std::uint64_t> dependencies,coast_dependencies;
  std::unordered_map<std::size_t,std::uint32_t> world_dependencies;
''' + lookup + r'''
  bool hit=ground_hit;
  if(record){
   CachedGroundGrid grid;grid.divisions=1;grid.layer=.5f;grid.vertices.resize(4);grid.samples.resize(4);
   pending_ground_grids.push_back(std::move(grid));
   dependencies[9]=semantic_by_coordinate.at(9);coast_dependencies[5]=world_coast.revision;world_dependencies[7]=world_coast.data.value;
  }else if(hit){assert(dependencies.at(9)==semantic_by_coordinate.at(9));assert(coast_dependencies.at(5)==world_coast.revision);assert(world_dependencies.at(7)==world_coast.data.value);}
''' + admission + r'''
  std::size_t sum=0;for(auto const& entry:ground_grid_cache)sum+=entry.second.bytes;
  assert(sum==ground_grid_cache_bytes && sum<=natural_mesh_cache_budget && ground_grid_cache.size()<=natural_mesh_cache_capacity);
  return hit;
 }
};
int main(){
 State s;assert(!s.run(0,true));assert(s.run(0,false));
 assert(!s.run(100,false) && !s.run(-100,false));
 s.world_coast.revision++;assert(!s.run(0,false) && !s.ground_grid_cache_bytes);
 s.run(0,true);s.world_coast.data.value++;assert(!s.run(0,false));
 s.run(0,true);s.semantic_by_coordinate[9]++;assert(!s.run(0,false));
 s.run(0,true);s.content_revision++;assert(!s.run(0,false));
 s.run(0,true);s.frame.world_topology_revision++;assert(!s.run(0,false));
 s.run(0,true);s.frame.world_width_tiles+=2;assert(!s.run(0,false));
 s.run(0,true);s.frame.world_wrap_x=0;assert(!s.run(0,false));
 s.run(0,true);s.frame.tile_width=192;s.frame.tile_height=96;assert(s.run(0,false));
 State bounded;for(int x:{-2,-1,0,1})bounded.run(x,true);
 assert(bounded.ground_grid_cache.size()==4);
 bounded.run(100,true);assert(bounded.ground_grid_cache.size()==4 && !bounded.run(100,false));
 State tiny;tiny.natural_mesh_cache_budget=1;tiny.run(0,true);assert(tiny.ground_grid_cache.empty());
 State extended;extended.run(0,true);auto before=extended.ground_grid_cache_bytes;
 assert(extended.run(0,true) && extended.ground_grid_cache_bytes>before);
 assert(extended.ground_grid_cache.begin()->second.grids.size()==2);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-ground-cache-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_retained_ground_grid_preserves_projection_and_raw_normals(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        retained = "struct CachedGroundGrid {" + source.split("struct CachedGroundGrid {", 1)[1].split("struct CachedGroundTile", 1)[0]
        program = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
''' + retained + r'''
int main(){
 for(int divisions:{8,12,16,24,32})for(float layer:{.5f,1.f,9.f}){
  CachedGroundGrid grid;grid.divisions=divisions;grid.layer=layer;
  grid.vertices.resize((divisions+1)*(divisions+1));grid.samples.resize(grid.vertices.size());
  for(unsigned i=0;i<grid.vertices.size();i++){
   auto& vertex=grid.vertices[i];auto* fields=reinterpret_cast<float*>(&vertex);
   for(unsigned j=0;j<sizeof(Vertex)/sizeof(float);j++)fields[j]=float(i+j)*.123f;
   grid.samples[i]={layer==.5f?0.f:float(i%37)*.73f,float(int(i%7)-3)*.031f,float(int(i%13)-6)*.019f};
  }
  for(int width:{64,96,128,160,192})for(int height:{32,48,64,80,96})
  for(unsigned i=0;i<grid.vertices.size();i++){
   float u=float(i%unsigned(divisions+1))/divisions,v=float(i/unsigned(divisions+1))/divisions;
   float half_w=float(width)*.5f,half_h=float(height)*.5f;
   Vertex expected=grid.vertices[i];
   float h=grid.samples[i][0]*(float(width)/224.f*.82f);
   float gx=0.f+(half_w+(u-v)*half_w),gy=0.f+(u+v)*half_h;
   expected.x=gx;expected.y=gy-h;expected.z=gy+h*.75f;
   if(layer==1.f || layer==9.f){
    float su=grid.samples[i][1]*1.f/(2.f*.006f*float(width));
    float sv=grid.samples[i][2]*-1.f/(2.f*.006f*float(width));
    float length=std::sqrt(su*su+sv*sv+1.f);
    expected.normal_x=-su/length;expected.normal_y=-sv/length;expected.normal_z=1.f/length;
   }
   auto actual=grid.project(i,width,height);
   assert(!std::memcmp(&actual,&expected,sizeof(Vertex)));
  }
 }
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-ground-reprojection-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_viewport_draw_identity_rejects_eviction_and_stale_frames(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        caches = "struct CachedViewport {" + source.split("struct CachedViewport {", 1)[1].split("struct CachedVertexChunk", 1)[0]
        restore = "    bool restore_viewport_geometry(" + source.split(
            "    bool restore_viewport_geometry(", 1)[1].split("    bool draw_cached_geometry(", 1)[0]
        program = r'''
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>
using c3x_renderer_u32=unsigned;
namespace c3x_renderer {struct TerrainFrameSignature {std::uint64_t complete=0;};}
struct c3x_renderer_tile_v1 {int x=0;};
struct c3x_renderer_frame_v1 {unsigned tile_count=2;c3x_renderer_tile_v1 const* tiles;int world_topology_revision=9;};
''' + caches + r'''
struct CachedTileGeometry {bool shared_natural=false;std::uint64_t version=0,natural_signature=0,natural_version=0;};
struct State {
 std::unordered_multimap<int,CachedTileGeometry> tile_geometry_cache;
 CachedGeometry geometry_cache;int geometry_world_revision=0,appends=0;
 void append_tile_geometry(CachedTileGeometry&,c3x_renderer_tile_v1 const&,bool animated){assert(animated);++appends;}
''' + restore + r'''
};
int main(){
 State state;CachedViewport view;view.signature.complete=10;
 view.tile_keys={{1,100},{2,200}};view.tiles={{3},{5}};
 view.replacement_flags={1,0};view.fallback_indices={1};view.rendered_tile_count=1;
 view.fallback_tile_count=1;view.textured_tile_count=1;
 c3x_renderer_frame_v1 frame={2,view.tiles.data(),9};
 state.tile_geometry_cache.emplace(1,CachedTileGeometry{false,100,99,100});
 state.tile_geometry_cache.emplace(99,CachedTileGeometry{true,100,0});
 assert(!state.restore_viewport_geometry(view,frame,{10}) && !state.appends);
 state.tile_geometry_cache.emplace(2,CachedTileGeometry{false,201,0});
 assert(!state.restore_viewport_geometry(view,frame,{10}) && !state.appends);
 state.tile_geometry_cache.emplace(2,CachedTileGeometry{false,200,0});
 assert(!state.restore_viewport_geometry(view,frame,{11}) && !state.appends);
 assert(state.restore_viewport_geometry(view,frame,{10}) && state.appends==2);
 assert(state.geometry_cache.valid && state.geometry_cache.tile_keys==view.tile_keys);
 assert(state.geometry_cache.replacement_flags==view.replacement_flags);
 assert(state.geometry_cache.fallback_indices==view.fallback_indices && state.geometry_world_revision==9);
 state.tile_geometry_cache.find(99)->second.version=101;state.appends=0;
 assert(!state.restore_viewport_geometry(view,frame,{10}) && !state.appends);
 state.tile_geometry_cache.erase(99);state.appends=0;state.geometry_cache.clear();
 assert(!state.restore_viewport_geometry(view,frame,{10}) && !state.appends && !state.geometry_cache.valid);
 frame.tile_count=1;assert(!state.restore_viewport_geometry(view,frame,{10}));
 // Zero identities are intentionally non-rendered records, not missing owners.
 frame.tile_count=2;view.tile_keys={{0,0},{2,200}};
 assert(state.restore_viewport_geometry(view,frame,{10}) && state.appends==1);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-viewport-identities-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_shared_draw_lists_pin_owners_and_grow_amortized(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        append = "    void append_tile_geometry(" + source.split(
            "    void append_tile_geometry(", 1)[1].split("    bool restore_viewport_geometry(", 1)[0]
        program = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>
constexpr int geometry_layer_count=2,C3X_RENDERER_TILE_RENDER=1;
struct Ref {int references=1;void AddRef(){++references;}void Release(){--references;}};
struct CachedVertexChunk {Ref *buffer=nullptr,*indices=nullptr;int translation_x=0,translation_y=0,projected=0;};
struct c3x_renderer_tile_v1 {int tile_flags=1,anchor_x=0,anchor_y=0;};
struct Anchor {int anchor_x=0,anchor_y=0;};
struct CachedTileGeometry {
 bool prefetched=false,shared_natural=false;std::size_t byte_count=0;
 std::uint64_t natural_signature=0,last_used=0,animation_epoch=0;
 int anchor_x=0,anchor_y=0;std::vector<Anchor> resource_anchors;
 std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
};
struct State {
 std::size_t prefetched_geometry_bytes=20;std::uint64_t tile_geometry_epoch=5;
 std::unordered_map<int,CachedTileGeometry> tile_geometry_cache;
 std::vector<Anchor> resource_anchors;std::vector<int> geometry_footprints;
 std::array<std::vector<CachedVertexChunk>,geometry_layer_count> geometry_vertex_buffers;
 int tile_footprint(CachedTileGeometry const&,c3x_renderer_tile_v1 const&){return 1;}
 CachedVertexChunk project_natural_chunk(CachedVertexChunk chunk,c3x_renderer_tile_v1 const&){chunk.projected=1;return chunk;}
''' + append + r'''
};
int main(){
 Ref ground,world,indices;State state;CachedTileGeometry tile;
 tile.prefetched=true;tile.byte_count=20;tile.natural_signature=99;
 tile.buffers[0].push_back({&ground,&indices});tile.resource_anchors.push_back({1,2});
 auto& owner=state.tile_geometry_cache[99];owner.shared_natural=true;owner.byte_count=100;
 owner.buffers[1].push_back({&world,&indices});
 std::size_t copied_capacity=0;
 for(int i=0;i<4000;i++){
  auto old=state.geometry_vertex_buffers[0].capacity();
  state.append_tile_geometry(tile,{1,i*2,40},true);
  if(state.geometry_vertex_buffers[0].capacity()!=old)copied_capacity+=old;
 }
 assert(copied_capacity<16000); // exact per-tile reserve is quadratic, ~8 million.
 assert(!tile.prefetched && !state.prefetched_geometry_bytes);
 assert(tile.last_used==5 && owner.last_used==5 && owner.animation_epoch==5);
 assert(ground.references==4001 && world.references==4001 && indices.references==8001);
 assert(state.resource_anchors.size()==4000 && state.resource_anchors.back().anchor_x==7999);
 assert(state.geometry_vertex_buffers[1].back().translation_x==7998);
 assert(state.geometry_vertex_buffers[1].back().projected && !owner.buffers[1][0].projected);
 for(auto& layer:state.geometry_vertex_buffers)for(auto& chunk:layer){chunk.buffer->Release();chunk.indices->Release();}
 assert(ground.references==1 && world.references==1 && indices.references==1);
 // Caster-only records must not publish resource anchors or pin indefinitely.
 state.tile_geometry_epoch=6;state.append_tile_geometry(tile,{0,0,0},false);
 assert(state.resource_anchors.size()==4000 && owner.last_used==6 && owner.animation_epoch==5);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-shared-draws-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_pixel_prefetch_borrows_both_camera_and_world_layers(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        borrow = "        for (auto const & contributor : pending_pixel_block.key) {" + source.split(
            "        for (auto const & contributor : pending_pixel_block.key) {", 1)[1].split(
            "        ViewportShaderSettings settings={};", 1)[0]
        program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>
constexpr int geometry_layer_count=3,geometry_natural_terrain=1;
struct CachedVertexChunk {int id,translation_x=0,translation_y=0,projected=0;};
struct c3x_renderer_tile_v1 {int tile_x=0,tile_y=0;};
struct Entry {
 bool shared_natural=false;std::uint64_t version=0,natural_signature=0,natural_version=0;
 int tile_x=0,tile_y=0;
 std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
};
struct State {
 std::unordered_multimap<int,Entry> tile_geometry_cache;
 struct Contributor {std::uint64_t mesh;int x,y;};
 struct {std::vector<Contributor> key;} pending_pixel_block;
 int pixel_prepare_cursor=0,submissions=0;
 std::array<std::vector<CachedVertexChunk>,geometry_layer_count> buffers;
 CachedVertexChunk project_natural_chunk(CachedVertexChunk chunk,c3x_renderer_tile_v1 record){
  chunk.projected=record.tile_x*100+record.tile_y;return chunk;
 }
 bool prepare(){
''' + borrow + r'''
  ++submissions;return true;
 }
};
int main(){
 State s;Entry camera;camera.version=7;camera.natural_signature=99;camera.natural_version=7;
 camera.tile_x=15;camera.tile_y=47;camera.buffers[0].push_back({1});
 Entry world;world.version=7;world.shared_natural=true;world.buffers[1].push_back({2});
 s.tile_geometry_cache.emplace(42,std::move(camera));s.tile_geometry_cache.emplace(99,std::move(world));
 s.pending_pixel_block.key.push_back({7,120,60});
 assert(s.prepare() && s.submissions==1 && !s.pixel_prepare_cursor);
 assert(s.buffers[0].size()==1 && s.buffers[1].size()==1 && s.buffers[2].empty());
 assert(s.buffers[0][0].id==1 && !s.buffers[0][0].projected);
 assert(s.buffers[1][0].id==2 && s.buffers[1][0].projected==1547);
 assert(s.buffers[1][0].translation_x==120 && s.buffers[1][0].translation_y==60);
 assert(!s.tile_geometry_cache.find(99)->second.buffers[1][0].projected);
 // An evicted owner must cancel preparation, never cache incomplete pixels.
 s.tile_geometry_cache.erase(99);s.buffers={};
 assert(s.prepare() && s.submissions==1 && s.pixel_prepare_cursor==1);
 // A shared owner's coincident version is not itself a camera contributor.
 s.tile_geometry_cache.clear();Entry orphan;orphan.version=7;orphan.shared_natural=true;
 s.tile_geometry_cache.emplace(99,std::move(orphan));
 assert(s.prepare() && s.submissions==1 && s.pixel_prepare_cursor==2);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-world-prefetch-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_shared_world_bounds_cover_every_zoom_and_reflected_height(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        project = "CachedVertexChunk project_natural_chunk(" + source.split(
            "CachedVertexChunk project_natural_chunk(", 1)[1].split(
            "    c3x_renderer::TileFootprint tile_footprint", 1)[0]
        program = r'''
#include <cassert>
#include <climits>
#include "Renderer/lab/shared/natural/ground.h"
using LONG=long;
struct c3x_renderer_tile_v1 {int tile_x,tile_y;};
struct CachedVertexChunk {
 float natural_projection[4]={};
 struct {float low[3],high[3];} world_bounds;
 struct {LONG left,top,right,bottom;} bounds;
};
struct State {int shadow_tile_width=128,shadow_tile_height=64,height=1192;
''' + project + r'''
};
int main(){
 State state;CachedVertexChunk stored={};
 stored.world_bounds={{30.5f,-17.25f,-.03f},{33.5f,-14.7f,1.73f}};
 for(int width:{64,96,128,160,192})for(int height:{480,1192,2160}){
  state.shadow_tile_width=width;state.shadow_tile_height=width/2;state.height=height;
  auto chunk=state.project_natural_chunk(stored,{15,47});
  assert(chunk.natural_projection[0]==31 && chunk.natural_projection[1]==-16);
  assert(chunk.natural_projection[2]==width && chunk.natural_projection[3]==height);
  c3x_renderer::fidelity::GroundProjection project{31,-16,width*.5f,width*.25f,width/224.f*.82f,float(height)};
  for(int i=0;i<10000;i++){
   float x=30.5f+(i%101)/100.f*3,y=-17.25f+(i%103)/102.f*2.55f,z=-.03f+(i%107)/106.f*1.76f;
   auto p=project(x,y,z*112.f);
   assert(p.x>=chunk.bounds.left && p.x<chunk.bounds.right);
   assert(p.y>=chunk.bounds.top && p.y<chunk.bounds.bottom);
   float reflection=2*(112.f*.82f*width/224.f)*std::max(0.f,z-2.5f/112.f);
   float max_reflection=2*(112.f*.82f*width/224.f)*std::max(0.f,stored.world_bounds.high[2]-2.5f/112.f);
   assert(p.y+reflection<chunk.bounds.bottom+max_reflection);
  }
 }
 // Computing a camera view must not mutate the cached world's projection.
 assert(stored.natural_projection[2]==0);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-shared-world-") as directory:
            cpp, binary = Path(directory) / "test.cpp", Path(directory) / "test"
            cpp.write_text(program)
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_benchmark_validates_centered_and_historical_ladders(self):
        from Renderer.native.compare_zoom_benchmark import run
        with tempfile.TemporaryDirectory(prefix="c3x-zoom-ladder-") as directory:
            root = Path(directory)
            for levels in ((128,112,96,80,64), (128,96,64,192,160), (128,64,96,192,160)):
                lines = []
                for cycle in range(2):
                    for width in levels:
                        lines.append(f"ZOOM cycle={cycle} width={width} result=1")
                        if cycle:
                            lines.append(f"ZOOM parity width={width} changed=0 error=0 status=pass")
                lines.append("BIQ 100x100 viewport: 100 visible tiles, 0 fallback")
                (root / "benchmark.log").write_text("\n".join(lines) + "\n")
                if levels[1] == 64:
                    with self.assertRaisesRegex(ValueError, "unexpected zoom ladder"):
                        run("test", root, "zoom")
                else:
                    self.assertEqual(len(run("test", root, "zoom")[1]), 10)

    def test_backdrop_budget_preserves_current_view_and_evicts_lru(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        eviction = "bool make_resource_backdrop_room(" + source.split("bool make_resource_backdrop_room(", 1)[1].split("    void reset_resource_buffers", 1)[0]
        program = r'''
#include <cassert>
#include <vector>
#include <cstdint>
struct State {
 struct Block {std::uint64_t signature,used;std::size_t bytes;int color=1,depth=1;};
 std::vector<Block> resource_backdrops={{1,1,30},{2,2,30},{3,3,30}};
 std::size_t resource_backdrop_bytes=90,resource_backdrop_cache_budget=100;
 unsigned releases=0;
 void release(int& resource){assert(resource);resource=0;++releases;}
''' + eviction + r'''
};
int main(){
 State state;
 assert(!state.make_resource_backdrop_room(101,2) && !state.releases);
 assert(state.make_resource_backdrop_room(20,2));
 assert(state.resource_backdrops.size()==2 && state.resource_backdrops[0].signature==2);
 assert(state.resource_backdrop_bytes==60 && state.releases==2);
 assert(state.make_resource_backdrop_room(60,2));
 assert(state.resource_backdrops.size()==1 && state.resource_backdrop_bytes==30 && state.releases==4);
 assert(!state.make_resource_backdrop_room(80,2) && state.resource_backdrop_bytes==30);
 // A changed static signature cannot keep using a stale background.
 assert(state.make_resource_backdrop_room(80,4) && state.resource_backdrop_bytes==0);
 assert(state.resource_backdrops.empty() && state.releases==6);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-backdrop-budget-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_direct_grid_preserves_complete_triangle_stream(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        grid = "auto append_ground_layer = " + source.split("auto append_ground_layer = ", 1)[1].split("            auto append_feature_instance", 1)[0]
        retained = "struct CachedGroundGrid {" + source.split("struct CachedGroundGrid {", 1)[1].split("struct RiverNode", 1)[0]
        program = r'''
#include <array>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
using UINT=unsigned;
''' + retained + r'''
int main(){
 bool ground_hit=false,retain_ground_grids=false,prewarming=false,reuse_nested_ground_grids=true;
 std::unordered_map<int,CachedGroundTile> retained_cache;
 auto retained_ground=retained_cache.end();
 std::vector<CachedGroundGrid> pending_ground_grids;
 unsigned frame_ground_grid_hits=0;
 struct {int tile_width=128,tile_height=64;} frame;
 auto river_node_distance=[](float,float,unsigned){return 1000.f;};
 struct Point {float relief[3]={},normal_delta[2]={};} point;
 auto ground_point_at=[&](float,float)->Point&{return point;};
 bool stop=false;unsigned samples=0;
 auto cancelled=[&](){return stop;};
 auto make_ground_vertex=[&](float u,float v,float layer){
  ++samples;Vertex out={};out.x=u*137;out.y=v*71;out.z=layer+u*v;
  out.u=u;out.v=v;out.normal_x=u-.5f;out.normal_y=v-.5f;out.normal_z=1;
  out.world_x=53+u;out.world_y=17-v;out.material_grass=layer;
  out.material_plains=u==0?-0.f:u;return out;
 };
''' + grid + r'''
 for(int n:{1,8,12,16,24,32})for(float layer:{.5f,1.f,4.f,5.f,9.f,10.f}){
  std::vector<Vertex> old,packed;std::vector<UINT> indices;
  append_ground_layer(old,layer,n);
  samples=0;append_ground_layer(packed,layer,n,&indices);
  assert(samples==unsigned((n+1)*(n+1)) && packed.size()==samples);
  assert(indices.size()==old.size() && indices.size()==unsigned(n*n*6));
  for(std::size_t i=0;i<indices.size();++i){
   assert(indices[i]<packed.size());
   assert(!std::memcmp(&packed[indices[i]],&old[i],sizeof(Vertex)));
  }
  // Reuse a larger scratch allocation for a smaller grid: no stale triangles.
  append_ground_layer(packed,layer,1,&indices);
  assert(packed.size()==4 && indices==std::vector<UINT>({0,1,3,0,3,2}));
 }
 // Execute the actual cached-grid selection/indexing against every supported
 // fine/coarse pair. Compare all channels, not only the projected position.
 ground_hit=true;retained_cache.emplace(0,CachedGroundTile{});retained_ground=retained_cache.find(0);
 for(int fine:{8,12,16,24,32})for(int coarse:{8,12,16,24,32})
 for(float layer:{.5f,1.f,9.f})for(int width:{64,96,128,160,192}){
  frame.tile_width=width;frame.tile_height=width/2;
  auto& grids=retained_ground->second.grids;grids.clear();grids.emplace_back();
  auto& cached=grids.back();cached.divisions=fine;cached.layer=layer;
  for(int y=0;y<=fine;++y)for(int x=0;x<=fine;++x){
   cached.vertices.push_back(make_ground_vertex(float(x)/fine,float(y)/fine,layer));
   cached.samples.push_back({float(x+y)*.25f,float(x)*.017f,float(y)*-.023f});
  }
  int stride=cached.sample_stride(coarse);
  assert(bool(stride)==(fine>=coarse && fine%coarse==0));
  assert(!cached.sample_stride(0) && !cached.sample_stride(-1));
  std::vector<Vertex> packed;std::vector<UINT> indices;samples=0;
  append_ground_layer(packed,layer,coarse,&indices);
  assert(indices.size()==unsigned(coarse*coarse*6));
  if(!stride){assert(samples==unsigned((coarse+1)*(coarse+1)));continue;}
  assert(!samples);
  CachedGroundGrid expected;expected.divisions=coarse;expected.layer=layer;
  for(int y=0;y<=coarse;++y)for(int x=0;x<=coarse;++x){
   auto from=(y*stride)*(fine+1)+x*stride;
   expected.vertices.push_back(cached.vertices[from]);expected.samples.push_back(cached.samples[from]);
  }
  for(unsigned i=0;i<packed.size();++i){
   auto vertex=expected.project(i,width,width/2);
   if(layer==9.f)vertex.river_branch_count=1000.f;
   assert(!std::memcmp(&packed[i],&vertex,sizeof(Vertex)));
  }
  reuse_nested_ground_grids=false;samples=0;append_ground_layer(packed,layer,coarse,&indices);
  assert(samples==(fine==coarse?0u:unsigned((coarse+1)*(coarse+1))));
  reuse_nested_ground_grids=true;
 }
 stop=true;samples=0;std::vector<Vertex> aborted;std::vector<UINT> empty;
 append_ground_layer(aborted,.5f,32,&empty);assert(!samples && empty.empty());
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-indexed-grid-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_memory_receipt_distinguishes_missing_samples(self):
        from Renderer.native.compare_zoom_benchmark import memory_samples
        with tempfile.TemporaryDirectory(prefix="c3x-camera-memory-") as directory:
            root = Path(directory)
            log = root / "benchmark.log"
            log.write_text("ZOOM cycle=0\n")
            self.assertIsNone(memory_samples(root))
            log.write_text("CAMERA memory available_virtual=2000 largest_free_region=1500 total_virtual=4000\n"
                           "CAMERA memory available_virtual=1800 largest_free_region=1200 total_virtual=4000\n")
            self.assertEqual(memory_samples(root), {"samples": 2,
                "minimum_available_virtual_bytes": 1800,
                "minimum_largest_free_region_bytes": 1200,
                "total_virtual_bytes": 4000})

    def test_budget_eviction_preserves_active_frame(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        eviction = "    bool make_tile_cache_room(" + source.split("    bool make_tile_cache_room(", 1)[1].split("    bool cache_geometry_layer(", 1)[0]
        program = r'''
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <unordered_map>
template<std::size_t N,class... A> int sprintf_s(char(&out)[N],char const*format,A...args){return std::snprintf(out,N,format,args...);}
struct State {
    struct Item {std::size_t byte_count;unsigned last_used;bool prefetched;int buffers=0;std::uint64_t animation_epoch=0;};
    using CachedTileGeometry=Item;
    unsigned viewport_cache_capacity=32;
    struct Trace {void write(char const*,char const*,bool){}} trace;
    std::unordered_map<int,Item> tile_geometry_cache;
    std::size_t tile_geometry_cache_bytes=60,prefetched_geometry_bytes=10;
    std::size_t tile_geometry_cache_budget=100,tile_geometry_cache_capacity=2048;
    unsigned tile_geometry_epoch=3,frame_tiles_evicted=0,cache_evictions=0;
    void release_geometry_vertex_buffers(int&){}
''' + eviction + r'''
};
int main(){
 State s;s.tile_geometry_cache={{1,{10,1,true}},{2,{20,2,false}},{3,{30,3,false}}};
 assert(s.make_tile_cache_room(10) && s.tile_geometry_cache.size()==3);
 assert(s.make_tile_cache_room(50) && s.tile_geometry_cache.size()==2);
 assert(!s.tile_geometry_cache.count(1) && s.tile_geometry_cache.count(3));
 assert(s.tile_geometry_cache_bytes==50 && !s.prefetched_geometry_bytes && s.frame_tiles_evicted==1);
 assert(!s.make_tile_cache_room(80) && s.tile_geometry_cache.size()==1);
 assert(s.tile_geometry_cache_bytes==30 && s.tile_geometry_cache.count(3));
 assert(!s.make_tile_cache_room(71) && s.tile_geometry_cache_bytes==30);
 assert(s.make_tile_cache_room(70) && s.tile_geometry_cache.size()==1);
 State favored;favored.tile_geometry_cache={{1,{10,1,true}},{2,{20,2,false}},{3,{30,3,false}}};
 favored.tile_geometry_cache.at(1).animation_epoch=1;
 assert(favored.make_tile_cache_room(50));
 assert(favored.tile_geometry_cache.count(1) && !favored.tile_geometry_cache.count(2));
 assert(favored.tile_geometry_cache.count(3) && favored.tile_geometry_cache_bytes==40);
 // Preference expires and never pins unused geometry against the hard cap.
 favored.tile_geometry_epoch=40;favored.tile_geometry_cache.at(3).last_used=40;
 assert(favored.make_tile_cache_room(70) && !favored.tile_geometry_cache.count(1));
 State bounded;bounded.tile_geometry_cache={{1,{10,1,true}},{2,{20,2,false}},{3,{30,3,false}}};
 bounded.tile_geometry_cache.at(1).animation_epoch=1;
 bounded.tile_geometry_cache.at(2).animation_epoch=2;
 assert(bounded.make_tile_cache_room(70) && bounded.tile_geometry_cache.size()==1);
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-cache-pressure-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            subprocess.run([compiler, "-std=c++17", "-O2", str(cpp), "-o", str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_exact_indexing_and_world_reprojection(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        source = (ROOT / "Renderer/native/c3x_renderer.cpp").read_text()
        equality = source.split("struct VertexHash {", 1)[1].split("struct GroundPoint {", 1)[0]
        indexing = "//" + source.split("        // Index into the packed array", 1)[1].split("        CachedVertexChunk chunk;", 1)[0]
        indexing = indexing.rsplit("        }", 1)[0]
        compact = source.split("        std::vector<std::uint16_t> narrow_indices;", 1)[1].split("        while (prefetch", 1)[0]
        program = r'''
#include <array>
#include <atomic>
#include <cassert>
#include <climits>
#include <cstring>
#include <unordered_map>
#include "Renderer/lab/shared/natural/ground.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
using UINT=unsigned;
''' + "struct VertexHash {" + equality + r'''
bool check(bool feature) {
    bool pickup_profile=true,compact_feature=feature,prefetch=false;
    std::atomic<bool> const* foreground_pending=nullptr;
    std::size_t hash_stride=sizeof(Vertex);
    std::vector<Vertex> vertices;
    for(unsigned i=0;i<30000;i++) {
        Vertex v={};unsigned key=(i*73)%1789;
        v.x=float(key%41);v.y=float(key/41);v.z=float(key%13);
        v.u=key*.125f;v.normal_z=1;v.world_x=key*.25f;
        // Compact features omit this channel. Signed zeros remain byte-distinct.
        v.material_grass=float(i%3);v.v=i%7==0?-0.f:0.f;
        vertices.push_back(v);
    }
    std::vector<Vertex> expected;
    std::vector<UINT> expected_indices;
    std::unordered_map<Vertex,UINT,VertexHash,VertexEqual> old(0,VertexHash{hash_stride,feature},VertexEqual{hash_stride,feature});
    for(auto const& v:vertices){auto at=old.emplace(v,UINT(expected.size()));
        if(at.second)expected.push_back(v);expected_indices.push_back(at.first->second);}
    std::vector<Vertex> packed;
    std::vector<UINT> indices;
''' + indexing + r'''
    assert(indices==expected_indices && packed.size()==expected.size());
    assert(std::memcmp(packed.data(),expected.data(),packed.size()*sizeof(Vertex))==0);
    return true;
}
void check_index_width(std::size_t vertex_count) {
    enum {DXGI_FORMAT_R16_UINT=16,DXGI_FORMAT_R32_UINT=32};
    struct {int index_format=DXGI_FORMAT_R32_UINT;std::size_t byte_count=0;} chunk;
    std::vector<Vertex> packed(vertex_count);
    std::vector<UINT> indices={0,1,UINT(vertex_count-1),0,UINT(vertex_count-1),1};
    std::size_t vertex_stride=76;
    std::vector<std::uint16_t> narrow_indices;
''' + compact + r'''
    bool narrow=vertex_count<=65535;
    assert(chunk.index_format==(narrow?DXGI_FORMAT_R16_UINT:DXGI_FORMAT_R32_UINT));
    assert(chunk.byte_count==vertex_count*vertex_stride+indices.size()*(narrow?2:4));
    if(narrow)for(std::size_t i=0;i<indices.size();++i)assert(narrow_indices[i]==indices[i]);
    else assert(narrow_indices.empty());
}
int main(){
    for(auto count:{3u,65535u,65536u,70000u})check_index_width(count);
    assert(check(false) && check(true));
    // IEEE float grids share low mantissa bits. A power-of-two table must
    // disperse high bits too, or every regular vertex starts in one bucket.
    std::vector<UINT> slots(8192,UINT_MAX);std::size_t probes=0;
    VertexHash hash;
    for(unsigned i=0;i<4096;++i){Vertex v={};v.x=(i%64)*.125f;v.y=(i/64)*.125f;
        auto slot=hash(v)&8191u;++probes;
        while(slots[slot]!=UINT_MAX){++probes;slot=(slot+1)&8191u;}
        slots[slot]=i;
    }
    assert(probes<4096*4);
    using c3x_renderer::fidelity::GroundProjection;
    for(int original:{64,80,96,112,128,160,192})for(int next:{64,80,96,112,128,160,192})
    for(int target:{480,640,1080})for(int i=0;i<10000;i++){
        float x=31.f+(i%65)/64.f,y=-17.f+(i%67)/64.f;
        float h=2.5f+(i%3001)*.1253f;
        GroundProjection before{31,-17,original*.5f,original*.25f,original/224.f*.82f,640};
        GroundProjection after{31,-17,next*.5f,next*.25f,next/224.f*.82f,float(target)};
        auto cached=before(x,y,h),fresh=after(x,y,h),restored=after(cached.world_x,cached.world_y,cached.world_z*112.f);
        assert(std::abs(fresh.x-restored.x)<1.f/256 && std::abs(fresh.y-restored.y)<1.f/256 && std::abs(fresh.z-restored.z)<1.f/256);
    }
}
'''
        with tempfile.TemporaryDirectory(prefix="c3x-zoom-mesh-") as directory:
            cpp = Path(directory) / "test.cpp"
            cpp.write_text(program)
            binary = Path(directory) / "test"
            compiled = subprocess.run([compiler, "-std=c++17", "-O2", "-I", str(ROOT), str(cpp), "-o", str(binary)], capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    unittest.main()
