"""Execute production indexing and projection against independent witnesses."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ZoomMeshTests(unittest.TestCase):
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
        program = r'''
#include <cassert>
#include <cstring>
#include <vector>
#include "Renderer/lab/shared/natural/vertex.h"
using Vertex=c3x_renderer::fidelity::MapVertex;
using UINT=unsigned;
int main(){
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
    for(int original:{64,80,96,112,128})for(int next:{64,80,96,112,128})
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
