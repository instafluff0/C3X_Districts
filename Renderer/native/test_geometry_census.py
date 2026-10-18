"""The stage 5 geometry census reports every layer within the trace line limit.

A trace line holds 1,024 bytes including its prefix (renderer_trace.h). With
exact strides the census outgrew one line and lost the natural layers after
layer 21; it now continues on geometry-census-more lines.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


class GeometryCensusTests(unittest.TestCase):
    def test_every_layer_reaches_the_trace(self):
        source = (ROOT / 'Renderer/native/c3x_renderer.cpp').read_text()
        method = source[source.index('    void geometry_census(){'):source.index('    std::size_t world_owner_gpu_bytes(')]
        prefix = len('[C3X renderer] qpc=113789049133 ms=4741210.381 process=1504 thread=1880 '
                     'sequence=39 stage=geometry-census-more \n')
        run_cpp(r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
enum DXGI_FORMAT {DXGI_FORMAT_R16_UINT=57,DXGI_FORMAT_R32_UINT=42};
struct D3D11_BUFFER_DESC {unsigned ByteWidth=0;};
struct ID3D11Buffer {unsigned size;void GetDesc(D3D11_BUFFER_DESC* d){d->ByteWidth=size;}};
constexpr unsigned geometry_layer_count=40;
struct ResourceAnchor {int x=0,y=0;};
struct Lighting {std::vector<std::array<float,8>> lights;std::vector<std::array<float,6>> blockers;};
struct Chunk {ID3D11Buffer *buffer=nullptr,*indices=nullptr;unsigned vertex_count=0,vertex_stride=0,index_count=0,byte_count=1;
 DXGI_FORMAT index_format=DXGI_FORMAT_R32_UINT;bool rigid_source=false;std::shared_ptr<std::vector<int>> instances;
 std::shared_ptr<Lighting> city_lighting;};
struct Proof {std::vector<std::pair<std::uint64_t,std::uint64_t>> appearance_dependencies,dependencies,coast_dependencies;
 std::vector<std::pair<std::size_t,std::uint32_t>> world_dependencies;std::vector<std::uint64_t> river_dependencies;
 std::vector<std::pair<std::uint64_t,std::array<int,2>>> anchor_dependencies;};
struct Mesh {std::array<std::vector<Chunk>,geometry_layer_count> layers;std::shared_ptr<Proof> proof=std::make_shared<Proof>();};
struct Owner {bool shared_natural=false;std::shared_ptr<Mesh> mesh;std::vector<ResourceAnchor> resource_anchors;};
using CachedVertexChunk=Chunk;using CachedMeshGeneration=Mesh;using CachedTileGeometry=Owner;
struct State {
 std::unordered_map<unsigned,Owner> tile_geometry_cache;std::map<unsigned,ID3D11Buffer*> terrain_patch_indices;
 std::size_t tile_geometry_cache_bytes=1u<<30,terrain_patch_index_bytes=1u<<20,tile_geometry_runtime_budget=1u<<31;
 std::size_t canonical_geometry_bytes()const{return 1u<<29;} // the fixed tier's canonical content (review 67)
 std::size_t texture_file_bytes()const{return 3u<<20;}
 struct {std::vector<std::pair<std::string,std::string>> lines;
  void write(char const* stage,char const* detail,bool){lines.emplace_back(stage,detail);}} trace;
''' + method + r'''};
int main(){
 State s;std::vector<std::unique_ptr<ID3D11Buffer>> buffers;
 for(unsigned owner=0;owner<4;++owner){auto& o=s.tile_geometry_cache[owner];o.mesh=std::make_shared<Mesh>();
  for(unsigned layer=0;layer<geometry_layer_count;++layer)for(unsigned stride:{32u,48u,56u,92u,168u}){
   buffers.push_back(std::make_unique<ID3D11Buffer>(ID3D11Buffer{4000000000u}));
   Chunk c;c.buffer=c.indices=buffers.back().get();c.vertex_count=4000000;c.vertex_stride=stride;c.index_count=9000000;
   o.mesh->layers[layer].push_back(c);}}
 // CPU census: each shared object once. Owner 4 shares owner 0's mesh.
 for(unsigned owner=0;owner<4;++owner)s.tile_geometry_cache[owner].mesh->proof->dependencies.reserve(1024); // 16 KiB each
 s.tile_geometry_cache[4].mesh=s.tile_geometry_cache[0].mesh;
 s.geometry_census();
 assert(s.trace.lines.size()>2 && s.trace.lines[0].first=="geometry-census");
 assert(s.trace.lines.back().first=="geometry-census-cpu");
 assert(s.trace.lines.back().second.find(" dependencies_kb=64 ")!=std::string::npos);
 assert(s.trace.lines.back().second.find("dds_kb=3072 ")==0);
 s.trace.lines.pop_back();
 std::string all;
 for(unsigned i=0;i<s.trace.lines.size();++i){
  assert(s.trace.lines[i].first==(i?"geometry-census-more":"geometry-census"));
  assert(s.trace.lines[i].second.size()+''' + str(prefix) + r'''<1024);
  all+=s.trace.lines[i].second;}
 for(unsigned layer=0;layer<geometry_layer_count;++layer)
  assert(all.find(" L"+std::to_string(layer)+"=")!=std::string::npos);
 assert(all.find("32/48/56/92")!=std::string::npos);
}
''')


if __name__ == '__main__':
    unittest.main()
