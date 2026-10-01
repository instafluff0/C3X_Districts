"""Immutable generic payload preparation, failed keys and display-tick deduplication."""
import unittest
from Renderer.native.native_cpp_test import run_cpp

class UnitAssetContentTests(unittest.TestCase):
    def test_decoded_budget_rejects_before_any_count_sized_allocation(self):
        from Renderer.lab.platform import ROOT
        fixture=(ROOT/'Renderer/native/test_animation_runtime.cpp').read_text().split('void append_u32(',1)[1].split('\nint main(',1)[0]
        run_cpp(r'''
#include "Renderer/native/unit_asset_content.h"
#include <cassert>
#include <cstdlib>
#include <new>
unsigned allocations=0;
void* operator new(std::size_t bytes){++allocations;if(auto p=std::malloc(bytes))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
void append_u32('''+fixture+r'''
int main(){using namespace c3x_renderer;
 auto data=fixture();AnimationMesh output;
 auto bytes=sizeof(AnimationMesh)+2*sizeof(void*)+3*sizeof(AnimationVertex)+3*sizeof(std::uint32_t)+32*sizeof(float);
 unsigned before=allocations;assert(!decode_animation_mesh(data,output,bytes-1));
 assert(allocations==before&&output.vertices.empty()&&output.palettes.empty());
 assert(decode_animation_mesh(data,output,bytes)&&allocations>before);
 assert(UnitAssetContent::mesh_bytes(output)==bytes);
}
''')

    def test_production_eviction_preserves_the_whole_frame_and_borrowed_leases(self):
        from Renderer.lab.platform import ROOT
        from Renderer.native.test_fresh_preparation_cancellation import block_at
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        body=block_at(source,source.index('    bool reserve_frame_unit_asset('))
        run_cpp(r'''
#include <vector>
#include <memory>
#include <cstdint>
#include <climits>
#include <cassert>
struct Owner {
 struct Mesh {std::size_t bytes=0;std::uint64_t used=0;int* indices=nullptr;std::shared_ptr<int> animation;};
 struct Texture {std::size_t bytes=0;std::uint64_t used=0;int* view=nullptr;std::vector<std::uint8_t> dds;};
 struct Bodies {std::vector<Mesh> meshes;std::vector<Texture> textures;std::size_t resident_bytes=0;
   void release(int*& value){value=nullptr;}}unit_bodies;
 std::vector<bool> frame_mesh_leases,frame_texture_leases;std::uint64_t unit_asset_evictions=0;
'''+body+r'''
};
int main(){
 constexpr std::size_t mib=1024u*1024u;Owner owner;
 owner.unit_bodies.meshes={{48*mib,1,nullptr,std::make_shared<int>(1)},
                          {32*mib,0,nullptr,std::make_shared<int>(2)}};
 owner.unit_bodies.textures={{16*mib,3,nullptr,{1,2,3}}};owner.unit_bodies.resident_bytes=96*mib;
 owner.frame_mesh_leases={true,false};owner.frame_texture_leases={false};
 auto frame_member=owner.unit_bodies.meshes[0].animation;auto borrowed=owner.unit_bodies.meshes[1].animation;
 assert(owner.reserve_frame_unit_asset(16*mib));assert(owner.unit_asset_evictions==1);
 assert(owner.unit_bodies.meshes[0].animation==frame_member&&owner.unit_bodies.meshes[1].animation==borrowed);
 assert(!owner.unit_bodies.textures[0].bytes&&owner.unit_bodies.textures[0].dds.empty());
 assert(!owner.reserve_frame_unit_asset(32*mib)); // never invalidate the union to squeeze in a later member
 assert(owner.unit_bodies.resident_bytes==80*mib);
 borrowed.reset();assert(owner.reserve_frame_unit_asset(32*mib));assert(!owner.unit_bodies.meshes[1].animation);
 assert(owner.unit_bodies.meshes[0].animation==frame_member&&owner.unit_asset_evictions==2);
 assert(!owner.reserve_frame_unit_asset(97*mib));
}
''')

    def test_worker_deduplication_failure_and_retirement(self):
        run_cpp(r'''
#include "Renderer/native/unit_asset_content.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 UnitAssetPreparation pool;std::atomic<unsigned> reads{0};std::atomic<bool> entered{false},release{false};
 auto read=[&](char const* name,std::vector<std::uint8_t>& bytes){
  ++reads;if(std::string(name)=="held"){entered=true;while(!release)std::this_thread::yield();}
  if(std::string(name)=="missing")return false;
  bytes.assign(156,0);std::memcpy(bytes.data(),"DDS ",4);std::memcpy(bytes.data()+84,"DX10",4);return true;
 };
 auto compile=[&](auto const& input,auto const& cancel,unsigned){return compile_unit_asset(input,cancel,read);};
 pool.schedule({{1,{"held"}}},compile,4,{1},64u*1024u*1024u,true);
 while(!entered)std::this_thread::yield();
 // Independent display ticks must not cancel or restart the active read.
 for(unsigned tick=0;tick<1000;++tick)pool.schedule({{1,{"held"}}},compile,4,{1},64u*1024u*1024u,true);
 assert(reads==1);release=true;
 auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(2);
 std::unique_ptr<UnitAssetContent> ready;
 while(!ready&&std::chrono::steady_clock::now()<deadline)ready=pool.take_ready(1);
 assert(ready&&!ready->failed&&ready->dds.size()==156);
 pool.schedule({{2,{"missing"}},{3,{"invalid-mesh",true}}},compile,4,{2,3},64u*1024u*1024u,true);
 auto missing=pool.take(2),invalid=pool.take(3);assert(missing&&missing->failed&&invalid&&invalid->failed);
 pool.clear();auto stats=pool.statistics();assert(!stats.active&&!stats.pending&&!stats.bytes&&stats.built==3&&stats.rejected==0);
 std::atomic<bool> cancelled{true};assert(!compile_unit_asset({"unused"},cancelled,read));assert(reads==3);
}
''')
