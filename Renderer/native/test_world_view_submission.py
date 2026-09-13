"""World validity and spatial selection use production owners, without a GPU."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class WorldViewSubmissionTests(unittest.TestCase):
    def test_river_cell_proofs_preserve_exact_values_and_local_validity(self):
        run_cpp(r'''
#include "Renderer/lab/shared/natural/world.h"
#include <cassert>
using namespace c3x_renderer;
using fidelity::NaturalWorld;
int main(){
 render_core::WorldTopology topology;
 std::vector<std::uint32_t> bits(2048,2|(2<<8));
 auto update=[&]{topology.update({64,64,true,true},bits.data(),bits.size());};update();
 NaturalWorld world;world.fields.resize(1);world.fields[0].width=world.fields[0].height=2;
 world.fields[0].pixels={0,64,128,255};world.update_rivers(topology,1);
 NaturalWorld::CellInputs inputs;
 {
  NaturalWorld::DependencyScope scope(world,&inputs);
  auto held=world.bind_river_page(12.5,4.5);
  auto a=held.sample({12.5,4.5});assert(a.distance==1000);
  hydro::P bank;world.river_affects(12,4);world.river_page(12.5,4.5).bank_point({12.5,4.5},1,1,bank);
  assert(inputs.size()==1); // Repeated consumers share an exact empty-cell proof.
  // Holding a page remains safe across the bounded page cache's eviction.
  for(int i=0;i<30;++i)world.river_sample({i*8+.5,4.5});
  auto b=held.sample({12.5,4.5});assert(a.distance==b.distance && a.source==b.source && a.mouth==b.mouth);
 }
 assert(!world.consumer && world.river_pages.size()==16);
 NaturalWorld::CellProof proof;
 auto key=NaturalWorld::CellKey{1,0,12,4};proof.push_back({key,inputs.at(key)});
 auto epoch=world.river_epoch;assert(world.valid(proof));assert(world.river_epoch==epoch);
 // A revision alone is not a changed river query result, even in this page.
 bits[topology.index(10,3)]=2|(5<<8);update();world.update_rivers(topology,2);
 assert(world.valid(proof));
 // Add a local river: absence is observed and outer compiled reuse is rejected.
 bits[topology.index(12,4)]|=10u<<16;update();world.update_rivers(topology,3);
 assert(!world.valid(proof));
 NaturalWorld::CellInputs changed;
 {NaturalWorld::DependencyScope scope(world,&changed);world.river_sample({12.5,4.5});}
 NaturalWorld::CellProof current(changed.begin(),changed.end());assert(world.valid(current));
 // Exact values survive independent page construction and wrapped lookup.
 NaturalWorld cold;cold.fields=world.fields;cold.update_rivers(topology,3);assert(cold.valid(current));
 for(auto const& p:world.river_pages){assert(p.cells->inputs->values.size()<=4096);for(auto const& input:p.cells->inputs->values)assert(topology.at(input.first)==input.second);}
 bits[topology.index(12,4)]&=~(255u<<16);update();world.update_rivers(topology,4);
 assert(world.valid(proof) && !world.valid(current));
 world.reset_world();assert(world.river_pages.empty() && !world.valid(proof));
}
''')

    def test_shared_patch_topology_and_explicit_detail_policy(self):
        run_cpp(r'''
#include "Renderer/lab/shared/natural/ground.h"
#include <cassert>
#include <cstring>
using namespace c3x_renderer::fidelity;
int main(){
 PatchLayouts layouts;
 for(unsigned n:{8u,16u,32u,48u,64u}){
  auto const& layout=layouts.get(n);assert(&layout==&layouts.get(n));
  assert(layout.corners.size()==(n+1)*(n+1) && layout.indices.size()==6*n*n);
  std::vector<MapVertex> grid((n+1)*(n+1));
  for(unsigned i=0;i<grid.size();++i){grid[i].x=float(i);grid[i].base_terrain=-9;}
  for(bool partial:{false,true}){
   if(partial)for(unsigned i=0;i<grid.size()/2;++i)grid[i].base_terrain=-10;
   std::vector<MapVertex> a,b;std::vector<unsigned> ia,ib;
   append_surface_grid(a,grid,n,true,&ia);
   append_surface_grid(b,grid,n,true,&ib,&layout);
   assert(ia==ib && a.size()==b.size());assert(!std::memcmp(a.data(),b.data(),a.size()*sizeof(MapVertex)));
  }
 }
 assert(PatchDetail(128,0)==PatchDetail());
 assert(PatchDetail(128,3).mountain==32 && PatchDetail(192,3).mountain==64);
 assert(PatchDetail(128,5).mountain==16);
 bool failed=false;try{layouts.get(65);}catch(std::length_error const&){failed=true;}assert(failed);
}
''')

    def test_rectangular_pass_selection_preserves_order_and_all_overhangs(self):
        run_cpp(r'''
#include "Renderer/native/render_core/region_contributor_index.h"
#include <cassert>
using Index=c3x_renderer::render_core::RegionContributorIndex;
int main(){
 Index index;std::vector<std::array<int,4>> bounds;
 for(unsigned i=0;i<600;++i){int x=int(i%40)*64-500,y=int(i/40)*80-300;
  bounds.push_back({x,y,x+180,y+240});assert(index.add(0,{i%8,i},x,y,x+180,y+240));}
 index.ready=true;
 for(int x:{-540,-1,100,700})for(int y:{-340,0,450})for(int w:{1,17,256,2248})for(int h:{1,31,1200}){
  std::vector<Index::Item> selected;assert(index.query_rectangle(0,x,y,w,h,selected));
  assert(std::is_sorted(selected.begin(),selected.end()));
  assert(std::adjacent_find(selected.begin(),selected.end())==selected.end());
  std::vector<Index::Item> exact,brute;
  auto hit=[&](unsigned i){auto b=bounds[i];return b[0]<x+w && b[2]>x && b[1]<y+h && b[3]>y;};
  for(auto item:selected)if(hit(item.second))exact.push_back(item);
  for(unsigned i=0;i<bounds.size();++i)if(hit(i))brute.push_back({i%8,i});std::sort(brute.begin(),brute.end());
  assert(exact==brute); // Same material/layer and occurrence order, no duplicates.
 }
 assert(index.bytes<=index.budget);index.clear();std::vector<Index::Item> out;
 assert(!index.query_rectangle(0,0,0,2248,1200,out)); // Caller must use the complete scan.
}
''')


if __name__ == '__main__':
    unittest.main()
