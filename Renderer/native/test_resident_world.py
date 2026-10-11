"""The resident world store: the explored world's draw records by world block.

Camera decoupling stage 6 (whole-world restructure, increment 1). Queries must
return the window's draw order (layer, occurrence row, column, position in the
tile, wrap turn), cull conservatively, and keep published generations immutable
while edits copy only the blocks they touch.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ResidentWorldTests(unittest.TestCase):
    def test_order_culling_wrap_and_generations(self):
        run_cpp(r'''
#include "Renderer/native/render_core/resident_world.h"
#include <cassert>
#include <cstdio>
#include <random>
#include <set>
#include <tuple>
using namespace c3x_renderer::render_core;
struct Record {int id=0,tile_x=0,tile_y=0;};
constexpr unsigned Layers=4;
using World=ResidentWorld<Record,Layers>;
using Extent=World::Extent;
int main(){
 std::mt19937 rng(7);
 World world;
 // Reference: per layer, records sorted by (row, column) then append order.
 struct Ref {int y,x,i,id;Extent e;};
 std::array<std::vector<Ref>,Layers> reference;
 int next_id=0;std::set<std::pair<int,int>> placed;
 for(int n=0;n<900;++n){
  int x=int(rng()%80)-20,y=int(rng()%60)-10;if((x+y)&1)++x;
  if(!placed.insert({x,y}).second)continue;
  World::Tile tile;tile.tile_x=x;tile.tile_y=y;
  for(unsigned layer=0;layer<Layers;++layer){unsigned count=rng()%4;
   for(unsigned i=0;i<count;++i){Record r{next_id++,x,y};
    double px=x*64.,py=y*32.;Extent e{px-40-double(rng()%30),py-30,px+40+double(rng()%30),py+30+double(rng()%40),
     float(rng()%100)/100.f,float(rng()%100)/100.f+1.f};
    tile.records[layer].push_back(r);tile.extents[layer].push_back(e);
    reference[layer].push_back({y,x,int(i),r.id,e});}}
  world.set(std::move(tile));
 }
 for(auto& layer:reference)std::sort(layer.begin(),layer.end(),[](Ref const& a,Ref const& b){
  return std::make_tuple(a.y,a.x,a.i)<std::make_tuple(b.y,b.x,b.i);});
 auto const& g=world.view();
 // 1. Every record, in the window's order, per layer.
 for(unsigned layer=0;layer<Layers;++layer){std::vector<int> got;
  World::each(g,layer,[&](Record const& r){got.push_back(r.id);});
  assert(got.size()==reference[layer].size());
  for(std::size_t i=0;i<got.size();++i)assert(got[i]==reference[layer][i].id);}
 // 2. Queries equal a brute-force exact selection, in order (no wrap).
 auto none=[](float){return 0.;};
 auto mirror=[](float z){return 2*30.*std::max(0.f,z-2.5f/112.f);};
 for(int q=0;q<300;++q){
  double l=double(int(rng()%6000))-1500,t=double(int(rng()%2400))-500;double r=l+1+rng()%1500,b=t+1+rng()%900;
  bool mirrored=q&1;
  std::vector<std::pair<unsigned,int>> got,want;
  World::query(g,l,t,r,b,0,0,[&](float z){return mirrored?mirror(z):none(z);},[&](unsigned layer,Record const& rec,unsigned turn,int shift){
   assert(turn==0&&shift==0);got.push_back({layer,rec.id});});
  for(unsigned layer=0;layer<Layers;++layer)for(auto const& ref:reference[layer]){
   double lo=mirrored?mirror(ref.e.low_z):0,hi=mirrored?mirror(ref.e.high_z):0;
   if(!(ref.e.right<=l||ref.e.left>=r||ref.e.bottom+hi<=t||ref.e.top+lo>=b))want.push_back({layer,ref.id});}
  assert(got==want);
 }
 // 3. Wrap turns: copies at -/+wrap in effective column order, turn last.
 {World w;int wrap_tiles=20,wrap=wrap_tiles*64;
  for(int x:{0,2,18}){World::Tile t;t.tile_x=x;t.tile_y=0;t.records[0].push_back({x,x,0});
   t.extents[0].push_back({x*64.-32,-16,x*64.+32,16,0,0});w.set(std::move(t));}
  // A view across the seam: columns 16..22 in world pixels 1000..1440.
  std::vector<std::tuple<int,unsigned,int>> got;
  World::query(w.view(),1000,-10,1440,10,wrap,wrap_tiles,none,[&](unsigned,Record const& r,unsigned turn,int shift){got.push_back({r.id,turn,shift});});
  // Column 18 as stored, then columns 0 and 2 wrapped forward (effective 20, 22).
  assert(got.size()==3);
  assert(got[0]==std::make_tuple(18,0u,0));
  assert(got[1]==std::make_tuple(0,2u,wrap)&&got[2]==std::make_tuple(2,2u,wrap));}
 // 4. Generations: a published view is immutable; edits copy only their block.
 {auto before=world.publish();auto revision=world.revision();
  auto first=before->blocks.begin();auto untouched=std::next(first)->second;
  World::Tile t=*first->second->tiles.front();int tx=t.tile_x,ty=t.tile_y;
  auto original=t.records[0].size()+1;t.records[0].push_back({-1,tx,ty});t.extents[0].push_back({});
  world.set(t);t.records[0].clear();t.extents[0].clear();world.set(std::move(t));
  assert(world.revision()!=revision);
  assert(before->tile(tx,ty)->records[0].size()==original-1); // unchanged by both edits
  auto after=world.publish();
  assert(after->blocks.at(std::next(first)->first)==untouched); // shared, not copied
  assert(after->blocks.at(first->first)!=first->second);        // copied
  assert(after->tile(tx,ty)->records[0].empty());
  assert(before->tile(tx,ty)==first->second->tiles.front().get());}
 // 5. Owners live while any generation borrows the tile.
 {World w;auto mesh=std::make_shared<int>(5);std::weak_ptr<int> weak=mesh;
  World::Tile t;t.tile_x=4;t.tile_y=4;t.owners.push_back({ContentHandle{3,9},mesh});mesh.reset();
  w.set(std::move(t));auto lease=w.publish();
  assert(lease->owner(4,4,ContentHandle{3,9})&&!lease->owner(4,4,ContentHandle{3,8}));
  World::Tile replacement;replacement.tile_x=4;replacement.tile_y=4;w.set(std::move(replacement));
  assert(!weak.expired());lease.reset();assert(weak.expired());
  // Erasing the last tile drops its block.
  assert(w.erase(4,4)&&w.view().blocks.empty()&&w.view().tiles==0&&!w.erase(4,4));}
 std::puts("PASS resident world: order, culling, wrap, copy-on-write, owners");
}
''')


if __name__ == '__main__':
    unittest.main()
