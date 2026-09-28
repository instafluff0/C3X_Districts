"""The promoted Lab library replaces every runtime city/wall combination."""
from pathlib import Path
import os
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT = Path(__file__).resolve().parents[2]


class CityRecipeTests(unittest.TestCase):
    def test_complete_library_selection_and_transactional_decode(self):
        pack = ROOT / os.environ.get('C3X_TEST_CITY_PACK', 'Renderer/packs/CityCompositionRuntime/city.bin')
        self.assertTrue(pack.is_file(), 'Compile the selected city recipes first')
        code = r'''
#include "Renderer/native/city_fidelity/compiler.h"
#include <cassert>
#include <fstream>
#include <iterator>
#include <iostream>
using namespace c3x_renderer;
int main(){
 std::ifstream stream(R"PATH(CITY_PATH)PATH",std::ios::binary);
 std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(stream)),{});
 city_fidelity::Library lib;assert(lib.decode(bytes));assert(lib.complete_city_set());
 unsigned variants=0;for(auto const& c:lib.compositions)variants=std::max(variants,c.variant+1);
 assert(lib.compositions.size()==160*variants);
 struct Land{int base=13,real=6;};struct Shore{double distance=-1;};
 auto select=[&](c3x_renderer_tile_v1 const& tile){return city_fidelity::select(lib,tile,20,40,
   [](int,int){return Land{};},[](float,float){return Shore{};},[](float,float){return 0.f;},
   [](float,float){return 2.5f;});};
 c3x_renderer_tile_v1 tile{};tile.city_id=8;
 for(int culture=0;culture<5;++culture)for(int era=0;era<4;++era)
 for(int size=0;size<3;++size)for(unsigned flags=0;flags<4;++flags)
 for(unsigned seed:{0u,1u,2u,777u,0xffffffffu}){
  tile.city_culture_group=culture;tile.city_era=era;tile.city_size=size;
  tile.city_flags=(flags&1?C3X_RENDERER_CITY_CAPITAL:0)|(flags&2?C3X_RENDERER_CITY_WALLED:0);
  tile.variant_seed=seed;auto c=select(tile);assert(c);
  assert(c->culture==unsigned(culture)&&c->era==unsigned(era)&&c->size==unsigned(size));
  assert(c->variant==seed%variants&&c->capital==unsigned((flags&1)!=0));
  assert(c->walled==unsigned(size==0&&(flags&2))&&c->owns_walls&&c->anchor_layout);
  assert(c->authority.find("generic-source-growth")==std::string::npos);
  tile.city_owner_id=7;tile.city_population=9;assert(select(tile)==c);
  tile.city_owner_id=2;tile.city_population=10;assert(select(tile)==c);
  unsigned capitals=0;for(auto const& i:c->instances)capitals+=i.capital;
  assert(capitals==c->capital);
 }
 tile.city_id=-1;assert(!select(tile));
 // The one immutable library may be read in any order; wrapping changes only placement.
 for(auto const& c:lib.compositions)for(auto const& i:c.instances){
  auto a=city_fidelity::place(i,20,40,2.5f),b=city_fidelity::place(i,37,33,2.5f);
  float source[3]={.13f,.27f,.4f},pa[3],pb[3];a.position(source,pa);b.position(source,pb);
  assert(std::abs(pb[0]-pa[0]-17)<1e-5f&&std::abs(pb[1]-pa[1]+7)<1e-5f&&pa[2]==pb[2]);
 }
 auto last=lib.compositions.back();lib.compositions.pop_back();assert(!lib.complete_city_set());
 lib.compositions.push_back(last);lib.compositions.push_back(last);assert(!lib.complete_city_set());
 lib.compositions.pop_back();assert(lib.complete_city_set());
 auto size=lib.byte_count;
 for(std::size_t n:{std::size_t(0),std::size_t(20),bytes.size()/2,bytes.size()-1}){
  std::vector<std::uint8_t> truncated(bytes.begin(),bytes.begin()+n);
  assert(!lib.decode(truncated)&&lib.byte_count==size&&lib.complete_city_set());
 }
 bytes.push_back(0);assert(!lib.decode(bytes));
 std::cout<<"PASS complete city recipes: "<<lib.compositions.size()<<" states, stable seeds, capture/growth, walls, coastal anchors, wrapping, malformed packs\n";
}
'''.replace('CITY_PATH', pack.as_posix())
        run_cpp(code, timeout=90)


if __name__ == '__main__':
    unittest.main()
