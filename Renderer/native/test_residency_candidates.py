"""Eviction preserves active pins and owner generations during streaming."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ResidencyCandidatesTests(unittest.TestCase):
    def test_order_pins_stale_bindings_and_linear_selection(self):
        run_cpp(r'''
#include "Renderer/native/render_core/residency_candidates.h"
#include <map>
#include <cassert>
using namespace c3x_renderer::render_core;
struct Content {std::uint64_t last_used;bool animated;ContentHandle binding;};
int main(){
 std::map<int,Content> cache;ResidentContent<Content> owner(2048);ResidencyCandidates candidates;
 auto add=[&](int key,unsigned used,bool animated){auto& value=cache[key];value={used,animated,{}};value.binding=owner.bind(value);};
 auto priority=[](auto const& value){return value.animated;};
 add(0,8,false);add(1,1,true);add(2,3,false);add(3,10,false);
 assert(candidates.next(cache,owner,10,priority)==cache[2].binding);
 // An entry becoming current after sorting must remain pinned.
 cache[0].last_used=10;
 assert(candidates.next(cache,owner,10,priority)==cache[1].binding);
 owner.release(cache[1].binding);cache.erase(1);
 owner.release(cache[2].binding);cache.erase(2);
 assert(!candidates.next(cache,owner,10,priority).generation);
 auto old=cache[0].binding;owner.release(old);cache.erase(0);add(0,10,false);
 assert(!owner.resolve(old));assert(!candidates.next(cache,owner,10,priority).generation);
 assert(candidates.next(cache,owner,11,priority).generation);
 candidates.clear();owner.clear();cache.clear();
 for(int i=0;i<1000;++i)add(i,unsigned(i),false);
 auto rebuilds=candidates.rebuilds,examined=candidates.examined;
 for(int i=0;i<1000;++i){auto handle=candidates.next(cache,owner,2000,priority);
  assert(handle==cache[i].binding);owner.release(handle);cache.erase(i);}
 assert(candidates.rebuilds==rebuilds+1 && candidates.examined==examined+1000);
 assert(candidates.bytes()<128u*1024u);
}
''')


if __name__ == '__main__':
    unittest.main()
