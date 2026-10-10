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


    def test_shared_components_outlive_their_resident_dependents(self):
        # Evicting a region's shared natural component first left every
        # resident tile drawing with it invalid; block crossings then rebuilt
        # them (performance review, section 55).
        run_cpp(r'''
#include "Renderer/native/render_core/residency_candidates.h"
#include <map>
#include <cassert>
using namespace c3x_renderer::render_core;
struct Content {std::uint64_t last_used;ContentHandle binding,natural_content;};
int main(){
 std::map<int,Content> cache;ResidentContent<Content> owner(2048);ResidencyCandidates candidates;
 auto add=[&](int key,unsigned used,int component=-1){auto& value=cache[key];value={used,{},{}};
  value.binding=owner.bind(value);if(component>=0)value.natural_content=cache[component].binding;};
 auto priority=[](auto const&){return false;};
 auto dependency=[](Content const& value){return value.natural_content;};
 auto evict=[&](std::uint64_t now){auto handle=candidates.next(cache,owner,now,priority,dependency);
  for(auto it=cache.begin();it!=cache.end();++it)if(it->second.binding==handle){int key=it->first;owner.release(handle);cache.erase(it);return key;}
  return -1;};
 // Loaded together: component 0 with tiles 1 and 2, all equally old.
 add(0,5);add(1,5,0);add(2,5,0);
 // A plain tile used more recently than the region.
 add(3,7);
 assert(evict(10)==1&&evict(10)==2);  // dependents first,
 assert(evict(10)==0&&evict(10)==3);  // then their component, then newer tiles.
 // A component whose dependent was used recently ranks as recent: the
 // older plain tile goes first, and the component still follows its dependent.
 candidates.clear();add(10,2);add(11,9,10);add(12,4);
 assert(evict(20)==12&&evict(20)==11&&evict(20)==10);
 // A dependent in the current frame pins its component.
 candidates.clear();add(20,3);add(21,30,20);add(22,8);
 assert(evict(30)==22&&evict(30)==-1);
}
''')


if __name__ == '__main__':
    unittest.main()
