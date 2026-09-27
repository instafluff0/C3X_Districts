"""Repeated sprite content reuses uploads; mutation and eviction remain ordered."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class NativeSpriteCacheTests(unittest.TestCase):
    def test_interleaved_content_dimensions_mutation_and_eviction(self):
        run_cpp(r'''
#include "Renderer/native/native_sprite_cache.h"
#include <cassert>
#include <map>
using namespace c3x_gpu_images;
struct GPU {
 Id next=0;unsigned uploaded=0;
 std::map<Id,std::vector<unsigned>> live;
 Id create(unsigned w,unsigned h,Format f){assert(f==Format::bgra32);live[++next].resize(w*h);return next;}
 bool upload(Id id,std::uint64_t version,unsigned const* p,std::size_t n){assert(version==1&&live.at(id).size()==n);live[id].assign(p,p+n);++uploaded;return true;}
 bool destroy(Id id){assert(live.erase(id)==1);return true;}
} gpu;
int main(){
 c3x_native_images::SpriteCache<GPU> cache;
 Id ids[32]={};
 for(unsigned n=0;n<10400;++n){
  unsigned variant=n%32;std::vector<unsigned> source(128*64,65536|variant);
  auto id=cache.select(gpu,source,128,64);assert(id);
  if(n<32)ids[variant]=id;else assert(id==ids[variant]);
  assert(gpu.live.at(id)[17]==(65536|variant));
 }
 assert(gpu.uploaded==32&&cache.hits==10368&&cache.bytes()==32*128*64*4);
 std::vector<unsigned> words(128*64,65536);words[411]^=1;
 auto changed=cache.select(gpu,words,128,64);assert(changed!=ids[0]);
 words.assign(128*64,65536);assert(cache.select(gpu,words,64,128)!=ids[0]);
 cache.clear(gpu);assert(gpu.live.empty()&&cache.bytes()==0);
 // Force LRU eviction and then reuse a previously evicted payload. A retained
 // native pointer and equal dimensions never grant stale-content reuse.
 c3x_native_images::SpriteCache<GPU,2> small;
 for(unsigned n=0;n<80;++n){words.assign(64,n%3);auto id=small.select(gpu,words,8,8);
  assert(gpu.live.at(id).front()==n%3&&gpu.live.size()<=2);}
 small.clear(gpu);assert(gpu.live.empty());
 // The byte cap also applies when fewer than Capacity images are resident.
 for(unsigned n=0;n<8;++n){words.assign(1024*1024,n);auto id=cache.select(gpu,words,1024,1024);
  assert(id&&cache.bytes()<=16u*1024u*1024u&&gpu.live.size()<=4);}
 cache.clear(gpu);assert(gpu.live.empty());
}
''')


if __name__ == '__main__':
    unittest.main()
