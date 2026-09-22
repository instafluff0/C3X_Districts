"""Whole-process pressure evicts reusable resources without retiring authority."""
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
import unittest

class ProcessHeadroomTests(unittest.TestCase):
    def test_hysteresis_and_resource_release(self):
        source=(ROOT/'Renderer/native/c3x_renderer.cpp').read_text()
        method='void preserve_process_headroom(){'+source.split('void preserve_process_headroom(){',1)[1].split('    ULONGLONG last_memory_status',1)[0]
        run_cpp(r'''
#include "Renderer/native/render_core/render_region_cache.h"
#include "Renderer/native/render_core/frame_working_set.h"
#include <cassert>
#include <cstdio>
#define sprintf_s std::snprintf
// Adapt only the CRT buffer-size overload, not the production pressure decision.
#undef sprintf_s
template<size_t N,class... T> void sprintf_s(char(&s)[N],char const* f,T...args){std::snprintf(s,N,f,args...);}
using ULONGLONG=unsigned long long;
ULONGLONG now=1000,available=2ull*1024*1024*1024;
ULONGLONG GetTickCount64(){return now;}
struct MEMORYSTATUSEX {unsigned dwLength=0;ULONGLONG ullAvailVirtual=0;};
bool GlobalMemoryStatusEx(MEMORYSTATUSEX* p){p->ullAvailVirtual=available;return true;}
int retired=0;struct Resource{void Release(){++retired;delete this;}};
struct State {
 bool memory_pressured=false,shared_scene_surface=false;ULONGLONG last_memory_control=0,last_unit_work=0;
 size_t attachment_bytes=0,composition_bytes=0;
 size_t composition_working_bytes()const{return composition_bytes;}
 struct {size_t n=0;size_t bytes()const{return n;}void reset(){n=0;}}unit_scene_work;
 size_t frame_working_bytes()const{return attachment_bytes+unit_scene_work.bytes();}
 c3x_renderer::render_core::RenderRegionCache<Resource> render_regions;
 struct {size_t gpu_content_bytes=192u*1024*1024,gpu_content_limit=192u*1024*1024;
 void set_gpu_content_limit(size_t n){gpu_content_limit=n;gpu_content_bytes=std::min(n,gpu_content_bytes);}}unit_bodies;
 struct {void write(char const*,char const*,bool){}}trace;
'''+method+r'''
};
int main(){
 State s;constexpr size_t mib=1024*1024;
 for(unsigned i=0;i<4;++i)assert(s.render_regions.insert({i+1},new Resource,64*mib));
 s.preserve_process_headroom();assert(!s.memory_pressured&&!retired);
 available=700*mib;now+=250;s.preserve_process_headroom();assert(s.memory_pressured&&retired==3);
 assert(s.render_regions.gpu_bytes==64*mib&&s.unit_bodies.gpu_content_limit==48*mib);
 available=900*mib;now+=250;s.preserve_process_headroom();assert(s.memory_pressured);
 available=1100*mib;now+=250;s.preserve_process_headroom();assert(!s.memory_pressured&&s.render_regions.gpu_limit==256*mib);
 assert(s.render_regions.gpu_bytes==64*mib); // restoring capacity never recreates evicted images
 auto resource=s.render_regions.find({4});assert(resource);s.render_regions.clear();assert(retired==4);
 s.shared_scene_surface=true;s.attachment_bytes=900*mib;now+=250;s.preserve_process_headroom();
 assert(s.render_regions.gpu_limit==0&&s.unit_bodies.gpu_content_limit==124*mib);
 s.unit_scene_work.n=50*mib;s.last_unit_work=now;now+=250;s.preserve_process_headroom();
 assert(s.unit_bodies.gpu_content_limit==74*mib&&s.unit_scene_work.bytes()==50*mib);
 now+=1000;s.preserve_process_headroom();assert(!s.unit_scene_work.bytes()&&s.unit_bodies.gpu_content_limit==124*mib);
 // Simultaneously held composition/front allocations compete with optional
 // caches; releasing a front restores capacity without recreating content.
 s.composition_bytes=100*mib;now+=250;s.preserve_process_headroom();assert(s.unit_bodies.gpu_content_limit==24*mib);
 s.composition_bytes=200*mib;now+=250;s.preserve_process_headroom();assert(!s.unit_bodies.gpu_content_limit);
 s.composition_bytes=0;now+=250;s.preserve_process_headroom();assert(s.unit_bodies.gpu_content_limit==124*mib);
 using Budget=c3x_renderer::render_core::FrameWorkingSet;
 assert(Budget::scene(2240,1260)==684111360&&Budget::mirror(2240,1260)==92116992);
 for(size_t bytes:{0u,100u,900u,1024u,1200u})for(bool pressure:{false,true})for(bool shared:{false,true}){
   auto limits=Budget::caches(bytes*mib,pressure,shared);
   assert(limits.regions+limits.units<= (bytes<1024?(1024-bytes)*mib:0));
   if(shared)assert(!limits.regions);
 }
}
''')

    def test_unit_cache_obeys_the_shared_remaining_budget(self):
        source=(ROOT/'Renderer/native/unit_body_renderer.h').read_text()
        method='    void set_gpu_content_limit('+source.split('    void set_gpu_content_limit(',1)[1].split('    int scene_output_width=',1)[0]
        run_cpp(r'''#include <vector>
#include <algorithm>
#include <cassert>
struct Cache {
 struct Entry {size_t used,bytes;};std::vector<Entry> gpu_content;
 size_t gpu_content_limit=0,gpu_content_bytes=0;
'''+method+r'''};
int main(){
 Cache c;c.gpu_content={{2,20},{1,40}};c.gpu_content_bytes=60;
 c.set_gpu_content_limit(25);assert(c.gpu_content.size()==1&&c.gpu_content[0].used==2&&c.gpu_content_bytes==20);
 c.set_gpu_content_limit(0);assert(c.gpu_content.empty()&&!c.gpu_content_bytes&&!c.gpu_content_limit);
 c.set_gpu_content_limit(192u*1024u*1024u);assert(c.gpu_content.empty());
}''')
