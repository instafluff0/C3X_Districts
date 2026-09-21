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
 bool memory_pressured=false;ULONGLONG last_memory_control=0;
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
}
''')
