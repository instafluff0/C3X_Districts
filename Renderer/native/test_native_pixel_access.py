"""Civ III's per-pixel loops stay cheap in the bridge.

The Military and Cultural advisors call JGL pixel_3/bits hundreds of thousands
of times while opening. Each call visits the lifetime registry and the native
adapter; a scan of every slot per call stalled those screens for 1-2 s.
"""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def adapter_contract():
    source = (ROOT / 'Renderer/native/native_image_adapter.h').read_text()
    find = source[source.index("    // Civ III's per-pixel loops (Military and Cultural advisors) alternate"):source.index('    void forget(Image& image)')]
    trim = source[source.index('    void trim_cpu_sources(){'):source.index('    Image* create(void* p,bool owned){')]
    create = source[source.index('    Image* create(void* p,bool owned){'):source.index('    bool refresh(Image& image){')]
    start = source.index('    int operation(int op,void* object')
    marker = '        auto destination=find(object);\n'
    operation = source[start:source.index(marker, start) + len(marker)]
    return find, trim, create, operation


class NativePixelAccessTests(unittest.TestCase):
    def test_lifetime_lookups_do_not_scan_every_slot(self):
        run_cpp(r'''
#include "Renderer/native/native_lifetime_registry.h"
#include <cassert>
#include <chrono>
#include <cstdio>
int main(){c3x_native_images::Lifetimes lifetimes;static char images[1025];
 for(int n=0;n<1024;++n)assert(lifetimes.observe(C3X_NATIVE_INIT,&images[n],0,1));
 assert(!lifetimes.observe(C3X_NATIVE_INIT,&images[1024],0,1)); // the registry stays bounded
 assert(!lifetimes.observe(C3X_NATIVE_PIXEL,&images[1023],0,1)); // escaped
 // A private lease keeps the first image; the escaped one is simply unknown.
 auto cost=[&](void* image,int context){double best=1e9;
  for(int round=0;round<5;++round){auto begin=std::chrono::steady_clock::now();
   for(int n=0;n<100000;++n)lifetimes.observe(C3X_NATIVE_BITS,image,context,1);
   best=std::min(best,std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count());}
  return best;};
 double kept=cost(&images[0],C3X_NATIVE_COPY),escaped=cost(&images[1023],0);
 std::printf("kept %.2f ms, escaped %.2f ms per 100k\n",kept*1000,escaped*1000);
 assert(escaped<kept*6);
 // Evidence semantics are unchanged.
 bool revoked=true,early=false;
 assert(lifetimes.observe(C3X_NATIVE_MAP,&images[0],0,1));
 assert(!lifetimes.observe(C3X_NATIVE_PIXEL,&images[0],0,1,&revoked,&early)&&revoked&&!early);
 assert(!lifetimes.observe(C3X_NATIVE_MAP,&images[0],0,1));
 assert(lifetimes.observe(C3X_NATIVE_INIT,&images[1024],0,1)&&lifetimes.observe(C3X_NATIVE_INIT,&images[1023],0,1)); // escapes freed slots
 assert(!lifetimes.observe(C3X_NATIVE_INIT,&images[0],0,1)); // full again
 assert(!lifetimes.observe(C3X_NATIVE_BITS,&images[5],0,2,&revoked,&early)&&!revoked&&early); // another thread
 assert(lifetimes.observe(C3X_NATIVE_INIT,&images[0],0,1));
 assert(!lifetimes.observe(C3X_NATIVE_INIT,&images[6],0,2)); // a non-owner INIT forgets the image
 assert(!lifetimes.observe(C3X_NATIVE_MAP,&images[6],0,1));
 assert(lifetimes.observe(C3X_NATIVE_VERIFY,nullptr,0,3)&&!lifetimes.observe(C3X_NATIVE_MAP,&images[1],0,3));
 std::printf("PASS lifetime lookups\n");
}
''')

    def test_adapter_pixel_access_finds_its_image_without_scans(self):
        find, trim, create, operation = adapter_contract()
        # Only create() can store a native pointer, so it alone ends an absence.
        self.assertIn('recent={};', create)
        run_cpp(r'''
#include "Renderer/native/c3x_renderer_api.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
using DWORD=unsigned;inline DWORD GetCurrentThreadId(){return 1;}
inline void OutputDebugStringA(char const*){}
struct Image {void* native=nullptr;bool owned=false,dirty=false;std::uint64_t used=0;unsigned width=0,height=0;};
unsigned scans=0;
template<class T,std::size_t N> struct Counted : std::array<T,N> {
 auto begin(){++scans;return std::array<T,N>::begin();}
 auto end(){return std::array<T,N>::end();}
};
struct Harness {
 Counted<Image,128> images{};std::uint64_t cpu_bytes=0,source_age=0;unsigned source_evictions=0,ownership=0;DWORD thread=1;
 void forget(Image& image){image={};}
 void cpu_ownership(Image& image,int){++ownership;image.owned=false;}
''' + find + trim + operation + r'''
  (void)destination;(void)source;(void)source_rect;(void)target_rect;(void)color;return 1;
 }
};
int main(){Harness h;static char natives[128];
 for(int n=0;n<100;++n)h.images[n].native=&natives[n];
 // The advisors read sprites and write a canvas: three admitted and three
 // unknown images in turn.
 scans=0;
 for(int n=0;n<50000;++n)for(int op:{C3X_NATIVE_PIXEL,C3X_NATIVE_BITS})for(int image:{90,120,10,121,50,122})
  assert(h.operation(op,&natives[image],nullptr,nullptr,nullptr,0)==0);
 std::printf("%u slot scans for 600k pixel/bits calls\n",scans);
 assert(h.ownership==300000&&scans<=12);
 // Least-recently-used order still sees the access.
 assert(h.images[50].used==h.source_age);
 // A retired slot is not found again; other images are.
 h.forget(h.images[90]);assert(!h.find(&natives[90])&&h.find(&natives[89])==&h.images[89]);
 assert(h.operation(C3X_NATIVE_DC,&natives[89],nullptr,nullptr,nullptr,0)==0&&h.ownership==300001);
 // Every other operation still trims CPU sources first.
 scans=0;assert(h.operation(C3X_NATIVE_COPY,&natives[89],nullptr,nullptr,nullptr,0)==1&&scans>=1);
 std::printf("PASS adapter pixel access\n");
}
''')


    def test_inactive_input_recording_allocates_nothing_per_call(self):
        # MSVC's std::map allocates on construction (libc++'s does not), so
        # this runs on the Windows VM like the bridge itself.
        run_cpp(r"""
#define NOMINMAX
#include <windows.h>
#pragma comment(lib,"gdi32.lib")
#pragma comment(lib,"user32.lib")
#include "Renderer/native/input_recording/native_calls.h"
#include <cstdio>
#include <cstdlib>
#include <new>
unsigned allocations=0;
void* operator new(std::size_t n){++allocations;if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}
void operator delete(void* p,std::size_t)noexcept{std::free(p);}
int main(){using namespace c3x_inputs;
 {NativeCall warm(1,[](auto&){});warm.result(0);} // the recorder service starts inactive
 allocations=0;
 for(int n=0;n<1000;++n){NativeCall input(1,[](auto&){});input.result(0);}
 std::printf("%u allocations for 1000 native calls without recording\n",allocations);
 if(allocations)return 1;
 std::printf("PASS inactive input recording\n");
}
""", timeout=120)

if __name__ == '__main__':
    unittest.main()
