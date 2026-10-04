"""CPU UI edits preserve exact, versioned sources without full-canvas uploads."""
import unittest
from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def refresh_contract():
    source = (ROOT / 'Renderer/native/native_image_adapter.h').read_text()
    start = source.index('    bool refresh(Image& image){')
    return source[start:source.index('    // A native-format GPU mirror', start)]


class NativeSourceUpdateTests(unittest.TestCase):
    def test_small_strided_edits_and_retained_versions(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#pragma comment(lib,"gdi32.lib")
#include "Renderer/native/test_retained_composition.cpp"
struct Native {unsigned width=257,height=128,stride=260;std::vector<std::uint16_t> words=std::vector<std::uint16_t>(stride*height,0x123);unsigned leases=0;};
int field(void* p,int offset){assert(offset==0x40);return static_cast<Native*>(p)->stride;}
namespace c3x_native_access {
 std::uint16_t* words(void* p,void*){auto& n=*static_cast<Native*>(p);++n.leases;return n.words.data();}
 void release_words(void* p,void*){--static_cast<Native*>(p)->leases;}
}
struct Image {void* native;Id gpu=0;unsigned width=257,height=128;bool cpu_uploaded=false,owned=false;Format format=Format::rgb555;
 std::vector<std::uint16_t> cpu=std::vector<std::uint16_t>(width*height);std::uint64_t revision=0;};
struct Gpu {
 Compositor& live;RetainedComposition& retained;Native& native;bool reject=false;unsigned temporary=0;
 Id create(unsigned w,unsigned h,Format f){assert(!native.leases);auto id=live.create(w,h,f);retained.create(id,w,h,f);++temporary;return id;}
 bool destroy(Id id){assert(!native.leases);retained.destroy(id);--temporary;return live.destroy(id);}
 bool upload(Id id,std::uint64_t rev,unsigned const* p,std::size_t n){assert(!native.leases);if(reject)return false;
  bool ok=live.upload(id,rev,p,n);if(ok)retained.source(id,live.texture(id));return ok;}
 bool submit(Command const* commands,unsigned count){assert(!native.leases);bool ok=live.submit(commands,count);
  if(ok)for(unsigned i=0;i<count;++i)retained.record(commands[i]);return ok;}
};
struct Harness {Gpu gpu;void* get_bits=nullptr;void* release_bits=nullptr;unsigned large_uploads=0,source_evictions=0;
 std::size_t cpu_bytes=257*128*2;struct{unsigned source_checks=0,source_reuses=0;std::size_t source_expanded_bytes=0;}counters;
''' + refresh_contract() + r'''
};
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 Compositor live(device.Get(),context.Get());RetainedComposition retained(device.Get(),context.Get());
 Native native;Image image{&native};image.gpu=live.create(image.width,image.height,image.format);
 retained.create(image.gpu,image.width,image.height,image.format);Harness h{{live,retained,native}};Rect full={0,0,int(image.width),int(image.height)};
 auto expected=[&]{std::vector<unsigned> out;for(unsigned y=0;y<native.height;++y)for(unsigned x=0;x<native.width;++x)out.push_back(native.words[y*native.stride+x]);return out;};
 auto check=[&]{retained.commit(image.gpu,full);auto want=expected();
  assert(retained_read(device.Get(),context.Get(),retained.sample(image.revision,1000).Get())==want);
  assert(retained_read(device.Get(),context.Get(),live.texture(image.gpu))==want);
  assert(!native.leases&&!h.gpu.temporary&&retained.node_count()<400);
 };
 assert(h.refresh(image));check();auto expanded=h.counters.source_expanded_bytes;
 native.words[257]=0xffff;assert(h.refresh(image));assert(h.counters.source_expanded_bytes==expanded); // padding only
 std::vector<std::pair<Id,std::vector<unsigned>>> saves;
 for(unsigned n=0;n<150;++n){
  unsigned x=n%2?256:(n*19)%257,y=n%2?127:(n*13)%128;native.words[y*native.stride+x]=0x200+n;
  auto revision=image.revision;expanded=h.counters.source_expanded_bytes;
  assert(h.refresh(image));check();
  assert(h.counters.source_expanded_bytes-expanded==(revision%64?4:257*128*4));
  if(n%30==0){Id saved=10000+n;retained.create(saved,image.width,image.height,image.format);retained.snapshot(saved,image.gpu);saves.push_back({saved,expected()});}
 }
 // A failed partial upload must neither commit the CPU cache nor leak its image.
 native.words[0]=0x777;auto revision=image.revision;auto before=image.cpu;
 h.gpu.reject=true;assert(!h.refresh(image));assert(image.revision==revision&&image.cpu==before&&!h.gpu.temporary);
 h.gpu.reject=false;assert(h.refresh(image));check();
 // Wide changes and GPU-to-CPU invalidation use complete authoritative images.
 std::fill(native.words.begin(),native.words.end(),0x345);expanded=h.counters.source_expanded_bytes;
 assert(h.refresh(image));assert(h.counters.source_expanded_bytes-expanded==257*128*4);check();
 image.cpu_uploaded=false;native.words[0]=0x111;expanded=h.counters.source_expanded_bytes;
 assert(h.refresh(image));assert(h.counters.source_expanded_bytes-expanded==257*128*4);check();
 for(auto const& saved:saves){retained.commit(saved.first,full);assert(retained_read(device.Get(),context.Get(),retained.sample(++image.revision,1000).Get())==saved.second);}
 std::printf("PASS 150 strided source edits, bounded history, padding, retries, full updates and saved versions\n");
 }catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}
}
''', timeout=120)


if __name__ == '__main__':
    unittest.main()
