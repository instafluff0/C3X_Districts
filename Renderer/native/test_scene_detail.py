import unittest
from Renderer.native.native_cpp_test import run_cpp


class SceneDetailTests(unittest.TestCase):
    def test_contrast_alpha_and_disabled_identity(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
#include <limits>
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 Compositor gpu(device.Get(),context.Get());constexpr unsigned w=64,h=48;
 std::vector<unsigned> pixels(w*h,0xff484848);
 for(unsigned y=8;y<24;++y)for(unsigned x=8;x<24;++x){unsigned v=x%2?112:80;pixels[y*w+x]=0xff000000|v|(v<<8)|(v<<16);}
 for(unsigned y=8;y<24;++y)for(unsigned x=32;x<48;++x)pixels[y*w+x]=x<40?0xff000000:0xffffffff;
 pixels[32*w+32]=0x80808080;
 D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
 d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.Usage=D3D11_USAGE_IMMUTABLE;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 D3D11_SUBRESOURCE_DATA data={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> source;
 checked(device->CreateTexture2D(&d,&data,&source));auto id=gpu.create(w,h,Format::bgra32);
 auto output=gpu.release_import_target(id);
 for(float invalid:{-.1f,1.1f,std::numeric_limits<float>::quiet_NaN()})assert(!gpu.import_bgra(output,source.Get(),0,0,invalid));
 assert(gpu.import_bgra(output,source.Get()));assert(retained_read(device.Get(),context.Get(),output.texture.Get())==pixels);
 auto allocations=gpu.stats().allocations;
 for(float amount:{.2f,.35f,1.f}){
  assert(gpu.import_bgra(output,source.Get(),0,0,amount));auto filtered=retained_read(device.Get(),context.Get(),output.texture.Get());
  for(unsigned i=0;i<pixels.size();++i)assert((filtered[i]>>24)==(pixels[i]>>24));
  for(unsigned y=0;y<6;++y)for(unsigned x=0;x<w;++x)assert(filtered[y*w+x]==pixels[y*w+x]);
  for(unsigned y=10;y<22;++y)for(unsigned x=34;x<46;++x)assert(filtered[y*w+x]==pixels[y*w+x]);
  for(unsigned y=31;y<=33;++y)for(unsigned x=31;x<=33;++x)assert(filtered[y*w+x]==pixels[y*w+x]);
  assert((filtered[16*w+16]&255)<80);assert((filtered[16*w+17]&255)>112);
 }
 assert(gpu.stats().allocations==allocations&&gpu.stats().uploads==0);
 std::puts("PASS scene CAS: disabled identity, increased interior contrast, preserved flat fields, extrema and alpha, no hot allocation or upload");
 }catch(std::exception const& e){std::printf("FAIL scene CAS: %s\n",e.what());return 1;}
}
''', timeout=90)


if __name__ == "__main__":
    unittest.main()
