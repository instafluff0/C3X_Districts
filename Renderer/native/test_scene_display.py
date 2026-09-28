"""Exercise the actual shared HDR transfer on D3D, including extreme radiance."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class SceneDisplayTests(unittest.TestCase):
    def test_black_monotonicity_bright_headroom_and_neutral_control(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
#include "Renderer/native/scene_display.h"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 std::string code=c3x_renderer::scene_display_shader();code+=R"(
RWTexture2D<float4> target:register(u0);
[numthreads(64,1,1)] void CS(uint3 p:SV_DispatchThreadID){
 float x=p.x==0?0:pow(10.,(float(p.x)-128)/32.);
 float amount=float(p.y)*.5;
 target[p.xy]=float4(scene_display_srgb(float3(x,x*.6,x*.2),amount),1);
})";
 ComPtr<ID3DBlob> blob,errors;checked(D3DCompile(code.data(),code.size(),"scene display oracle",nullptr,nullptr,"CS","cs_5_0",0,0,&blob,&errors));
 ComPtr<ID3D11ComputeShader> shader;checked(device->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&shader));
 D3D11_TEXTURE2D_DESC desc={};desc.Width=256;desc.Height=3;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
 desc.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
 ComPtr<ID3D11Texture2D> output,readback;checked(device->CreateTexture2D(&desc,nullptr,&output));
 ComPtr<ID3D11UnorderedAccessView> view;checked(device->CreateUnorderedAccessView(output.Get(),nullptr,&view));
 auto uav=view.Get();context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);context->CSSetShader(shader.Get(),nullptr,0);
 context->Dispatch(4,3,1);context->ClearState();
 desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 checked(device->CreateTexture2D(&desc,nullptr,&readback));context->CopyResource(readback.Get(),output.Get());
 D3D11_MAPPED_SUBRESOURCE map={};checked(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&map));
 auto srgb=[](double v){return v<=.0031308?v*12.92:1.055*std::pow(v,1./2.4)-.055;};
 for(unsigned row=0;row<3;++row){auto values=reinterpret_cast<float const*>(static_cast<char const*>(map.pData)+row*map.RowPitch);
  for(unsigned x=0;x<256;++x)for(unsigned c=0;c<3;++c){float v=values[x*4+c];
   assert(std::isfinite(v)&&v>=0&&v<=1.00001f);
   if(!x)assert(v==0);else assert(v>=values[(x-1)*4+c]-1e-6f);
   if(!row){double input=x?std::pow(10.,(double(x)-128)/32.):0.;double channel=c==0?1:c==1?.6:.2;
    assert(std::abs(v-srgb(input*channel/(1+input)))<1e-5);
   }
  }
  assert(values[32*4]>0&&values[224*4]<1.00001f); // retain near-black and bright range
 }
 context->Unmap(readback.Get(),0);
 std::puts("PASS scene display: exact neutral control, black, finite HDR range and monotone channels");
 }catch(std::exception const& e){std::printf("FAIL scene display: %s\n",e.what());return 1;}
}
''', timeout=90)


if __name__ == '__main__':
    unittest.main()
