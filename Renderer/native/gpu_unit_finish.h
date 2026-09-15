#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <cstring>
#include <stdexcept>
namespace c3x_renderer {
// Worker-owned finishing consumes the resident body and shadow pass outputs.
// Only projection metadata crosses from the CPU; no intermediate pixel upload.
class GpuUnitFinish {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D11ComputeShader> shader;
    Ptr<ID3D11Buffer> projection;
    static void check(HRESULT hr){if(FAILED(hr))throw std::runtime_error("GPU unit finishing failed");}
public:
    void reset(){shader.Reset();projection.Reset();}
    Ptr<ID3D11Texture2D> finish(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* body,
                              unsigned w,unsigned h,ID3D11ShaderResourceView* heights,std::array<float,16> const& ground){
        if(!w||!h||w>1024||h>1024||!heights)throw std::runtime_error("invalid prepared unit shadow");
        if(!shader){char const* source=R"(
Texture2D<float4> body:register(t0);Texture2D<float> heights:register(t1);RWTexture2D<uint> output:register(u0);
cbuffer Ground:register(b0){float4 pose,bounds,quality;};
[numthreads(8,8,1)] void main(uint3 at:SV_DispatchThreadID){
 uint w,h;output.GetDimensions(w,h);if(at.x>=w||at.y>=h)return;
 uint4 c=uint4(round(saturate(body.Load(int3(at.xy,0)))*255));
  precise float sx=(float(at.x)+.5f-pose.x)/(64*pose.z),sy=(float(at.y)+.5f-pose.y)/(32*pose.z);
  precise float x=(sx+sy)*.5f,y=(sy-sx)*.5f;
  int px=int(floor((x-bounds.x)/bounds.z*quality.x)),py=int(floor((y-bounds.y)/bounds.w*quality.x));
  precise float count=0;
  [unroll]for(int oy=-1;oy<=1;++oy)[unroll]for(int ox=-1;ox<=1;++ox){
   int2 q=int2(px+ox,py+oy);
   if(all(q>=0)&&all(q<int(quality.x))&&heights.Load(int3(q,0))>.006f)count+=1.f/9;
  }
  precise float fade=clamp(float(min(min(at.x,at.y),min(w-1-at.x,h-1-at.y)))/3,0.f,1.f);
  precise float shadow_alpha=pose.w*fade*count;uint shade=uint(shadow_alpha);
 uint alpha=c.a+(shade*(255-c.a)+127)/255;
 uint3 color=(c.rgb*c.a+127)/255;output[at.xy]=color.b|(color.g<<8)|(color.r<<16)|(alpha<<24);
})";
            Ptr<ID3DBlob> code,error;check(D3DCompile(source,std::strlen(source),"unit alpha and shadow",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS,0,&code,&error));
            check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader));
        }
        if(!projection){D3D11_BUFFER_DESC d={};d.ByteWidth=48;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;check(device->CreateBuffer(&d,nullptr,&projection));}
        context->UpdateSubresource(projection.Get(),0,nullptr,ground.data(),0,0);
        auto settings=projection.Get();context->CSSetConstantBuffers(0,1,&settings);
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_R32_UINT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        Ptr<ID3D11Texture2D> result;Ptr<ID3D11UnorderedAccessView> write;Ptr<ID3D11ShaderResourceView> input;
        check(device->CreateTexture2D(&desc,nullptr,&result));check(device->CreateUnorderedAccessView(result.Get(),nullptr,&write));check(device->CreateShaderResourceView(body,nullptr,&input));
        ID3D11ShaderResourceView* reads[2]={input.Get(),heights};auto output=write.Get();
        context->CSSetShaderResources(0,2,reads);context->CSSetUnorderedAccessViews(0,1,&output,nullptr);context->CSSetShader(shader.Get(),nullptr,0);
        context->Dispatch((w+7)/8,(h+7)/8,1);
        reads[0]=reads[1]=nullptr;output=nullptr;context->CSSetShaderResources(0,2,reads);context->CSSetUnorderedAccessViews(0,1,&output,nullptr);
        return result;
    }
};
}
