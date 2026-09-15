#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <vector>
#include <cstring>
#include <stdexcept>
namespace c3x_renderer {
// Worker-owned finishing resources. The prepared shadow plane is source data;
// rendered body pixels never visit CPU memory on this path.
class GpuUnitFinish {
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    Ptr<ID3D11ComputeShader> shader;
    Ptr<ID3D11Texture2D> shadow;
    Ptr<ID3D11ShaderResourceView> shadow_view;
    unsigned width=0,height=0;
    static void check(HRESULT hr){if(FAILED(hr))throw std::runtime_error("GPU unit finishing failed");}
public:
    void reset(){shader.Reset();shadow.Reset();shadow_view.Reset();width=height=0;}
    Ptr<ID3D11Texture2D> finish(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* body,
                              unsigned w,unsigned h,std::vector<unsigned char> const& coverage){
        if(!w||!h||w>1024||h>1024||coverage.size()!=std::size_t(w)*h)throw std::runtime_error("invalid prepared unit shadow");
        if(!shader){char const* source=R"(
Texture2D<float4> body:register(t0);Texture2D<uint> shadow:register(t1);RWTexture2D<uint> output:register(u0);
[numthreads(8,8,1)] void main(uint3 at:SV_DispatchThreadID){
 uint w,h;output.GetDimensions(w,h);if(at.x>=w||at.y>=h)return;
 uint4 c=uint4(round(saturate(body.Load(int3(at.xy,0)))*255));uint shade=shadow.Load(int3(at.xy,0));
 uint alpha=c.a+(shade*(255-c.a)+127)/255;
 uint3 color=(c.rgb*c.a+127)/255;output[at.xy]=color.b|(color.g<<8)|(color.r<<16)|(alpha<<24);
})";
            Ptr<ID3DBlob> code,error;check(D3DCompile(source,std::strlen(source),"unit alpha and shadow",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
            check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader));
        }
        if(width!=w||height!=h){shadow_view.Reset();shadow.Reset();width=height=0;
            D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
            d.Format=DXGI_FORMAT_R8_UINT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            check(device->CreateTexture2D(&d,nullptr,&shadow));check(device->CreateShaderResourceView(shadow.Get(),nullptr,&shadow_view));width=w;height=h;
        }
        context->UpdateSubresource(shadow.Get(),0,nullptr,coverage.data(),w,0);
        D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
        desc.Format=DXGI_FORMAT_R32_UINT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        Ptr<ID3D11Texture2D> result;Ptr<ID3D11UnorderedAccessView> write;Ptr<ID3D11ShaderResourceView> input;
        check(device->CreateTexture2D(&desc,nullptr,&result));check(device->CreateUnorderedAccessView(result.Get(),nullptr,&write));check(device->CreateShaderResourceView(body,nullptr,&input));
        ID3D11ShaderResourceView* reads[2]={input.Get(),shadow_view.Get()};auto output=write.Get();
        context->CSSetShaderResources(0,2,reads);context->CSSetUnorderedAccessViews(0,1,&output,nullptr);context->CSSetShader(shader.Get(),nullptr,0);
        context->Dispatch((w+7)/8,(h+7)/8,1);
        reads[0]=reads[1]=nullptr;output=nullptr;context->CSSetShaderResources(0,2,reads);context->CSSetUnorderedAccessViews(0,1,&output,nullptr);
        return result;
    }
};
}
