#pragma once
#include <array>
// Shared final image pass: full-color map pixels and packed native UI sources.
namespace c3x_gpu_images {
class ImageDisplay {
    Microsoft::WRL::ComPtr<ID3D11VertexShader> vertex;
    Microsoft::WRL::ComPtr<ID3D11PixelShader> pixel[3];
    Microsoft::WRL::ComPtr<ID3D11RasterizerState> raster;
public:
    bool draw(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11ShaderResourceView* input,
              ID3D11RenderTargetView* target,unsigned width,unsigned height,RECT clip,unsigned native_format=0,
              std::array<LONGLONG,8>* phase_ticks=nullptr){
        if(!input||!target||native_format>2)return false;
        LARGE_INTEGER mark={};
        auto record=[&](unsigned field){if(phase_ticks){LARGE_INTEGER next={};QueryPerformanceCounter(&next);(*phase_ticks)[field]=next.QuadPart-mark.QuadPart;mark=next;}};
        if(phase_ticks)QueryPerformanceCounter(&mark);
        // Window/GDI handoffs retain these shared shaders. Device replacement
        // invalidates them explicitly on the GPU owner before the next draw.
        if(vertex){Microsoft::WRL::ComPtr<ID3D11Device> previous;vertex->GetDevice(&previous);
            if(previous.Get()!=device){vertex.Reset();for(auto& shader:pixel)shader.Reset();raster.Reset();}}
        auto check=[](HRESULT hr){if(FAILED(hr))throw std::runtime_error("native image display failed");};
        char const* source=R"(
Texture2D<uint> input_image:register(t0);
float4 vs(uint id:SV_VertexID):SV_Position {return float4(id==2?3:-1,id==1?-3:1,0,1);}
float4 ps(float4 at:SV_Position):SV_Target {uint c=input_image.Load(int3(int2(at.xy),0));return float4((c>>16)&255,(c>>8)&255,c&255,255)/255.0;}
float4 native_color(float4 at,uint rb,uint mask):SV_Target {uint c=input_image.Load(int3(int2(at.xy),0));uint r=(c>>rb)&31,g=(c>>5)&mask,b=c&31;
return float4((r<<3)|(r>>2),mask==63?((g<<2)|(g>>4)):((g<<3)|(g>>2)),(b<<3)|(b>>2),255)/255.0;}
float4 ps555(float4 at:SV_Position):SV_Target {return native_color(at,10,31);}
float4 ps565(float4 at:SV_Position):SV_Target {return native_color(at,11,63);}
)";
        Microsoft::WRL::ComPtr<ID3DBlob> code,error;
        if(!vertex){
            check(D3DCompile(source,std::strlen(source),"native final transfer",nullptr,nullptr,"vs","vs_4_0",0,0,&code,&error));
            check(device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&vertex));
            D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.ScissorEnable=TRUE;r.DepthClipEnable=TRUE;
            check(device->CreateRasterizerState(&r,&raster));
        }
        if(!pixel[native_format]){
            check(D3DCompile(source,std::strlen(source),"native final transfer",nullptr,nullptr,native_format==0?"ps":native_format==1?"ps555":"ps565","ps_4_0",0,0,&code,&error));
            check(device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&pixel[native_format]));
        }
        record(0);
        context->ClearState();record(1);
        D3D11_VIEWPORT viewport={0,0,float(width),float(height),0,1};context->RSSetViewports(1,&viewport);
        context->RSSetScissorRects(1,&clip);context->RSSetState(raster.Get());record(2);
        context->OMSetRenderTargets(1,&target,nullptr);record(3);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);context->VSSetShader(vertex.Get(),nullptr,0);
        context->PSSetShader(pixel[native_format].Get(),nullptr,0);record(4);
        context->PSSetShaderResources(0,1,&input);record(5);
        context->Draw(3,0);record(6);
        context->ClearState();record(7);
        return true;
    }
};
}
