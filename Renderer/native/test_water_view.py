"""GPU water view continuity across wrapped occurrences and display zoom."""
from pathlib import Path
import unittest
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.source_fidelity.prepare import function

ROOT = Path(__file__).resolve().parents[2]


class WaterViewTests(unittest.TestCase):
    def test_projected_world_ray_survives_wrap_zoom_and_keeps_glint(self):
        helper = function((ROOT/'Renderer/sandbox/water_surface.hlsl').read_text(),
                          'water_view_direction')
        shader = r'''
cbuffer Basis : register(b0) {float2 origin; float zoom; float pad;};
cbuffer WaterView : register(b11) {float4 bounds; float4 water_view;};
struct PixelInput {float4 position:SV_Position; float4 q6_world:TEXCOORD0;};
PixelInput VS(uint id:SV_VertexID) {
 float2 p=float2((id<<1)&2,id&2);
 PixelInput o;o.position=float4(p*float2(2,-2)+float2(-1,1),0,1);
 o.q6_world=float4(origin+float2(p.x+p.y,p.x-p.y)/zoom,0,1);return o;
}
''' + helper + r'''
float4 PS(PixelInput input):SV_Target {
 float3 eye=water_view_direction(input);
 float3 normal=float3(0,0,1),sun=normalize(float3(-.43,.43,1));
 float glint=pow(saturate(dot(reflect(-sun,normal),eye)),96);
 return float4(eye,glint);
}
'''
        code = r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
#include <array>
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 char const* source=R"WATER(SHADER)WATER";
 ComPtr<ID3DBlob> vs,ps,error;
 checked(D3DCompile(source,strlen(source),"water",nullptr,nullptr,"VS","vs_5_0",0,0,&vs,&error));
 checked(D3DCompile(source,strlen(source),"water",nullptr,nullptr,"PS","ps_5_0",0,0,&ps,&error));
 ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel;
 checked(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex));
 checked(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel));
 auto buffer=[&](unsigned bytes){D3D11_BUFFER_DESC d={};d.ByteWidth=bytes;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
  ComPtr<ID3D11Buffer> b;checked(device->CreateBuffer(&d,nullptr,&b));return b;};
 auto basis=buffer(16),view=buffer(32);
 float water[8]={0,0,0,0,64,32,0,0};context->UpdateSubresource(view.Get(),0,nullptr,water,0,0);
 auto b=basis.Get(),v=view.Get();context->VSSetConstantBuffers(0,1,&b);context->PSSetConstantBuffers(11,1,&v);
 D3D11_TEXTURE2D_DESC t={};t.Width=128;t.Height=64;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
 t.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;t.BindFlags=D3D11_BIND_RENDER_TARGET;
 ComPtr<ID3D11Texture2D> texture,read;checked(device->CreateTexture2D(&t,nullptr,&texture));
 ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(texture.Get(),nullptr,&target));
 t.BindFlags=0;t.Usage=D3D11_USAGE_STAGING;t.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&t,nullptr,&read));
 D3D11_RASTERIZER_DESC r={};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.ScissorEnable=TRUE;r.DepthClipEnable=TRUE;
 ComPtr<ID3D11RasterizerState> raster;checked(device->CreateRasterizerState(&r,&raster));context->RSSetState(raster.Get());
 D3D11_VIEWPORT viewport={0,0,128,64,0,1};context->RSSetViewports(1,&viewport);
 auto output=target.Get();context->OMSetRenderTargets(1,&output,nullptr);
 context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
 context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
 for(float zoom:{1.f,1.125f,1.25f,1.5f})for(float wrap:{0.f,8.f,50.f,100.f,-100.f}){
  for(int side=0;side<2;++side){
   float values[4]={side?wrap:0,side?-wrap:0,zoom,0};context->UpdateSubresource(basis.Get(),0,nullptr,values,0,0);
   D3D11_RECT scissor={side*64,0,(side+1)*64,64};context->RSSetScissorRects(1,&scissor);context->Draw(3,0);
  }
  context->CopyResource(read.Get(),texture.Get());D3D11_MAPPED_SUBRESOURCE mapped={};
  checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&mapped));
  float brightest=0;
  for(unsigned y=1;y<63;++y)for(unsigned x=1;x<127;++x){
   auto actual=reinterpret_cast<float const*>(static_cast<char const*>(mapped.pData)+y*mapped.RowPitch)+x*4;
   float dx=(float(x)+.5f-64)/128/zoom,dy=(float(y)+.5f-32)/64/zoom;
   float expected[3]={1.075f-dx-dy,-1.075f-dx+dy,2.5f};
   float length=std::sqrt(expected[0]*expected[0]+expected[1]*expected[1]+expected[2]*expected[2]);
   for(unsigned a=0;a<3;++a)assert(std::abs(actual[a]-expected[a]/length)<.0002f);
   brightest=std::max(brightest,actual[3]);
  }
  context->Unmap(read.Get(),0);assert(brightest>.99f);
 }
 std::puts("PASS water view: continuous projected world basis across 5 wrap offsets and 4 zooms; specular glint retained");
 }catch(std::exception const& e){std::printf("FAIL water view: %s\n",e.what());return 1;}
}
'''.replace('SHADER', shader)
        run_cpp(code, timeout=90)


if __name__ == '__main__':
    unittest.main()
