"""The opaque underlay alone may force early tests; coverage clips stay late."""
import unittest
from unittest.mock import patch
from Renderer.native import test_underlay_occlusion as fixture
from Renderer.native.native_cpp_test import run_cpp


class EarlyUnderlayTests(unittest.TestCase):
    def test_msaa_fallback_opaque_underlay_preserves_every_sample(self):
        # The optimization's stencil prepasses are disabled for MSAA. The
        # underlay entry still has its early flag, so check that fallback too.
        run_cpp(r'''
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cassert>
#include <cstring>
#include <string>
#include <vector>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 assert(SUCCEEDED(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,
  D3D11_SDK_VERSION,&device,nullptr,&context)));
 for(unsigned samples:{2u,4u}){
  std::string program="static const uint Samples="+std::to_string(samples)+R"(;
   cbuffer Values:register(b0){float4 color;float4 setup;};
   float4 VS(uint id:SV_VertexID):SV_Position {
    float2 p=id==0?float2(-1,-1):id==1?float2(-1,3):float2(3,-1);
    return float4(p,setup.x,1);
   }
   float4 PS(float4 p:SV_Position):SV_Target {
    if(setup.y>0)clip(setup.y-p.x);clip(color.a-.000001);
    return float4(color.rgb*color.a,color.a);
   }
   [earlydepthstencil]
   float4 Underlay(float4 p:SV_Position):SV_Target {
    clip(color.a-.000001);return float4(color.rgb*color.a,color.a);
   }
   Texture2DMS<float4,Samples> C:register(t0);
   Texture2DMS<float,Samples> D:register(t1);
   uint2 ReadSamples(float4 p:SV_Position):SV_Target {
    uint x=uint(p.x);int2 pixel=int2(x/Samples,uint(p.y));uint s=x%Samples;
    uint4 c=uint4(round(C.Load(pixel,s)*255));
    return uint2(c.x|(c.y<<8)|(c.z<<16)|(c.w<<24),asuint(D.Load(pixel,s)));
   }
  )";
  auto compile=[&](char const* entry,char const* profile){ComPtr<ID3DBlob> blob,error;
   assert(SUCCEEDED(D3DCompile(program.data(),program.size(),nullptr,nullptr,nullptr,entry,profile,
    D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&blob,&error)));return blob;};
  auto vb=compile("VS","vs_5_0"),pb=compile("PS","ps_5_0"),eb=compile("Underlay","ps_5_0"),rb=compile("ReadSamples","ps_5_0");
  ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps,early,reader;
  assert(SUCCEEDED(device->CreateVertexShader(vb->GetBufferPointer(),vb->GetBufferSize(),nullptr,&vs)));
  assert(SUCCEEDED(device->CreatePixelShader(pb->GetBufferPointer(),pb->GetBufferSize(),nullptr,&ps)));
  assert(SUCCEEDED(device->CreatePixelShader(eb->GetBufferPointer(),eb->GetBufferSize(),nullptr,&early)));
  assert(SUCCEEDED(device->CreatePixelShader(rb->GetBufferPointer(),rb->GetBufferSize(),nullptr,&reader)));
  D3D11_TEXTURE2D_DESC td={};td.Width=33;td.Height=17;td.MipLevels=td.ArraySize=1;
  td.SampleDesc.Count=samples;td.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
  td.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
  ComPtr<ID3D11Texture2D> color,depth,packed,readback;ComPtr<ID3D11RenderTargetView> target,packed_target;
  ComPtr<ID3D11ShaderResourceView> color_view,depth_view;
  assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&color)));
  assert(SUCCEEDED(device->CreateRenderTargetView(color.Get(),nullptr,&target)));
  assert(SUCCEEDED(device->CreateShaderResourceView(color.Get(),nullptr,&color_view)));
  td.Format=DXGI_FORMAT_R24G8_TYPELESS;td.BindFlags=D3D11_BIND_DEPTH_STENCIL|D3D11_BIND_SHADER_RESOURCE;
  assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&depth)));
  D3D11_DEPTH_STENCIL_VIEW_DESC dd={};dd.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;dd.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2DMS;
  ComPtr<ID3D11DepthStencilView> dsv;assert(SUCCEEDED(device->CreateDepthStencilView(depth.Get(),&dd,&dsv)));
  D3D11_SHADER_RESOURCE_VIEW_DESC sd={};sd.Format=DXGI_FORMAT_R24_UNORM_X8_TYPELESS;sd.ViewDimension=D3D11_SRV_DIMENSION_TEXTURE2DMS;
  assert(SUCCEEDED(device->CreateShaderResourceView(depth.Get(),&sd,&depth_view)));
  td.Width=33*samples;td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R32G32_UINT;td.BindFlags=D3D11_BIND_RENDER_TARGET;
  assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&packed)));
  assert(SUCCEEDED(device->CreateRenderTargetView(packed.Get(),nullptr,&packed_target)));
  td.BindFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  assert(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&readback)));
  D3D11_BUFFER_DESC bd={};bd.ByteWidth=32;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
  ComPtr<ID3D11Buffer> values;assert(SUCCEEDED(device->CreateBuffer(&bd,nullptr,&values)));
  D3D11_DEPTH_STENCIL_DESC ds={};ds.DepthEnable=TRUE;ds.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;ds.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
  ComPtr<ID3D11DepthStencilState> original_depth,no_depth;
  assert(SUCCEEDED(device->CreateDepthStencilState(&ds,&original_depth)));ds.DepthEnable=FALSE;
  assert(SUCCEEDED(device->CreateDepthStencilState(&ds,&no_depth)));
  D3D11_BLEND_DESC blend={};auto& b=blend.RenderTarget[0];b.BlendEnable=TRUE;
  b.SrcBlend=b.SrcBlendAlpha=D3D11_BLEND_ONE;b.DestBlend=b.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;
  b.BlendOp=b.BlendOpAlpha=D3D11_BLEND_OP_ADD;b.RenderTargetWriteMask=15;
  ComPtr<ID3D11BlendState> original_blend,no_blend;
  assert(SUCCEEDED(device->CreateBlendState(&blend,&original_blend)));b.BlendEnable=FALSE;
  assert(SUCCEEDED(device->CreateBlendState(&blend,&no_blend)));
  D3D11_RASTERIZER_DESC rd={};rd.FillMode=D3D11_FILL_SOLID;rd.CullMode=D3D11_CULL_NONE;
  rd.DepthClipEnable=rd.ScissorEnable=rd.MultisampleEnable=TRUE;
  ComPtr<ID3D11RasterizerState> raster;assert(SUCCEEDED(device->CreateRasterizerState(&rd,&raster)));
  auto draw=[&](float z,float alpha,float edge,float r,float g,float blue){float data[]={r,g,blue,alpha,z,edge,0,0};
   context->UpdateSubresource(values.Get(),0,nullptr,data,0,0);context->Draw(3,0);};
  auto render=[&](bool candidate,float z,float alpha,bool late){
   ID3D11ShaderResourceView* nulls[2]={};context->PSSetShaderResources(0,2,nulls);
   float clear[4]={};context->ClearRenderTargetView(target.Get(),clear);
   context->ClearDepthStencilView(dsv.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
   auto* rt=target.Get();context->OMSetRenderTargets(1,&rt,dsv.Get());
   context->VSSetShader(vs.Get(),nullptr,0);auto* cb=values.Get();context->VSSetConstantBuffers(0,1,&cb);context->PSSetConstantBuffers(0,1,&cb);
   context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);context->RSSetState(raster.Get());
   D3D11_VIEWPORT vp={0,0,33,17,0,1};context->RSSetViewports(1,&vp);D3D11_RECT rect={3,2,30,16};context->RSSetScissorRects(1,&rect);
   context->OMSetDepthStencilState(original_depth.Get(),0);context->OMSetBlendState(original_blend.Get(),nullptr,~0u);
   context->PSSetShader(candidate?early.Get():ps.Get(),nullptr,0);draw(.5f,1,0,.2f,.3f,.5f);
   context->PSSetShader(ps.Get(),nullptr,0);draw(z,alpha,16,1,0,0);if(late)draw(.5f,.25f,0,0,1,0);
   context->OMSetRenderTargets(0,nullptr,nullptr);rt=packed_target.Get();context->OMSetRenderTargets(1,&rt,nullptr);
   context->OMSetDepthStencilState(no_depth.Get(),0);context->OMSetBlendState(no_blend.Get(),nullptr,~0u);
   vp.Width=float(33*samples);context->RSSetViewports(1,&vp);rect={0,0,LONG(33*samples),17};context->RSSetScissorRects(1,&rect);
   ID3D11ShaderResourceView* inputs[2]={color_view.Get(),depth_view.Get()};context->PSSetShaderResources(0,2,inputs);
   context->PSSetShader(reader.Get(),nullptr,0);draw(0,1,0,0,0,0);context->PSSetShaderResources(0,2,nulls);
   context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(readback.Get(),packed.Get());
   D3D11_MAPPED_SUBRESOURCE mapped={};assert(SUCCEEDED(context->Map(readback.Get(),0,D3D11_MAP_READ,0,&mapped)));
   std::vector<unsigned> result;
   for(unsigned y=0;y<17;++y)for(unsigned x=0;x<33*samples*2;++x)
    result.push_back(reinterpret_cast<unsigned const*>(static_cast<unsigned char const*>(mapped.pData)+y*mapped.RowPitch)[x]);
   context->Unmap(readback.Get(),0);return result;
  };
  for(float z:{.45f,.5f,.55f})for(float alpha:{0.f,.5f,.99999994f,1.f})for(bool late:{false,true})
   assert(render(false,z,alpha,late)==render(true,z,alpha,late));
  assert(render(false,.45f,1,false)!=render(false,.55f,1,false));
 }
}
''', timeout=90)

    def test_early_underlay_preserves_partial_coverage_and_equal_depth(self):
        # Reuse the existing executable geometry/depth fixture without changing
        # its original source or applying early tests to its clipping mask.
        def execute(program, **options):
            def replace(old, new):
                nonlocal program
                self.assertEqual(program.count(old), 1)
                program = program.replace(old, new)
            replace('void Mask(float4 p:SV_Position){', '''[earlydepthstencil]
 float4 EarlyUnderlay(float4 p:SV_Position):SV_Target {
  clip(color.a-.000001);return float4(color.rgb*color.a,color.a);
 }
 void Mask(float4 p:SV_Position){''')
            replace('auto vb=compile("VS","vs_5_0"),pb=compile("PS","ps_5_0"),mb=compile("Mask","ps_5_0");',
                    '''auto vb=compile("VS","vs_5_0"),pb=compile("PS","ps_5_0"),mb=compile("Mask","ps_5_0"),eb=compile("EarlyUnderlay","ps_5_0");
 ComPtr<ID3DBlob> early_text,mask_text;
 assert(SUCCEEDED(D3DDisassemble(eb->GetBufferPointer(),eb->GetBufferSize(),0,nullptr,&early_text)));
 assert(SUCCEEDED(D3DDisassemble(mb->GetBufferPointer(),mb->GetBufferSize(),0,nullptr,&mask_text)));
 assert(std::strstr(static_cast<char*>(early_text->GetBufferPointer()),"forceEarlyDepthStencil"));
 assert(!std::strstr(static_cast<char*>(mask_text->GetBufferPointer()),"forceEarlyDepthStencil"));''')
            replace('ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps,mask;',
                    'ComPtr<ID3D11VertexShader> vs;ComPtr<ID3D11PixelShader> ps,mask,early;')
            anchor='assert(SUCCEEDED(device->CreatePixelShader(mb->GetBufferPointer(),mb->GetBufferSize(),nullptr,&mask)));'
            replace(anchor, anchor+'\n assert(SUCCEEDED(device->CreatePixelShader(eb->GetBufferPointer(),eb->GetBufferSize(),nullptr,&early)));')
            replace('context->OMSetBlendState(original_blend.Get(),nullptr,~0u);context->PSSetShader(ps.Get(),nullptr,0);draw(.5f,1,0,.2f,.3f,.5f);',
                    'context->OMSetBlendState(original_blend.Get(),nullptr,~0u);context->PSSetShader(candidate?early.Get():ps.Get(),nullptr,0);draw(.5f,1,0,.2f,.3f,.5f);')
            replace('context->OMSetDepthStencilState(original_depth.Get(),0);draw(z,alpha,16,1,0,0);',
                    'context->OMSetDepthStencilState(original_depth.Get(),0);context->PSSetShader(ps.Get(),nullptr,0);draw(z,alpha,16,1,0,0);')
            run_cpp(program, **options)
        with patch.object(fixture, 'run_cpp', execute):
            case = fixture.UnderlayOcclusionTests('test_partial_opaque_behind_equal_front_and_later_overlay')
            case.test_partial_opaque_behind_equal_front_and_later_overlay()


if __name__ == '__main__':
    unittest.main()
