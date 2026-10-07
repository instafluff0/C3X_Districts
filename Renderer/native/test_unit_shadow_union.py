"""Unit ground shadows: overlapping parts must darken the ground once.

Every part of a unit (and every face of a part) is drawn as its own
ground-projected shadow. Without a per-pixel guard those layers compound into
separate darker patches, the defect seen in game. The stencil marks shadowed
pixels so each darkens once. This GPU oracle runs the actual shader text and
depth-stencil descriptor from `sandbox/direct_units.h`.
"""
import pathlib
import unittest

from Renderer.native.native_cpp_test import run_cpp

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIRECT = ROOT / "Renderer/sandbox/direct_units.h"


def shadow_once_block(text):
    start = text.index("D3D11_DEPTH_STENCIL_DESC d={};renderer.natural.decal_depth->GetDesc(&d);")
    end = text.index("CreateDepthStencilState(&d,&shadow_once)))return false;", start)
    return text[start:end] + "CreateDepthStencilState(&d,&shadow_once)))return false;"


def program(text):
    source = text.split('char const* source=R"(', 1)[1].split(')";', 1)[0]
    return r'''
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
void checked(HRESULT hr){if(FAILED(hr))throw std::runtime_error("D3D call failed");}
struct Natural{ID3D11DepthStencilState* decal_depth=nullptr;};
struct Renderer{ID3D11Device* device=nullptr;Natural natural;} renderer;
bool make_shadow_once(ID3D11DepthStencilState*& shadow_once){
 ''' + shadow_once_block(text) + r'''
 return true;
}
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 renderer.device=device.Get();
 char const* source=R"UNITSHADOW(''' + source + r''')UNITSHADOW";
 ComPtr<ID3DBlob> vs,ps,error;
 checked(D3DCompile(source,strlen(source),"unit",nullptr,nullptr,"VS","vs_5_0",0,0,&vs,&error));
 checked(D3DCompile(source,strlen(source),"unit",nullptr,nullptr,"PSShadow","ps_5_0",0,0,&ps,&error));
 ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel;
 checked(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex));
 checked(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel));
 D3D11_INPUT_ELEMENT_DESC desc[]={
 {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
 {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
 {"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,24,D3D11_INPUT_PER_VERTEX_DATA,0},
 {"TANGENT",0,DXGI_FORMAT_R32G32B32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0},
 {"BINORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,44,D3D11_INPUT_PER_VERTEX_DATA,0},
 {"BLENDINDICES",0,DXGI_FORMAT_R32G32B32A32_UINT,0,56,D3D11_INPUT_PER_VERTEX_DATA,0},
 {"BLENDWEIGHT",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,72,D3D11_INPUT_PER_VERTEX_DATA,0}};
 ComPtr<ID3D11InputLayout> layout;checked(device->CreateInputLayout(desc,7,vs->GetBufferPointer(),vs->GetBufferSize(),&layout));
 // Two parts sharing one footprint, as body/armor/head layers do: a raised
 // triangle whose ground projection covers the centre of a 32x32 target.
 struct Vertex{float p[3],n[3],uv[2],t[3],b[3];unsigned joints[4];float weights[4];};
 Vertex vertices[6]={};float xyz[9]={-.12f,-.12f,.2f, .12f,-.12f,.3f, -.12f,.12f,.4f};
 for(unsigned i=0;i<6;++i){auto& v=vertices[i];std::memcpy(v.p,xyz+(i%3)*3,12);v.n[2]=v.t[0]=v.b[1]=1;v.weights[0]=1;}
 auto buffer=[&](void const* data,unsigned bytes,unsigned bind,unsigned stride=0){
  D3D11_BUFFER_DESC d={};d.ByteWidth=bytes;d.BindFlags=bind;
  if(stride){d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=stride;}
  D3D11_SUBRESOURCE_DATA s={data,0,0};ComPtr<ID3D11Buffer> b;checked(device->CreateBuffer(&d,&s,&b));return b;};
 float palette[16]={1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
 auto vb=buffer(vertices,sizeof(vertices),D3D11_BIND_VERTEX_BUFFER);
 auto pb=buffer(palette,sizeof(palette),D3D11_BIND_SHADER_RESOURCE,16);
 ComPtr<ID3D11ShaderResourceView> palettes;checked(device->CreateShaderResourceView(pb.Get(),nullptr,&palettes));
 // origin, extent, scale, depth, skin frame (frame 0, 1 bone, no yaw),
 // skin shape (scale, ground, Lab alpha), ground pass with a light offset.
 float settings[28]={16,16,32,32,1,0,0,0, 0,1,1,0, 1,0,0,0, 1,.3f,-.2f,0};
 auto cb=buffer(settings,sizeof(settings),D3D11_BIND_CONSTANT_BUFFER);
 D3D11_TEXTURE2D_DESC t={};t.Width=t.Height=32;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
 t.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;t.BindFlags=D3D11_BIND_RENDER_TARGET;
 ComPtr<ID3D11Texture2D> color,read;checked(device->CreateTexture2D(&t,nullptr,&color));
 ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(color.Get(),nullptr,&target));
 t.Format=DXGI_FORMAT_R32G32B32A32_FLOAT;t.BindFlags=0;t.Usage=D3D11_USAGE_STAGING;t.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 t.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;checked(device->CreateTexture2D(&t,nullptr,&read));
 t.Usage=D3D11_USAGE_DEFAULT;t.CPUAccessFlags=0;t.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;t.BindFlags=D3D11_BIND_DEPTH_STENCIL;
 ComPtr<ID3D11Texture2D> depth;checked(device->CreateTexture2D(&t,nullptr,&depth));
 ComPtr<ID3D11DepthStencilView> depth_view;checked(device->CreateDepthStencilView(depth.Get(),nullptr,&depth_view));
 D3D11_DEPTH_STENCIL_DESC decal={};decal.DepthEnable=TRUE;decal.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;decal.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
 ComPtr<ID3D11DepthStencilState> decal_depth;checked(device->CreateDepthStencilState(&decal,&decal_depth));
 renderer.natural.decal_depth=decal_depth.Get();
 ID3D11DepthStencilState* shadow_once=nullptr;if(!make_shadow_once(shadow_once))throw std::runtime_error("shadow_once");
 D3D11_BLEND_DESC blend={};auto& rt=blend.RenderTarget[0];rt.BlendEnable=TRUE;
 rt.SrcBlend=D3D11_BLEND_SRC_ALPHA;rt.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOp=D3D11_BLEND_OP_ADD;
 rt.SrcBlendAlpha=D3D11_BLEND_ONE;rt.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;
 rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;ComPtr<ID3D11BlendState> over;checked(device->CreateBlendState(&blend,&over));
 D3D11_RASTERIZER_DESC raster={};raster.FillMode=D3D11_FILL_SOLID;raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=TRUE;
 ComPtr<ID3D11RasterizerState> rs;checked(device->CreateRasterizerState(&raster,&rs));
 D3D11_VIEWPORT viewport={0,0,32,32,0,1};
 auto v=vb.Get(),c=cb.Get();UINT stride=sizeof(Vertex),offset=0;auto pv=palettes.Get();auto output=target.Get();
 float centre[2]={};
 auto render=[&](ID3D11DepthStencilState* state,float alpha,unsigned& shaded){
  settings[14]=alpha;context->UpdateSubresource(c,0,nullptr,settings,0,0);
  float clear[4]={.5f,.5f,.5f,1};context->ClearRenderTargetView(output,clear);
  context->ClearDepthStencilView(depth_view.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
  context->RSSetState(rs.Get());context->RSSetViewports(1,&viewport);
  context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->IASetVertexBuffers(0,1,&v,&stride,&offset);context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
  context->VSSetConstantBuffers(2,1,&c);context->PSSetConstantBuffers(2,1,&c);context->VSSetShaderResources(0,1,&pv);
  context->OMSetRenderTargets(1,&output,depth_view.Get());context->OMSetDepthStencilState(state,0);context->OMSetBlendState(over.Get(),nullptr,~0u);
  context->Draw(3,0);context->Draw(3,3); // two parts, as the production loop issues them
  context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(read.Get(),color.Get());
  D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m));
  float darkest=1;shaded=0;centre[0]=centre[1]=0;
  for(unsigned y=0;y<32;++y)for(unsigned x=0;x<32;++x){
   auto h=reinterpret_cast<unsigned short*>(static_cast<char*>(m.pData)+y*m.RowPitch)[x*4];
   unsigned sign=h>>15,exponent=(h>>10)&31,mantissa=h&1023;
   float value=std::ldexp(float(mantissa|1024),int(exponent)-25);if(sign)value=-value;
   if(value<.499f){++shaded;darkest=std::min(darkest,value);centre[0]+=x;centre[1]+=y;}
  }
  if(shaded){centre[0]/=shaded;centre[1]/=shaded;}
  context->Unmap(read.Get(),0);return darkest;};
 unsigned covered=0,again=0;
 float stacked=render(decal_depth.Get(),0,covered);
 float once=render(shadow_once,0,again);
 float strong=render(shadow_once,.45f,again);
 std::printf("shadow pixels=%u stacked=%.4f once=%.4f strong=%.4f\n",covered,stacked,once,strong);
 if(covered<20||again!=covered)throw std::runtime_error("ground footprint changed");
 // Existing path: the second part darkens the first part's shadow again.
 if(std::fabs(stacked-.5f*.72f*.72f)>.003f)throw std::runtime_error("expected the existing per-part compounding");
 // Candidate: one layer at the existing .28 strength, or the Lab strength.
 if(std::fabs(once-.5f*.72f)>.003f)throw std::runtime_error("overlapping unit parts darkened the ground more than once");
 if(std::fabs(strong-.5f*.55f)>.003f)throw std::runtime_error("Lab shadow strength not applied");
 // A floating hull below the water plane (z<0) is clipped from view. Its
 // shadow is its own footprint, not a sliver projected toward the light.
 auto footprint=[&](float z,unsigned& shaded,float* at){
  float flat[9]={-.12f,-.12f,z, .12f,-.12f,z, -.12f,.12f,z};
  for(unsigned i=0;i<6;++i)std::memcpy(vertices[i].p,flat+(i%3)*3,12);
  context->UpdateSubresource(vb.Get(),0,nullptr,vertices,0,0);
  render(shadow_once,0,shaded);at[0]=centre[0];at[1]=centre[1];};
 unsigned at_plane=0,submerged=0;float plane_centre[2],submerged_centre[2];
 footprint(0,at_plane,plane_centre);footprint(-.3f,submerged,submerged_centre);
 std::printf("waterline footprint pixels=%u/%u centre=%.2f,%.2f/%.2f,%.2f\n",at_plane,submerged,
  plane_centre[0],plane_centre[1],submerged_centre[0],submerged_centre[1]);
 if(at_plane<20||submerged!=at_plane||std::fabs(submerged_centre[0]-plane_centre[0])>.01f||
    std::fabs(submerged_centre[1]-plane_centre[1])>.01f)throw std::runtime_error("submerged hull shadow projected toward the light");
 shadow_once->Release();
 std::puts("PASS unit ground shadow union: overlapping parts darken once; Lab strength applied; submerged hulls shade their footprint");
}catch(std::exception const& e){std::printf("FAIL unit shadow union: %s\n",e.what());return 1;}}
'''


class UnitShadowUnionTests(unittest.TestCase):
    def test_layer_zero_selects_single_darkening_state(self):
        text = DIRECT.read_text()
        self.assertIn("context->OMSetDepthStencilState(shadow_once,0);", text)
        # One shadow at the shared dynamic-shadow strength animated resources use.
        self.assertIn("placement_values[14]=environment.shadow_strength*c3x_renderer::lighting::c3x_dynamic_shadow_opacity;", text)
        # Floating hulls below z=0 shade their footprint (see the GPU test).
        self.assertIn("float above=max(z,0);x+=above*pass_control.y;y+=above*pass_control.z;z=0;", text)

    def test_gpu_overlapping_parts_darken_ground_once(self):
        run_cpp(program(DIRECT.read_text()), timeout=90)


if __name__ == "__main__":
    unittest.main()
