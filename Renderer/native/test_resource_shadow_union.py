"""Animated-resource ground shadows: a leg over the body darkens the ground once.

Every posed triangle of an animated resource is blended as its own ground
footprint. Without a per-pixel guard, a leg's footprint over the body's
footprint darkened the ground again, a darker patch inside the shadow. As for
unit ground shadows, the pass clears the stencil and keeps the first fragment
at each pixel. This GPU oracle runs the actual state and clear code of both
resource-shadow submitters and checks a uniform alpha in both shaders.
"""
import pathlib
import unittest

from Renderer.native.native_cpp_test import run_cpp

ROOT = pathlib.Path(__file__).resolve().parents[2]
FRESH = ROOT / "Renderer/sandbox/fresh_pipeline.h"
LEGACY = ROOT / "Renderer/native/c3x_renderer.cpp"
SHADERS = (ROOT / "Renderer/native/city_fidelity/resource_shadow.hlsl",
           ROOT / "Renderer/native/city_fidelity/feature.hlsl")


def guard_block(text):
    start = text.index("if(!resource_shadow_once){")
    end = text.index("OMSetDepthStencilState(resource_shadow_once,0);", start)
    return text[start:end] + "OMSetDepthStencilState(resource_shadow_once,0);"


def program():
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
struct Work{void clear(ID3D11View*){}} work;
struct Linear{ID3D11DepthStencilView* depth=nullptr;};
bool fresh_guard(ID3D11DeviceContext* context,ID3D11DepthStencilView* depth,ID3D11DepthStencilState*& resource_shadow_once){
 ''' + guard_block(FRESH.read_text()) + r'''
 return true;
}
bool legacy_guard(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11DepthStencilState* depth_state,
                  Linear& linear,ID3D11DepthStencilState*& resource_shadow_once){
 ''' + guard_block(LEGACY.read_text()) + r'''
 return true;
}
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 renderer.device=device.Get();
 // Two posed triangles sharing ground, as a leg over a body does. Uniform
 // shadow alpha .4 over mid-grey linear radiance.
 char const* source="float4 VS(float3 p:POSITION):SV_Position{return float4(p,1);}\n"
  "float4 PS():SV_Target{return float4(.008,.011,.016,.4);}";
 ComPtr<ID3DBlob> vs,ps,error;
 checked(D3DCompile(source,strlen(source),"resource",nullptr,nullptr,"VS","vs_5_0",0,0,&vs,&error));
 checked(D3DCompile(source,strlen(source),"resource",nullptr,nullptr,"PS","ps_5_0",0,0,&ps,&error));
 ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> pixel;
 checked(device->CreateVertexShader(vs->GetBufferPointer(),vs->GetBufferSize(),nullptr,&vertex));
 checked(device->CreatePixelShader(ps->GetBufferPointer(),ps->GetBufferSize(),nullptr,&pixel));
 D3D11_INPUT_ELEMENT_DESC element={"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0};
 ComPtr<ID3D11InputLayout> layout;checked(device->CreateInputLayout(&element,1,vs->GetBufferPointer(),vs->GetBufferSize(),&layout));
 float triangles[18]={-.8f,-.8f,.5f, .6f,-.8f,.5f, -.8f,.6f,.5f, -.6f,-.6f,.5f, .8f,-.6f,.5f, -.6f,.8f,.5f};
 D3D11_BUFFER_DESC bd={};bd.ByteWidth=sizeof(triangles);bd.BindFlags=D3D11_BIND_VERTEX_BUFFER;
 D3D11_SUBRESOURCE_DATA init={triangles,0,0};ComPtr<ID3D11Buffer> vb;checked(device->CreateBuffer(&bd,&init,&vb));
 D3D11_TEXTURE2D_DESC t={};t.Width=t.Height=32;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
 t.Format=DXGI_FORMAT_R16G16B16A16_FLOAT;t.BindFlags=D3D11_BIND_RENDER_TARGET;
 ComPtr<ID3D11Texture2D> color,read;checked(device->CreateTexture2D(&t,nullptr,&color));
 ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(color.Get(),nullptr,&target));
 t.BindFlags=0;t.Usage=D3D11_USAGE_STAGING;t.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&t,nullptr,&read));
 t.Usage=D3D11_USAGE_DEFAULT;t.CPUAccessFlags=0;t.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;t.BindFlags=D3D11_BIND_DEPTH_STENCIL;
 ComPtr<ID3D11Texture2D> depth;checked(device->CreateTexture2D(&t,nullptr,&depth));
 ComPtr<ID3D11DepthStencilView> depth_view;checked(device->CreateDepthStencilView(depth.Get(),nullptr,&depth_view));
 // Production decal/ground tests: no depth write, test against terrain.
 D3D11_DEPTH_STENCIL_DESC decal={};decal.DepthEnable=TRUE;decal.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;decal.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
 ComPtr<ID3D11DepthStencilState> decal_depth,depth_state;checked(device->CreateDepthStencilState(&decal,&decal_depth));
 decal.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;checked(device->CreateDepthStencilState(&decal,&depth_state));
 renderer.natural.decal_depth=decal_depth.Get();
 D3D11_BLEND_DESC blend={};auto& rt=blend.RenderTarget[0];rt.BlendEnable=TRUE;
 rt.SrcBlend=D3D11_BLEND_SRC_ALPHA;rt.DestBlend=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOp=D3D11_BLEND_OP_ADD;
 rt.SrcBlendAlpha=D3D11_BLEND_ONE;rt.DestBlendAlpha=D3D11_BLEND_INV_SRC_ALPHA;rt.BlendOpAlpha=D3D11_BLEND_OP_ADD;
 rt.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;ComPtr<ID3D11BlendState> over;checked(device->CreateBlendState(&blend,&over));
 D3D11_RASTERIZER_DESC raster={};raster.FillMode=D3D11_FILL_SOLID;raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=TRUE;
 ComPtr<ID3D11RasterizerState> rs;checked(device->CreateRasterizerState(&raster,&rs));
 D3D11_VIEWPORT viewport={0,0,32,32,0,1};auto output=target.Get();auto v=vb.Get();UINT stride=12,offset=0;
 // guard: 0 plain decal depth, 1 fresh pipeline, 2 legacy region submitter.
 auto render=[&](int guard,float* overlap,float* single){
  float clear[4]={.5f,.5f,.5f,1};context->ClearRenderTargetView(output,clear);
  // A previous pass left stencil marks; the shadow pass must not inherit them.
  context->ClearDepthStencilView(depth_view.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,7);
  context->RSSetState(rs.Get());context->RSSetViewports(1,&viewport);
  context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->IASetVertexBuffers(0,1,&v,&stride,&offset);
  context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
  context->OMSetRenderTargets(1,&output,depth_view.Get());context->OMSetBlendState(over.Get(),nullptr,~0u);
  ID3D11DepthStencilState* state=nullptr;Linear linear{depth_view.Get()};
  if(guard==0)context->OMSetDepthStencilState(decal_depth.Get(),0);
  else if(guard==1){if(!fresh_guard(context.Get(),depth_view.Get(),state))throw std::runtime_error("fresh guard");}
  else if(!legacy_guard(device.Get(),context.Get(),depth_state.Get(),linear,state))throw std::runtime_error("legacy guard");
  context->Draw(3,0);context->Draw(3,3); // body, then a leg over it
  if(state)state->Release();
  context->OMSetRenderTargets(0,nullptr,nullptr);context->CopyResource(read.Get(),color.Get());
  D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m));
  auto at=[&](unsigned x,unsigned y){
   auto h=reinterpret_cast<unsigned short*>(static_cast<char*>(m.pData)+y*m.RowPitch)[x*4];
   unsigned exponent=(h>>10)&31,mantissa=h&1023;return std::ldexp(float(mantissa|1024),int(exponent)-25);};
  *overlap=at(10,21);*single=at(3,28);
  context->Unmap(read.Get(),0);};
 float stacked,stacked_single,fresh,fresh_single,legacy,legacy_single;
 render(0,&stacked,&stacked_single);render(1,&fresh,&fresh_single);render(2,&legacy,&legacy_single);
 float once=.5f+(.008f-.5f)*.4f,twice=once+(.008f-once)*.4f;
 std::printf("overlap stacked=%.4f fresh=%.4f legacy=%.4f single=%.4f/%.4f/%.4f once=%.4f\n",
  stacked,fresh,legacy,stacked_single,fresh_single,legacy_single,once);
 // The control reproduces the reported defect: the overlap darkens twice.
 if(std::fabs(stacked-twice)>.003f || std::fabs(stacked_single-once)>.003f)
  throw std::runtime_error("control did not reproduce per-triangle compounding");
 if(std::fabs(fresh-once)>.003f || std::fabs(fresh_single-once)>.003f)
  throw std::runtime_error("fresh pipeline resource shadow darkened overlap twice or lost stale-stencil pixels");
 if(std::fabs(legacy-once)>.003f || std::fabs(legacy_single-once)>.003f)
  throw std::runtime_error("legacy resource shadow darkened overlap twice or lost stale-stencil pixels");
 std::puts("PASS resource ground shadow union: overlapping posed triangles darken the ground once");
 return 0;
}catch(std::exception const& e){std::printf("FAIL resource shadow union: %s\n",e.what());return 1;}}
'''


class ResourceShadowUnionTests(unittest.TestCase):
    def test_shadow_alpha_is_uniform_inside_the_silhouette(self):
        # A first-fragment stencil is order independent only for one alpha.
        for path in SHADERS:
            text = path.read_text()
            branch = text.index("Animated resources project their current posed source triangles.")
            body = text[branch:text.index("return float4(0.008, 0.011, 0.016, alpha);", branch)]
            alpha = body[body.index("float alpha ="):]
            self.assertNotIn("coverage", alpha, path.name)
            self.assertIn("clip(coverage - 0.08);", body, path.name)

    def test_body_pass_restores_ordinary_depth(self):
        # Bodies after the shadow must not test against the shadow's stencil.
        fresh = FRESH.read_text()
        draw = fresh[fresh.index("bool draw_resource_poses("):]
        self.assertLess(draw.index("OMSetDepthStencilState(resource_shadow_once,0);"),
                        draw.index("if(!draw_features(inputs,"))
        features = fresh[fresh.index("bool draw_features("):]
        self.assertLess(features.index("aquatic_depth:renderer.depth_state,0);"), features.index("issue_records("))
        legacy = LEGACY.read_text()
        region = legacy[legacy.index("OMSetDepthStencilState(resource_shadow_once,0);"):]
        self.assertTrue(region.split("draw_cached_geometry(geometry_shadow", 1)[1]
                        .split("\n", 2)[1].strip() == "context->OMSetDepthStencilState(depth_state,0);")

    def test_gpu_overlapping_triangles_darken_ground_once(self):
        run_cpp(program(), timeout=90)


if __name__ == "__main__":
    unittest.main()
