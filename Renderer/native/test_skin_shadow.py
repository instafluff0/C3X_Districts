"""Conservative pose bounds and the resident skin shader's self-shadow pass."""
import pathlib
import subprocess
import tempfile
import unittest
from Renderer.native.native_cpp_test import run_cpp

ROOT=pathlib.Path(__file__).resolve().parents[2]

class SkinShadowTests(unittest.TestCase):
    def test_bounds_include_weighted_pose_under_yaw_and_scale(self):
        code=r'''
#include "Renderer/native/render_core/skin_shadow_bounds.h"
#include <cassert>
using namespace c3x_renderer;
int main(){
 AnimationMesh mesh;mesh.bones=2;mesh.vertices.resize(4);
 for(unsigned i=0;i<4;++i){auto& v=mesh.vertices[i];v.source.position[0]=i&1?1.f:-1.f;
  v.source.position[1]=i&2?1.f:-1.f;v.source.position[2]=.1f;
  v.joints={0,1,0,0};v.weights={.25f,.75f,0,0};}
 render_core::SkinShadowBounds bounds;bounds.prepare(mesh);
 float palette[32]={};for(unsigned i=0;i<2;++i)for(unsigned a=0;a<4;++a)palette[i*16+a*5]=1;
 palette[14]=-1;palette[28]=2;palette[30]=1;
 for(float angle:{0.f,.4f,1.7f,3.14f})for(float scale:{.2f,1.f,2.f}){
  std::vector<UnitShadow::Point> corners;bounds.append(palette,angle,scale,.1f,corners);
  assert(corners.size()==16);UnitShadow fit(512,false);assert(fit.fit(corners,-.8f,.6f,true));
  assert(fit.heights.empty());
  for(auto const& v:mesh.vertices){
   float x=v.source.position[0]+1.5f,y=v.source.position[1],z=.7f;
   auto p=fit.project({(x*std::cos(angle)-y*std::sin(angle))*scale,
    (x*std::sin(angle)+y*std::cos(angle))*scale,z*scale});
   assert(p[0]>fit.left&&p[0]<fit.left+fit.width);
   assert(p[1]>fit.top&&p[1]<fit.top+fit.height);
  }
 }
}
'''
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);(p/'test.cpp').write_text(code)
            subprocess.run(['c++','-std=c++17','-O2','-I',str(ROOT),str(p/'test.cpp'),'-o',str(p/'test')],check=True)
            subprocess.run([str(p/'test')],check=True)

    def test_gpu_height_cutout_and_maximum(self):
        source=(ROOT/'Renderer/sandbox/direct_units.h').read_text().split('char const* source=R"(',1)[1].split(')";',1)[0]
        code=r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
#include "Renderer/native/scene_projection.h"
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 char const* source=R"SKINSHADOW(SHADER_SOURCE)SKINSHADOW";
 ComPtr<ID3DBlob> vs,ps,error;
 checked(D3DCompile(source,strlen(source),"skin",nullptr,nullptr,"VS","vs_5_0",0,0,&vs,&error));
 checked(D3DCompile(source,strlen(source),"skin",nullptr,nullptr,"PSHeight","ps_5_0",0,0,&ps,&error));
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
 struct Vertex{float p[3],n[3],uv[2],t[3],b[3];unsigned joints[4];float weights[4];};
 Vertex vertices[3]={};float xy[6]={.15f,.15f,.85f,.15f,.5f,.85f};
 for(unsigned i=0;i<3;++i){auto& v=vertices[i];v.p[0]=xy[i*2];v.p[1]=xy[i*2+1];v.p[2]=.25f;
  v.n[2]=v.t[0]=v.b[1]=1;v.uv[0]=v.uv[1]=.5f;v.joints[1]=1;v.weights[0]=v.weights[1]=.5f;}
 auto buffer=[&](void const* data,unsigned bytes,unsigned bind,unsigned stride=0){
  D3D11_BUFFER_DESC d={};d.ByteWidth=bytes;d.BindFlags=bind;
  if(stride){d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;d.StructureByteStride=stride;}
  D3D11_SUBRESOURCE_DATA s={data,0,0};ComPtr<ID3D11Buffer> b;checked(device->CreateBuffer(&d,&s,&b));return b;};
 float palette[32]={};for(unsigned j=0;j<2;++j)for(unsigned a=0;a<4;++a)palette[j*16+a*5]=1;palette[30]=.5f;
 auto vb=buffer(vertices,sizeof(vertices),D3D11_BIND_VERTEX_BUFFER);
 auto pb=buffer(palette,sizeof(palette),D3D11_BIND_SHADER_RESOURCE,16);
 ComPtr<ID3D11ShaderResourceView> palettes;checked(device->CreateShaderResourceView(pb.Get(),nullptr,&palettes));
 float settings[28]={0,0,1,1,1,0,0,0,0,2,1,0,1,0,0,0,3,0,0,1,0,0,1,1,0,0};
 auto cb=buffer(settings,sizeof(settings),D3D11_BIND_CONSTANT_BUFFER);
 D3D11_TEXTURE2D_DESC t={};t.Width=t.Height=32;t.MipLevels=t.ArraySize=t.SampleDesc.Count=1;
 t.Format=DXGI_FORMAT_R32_FLOAT;t.BindFlags=D3D11_BIND_RENDER_TARGET;
 ComPtr<ID3D11Texture2D> texture,read;checked(device->CreateTexture2D(&t,nullptr,&texture));
 ComPtr<ID3D11RenderTargetView> target;checked(device->CreateRenderTargetView(texture.Get(),nullptr,&target));
 t.BindFlags=0;t.Usage=D3D11_USAGE_STAGING;t.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&t,nullptr,&read));
 t.Width=t.Height=1;t.Format=DXGI_FORMAT_R8G8B8A8_UNORM;t.Usage=D3D11_USAGE_DEFAULT;t.CPUAccessFlags=0;t.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 unsigned rgba=0xffffffff;D3D11_SUBRESOURCE_DATA image={&rgba,4,0};ComPtr<ID3D11Texture2D> base;
 checked(device->CreateTexture2D(&t,&image,&base));ComPtr<ID3D11ShaderResourceView> base_view;checked(device->CreateShaderResourceView(base.Get(),nullptr,&base_view));
 D3D11_SAMPLER_DESC sampler={};sampler.Filter=D3D11_FILTER_MIN_MAG_MIP_POINT;sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D11_TEXTURE_ADDRESS_CLAMP;
 ComPtr<ID3D11SamplerState> sample;checked(device->CreateSamplerState(&sampler,&sample));
 D3D11_BLEND_DESC blend={};auto& rt=blend.RenderTarget[0];rt.BlendEnable=TRUE;
 rt.SrcBlend=rt.DestBlend=rt.SrcBlendAlpha=rt.DestBlendAlpha=D3D11_BLEND_ONE;rt.BlendOp=rt.BlendOpAlpha=D3D11_BLEND_OP_MAX;rt.RenderTargetWriteMask=1;
 ComPtr<ID3D11BlendState> maximum;checked(device->CreateBlendState(&blend,&maximum));
 D3D11_RASTERIZER_DESC raster={};raster.FillMode=D3D11_FILL_SOLID;raster.CullMode=D3D11_CULL_NONE;raster.DepthClipEnable=TRUE;
 ComPtr<ID3D11RasterizerState> rs;checked(device->CreateRasterizerState(&raster,&rs));
 context->RSSetState(rs.Get());D3D11_VIEWPORT viewport={0,0,32,32,0,1};context->RSSetViewports(1,&viewport);
 auto v=vb.Get(),c=cb.Get();UINT stride=sizeof(Vertex),offset=0;auto pv=palettes.Get(),bv=base_view.Get();auto ss=sample.Get();auto output=target.Get();
 context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
 context->IASetVertexBuffers(0,1,&v,&stride,&offset);context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel.Get(),nullptr,0);
 context->VSSetConstantBuffers(2,1,&c);context->PSSetConstantBuffers(2,1,&c);context->VSSetShaderResources(0,1,&pv);
 context->PSSetShaderResources(0,1,&bv);context->PSSetSamplers(0,1,&ss);context->OMSetBlendState(maximum.Get(),nullptr,~0u);
 auto draw=[&]{context->OMSetRenderTargets(1,&output,nullptr);context->Draw(3,0);context->OMSetRenderTargets(0,nullptr,nullptr);};
 auto verify=[&](float expected){context->CopyResource(read.Get(),texture.Get());D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m));unsigned covered=0;
  for(unsigned y=0;y<32;++y)for(unsigned x=0;x<32;++x){float h=reinterpret_cast<float*>(static_cast<char*>(m.pData)+y*m.RowPitch)[x];
   if(h>=0){assert(std::abs(h-expected)<1e-5f);++covered;}}
  context->Unmap(read.Get(),0);assert(expected<0?covered==0:covered>150);};
 float clear[4]={-1,-1,-1,-1};context->ClearRenderTargetView(output,clear);draw();verify(.5f);
 settings[13]=-.2f;context->UpdateSubresource(c,0,nullptr,settings,0,0);draw();verify(.5f); // MAX preserves top caster
 rgba=0;context->UpdateSubresource(base.Get(),0,nullptr,&rgba,4,0);context->ClearRenderTargetView(output,clear);draw();verify(-1);
 settings[19]=0;context->UpdateSubresource(c,0,nullptr,settings,0,0);draw();verify(.3f); // opaque owner-mask alpha isn't a cutout
 settings[13]=-1;context->UpdateSubresource(c,0,nullptr,settings,0,0);context->ClearRenderTargetView(output,clear);draw();verify(-1);
 // Exercise the actual resident body VS at both displayed scales. Its
 // footprint must grow, not just its world position or a finished bitmap.
 settings[0]=16;settings[1]=22;settings[2]=settings[3]=32;
 settings[12]=.1f;settings[13]=0;settings[16]=0;
 context->UpdateSubresource(c,0,nullptr,settings,0,0);
 unsigned widths[2]={};
 for(unsigned pass=0;pass<2;++pass){
  D3D11_VIEWPORT v={0,0,32,32,0,1};c3x_renderer::SceneProjection(32,32,pass?1.5f:1.f).viewport(v);
  context->RSSetViewports(1,&v);context->ClearRenderTargetView(output,clear);draw();
  context->CopyResource(read.Get(),texture.Get());D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&m));
  unsigned lo=32,hi=0;
  for(unsigned y=0;y<32;++y)for(unsigned x=0;x<32;++x)
   if(reinterpret_cast<float*>(static_cast<char*>(m.pData)+y*m.RowPitch)[x]>=0){lo=std::min(lo,x);hi=std::max(hi,x);}
  context->Unmap(read.Get(),0);assert(hi>=lo);widths[pass]=hi-lo+1;
 }
 assert(widths[0]>=5&&widths[1]>=widths[0]+2);
 std::puts("PASS resident skinned shadow: weighted GPU palette, height MAX, cutouts, clipping; body grows at geometry zoom");
}catch(std::exception const& e){std::printf("FAIL skin shadow: %s\n",e.what());return 1;}}
'''.replace('SHADER_SOURCE',source)
        run_cpp(code,timeout=90)

if __name__=='__main__':unittest.main()
