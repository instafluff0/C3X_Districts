"""Windows-only GPU oracle for sample-preserving scrolling depth restoration.

The Windows include sends run_cpp to the Windows VM on a Mac. This test must
only be run by the current VM reservation owner; it is not a host-only check.
"""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ScrollDepthRestore(unittest.TestCase):
    def test_clear_depth_translation_and_foreground_order(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include <string>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
#define main unused_redraw_fixture_main
#include "Renderer/native/test_scene_redraw.cpp"
#undef main

void replace_all(std::string& text,char const* from,char const* to){
 std::size_t p=0;while((p=text.find(from,p))!=std::string::npos){text.replace(p,std::strlen(from),to);p+=std::strlen(to);}
}
std::string sampled(std::string text,unsigned count){
 if(count==1){
  replace_all(text,"Texture2DMS<float4,4>","Texture2D<float4>");
  replace_all(text,"Texture2DMS<float,4>","Texture2D<float>");
  replace_all(text,"color.Load(p,s)","color.Load(int3(p,0))");
  replace_all(text,"depth.Load(p,s)","depth.Load(int3(p,0))");
 }else if(count==2)replace_all(text,",4>",",2>");return text;
}

int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 verify_redraw(level>=D3D_FEATURE_LEVEL_11_0,"hardware feature level 11 required");
 constexpr unsigned w=32,h=24;
 const char* source=R"(
float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
struct O{float4 color:SV_Target;float depth:SV_Depth;};
O Seed(float4 position:SV_Position,uint s:SV_SampleIndex){int2 p=int2(position.xy);
 if(p.x<8 || (p.x+2*p.y+s)%7==0)discard;
 O o;o.color=float4((p.x+1)*.03125,(p.y+1)*.0625,(s+1)*.25,(s+1)*.125);
 o.depth=.25+s*.0625+(p.x%4)*.00390625;return o;}
O Probe(float4 position:SV_Position,uint s:SV_SampleIndex){O o;
 o.color=float4(.1875+s*.125,.375,.5625,.75);o.depth=.99609375;return o;}
)";
 ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> seed,probe;
 auto blob=redraw_shader(source,"VS","vs_5_0");checked(device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vertex));
 blob=redraw_shader(source,"Seed","ps_5_0");checked(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&seed));
 blob=redraw_shader(source,"Probe","ps_5_0");checked(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&probe));
 D3D11_DEPTH_STENCIL_DESC zd={};zd.DepthEnable=true;zd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;zd.DepthFunc=D3D11_COMPARISON_LESS;
 ComPtr<ID3D11DepthStencilState> less;checked(device->CreateDepthStencilState(&zd,&less));
 D3D11_BUFFER_DESC bd={};bd.ByteWidth=32;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
 ComPtr<ID3D11Buffer> constants;checked(device->CreateBuffer(&bd,nullptr,&constants));
 unsigned cases=0,negative_controls=0;
 for(unsigned count:{1u,2u,4u}){
  LinearRestore restore;verify_redraw(restore.ensure(device.Get(),count),"restore sample setup");
  LinearTarget retained,actual;
  for(auto target:{&retained,&actual})verify_redraw(target->ensure(device.Get(),w,h,true,false,count),"sample target allocation");
  auto draw=[&](LinearTarget& target,ID3D11PixelShader* pixel,ID3D11DepthStencilState* depth){
   context->OMSetRenderTargets(1,&target.target,target.depth);context->OMSetBlendState(nullptr,nullptr,~0u);
   context->OMSetDepthStencilState(depth,0);context->RSSetState(restore.rasterizer);
   D3D11_VIEWPORT vp={0,0,float(w),float(h),0,1};D3D11_RECT rect={0,0,LONG(w),LONG(h)};
   context->RSSetViewports(1,&vp);context->RSSetScissorRects(1,&rect);
   context->IASetInputLayout(nullptr);context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
   context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel,nullptr,0);context->Draw(3,0);context->OMSetRenderTargets(0,nullptr,nullptr);
  };
  float zero[4]={};context->ClearRenderTargetView(retained.target,zero);
  context->ClearDepthStencilView(retained.depth,D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,0);
  draw(retained,seed.Get(),restore.depth);
  auto oracle=sampled(R"(
Texture2DMS<float4,4> color:register(t0);
Texture2DMS<float,4> depth:register(t1);
Texture2DMS<float4,4> actual_color:register(t2);
Texture2DMS<float,4> actual_depth:register(t3);
cbuffer Move:register(b0){int4 move;float4 adjust;};
RWTexture2D<uint> result:register(u0);
[numthreads(8,8,1)]void CS(uint3 id:SV_DispatchThreadID){
 uint width,height;result.GetDimensions(width,height);if(id.x>=width||id.y>=height)return;
 int2 p=int2(id.xy)-move.xy;bool inside=all(p>=0)&&all(p<int2(width,height));uint errors=0;
 for(int s=0;s<move.z;++s){
  float4 expected=inside?color.Load(p,s):float4(0,0,0,0);
  float d=inside?depth.Load(p,s):1;
  float wanted=d<1?d+adjust.x:1;
  if(move.w!=0 && d==1){expected=float4(.1875+s*.125,.375,.5625,.75);wanted=.99609375;}
  float4 got=actual_color.Load(id.xy,s);float z=actual_depth.Load(id.xy,s);
  if(any(asuint(got)!=asuint(expected)))errors|=1;
  // Foreground is already quantized D24. A second conversion permits one
  // additional D24 code; clear sentinel1 is required to remain exactly1.
  if(d==1 && move.w==0){if(z!=1)errors|=2;}
  else if(abs(round(z*16777215.)-round(wanted*16777215.))>1)errors|=4;
 }
 result[id.xy]=errors;
})",count);
  if(count==1){
   replace_all(oracle,"actual_color.Load(id.xy,s)","actual_color.Load(int3(id.xy,0))");
   replace_all(oracle,"actual_depth.Load(id.xy,s)","actual_depth.Load(int3(id.xy,0))");
  }
  blob=redraw_shader(oracle.c_str(),"CS","cs_5_0");ComPtr<ID3D11ComputeShader> compute;
  checked(device->CreateComputeShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&compute));
  D3D11_TEXTURE2D_DESC td={};td.Width=w;td.Height=h;td.MipLevels=td.ArraySize=td.SampleDesc.Count=1;
  td.Format=DXGI_FORMAT_R32_UINT;td.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
  ComPtr<ID3D11Texture2D> differences,staging;ComPtr<ID3D11UnorderedAccessView> output;
  checked(device->CreateTexture2D(&td,nullptr,&differences));checked(device->CreateUnorderedAccessView(differences.Get(),nullptr,&output));
  td.BindFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&td,nullptr,&staging));
  auto differences_at=[&](int dx,int dy,float shift,bool after_probe){
   struct Values{int move[4];float adjust[4];} values={{dx,dy,int(count),after_probe?1:0},{shift,0,0,0}};
   context->UpdateSubresource(constants.Get(),0,nullptr,&values,0,0);auto cb=constants.Get();auto uav=output.Get();
   ID3D11ShaderResourceView* inputs[]={retained.samples,retained.depth_samples,actual.samples,actual.depth_samples};
   context->CSSetConstantBuffers(0,1,&cb);context->CSSetShader(compute.Get(),nullptr,0);
   context->CSSetShaderResources(0,4,inputs);context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);context->Dispatch((w+7)/8,(h+7)/8,1);
   for(auto& input:inputs)input=nullptr;uav=nullptr;
   context->CSSetShaderResources(0,4,inputs);context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);context->CSSetShader(nullptr,nullptr,0);
   context->CopyResource(staging.Get(),differences.Get());D3D11_MAPPED_SUBRESOURCE mapped={};checked(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped));
   unsigned errors=0;for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)errors+=reinterpret_cast<unsigned*>(static_cast<char*>(mapped.pData)+y*mapped.RowPitch)[x]!=0;
   context->Unmap(staging.Get(),0);return errors;
  };
  for(auto move:std::vector<std::array<int,2>>{{0,0},{3,-2},{-3,2}})for(float shift:{1.f/128,-1.f/128}){
   verify_redraw(restore.draw(context.Get(),actual,retained.samples,retained.depth_samples,move[0],move[1],{},nullptr,w,h,false,false,0,nullptr,1,shift),"production restore");
   verify_redraw(differences_at(move[0],move[1],shift,false)==0,"restore changed HDR alpha/sample, clear sentinel or translated occupied depth");
   draw(actual,probe.Get(),less.Get());
   verify_redraw(differences_at(move[0],move[1],shift,true)==0,"clear or newly exposed samples occlude far foreground; occupied sample ordering changed");++cases;
  }
  // Known failing control executes the former sentinel translation and proves
  // this oracle detects its resulting far-background occlusion.
  auto old=sampled(R"(
Texture2DMS<float4,4> color:register(t0);Texture2DMS<float,4> depth:register(t1);
struct O{float4 color:SV_Target;float depth:SV_Depth;};
O PS(float4 position:SV_Position,uint s:SV_SampleIndex){int2 p=int2(position.xy);O o;
 o.color=color.Load(p,s);o.depth=depth.Load(p,s)-.0078125;return o;}
)",count);
  blob=redraw_shader(old.c_str(),"PS","ps_5_0");ComPtr<ID3D11PixelShader> legacy;
  checked(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&legacy));
  ID3D11ShaderResourceView* inputs[]={retained.samples,retained.depth_samples};context->PSSetShaderResources(0,2,inputs);
  draw(actual,legacy.Get(),restore.depth);inputs[0]=inputs[1]=nullptr;context->PSSetShaderResources(0,2,inputs);
  verify_redraw(differences_at(0,0,-1.f/128,false)>0,"former clear-depth bug must fail sentinel comparison");
  draw(actual,probe.Get(),less.Get());
  verify_redraw(differences_at(0,0,-1.f/128,true)>0,"former clear-depth bug must reject eligible far foreground");++negative_controls;
 }
 context->ClearState();std::printf("PASS scroll depth restore: cases=%u samples_1_2_4=1 positive_negative_adjustment=1 color_alpha_sample_exact=1 clear_depth_exact=1 depth_d24_tolerance=1 foreground_order=1 legacy_negative_controls=%u\n",cases,negative_controls);return 0;
}catch(std::exception const& error){std::fprintf(stderr,"FAIL scroll depth restore: %s\n",error.what());return 1;}}
''', timeout=90)


if __name__ == '__main__':
    unittest.main()
