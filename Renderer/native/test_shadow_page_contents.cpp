// Standalone native oracle: exact retained page decisions, production caster
// shaders and production receiver PCF. No assets or game process are required.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include "render_core/shadow_page_contents.h"
#include <array>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
using Grid=c3x_renderer::render_core::ShadowSamplingGrid;
using Key=std::array<std::uint64_t,20>;
using Pages=c3x_renderer::render_core::ShadowPageContents<Key>;
void check(bool ok,char const* detail){if(!ok){std::cerr<<"SHADOW_PAGE_NATIVE_FAIL "<<detail<<"\n";std::exit(1);}}
void checked(HRESULT hr,char const* detail){check(SUCCEEDED(hr),detail);}
std::string read(char const* path){std::ifstream file(path,std::ios::binary);check(bool(file),path);return {std::istreambuf_iterator<char>(file),{}};}
struct Caster {float low[2],high[2],z;bool cutout=false;std::uint64_t generation=1;};
struct Oracle {
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 ComPtr<ID3D11VertexShader> caster_vs,screen_vs;ComPtr<ID3D11PixelShader> opaque,cutout,sample;
 ComPtr<ID3D11InputLayout> layout;ComPtr<ID3D11Buffer> settings,table,queries,vertices;
 ComPtr<ID3D11BlendState> maximum;ComPtr<ID3D11RasterizerState> raster;
 ComPtr<ID3D11Texture2D> mask,output,output_read;ComPtr<ID3D11ShaderResourceView> mask_view;
 ComPtr<ID3D11RenderTargetView> output_target;
 struct Field {Pages pages;ComPtr<ID3D11Texture2D> texture,read;ComPtr<ID3D11ShaderResourceView> view;
  std::array<ComPtr<ID3D11RenderTargetView>,Grid::max_pages> targets;};
 std::array<float,12> basis={1,0,0,0,0,1,0,0,0,0,1,0};
 unsigned raster_pages=0,draws=0;std::uint64_t compared=0;
 ComPtr<ID3DBlob> compile(std::string const& code,char const* entry,char const* profile){
  ComPtr<ID3DBlob> result,errors;auto hr=D3DCompile(code.data(),code.size(),nullptr,nullptr,nullptr,entry,profile,D3DCOMPILE_ENABLE_STRICTNESS,0,&result,&errors);
  if(FAILED(hr)&&errors)std::cerr<<static_cast<char const*>(errors->GetBufferPointer());checked(hr,entry);return result;
 }
 void buffer(ComPtr<ID3D11Buffer>& value,unsigned bytes,unsigned flags){D3D11_BUFFER_DESC desc{};desc.ByteWidth=bytes;desc.BindFlags=flags;checked(device->CreateBuffer(&desc,nullptr,&value),"buffer");}
 Oracle(){
  auto hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context);
  if(FAILED(hr))hr=D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context);checked(hr,"device");
  auto source=read("Renderer/native/environment_refresh/source_caster.hlsl");auto code=compile(source,"VS","vs_5_0");
  checked(device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&caster_vs),"caster VS");
  D3D11_INPUT_ELEMENT_DESC elements[]={{"TEXCOORD",0,DXGI_FORMAT_R32G32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
   {"TEXCOORD",1,DXGI_FORMAT_R32_FLOAT,0,8,D3D11_INPUT_PER_VERTEX_DATA,0},
   {"TEXCOORD",2,DXGI_FORMAT_R32G32B32A32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
   {"TEXCOORD",3,DXGI_FORMAT_R32_FLOAT,0,28,D3D11_INPUT_PER_VERTEX_DATA,0},
   {"TEXCOORD",4,DXGI_FORMAT_R32G32B32A32_FLOAT,0,32,D3D11_INPUT_PER_VERTEX_DATA,0}};
  checked(device->CreateInputLayout(elements,5,code->GetBufferPointer(),code->GetBufferSize(),&layout),"layout");
  for(auto pair:{std::pair<char const*,ComPtr<ID3D11PixelShader>*>("PSOpaque",&opaque),{"PSCutout",&cutout}}){code=compile(source,pair.first,"ps_5_0");checked(device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,pair.second->GetAddressOf()),pair.first);}
  auto header=read("Renderer/sandbox/fresh_pipeline.h");auto start=header.find("// Only the shadow query uses");auto end=header.find("\n)\");",start);check(start!=std::string::npos&&end!=std::string::npos,"production sampling source");
  auto sampling=header.substr(start,end-start);
  source="cbuffer Table:register(b0){float4 pickup_pages[64];};cbuffer Query:register(b1){float4 U,V,L;float4 domain;};Texture2DArray field:register(t0);\n"+sampling+R"(
float4 ScreenVS(uint n:SV_VertexID):SV_POSITION {float2 p=float2((n<<1)&2,n&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
float SamplePS(float4 p:SV_POSITION):SV_TARGET {float2 xy=domain.xy+p.xy/128*domain.zw;
 return c3x_paged_visibility(field,float4(xy,.1,1),float3(0,0,1),false,U,V,L,float4(1,0,0,0));}
)";
  code=compile(source,"ScreenVS","vs_5_0");checked(device->CreateVertexShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&screen_vs),"screen VS");
  code=compile(source,"SamplePS","ps_5_0");checked(device->CreatePixelShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&sample),"receiver PCF");
  buffer(settings,80,D3D11_BIND_CONSTANT_BUFFER);buffer(table,64*16,D3D11_BIND_CONSTANT_BUFFER);buffer(queries,64,D3D11_BIND_CONSTANT_BUFFER);buffer(vertices,6*48,D3D11_BIND_VERTEX_BUFFER);
  D3D11_BLEND_DESC blend{};auto& b=blend.RenderTarget[0];b.BlendEnable=TRUE;b.SrcBlend=b.DestBlend=b.SrcBlendAlpha=b.DestBlendAlpha=D3D11_BLEND_ONE;b.BlendOp=b.BlendOpAlpha=D3D11_BLEND_OP_MAX;b.RenderTargetWriteMask=D3D11_COLOR_WRITE_ENABLE_ALL;checked(device->CreateBlendState(&blend,&maximum),"maximum blend");
  D3D11_RASTERIZER_DESC r{};r.FillMode=D3D11_FILL_SOLID;r.CullMode=D3D11_CULL_NONE;r.DepthClipEnable=FALSE;checked(device->CreateRasterizerState(&r,&raster),"raster");
  D3D11_TEXTURE2D_DESC d{};d.Width=d.Height=2;d.ArraySize=d.MipLevels=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_R32_FLOAT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
  float alpha[]={1,0,0,1};D3D11_SUBRESOURCE_DATA initial{alpha,8,0};checked(device->CreateTexture2D(&d,&initial,&mask),"mask");checked(device->CreateShaderResourceView(mask.Get(),nullptr,&mask_view),"mask view");
  d.Width=d.Height=128;d.BindFlags=D3D11_BIND_RENDER_TARGET;checked(device->CreateTexture2D(&d,nullptr,&output),"output");checked(device->CreateRenderTargetView(output.Get(),nullptr,&output_target),"output target");d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&d,nullptr,&output_read),"output read");
 }
 void field(Field& f){D3D11_TEXTURE2D_DESC d{};d.Width=d.Height=Grid::page_texels;d.ArraySize=Grid::max_pages;d.MipLevels=d.SampleDesc.Count=1;d.Format=DXGI_FORMAT_R32_FLOAT;d.BindFlags=D3D11_BIND_RENDER_TARGET|D3D11_BIND_SHADER_RESOURCE;
  checked(device->CreateTexture2D(&d,nullptr,&f.texture),"field");checked(device->CreateShaderResourceView(f.texture.Get(),nullptr,&f.view),"field view");
  for(unsigned i=0;i<Grid::max_pages;++i){D3D11_RENDER_TARGET_VIEW_DESC target{};target.Format=d.Format;target.ViewDimension=D3D11_RTV_DIMENSION_TEXTURE2DARRAY;target.Texture2DArray.ArraySize=1;target.Texture2DArray.FirstArraySlice=i;checked(device->CreateRenderTargetView(f.texture.Get(),&target,&f.targets[i]),"page target");}
  d.ArraySize=1;d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;checked(device->CreateTexture2D(&d,nullptr,&f.read),"field read");
 }
 std::array<float,4> projected(Caster const& c){std::array<float,4> p={1e9,1e9,-1e9,-1e9};
  for(unsigned corner=0;corner<4;++corner){float x=corner&1?c.high[0]:c.low[0],y=corner&2?c.high[1]:c.low[1];float u=x*basis[0]+y*basis[1]+c.z*basis[2],v=x*basis[4]+y*basis[5]+c.z*basis[6];p[0]=std::min(p[0],u);p[1]=std::min(p[1],v);p[2]=std::max(p[2],u);p[3]=std::max(p[3],v);}return p;}
 bool intersects(Caster const& c,Grid const& g,unsigned i){auto p=projected(c);auto b=g.page_box(i);return !(p[2]<b[0]||p[0]>b[0]+b[2]||p[3]<b[1]||p[1]>b[1]+b[3]);}
 Key key(Caster const& c){Key value{c.generation,unsigned(c.cutout)};std::memcpy(value.data()+2,&c.low,sizeof(c.low));std::memcpy(value.data()+3,&c.high,sizeof(c.high));std::memcpy(value.data()+4,&c.z,4);return value;}
 Pages::Inputs proofs(Grid const& g,std::vector<Caster> const& casters){Pages::Inputs facts;for(unsigned i=0;i<g.pages();++i){
  for(auto const& c:casters)if(intersects(c,g,i)){facts[i].push_back(key(c));}std::sort(facts[i].begin(),facts[i].end());}return facts;}
 void render(Field& f,Grid const& g,Pages::Context const& proof_context,std::vector<Caster> const& casters,bool forced){
  Pages::Inputs facts;
  if(forced){f.pages.clear();facts=proofs(g,casters);f.pages.select(g,proof_context,facts,true);}
  else {f.pages.begin_incremental(g,proof_context,basis);
   for(auto const& c:casters)check(f.pages.update(key(c),g,[&]{return projected(c);},[](auto bytes){return bytes<=16u*1024u*1024u;}),"incremental admission");
   check(f.pages.finish_incremental(g,true,[](auto){return true;}),"incremental membership");}
  ID3D11ShaderResourceView* empty=nullptr;context->PSSetShaderResources(0,1,&empty);context->VSSetShader(caster_vs.Get(),nullptr,0);context->IASetInputLayout(layout.Get());context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);context->RSSetState(raster.Get());context->OMSetDepthStencilState(nullptr,0);context->OMSetBlendState(maximum.Get(),nullptr,~0u);
  D3D11_VIEWPORT vp{0,0,float(Grid::page_texels),float(Grid::page_texels),0,1};context->RSSetViewports(1,&vp);ID3D11Buffer* cb=settings.Get();context->VSSetConstantBuffers(0,1,&cb);ID3D11ShaderResourceView* mask_resource=mask_view.Get();context->PSSetShaderResources(33,1,&mask_resource);
  for(unsigned i=0;i<g.pages();++i){if(f.pages.reused[i])continue;auto* target=f.targets[f.pages.slots[i]].Get();float clear[]={-1e6,-1e6,-1e6,-1e6};context->ClearRenderTargetView(target,clear);context->OMSetRenderTargets(1,&target,nullptr);++raster_pages;
   for(auto const& c:casters){if(!intersects(c,g,i))continue;
    float values[20]{};std::copy(basis.begin(),basis.end(),values);for(unsigned k=0;k<3;++k){values[k]*=6/g.page_span(0);values[4+k]*=6/g.page_span(1);}auto coord=g.page(i);values[12]=float(coord[0]);values[13]=float(coord[1]);context->UpdateSubresource(settings.Get(),0,nullptr,values,0,0);
    float points[6][12]{};unsigned corners[]={0,1,2,2,1,3};for(unsigned n=0;n<6;++n){unsigned corner=corners[n];points[n][0]=corner&1?1.f:0.f;points[n][1]=corner&2?1.f:0.f;points[n][2]=40;points[n][3]=corner&1?c.high[0]:c.low[0];points[n][4]=corner&2?c.high[1]:c.low[1];points[n][5]=c.z;points[n][6]=1;points[n][7]=1;}
    context->UpdateSubresource(vertices.Get(),0,nullptr,points,0,0);ID3D11Buffer* stream=vertices.Get();UINT stride=48,offset=0;context->IASetVertexBuffers(0,1,&stream,&stride,&offset);context->PSSetShader(c.cutout?cutout.Get():opaque.Get(),nullptr,0);context->Draw(6,0);++draws;
   }check(forced?f.pages.complete(i,g,proof_context,facts[i]):f.pages.complete_incremental(i),"complete");
   auto exact=forced?facts[i]:proofs(g,casters)[i];
   check(f.pages.exact_inputs(f.pages.slots[i],exact),"compact membership resolves every exact caster key");
  }context->OMSetRenderTargets(0,nullptr,nullptr);
 }
 std::vector<unsigned char> depth(Field& f,unsigned physical){context->CopySubresourceRegion(f.read.Get(),0,0,0,0,f.texture.Get(),physical,nullptr);D3D11_MAPPED_SUBRESOURCE mapped{};checked(context->Map(f.read.Get(),0,D3D11_MAP_READ,0,&mapped),"depth map");std::vector<unsigned char> pixels(Grid::page_texels*Grid::page_texels*4);
  for(unsigned y=0;y<Grid::page_texels;++y)std::memcpy(pixels.data()+std::size_t(y)*Grid::page_texels*4,static_cast<unsigned char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,Grid::page_texels*4);context->Unmap(f.read.Get(),0);return pixels;}
 std::vector<unsigned char> receiver(Field& f,Grid const& g,std::array<float,4> wrap){std::array<std::array<float,4>,64> values{};values[0]={float(g.low[0]),float(g.low[1]),float(g.count[0]),float(g.count[1])};values[1]=wrap;values[2]={g.inverse_pitch(0),g.inverse_pitch(1),g.page_span(0),g.page_span(1)};for(unsigned i=0;i<g.pages();++i)values[3+i][0]=float(f.pages.slots[i]);context->UpdateSubresource(table.Get(),0,nullptr,values.data(),0,0);
  float query[16]{};std::copy(basis.begin(),basis.end(),query);auto box=g.coverage();std::copy(box.begin(),box.end(),query+12);context->UpdateSubresource(queries.Get(),0,nullptr,query,0,0);
  context->OMSetBlendState(nullptr,nullptr,~0u);ID3D11RenderTargetView* target=output_target.Get();context->OMSetRenderTargets(1,&target,nullptr);D3D11_VIEWPORT vp{0,0,128,128,0,1};context->RSSetViewports(1,&vp);context->IASetInputLayout(nullptr);context->VSSetShader(screen_vs.Get(),nullptr,0);context->PSSetShader(sample.Get(),nullptr,0);ID3D11Buffer* cb[]={table.Get(),queries.Get()};context->PSSetConstantBuffers(0,2,cb);ID3D11ShaderResourceView* srv=f.view.Get();context->PSSetShaderResources(0,1,&srv);context->Draw(3,0);context->OMSetRenderTargets(0,nullptr,nullptr);srv=nullptr;context->PSSetShaderResources(0,1,&srv);
  context->CopyResource(output_read.Get(),output.Get());D3D11_MAPPED_SUBRESOURCE mapped{};checked(context->Map(output_read.Get(),0,D3D11_MAP_READ,0,&mapped),"receiver map");std::vector<unsigned char> pixels(128*128*4);for(unsigned y=0;y<128;++y)std::memcpy(pixels.data()+y*128*4,static_cast<unsigned char const*>(mapped.pData)+y*mapped.RowPitch,128*4);context->Unmap(output_read.Get(),0);return pixels;
 }
 void compare(Field& warm,Field& cold,Grid const& g,std::array<float,4> wrap){bool nonblank=false;
  for(unsigned i=0;i<g.pages();++i){auto a=depth(warm,warm.pages.slots[i]),b=depth(cold,cold.pages.slots[i]);check(a==b,"retained vs rebuild depth");for(std::size_t j=0;j<a.size();j+=4){float z;std::memcpy(&z,a.data()+j,4);nonblank|=z>-.5f;}compared+=Grid::page_texels*Grid::page_texels;}
  check(nonblank,"nonblank depth");auto a=receiver(warm,g,wrap),b=receiver(cold,g,wrap);check(a==b,"retained vs rebuild PCF");bool shadowed=false,lit=false;
  for(std::size_t i=0;i<a.size();i+=4){float value;std::memcpy(&value,a.data()+i,4);check(value>=0&&value<=1,"PCF range");shadowed|=value<.5f;lit|=value>.5f;}
  check(shadowed&&lit,"nontrivial receiver PCF");compared+=128*128;
 }
};
int main(){
 Oracle oracle;Oracle::Field warm,cold;oracle.field(warm);oracle.field(cold);Grid grid;grid.valid=true;grid.quality_span={40,40};grid.low={-1,-1};grid.count={2,2};
 Pages::Context context={1,2,3,1};std::array<float,4> wrap{};
 std::vector<Caster> casters={{{-6,-6},{4,5},.6f,false,1},{{8,-4},{14,4},.9f,true,2}};
 auto phase=[&](char const* name,unsigned minimum_hits,unsigned maximum_hits){auto hits=warm.pages.hits,rebuilds=warm.pages.rebuilt;unsigned before=oracle.draws;
  oracle.render(warm,grid,context,casters,false);auto reused=unsigned(warm.pages.hits-hits);check(reused>=minimum_hits&&reused<=maximum_hits,"expected affected pages");oracle.render(cold,grid,context,casters,true);oracle.compare(warm,cold,grid,wrap);
  std::cout<<"SHADOW_PAGE_PHASE name="<<name<<" reused="<<reused<<" rebuilt="<<warm.pages.rebuilt-rebuilds<<" draws="<<oracle.draws-before<<" compared_pixels="<<oracle.compared<<"\n";
 };
 phase("cold",0,0);phase("same",4,4);grid.low[0]=0;phase("camera-shift",2,2);
 casters.push_back({{12,2},{17,7},1.2f,false,3});phase("entering-offscreen",3,3);
 casters.erase(casters.begin()+2);phase("removal",3,3);
 casters[1].cutout=false;++casters[1].generation;phase("cutout-variant",0,3);
 oracle.basis[0]=.8f;oracle.basis[1]=.6f;oracle.basis[4]=-.6f;oracle.basis[5]=.8f;++context[7];phase("light-season",0,0);
 wrap={0,0,32,24};++context[19];phase("wrap",0,0);++context[1];phase("asset-config",0,0);++context[2];phase("device-generation",0,0);++context[0];phase("content-scope",0,0);
 grid.quality_span={48,48};++context[5];phase("current-quality",0,0);
 check(warm.pages.hits>0 && warm.pages.rebuilt<cold.pages.rebuilt,"structural reuse");
 std::cout<<"SHADOW_PAGE_NATIVE_PASS compared_pixels="<<oracle.compared<<" projections="<<warm.pages.projections<<" projection_reuses="<<warm.pages.projection_reuses<<" page_tests="<<warm.pages.page_tests<<" contributor_edits="<<warm.pages.contributor_edits<<" retained_hits="<<warm.pages.hits<<" retained_rebuilds="<<warm.pages.rebuilt<<" forced_rebuilds="<<cold.pages.rebuilt<<"\n";
}
