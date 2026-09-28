"""Exercise production GPU borders against zoom, ownership and foreground depth."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class TerritoryBorderTests(unittest.TestCase):
    def test_gpu_color_zoom_wrap_and_occlusion(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/native/test_retained_composition.cpp"
#include "Renderer/native/render_core/linear_target.h"
#include "Renderer/native/gpu_territory_borders.h"
#include <array>
struct Chunk {ID3D11Buffer* buffer=nullptr,*indices=nullptr;unsigned vertex_stride=92,vertex_offset=0,index_count=6,index_offset=0;
 DXGI_FORMAT index_format=DXGI_FORMAT_R32_UINT;};
struct Record {Chunk mesh;unsigned territory_edges=15,territory_rgb=0x20c080;int translation_x=0,translation_y=0;
 float natural_projection[4]={0,0,128,192};Chunk const& content()const{return mesh;}};
struct Settings {float translation[2]={68,64},inverse_size[2]={1.f/264,1.f/200},depth_translation=0;};
float half(unsigned h){return (h&0x7fff)?std::ldexp(float((h&1023)+((h&0x7c00)?1024:0)),int((h>>10)&31)-25+(!(h&0x7c00))):0;}
int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL level;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
 float vertices[4][23]={};float xy[4][2]={{0,0},{1,0},{1,1},{0,1}};
 for(unsigned i=0;i<4;++i){vertices[i][3]=xy[i][0];vertices[i][4]=xy[i][1];vertices[i][5]=2.5f/112;}
 unsigned indices[]={0,1,2,0,2,3};D3D11_BUFFER_DESC desc={};desc.ByteWidth=sizeof(vertices);desc.BindFlags=D3D11_BIND_VERTEX_BUFFER;
 D3D11_SUBRESOURCE_DATA data={vertices,0,0};ComPtr<ID3D11Buffer> vb,ib;
 checked(device->CreateBuffer(&desc,&data,&vb));desc.ByteWidth=sizeof(indices);desc.BindFlags=D3D11_BIND_INDEX_BUFFER;data.pSysMem=indices;
 checked(device->CreateBuffer(&desc,&data,&ib));
 std::array<Record,1> records;records[0].mesh.buffer=vb.Get();records[0].mesh.indices=ib.Get();
 Settings settings;c3x_renderer::TerritoryBorders borders;
 for(unsigned count:{1u,2u,4u}){
  c3x_renderer::render_core::LinearTarget target;assert(target.ensure(device.Get(),264,200,true,true,count));
  D3D11_TEXTURE2D_DESC td={};target.resolved->GetDesc(&td);td.BindFlags=0;td.Usage=D3D11_USAGE_STAGING;td.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
  ComPtr<ID3D11Texture2D> read;checked(device->CreateTexture2D(&td,nullptr,&read));
  auto render=[&](float depth,float zoom){
   float clear[4]={};context->ClearRenderTargetView(target.target,clear);context->ClearDepthStencilView(target.depth,D3D11_CLEAR_DEPTH,depth,0);
   assert(borders.draw(device.Get(),context.Get(),records,settings,target,256,192,zoom,1.f,[](auto const&){return true;}));
   if(count==1)context->CopyResource(target.resolved,target.color);
   else context->ResolveSubresource(target.resolved,0,target.color,0,DXGI_FORMAT_R16G16B16A16_FLOAT);
   context->CopyResource(read.Get(),target.resolved);D3D11_MAPPED_SUBRESOURCE mapped={};checked(context->Map(read.Get(),0,D3D11_MAP_READ,0,&mapped));
   std::array<double,6> sums{};
   for(unsigned y=0;y<200;++y)for(unsigned x=0;x<264;++x){auto p=reinterpret_cast<unsigned short const*>(static_cast<char*>(mapped.pData)+y*mapped.RowPitch)+4*x;
    for(unsigned j=0;j<4;++j)sums[j]+=half(p[j]);sums[4]+=x*half(p[3]);sums[5]+=y*half(p[3]);}
   context->Unmap(read.Get(),0);return sums;
  };
  records[0].territory_edges=15;records[0].territory_rgb=0x20c080;records[0].translation_x=0;
  auto shown=render(1,1),hidden=render(.3f,1);assert(shown[3]>100&&shown[1]>shown[2]&&shown[2]>shown[0]);
  assert(std::abs(hidden[3]/shown[3]-.34)<.002);
  auto zoomed=render(1,1.25f);assert(zoomed[3]>shown[3]*1.4&&zoomed[3]<shown[3]*1.7);
  records[0].translation_x=16;auto moved=render(1,1);assert(std::abs(moved[3]-shown[3])<.01);
  assert(std::abs(moved[4]/moved[3]-shown[4]/shown[3]-16)<.01);
  records[0].territory_rgb=0xff2010;auto captured=render(1,1);assert(captured[0]>captured[1]*10);
  records[0].territory_edges=0;auto removed=render(1,1);assert(removed[3]==0);
 }
 std::puts("PASS GPU borders: native colors, ownership removal/capture, zoom, translated occurrence, translucent occlusion, 1/2/4 samples");
 }catch(std::exception const&e){std::printf("FAIL borders: %s\n",e.what());return 1;}
}
''', timeout=120)


if __name__ == '__main__':
    unittest.main()
