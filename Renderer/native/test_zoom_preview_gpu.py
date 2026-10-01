"""Completed-map filtering, coverage and reprojected depth on the WARP adapter."""
import unittest
from Renderer.native.native_cpp_test import run_cpp


class ZoomPreviewGpuTests(unittest.TestCase):
    def test_original_sources_relative_zoom_and_guarded_depth(self):
        run_cpp(r'''
#define NOMINMAX
#include <windows.h>
#include "Renderer/sandbox/zoom_preview_gpu.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <limits>
#include <vector>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using Microsoft::WRL::ComPtr;
using c3x_renderer::sandbox::CompletedMapPreview;
void check(HRESULT h){assert(SUCCEEDED(h));}
struct Output {
 unsigned width,height;
 ComPtr<ID3D11Texture2D> color,depth;
 ComPtr<ID3D11RenderTargetView> target;
 ComPtr<ID3D11DepthStencilView> dsv;
 Output(ID3D11Device* device,unsigned w,unsigned h):width(w),height(h){
  D3D11_TEXTURE2D_DESC d={};d.Width=w;d.Height=h;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
  d.Format=DXGI_FORMAT_B8G8R8A8_UNORM;d.BindFlags=D3D11_BIND_RENDER_TARGET;
  check(device->CreateTexture2D(&d,nullptr,&color));check(device->CreateRenderTargetView(color.Get(),nullptr,&target));
  d.Format=DXGI_FORMAT_R24G8_TYPELESS;d.BindFlags=D3D11_BIND_DEPTH_STENCIL;
  check(device->CreateTexture2D(&d,nullptr,&depth));
  D3D11_DEPTH_STENCIL_VIEW_DESC ds={};ds.Format=DXGI_FORMAT_D24_UNORM_S8_UINT;ds.ViewDimension=D3D11_DSV_DIMENSION_TEXTURE2D;
  check(device->CreateDepthStencilView(depth.Get(),&ds,&dsv));
 }
};
std::vector<unsigned> read(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* input){
 D3D11_TEXTURE2D_DESC d={};input->GetDesc(&d);d.BindFlags=d.MiscFlags=0;
 d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Texture2D> staging;check(device->CreateTexture2D(&d,nullptr,&staging));
 context->CopyResource(staging.Get(),input);D3D11_MAPPED_SUBRESOURCE map={};
 check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&map));std::vector<unsigned> result(d.Width*d.Height);
 for(unsigned y=0;y<d.Height;++y)std::memcpy(result.data()+y*d.Width,
  static_cast<unsigned char const*>(map.pData)+std::size_t(y)*map.RowPitch,d.Width*4);
 context->Unmap(staging.Get(),0);return result;
}
std::vector<unsigned> pattern(unsigned w,unsigned h,bool close_marker=false){
 std::vector<unsigned> result(w*h);
 for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
  result[y*w+x]=((x+y)&1)?0xffc03090u:0xff2080d0u;
  // An off-center attachment marker remains visibly attached to the map.
  unsigned cx=w/2+(close_marker?10:8),cy=h/2+(close_marker?5:4);
  if(x>=cx-2&&x<cx+3&&y>=cy-2&&y<cy+3)result[y*w+x]=0xff00ff00u;
 }
 return result;
}
std::vector<unsigned> depth_pattern(unsigned w,unsigned h){
 std::vector<unsigned> result((w+8)*(h+8));
 for(unsigned y=0;y<h+8;++y)for(unsigned x=0;x<w+8;++x)
  result[y*(w+8)+x]=0x4d000000u|((x*773u+y*1931u+0x200000u)&0xffffffu);
 return result;
}
ComPtr<ID3D11Texture2D> depth_texture(ID3D11Device* device,unsigned w,unsigned h,std::vector<unsigned> const& values){
 D3D11_TEXTURE2D_DESC d={};d.Width=w+8;d.Height=h+8;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
 d.Format=DXGI_FORMAT_R24G8_TYPELESS;d.BindFlags=D3D11_BIND_SHADER_RESOURCE;
 D3D11_SUBRESOURCE_DATA data={values.data(),(w+8)*4,0};ComPtr<ID3D11Texture2D> out;
 check(device->CreateTexture2D(&d,&data,&out));return out;
}
void fill(ID3D11DeviceContext* context,CompletedMapPreview& preview,unsigned slot,unsigned w,
 std::vector<unsigned> const& values){
 ComPtr<ID3D11Resource> color;preview.target(slot)->GetResource(&color);
 context->UpdateSubresource(color.Get(),0,nullptr,values.data(),w*4,0);
}
unsigned expected(std::vector<unsigned> const& pixels,unsigned w,unsigned h,float x,float y,bool identity){
 if(x<0||y<0||x>=float(w)||y>=float(h))return 0;
 if(identity)return pixels[unsigned(y)*w+unsigned(x)];
 float px=x-.5f,py=y-.5f;int lx=int(std::floor(px)),ly=int(std::floor(py));
 float fx=px-float(lx),fy=py-float(ly);
 auto pixel=[&](int atx,int aty){return pixels[unsigned(std::clamp(aty,0,int(h)-1))*w+unsigned(std::clamp(atx,0,int(w)-1))];};
 unsigned out=0;
 for(unsigned shift:{0u,8u,16u,24u}){
  auto channel=[&](int atx,int aty){return float((pixel(atx,aty)>>shift)&255u);};
  float top=channel(lx,ly)*(1-fx)+channel(lx+1,ly)*fx;
  float bottom=channel(lx,ly+1)*(1-fx)+channel(lx+1,ly+1)*fx;
  out|=unsigned(std::lround(top*(1-fy)+bottom*fy))<<shift;
 }
 return out;
}
void compare(std::vector<unsigned> const& actual,std::vector<unsigned> const& source,unsigned sw,unsigned sh,
 unsigned ow,unsigned oh,float ratio){
 for(unsigned y=0;y<oh;++y)for(unsigned x=0;x<ow;++x){
  float sx=(float(x)+.5f-float(ow/2))/ratio+float(sw/2);
  float sy=(float(y)+.5f-float(oh/2))/ratio+float(sh/2);
  unsigned want=expected(source,sw,sh,sx,sy,ratio==1.f),got=actual[y*ow+x];
  for(unsigned shift:{0u,8u,16u,24u})assert(std::abs(int((got>>shift)&255)-int((want>>shift)&255))<=(ratio==1.f?0:2));
 }
}
int main(){
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 check(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_WARP,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context));
 CompletedMapPreview preview;Output output(device.Get(),40,32);
 auto wide=pattern(40,32),wide_depth=depth_pattern(40,32);
 auto original=depth_texture(device.Get(),40,32,wide_depth);
 assert(preview.prepare(device.Get(),0,40,32));fill(context.Get(),preview,0,40,wide);
 assert(preview.commit(context.Get(),0,original.Get(),1.f,1.f,17));
 assert(preview.bytes()==(40u*32u+48u*40u)*4u);
 assert(preview.width(0)==40&&preview.height(0)==32&&preview.completed_zoom(0)==1.f);
 auto draw=[&](unsigned slot,float zoom,bool depth=false){
  assert(preview.select(slot,zoom,17));assert(preview.draw(device.Get(),context.Get(),output.target.Get(),40,32,depth?output.dsv.Get():nullptr));
  return read(device.Get(),context.Get(),output.color.Get());
 };
 auto identity=draw(0,1.f);assert(identity==wide);
 auto enlarged=draw(0,1.25f);compare(enlarged,wide,40,32,40,32,1.25f);
 assert(enlarged[22*40+30]==0xff00ff00u); // canonical marker at (28,20) projects to (30,21)
 for(float zoom:{1.05f,1.5f,1.125f,1.f,1.25f}){
  auto frame=draw(0,zoom);compare(frame,wide,40,32,40,32,zoom);
  if(zoom==1.25f)assert(frame==enlarged); // no repeated resampling through reversals
 }
 // A larger valid completed map supplies actual surrounding pixels when the
 // relative scale is below one. This is not permission to stretch a close crop.
 auto large=pattern(64,48),large_depth=depth_pattern(64,48);
 auto large_original=depth_texture(device.Get(),64,48,large_depth);
 assert(preview.prepare(device.Get(),1,64,48));fill(context.Get(),preview,1,64,large);
 assert(preview.commit(context.Get(),1,large_original.Get(),1.25f,1.f,17));
 auto reduced_frame=draw(1,1.f);assert(std::abs(preview.relative_zoom()-.8f)<.00001f);
 compare(reduced_frame,large,64,48,40,32,.8f);
 // The original D24 input can subsequently change. The committed depth remains
 // its copied snapshot, and the witness reads nearest depth at the same affine coordinate.
 std::vector<unsigned> changed(72u*56u,0x00ffffffu);
 auto changed_depth=depth_texture(device.Get(),64,48,changed);
 context->CopyResource(large_original.Get(),changed_depth.Get());
 context->ClearDepthStencilView(output.dsv.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,1,123);
 auto witness=draw(1,1.f,true);assert(witness==reduced_frame);
 auto packed=read(device.Get(),context.Get(),output.depth.Get());
 for(unsigned y=0;y<32;++y)for(unsigned x=0;x<40;++x){
  unsigned sx=unsigned((float(x)+.5f-20.f)/.8f+32.f)+4;
  unsigned sy=unsigned((float(y)+.5f-16.f)/.8f+24.f)+4;
  unsigned want=large_depth[sy*72+sx]&0xffffffu;
  assert(std::abs(int(packed[y*40+x]&0xffffffu)-int(want))<=1);
  assert(packed[y*40+x]>>24==123); // witness reprojection does not claim original S8
 }
 // Narrow close output refuses uncovered zoom-out; the wide slot remains usable.
 auto close=pattern(40,32,true);
 assert(preview.prepare(device.Get(),1,40,32));fill(context.Get(),preview,1,40,close);
 assert(preview.commit(context.Get(),1,original.Get(),1.25f,1.25f,17));
 assert(!preview.select(1,1.f,17));assert(!preview.draw(device.Get(),context.Get(),output.target.Get(),40,32));
 assert(draw(0,1.f)==wide);
 auto settled=draw(1,1.25f);assert(settled==close);
 assert(settled[22*40+30]==enlarged[22*40+30]); // attachment anchor at handoff
 auto green_center=[](std::vector<unsigned> const& pixels){
  double x=0,y=0,count=0;
  for(unsigned i=0;i<pixels.size();++i){auto p=pixels[i];
   if(((p>>8)&255)>220 && ((p>>16)&255)<35 && (p&255)<35){x+=i%40+.5;y+=i/40+.5;++count;}}
  assert(count>0);return std::make_pair(x/count,y/count);
 };
 auto preview_anchor=green_center(enlarged),settled_anchor=green_center(settled);
 assert(std::abs(preview_anchor.first-settled_anchor.first)<.5);
 assert(std::abs(preview_anchor.second-settled_anchor.second)<.5);
 // Missing area is explicitly transparent, even if a caller overstates coverage.
 assert(preview.commit(context.Get(),1,original.Get(),1.25f,1.f,17));
 context->ClearDepthStencilView(output.dsv.Get(),D3D11_CLEAR_DEPTH|D3D11_CLEAR_STENCIL,0,0);
 auto uncovered=draw(1,1.f,true);auto uncovered_depth=read(device.Get(),context.Get(),output.depth.Get());
 compare(uncovered,close,40,32,40,32,.8f);
 assert(uncovered.front()==0&&uncovered.back()==0);
 assert((uncovered_depth.front()&0xffffffu)==0xffffffu);
 assert(!preview.select(2,1.f,17));
 assert(!preview.select(0,std::numeric_limits<float>::quiet_NaN(),17));
 assert(!preview.select(0,0.f,17));assert(!preview.select(0,-1.f,17));
 auto allocation=preview.bytes();
 // Scene/view replacement invalidates every old completed image. Refusing an
 // unavailable serial must not leave a previously selected image drawable.
 assert(!preview.select(0,1.f,18));assert(!preview.select(0,1.f,17));
 assert(!preview.select(1,1.25f,17));assert(preview.bytes()==allocation);
 assert(!preview.draw(device.Get(),context.Get(),output.target.Get(),40,32));
 assert(preview.commit(context.Get(),0,original.Get(),1.f,1.f,18));
 assert(preview.select(0,1.25f,18));assert(preview.draw(device.Get(),context.Get(),output.target.Get(),40,32));
 assert(read(device.Get(),context.Get(),output.color.Get())==enlarged);
 preview.invalidate();assert(!preview.select(0,1.f,18));
 preview.reset();assert(preview.bytes()==0&&preview.target(0)==nullptr);
 std::puts("PASS completed-map preview: original source, relative zoom-out, coverage, handoff anchor, serial and guarded depth");
}
''', timeout=90)


if __name__ == "__main__":
    unittest.main()
