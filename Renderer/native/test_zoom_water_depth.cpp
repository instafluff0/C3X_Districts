#define NOMINMAX
#include <windows.h>
#include "Renderer/native/gpu_image_compositor.h"
#include "Renderer/native/render_core/linear_target.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <cstdio>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_gpu_images;
using namespace c3x_renderer::render_core;

void require_water_depth(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
ComPtr<ID3DBlob> water_depth_shader(std::string const& source,char const* entry,char const* model){
 ComPtr<ID3DBlob> code,error;
 auto hr=D3DCompile(source.data(),source.size(),"zoom_water_depth",nullptr,nullptr,entry,model,
     D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&error);
 if(FAILED(hr)&&error)std::fprintf(stderr,"%s",static_cast<char const*>(error->GetBufferPointer()));
 checked(hr);return code;
}
template<class T> std::vector<T> water_depth_read(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
 D3D11_TEXTURE2D_DESC desc={};texture->GetDesc(&desc);
 desc.BindFlags=0;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Texture2D> staging;checked(device->CreateTexture2D(&desc,nullptr,&staging));
 context->CopyResource(staging.Get(),texture);D3D11_MAPPED_SUBRESOURCE mapped={};
 checked(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped));
 std::vector<T> result(desc.Width*desc.Height);
 for(unsigned y=0;y<desc.Height;++y)std::memcpy(result.data()+y*desc.Width,
     static_cast<char const*>(mapped.pData)+y*mapped.RowPitch,desc.Width*sizeof(T));
 context->Unmap(staging.Get(),0);return result;
}

int main(){try{
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
 checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context));
 LinearResample corrected,nearest;
 require_water_depth(corrected.ensure(device.Get())&&nearest.ensure(device.Get()),"production resample setup");
 // Execute the old nearest-depth path as a negative control using the same
 // production color, coverage and secondary-source code.
 auto root=std::filesystem::current_path();
 while(!std::filesystem::exists(root/"Renderer/native/render_core/linear_target.h")){
  auto parent=root.parent_path();require_water_depth(parent!=root,"checkout not found");root=parent;
 }
 std::ifstream input(root/"Renderer/native/render_core/linear_target.h");
 std::string header((std::istreambuf_iterator<char>(input)),std::istreambuf_iterator<char>());
 auto begin=header.find("char const* source=R\"(",header.find("struct LinearResample"));
 require_water_depth(begin!=std::string::npos,"resample shader marker");begin+=std::strlen("char const* source=R\"(");
 auto end=header.find(")\";",begin);require_water_depth(end!=std::string::npos,"resample shader end");
 auto legacy=header.substr(begin,end-begin);
 begin=legacy.find("float retained_depth(");end=legacy.find("Output fetch0",begin);
 require_water_depth(begin!=std::string::npos&&end!=std::string::npos,"depth helper markers");
 legacy.replace(begin,end-begin,"float retained_depth(Texture2D<float> field,float2 p,float2 size,float shift,float4 covered){float z=field.Load(int3(int2(floor(p)),0));return z<1?z+shift:1;}\n");
 auto blob=water_depth_shader(legacy,"PS","ps_5_0");nearest.pixel->Release();nearest.pixel=nullptr;
 checked(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&nearest.pixel));

 // Sloping map depth with almost coplanar animated water. The water's depth is
 // evaluated independently at the destination projection, like the live pass.
 char const* shader=R"(
cbuffer Parameters:register(b0){float4 affine;float4 plane;float4 extent;};
float4 VS(uint id:SV_VertexID):SV_Position{float2 p=float2((id<<1)&2,id&2);return float4(p*float2(2,-2)+float2(-1,1),0,1);}
struct Output{float4 color:SV_Target;float depth:SV_Depth;};
Output Seed(float4 p:SV_Position){Output o;o.color=float4(1,0,0,1);
 o.depth=.5+dot(plane.xy,p.xy-extent.xy);
 if(extent.z==1&&p.x<extent.x)o.depth-=.02;
 if(extent.z==2&&p.x>=extent.x){o.depth=1;o.color=0;}
 if(extent.z==3){uint2 q=uint2(p.xy);o.color=float4((q.x+q.y)%2,q.x%3==0,q.y%5==0,1);o.depth=.5;}
 return o;}
Output Water(float4 p:SV_Position){Output o;o.color=float4(0,1,0,1);
 o.depth=.5+dot(plane.xy,p.xy*affine.xy+affine.zw-extent.xy)+plane.z-plane.w;return o;}
)";
 ComPtr<ID3D11VertexShader> vertex;ComPtr<ID3D11PixelShader> seed,water;
 blob=water_depth_shader(shader,"VS","vs_5_0");checked(device->CreateVertexShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&vertex));
 blob=water_depth_shader(shader,"Seed","ps_5_0");checked(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&seed));
 blob=water_depth_shader(shader,"Water","ps_5_0");checked(device->CreatePixelShader(blob->GetBufferPointer(),blob->GetBufferSize(),nullptr,&water));
 D3D11_DEPTH_STENCIL_DESC zd={};zd.DepthEnable=true;zd.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ALL;zd.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
 ComPtr<ID3D11DepthStencilState> less;checked(device->CreateDepthStencilState(&zd,&less));
 D3D11_BUFFER_DESC bd={};bd.ByteWidth=48;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
 ComPtr<ID3D11Buffer> constants;checked(device->CreateBuffer(&bd,nullptr,&constants));
 LinearTarget retained,actual;
 require_water_depth(retained.ensure(device.Get(),256,192,true,false,1)&&actual.ensure(device.Get(),128,96,true,false,1),"fixture targets");
 struct Values{float affine[4],plane[4],extent[4];} values={};
 auto draw=[&](LinearTarget const& target,ID3D11PixelShader* pixel,ID3D11DepthStencilState* depth){
  context->UpdateSubresource(constants.Get(),0,nullptr,&values,0,0);auto cb=constants.Get();
  context->PSSetConstantBuffers(0,1,&cb);context->OMSetRenderTargets(1,&target.target,target.depth);
  context->OMSetDepthStencilState(depth,0);context->OMSetBlendState(nullptr,nullptr,~0u);
  context->RSSetState(corrected.rasterizer);D3D11_VIEWPORT viewport={0,0,float(target.width),float(target.height),0,1};
  context->RSSetViewports(1,&viewport);context->IASetInputLayout(nullptr);
  context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  context->VSSetShader(vertex.Get(),nullptr,0);context->PSSetShader(pixel,nullptr,0);context->Draw(3,0);
  context->OMSetRenderTargets(0,nullptr,nullptr);
 };
 unsigned cases=0,legacy_rejected=0,corrected_rejected=0,clear_checks=0,occluder_checks=0;
 for(float source_scale:{.5f,1.f})for(float zoom:{.5f,.625f,.75f,.875f,1.037f,1.125f,1.375f,1.75f,2.625f,3.f})
 for(float shift:{-5.f/16384,5.f/16384})for(unsigned profile=0;profile<3;++profile){
  if(source_scale==1.f&&zoom==.5f)continue; // This fixture has only a 2:1 source extent.
  float ratio=source_scale/zoom;
  values={{ratio,ratio,128-64*ratio+.17f,96-48*ratio-.29f},
      {.17f/(16384*source_scale),-1.f/(16384*source_scale),shift,.000006f*128/16384},{128,96,float(profile),0}};
  draw(retained,seed.Get(),corrected.depth);
  LinearResample::Source source;source.color=retained.samples;source.depth=retained.depth_samples;
  std::copy(values.affine,values.affine+4,source.map);source.covered[0]=source.covered[1]=1;
  source.covered[2]=255;source.covered[3]=191;source.size[0]=256;source.size[1]=192;source.depth_shift=shift;
  for(unsigned control=0;control<2;++control){
   auto& resample=control?nearest:corrected;
   // Exercise fallback selection as well as the primary retained image.
   auto primary=source;bool fallback=(cases&1)!=0;if(fallback)primary.covered[2]=primary.covered[3]=0;
   require_water_depth(resample.draw(context.Get(),actual,primary,fallback?&source:nullptr),"resample draw");
   if(!control){
    auto depths=water_depth_read<unsigned>(device.Get(),context.Get(),actual.depth_texture);
    for(unsigned y=0;y<96;++y)for(unsigned x=0;x<128;++x){
     float sx=(x+.5f)*ratio+values.affine[2],sy=(y+.5f)*ratio+values.affine[3];
     bool occluder=profile==1&&int(std::floor(sx))<128,empty=profile==2&&int(std::floor(sx))>=128;
     unsigned got=depths[y*128+x]&0xffffff;
     if(empty){require_water_depth(got==0xffffff,"clear depth sentinel changed");++clear_checks;continue;}
     // At a hard silhouette the one-sided reconstruction intentionally limits
     // the slope. Check exact planes away from that one-texel neighborhood.
     if(profile&&std::abs(sx-128)<1.5f)continue;
     float expected=.5f+values.plane[0]*(sx-128)+values.plane[1]*(sy-96)+shift-(occluder?.02f:0.f);
     require_water_depth(std::abs(double(got)-double(expected)*16777215)<=4,"continuous plane depth mismatch");
    }
   }
   draw(actual,water.Get(),less.Get());
   auto colors=water_depth_read<std::array<unsigned short,4>>(device.Get(),context.Get(),actual.color);
   for(unsigned y=0;y<96;++y)for(unsigned x=0;x<128;++x){
    float sx=(x+.5f)*ratio+values.affine[2];bool occluder=profile==1&&int(std::floor(sx))<128;
    bool visible=colors[y*128+x][1]==0x3c00; // exactly 1 in float16
    if(!control&&occluder){require_water_depth(!visible,"water leaked through foreground silhouette");++occluder_checks;}
    if(!occluder&&!visible){if(control)++legacy_rejected;else ++corrected_rejected;}
   }
  }
  ++cases;
 }
 require_water_depth(legacy_rejected>10000,"negative control did not reproduce water stripes");
 require_water_depth(corrected_rejected==0,"corrected depth still rejects visible water");
 // Pixel-sharp terrain must survive an unchanged view and integer camera
 // translations through both retained restore and fallback composition.
 // One-pixel detail makes a half-resolution fallback fail deterministically.
 values={{1,1,0,0},{0,0,0,0},{128,96,3,0}};draw(retained,seed.Get(),corrected.depth);
 auto reference=water_depth_read<std::array<unsigned short,4>>(device.Get(),context.Get(),retained.color);
 LinearRestore restore;require_water_depth(restore.ensure(device.Get(),1),"sharpness restore setup");
 unsigned sharp_checks=0,blur_rejected=0;
 for(int dx:{0,1,17,63})for(int dy:{0,1,19})for(unsigned mode=0;mode<3;++mode){
  LinearResample::Source source;source.color=retained.samples;source.depth=retained.depth_samples;
  source.map[0]=source.map[1]=1;source.map[2]=float(dx);source.map[3]=float(dy);
  source.covered[2]=source.size[0]=256;source.covered[3]=source.size[1]=192;
  if(mode==0)require_water_depth(restore.draw(context.Get(),actual,retained.samples,retained.depth_samples,
      -dx,-dy,{},nullptr,256,192,false,false,0,nullptr,1),"sharpness restore");
  else {auto primary=source;if(mode==2)primary.covered[2]=primary.covered[3]=0;
   require_water_depth(corrected.draw(context.Get(),actual,primary,mode==2?&source:nullptr),"sharpness resample");}
  auto pixels=water_depth_read<std::array<unsigned short,4>>(device.Get(),context.Get(),actual.color);
  for(unsigned y=0;y<96;++y)for(unsigned x=0;x<128;++x){
   require_water_depth(pixels[y*128+x]==reference[(y+dy)*256+x+dx],"stationary terrain lost pixel detail");++sharp_checks;
  }
 }
 LinearTarget reduced;require_water_depth(reduced.ensure(device.Get(),128,96,true,false,1),"blur control target");
 LinearResample::Source source;source.color=retained.samples;source.depth=retained.depth_samples;
 source.map[0]=source.map[1]=2;source.covered[2]=source.size[0]=256;source.covered[3]=source.size[1]=192;
 require_water_depth(corrected.draw(context.Get(),reduced,source,nullptr),"blur control downsample");
 source.color=reduced.samples;source.depth=reduced.depth_samples;source.map[0]=source.map[1]=.5f;
 source.covered[2]=source.size[0]=128;source.covered[3]=source.size[1]=96;
 require_water_depth(corrected.draw(context.Get(),actual,source,nullptr),"blur control upsample");
 auto blurred=water_depth_read<std::array<unsigned short,4>>(device.Get(),context.Get(),actual.color);
 for(unsigned y=0;y<96;++y)for(unsigned x=0;x<128;++x)blur_rejected+=blurred[y*128+x]!=reference[y*256+x];
 require_water_depth(blur_rejected>12000,"negative control did not detect reduced-resolution detail loss");
 std::printf("PASS stationary detail: exact_pixel_checks=%u half_resolution_rejected=%u restore_primary_fallback=1\n",sharp_checks,blur_rejected);
 context->ClearState();
 std::printf("PASS zoom water depth: cases=%u legacy_rejected=%u corrected_rejected=%u clear_checks=%u occluder_checks=%u primary_and_fallback=1 half_resolution=1\n",
     cases,legacy_rejected,corrected_rejected,clear_checks,occluder_checks);return 0;
}catch(std::exception const& error){std::fprintf(stderr,"FAIL zoom water depth: %s\n",error.what());return 1;}}
