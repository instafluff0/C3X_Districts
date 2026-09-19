#pragma once
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <cstring>
#include <stdexcept>

namespace c3x_renderer {
// Borrowed for one ordered draw on the renderer worker. These are reusable HDR
// raster/depth and transfer scratch, never a separately published finished pose.
struct UnitSceneSample {
    ID3D11ShaderResourceView* body=nullptr;
    ID3D11ShaderResourceView* heights=nullptr;
    std::array<float,16> ground{};
    unsigned width=0,height=0;
    std::array<int,4> coverage{};
    std::array<int,2> origin{};
};
class GpuUnitScene {
    Microsoft::WRL::ComPtr<ID3D11ComputeShader> shader;
    Microsoft::WRL::ComPtr<ID3D11Buffer> projection;
public:
    void draw(ID3D11Device* device,ID3D11DeviceContext* context,UnitSceneSample const& sample,
              unsigned width,unsigned height){
        auto check=[](HRESULT hr){if(FAILED(hr))throw std::runtime_error("unit scene composition failed");};
        if(!shader){char const* source=R"(
cbuffer Params:register(b0){int4 area;int2 offset;uint mode;uint color;};
cbuffer Ground:register(b1){float4 pose,bounds,quality;int4 canvas;};
Texture2D<float4> body:register(t0);Texture2D<uint> native_below:register(t1);Texture2D<uint> native_ground:register(t2);
Texture2D<uint> detail_below:register(t3);Texture2D<uint> detail_ground:register(t4);Texture2D<float> heights:register(t5);
RWTexture2D<uint> native_result:register(u0);RWTexture2D<uint> detail_result:register(u1);
uint finished(int2 at){
 uint w=canvas.z,h=canvas.w;uint4 c=uint4(round(saturate(body.Load(int3(at-canvas.xy,0)))*255));
 precise float sx=(float(at.x)+.5f-pose.x)/(64*pose.z),sy=(float(at.y)+.5f-pose.y)/(32*pose.z);
 precise float x=(sx+sy)*.5f,y=(sy-sx)*.5f;
 precise float projected_x=(x-bounds.x)/bounds.z*quality.x,projected_y=(y-bounds.y)/bounds.w*quality.x;
 int px=int(floor(projected_x)),py=int(floor(projected_y));
 precise float count=0;
 [unroll]for(int oy=-1;oy<=1;++oy)[unroll]for(int ox=-1;ox<=1;++ox){int2 q=int2(px+ox,py+oy);
  if(all(q>=0)&&all(q<int(quality.x))&&heights.Load(int3(q,0))>.006f)count+=1.f/9;}
 precise float fade=clamp(float(min(min(at.x,at.y),min(int(w)-1-at.x,int(h)-1-at.y)))/3,0.f,1.f);
 precise float shadow_alpha=pose.w*fade*count;uint shade=uint(shadow_alpha);
 uint alpha=c.a+(shade*(255-c.a)+127)/255;uint3 rgb=(c.rgb*c.a+127)/255;
 return rgb.b|(rgb.g<<8)|(rgb.r<<16)|(alpha<<24);
}
uint expanded(uint c){uint b=((c&31)<<3)|((c&31)>>2),r,g;
 if(mode==1){g=((c>>3)&252)|((c>>9)&3);r=((c>>8)&248)|((c>>13)&7);}
 else {g=((c>>2)&248)|((c>>7)&7);r=((c>>7)&248)|((c>>12)&7);}return b|(g<<8)|(r<<16);}
bool keyed(uint c){return (c&0xf800f8)==0xf800f8&&(c&0xf800)==0;}
uint blend(uint source,uint below,uint alpha){
 uint3 s=uint3(source&255,(source>>8)&255,(source>>16)&255),b=uint3(below&255,(below>>8)&255,(below>>16)&255);
 uint3 c=min(s+(b*(255-alpha)+127)/255,255);uint result=c.x|(c.y<<8)|(c.z<<16);
 if(keyed(result))result^=0x800;return result;
}
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
 int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
 uint source=finished(at+offset),alpha=source>>24;if(!alpha)return;
 uint below=expanded(native_below.Load(int3(at-area.xy,0)));bool ground=false;
 if(alpha<255&&keyed(below)){
  int2 ground_at=(color&4)?at-area.xy:at;uint w,h;native_ground.GetDimensions(w,h);if(any(ground_at>=int2(w,h)))return;
  below=expanded(native_ground.Load(int3(ground_at,0)));if(keyed(below))return;ground=true;
 }
 uint c=blend(source,below,alpha);
 if(mode==1)native_result[at]=(c>>3&31)|((c>>10&63)<<5)|((c>>19&31)<<11);
 else native_result[at]=(c>>3&31)|((c>>11&31)<<5)|((c>>19&31)<<10);
 if(color&1){uint full=ground?((color&2)?detail_ground.Load(int3((color&8)?at-area.xy:at,0)):below):detail_below.Load(int3(at-area.xy,0));
  detail_result[at]=blend(source,full,alpha)|0xff000000;}
})";
            Microsoft::WRL::ComPtr<ID3DBlob> code,error;
            auto hr=D3DCompile(source,std::strlen(source),"direct unit scene",nullptr,nullptr,"main","cs_5_0",
                D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS,0,&code,&error);
            if(error)OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));check(hr);
            check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader));
            D3D11_BUFFER_DESC d={};d.ByteWidth=64;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;check(device->CreateBuffer(&d,nullptr,&projection));
        }
        struct Constants {float ground[12];int canvas[4];} values;
        std::memcpy(values.ground,sample.ground.data(),sizeof(values.ground));
        values.canvas[0]=sample.origin[0];values.canvas[1]=sample.origin[1];values.canvas[2]=int(sample.width);values.canvas[3]=int(sample.height);
        context->UpdateSubresource(projection.Get(),0,nullptr,&values,0,0);
        auto cb=projection.Get();context->CSSetConstantBuffers(1,1,&cb);
        context->CSSetShaderResources(0,1,&sample.body);context->CSSetShaderResources(5,1,&sample.heights);
        context->CSSetShader(shader.Get(),nullptr,0);context->Dispatch((width+7)/8,(height+7)/8,1);
        ID3D11ShaderResourceView* empty=nullptr;context->CSSetShaderResources(5,1,&empty);
    }
};
}
