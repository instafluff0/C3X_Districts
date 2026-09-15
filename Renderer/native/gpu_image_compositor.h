#pragma once
// Ordered packed-pixel operations on a borrowed D3D device/context. No presenter,
// native pointers, GDI leases, worker scheduling or implicit readback lives here.
#include "gpu_image_commands.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>

#include "gpu_image_display.h"
namespace c3x_gpu_images {
using Microsoft::WRL::ComPtr;
struct Counts {std::uint64_t uploads=0,upload_bytes=0,commands=0,snapshots=0,resident_bytes=0;};
inline void checked(HRESULT hr){if(FAILED(hr))throw std::runtime_error("GPU image operation failed");}
class Compositor {
    struct Image {Id id=0;unsigned width=0,height=0;Format format=Format::rgb555;std::uint64_t revision=0;bool cpu_current=false;
        ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11ShaderResourceView> read;ComPtr<ID3D11UnorderedAccessView> write;};
    struct Constants {int area[4],offset[2];unsigned mode,color;};
    ID3D11Device* device;ID3D11DeviceContext* context;
    std::array<Image,32> images={};Image scratch,detail_scratch;Id serial=0;
    ImageDisplay display_program;
    ComPtr<ID3D11ComputeShader> shader,import_shader,unit_shader;ComPtr<ID3D11Buffer> constants;
    Counts counters;std::uint64_t budget;
    Image* find(Id id){if(!id)return nullptr;for(auto& image:images)if(image.id==id)return &image;return nullptr;}
    static std::uint64_t bytes(Image const& image){return std::uint64_t(image.width)*image.height*4;}
    void make(Image& image,unsigned width,unsigned height){
        D3D11_TEXTURE2D_DESC d={};d.Width=width;d.Height=height;d.MipLevels=d.ArraySize=d.SampleDesc.Count=1;
        d.Format=DXGI_FORMAT_R32_UINT;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_SHADER_RESOURCE|D3D11_BIND_UNORDERED_ACCESS;
        checked(device->CreateTexture2D(&d,nullptr,&image.texture));
        checked(device->CreateShaderResourceView(image.texture.Get(),nullptr,&image.read));
        checked(device->CreateUnorderedAccessView(image.texture.Get(),nullptr,&image.write));image.width=width;image.height=height;
    }
    void unbind(){ID3D11ShaderResourceView* r[5]={};ID3D11UnorderedAccessView* w[2]={};
        context->CSSetShaderResources(0,5,r);context->CSSetUnorderedAccessViews(0,2,w,nullptr);}
    static Rect selected(Command const& command,Image const& image){return {
        std::max({0,command.area.left,command.clip.left}),std::max({0,command.area.top,command.clip.top}),
        std::min({int(image.width),command.area.right,command.clip.right}),std::min({int(image.height),command.area.bottom,command.clip.bottom})};}
    bool valid(Command const& command){
        auto d=find(command.destination);if(!d)return false;
        if(command.area.left>command.area.right || command.area.top>command.area.bottom ||
           command.clip.left>command.clip.right || command.clip.top>command.clip.bottom)return false;
        if(command.kind!=Kind::copy&&command.kind!=Kind::fill&&command.kind!=Kind::color_key&&command.kind!=Kind::invert&&command.kind!=Kind::quantize&&command.kind!=Kind::expand&&command.kind!=Kind::native_sprite&&command.kind!=Kind::unit_over)return false;
        if(command.kind!=Kind::unit_over&&(command.background||command.detail||command.background_detail))return false;
        if(command.kind==Kind::fill||command.kind==Kind::invert)return d->format==Format::bgra32||command.color<=65535;
        auto s=find(command.source);if(!s)return false;
        if(command.kind==Kind::unit_over){
            auto b=find(command.background),detail=find(command.detail),bd=find(command.background_detail);
            if(s->format!=Format::bgra32||d->format==Format::bgra32||!b||b->format!=d->format||command.color)return false;
            if(command.detail&&(!detail||detail->format!=Format::bgra32||detail->width!=d->width||detail->height!=d->height||detail==s))return false;
            if(command.background_detail&&(!detail||!bd||bd->format!=Format::bgra32||bd->width!=b->width||bd->height!=b->height))return false;
            if(bd==detail&&b!=d)return false;
        }else if(command.kind==Kind::quantize){
            if(s->format!=Format::bgra32||d->format==Format::bgra32||command.color>63)return false;
        }else if(command.kind==Kind::expand){
            if(s->format==Format::bgra32||d->format!=Format::bgra32||command.color>65536)return false;
        }else if(command.kind==Kind::native_sprite){
            // Decoded native source: low 16 bits are a native color, bit 16 is
            // coverage. This is independent of the destination's pixel values.
            if(s->format!=Format::bgra32||command.color>2||
               (d->format==Format::bgra32?command.color==0:command.color!=0))return false;
        }else if(s->format!=d->format)return false;
        auto r=selected(command,*d);if(r.left>=r.right||r.top>=r.bottom)return true;
        auto x=std::int64_t(command.source_x)+r.left-command.area.left;
        auto y=std::int64_t(command.source_y)+r.top-command.area.top;
        return x>=0&&y>=0&&x+(r.right-r.left)<=s->width&&y+(r.bottom-r.top)<=s->height;
    }
    void unit_over(Command const& op,Rect r){
        if(!unit_shader){
            char const* source=R"(
cbuffer Params:register(b0){int4 area;int2 offset;uint mode;uint color;};
Texture2D<uint> body:register(t0);Texture2D<uint> native_below:register(t1);Texture2D<uint> native_ground:register(t2);
Texture2D<uint> detail_below:register(t3);Texture2D<uint> detail_ground:register(t4);
RWTexture2D<uint> native_result:register(u0);RWTexture2D<uint> detail_result:register(u1);
uint expanded(uint c){uint b=((c&31)<<3)|((c&31)>>2),r,g;
 if(mode==1){g=((c>>3)&252)|((c>>9)&3);r=((c>>8)&248)|((c>>13)&7);}
 else {g=((c>>2)&248)|((c>>7)&7);r=((c>>7)&248)|((c>>12)&7);}
 return b|(g<<8)|(r<<16);}
bool keyed(uint c){return (c&0xf800f8)==0xf800f8&&(c&0xf800)==0;}
uint blend(uint source,uint below,uint alpha){
 uint3 s=uint3(source&255,(source>>8)&255,(source>>16)&255),b=uint3(below&255,(below>>8)&255,(below>>16)&255);
 uint3 c=min(s+(b*(255-alpha)+127)/255,255);uint result=c.x|(c.y<<8)|(c.z<<16);
 if(keyed(result))result^=0x800;return result;
}
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
 int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
 uint source=body.Load(int3(at+offset,0)),alpha=source>>24;if(!alpha)return;
 uint below=expanded(native_below.Load(int3(at,0)));bool ground=false;
 if(alpha<255&&keyed(below)){
   uint w,h;native_ground.GetDimensions(w,h);if(any(at>=int2(w,h)))return;
   below=expanded(native_ground.Load(int3(at,0)));if(keyed(below))return;ground=true;
 }
 uint c=blend(source,below,alpha);
 if(mode==1)native_result[at]=(c>>3&31)|((c>>10&63)<<5)|((c>>19&31)<<11);
 else native_result[at]=(c>>3&31)|((c>>11&31)<<5)|((c>>19&31)<<10);
 if(color&1){
   uint full=ground?((color&2)?detail_ground.Load(int3(at,0)):below):detail_below.Load(int3(at,0));
   detail_result[at]=blend(source,full,alpha)|0xff000000;
 }
})";
            ComPtr<ID3DBlob> code,error;checked(D3DCompile(source,std::strlen(source),"native unit composition",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
            checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&unit_shader));
        }
        auto d=find(op.destination),b=find(op.background),detail=find(op.detail),bd=find(op.background_detail);
        unbind();D3D11_BOX box={unsigned(r.left),unsigned(r.top),0,unsigned(r.right),unsigned(r.bottom),1};
        // Only pixels in the selected body/shadow rectangle are read or copied.
        context->CopySubresourceRegion(scratch.texture.Get(),0,r.left,r.top,0,d->texture.Get(),0,&box);++counters.snapshots;
        if(detail){context->CopySubresourceRegion(detail_scratch.texture.Get(),0,r.left,r.top,0,detail->texture.Get(),0,&box);++counters.snapshots;}
        auto ground=b==d?&scratch:b;auto full_ground=bd==detail?&detail_scratch:bd;
        ID3D11ShaderResourceView* reads[5]={find(op.source)->read.Get(),scratch.read.Get(),ground->read.Get(),detail?detail_scratch.read.Get():nullptr,bd?full_ground->read.Get():nullptr};
        ID3D11UnorderedAccessView* writes[2]={d->write.Get(),detail?detail->write.Get():nullptr};
        Constants p={{r.left,r.top,r.right,r.bottom},{int(std::int64_t(op.source_x)-op.area.left),int(std::int64_t(op.source_y)-op.area.top)},d->format==Format::rgb565?1u:0u,(detail?1u:0u)|(bd?2u:0u)};
        context->UpdateSubresource(constants.Get(),0,nullptr,&p,0,0);auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
        context->CSSetShaderResources(0,5,reads);context->CSSetUnorderedAccessViews(0,2,writes,nullptr);context->CSSetShader(unit_shader.Get(),nullptr,0);
        context->Dispatch(unsigned(r.right-r.left+7)/8,unsigned(r.bottom-r.top+7)/8,1);unbind();
        d->cpu_current=false;if(detail)detail->cpu_current=false;++counters.commands;
    }
public:
    // R32_UINT stores native 16-bit words exactly too. Its explicit 4-byte budget
    // avoids format-dependent typed-UAV support and rounding intermediate images.
    Compositor(ID3D11Device* d,ID3D11DeviceContext* c,std::uint64_t cap=64u*1024u*1024u):device(d),context(c),budget(cap){
        if(!d||!c||d->GetFeatureLevel()<D3D_FEATURE_LEVEL_11_0)throw std::runtime_error("GPU image operations require feature level 11");
        char const* source=R"(
cbuffer Params:register(b0){int4 area;int2 offset;uint mode;uint color;};
Texture2D<uint> input_image:register(t0);RWTexture2D<uint> output_image:register(u0);
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
 int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
 uint value=color;if(mode!=1)value=input_image.Load(int3(at+offset,0));
 if(mode==2&&value==color)return;
 if(mode==4&&(value&0xffffff)==(color&0xffffff))return;
 if(mode==3)value^=color;
 if(mode==5||mode==6){
   uint2 xy=uint2(at+offset)-uint2(color&7,(color>>3)&7);uint threshold=0;
   [unroll]for(uint bit=0;bit<3;++bit){uint a=(xy.x>>bit)&1,b=(xy.y>>bit)&1;threshold=(threshold<<2)|((a^b)<<1)|b;}
   uint3 levels=uint3(31,mode==6?63:31,31);
   uint3 scaled=uint3(value&255,(value>>8)&255,(value>>16)&255)*levels;
   uint3 q=scaled/255+uint3((scaled%255)*128>(threshold*2+1)*255);
   value=q.x|(q.y<<5)|(q.z<<(mode==6?11:10));
 }
 if(mode>=9){
   if(!(value&65536))return;
   value&=65535;
   if(mode==9){output_image[at]=value;return;}
 }
 if(mode==7||mode==8||mode==10||mode==11){
   if(mode<9&&value==color)return;
   uint b=((value&31)<<3)|((value&31)>>2);
   uint r,g;
   if(mode==8||mode==11){g=((value>>3)&252)|((value>>9)&3);r=((value>>8)&248)|((value>>13)&7);}
   else {g=((value>>2)&248)|((value>>7)&7);r=((value>>7)&248)|((value>>12)&7);}
   value=b|(g<<8)|(r<<16)|0xff000000;
 }
 output_image[at]=value;
})";
        ComPtr<ID3DBlob> code,error;checked(D3DCompile(source,std::strlen(source),"packed image operations",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
        checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader));
        D3D11_BUFFER_DESC bd={};bd.ByteWidth=sizeof(Constants);bd.Usage=D3D11_USAGE_DEFAULT;bd.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        checked(device->CreateBuffer(&bd,nullptr,&constants));
    }
    ~Compositor(){unbind();}
    Compositor(Compositor const&)=delete;Compositor& operator=(Compositor const&)=delete;
    Id create(unsigned width,unsigned height,Format format){
        if((format!=Format::rgb555&&format!=Format::rgb565&&format!=Format::bgra32)||!width||!height||width>2240||height>1192||std::uint64_t(width)*height*4>budget-counters.resident_bytes)return 0;
        for(auto& image:images)if(!image.id){Image next;make(next,width,height);next.id=++serial;next.format=format;
            unsigned zero[4]={};context->ClearUnorderedAccessViewUint(next.write.Get(),zero);
            counters.resident_bytes+=bytes(next);image=std::move(next);return image.id;}return 0;
    }
    bool destroy(Id id){auto image=find(id);if(!image)return false;unbind();counters.resident_bytes-=bytes(*image);*image={};return true;}
    bool upload(Id id,std::uint64_t revision,std::uint32_t const* pixels,std::size_t count){
        auto image=find(id);if(!image||!revision||!pixels||count!=std::size_t(image->width)*image->height)return false;
        if(image->revision==revision)return image->cpu_current;
        if(image->revision>revision)return false;
        if(image->format!=Format::bgra32)for(std::size_t n=0;n<count;++n)if(pixels[n]>65535)return false;
        unbind();context->UpdateSubresource(image->texture.Get(),0,nullptr,pixels,image->width*4,0);
        image->revision=revision;image->cpu_current=true;++counters.uploads;counters.upload_bytes+=count*4;return true;
    }
    // Validate the entire transaction and reserve overlap scratch before any draw.
    // Rejection leaves the destination unchanged; hardware errors must invalidate
    // the caller's unpublished transaction, never publish a partially drawn image.
    bool submit(Command const* commands,std::size_t count){
        if(!commands||!count||count>2048)return false;
        unsigned width=0,height=0,detail_width=0,detail_height=0;
        for(std::size_t n=0;n<count;++n){auto const& op=commands[n];if(!valid(op))return false;
            if(op.kind==Kind::unit_over&&op.detail){auto image=find(op.detail);detail_width=std::max(detail_width,image->width);detail_height=std::max(detail_height,image->height);}
            if(op.kind==Kind::unit_over || op.kind==Kind::invert || (op.kind!=Kind::fill&&op.source==op.destination)){
                auto image=find(op.destination);width=std::max(width,image->width);height=std::max(height,image->height);}}
        // Reserve both destination snapshots before any operation can mutate a
        // native/full-color pair. Old allocations count until replacement ends.
        bool grow=width&&(scratch.width<width||scratch.height<height);
        bool grow_detail=detail_width&&(detail_scratch.width<detail_width||detail_scratch.height<detail_height);
        auto needed=(grow?std::uint64_t(width)*height*4:0)+(grow_detail?std::uint64_t(detail_width)*detail_height*4:0);
        if(needed>budget-counters.resident_bytes)return false;
        Image next,next_detail;if(grow)make(next,width,height);if(grow_detail)make(next_detail,detail_width,detail_height);
        if(grow){counters.resident_bytes+=bytes(next)-bytes(scratch);scratch=std::move(next);}
        if(grow_detail){counters.resident_bytes+=bytes(next_detail)-bytes(detail_scratch);detail_scratch=std::move(next_detail);}
        for(std::size_t n=0;n<count;++n){auto const& op=commands[n];auto d=find(op.destination);auto r=selected(op,*d);
            if(r.left>=r.right||r.top>=r.bottom)continue;
            if(op.kind==Kind::unit_over){unit_over(op,r);continue;}
            auto s=op.kind==Kind::invert?d:find(op.source);unbind();
            if(s==d){D3D11_BOX box={0,0,0,d->width,d->height,1};context->CopySubresourceRegion(scratch.texture.Get(),0,0,0,0,d->texture.Get(),0,&box);s=&scratch;++counters.snapshots;}
            if(op.kind==Kind::copy){
                unsigned x=unsigned(std::int64_t(op.source_x)+r.left-op.area.left),y=unsigned(std::int64_t(op.source_y)+r.top-op.area.top);
                D3D11_BOX box={x,y,0,x+unsigned(r.right-r.left),y+unsigned(r.bottom-r.top),1};
                context->CopySubresourceRegion(d->texture.Get(),0,r.left,r.top,0,s->texture.Get(),0,&box);
                d->cpu_current=false;++counters.commands;continue;
            }
            if(op.kind==Kind::fill && r.left==0 && r.top==0 && r.right==int(d->width) && r.bottom==int(d->height)){
                unsigned value[4]={op.color,op.color,op.color,op.color};context->ClearUnorderedAccessViewUint(d->write.Get(),value);
                d->cpu_current=false;++counters.commands;continue;
            }
            Constants p={{r.left,r.top,r.right,r.bottom},{0,0},op.kind==Kind::fill?1u:op.kind==Kind::color_key?(d->format==Format::bgra32?4u:2u):op.kind==Kind::invert?3u:op.kind==Kind::quantize?(d->format==Format::rgb565?6u:5u):op.kind==Kind::expand?(s->format==Format::rgb565?8u:7u):op.kind==Kind::native_sprite?9u+op.color:0u,op.color};
            if(op.kind!=Kind::fill&&op.kind!=Kind::invert){p.offset[0]=int(std::int64_t(op.source_x)-op.area.left);p.offset[1]=int(std::int64_t(op.source_y)-op.area.top);}
            context->UpdateSubresource(constants.Get(),0,nullptr,&p,0,0);auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
            auto read=s?s->read.Get():nullptr;auto write=d->write.Get();context->CSSetShaderResources(0,1,&read);context->CSSetUnorderedAccessViews(0,1,&write,nullptr);
            context->CSSetShader(shader.Get(),nullptr,0);context->Dispatch(unsigned(r.right-r.left+7)/8,unsigned(r.bottom-r.top+7)/8,1);unbind();
            // GPU mutation invalidates a prior CPU revision; the caller must supply
            // a strictly newer revision to replace this image from CPU content.
            d->cpu_current=false;++counters.commands;
        }return true;
    }
    // GPU-to-GPU import of a completed display texture; no staging or CPU seed.
    bool import_bgra(Id id,ID3D11Texture2D* source){
        auto destination=find(id);if(!destination||!source||destination->format!=Format::bgra32)return false;
        D3D11_TEXTURE2D_DESC desc={};source->GetDesc(&desc);
        if(desc.Width!=destination->width||desc.Height!=destination->height||desc.SampleDesc.Count!=1||
           desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM||!(desc.BindFlags&D3D11_BIND_SHADER_RESOURCE))return false;
        ComPtr<ID3D11Device> source_device;source->GetDevice(&source_device);if(source_device.Get()!=device)return false;
        if(!import_shader){
            char const* hlsl=R"(
Texture2D<float4> input_image:register(t0);RWTexture2D<uint> output_image:register(u0);
[numthreads(8,8,1)] void main(uint3 at:SV_DispatchThreadID){
 uint w,h;output_image.GetDimensions(w,h);if(at.x>=w||at.y>=h)return;
 uint4 c=uint4(round(saturate(input_image.Load(int3(at.xy,0)))*255.0));
 output_image[at.xy]=c.b|(c.g<<8)|(c.r<<16)|(c.a<<24);
})";
            ComPtr<ID3DBlob> code,error;checked(D3DCompile(hlsl,std::strlen(hlsl),"resident map import",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
            checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&import_shader));
        }
        ComPtr<ID3D11ShaderResourceView> input;checked(device->CreateShaderResourceView(source,nullptr,&input));
        unbind();auto read=input.Get();auto write=destination->write.Get();context->CSSetShaderResources(0,1,&read);context->CSSetUnorderedAccessViews(0,1,&write,nullptr);
        context->CSSetShader(import_shader.Get(),nullptr,0);context->Dispatch((desc.Width+7)/8,(desc.Height+7)/8,1);unbind();
        destination->cpu_current=false;return true;
    }
    bool display(Id id,ID3D11RenderTargetView* target,unsigned width,unsigned height,Rect area){
        auto image=find(id);if(!image||image->format!=Format::bgra32||image->width!=width||image->height!=height||!target)return false;
        RECT clip={std::max(0,area.left),std::max(0,area.top),std::min(int(width),area.right),std::min(int(height),area.bottom)};
        return display_program.draw(device,context,image->read.Get(),target,width,height,clip);
    }

    ID3D11Texture2D* texture(Id id){auto image=find(id);return image?image->texture.Get():nullptr;}
    ID3D11ShaderResourceView* view(Id id){auto image=find(id);return image?image->read.Get():nullptr;}
    Counts stats()const{return counters;}
};
} // namespace c3x_gpu_images
