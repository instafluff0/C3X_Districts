#pragma once
// Ordered packed-pixel operations on a borrowed D3D device/context. No presenter,
// native pointers, GDI leases, worker scheduling or implicit readback lives here.
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <array>
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace c3x_gpu_images {
using Microsoft::WRL::ComPtr;
using Id=std::uint64_t;
enum class Format { rgb555, rgb565, bgra32 };
enum class Kind { copy, fill, color_key, invert };
struct Rect { int left,top,right,bottom; };
struct Command {Kind kind;Id destination,source;Rect area,clip;int source_x=0,source_y=0;std::uint32_t color=0;};
struct Counts {std::uint64_t uploads=0,upload_bytes=0,commands=0,snapshots=0,resident_bytes=0;};
inline void checked(HRESULT hr){if(FAILED(hr))throw std::runtime_error("GPU image operation failed");}
class Compositor {
    struct Image {Id id=0;unsigned width=0,height=0;Format format=Format::rgb555;std::uint64_t revision=0;bool cpu_current=false;
        ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11ShaderResourceView> read;ComPtr<ID3D11UnorderedAccessView> write;};
    struct Constants {int area[4],offset[2];unsigned mode,color;};
    ID3D11Device* device;ID3D11DeviceContext* context;
    std::array<Image,32> images={};Image scratch;Id serial=0;
    ComPtr<ID3D11ComputeShader> shader;ComPtr<ID3D11Buffer> constants;
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
    void unbind(){ID3D11ShaderResourceView* r=nullptr;ID3D11UnorderedAccessView* w=nullptr;
        context->CSSetShaderResources(0,1,&r);context->CSSetUnorderedAccessViews(0,1,&w,nullptr);}
    static Rect selected(Command const& command,Image const& image){return {
        std::max({0,command.area.left,command.clip.left}),std::max({0,command.area.top,command.clip.top}),
        std::min({int(image.width),command.area.right,command.clip.right}),std::min({int(image.height),command.area.bottom,command.clip.bottom})};}
    bool valid(Command const& command){
        auto d=find(command.destination);if(!d)return false;
        if(command.area.left>command.area.right || command.area.top>command.area.bottom ||
           command.clip.left>command.clip.right || command.clip.top>command.clip.bottom)return false;
        if(command.kind!=Kind::copy&&command.kind!=Kind::fill&&command.kind!=Kind::color_key&&command.kind!=Kind::invert)return false;
        if(command.kind==Kind::fill||command.kind==Kind::invert)return d->format==Format::bgra32||command.color<=65535;
        auto s=find(command.source);if(!s||s->format!=d->format)return false;
        auto r=selected(command,*d);if(r.left>=r.right||r.top>=r.bottom)return true;
        auto x=std::int64_t(command.source_x)+r.left-command.area.left;
        auto y=std::int64_t(command.source_y)+r.top-command.area.top;
        return x>=0&&y>=0&&x+(r.right-r.left)<=s->width&&y+(r.bottom-r.top)<=s->height;
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
        unsigned width=0,height=0;
        for(std::size_t n=0;n<count;++n){auto const& op=commands[n];if(!valid(op))return false;
            if(op.kind==Kind::invert || (op.kind!=Kind::fill&&op.source==op.destination)){
                auto image=find(op.destination);width=std::max(width,image->width);height=std::max(height,image->height);}}
        if(width && (scratch.width<width||scratch.height<height)){
            auto requested=std::uint64_t(width)*height*4;
            if(requested>budget-counters.resident_bytes+bytes(scratch))return false;
            // Replacement temporarily retains both allocations; account for that peak too.
            if(requested>budget-counters.resident_bytes)return false;
            Image next;make(next,width,height);counters.resident_bytes+=requested-bytes(scratch);scratch=std::move(next);
        }
        for(std::size_t n=0;n<count;++n){auto const& op=commands[n];auto d=find(op.destination);auto r=selected(op,*d);
            if(r.left>=r.right||r.top>=r.bottom)continue;
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
            Constants p={{r.left,r.top,r.right,r.bottom},{0,0},op.kind==Kind::fill?1u:op.kind==Kind::color_key?(d->format==Format::bgra32?4u:2u):op.kind==Kind::invert?3u:0u,op.color};
            if(op.kind!=Kind::fill&&op.kind!=Kind::invert){p.offset[0]=int(std::int64_t(op.source_x)-op.area.left);p.offset[1]=int(std::int64_t(op.source_y)-op.area.top);}
            context->UpdateSubresource(constants.Get(),0,nullptr,&p,0,0);auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
            auto read=s?s->read.Get():nullptr;auto write=d->write.Get();context->CSSetShaderResources(0,1,&read);context->CSSetUnorderedAccessViews(0,1,&write,nullptr);
            context->CSSetShader(shader.Get(),nullptr,0);context->Dispatch(unsigned(r.right-r.left+7)/8,unsigned(r.bottom-r.top+7)/8,1);unbind();
            // GPU mutation invalidates a prior CPU revision; the caller must supply
            // a strictly newer revision to replace this image from CPU content.
            d->cpu_current=false;++counters.commands;
        }return true;
    }
    ID3D11Texture2D* texture(Id id){auto image=find(id);return image?image->texture.Get():nullptr;}
    ID3D11ShaderResourceView* view(Id id){auto image=find(id);return image?image->read.Get():nullptr;}
    Counts stats()const{return counters;}
};
} // namespace c3x_gpu_images
