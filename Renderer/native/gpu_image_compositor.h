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
    struct Image {Id id=0;unsigned width=0,height=0;Format format=Format::rgb555;std::uint64_t revision=0;bool cpu_current=false,read_only=false;
        ComPtr<ID3D11Texture2D> texture;ComPtr<ID3D11ShaderResourceView> read;ComPtr<ID3D11UnorderedAccessView> write;};
    struct Constants {int area[4],offset[2];unsigned mode,color;};
    ID3D11Device* device;ID3D11DeviceContext* context;
    std::array<Image,128> images={};Image scratch,detail_scratch;Id serial=0;
    ImageDisplay display_program;
    ComPtr<ID3D11ComputeShader> shader,import_shader,unit_shader,image_shader,blend_shader,lookup_shader;ComPtr<ID3D11Buffer> constants,image_constants;
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
        auto d=find(command.destination);if(!d||d->read_only)return false;
        if(command.area.left>command.area.right || command.area.top>command.area.bottom ||
           command.clip.left>command.clip.right || command.clip.top>command.clip.bottom)return false;
        if(command.kind!=Kind::copy&&command.kind!=Kind::fill&&command.kind!=Kind::color_key&&command.kind!=Kind::invert&&command.kind!=Kind::quantize&&command.kind!=Kind::expand&&command.kind!=Kind::native_sprite&&command.kind!=Kind::unit_over&&command.kind!=Kind::native_text&&command.kind!=Kind::native_image&&command.kind!=Kind::native_blend&&command.kind!=Kind::native_lookup)return false;
        if(command.kind!=Kind::unit_over&&command.kind!=Kind::native_text&&command.kind!=Kind::native_image&&command.kind!=Kind::native_blend&&command.kind!=Kind::native_lookup&&(command.background||command.detail||command.background_detail))return false;
        if(command.kind!=Kind::native_image&&!(command.kind==Kind::native_blend&&command.color==2)&&(command.source_width||command.source_height))return false;
        if(command.kind!=Kind::native_lookup&&command.program)return false;
        if(command.kind==Kind::fill||command.kind==Kind::invert)return d->format==Format::bgra32||command.color<=65535;
        auto s=find(command.source);if(!s)return false;
        if(command.kind==Kind::native_lookup){
            auto b=find(command.background),detail=find(command.detail),bd=find(command.background_detail),program=find(command.program);
            if(d->format==Format::bgra32||s==d||s->format!=Format::bgra32||s->width!=1024||(s->height!=1024&&(s->height!=128||command.color!=32||!command.program))||
               !b||b->format!=d->format||b->width!=d->width||b->height!=d->height||(command.color>15&&!(command.program&&command.color==32)))return false;
            if(command.detail&&(!detail||detail==s||detail->read_only||detail->format!=Format::bgra32||detail->width!=d->width||detail->height!=d->height))return false;
            if(command.background_detail&&(!detail||!bd||bd->format!=Format::bgra32||bd->width!=b->width||bd->height!=b->height||((bd==detail)!=(b==d))))return false;
            if(!command.program)return true;
            auto r=selected(command,*d);auto x=std::int64_t(command.source_x)+r.left-command.area.left,y=std::int64_t(command.source_y)+r.top-command.area.top;
            return program&&program!=d&&program!=detail&&program->format==Format::bgra32&&x>=0&&y>=0&&x+r.right-r.left<=program->width&&y+r.bottom-r.top<=program->height;
        }
        if(command.kind==Kind::native_blend){
            auto b=find(command.background),detail=find(command.detail),bd=find(command.background_detail);
            if(d->format==Format::bgra32||!b||b->format!=d->format||command.color>3||
               (command.color==2?(s!=d||b!=d||command.source_width<0||command.source_width>65535||command.source_height<0||command.source_height>256):(s->format!=Format::bgra32||s==d)))return false;
            if(command.detail&&(!detail||detail->read_only||detail->format!=Format::bgra32||detail->width!=d->width||detail->height!=d->height||detail==s))return false;
            if(command.background_detail&&(!detail||!bd||bd->format!=Format::bgra32||bd->width!=b->width||bd->height!=b->height||((bd==detail)!=(b==d))))return false;
            auto r=selected(command,*d);
            if(r.right>int(b->width)||r.bottom>int(b->height))return false;
            if(command.color==2)return true;
        }else if(command.kind==Kind::native_image){
            auto detail=find(command.detail),sd=find(command.background_detail);
            auto dw=std::int64_t(command.area.right)-command.area.left,dh=std::int64_t(command.area.bottom)-command.area.top;
            if(d->format==Format::bgra32||s->format!=d->format||command.background||command.color>65536||
               command.source_width<=0||command.source_height<=0||dw<=0||dh<=0||dw>65535||dh>65535||
               command.source_x<0||command.source_y<0||std::int64_t(command.source_x)+command.source_width>s->width||
               std::int64_t(command.source_y)+command.source_height>s->height)return false;
            if(command.detail&&(!detail||detail->read_only||detail->format!=Format::bgra32||detail->width!=d->width||detail->height!=d->height))return false;
            if(command.background_detail&&(!detail||!sd||sd->format!=Format::bgra32||sd->width!=s->width||sd->height!=s->height||((sd==detail)!=(s==d))))return false;
            if(command.color!=65536&&(dw!=command.source_width||dh!=command.source_height))return false;
            return true;
        }
        if(command.kind==Kind::native_text){
            auto lut=find(command.background);
            if(!lut||lut==d||s==d||s->format!=Format::bgra32||lut->format!=Format::bgra32||lut->width!=17||lut->height>1024||command.detail||command.background_detail||command.color)return false;
        }else if(command.kind==Kind::unit_over){
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
        }else if(command.kind!=Kind::native_blend&&s->format!=d->format)return false;
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
    void native_lookup(Command const& op,Rect r){
        if(!lookup_shader){
            char const* program=R"(
cbuffer Params:register(b0){int4 area;int2 offset;uint mode;uint flags;};
Texture2D<uint> table:register(t0);Texture2D<uint> background:register(t1);Texture2D<uint> background_detail:register(t2);Texture2D<uint> coverage:register(t3);
RWTexture2D<uint> output:register(u0);RWTexture2D<uint> detail_output:register(u1);
uint3 channels(uint word){return uint3(word&31,(word>>5)&(mode?63:31),(word>>(mode?11:10))&31);}
uint3 expanded_channels(uint3 q){return (q<<uint3(3,mode?2:3,3))|(q>>uint3(2,mode?4:2,2));}
uint3 expanded(uint word){return expanded_channels(channels(word));}
uint mapped(uint word,uint block){uint index=(block<<15)|word;return table.Load(int3(index&1023,index>>10,0));}
float3 graded(uint full,uint block){
 uint3 rgb=uint3(full&255,(full>>8)&255,(full>>16)&255),q=rgb>>uint3(3,mode?2:3,3);
 q-=uint3(expanded_channels(q)>rgb);uint3 hi=min(q+1,uint3(31,mode?63:31,31));
 uint3 lo_rgb=expanded_channels(q),hi_rgb=expanded_channels(hi);float3 f=float3(rgb-lo_rgb)/float3(max(hi_rgb-lo_rgb,1));float3 result=0;
 [unroll] for(uint z=0;z<2;++z)[unroll] for(uint y=0;y<2;++y)[unroll] for(uint x=0;x<2;++x){
   uint3 c=uint3(x?hi.x:q.x,y?hi.y:q.y,z?hi.z:q.z);uint word=c.x|(c.y<<5)|(c.z<<(mode?11:10));
   result+=float3(expanded(mapped(word,block)))*(x?f.x:1-f.x)*(y?f.y:1-f.y)*(z?f.z:1-f.z);
 }return result;
}
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
 int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
 uint prior=output[at],block=flags>>8,opaque_word=0;bool from_background=false,black=false,opaque=false;
 if(flags&16){uint code=coverage.Load(int3(at+offset,0));if(code==0xffffffff)return;
   opaque=bool(code&65536);opaque_word=code&65535;block=code;from_background=!opaque&&prior==0x7c1f;
 }else if(flags&4){uint code=coverage.Load(int3(at+offset,0));if(code>15)return;black=code==0;block=code-1;}
 else from_background=prior==0x7c1f;
 uint word=from_background?((flags&8)?prior:background.Load(int3(at,0))):prior;
 uint result=opaque?opaque_word:black?0:mapped(word,block);
 if(flags&1){
   uint3 rgb=expanded(result);bool has_full=!from_background||bool(flags&2);
   if(!opaque&&!black&&has_full){uint full=from_background&&!(flags&8)?background_detail.Load(int3(at,0)):detail_output[at];
     // The native word remains exact. Evaluate the same color table between
     // its lattice entries for independent full-color map pixels.
     if(any(uint3(full&255,(full>>8)&255,(full>>16)&255)!=expanded(word)))rgb=uint3(clamp(round(graded(full,block)),0,255));
   }
   detail_output[at]=rgb.x|(rgb.y<<8)|(rgb.z<<16)|0xff000000;
 }
 output[at]=result;
})";
            ComPtr<ID3DBlob> code,error;checked(D3DCompile(program,std::strlen(program),"native lookup composition",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
            checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&lookup_shader));
        }
        auto d=find(op.destination),b=find(op.background),detail=find(op.detail),bd=find(op.background_detail);unbind();
        Constants params={{r.left,r.top,r.right,r.bottom},{op.program?op.source_x-op.area.left:0,op.program?op.source_y-op.area.top:0},d->format==Format::rgb565?1u:0u,
            (detail?1u:0u)|(bd?2u:0u)|(op.program?4u:0u)|(b==d?8u:0u)|(op.color==32?16u:op.color<<8)};
        context->UpdateSubresource(constants.Get(),0,nullptr,&params,0,0);auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
        ID3D11ShaderResourceView* reads[4]={find(op.source)->read.Get(),b==d?nullptr:b->read.Get(),bd&&bd!=detail?bd->read.Get():nullptr,op.program?find(op.program)->read.Get():nullptr};
        ID3D11UnorderedAccessView* writes[2]={d->write.Get(),detail?detail->write.Get():nullptr};
        context->CSSetShaderResources(0,4,reads);context->CSSetUnorderedAccessViews(0,2,writes,nullptr);context->CSSetShader(lookup_shader.Get(),nullptr,0);
        context->Dispatch(unsigned(r.right-r.left+7)/8,unsigned(r.bottom-r.top+7)/8,1);unbind();
        d->cpu_current=false;if(detail)detail->cpu_current=false;++counters.commands;
    }
    void native_blend(Command const& op,Rect r){
        if(!blend_shader){
            char const* program=R"(
cbuffer Params:register(b0){int4 area;int2 offset;uint mode;uint flags;};
Texture2D<uint> source:register(t0);Texture2D<uint> background:register(t1);Texture2D<uint> background_detail:register(t2);
RWTexture2D<uint> output:register(u0);RWTexture2D<uint> detail_output:register(u1);
uint3 unpack(uint c){return uint3(c&31,(c>>5)&(mode==1?63:31),(c>>(mode==1?11:10))&31);}
uint3 expanded(uint c){uint3 q=unpack(c);return (q<<uint3(3,mode==1?2:3,3))|(q>>uint3(2,mode==1?4:2,2));}
uint packed(uint3 rgb){uint3 q=rgb>>uint3(3,mode==1?2:3,3);return q.x|(q.y<<5)|(q.z<<(mode==1?11:10));}
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
 int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
 uint pixel=(flags&8)?uint(offset.x):source.Load(int3(at+offset,0)),weight=(flags&8)?uint(offset.y):pixel>>24;if(!(flags&8)&&weight==255)return;
 uint below=background.Load(int3(at,0)),word=pixel&65535,detail_weight=weight;
 if(flags&16){
   uint3 q=(unpack(word)*weight+unpack(below)*(16-weight))>>4;
   word=q.x|(q.y<<5)|(q.z<<(mode==1?11:10));detail_weight=(16-weight)*16;
 }else if(flags&8){
   uint3 rgb=unpack(word)<<uint3(3,mode==1?2:3,3);
   uint3 b=unpack(below)<<uint3(3,mode==1?2:3,3);
   word=packed((rgb*(256-weight)>>8)+(b*weight>>8));
 }else if(flags&4){
   uint3 rgb=uint3(pixel&255,(pixel>>8)&255,(pixel>>16)&255);
   // JGL first writes an opaque source at weight zero, then still executes its
   // (weight+1)/256 blend. It truncates each product separately.
   uint initial=weight?below:packed(rgb);detail_weight=weight?weight+1:0;
   uint3 b=unpack(initial)<<uint3(3,mode==1?2:3,3);
   word=packed((rgb*(255-weight)>>8)+(b*(weight+1)>>8));
 }else if(weight){
   // This native HUD helper always uses 5:5:5 channel arithmetic, including
   // its observable carry/overflow behavior, even with a 565 palette table.
   uint3 s=uint3(word&31,(word>>5)&31,(word>>10)&31)<<3;
   uint3 b=uint3(below&31,(below>>5)&31,(below>>10)&31)<<3;
   uint3 c=(s+(b*weight>>8))>>3;word=(c.x|(c.y<<5)|(c.z<<10))&65535;
 }
 output[at]=word;
 if(flags&1){
   int3 result=int3(expanded(word));
   // Native UI colors remain exact over a native-color background. Carry the
   // independent map precision through the same background contribution.
   if((flags&2)&&detail_weight){uint full=background_detail.Load(int3(at,0));
     int3 delta=int3(full&255,(full>>8)&255,(full>>16)&255)-int3(expanded(below));
     result+=delta*int(detail_weight)/256;
   }
   uint3 c=uint3(clamp(result,0,255));detail_output[at]=c.x|(c.y<<8)|(c.z<<16)|0xff000000;
 }
})";
            ComPtr<ID3DBlob> code,error;checked(D3DCompile(program,std::strlen(program),"native HUD blend",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
            checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&blend_shader));
        }
        auto d=find(op.destination),b=find(op.background),detail=find(op.detail),bd=find(op.background_detail);unbind();
        D3D11_BOX box={unsigned(r.left),unsigned(r.top),0,unsigned(r.right),unsigned(r.bottom),1};
        if(b==d){context->CopySubresourceRegion(scratch.texture.Get(),0,r.left,r.top,0,b->texture.Get(),0,&box);b=&scratch;++counters.snapshots;}
        if(bd&&bd==detail){context->CopySubresourceRegion(detail_scratch.texture.Get(),0,r.left,r.top,0,bd->texture.Get(),0,&box);bd=&detail_scratch;++counters.snapshots;}
        Constants params={{r.left,r.top,r.right,r.bottom},{int(std::int64_t(op.source_x)-op.area.left),int(std::int64_t(op.source_y)-op.area.top)},d->format==Format::rgb565?1u:0u,(detail?1u:0u)|(bd?2u:0u)|(op.color==3?16u:op.color==2?8u:op.color?4u:0u)};
        if(op.color==2){params.offset[0]=op.source_width;params.offset[1]=op.source_height;}
        context->UpdateSubresource(constants.Get(),0,nullptr,&params,0,0);auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
        ID3D11ShaderResourceView* reads[3]={op.color==2?nullptr:find(op.source)->read.Get(),b->read.Get(),bd?bd->read.Get():nullptr};
        ID3D11UnorderedAccessView* writes[2]={d->write.Get(),detail?detail->write.Get():nullptr};
        context->CSSetShaderResources(0,3,reads);context->CSSetUnorderedAccessViews(0,2,writes,nullptr);context->CSSetShader(blend_shader.Get(),nullptr,0);
        context->Dispatch(unsigned(r.right-r.left+7)/8,unsigned(r.bottom-r.top+7)/8,1);unbind();
        d->cpu_current=false;if(detail)detail->cpu_current=false;++counters.commands;
    }
    void native_image(Command const& op,Rect r){
        // JGL StretchBlt uses BLACKONWHITE. Shrink combines the pixels preceding
        // each center sample; enlargement takes the center sample alone. Clip
        // after deriving the original mapping, so scrolling preserves its phase.
        struct ImageConstants {int area[4],target[4],source[4];unsigned mode,key,flags,padding;};
        if(!image_shader){
            char const* program=R"(
cbuffer Params:register(b0){int4 area;int4 target;int4 source;uint mode;uint key;uint flags;uint padding;};
Texture2D<uint> native_input:register(t0);Texture2D<uint> detail_input:register(t1);
RWTexture2D<uint> native_output:register(u0);RWTexture2D<uint> detail_output:register(u1);
int2 interval(int at,int src,int dst){
 int last=((2*at+1)*src)/(2*dst);
 if(src<=dst)return int2(last,last+1);
 int first=at==0?0:((2*at-1)*src)/(2*dst)+1;
 return int2(first,last+1);
}
uint expanded(uint c){uint b=((c&31)<<3)|((c&31)>>2),r,g;
 if(mode==1){g=((c>>3)&252)|((c>>9)&3);r=((c>>8)&248)|((c>>13)&7);}
 else {g=((c>>2)&248)|((c>>7)&7);r=((c>>7)&248)|((c>>12)&7);}
 return b|(g<<8)|(r<<16)|0xff000000;
}
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
 int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
 int2 xr=interval(at.x-target.x,source.z,target.z),yr=interval(at.y-target.y,source.w,target.w);
 uint word=65535,full=0xffffffff;
 for(int y=yr.x;y<yr.y;++y)for(int x=xr.x;x<xr.y;++x){
   int3 sample=int3(source.xy+int2(x,y),0);word&=native_input.Load(sample);
   if(flags&2)full&=detail_input.Load(sample);
 }
 if(word==key)return;
 native_output[at]=word;
 if(flags&1)detail_output[at]=(flags&2)?full:expanded(word);
})";
            ComPtr<ID3DBlob> code,error;checked(D3DCompile(program,std::strlen(program),"native image transfer",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
            checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&image_shader));
            D3D11_BUFFER_DESC desc={};desc.ByteWidth=sizeof(ImageConstants);desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            checked(device->CreateBuffer(&desc,nullptr,&image_constants));
        }
        auto d=find(op.destination),s=find(op.source),detail=find(op.detail),sd=find(op.background_detail);unbind();
        if(s==d){D3D11_BOX box={0,0,0,d->width,d->height,1};context->CopySubresourceRegion(scratch.texture.Get(),0,0,0,0,d->texture.Get(),0,&box);s=&scratch;++counters.snapshots;
            if(sd){context->CopySubresourceRegion(detail_scratch.texture.Get(),0,0,0,0,sd->texture.Get(),0,&box);sd=&detail_scratch;++counters.snapshots;}}
        ImageConstants params={{r.left,r.top,r.right,r.bottom},{op.area.left,op.area.top,op.area.right-op.area.left,op.area.bottom-op.area.top},
            {op.source_x,op.source_y,op.source_width,op.source_height},d->format==Format::rgb565?1u:0u,op.color,(detail?1u:0u)|(sd?2u:0u),0};
        context->UpdateSubresource(image_constants.Get(),0,nullptr,&params,0,0);auto cb=image_constants.Get();context->CSSetConstantBuffers(0,1,&cb);
        ID3D11ShaderResourceView* reads[2]={s->read.Get(),sd?sd->read.Get():nullptr};ID3D11UnorderedAccessView* writes[2]={d->write.Get(),detail?detail->write.Get():nullptr};
        context->CSSetShaderResources(0,2,reads);context->CSSetUnorderedAccessViews(0,2,writes,nullptr);context->CSSetShader(image_shader.Get(),nullptr,0);
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
Texture2D<uint> input_image:register(t0);Texture2D<uint> text_curves:register(t1);RWTexture2D<uint> output_image:register(u0);
[numthreads(8,8,1)] void main(uint3 thread:SV_DispatchThreadID){
 int2 at=area.xy+int2(thread.xy);if(any(at>=area.zw))return;
 uint value=color;if(mode!=1)value=input_image.Load(int3(at+offset,0));
 if(mode>=12){
   uint below=output_image[at],rgb=below,result=0;
   if(mode!=14){uint b=((below&31)<<3)|((below&31)>>2),r,g;
     if(mode==13){g=((below>>3)&252)|((below>>9)&3);r=((below>>8)&248)|((below>>13)&7);}
     else {g=((below>>2)&248)|((below>>7)&7);r=((below>>7)&248)|((below>>12)&7);}
     rgb=b|(g<<8)|(r<<16);
   }
   [unroll]for(uint c=0;c<3;++c){
     uint level=(rgb>>(c*8))&255,index=min(level/16,15),fraction=level-index*16,span=index==15?15:16,id=(value>>(c*10))&1023;
     uint a=text_curves.Load(int3(index,id,0)),b=text_curves.Load(int3(index+1,id,0));
     result|=((a*(span-fraction)+b*fraction+span/2)/span)<<(8*c);
   }
   if(mode==14)result|=0xff000000;
   else {if(mode==13)result=(result>>3&31)|((result>>10&63)<<5)|((result>>19&31)<<11);
     else result=(result>>3&31)|((result>>11&31)<<5)|((result>>19&31)<<10);
     if(mode==12&&!(value&0x40000000))result|=below&32768;
   }
   output_image[at]=result;return;
 }
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
    // Borrow an immutable finished unit from its pose owner. COM retains the
    // texture through queued draws; this creates no texture copy or CPU upload.
    Id attach_source(ID3D11Texture2D* texture){
        if(!texture)return 0;D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
        ComPtr<ID3D11Device> owner;texture->GetDevice(&owner);
        if(owner.Get()!=device||d.Format!=DXGI_FORMAT_R32_UINT||d.SampleDesc.Count!=1||d.ArraySize!=1||d.MipLevels!=1||
           !(d.BindFlags&D3D11_BIND_SHADER_RESOURCE)||!d.Width||!d.Height||d.Width>1024||d.Height>1024||std::uint64_t(d.Width)*d.Height*4>budget-counters.resident_bytes)return 0;
        for(auto& image:images)if(!image.id){Image next;next.texture=texture;next.width=d.Width;next.height=d.Height;next.format=Format::bgra32;next.read_only=true;
            checked(device->CreateShaderResourceView(texture,nullptr,&next.read));next.id=++serial;counters.resident_bytes+=bytes(next);image=std::move(next);return image.id;
        }return 0;
    }
    bool destroy(Id id){auto image=find(id);if(!image)return false;unbind();counters.resident_bytes-=bytes(*image);*image={};return true;}
    bool upload(Id id,std::uint64_t revision,std::uint32_t const* pixels,std::size_t count){
        auto image=find(id);if(!image||image->read_only||!revision||!pixels||count!=std::size_t(image->width)*image->height)return false;
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
            if((op.kind==Kind::unit_over||(op.kind==Kind::native_image&&op.source==op.destination)||(op.kind==Kind::native_blend&&op.background_detail==op.detail))&&op.detail){auto image=find(op.detail);detail_width=std::max(detail_width,image->width);detail_height=std::max(detail_height,image->height);}
            if(op.kind==Kind::unit_over || (op.kind==Kind::native_blend&&op.background==op.destination) || op.kind==Kind::invert || (op.kind!=Kind::fill&&op.source==op.destination)){
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
            if(op.kind==Kind::native_image){native_image(op,r);continue;}
            if(op.kind==Kind::native_blend){native_blend(op,r);continue;}
            if(op.kind==Kind::native_lookup){native_lookup(op,r);continue;}
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
            Constants p={{r.left,r.top,r.right,r.bottom},{0,0},op.kind==Kind::fill?1u:op.kind==Kind::color_key?(d->format==Format::bgra32?4u:2u):op.kind==Kind::invert?3u:op.kind==Kind::quantize?(d->format==Format::rgb565?6u:5u):op.kind==Kind::expand?(s->format==Format::rgb565?8u:7u):op.kind==Kind::native_sprite?9u+op.color:op.kind==Kind::native_text?12u+(d->format==Format::rgb555?0u:d->format==Format::rgb565?1u:2u):0u,op.color};
            if(op.kind!=Kind::fill&&op.kind!=Kind::invert){p.offset[0]=int(std::int64_t(op.source_x)-op.area.left);p.offset[1]=int(std::int64_t(op.source_y)-op.area.top);}
            context->UpdateSubresource(constants.Get(),0,nullptr,&p,0,0);auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
            auto read=s?s->read.Get():nullptr;auto write=d->write.Get();ID3D11ShaderResourceView* reads[2]={read,op.kind==Kind::native_text?find(op.background)->read.Get():nullptr};context->CSSetShaderResources(0,2,reads);context->CSSetUnorderedAccessViews(0,1,&write,nullptr);
            context->CSSetShader(shader.Get(),nullptr,0);context->Dispatch(unsigned(r.right-r.left+7)/8,unsigned(r.bottom-r.top+7)/8,1);unbind();
            // GPU mutation invalidates a prior CPU revision; the caller must supply
            // a strictly newer revision to replace this image from CPU content.
            d->cpu_current=false;++counters.commands;
        }return true;
    }
    // GPU-to-GPU import of a completed display texture; no staging or CPU seed.
    bool import_bgra(Id id,ID3D11Texture2D* source,int x=0,int y=0){
        auto destination=find(id);if(!destination||!source||destination->format!=Format::bgra32)return false;
        D3D11_TEXTURE2D_DESC desc={};source->GetDesc(&desc);
        if(x<0||y<0||desc.Width<destination->width||desc.Height<destination->height||
           unsigned(x)>desc.Width-destination->width||unsigned(y)>desc.Height-destination->height||desc.SampleDesc.Count!=1||
           desc.Format!=DXGI_FORMAT_B8G8R8A8_UNORM||!(desc.BindFlags&D3D11_BIND_SHADER_RESOURCE))return false;
        ComPtr<ID3D11Device> source_device;source->GetDevice(&source_device);if(source_device.Get()!=device)return false;
        if(!import_shader){
            char const* hlsl=R"(
cbuffer Params:register(b0){int4 area;int2 offset;uint mode;uint color;};
Texture2D<float4> input_image:register(t0);RWTexture2D<uint> output_image:register(u0);
[numthreads(8,8,1)] void main(uint3 at:SV_DispatchThreadID){
 uint w,h;output_image.GetDimensions(w,h);if(at.x>=w||at.y>=h)return;
 uint4 c=uint4(round(saturate(input_image.Load(int3(int2(at.xy)+offset,0)))*255.0));
 output_image[at.xy]=c.b|(c.g<<8)|(c.r<<16)|(c.a<<24);
})";
            ComPtr<ID3DBlob> code,error;checked(D3DCompile(hlsl,std::strlen(hlsl),"resident map import",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
            checked(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&import_shader));
        }
        ComPtr<ID3D11ShaderResourceView> input;checked(device->CreateShaderResourceView(source,nullptr,&input));
        unbind();auto read=input.Get();auto write=destination->write.Get();context->CSSetShaderResources(0,1,&read);context->CSSetUnorderedAccessViews(0,1,&write,nullptr);
        if(!constants){D3D11_BUFFER_DESC d={};d.ByteWidth=sizeof(Constants);d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;checked(device->CreateBuffer(&d,nullptr,&constants));}
        Constants params={};params.offset[0]=x;params.offset[1]=y;
        context->UpdateSubresource(constants.Get(),0,nullptr,&params,0,0);auto cb=constants.Get();context->CSSetConstantBuffers(0,1,&cb);
        context->CSSetShader(import_shader.Get(),nullptr,0);context->Dispatch((destination->width+7)/8,(destination->height+7)/8,1);unbind();
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
