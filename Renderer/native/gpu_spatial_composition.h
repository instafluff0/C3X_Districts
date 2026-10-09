#pragma once
#include "gpu_image_commands.h"
#include "composition_storage.h"
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <vector>
#include <map>
#include <array>
#include <functional>
#include <unordered_set>
#include <cstring>
#include <string>
#include <stdexcept>

namespace c3x_gpu_images {
// A captured command sequence is still the authority. Each tile executes its
// ordered pointwise subsequence, carrying both native words and map detail in
// registers. Cross-position reads stay in the original interpreter.
class SpatialComposition {
    using MicrosoftTexture=Microsoft::WRL::ComPtr<ID3D11Texture2D>;
    template<class T> using Ptr=Microsoft::WRL::ComPtr<T>;
    static constexpr unsigned page_size=1024,tile_size=32,max_pages=8,max_commands=8192,max_indices=1048576;
    struct GpuCommand {int area[4],source[4],auxiliary[4];unsigned kind,color,target,source_page,auxiliary_page,flags,mode,padding;};
    struct Tile {unsigned x,y,offset,count;};
    struct Run {unsigned command=0,count=0,tile=0,tiles=0;bool spatial=false;};
    struct Source {MicrosoftTexture texture;Rect area{};unsigned x=0,y=0,page=0;};
    struct SourceKey {ID3D11Texture2D* texture=nullptr;Rect area{};unsigned x=0,y=0,page=0;};
    ID3D11Device* device;ID3D11DeviceContext* context;
    Ptr<ID3D11ComputeShader> shader,fused_shader;Ptr<ID3D11Buffer> constants,fused_constants;
    CompositionStorage allocations,physical;
    std::uint64_t dispatches_=0,commands_=0,compilations_=0,source_copies_=0;
    static void check(HRESULT result){if(FAILED(result))throw std::runtime_error("spatial native composition resource");}
    static bool empty(Rect r){return r.left>=r.right||r.top>=r.bottom;}
    static bool same(Rect a,Rect b){return a.left==b.left&&a.top==b.top&&a.right==b.right&&a.bottom==b.bottom;}
    void program(){
        if(shader && constants && fused_shader && fused_constants)return;
        char const* common=R"(
struct Op {int4 area;int4 source;int4 auxiliary;uint kind,color,target,source_page,auxiliary_page,flags,mode,padding;};
struct Tile {uint x,y,offset,count;};
cbuffer Params:register(b0){uint first_tile,width,height,padding;};
Texture2DArray<uint> inputs:register(t0);StructuredBuffer<Op> commands:register(t1);
StructuredBuffer<Tile> tiles:register(t2);StructuredBuffer<uint> order:register(t3);
RWTexture2D<uint> words:register(u0);RWTexture2D<uint> detail:register(u1);
// Three R32_UINT slices keep typed UAV loads valid on D3D11.0 hardware.
// Cache only pixels proved independent of BOTH changing before-images.
RWTexture2DArray<uint> hud:register(u2);
uint3 channels(uint v,uint mode){return uint3(v&31,(v>>5)&(mode==1?63:31),(v>>(mode==1?11:10))&31);}
uint3 expanded_channels(uint3 q,uint mode){return (q<<uint3(3,mode==1?2:3,3))|(q>>uint3(2,mode==1?4:2,2));}
uint3 expanded_native(uint v,uint mode){uint b=((v&31)<<3)|((v&31)>>2),r,g;
 if(mode==1){g=((v>>3)&252)|((v>>9)&3);r=((v>>8)&248)|((v>>13)&7);}
 else {g=((v>>2)&248)|((v>>7)&7);r=((v>>7)&248)|((v>>12)&7);}return uint3(b,g,r);}
uint3 expanded(uint v,uint mode){return expanded_channels(channels(v,mode),mode);}
uint full_color(uint v,uint mode){uint3 c=expanded_native(v,mode);return c.x|(c.y<<8)|(c.z<<16)|0xff000000;}
bool native_keyed(uint c){return (c&0xf800f8)==0xf800f8&&(c&0xf800)==0;}
uint unit_mixed(uint source,uint below,uint alpha){
 uint3 s=uint3(source&255,(source>>8)&255,(source>>16)&255),b=uint3(below&255,(below>>8)&255,(below>>16)&255);
 uint3 c=min(s+(b*(255-alpha)+127)/255,255);uint result=c.x|(c.y<<8)|(c.z<<16);if(native_keyed(result))result^=0x800;return result;
}
uint packed(uint3 c,uint mode){uint3 q=c>>uint3(3,mode==1?2:3,3);return q.x|(q.y<<5)|(q.z<<(mode==1?11:10));}
uint read_source(Op op,int2 at,uint native,uint full){
 if(op.flags&4)return native;if(op.flags&8)return full;
 return inputs.Load(int4(at+op.source.xy,op.source_page,0));
}
uint mapped(Op op,uint v,uint block){uint index=(block<<15)|v;return inputs.Load(int4(op.source.zw+int2(index&1023,index>>10),op.source_page,0));}
float3 graded(Op op,uint full,uint block){
 uint3 rgb=uint3(full&255,(full>>8)&255,(full>>16)&255),q=rgb>>uint3(3,op.mode==1?2:3,3);
 q-=uint3(expanded_channels(q,op.mode)>rgb);uint3 hi=min(q+1,uint3(31,op.mode==1?63:31,31));
 uint3 lo_rgb=expanded_channels(q,op.mode),hi_rgb=expanded_channels(hi,op.mode);
 float3 f=float3(rgb-lo_rgb)/float3(max(hi_rgb-lo_rgb,1));float3 result=0;
 [unroll]for(uint z=0;z<2;++z)[unroll]for(uint y=0;y<2;++y)[unroll]for(uint x=0;x<2;++x){
  uint3 c=uint3(x?hi.x:q.x,y?hi.y:q.y,z?hi.z:q.z);uint v=c.x|(c.y<<5)|(c.z<<(op.mode==1?11:10));
  result+=float3(expanded(mapped(op,v,block),op.mode))*(x?f.x:1-f.x)*(y?f.y:1-f.y)*(z?f.z:1-f.z);
 }return result;
}
void apply(Op op,int2 at,inout uint native,inout uint full){
 uint prior=op.target?full:native,value=op.color;
 if(op.kind!=1&&op.kind!=3&&!(op.kind==10&&op.color==2)&&op.kind!=11)value=read_source(op,at,native,full);
 if(op.kind==7){
  uint alpha=value>>24;if(!alpha)return;uint below=full_color(native,op.mode);
  // An aliased native ground is the same captured point. A keyed value
  // cannot supply the missing ground, just as in the unit interpreter.
  if(alpha<255&&native_keyed(below))return;
  uint color=unit_mixed(value,below,alpha);native=packed(uint3(color&255,(color>>8)&255,(color>>16)&255),op.mode);
  if(op.flags&1)full=unit_mixed(value,full,alpha)|0xff000000;return;
 }
 if(op.kind==8){
  uint rgb=prior,result=0;
  if(op.mode!=2){uint3 c=expanded_native(prior,op.mode);rgb=c.x|(c.y<<8)|(c.z<<16);}
  [unroll]for(uint c=0;c<3;++c){
   uint level=(rgb>>(c*8))&255,index=min(level/16,15),fraction=level-index*16,span=index==15?15:16,id=(value>>(c*10))&1023;
   uint a=inputs.Load(int4(op.auxiliary.xy+int2(index,id),op.auxiliary_page,0));
   uint b=inputs.Load(int4(op.auxiliary.xy+int2(index+1,id),op.auxiliary_page,0));
   result|=((a*(span-fraction)+b*fraction+span/2)/span)<<(8*c);
  }
  if(op.mode==2)full=result|0xff000000;
  else {result=packed(uint3(result&255,(result>>8)&255,(result>>16)&255),op.mode);
   if(op.mode==0&&!(value&0x40000000))result|=prior&32768;native=result;}
  return;
 }
 if(op.kind==10){
  uint weight=op.color==2?uint(op.auxiliary.y):value>>24;
  if(op.color!=2&&op.color!=4&&weight==255)return;
  uint below=native,v=op.color==2?uint(op.auxiliary.x):value&65535,detail_weight=weight;
  if(op.color==4){if(below!=(value>>16))return;detail_weight=0;}
  else if(op.color==3){uint3 q=(channels(v,op.mode)*weight+channels(below,op.mode)*(16-weight))>>4;
   v=q.x|(q.y<<5)|(q.z<<(op.mode==1?11:10));detail_weight=(16-weight)*16;}
  else if(op.color==2){uint3 rgb=channels(v,op.mode)<<uint3(3,op.mode==1?2:3,3);
   uint3 b=channels(below,op.mode)<<uint3(3,op.mode==1?2:3,3);v=packed((rgb*(256-weight)>>8)+(b*weight>>8),op.mode);}
  else if(op.color==1){uint3 rgb=uint3(value&255,(value>>8)&255,(value>>16)&255);
   uint initial=weight?below:packed(rgb,op.mode);detail_weight=weight?weight+1:0;
   uint3 b=channels(initial,op.mode)<<uint3(3,op.mode==1?2:3,3);v=packed((rgb*(255-weight)>>8)+(b*(weight+1)>>8),op.mode);}
  else if(weight){uint3 s=uint3(v&31,(v>>5)&31,(v>>10)&31)<<3;
   uint3 b=uint3(below&31,(below>>5)&31,(below>>10)&31)<<3;uint3 c=(s+(b*weight>>8))>>3;
   v=(c.x|(c.y<<5)|(c.z<<10))&65535;}
  native=v;
  if(op.flags&1){int3 result=int3(expanded(v,op.mode));
   if((op.flags&2)&&detail_weight){int3 delta=int3(full&255,(full>>8)&255,(full>>16)&255)-int3(expanded(below,op.mode));result+=delta*int(detail_weight)/256;}
   uint3 c=uint3(clamp(result,0,255));full=c.x|(c.y<<8)|(c.z<<16)|0xff000000;
  }return;
 }
 if(op.kind==11){
  uint block=op.color,opaque_word=0;bool black=false,opaque=false;
  if(op.flags&16){uint code=inputs.Load(int4(at+op.auxiliary.xy,op.auxiliary_page,0));
   if(op.color==32){if(code==0xffffffff)return;opaque=bool(code&65536);opaque_word=code&65535;block=code;}
   else {if(code>15)return;black=code==0;block=code-1;}
  }
  uint result=opaque?opaque_word:black?0:mapped(op,native,block);
  if(op.flags&1){uint3 rgb=expanded(result,op.mode);
   bool from_background=(!(op.flags&16)||op.color==32)&&!opaque&&native==0x7c1f;
   if(!opaque&&!black&&(!from_background||bool(op.flags&2))&&any(uint3(full&255,(full>>8)&255,(full>>16)&255)!=expanded(native,op.mode)))rgb=uint3(clamp(round(graded(op,full,block)),0,255));
   full=rgb.x|(rgb.y<<8)|(rgb.z<<16)|0xff000000;
  }native=result;return;
 }
 if(op.kind==3)value=prior^op.color;
 if(op.kind==2&&(op.mode==2?(value&0xffffff)==(op.color&0xffffff):value==op.color))return;
 if(op.kind==9){
  value&=65535;
  if(value==op.color)return;native=value;
  if(op.flags&1)full=(op.flags&2)?((op.flags&32)?full:inputs.Load(int4(at+op.auxiliary.xy,op.auxiliary_page,0))):full_color(value,op.mode);
  return;
 }
 if(op.kind==4){
  uint2 xy=uint2(at+op.source.zw)-uint2(op.color&7,(op.color>>3)&7);uint threshold=0;
  [unroll]for(uint bit=0;bit<3;++bit){uint a=(xy.x>>bit)&1,b=(xy.y>>bit)&1;threshold=(threshold<<2)|((a^b)<<1)|b;}
  uint3 levels=uint3(31,op.mode==1?63:31,31),scaled=uint3(value&255,(value>>8)&255,(value>>16)&255)*levels;
  uint3 q=scaled/255+uint3((scaled%255)*128>(threshold*2+1)*255);value=q.x|(q.y<<5)|(q.z<<(op.mode==1?11:10));
 }
 if(op.kind==6){if(!(value&65536))return;value&=65535;if(op.color)value=full_color(value,op.color==2?1:0);}
 if(op.kind==5){if(value==op.color)return;value=full_color(value,op.source[2]);}
 if(op.target)full=value;else native=value;
}
void dependencies(Op op,int2 at,uint native,uint full,inout bool dn,inout bool df){
 bool source_dependent=((op.flags&4)!=0&&dn)||((op.flags&8)!=0&&df);
 if(op.kind==7||op.kind==10||op.kind==11){bool depends=dn||df||source_dependent;dn=depends;df=depends;return;}
 if(op.kind==8){if(op.target)df=df||source_dependent;else dn=dn||source_dependent;return;}
 if(op.kind==3)return; // XOR depends only on its prior target.
 uint value=op.kind==1?op.color:read_source(op,at,native,full);
 bool assigned=op.kind==0||op.kind==1||op.kind==4;
 if(op.kind==2)assigned=op.mode==2?(value&0xffffff)!=(op.color&0xffffff):value!=op.color;
 if(op.kind==5)assigned=value!=op.color;
 if(op.kind==6)assigned=(value&65536)!=0;
 if(op.kind==9){
  // Key tests on a changing source can themselves vary on the next frame.
  if(source_dependent){bool depends=dn||df;dn=depends;if(op.flags&1)df=depends;return;}
  if((value&65535)==op.color)return;
  dn=false;
  if((op.flags&1)&&!((op.flags&2)&&(op.flags&32)))df=false;
  return;
 }
 bool depends=source_dependent;
 // A conditional write sourced from a before-image may become a no-op.
 if((op.kind==2||op.kind==5||op.kind==6)&&source_dependent){if(op.target)df=true;else dn=true;return;}
 if(op.kind==1)depends=false;
 if(assigned){if(op.target)df=depends;else dn=depends;}
}
)";
        char const* hud_main=R"(
[numthreads(8,8,1)]void main(uint3 group:SV_GroupID,uint3 thread:SV_GroupThreadID){
 Tile tile=tiles[first_tile+group.x];
 [loop]for(uint y=0;y<4;++y)[loop]for(uint x=0;x<4;++x){
  int2 at=int2(tile.x*32+thread.x+x*8,tile.y*32+thread.y+y*8);if(at.x>=int(width)||at.y>=int(height))continue;
  uint classification=padding!=0?hud[int3(at,2)]:2;
  if(classification==1){words[at]=hud[int3(at,0)];detail[at]=hud[int3(at,1)];continue;}
  uint native=words[at],full=detail[at];bool dn=true,df=true;
  for(uint n=0;n<tile.count;++n){Op op=commands[order[tile.offset+n]];if(all(at>=op.area.xy)&&all(at<op.area.zw)){
   if(classification==0)dependencies(op,at,native,full,dn,df);
   apply(op,at,native,full);
  }}
  words[at]=native;detail[at]=full;
  // Dependence classification is invariant for this immutable program:
  // conditional writes from changing inputs never clear their dependence.
  if(classification==0){
   if(!dn && !df){hud[int3(at,0)]=native;hud[int3(at,1)]=full;hud[int3(at,2)]=1;}
   else hud[int3(at,2)]=2;
  }
 }
})";
        // The whole interface over the scene in one pass (stage 3): each pixel
        // starts from the projected scene, takes the world view's native word
        // (the view transform's ordered quantization), runs its tile's HUD
        // program and the keyed screen-canvas transfer, and writes the front.
        char const* fused_main=R"(
cbuffer Fused:register(b1){uint columns,native_mode,key,transfer_flags;};
Texture2D<uint> scene:register(t4);Texture2D<uint> screen_words:register(t5);Texture2D<uint> screen_detail:register(t6);
StructuredBuffer<Tile> table:register(t7);RWTexture2D<uint> front:register(u3);RWTexture2D<uint> front_words:register(u4);
[numthreads(8,8,1)]void fused(uint3 id:SV_DispatchThreadID){
 int2 at=int2(id.xy);if(at.x>=int(width)||at.y>=int(height))return;
 uint full=scene.Load(int3(at,0));
 uint threshold=0;
 [unroll]for(uint bit=0;bit<3;++bit){uint a=(uint(at.x)>>bit)&1,b=(uint(at.y)>>bit)&1;threshold=(threshold<<2)|((a^b)<<1)|b;}
 uint3 c=uint3(full&255,(full>>8)&255,(full>>16)&255),levels=uint3(31,native_mode==1?63:31,31),scaled=c*levels;
 uint3 q=scaled/255+uint3((scaled%255)*128>(threshold*2+1)*255);
 uint native=q.x|(q.y<<5)|(q.z<<(native_mode==1?11:10));
 Tile tile=table[(uint(at.y)/32)*columns+uint(at.x)/32];
 for(uint n=0;n<tile.count;++n){Op op=commands[order[tile.offset+n]];if(all(at>=op.area.xy)&&all(at<op.area.zw))apply(op,at,native,full);}
 uint value=screen_words.Load(int3(at,0))&65535;
 if(value!=key){native=value;
  if(transfer_flags&1)full=(transfer_flags&2)?((transfer_flags&32)?full:screen_detail.Load(int3(at,0))):full_color(value,native_mode);}
 front[at]=full;front_words[at]=native;
}
)";
        auto compile_entry=[&](char const* entry_source,char const* entry,Ptr<ID3D11ComputeShader>& target){
            std::string source=std::string(common)+entry_source;Ptr<ID3DBlob> code,error;
            auto hr=D3DCompile(source.c_str(),source.size(),"ordered spatial native composition",nullptr,nullptr,entry,"cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error);
            if(FAILED(hr)){if(error)OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));check(hr);}
            check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&target));
        };
        if(!shader)compile_entry(hud_main,"main",shader);
        if(!fused_shader)compile_entry(fused_main,"fused",fused_shader);
        D3D11_BUFFER_DESC d={};d.ByteWidth=16;d.Usage=D3D11_USAGE_DEFAULT;d.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        if(!constants)check(device->CreateBuffer(&d,nullptr,&constants));
        if(!fused_constants)check(device->CreateBuffer(&d,nullptr,&fused_constants));
    }
public:
    struct Image {ID3D11Texture2D* texture=nullptr;unsigned width=0,height=0;Format format=Format::bgra32;};
    struct Plan {
        Id words=0,detail=0;Format words_format=Format::rgb555;unsigned width=0,height=0;std::vector<Command> original;std::vector<Run> runs;
        MicrosoftTexture atlas;Ptr<ID3D11ShaderResourceView> atlas_view;
        MicrosoftTexture hud;Ptr<ID3D11UnorderedAccessView> hud_write;
        Ptr<ID3D11Buffer> commands,tiles,order;Ptr<ID3D11ShaderResourceView> command_view,tile_view,order_view;
        // Every 32-pixel tile of a single spatial run, touched or not, for the
        // fused interface pass (fused()); empty otherwise.
        Ptr<ID3D11Buffer> table;Ptr<ID3D11ShaderResourceView> table_view;unsigned columns=0;
        std::array<CompositionStorage::Lease,6> owned,physical;
        std::vector<SourceKey> source_keys;std::vector<ID3D11Texture2D*> external_allocations;
        std::uint64_t bytes=0;unsigned spatial_commands=0,spatial_runs=0,fallback_commands=0;
        Plan()=default;Plan(Plan&&)=default;Plan& operator=(Plan&&)=default;
        Plan(Plan const&)=delete;Plan& operator=(Plan const&)=delete;
        explicit operator bool()const{return !original.empty();}
    };
    SpatialComposition(ID3D11Device* d,ID3D11DeviceContext* c):device(d),context(c){}
    void prepare_assets(){program();}
    void share_storage(CompositionStorage const& tracker){physical=tracker;}
    std::uint64_t bytes()const{return allocations.bytes();}
    std::uint64_t dispatches()const{return dispatches_;}
    std::uint64_t commands()const{return commands_;}
    std::uint64_t compilations()const{return compilations_;}
    std::uint64_t source_copies()const{return source_copies_;}
    template<class Resolve,class Admit> bool compile(Plan& old,Command const* commands,std::size_t count,Id words,Id detail,Resolve resolve,Admit admit,bool reuse_sources=false){
        if(!commands||!count||count>max_commands)return false;
        auto w=resolve(words),d=resolve(detail);
        if(!w.texture||!d.texture||w.texture==d.texture||w.format==Format::bgra32||d.format!=Format::bgra32||w.width!=d.width||w.height!=d.height)return false;
        Plan next;next.words=words;next.detail=detail;next.words_format=w.format;next.width=w.width;next.height=w.height;next.original.assign(commands,commands+count);
        std::vector<GpuCommand> encoded(count);std::vector<Tile> tiles;std::vector<unsigned> indices;std::vector<Source> sources;
        unsigned cursor_x=0,cursor_y=0,row_height=0,page=0;
        // An interpreter boundary may write another image which a later
        // command reads. Only sources immutable throughout this whole plan may
        // enter the copied atlas, including aliases of a written allocation.
        std::unordered_set<ID3D11Texture2D*> written,external;
        for(unsigned n=0;n<count;++n)for(Id id:{commands[n].destination,commands[n].detail})
            if(id){auto image=resolve(id);if(image.texture)written.insert(image.texture);}
        for(unsigned n=0;n<count;++n){auto const& c=commands[n];for(Id id:{c.destination,c.source,c.background,c.detail,c.background_detail,c.program})
            if(id&&id!=words&&id!=detail){auto image=resolve(id);if(!image.texture||image.texture==w.texture||image.texture==d.texture)return false;
                if(external.insert(image.texture).second)next.external_allocations.push_back(image.texture);}}
        auto input=[&](Id id,Rect region,int* mapping,unsigned& slice)->bool{
            auto image=resolve(id);if(!image.texture||written.count(image.texture)||empty(region)||region.left<0||region.top<0||region.right>int(image.width)||region.bottom>int(image.height))return false;
            auto width=unsigned(region.right-region.left),height=unsigned(region.bottom-region.top);if(width>page_size||height>page_size)return false;
            for(auto const& source:sources)if(source.texture.Get()==image.texture&&same(source.area,region)){
                mapping[0]=int(source.x)-region.left;mapping[1]=int(source.y)-region.top;mapping[2]=int(source.x);mapping[3]=int(source.y);slice=source.page;return true;}
            if(cursor_x+width>page_size){cursor_x=0;cursor_y+=row_height;row_height=0;}
            if(cursor_y+height>page_size){++page;cursor_x=cursor_y=row_height=0;}
            if(page>=max_pages)return false;
            sources.push_back({MicrosoftTexture(image.texture),region,cursor_x,cursor_y,page});
            mapping[0]=int(cursor_x)-region.left;mapping[1]=int(cursor_y)-region.top;mapping[2]=int(cursor_x);mapping[3]=int(cursor_y);slice=page;
            cursor_x+=width;row_height=std::max(row_height,height);return true;
        };
        std::vector<bool> supported(count);
        for(unsigned n=0;n<count;++n){auto const& c=commands[n];auto& op=encoded[n];
            auto r=intersection(intersection(c.area,c.clip),{0,0,int(w.width),int(w.height)});
            op.area[0]=r.left;op.area[1]=r.top;op.area[2]=r.right;op.area[3]=r.bottom;
            op.kind=unsigned(c.kind);op.color=c.color;op.target=c.destination==detail;op.mode=op.target?2u:w.format==Format::rgb565?1u:0u;
            op.flags=(c.detail?1u:0u)|(c.background_detail?2u:0u);
            if(c.destination!=words&&c.destination!=detail)continue;
            if(c.detail&&c.detail!=detail)continue;
            if(c.kind>=Kind::world_begin)continue;
            if(c.kind==Kind::unit_over&&(c.background!=words||(c.background_detail&&c.background_detail!=detail)))continue;
            if((c.kind==Kind::native_blend||c.kind==Kind::native_lookup)&&(c.background!=words||(c.background_detail&&c.background_detail!=detail)))continue;
            if(c.kind==Kind::native_image&&(c.source_width!=c.area.right-c.area.left||c.source_height!=c.area.bottom-c.area.top))continue;
            bool ok=true;auto previous_sources=sources.size();auto previous_x=cursor_x,previous_y=cursor_y,previous_row=row_height,previous_page=page;
            auto footprint=[&](Id id,int dx,int dy){auto image=resolve(id);return intersection({c.area.left+dx,c.area.top+dy,c.area.right+dx,c.area.bottom+dy},{0,0,int(image.width),int(image.height)});};
            if(!empty(r)&&c.kind!=Kind::fill&&c.kind!=Kind::invert&&c.kind!=Kind::native_lookup&&!(c.kind==Kind::native_blend&&c.color==2)){
                int dx=c.source_x-c.area.left,dy=c.source_y-c.area.top;
                if(c.source==words||c.source==detail){if(dx||dy)ok=false;else op.flags|=c.source==words?4u:8u;}
                else {Rect area=footprint(c.source,dx,dy);ok=input(c.source,area,op.source,op.source_page);
                    op.source[0]+=dx;op.source[1]+=dy;}
                if(c.kind==Kind::quantize){op.source[2]=dx;op.source[3]=dy;}
                if(c.kind==Kind::expand)op.source[2]=resolve(c.source).format==Format::rgb565?1:0;
            }
            if(ok&&!empty(r)&&c.kind==Kind::native_text){auto lut=resolve(c.background);ok=input(c.background,{0,0,int(lut.width),int(lut.height)},op.auxiliary,op.auxiliary_page);}
            if(ok&&!empty(r)&&c.kind==Kind::native_image&&c.background_detail){
                int dx=c.source_x-c.area.left,dy=c.source_y-c.area.top;
                if(c.background_detail==detail){if(dx||dy)ok=false;else op.flags|=32u;}
                else {ok=input(c.background_detail,footprint(c.background_detail,dx,dy),op.auxiliary,op.auxiliary_page);op.auxiliary[0]+=dx;op.auxiliary[1]+=dy;}
            }
            if(ok&&c.kind==Kind::native_blend&&c.color==2){op.auxiliary[0]=c.source_width;op.auxiliary[1]=c.source_height;}
            if(ok&&!empty(r)&&c.kind==Kind::native_lookup){auto table=resolve(c.source);ok=input(c.source,{0,0,int(table.width),int(table.height)},op.source,op.source_page);
                if(ok&&c.program){int dx=c.source_x-c.area.left,dy=c.source_y-c.area.top;ok=input(c.program,footprint(c.program,dx,dy),op.auxiliary,op.auxiliary_page);op.auxiliary[0]+=dx;op.auxiliary[1]+=dy;op.flags|=16u;}
            }
            if(!ok){sources.resize(previous_sources);cursor_x=previous_x;cursor_y=previous_y;row_height=previous_row;page=previous_page;}
            supported[n]=ok;
        }
        for(unsigned first=0;first<count;){unsigned end=first+1;while(end<count&&supported[end]==supported[first])++end;
            Run run={first,end-first,unsigned(tiles.size()),0,supported[first]};
            if(run.spatial){
                std::map<unsigned,std::vector<unsigned>> selected;std::size_t selected_indices=0;
                unsigned columns=(w.width+tile_size-1)/tile_size;
                for(unsigned n=first;n<end;++n){auto const& r=encoded[n].area;if(r[0]>=r[2]||r[1]>=r[3])continue;
                    for(unsigned y=unsigned(r[1])/tile_size;y<=unsigned(r[3]-1)/tile_size;++y)
                    for(unsigned x=unsigned(r[0])/tile_size;x<=unsigned(r[2]-1)/tile_size;++x){
                        if(indices.size()+selected_indices>=max_indices)return false;++selected_indices;selected[y*columns+x].push_back(n);}
                }
                for(auto const& tile:selected){auto offset=unsigned(indices.size());
                    if(tile.second.size()>max_indices-indices.size())return false;
                    indices.insert(indices.end(),tile.second.begin(),tile.second.end());
                    tiles.push_back({tile.first%columns,tile.first/columns,offset,unsigned(tile.second.size())});}
                run.tiles=unsigned(tiles.size())-run.tile;next.spatial_commands+=end-first;++next.spatial_runs;
            }else next.fallback_commands+=end-first;
            next.runs.push_back(run);first=end;
        }
        // A no-op wrapper around the interpreter is not a compiled plan.
        if(!next.spatial_commands)return false;
        unsigned pages=sources.empty()?1:page+1;
        auto command_bytes=unsigned(encoded.size()*sizeof(GpuCommand));
        auto tile_bytes=unsigned(std::max<std::size_t>(tiles.size(),1)*sizeof(Tile));
        auto index_bytes=unsigned(std::max<std::size_t>(indices.size(),1)*sizeof(unsigned));
        std::vector<Tile> table;
        if(next.runs.size()==1&&next.runs[0].spatial){
            next.columns=(w.width+tile_size-1)/tile_size;unsigned rows=(w.height+tile_size-1)/tile_size;
            table.resize(std::size_t(next.columns)*rows);
            for(unsigned y=0;y<rows;++y)for(unsigned x=0;x<next.columns;++x)table[std::size_t(y)*next.columns+x]={x,y,0,0};
            for(auto const& tile:tiles)table[std::size_t(tile.y)*next.columns+tile.x]=tile;
        }
        auto table_bytes=unsigned(table.size()*sizeof(Tile));
        next.bytes=std::uint64_t(pages)*page_size*page_size*4+command_bytes+tile_bytes+index_bytes+table_bytes;
        bool same_sources=reuse_sources&&old.atlas&&old.source_keys.size()==sources.size();
        for(unsigned i=0;same_sources&&i<sources.size();++i){auto const& a=old.source_keys[i];auto const& b=sources[i];
            same_sources=a.texture==b.texture.Get()&&same(a.area,b.area)&&a.x==b.x&&a.y==b.y&&a.page==b.page;
        }
        auto atlas_bytes=std::uint64_t(pages)*page_size*page_size*4;
        if(!admit(next.bytes-(same_sources?atlas_bytes:0)))return false;program();
        for(auto const& source:sources)next.source_keys.push_back({source.texture.Get(),source.area,source.x,source.y,source.page});
        if(same_sources){next.atlas=old.atlas;next.atlas_view=old.atlas_view;next.owned[0]=old.owned[0];next.physical[0]=old.physical[0];}
        else {
        D3D11_TEXTURE2D_DESC td={};td.Width=td.Height=page_size;td.ArraySize=pages;td.MipLevels=td.SampleDesc.Count=1;
        td.Format=DXGI_FORMAT_R32_UINT;td.Usage=D3D11_USAGE_DEFAULT;td.BindFlags=D3D11_BIND_SHADER_RESOURCE;
        check(device->CreateTexture2D(&td,nullptr,&next.atlas));check(device->CreateShaderResourceView(next.atlas.Get(),nullptr,&next.atlas_view));
        next.owned[0]=allocations.retain(next.atlas.Get());next.physical[0]=physical.retain(next.atlas.Get());
        for(auto const& source:sources){auto a=source.area;D3D11_BOX box={unsigned(a.left),unsigned(a.top),0,unsigned(a.right),unsigned(a.bottom),1};
            context->CopySubresourceRegion(next.atlas.Get(),source.page,source.x,source.y,0,source.texture.Get(),0,&box);++source_copies_;}
        }
        auto buffer=[&](auto const& values,unsigned stride,unsigned bytes,auto& resource,auto& view,unsigned slot){
            unsigned zero[4]={};D3D11_BUFFER_DESC bd={};bd.ByteWidth=bytes;bd.Usage=D3D11_USAGE_IMMUTABLE;bd.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            bd.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;bd.StructureByteStride=stride;
            D3D11_SUBRESOURCE_DATA data={values.empty()?static_cast<void const*>(zero):static_cast<void const*>(values.data())};
            check(device->CreateBuffer(&bd,&data,&resource));check(device->CreateShaderResourceView(resource.Get(),nullptr,&view));
            next.owned[slot]=allocations.retain(resource.Get());next.physical[slot]=physical.retain(resource.Get());
        };
        buffer(encoded,sizeof(GpuCommand),command_bytes,next.commands,next.command_view,1);
        buffer(tiles,sizeof(Tile),tile_bytes,next.tiles,next.tile_view,2);
        buffer(indices,sizeof(unsigned),index_bytes,next.order,next.order_view,3);
        if(!table.empty())buffer(table,sizeof(Tile),table_bytes,next.table,next.table_view,5);
        // Commands, placement and external operands belong to this immutable
        // plan. Recompiling any of them discards the optional resolved HUD.
        // Ordered interpreter boundaries cannot share a cache across runs.
        char reference[8]={};
        bool reference_hud=GetEnvironmentVariableA("C3X_RENDERER_HUD_CACHE_REFERENCE",reference,sizeof(reference))&&reference[0]=='1';
        auto hud_bytes=std::uint64_t(w.width)*w.height*12;
        if(!reference_hud && next.runs.size()==1 && next.runs[0].spatial && admit(next.bytes+hud_bytes)){
            D3D11_TEXTURE2D_DESC td={};td.Width=w.width;td.Height=w.height;td.ArraySize=3;
            td.MipLevels=td.SampleDesc.Count=1;td.Format=DXGI_FORMAT_R32_UINT;td.BindFlags=D3D11_BIND_UNORDERED_ACCESS;
            if(SUCCEEDED(device->CreateTexture2D(&td,nullptr,&next.hud)) &&
                    SUCCEEDED(device->CreateUnorderedAccessView(next.hud.Get(),nullptr,&next.hud_write))){
                next.owned[4]=allocations.retain(next.hud.Get());next.physical[4]=physical.retain(next.hud.Get());
                next.bytes+=hud_bytes;unsigned clear[4]={};context->ClearUnorderedAccessViewUint(next.hud_write.Get(),clear);
            }else{next.hud_write.Reset();next.hud.Reset();}
        }
        old=std::move(next);++compilations_;return true;
    }
    // The fused interface pass over a single-run plan (stage 3). mode is the
    // native word format (0 = 555, 1 = 565); key and flags describe the keyed
    // screen-canvas transfer exactly as the spatial program encodes kind 9.
    bool fused(Plan const& plan,ID3D11ShaderResourceView* scene,ID3D11ShaderResourceView* screen_words,ID3D11ShaderResourceView* screen_detail,
               ID3D11UnorderedAccessView* front,ID3D11UnorderedAccessView* front_words,unsigned width,unsigned height,unsigned mode,unsigned key,unsigned flags){
        if(!plan||!plan.table_view||plan.width!=width||plan.height!=height||!scene||!screen_words||!front||!front_words||((flags&2)&&!(flags&32)&&!screen_detail))return false;
        program();
        unsigned params[4]={0,width,height,0},fused_params[4]={plan.columns,mode,key,flags};
        context->UpdateSubresource(constants.Get(),0,nullptr,params,0,0);context->UpdateSubresource(fused_constants.Get(),0,nullptr,fused_params,0,0);
        ID3D11Buffer* buffers[]={constants.Get(),fused_constants.Get()};context->CSSetConstantBuffers(0,2,buffers);
        ID3D11ShaderResourceView* reads[]={plan.atlas_view.Get(),plan.command_view.Get(),plan.tile_view.Get(),plan.order_view.Get(),
            scene,screen_words,screen_detail,plan.table_view.Get()};
        context->CSSetShaderResources(0,8,reads);
        ID3D11UnorderedAccessView* writes[]={nullptr,nullptr,nullptr,front,front_words};context->CSSetUnorderedAccessViews(0,5,writes,nullptr);
        context->CSSetShader(fused_shader.Get(),nullptr,0);context->Dispatch((width+7)/8,(height+7)/8,1);
        ID3D11ShaderResourceView* none[8]={};ID3D11UnorderedAccessView* no_writes[5]={};
        context->CSSetShaderResources(0,8,none);context->CSSetUnorderedAccessViews(0,5,no_writes,nullptr);
        ++dispatches_;commands_+=unsigned(plan.original.size());return true;
    }
    template<class Legacy> bool submit(Plan const& plan,Id words_id,Id detail_id,ID3D11UnorderedAccessView* words,ID3D11UnorderedAccessView* detail,Legacy legacy){
        if(!plan||!words||!detail)return false;
        for(auto const& run:plan.runs){
            if(!run.spatial){for(unsigned n=run.command;n<run.command+run.count;++n){auto c=plan.original[n];
                Id* ids[]={&c.destination,&c.source,&c.background,&c.detail,&c.background_detail,&c.program};
                for(auto id:ids){if(*id==plan.words)*id=words_id;else if(*id==plan.detail)*id=detail_id;}
                if(!legacy(c))return false;}continue;}
            if(!run.tiles)continue;
            unsigned params[4]={run.tile,plan.width,plan.height,plan.hud?1u:0u};context->UpdateSubresource(constants.Get(),0,nullptr,params,0,0);
            auto cb=constants.Get();ID3D11ShaderResourceView* reads[]={plan.atlas_view.Get(),plan.command_view.Get(),plan.tile_view.Get(),plan.order_view.Get()};
            ID3D11UnorderedAccessView* writes[]={words,detail,plan.hud_write.Get()};context->CSSetConstantBuffers(0,1,&cb);context->CSSetShaderResources(0,4,reads);
            context->CSSetUnorderedAccessViews(0,3,writes,nullptr);context->CSSetShader(shader.Get(),nullptr,0);context->Dispatch(run.tiles,1,1);
            ID3D11ShaderResourceView* none[4]={};ID3D11UnorderedAccessView* no_writes[3]={};context->CSSetShaderResources(0,4,none);context->CSSetUnorderedAccessViews(0,3,no_writes,nullptr);
            ++dispatches_;commands_+=run.count;
        }return true;
    }
};
}
