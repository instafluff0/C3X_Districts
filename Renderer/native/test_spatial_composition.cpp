#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <memory>
#include "gpu_image_compositor.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
using namespace c3x_gpu_images;
namespace {
void require(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
std::vector<unsigned> read(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
 D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
 D3D11_MAPPED_SUBRESOURCE mapped={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&mapped));
 std::vector<unsigned> values(std::size_t(d.Width)*d.Height);
 for(unsigned y=0;y<d.Height;++y)std::memcpy(values.data()+std::size_t(y)*d.Width,static_cast<char*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,d.Width*4);
 context->Unmap(stage.Get(),0);return values;
}
void exact(ID3D11Device* d,ID3D11DeviceContext* c,Compositor& a,Compositor& b,Id image,char const* label){
 auto expected=read(d,c,a.texture(image)),actual=read(d,c,b.texture(image));
 for(std::size_t i=0;i<expected.size();++i)if(expected[i]!=actual[i]){
  std::fprintf(stderr,"%s mismatch pixel=%zu expected=%08x actual=%08x\n",label,i,expected[i],actual[i]);throw std::runtime_error("same-clock spatial/interpreter pixel equality");}
}
}
int test_spatial_composition(){
 try{
  ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
  checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context));
  for(auto format:{Format::rgb555,Format::rgb565}){
   auto a_owner=std::make_unique<Compositor>(device.Get(),context.Get(),128u*1024u*1024u);
   auto b_owner=std::make_unique<Compositor>(device.Get(),context.Get(),128u*1024u*1024u);auto& a=*a_owner;auto& b=*b_owner;
   constexpr unsigned width=256,height=192;Rect full={0,0,width,height};
   Id words=0,detail=0,sprite=0,mask=0,curves=0,image=0,image_detail=0,blend=0,weighted=0,lookup=0,coverage=0;
   auto create=[&](unsigned w,unsigned h,Format f){auto x=a.create(w,h,f),y=b.create(w,h,f);require(x&&x==y,"paired fixture handles");return x;};
   auto upload=[&](Id id,unsigned revision,std::vector<unsigned> const& pixels){require(a.upload(id,revision,pixels.data(),pixels.size())&&b.upload(id,revision,pixels.data(),pixels.size()),"fixture upload");};
   words=create(width,height,format);detail=create(width,height,Format::bgra32);
   sprite=create(48,32,Format::bgra32);mask=create(48,32,Format::bgra32);curves=create(17,3,Format::bgra32);
   image=create(48,32,format);image_detail=create(48,32,Format::bgra32);blend=create(48,32,Format::bgra32);weighted=create(48,32,Format::bgra32);
   lookup=create(1024,1024,Format::bgra32);coverage=create(48,32,Format::bgra32);
   std::vector<unsigned> values(48*32),full_values(values.size());
   for(unsigned i=0;i<values.size();++i)values[i]=(i%7?65536:0)|(i*37&65535);upload(sprite,1,values);
   for(unsigned i=0;i<values.size();++i)values[i]=(i%3)|((i/3%3)<<10)|((i/7%3)<<20)|(i%5?0x40000000u:0);upload(mask,1,values);
   values.resize(17*3);for(unsigned i=0;i<values.size();++i)values[i]=i/17==0?i%17*15:i/17==1?255-i%17*15:127;upload(curves,1,values);
   values.resize(48*32);for(unsigned i=0;i<values.size();++i){values[i]=i%11?i*79&65535:0x7c1f;full_values[i]=0xff000000u|(i*4371&0xffffff);}
   upload(image,1,values);upload(image_detail,1,full_values);
   for(unsigned i=0;i<values.size();++i)values[i]=(i%256<<24)|(i*773&0xffffff);upload(blend,1,values);
   for(unsigned i=0;i<values.size();++i)values[i]=(i%17<<24)|(i*37&65535);upload(weighted,1,values);
   values.resize(1024*1024);for(unsigned i=0;i<values.size();++i)values[i]=(i*3+(i>>15)*11)&65535;upload(lookup,1,values);
   values.resize(48*32);for(unsigned i=0;i<values.size();++i)values[i]=i%19;upload(coverage,1,values);
   std::vector<Command> commands;
   for(unsigned n=0;n<384;++n){int x=int(n*31%280)-12,y=int(n*17%210)-10;Rect r={x,y,x+48,y+32};Rect clip={5,3,251,188};
    switch(n%13){
    case 0:commands.push_back({Kind::native_text,words,mask,r,clip,0,0,0,curves});break;
    case 1:commands.push_back({Kind::native_text,detail,mask,r,clip,0,0,0,curves});break;
    case 2:commands.push_back({Kind::native_sprite,words,sprite,r,clip});break;
    case 3:commands.push_back({Kind::native_sprite,detail,sprite,r,clip,0,0,format==Format::rgb565?2u:1u});break;
    case 4:commands.push_back({Kind::native_image,words,image,r,clip,0,0,0x7c1f,0,detail,image_detail,48,32});break;
    case 5:commands.push_back({Kind::native_blend,words,blend,r,clip,0,0,0,words,detail,detail});break;
    case 6:commands.push_back({Kind::native_blend,words,blend,r,clip,0,0,1,words,detail,detail});break;
    case 7:commands.push_back({Kind::native_blend,words,words,r,clip,0,0,2,words,detail,detail,0x34d1,96});break;
    case 8:commands.push_back({Kind::native_blend,words,weighted,r,clip,0,0,3,words,detail,detail});break;
    case 9:commands.push_back({Kind::color_key,words,image,r,clip,0,0,0x7c1f});break;
    case 10:commands.push_back({Kind::invert,detail,0,r,clip,0,0,0x0040a090});break;
    case 11:commands.push_back({Kind::native_lookup,words,lookup,r,clip,0,0,3,words,detail,detail,0,0,coverage});break;
    case 12:commands.push_back({Kind::unit_over,words,blend,r,clip,0,0,0,words,detail,detail});break;
    }
   }
   // The cross-position alias remains a materialization/interpreter boundary.
   commands.insert(commands.begin()+173,{Kind::copy,words,words,{21,22,69,54},full,3,7});
   Compositor::SpatialPlan plan;
   require(b.compile_spatial(plan,commands.data(),commands.size(),words,detail,64u*1024u*1024u),"bounded spatial compile");
   require(plan.fallback_commands==1&&plan.spatial_runs==2&&plan.spatial_commands==384,"pointwise runs split at self-copy");
   auto source_copies=b.stats().spatial_source_copies;
   for(unsigned frame=1;frame<=12;++frame){values.resize(width*height);full_values.resize(values.size());
    for(unsigned i=0;i<values.size();++i){values[i]=(i*17+frame*811)&65535;full_values[i]=0xff000000u|((i*107+frame*973)&0xffffff);}
    upload(words,frame,values);upload(detail,frame,full_values);
    require(a.submit(commands.data(),commands.size()),"legacy ordered command sequence");require(b.submit_spatial(plan,words,detail),"spatial ordered command sequence");
    try{exact(device.Get(),context.Get(),a,b,words,"packed");exact(device.Get(),context.Get(),a,b,detail,"detail");}
    catch(...){
     upload(words,1000,values);upload(detail,1000,full_values);
     for(unsigned n=0;n<commands.size();++n){auto const& command=commands[n];Compositor::SpatialPlan single;
      require(a.submit(&command,1),"single oracle command");
      if(b.compile_spatial(single,&command,1,words,detail,64u*1024u*1024u))require(b.submit_spatial(single,words,detail),"single spatial command");
      else require(b.submit(&command,1),"single interpreter boundary");
      std::printf("checking command=%u kind=%u color=%u format=%u\n",n,unsigned(command.kind),command.color,unsigned(format));std::fflush(stdout);
      exact(device.Get(),context.Get(),a,b,words,"single packed");exact(device.Get(),context.Get(),a,b,detail,"single detail");
     }throw;
    }
   }
   require(b.stats().spatial_compilations==1&&b.stats().spatial_source_copies==source_copies,"unchanged immutable sources compile/copy once");
   require(b.stats().spatial_dispatches==24,"384 pointwise commands use two dispatches per frame");
   auto old_words=read(device.Get(),context.Get(),b.texture(words));
   auto new_words=create(width,height,format),new_detail=create(width,height,Format::bgra32);
   std::vector<Command> rebound=commands;
   for(auto& command:rebound){Id* ids[]={&command.destination,&command.source,&command.background,&command.detail,&command.background_detail,&command.program};
    for(auto id:ids){if(*id==words)*id=new_words;else if(*id==detail)*id=new_detail;}}
   require(a.submit(rebound.data(),rebound.size()),"recreated-target mixed interpreter oracle");
   require(b.submit_spatial(plan,new_words,new_detail),"new target pair rebound without source recompilation");
   exact(device.Get(),context.Get(),a,b,new_words,"rebound packed");exact(device.Get(),context.Get(),a,b,new_detail,"rebound detail");
   require(read(device.Get(),context.Get(),b.texture(words))==old_words,"old target reader immutable after pair rebinding");
   // A complete pointwise HUD plan caches map-independent pixels. Change
   // every underlay pixel between frames: keyed holes, blends and text must
   // still exactly match the native interpreter, including 555/565 rounding.
   auto hud_commands=commands;hud_commands.erase(hud_commands.begin()+173);
   Compositor::SpatialPlan hud_plan;
   require(b.compile_spatial(hud_plan,hud_commands.data(),hud_commands.size(),words,detail,64u*1024u*1024u),"HUD cache compile");
   require(bool(hud_plan.hud)&&hud_plan.fallback_commands==0,"HUD cache admitted for pointwise plan");
   for(unsigned frame=1;frame<=6;++frame){
    values.resize(width*height);full_values.resize(values.size());
    for(unsigned i=0;i<values.size();++i){values[i]=(i*739+frame*1709)&65535;full_values[i]=0xff000000u|((i*331+frame*1783)&0xffffff);}
    upload(words,3000+frame,values);upload(detail,3000+frame,full_values);
    require(a.submit(hud_commands.data(),hud_commands.size())&&b.submit_spatial(hud_plan,words,detail),"HUD over changing map");
    exact(device.Get(),context.Get(),a,b,words,"cached HUD packed");exact(device.Get(),context.Get(),a,b,detail,"cached HUD detail");
   }
   D3D11_TEXTURE2D_DESC mask_desc={};hud_plan.hud->GetDesc(&mask_desc);
   mask_desc.ArraySize=1;mask_desc.Usage=D3D11_USAGE_STAGING;mask_desc.BindFlags=0;mask_desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
   ComPtr<ID3D11Texture2D> cached_mask;checked(device->CreateTexture2D(&mask_desc,nullptr,&cached_mask));
   context->CopySubresourceRegion(cached_mask.Get(),0,0,0,0,hud_plan.hud.Get(),2,nullptr);
   D3D11_MAPPED_SUBRESOURCE cached_pixels={};checked(context->Map(cached_mask.Get(),0,D3D11_MAP_READ,0,&cached_pixels));
   unsigned resolved=0,dependent=0;for(unsigned y=0;y<height;++y){auto row=reinterpret_cast<unsigned const*>(static_cast<char const*>(cached_pixels.pData)+y*cached_pixels.RowPitch);
    for(unsigned x=0;x<width;++x){resolved+=row[x]==1;dependent+=row[x]==2;}}
   context->Unmap(cached_mask.Get(),0);require(resolved>0&&dependent>0&&resolved+dependent==width*height,"HUD and map dependence classified once");
   std::printf("PASS HUD cache: format=%u cached_pixels=%u changing_underlays=6 exact=1\n",unsigned(format),resolved);
   // The fused interface pass (stage 3) classifies and reuses the same cache:
   // without it every touched pixel re-ran its whole program each frame.
   {Compositor::SpatialPlan fused_plan;
    require(b.compile_spatial(fused_plan,hud_commands.data(),hud_commands.size(),words,detail,64u*1024u*1024u)&&fused_plan.table_view&&fused_plan.hud,"fused plan with HUD cache");
    auto scene=create(width,height,Format::bgra32),screen=create(width,height,format),front=create(width,height,Format::bgra32),front_words=create(width,height,format);
    unsigned key=format==Format::rgb565?0xf81fu:0x7c1fu;
    upload(screen,1,std::vector<unsigned>(width*height,key));
    for(unsigned frame=1;frame<=3;++frame){
     for(unsigned i=0;i<full_values.size();++i)full_values[i]=0xff000000u|((i*211+frame*997)&0xffffff);
     upload(scene,4000+frame,full_values);
     require(b.fused_spatial(fused_plan,scene,screen,0,front,front_words,key,1u),"fused HUD over a changing scene");
    }
    context->CopySubresourceRegion(cached_mask.Get(),0,0,0,0,fused_plan.hud.Get(),2,nullptr);
    checked(context->Map(cached_mask.Get(),0,D3D11_MAP_READ,0,&cached_pixels));unsigned fused_resolved=0;
    for(unsigned y=0;y<height;++y){auto row=reinterpret_cast<unsigned const*>(static_cast<char const*>(cached_pixels.pData)+y*cached_pixels.RowPitch);
     for(unsigned x=0;x<width;++x)fused_resolved+=row[x]==1;}
    context->Unmap(cached_mask.Get(),0);require(fused_resolved>0,"fused pass classifies map-independent HUD pixels");
    std::printf("PASS fused HUD cache: format=%u cached_pixels=%u\n",unsigned(format),fused_resolved);}
   // A fallback writer makes its later source mutable; it cannot be
   // captured into the immutable atlas before the write actually executes.
   auto scratch=create(width,height,format);
   Command mixed[]={{Kind::copy,scratch,new_words,full,full},
       {Kind::color_key,new_words,scratch,full,full,0,0,0x7c1f},
       {Kind::invert,new_detail,0,{3,4,81,59},full,0,0,0x102030}};
   Compositor::SpatialPlan mixed_plan;
   require(b.compile_spatial(mixed_plan,mixed,3,new_words,new_detail,64u*1024u*1024u),"mixed source-write boundary compile");
   require(mixed_plan.fallback_commands==2,"mutable source reader kept in ordered interpreter");
   require(a.submit(mixed,3)&&b.submit_spatial(mixed_plan,new_words,new_detail),"mixed source-write boundary execution");
   exact(device.Get(),context.Get(),a,b,new_words,"source-write packed");exact(device.Get(),context.Get(),a,b,new_detail,"source-write detail");
   require(!b.submit_spatial(mixed_plan,scratch,new_detail),"external fallback/source allocation alias refused");
   auto high=create(48,32,format);std::vector<unsigned> high_values(48*32,0x10000);
   // Imported typed R32 sources can contain high bits even when their native
   // format is packed; native image transfer still starts from 65535.
   D3D11_BOX high_box={0,0,0,48,32,1};context->UpdateSubresource(a.texture(high),0,&high_box,high_values.data(),48*4,0);
   context->UpdateSubresource(b.texture(high),0,&high_box,high_values.data(),48*4,0);
   Command high_command={Kind::native_image,new_words,high,{8,9,56,41},full,0,0,65536,0,new_detail,0,48,32};
   Compositor::SpatialPlan high_plan;require(b.compile_spatial(high_plan,&high_command,1,new_words,new_detail,64u*1024u*1024u),"high-bit input compile");
   require(a.submit(&high_command,1)&&b.submit_spatial(high_plan,new_words,new_detail),"high-bit native image transfer");
   exact(device.Get(),context.Get(),a,b,new_words,"high-bit packed");exact(device.Get(),context.Get(),a,b,new_detail,"high-bit detail");
   Command units[]={{Kind::fill,new_words,0,{4,5,52,37},full,0,0,0x7c1f},
       {Kind::fill,new_detail,0,{4,5,52,37},full,0,0,0xffec19f1},
       {Kind::unit_over,new_words,blend,{4,5,52,37},full,0,0,0,new_words,new_detail,new_detail}};
   Compositor::SpatialPlan unit_plan;
   require(b.compile_spatial(unit_plan,units,3,new_words,new_detail,64u*1024u*1024u),"paired native unit-over compile");
   require(unit_plan.fallback_commands==0,"pointwise paired ground unit-over fused");
   require(a.submit(units,3)&&b.submit_spatial(unit_plan,new_words,new_detail),"paired keyed ground unit-over execution");
   exact(device.Get(),context.Get(),a,b,new_words,"unit-over keyed packed");exact(device.Get(),context.Get(),a,b,new_detail,"unit-over keyed detail");
   auto wrong_format=create(width,height,format==Format::rgb555?Format::rgb565:Format::rgb555);
   require(!b.submit_spatial(plan,wrong_format,new_detail)&&!b.submit_spatial(plan,new_words,new_words),"pair format/alias rebinding refused");
   Compositor::SpatialPlan refused;require(!b.compile_spatial(refused,commands.data(),commands.size(),words,detail,1),"budget refusal retains existing plan");
   require(plan&&plan.spatial_commands==384,"previous bounded plan remains usable");
   // Ragged dimensions exercise all sixteen 8x8 blocks and partial edge
   // groups. Each full-frame operation admits the edge tile itself.
   for(auto extent:std::array<std::array<unsigned,2>,4>{{{1,1},{33,17},{65,33},{257,193}}}){
    unsigned w=extent[0],h=extent[1];Rect area={0,0,int(w),int(h)};
    auto edge_a=std::make_unique<Compositor>(device.Get(),context.Get(),8u*1024u*1024u);
    auto edge_b=std::make_unique<Compositor>(device.Get(),context.Get(),8u*1024u*1024u);
    auto edge_words=edge_a->create(w,h,format),edge_detail=edge_a->create(w,h,Format::bgra32);
    require(edge_words&&edge_detail&&edge_b->create(w,h,format)==edge_words&&
        edge_b->create(w,h,Format::bgra32)==edge_detail,"ragged paired fixture handles");
    Command edge[]={{Kind::fill,edge_words,0,area,area,0,0,0x3517},
        {Kind::fill,edge_detail,0,area,area,0,0,0xff6192ac},
        {Kind::native_blend,edge_words,edge_words,area,area,0,0,2,edge_words,edge_detail,edge_detail,0x731b,91},
        {Kind::invert,edge_detail,0,{int(w/2),int(h/2),int(w),int(h)},area,0,0,0x183c72}};
    Compositor::SpatialPlan edge_plan;
    require(edge_b->compile_spatial(edge_plan,edge,4,edge_words,edge_detail,64u*1024u*1024u),"ragged spatial compile");
    require(edge_a->submit(edge,4)&&edge_b->submit_spatial(edge_plan,edge_words,edge_detail),"ragged spatial execution");
    exact(device.Get(),context.Get(),*edge_a,*edge_b,edge_words,"ragged packed");
    exact(device.Get(),context.Get(),*edge_a,*edge_b,edge_detail,"ragged detail");
    require(edge_a->destroy(edge_words)&&edge_a->destroy(edge_detail)&&edge_b->destroy(edge_words)&&edge_b->destroy(edge_detail),"ragged resources retire");
   }
   // Match the live HUD's independent glyph/image operands without adding
   // public image handles. Interpreter boundaries bind one command at a time.
   std::vector<Compositor::SpatialSource> inputs;std::vector<Command> many;
   for(unsigned n=0;n<900;++n){auto id=b.create(2,2,format);require(bool(id),"bounded transient source handle");
    std::vector<unsigned> pixels(4);for(unsigned i=0;i<4;++i)pixels[i]=(n*131+i*257)&65535;
    require(b.upload(id,1,pixels.data(),pixels.size()),"independent source upload");
    inputs.push_back({Compositor::spatial_source_id(n),b.texture(id),2,2,format});require(b.destroy(id),"source handle retired after immutable ownership transfer");
    int x=int(n*7%(width-4)),y=int(n*13%(height-4));
    many.push_back({Kind::native_image,new_words,inputs.back().id,{x,y,x+2,y+2},full,0,0,65536,0,new_detail,0,2,2});
   }
   many.insert(many.begin()+450,{Kind::copy,new_words,new_words,{21,22,23,24},full,3,7});
   many.push_back({Kind::native_image,new_words,inputs.front().id,{5,6,9,10},full,0,0,65536,0,new_detail,0,2,2});
   Compositor::SpatialPlan many_plan;
   require(b.compile_spatial_sources(many_plan,many.data(),many.size(),new_words,new_detail,64u*1024u*1024u,inputs),"900 immutable sources admitted without handle growth");
   require(many_plan.spatial_commands==900&&many_plan.fallback_commands==2,"external scaled source and alias interpreter boundaries");
   for(unsigned frame=1;frame<=4;++frame){values.assign(width*height,frame*731);full_values.assign(width*height,0xff175e93);
    upload(new_words,2000+frame,values);upload(new_detail,2000+frame,full_values);
    require(a.submit_source_commands(many.data(),many.size(),inputs)&&b.submit_spatial_sources(many_plan,new_words,new_detail,inputs),"handleless sources and lazy fallback execute");
    exact(device.Get(),context.Get(),a,b,new_words,"900-source packed");exact(device.Get(),context.Get(),a,b,new_detail,"900-source detail");
   }
   auto malformed=inputs;malformed.front().width=3;
   require(!b.compile_spatial_sources(many_plan,many.data(),many.size(),new_words,new_detail,64u*1024u*1024u,malformed),"external resource extent verified before replacement");
   require(many_plan.spatial_commands==900,"failed external compile preserves previous plan");
   std::printf("PASS handleless spatial inputs: format=%u sources=900 frames=4 scaled_external_boundary=1 native_alias_boundary=1 exact_words_and_detail=1 bounded_handle_table=512\n",unsigned(format));
   std::printf("PASS spatial composition: format=%u twelve dynamic underlays, dense text/key/blends, self-copy boundary, old reader, %u -> %llu recurring dispatches, %llu charged bytes\n",unsigned(format),384u,b.stats().spatial_dispatches/13,plan.bytes);
  }
  return 0;
 }catch(std::exception const& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
#ifdef C3X_SPATIAL_TEST_MAIN
int main(){return test_spatial_composition();}
#endif
