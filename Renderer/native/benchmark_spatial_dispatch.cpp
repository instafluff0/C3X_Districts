// Quiet, isolated production spatial-submit probe. No game, window or presenter.
// Build this same file beside each frozen gpu_image_compositor.h/header closure:
// cl /nologo /O2 /EHsc /std:c++17 benchmark_spatial_dispatch.cpp
// Run: benchmark_spatial_dispatch RESULTS.json ARM_LABEL [SAMPLES]
// The driver runs separate frozen executables in ABBA order. No production switch.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "gpu_image_compositor.h"
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")

namespace {
using namespace c3x_gpu_images;
constexpr unsigned benchmark_width=2240,benchmark_height=1260,commands_per_frame=2124,warmup=4;
void require(bool ok,char const* text){if(!ok)throw std::runtime_error(text);}
double now(){LARGE_INTEGER ticks,frequency;QueryPerformanceCounter(&ticks);QueryPerformanceFrequency(&frequency);
 return double(ticks.QuadPart)*1000.0/double(frequency.QuadPart);}
std::uint64_t hash(std::vector<unsigned> const& words){std::uint64_t value=14695981039346656037ull;
 for(auto word:words)for(unsigned b=0;b<4;++b){value^=(word>>(8*b))&255;value*=1099511628211ull;}return value;}
std::vector<unsigned> read(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
 D3D11_TEXTURE2D_DESC d{};texture->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=d.MiscFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
 ComPtr<ID3D11Texture2D> staging;checked(device->CreateTexture2D(&d,nullptr,&staging));context->CopyResource(staging.Get(),texture);
 D3D11_MAPPED_SUBRESOURCE mapped{};checked(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped));
 std::vector<unsigned> result(std::size_t(d.Width)*d.Height);
 for(unsigned y=0;y<d.Height;++y)std::memcpy(result.data()+std::size_t(y)*d.Width,
  static_cast<unsigned char const*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,d.Width*4);
 context->Unmap(staging.Get(),0);return result;
}
struct Fixture {
 ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;ComPtr<ID3D11Query> complete;
 std::unique_ptr<Compositor> spatial,interpreter;
 Id words=0,detail=0,sprite=0,mask=0,curves=0,image=0,image_detail=0,blend=0,weighted=0;
 Compositor::SpatialPlan plan;
 std::vector<Command> commands;
 std::array<std::vector<unsigned>,2> underlay,full_color;
 std::array<std::uint64_t,2> word_hash{},detail_hash{},input_word_hash{},input_detail_hash{};
 unsigned revision=0;
 Fixture(){
  checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context));
  D3D11_QUERY_DESC q{};q.Query=D3D11_QUERY_EVENT;checked(device->CreateQuery(&q,&complete));
  spatial=std::make_unique<Compositor>(device.Get(),context.Get(),128u*1024u*1024u);
  interpreter=std::make_unique<Compositor>(device.Get(),context.Get(),128u*1024u*1024u);
  auto create=[&](unsigned w,unsigned h,Format f){auto a=spatial->create(w,h,f),b=interpreter->create(w,h,f);
   require(a&&a==b,"paired bounded handles");return a;};
  words=create(benchmark_width,benchmark_height,Format::rgb555);detail=create(benchmark_width,benchmark_height,Format::bgra32);
  sprite=create(48,32,Format::bgra32);mask=create(48,32,Format::bgra32);curves=create(17,3,Format::bgra32);
  image=create(48,32,Format::rgb555);image_detail=create(48,32,Format::bgra32);
  blend=create(48,32,Format::bgra32);weighted=create(48,32,Format::bgra32);
  auto upload=[&](Id id,std::vector<unsigned> const& values){require(spatial->upload(id,1,values.data(),values.size())&&
   interpreter->upload(id,1,values.data(),values.size()),"immutable source seed");};
  std::vector<unsigned> values(48*32);
  for(unsigned i=0;i<values.size();++i)values[i]=(i%7?65536:0)|(i*37&65535);upload(sprite,values);
  for(unsigned i=0;i<values.size();++i)values[i]=(i%3)|((i/3%3)<<10)|((i/7%3)<<20)|(i%5?0x40000000u:0);upload(mask,values);
  values.resize(17*3);for(unsigned i=0;i<values.size();++i)values[i]=i/17==0?i%17*15:i/17==1?255-i%17*15:127;upload(curves,values);
  values.resize(48*32);for(unsigned i=0;i<values.size();++i)values[i]=i%11?i*79&65535:0x7c1f;upload(image,values);
  for(unsigned i=0;i<values.size();++i)values[i]=0xff000000u|(i*4371&0xffffff);upload(image_detail,values);
  for(unsigned i=0;i<values.size();++i)values[i]=(i%256<<24)|(i*773&0xffffff);upload(blend,values);
  for(unsigned i=0;i<values.size();++i)values[i]=(i%17<<24)|(i*37&65535);upload(weighted,values);
  Rect full={0,0,benchmark_width,benchmark_height},clip={5,3,benchmark_width-5,benchmark_height-3};commands.reserve(commands_per_frame+1);
  for(unsigned n=0;n<commands_per_frame;++n){int x=int(n*31%(benchmark_width+24))-12,y=int(n*17%(benchmark_height+20))-10;Rect r={x,y,x+48,y+32};
   switch(n%12){
   case 0:commands.push_back({Kind::native_text,words,mask,r,clip,0,0,0,curves});break;
   case 1:commands.push_back({Kind::native_text,detail,mask,r,clip,0,0,0,curves});break;
   case 2:commands.push_back({Kind::native_sprite,words,sprite,r,clip});break;
   case 3:commands.push_back({Kind::native_sprite,detail,sprite,r,clip,0,0,1});break;
   case 4:commands.push_back({Kind::native_image,words,image,r,clip,0,0,0x7c1f,0,detail,image_detail,48,32});break;
   case 5:commands.push_back({Kind::native_blend,words,blend,r,clip,0,0,0,words,detail,detail});break;
   case 6:commands.push_back({Kind::native_blend,words,blend,r,clip,0,0,1,words,detail,detail});break;
   case 7:commands.push_back({Kind::native_blend,words,words,r,clip,0,0,2,words,detail,detail,0x34d1,96});break;
   case 8:commands.push_back({Kind::native_blend,words,weighted,r,clip,0,0,3,words,detail,detail});break;
   case 9:commands.push_back({Kind::color_key,words,image,r,clip,0,0,0x7c1f});break;
   case 10:commands.push_back({Kind::invert,detail,0,r,clip,0,0,0x0040a090});break;
   case 11:commands.push_back({Kind::unit_over,words,blend,r,clip,0,0,0,words,detail,detail});break;
   }
  }
  commands.insert(commands.begin()+1062,{Kind::copy,words,words,{21,22,69,54},full,3,7});
  require(spatial->compile_spatial(plan,commands.data(),commands.size(),words,detail,64u*1024u*1024u),"production spatial compile");
  require(plan.fallback_commands==1&&plan.spatial_runs==2&&plan.spatial_commands==commands_per_frame,"two ordered runs and exact self-copy boundary");
  for(unsigned phase=0;phase<2;++phase){underlay[phase].resize(benchmark_width*benchmark_height);full_color[phase].resize(benchmark_width*benchmark_height);
   for(unsigned i=0;i<benchmark_width*benchmark_height;++i){underlay[phase][i]=(i*17+(phase+1)*811)&65535;
    full_color[phase][i]=0xff000000u|((i*107+(phase+1)*973)&0xffffff);}
   input_word_hash[phase]=hash(underlay[phase]);input_detail_hash[phase]=hash(full_color[phase]);
  }drain();
 }
 ~Fixture(){context->ClearState();context->Flush();}
 void drain(){context->End(complete.Get());context->Flush();BOOL done=FALSE;auto begin=now();
  for(;;){auto hr=context->GetData(complete.Get(),&done,sizeof(done),D3D11_ASYNC_GETDATA_DONOTFLUSH);
   checked(hr);if(hr==S_OK&&done)break;require(now()-begin<30000,"bounded diagnostic event wait");Sleep(0);}}
 void seed(unsigned phase,bool both=false){++revision;
  require(spatial->upload(words,revision,underlay[phase].data(),underlay[phase].size())&&
   spatial->upload(detail,revision,full_color[phase].data(),full_color[phase].size()),"spatial dynamic underlay");
  if(both)require(interpreter->upload(words,revision,underlay[phase].data(),underlay[phase].size())&&
   interpreter->upload(detail,revision,full_color[phase].data(),full_color[phase].size()),"interpreter dynamic underlay");
  drain(); // Reset upload/previous work is outside both measured intervals.
 }
 void parity(){for(unsigned phase=0;phase<2;++phase){seed(phase,true);
  for(std::size_t first=0;first<commands.size();first+=2048)
   require(interpreter->submit(commands.data()+first,std::min<std::size_t>(2048,commands.size()-first)),"untimed ordered interpreter chunk");
  require(spatial->submit_spatial(plan,words,detail),"same fixed spatial transaction");drain();
  auto a=read(device.Get(),context.Get(),interpreter->texture(words)),b=read(device.Get(),context.Get(),spatial->texture(words));
  require(a==b,"exact packed output against interpreter");word_hash[phase]=hash(b);
  a=read(device.Get(),context.Get(),interpreter->texture(detail));b=read(device.Get(),context.Get(),spatial->texture(detail));
  require(a==b,"exact detail output against interpreter");detail_hash[phase]=hash(b);
 }}
 std::array<double,3> sample(unsigned phase){seed(phase);auto begin=now();
  require(spatial->submit_spatial(plan,words,detail),"production spatial transaction");auto submitted=now();drain();auto completed=now();
  return {submitted-begin,completed-begin,completed-submitted};}
};
void metric(FILE* file,char const* name,std::vector<double> values){std::sort(values.begin(),values.end());double sum=0;for(auto v:values)sum+=v;
 std::fprintf(file,"\"%s\":{\"mean_ms\":%.6f,\"median_ms\":%.6f,\"p95_ms\":%.6f,\"max_ms\":%.6f}",
  name,sum/values.size(),values[values.size()/2],values[(values.size()*95-1)/100],values.back());}
}
int main(int argc,char** argv){
 if(argc<3||argc>4){std::fprintf(stderr,"usage: benchmark_spatial_dispatch RESULTS.json ARM_LABEL [SAMPLES]\n");return 2;}
 SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
 try{std::string label=argv[2];require(!label.empty()&&label.size()<64&&label.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")==std::string::npos,"portable arm label");
  unsigned samples=argc==4?unsigned(std::stoul(argv[3])):12;require(samples>=4&&samples<=48,"bounded samples");
  Fixture fixture;fixture.parity();for(unsigned n=0;n<warmup;++n)fixture.sample(n%2);
  std::array<std::vector<double>,3> timings;std::vector<std::array<double,3>> raw;
  for(unsigned n=0;n<samples;++n){auto value=fixture.sample(n%2);raw.push_back(value);for(unsigned i=0;i<3;++i)timings[i].push_back(value[i]);}
  FILE* output=nullptr;require(fopen_s(&output,argv[1],"wb")==0,"small timing receipt");
  std::fprintf(output,"{\"schema\":1,\"arm\":\"%s\",\"width\":%u,\"height\":%u,\"commands\":%zu,\"pointwise_commands\":%u,\"runs\":2,\"fallback_commands\":1,\"plan_bytes\":%zu,\"warmup\":%u,\"samples\":%u,\"exact_interpreter_parity\":true,\"present\":false,\"timed_underlay_upload\":false,\"completion_barrier\":\"D3D11_QUERY_EVENT_Flush_GetData\",\"isolated_gpu_shader_time\":false,",
   label.c_str(),benchmark_width,benchmark_height,fixture.commands.size(),commands_per_frame,fixture.plan.bytes,warmup,samples);
  for(unsigned i=0;i<3;++i){if(i)std::fputc(',',output);metric(output,i==0?"cpu_submit":i==1?"completed_transaction":"completion_wait",timings[i]);}
  std::fprintf(output,",\"input_word_fnv1a64\":[\"%016llx\",\"%016llx\"],\"input_detail_fnv1a64\":[\"%016llx\",\"%016llx\"],\"output_word_fnv1a64\":[\"%016llx\",\"%016llx\"],\"output_detail_fnv1a64\":[\"%016llx\",\"%016llx\"],\"samples_ms\":[",
   fixture.input_word_hash[0],fixture.input_word_hash[1],fixture.input_detail_hash[0],fixture.input_detail_hash[1],fixture.word_hash[0],fixture.word_hash[1],fixture.detail_hash[0],fixture.detail_hash[1]);
  for(unsigned n=0;n<raw.size();++n)std::fprintf(output,"%s[%.6f,%.6f,%.6f]",n?",":"",raw[n][0],raw[n][1],raw[n][2]);
  std::fprintf(output,"]}\n");require(fclose(output)==0,"receipt closed");return 0;
 }catch(std::exception const& error){std::fprintf(stderr,"SPATIAL_DISPATCH_BENCHMARK_FAILED %s\n",error.what());return 1;}
}
