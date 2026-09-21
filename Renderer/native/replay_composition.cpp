#define NOMINMAX
#include <windows.h>
#include "gpu_native_presenter.h"
#include "native_lifetime_registry.h"
#include <memory>
#include <map>
#include <string>
#include <cassert>
#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"user32.lib")
#pragma comment(lib,"gdi32.lib")
using namespace c3x_gpu_images;
namespace recording=c3x_recording;
void require(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
std::vector<unsigned> read_pixels(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
    require(texture!=nullptr,"missing oracle texture");D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);
    d.BindFlags=0;d.Usage=D3D11_USAGE_STAGING;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;d.MiscFlags=0;
    ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
    std::vector<unsigned> out(std::size_t(d.Width)*d.Height);D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));
    for(unsigned y=0;y<d.Height;++y)std::memcpy(out.data()+std::size_t(y)*d.Width,static_cast<char*>(m.pData)+std::size_t(y)*m.RowPitch,d.Width*4);
    context->Unmap(stage.Get(),0);return out;
}
struct Device {
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
    Device(){D3D_FEATURE_LEVEL level;checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));}
};
struct Stream {
    std::unique_ptr<Compositor> gpu;
    std::map<Id,Id> ids;
    Id id(Id recorded)const{if(!recorded)return 0;auto it=ids.find(recorded);return it==ids.end()?UINT64_MAX:it->second;}
};
struct Reader {
    FILE* file=nullptr;std::uint64_t frequency=0,last=0,last_ticks=0;bool footer=false;unsigned version=0;
    Reader(wchar_t const* path){require(_wfopen_s(&file,path,L"rb")==0&&file,"cannot open recording");
        recording::Bytes b(16);require(fread(b.data(),1,b.size(),file)==b.size(),"missing recording header");recording::Cursor c{b};
        require(c.u32()==0x52433343,"unsupported recording magic");version=c.u32();require(version==2||version==3,"unsupported recording version");frequency=c.u64();require(frequency>0&&frequency<1000000000000ull,"invalid recording clock");}
    ~Reader(){if(file)fclose(file);}
    bool next(unsigned& kind,std::uint64_t& stream,std::uint64_t& ticks,recording::Bytes& payload){
        recording::Bytes b(40);auto size=fread(b.data(),1,b.size(),file);if(!size){require(!ferror(file),"recording read error");return false;}
        require(!footer,"data after recording footer");require(size==b.size(),"truncated recording header");recording::Cursor c{b};
        require(c.u32()==0x31523343,"invalid recording marker");kind=c.u32();auto bytes=c.u32(),hash=c.u32();
        require(kind>=recording::begin&&kind<=recording::stop&&bytes<=12u*1024u*1024u,"invalid recording event");
        require(c.u64()==++last,"recording sequence gap");stream=c.u64();ticks=c.u64();
        require(ticks>=last_ticks,"recording clock moved backwards");last_ticks=ticks;payload.resize(bytes);
        require(fread(payload.data(),1,bytes,file)==bytes,"truncated recording payload");require(recording::checksum(payload)==hash,"corrupt recording payload");return true;
    }
};
void generate(wchar_t const* path){
    SetEnvironmentVariableW(L"C3X_RENDERER_RECORD_FILE",path);Device d;
    c3x_native_images::Lifetimes tracked;int object=0;
    auto observe=[&](int operation,void* image,int context){bool revoked=false;
        auto eligible=tracked.observe(operation,image,context,GetCurrentThreadId(),&revoked);
        recording::journal().native(recording::lifetime,operation,image,nullptr,unsigned(context),int(eligible)|(int(revoked)<<1),operation==C3X_NATIVE_DESTROY);
    };
    observe(C3X_NATIVE_VERIFY,nullptr,0);observe(C3X_NATIVE_INIT,&object,0);observe(C3X_NATIVE_MAP,&object,0);
    observe(C3X_NATIVE_DC,&object,C3X_NATIVE_COPY);observe(C3X_NATIVE_BITS,&object,0);
    observe(C3X_NATIVE_MAP,&object,0);observe(C3X_NATIVE_INIT,&object,0);observe(C3X_NATIVE_DESTROY,&object,0);
    {
        Compositor gpu(d.device.Get(),d.context.Get(),256u*1024u*1024u,true);
        // The formerly insufficient 128 MiB live-image budget must fail replay
        // of the same admitted surface family, without changing the recording.
        std::vector<Id> dense;
        for(unsigned n=0;n<16;++n){auto id=gpu.create(2240,1260,n%2?Format::bgra32:Format::rgb555);require(id!=0,"dense source admission");dense.push_back(id);}
        for(auto id:dense)require(gpu.destroy(id),"dense retirement");
        constexpr unsigned w=48,h=32;Rect full={0,0,w,h},part={4,6,40,28};
        for(auto format:{Format::rgb555,Format::rgb565}){
            auto target=gpu.create(w,h,format),detail=gpu.create(w,h,Format::bgra32),sprite=gpu.create(w,h,Format::bgra32);
            std::vector<unsigned> pixels(w*h);for(unsigned n=0;n<pixels.size();++n)pixels[n]=0x80001020u+n%32;
            require(gpu.upload(sprite,1,pixels.data(),pixels.size()),"sprite upload");
            Command seed={Kind::fill,target,0,full,full,0,0,123};require(gpu.submit(&seed,1),"seed");
            Command expand={Kind::expand,detail,target,full,full,0,0,65536};require(gpu.submit(&expand,1),"expand");
            Command blend={Kind::native_blend,target,sprite,full,part,0,0,1,target,detail,detail};require(gpu.submit(&blend,1),"HUD blend");
            Command self={Kind::copy,target,target,{4,4,40,28},full,0,0};require(gpu.submit(&self,1),"overlap");
            auto words=read_pixels(d.device.Get(),d.context.Get(),gpu.texture(target));gpu.record_readback(target,words.data(),words.size());
            words=read_pixels(d.device.Get(),d.context.Get(),gpu.texture(detail));gpu.record_readback(detail,words.data(),words.size());
            auto source=gpu.attach_source(gpu.texture(sprite));require(source!=0,"readonly source");
            require(!gpu.upload(source,1,pixels.data(),pixels.size()),"readonly upload rejection");
            require(gpu.destroy(source),"readonly retirement");
            // External map import, followed by a real recorded display boundary.
            D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
            desc.Format=DXGI_FORMAT_B8G8R8A8_UNORM;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;
            D3D11_SUBRESOURCE_DATA initial={pixels.data(),w*4,0};ComPtr<ID3D11Texture2D> texture;checked(d.device->CreateTexture2D(&desc,&initial,&texture));
            require(gpu.import_bgra(detail,texture.Get()),"external map import");
            // A resolved direct unit patch replaces only its footprint; pixels
            // outside it must survive replay unchanged.
            unsigned patch[4]={0xffabcdef,0xffabcdef,0xfffedcba,0xfffedcba};D3D11_BOX box={3,5,0,5,7,1};
            d.context->UpdateSubresource(gpu.texture(detail),0,&box,patch,8,0);gpu.record_external(detail,{3,5,5,7});
            desc.BindFlags=D3D11_BIND_RENDER_TARGET;ComPtr<ID3D11Texture2D> screen;ComPtr<ID3D11RenderTargetView> view;
            checked(d.device->CreateTexture2D(&desc,nullptr,&screen));checked(d.device->CreateRenderTargetView(screen.Get(),nullptr,&view));
            require(gpu.display(detail,view.Get(),w,h,full),"recorded display");
            require(gpu.destroy(target)&&gpu.destroy(detail)&&gpu.destroy(sprite),"retire sources");require(!gpu.destroy(sprite),"stale identity rejection");
        }
    }
    recording::journal().finish(recording::closed);std::puts("PASS generated production composition recording");
}
struct FrameSelection {
    std::uint64_t first=1,last=UINT64_MAX,step=32,seconds_first=0,seconds_last=UINT64_MAX;
    bool explicit_range=false;
    bool includes(std::uint64_t frame,std::uint64_t ticks,std::uint64_t frequency)const{
        if(frame<first||frame>last||ticks/frequency<seconds_first||ticks/frequency>=seconds_last)return false;
        return explicit_range?(frame-first)%step==0:frame==1||frame%step==0;
    }
};
int replay(wchar_t const* path,bool paced,bool show,std::uint64_t cap,wchar_t const* frames,bool require_ambient,FrameSelection const& selection){
    Reader reader(path);Device d;std::map<std::uint64_t,Stream> streams;
    c3x_native_images::Lifetimes lifetimes;
    struct Call {unsigned operation,value;};std::map<std::uint64_t,Call> calls;
    std::uint64_t lifetime_checks=0,lifetime_mismatches=0,revocations=0,ready_checks=0,ready_losses=0;
    std::uint64_t lost_at=0,longest_loss=0;bool had_ready=false,ready=false;
    if(frames)require(CreateDirectoryW(frames,nullptr)||GetLastError()==ERROR_ALREADY_EXISTS,"cannot create frame directory");
    NativePresenter presenter;HWND window=nullptr;
    unsigned kind=0,reason=999;std::uint64_t owner=0,ticks=0,events=0,checks=0,commands=0,externals=0,displays=0,native=0,visuals=0,peak=0,exports=0;
    double submit_ms=0;LARGE_INTEGER frequency={},started={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&started);
    recording::Bytes bytes;std::uint64_t first_tick=0;bool first=true;
    auto cleanup=[&]{presenter.reset();if(window){DestroyWindow(window);window=nullptr;}};
    try{while(reader.next(kind,owner,ticks,bytes)){
        ++events;if(first){first_tick=ticks;first=false;}
        if(paced){LARGE_INTEGER now={};QueryPerformanceCounter(&now);auto due=double(ticks-first_tick)/double(reader.frequency);
            auto elapsed=double(now.QuadPart-started.QuadPart)/double(frequency.QuadPart);
            require(due<=900,"recording exceeds bounded pacing window");if(due>elapsed)Sleep(DWORD((due-elapsed)*1000));}
        MSG msg;while(PeekMessageA(&msg,nullptr,0,0,PM_REMOVE)){TranslateMessage(&msg);DispatchMessageA(&msg);}
        recording::Cursor c{bytes};
        if(kind==recording::begin){auto budget=c.u64();require(owner&&streams.size()<16&&!streams.count(owner)&&budget<=1024ull*1024*1024,"invalid compositor session");
            streams[owner].gpu=std::make_unique<Compositor>(d.device.Get(),d.context.Get(),cap?cap:budget);}
        else if(kind==recording::end){require(streams.erase(owner)==1,"unknown compositor end");}
        else if(kind==recording::stop){require(!owner,"invalid footer owner");reason=c.u32();require(reason<=recording::unsupported,"invalid stop reason");reader.footer=true;}
        else if(kind==recording::native_begin||kind==recording::lifetime){
            auto token=c.u64();auto operation=c.u32();auto image=c.u64();c.u64();auto value=c.u32(),expected=c.u32();
            auto thread=reader.version>=3?c.u32():1u;++native;
            if(kind==recording::lifetime){
                require(image<=UINT32_MAX,"invalid native identity");bool revoked=false;
                auto eligible=lifetimes.observe(int(operation),reinterpret_cast<void*>(std::uintptr_t(image)),int(value),thread,&revoked);
                ++lifetime_checks;lifetime_mismatches+=(unsigned(eligible)|(unsigned(revoked)<<1))!=expected;
                if(revoked){++revocations;std::printf("LIFETIME_REVOKED event=%llu seconds=%.6f image=%llu operation=%u context=%u\n",events,double(ticks)/reader.frequency,image,operation,value);}
            }else{require(calls.size()<8192&&!calls.count(token),"invalid native call nesting");calls[token]={operation,value};}
        }
        else if(kind==recording::native_end){auto token=c.u64();int result=int(c.u32());auto call=calls.find(token);require(call!=calls.end(),"unmatched native completion");
            if(call->second.operation==C3X_NATIVE_VISUAL_POLICY){
                ++ready_checks;bool next=result>0;
                if(ready&&!next){lost_at=ticks;++ready_losses;std::printf("AMBIENT_DEPENDENCY_LOST event=%llu seconds=%.6f\n",events,double(ticks)/reader.frequency);}
                if(!ready&&next&&had_ready)longest_loss=std::max(longest_loss,ticks-lost_at);
                ready=next;had_ready|=ready;
            }calls.erase(call);
        }
        else if(kind==recording::visual){c.u64();c.u64();c.u32();c.u64();c.u64();c.u64();c.u32();++visuals;}
        else{
            auto it=streams.find(owner);require(it!=streams.end(),"missing compositor session");auto& s=it->second;auto& gpu=*s.gpu;
            if(kind==recording::create){auto id=c.u64();auto w=c.u32(),h=c.u32(),format=c.u32(),borrowed=c.u32();require(format<=2&&borrowed<=1,"invalid image format");
                Id actual=0;
                if(borrowed){require(w&&h&&w<=2240&&h<=1260,"invalid borrowed image dimensions");D3D11_TEXTURE2D_DESC desc={};desc.Width=w;desc.Height=h;desc.MipLevels=desc.ArraySize=desc.SampleDesc.Count=1;
                    desc.Format=DXGI_FORMAT_R32_UINT;desc.BindFlags=D3D11_BIND_SHADER_RESOURCE;ComPtr<ID3D11Texture2D> texture;checked(d.device->CreateTexture2D(&desc,nullptr,&texture));actual=gpu.attach_source(texture.Get(),Format(format));}
                else actual=gpu.create(w,h,Format(format));
                require(bool(id)==bool(actual),"image admission differs from recording");if(id){require(!s.ids.count(id),"duplicate live image identity");s.ids[id]=actual;}}
            else if(kind==recording::destroy){auto id=c.u64();auto expected=c.u32();require(gpu.destroy(s.id(id))==bool(expected),"destroy result differs");if(expected)s.ids.erase(id);}
            else if(kind==recording::upload){auto id=c.u64(),revision=c.u64();auto expected=c.u32();auto pixels=c.pixels();require(gpu.upload(s.id(id),revision,pixels.data(),pixels.size())==bool(expected),"upload result differs");}
            else if(kind==recording::external||kind==recording::checkpoint){auto id=c.u64();Rect area={};if(kind==recording::external)area={int(c.u32()),int(c.u32()),int(c.u32()),int(c.u32())};auto pixels=c.pixels();
                if(kind==recording::external){require(gpu.replay_external(s.id(id),pixels.data(),pixels.size(),area),"external input rejected");++externals;}
                else{require(read_pixels(d.device.Get(),d.context.Get(),gpu.texture(s.id(id)))==pixels,"pixel checkpoint differs");++checks;}}
            else if(kind==recording::submit){auto expected=c.u32(),count=c.u32();require(count&&count<=2048,"invalid command count");std::vector<Command> batch;batch.reserve(count);
                for(unsigned n=0;n<count;++n){auto op=c.command();op.destination=s.id(op.destination);op.source=s.id(op.source);op.background=s.id(op.background);op.detail=s.id(op.detail);op.background_detail=s.id(op.background_detail);op.program=s.id(op.program);batch.push_back(op);}
                LARGE_INTEGER a={},b={};QueryPerformanceCounter(&a);auto ok=gpu.submit(batch.data(),batch.size());QueryPerformanceCounter(&b);
                submit_ms+=1000.*double(b.QuadPart-a.QuadPart)/double(frequency.QuadPart);require(ok==bool(expected),"draw result differs");commands+=count;}
            else if(kind==recording::display){auto id=c.u64();auto expected=c.u32();Rect area={int(c.u32()),int(c.u32()),int(c.u32()),int(c.u32())};
                auto texture=gpu.texture(s.id(id));require(texture!=nullptr,"missing display image");D3D11_TEXTURE2D_DESC desc={};texture->GetDesc(&desc);
                if(!window){WNDCLASSW wc={};wc.lpfnWndProc=DefWindowProcW;wc.hInstance=GetModuleHandleW(nullptr);wc.lpszClassName=L"C3XCompositionReplay";RegisterClassW(&wc);
                    window=CreateWindowW(wc.lpszClassName,L"C3X composition replay",WS_POPUP|(show?WS_VISIBLE:0),0,0,int(desc.Width),int(desc.Height),nullptr,nullptr,wc.hInstance,nullptr);require(window!=nullptr,"replay window creation failed");}
                SetWindowPos(window,nullptr,0,0,int(desc.Width),int(desc.Height),SWP_NOACTIVATE|SWP_NOZORDER);
                require(presenter.prepare(window,d.device.Get(),desc.Width,desc.Height,true),"replay presenter preparation failed");
                auto ok=gpu.display(s.id(id),presenter.view(),desc.Width,desc.Height,area);require(ok==bool(expected),"display result differs");
                if(ok){d.context->CopyResource(presenter.buffer(),presenter.retained());require(presenter.present()==C3X_RENDERER_RESULT_OK,"replay present failed");++displays;
                    if(frames&&selection.includes(displays,ticks,reader.frequency)){
                        auto pixels=read_pixels(d.device.Get(),d.context.Get(),presenter.retained());
                        BITMAPFILEHEADER file={};BITMAPINFOHEADER info={};info.biSize=sizeof(info);info.biWidth=desc.Width;info.biHeight=-LONG(desc.Height);info.biPlanes=1;info.biBitCount=32;
                        file.bfType=0x4d42;file.bfOffBits=sizeof(file)+sizeof(info);file.bfSize=file.bfOffBits+DWORD(pixels.size()*4);
                        wchar_t name[64];swprintf_s(name,L"/frame-%06llu.bmp",displays);auto output=std::wstring(frames)+name;FILE* f=nullptr;
                        require(GetFileAttributesW(output.c_str())==INVALID_FILE_ATTRIBUTES,"replay frame already exists");
                        require(!_wfopen_s(&f,output.c_str(),L"wb")&&f,"cannot save replay frame");
                        bool saved=fwrite(&file,sizeof(file),1,f)==1&&fwrite(&info,sizeof(info),1,f)==1&&fwrite(pixels.data(),4,pixels.size(),f)==pixels.size();auto closed=fclose(f);
                        require(saved&&!closed,"replay frame write failed");++exports;std::printf("REPLAY_FRAME display=%llu event=%llu ticks=%llu frequency=%llu seconds=%.6f endpoint=composition_draw_accepted\n",displays,events,ticks,reader.frequency,double(ticks)/reader.frequency);
                    }
                }}
            else throw std::runtime_error("unknown composition event");
            std::uint64_t total=0;for(auto const& item:streams)total+=item.second.gpu->stats().resident_bytes;peak=std::max(peak,total);
        }
        c.done();
    }
    cleanup();
    require(!selection.explicit_range||exports>0,"no recorded display in requested range");
    if(had_ready&&!ready)longest_loss=std::max(longest_loss,ticks-lost_at);
    std::printf("{\"native_lifetime_checks\":%llu,\"native_lifetime_mismatches\":%llu,\"thread_evidence\":\"%s\",\"demanded_revocations\":%llu,\"ambient_readiness_checks\":%llu,\"ambient_dependency_losses\":%llu,\"longest_observed_loss_seconds\":%.6f,\"ambient_continuity_accepted\":false}\n",
        lifetime_checks,lifetime_mismatches,reader.version>=3?"recorded":"legacy_assumed_owner",revocations,ready_checks,ready_losses,double(longest_loss)/reader.frequency);
    require(!lifetime_mismatches,"native lifetime decisions differ from recording");
    if(require_ambient)require(had_ready&&ready_checks&&!ready_losses,"recorded workload loses ambient readiness");
    bool verified=reader.footer&&checks&&commands&&reason==recording::closed;
    // A budget/time/memory stop is a checked prefix, not session acceptance.
    std::printf("{\"status\":\"%s\",\"scope\":\"native_gpu_composition_only\",\"events\":%llu,\"commands\":%llu,\"pixel_checks\":%llu,\"external_snapshots\":%llu,\"display_boundaries\":%llu,\"native_observations\":%llu,\"visual_observations\":%llu,\"peak_compositor_bytes\":%llu,\"submit_cpu_ms\":%.3f,\"footer\":%s,\"stop_reason\":%u,\"full_gameplay_replayed\":false,\"retained_animation_replayed\":false,\"performance_accepted\":false}\n",
        verified?"verified_composition":"verified_prefix",events,commands,checks,externals,displays,native,visuals,peak,submit_ms,reader.footer?"true":"false",reason);
    return checks&&commands?0:2;
    }catch(...){cleanup();throw;}
}
int wmain(int argc,wchar_t** argv){
    try{require(argc>=2,"usage: replay_composition recording [--paced] [--show] [--budget-mib N], or --generate file");
        if(!wcscmp(argv[1],L"--generate")){require(argc==3,"generate requires a file");generate(argv[2]);return 0;}
        bool paced=false,show=false,ambient=false,explicit_step=false;std::uint64_t cap=0;wchar_t const* frames=nullptr;FrameSelection selection;
        auto number=[](wchar_t const* value,std::uint64_t maximum){
            require(*value>=L'0'&&*value<=L'9',"invalid numeric replay option");wchar_t* end=nullptr;
            auto result=_wcstoui64(value,&end,10);require(end&&!*end&&result<=maximum,"invalid numeric replay option");return std::uint64_t(result);
        };
        for(int n=2;n<argc;++n){if(!wcscmp(argv[n],L"--paced"))paced=true;else if(!wcscmp(argv[n],L"--show"))show=true;
            else if(!wcscmp(argv[n],L"--frames")&&n+1<argc)frames=argv[++n];
            else if(!wcscmp(argv[n],L"--frame-range")&&n+2<argc){selection.first=number(argv[++n],1000000000);selection.last=number(argv[++n],1000000000);
                require(selection.first&&selection.first<=selection.last,"invalid frame range");selection.explicit_range=true;}
            else if(!wcscmp(argv[n],L"--seconds")&&n+2<argc){selection.seconds_first=number(argv[++n],900);selection.seconds_last=number(argv[++n],900);
                require(selection.seconds_first<selection.seconds_last,"invalid seconds range");selection.explicit_range=true;}
            else if(!wcscmp(argv[n],L"--frame-step")&&n+1<argc){selection.step=number(argv[++n],1000000);require(selection.step>0,"invalid frame step");explicit_step=true;}
            else if(!wcscmp(argv[n],L"--require-ambient"))ambient=true;
            else if(!wcscmp(argv[n],L"--budget-mib")&&n+1<argc){auto value=_wtoi(argv[++n]);require(value>0&&value<=1024,"invalid budget");cap=std::uint64_t(value)*1024*1024;}
            else throw std::runtime_error("unknown replay option");}
        require(frames||(!selection.explicit_range&&!explicit_step),"frame selection requires --frames");
        if(selection.explicit_range&&!explicit_step)selection.step=1;
        return replay(argv[1],paced,show,cap,frames,ambient,selection);
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL composition recording/replay: %s\n",e.what());return 1;}
}
