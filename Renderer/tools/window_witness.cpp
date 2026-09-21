// External, bounded window evidence. This process never injects, sends game
// input, reads game memory, or creates a competing game presenter.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>
#include <wincodec.h>
#include <psapi.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cstdint>

using namespace winrt;
using namespace winrt::Windows::Graphics::Capture;
namespace fs=std::filesystem;
constexpr unsigned maximum_width=2400,maximum_height=1400;
constexpr std::uint64_t disk_limit=2ull*1024*1024*1024;
void require(bool value,char const* message){if(!value)throw std::runtime_error(message);}
long long qpc(){LARGE_INTEGER value={};QueryPerformanceCounter(&value);return value.QuadPart;}
bool valid_size(int width,int height){return width>0&&height>0&&width<=int(maximum_width)&&height<=int(maximum_height);}
struct Target {DWORD pid=0;HWND window=nullptr;long long area=0;};
BOOL CALLBACK find_window(HWND window,LPARAM parameter){
    auto& target=*reinterpret_cast<Target*>(parameter);DWORD pid=0;GetWindowThreadProcessId(window,&pid);
    RECT bounds={};if(pid==target.pid&&IsWindowVisible(window)&&!GetWindow(window,GW_OWNER)&&GetWindowRect(window,&bounds)){
        auto area=1ll*(bounds.right-bounds.left)*(bounds.bottom-bounds.top);
        if(area>target.area){target.area=area;target.window=window;}}
    return TRUE;
}
LRESULT CALLBACK test_window(HWND window,UINT message,WPARAM wp,LPARAM lp){
    if(message==WM_PAINT){PAINTSTRUCT paint={};auto dc=BeginPaint(window,&paint);RECT rect={};GetClientRect(window,&rect);
        auto brush=CreateSolidBrush((GetTickCount64()/200)%2?RGB(230,20,10):RGB(10,20,230));FillRect(dc,&rect,brush);DeleteObject(brush);EndPaint(window,&paint);return 0;}
    return DefWindowProcW(window,message,wp,lp);
}
struct AddressSpace {
    std::uint64_t limit=0;USHORT machine=0,native_machine=0;bool large_address_aware=false;
    explicit AddressSpace(HANDLE process){
        require(IsWow64Process2(process,&machine,&native_machine)!=FALSE,"target architecture unavailable");
        SYSTEM_INFO system={};GetNativeSystemInfo(&system);
        limit=reinterpret_cast<std::uintptr_t>(system.lpMaximumApplicationAddress)+1ull;
        wchar_t name[32768];DWORD count=32768;require(QueryFullProcessImageNameW(process,0,name,&count)!=FALSE,"target executable unavailable");
        // Read the executable header, never target process bytes. A 64-bit
        // observer must not count addresses a non-LAA 32-bit game cannot use.
        std::ifstream file(fs::path(name),std::ios::binary);IMAGE_DOS_HEADER dos={};
        require(bool(file.read(reinterpret_cast<char*>(&dos),sizeof(dos)))&&dos.e_magic==IMAGE_DOS_SIGNATURE&&dos.e_lfanew>0&&dos.e_lfanew<16*1024*1024,"target DOS header invalid");
        file.seekg(dos.e_lfanew);DWORD signature=0;IMAGE_FILE_HEADER header={};
        require(bool(file.read(reinterpret_cast<char*>(&signature),sizeof(signature)))&&signature==IMAGE_NT_SIGNATURE&&
            bool(file.read(reinterpret_cast<char*>(&header),sizeof(header))),"target PE header invalid");
        large_address_aware=(header.Characteristics&IMAGE_FILE_LARGE_ADDRESS_AWARE)!=0;
        if(header.Machine==IMAGE_FILE_MACHINE_I386||header.Machine==IMAGE_FILE_MACHINE_ARMNT){
            limit=large_address_aware?0x100000000ull:0x80000000ull;
        }else if(!large_address_aware)limit=0x80000000ull;
    }
};
void process_memory(std::ostream& stream,HANDLE process,AddressSpace const& space){
    auto began=qpc();
    PROCESS_MEMORY_COUNTERS_EX memory={};memory.cb=sizeof(memory);
    require(GetProcessMemoryInfo(process,reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),sizeof(memory))!=FALSE,"target memory counters unavailable");
    std::uintptr_t address=0;std::uint64_t free_bytes=0,largest=0,committed=0,reserved=0;unsigned regions=0;
    MEMORY_BASIC_INFORMATION block={};
    while(address<space.limit&&VirtualQueryEx(process,reinterpret_cast<void*>(address),&block,sizeof(block))){
        auto next=std::uint64_t(reinterpret_cast<std::uintptr_t>(block.BaseAddress))+block.RegionSize;
        require(next>address,"invalid target memory region");next=std::min(next,space.limit);auto length=next-address;
        if(block.State==MEM_FREE){free_bytes+=length;largest=std::max(largest,length);}
        else if(block.State==MEM_COMMIT)committed+=length;else if(block.State==MEM_RESERVE)reserved+=length;
        ++regions;address=std::uintptr_t(next);
        require(regions<=1000000,"target memory region bound");
    }
    PROCESS_MEMORY_COUNTERS_EX observer={};observer.cb=sizeof(observer);
    FILETIME created={},exited={},kernel={},user={};
    require(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&observer),sizeof(observer))&&
        GetProcessTimes(GetCurrentProcess(),&created,&exited,&kernel,&user),"observer resource counters unavailable");
    auto ticks=[](FILETIME value){return std::uint64_t(value.dwHighDateTime)<<32|value.dwLowDateTime;};
    stream<<"{\"event\":\"process_memory\",\"begin_qpc\":"<<began<<",\"qpc\":"<<qpc()<<",\"private_bytes\":"<<memory.PrivateUsage
        <<",\"working_set_bytes\":"<<memory.WorkingSetSize<<",\"peak_working_set_bytes\":"<<memory.PeakWorkingSetSize
        <<",\"free_bytes\":"<<free_bytes<<",\"largest_free_bytes\":"<<largest<<",\"committed_bytes\":"<<committed
        <<",\"reserved_bytes\":"<<reserved<<",\"regions\":"<<regions<<",\"enumerated_through\":"<<address
        <<",\"address_limit\":"<<space.limit<<",\"address_scan_complete\":"<<(address==space.limit?"true":"false")
        <<",\"observer_private_bytes\":"<<observer.PrivateUsage<<",\"observer_working_set_bytes\":"<<observer.WorkingSetSize
        <<",\"observer_cpu_100ns\":"<<(ticks(kernel)+ticks(user))<<"}\n";
}
struct Witness {
    com_ptr<ID3D11Device> device;com_ptr<ID3D11DeviceContext> context;
    com_ptr<ID3D11Texture2D> staging;com_ptr<IWICImagingFactory> encoder;
    unsigned width=0,height=0;std::vector<unsigned char> bgr;
    Witness(){
        D3D_FEATURE_LEVEL level={};check_hresult(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            nullptr,0,D3D11_SDK_VERSION,device.put(),&level,context.put()));
        check_hresult(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,__uuidof(IWICImagingFactory),encoder.put_void()));
    }
    unsigned save(Direct3D11CaptureFrame const& frame,fs::path const& path){
        auto size=frame.ContentSize();require(valid_size(size.Width,size.Height),"window witness extent exceeds supported bound");
        auto access=frame.Surface().as<::Windows::Graphics::DirectX::Direct3D11::IDirect3DDxgiInterfaceAccess>();
        com_ptr<ID3D11Texture2D> source;check_hresult(access->GetInterface(__uuidof(ID3D11Texture2D),source.put_void()));
        D3D11_TEXTURE2D_DESC description={};source->GetDesc(&description);
        require(description.Format==DXGI_FORMAT_B8G8R8A8_UNORM&&description.SampleDesc.Count==1&&
            unsigned(size.Width)<=description.Width&&unsigned(size.Height)<=description.Height,"unsupported witness surface");
        if(width!=unsigned(size.Width)||height!=unsigned(size.Height)){
            staging=nullptr;width=unsigned(size.Width);height=unsigned(size.Height);
            description.Width=width;description.Height=height;description.MipLevels=description.ArraySize=1;
            description.BindFlags=0;description.MiscFlags=0;description.Usage=D3D11_USAGE_STAGING;description.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
            check_hresult(device->CreateTexture2D(&description,nullptr,staging.put()));bgr.resize(std::size_t(width)*height*3);
        }
        D3D11_BOX box={0,0,0,width,height,1};context->CopySubresourceRegion(staging.get(),0,0,0,0,source.get(),0,&box);
        D3D11_MAPPED_SUBRESOURCE mapped={};check_hresult(context->Map(staging.get(),0,D3D11_MAP_READ,0,&mapped));
        auto data=static_cast<unsigned char const*>(mapped.pData);
        for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x){auto from=data+std::size_t(y)*mapped.RowPitch+x*4;
            auto to=bgr.data()+(std::size_t(y)*width+x)*3;std::copy_n(from,3,to);}
        context->Unmap(staging.get(),0);
        auto middle=bgr.data()+(std::size_t(height/2)*width+width/2)*3;
        unsigned color=unsigned(middle[2])<<16|unsigned(middle[1])<<8|middle[0];
        com_ptr<IWICStream> stream;check_hresult(encoder->CreateStream(stream.put()));check_hresult(stream->InitializeFromFilename(path.c_str(),GENERIC_WRITE));
        com_ptr<IWICBitmapEncoder> jpeg;check_hresult(encoder->CreateEncoder(GUID_ContainerFormatJpeg,nullptr,jpeg.put()));
        check_hresult(jpeg->Initialize(stream.get(),WICBitmapEncoderNoCache));
        com_ptr<IWICBitmapFrameEncode> output;com_ptr<IPropertyBag2> options;check_hresult(jpeg->CreateNewFrame(output.put(),options.put()));
        PROPBAG2 option={};option.pstrName=const_cast<wchar_t*>(L"ImageQuality");VARIANT quality={};quality.vt=VT_R4;quality.fltVal=.92f;
        check_hresult(options->Write(1,&option,&quality));check_hresult(output->Initialize(options.get()));check_hresult(output->SetSize(width,height));
        auto format=GUID_WICPixelFormat24bppBGR;check_hresult(output->SetPixelFormat(&format));require(format==GUID_WICPixelFormat24bppBGR,"unsupported witness encoder format");
        check_hresult(output->WritePixels(height,width*3,unsigned(bgr.size()),bgr.data()));check_hresult(output->Commit());check_hresult(jpeg->Commit());return color;
    }
};
int wmain(int argc,wchar_t** argv){
    fs::path out;HWND owned=nullptr;bool complete=false,created=false;std::string reason="error";std::uint64_t frames=0,bytes=0,drained=0;
    handle fixture_finished;
    try{
        require(argc==6||argc==7,"usage: window_witness PID|self-test NEW_DIRECTORY SECONDS FPS sampled-window-evidence [FIXTURE_EVENT]");
        require(std::wstring(argv[5])==L"sampled-window-evidence","explicit sampled-window scope is required");
        bool self_test=std::wstring(argv[1])==L"self-test";
        auto number=[](wchar_t const* value){std::size_t end=0;auto n=std::stoul(value,&end);require(value[end]==0,"invalid witness argument");return n;};
        unsigned seconds=number(argv[3]),fps=number(argv[4]);require(seconds>=1&&seconds<=900&&fps>=1&&fps<=10,"witness duration/cadence bounds");
        out=argv[2];require(!fs::exists(out)&&fs::create_directories(out),"window evidence directory must be new");created=true;
        require(fs::space(out).available>=disk_limit+64ull*1024*1024,"insufficient witness disk space");
        init_apartment(apartment_type::multi_threaded);SetProcessDPIAware();require(GraphicsCaptureSession::IsSupported(),"Windows window capture unsupported");
        Target target;target.pid=self_test?GetCurrentProcessId():number(argv[1]);require(target.pid!=0,"invalid target process");
        if(self_test){WNDCLASSW window_class={};window_class.lpfnWndProc=test_window;window_class.hInstance=GetModuleHandleW(nullptr);window_class.lpszClassName=L"C3XWindowWitnessTest";
            require(RegisterClassW(&window_class)!=0,"witness test class");owned=CreateWindowW(window_class.lpszClassName,L"C3X window witness test",WS_POPUP|WS_VISIBLE,100,100,320,240,nullptr,nullptr,window_class.hInstance,nullptr);
            require(owned!=nullptr,"witness test window");UpdateWindow(owned);target.window=owned;}
        else{auto deadline=GetTickCount64()+30000;while(!target.window&&GetTickCount64()<deadline){EnumWindows(find_window,reinterpret_cast<LPARAM>(&target));if(!target.window)Sleep(100);}}
        require(target.window!=nullptr,"no target window appeared");
        handle process{OpenProcess(PROCESS_QUERY_INFORMATION|PROCESS_VM_READ,FALSE,target.pid)};require(bool(process),"target memory access unavailable");
        AddressSpace space(process.get());
        if(argc==7){
            std::wstring name=argv[6];std::wstring prefix=L"Local\\C3XRendererWitness-";
            require(!self_test&&name.rfind(prefix,0)==0&&name.size()==prefix.size()+32&&
                name.find_first_not_of(L"0123456789abcdef",prefix.size())==std::wstring::npos,"invalid fixture witness event");
            handle ready{CreateEventW(nullptr,TRUE,FALSE,name.c_str())};
            fixture_finished=handle{CreateEventW(nullptr,TRUE,FALSE,(name+L"-done").c_str())};
            require(bool(ready)&&bool(fixture_finished),"fixture witness events unavailable");
            auto deadline=GetTickCount64()+300000;
            while(WaitForSingleObject(ready.get(),100)!=WAIT_OBJECT_0){
                DWORD code=0;require(GetExitCodeProcess(process.get(),&code)&&code==STILL_ACTIVE&&IsWindow(target.window),"fixture ended before window evidence interval");
                require(GetTickCount64()<deadline,"fixture window evidence interval timed out");
            }
        }
        auto factory=get_activation_factory<GraphicsCaptureItem,IGraphicsCaptureItemInterop>();GraphicsCaptureItem item{nullptr};
        check_hresult(factory->CreateForWindow(target.window,guid_of<GraphicsCaptureItem>(),put_abi(item)));
        auto size=item.Size();require(valid_size(size.Width,size.Height),"initial window extent exceeds witness bound");Witness witness;
        auto dxgi=witness.device.as<IDXGIDevice>();com_ptr<IInspectable> inspectable;
        check_hresult(CreateDirect3D11DeviceFromDXGIDevice(dxgi.get(),inspectable.put()));
        auto direct=inspectable.as<winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice>();
        auto format=winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized;
        auto pool=Direct3D11CaptureFramePool::CreateFreeThreaded(direct,format,2,size);auto session=pool.CreateCaptureSession(item);
        LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);auto started=qpc();FILETIME utc={};GetSystemTimePreciseAsFileTime(&utc);auto after_utc=qpc();
        std::ofstream metadata(out/"started.json");metadata<<"{\"schema\":1,\"scope\":\"sampled compositor window evidence; not scanout or all frames\",\"qpc_frequency\":"<<frequency.QuadPart
            <<",\"started_qpc\":"<<started<<",\"utc_filetime_100ns\":"<<(std::uint64_t(utc.dwHighDateTime)<<32|utc.dwLowDateTime)
            <<",\"qpc_after_utc\":"<<after_utc<<",\"process_machine\":"<<space.machine<<",\"native_machine\":"<<space.native_machine
            <<",\"address_limit\":"<<space.limit<<",\"large_address_aware\":"<<(space.large_address_aware?"true":"false")
            <<",\"fps_limit\":"<<fps<<",\"seconds_limit\":"<<seconds<<",\"disk_limit\":"<<disk_limit<<",\"lossy_jpeg\":true,\"uncaptured_frames_unknown\":true}\n";
        metadata.close();std::ofstream timeline(out/"timeline.jsonl");require(bool(timeline),"witness timeline unavailable");session.StartCapture();
        auto begin=GetTickCount64(),next=begin,next_memory=begin;long long previous_source=0;std::set<unsigned> colors;std::set<std::pair<int,int>> extents;
        while(GetTickCount64()-begin<seconds*1000ull){
            if(fs::exists(out/"stop.txt")){reason="requested";break;}
            DWORD current_pid=0;GetWindowThreadProcessId(target.window,&current_pid);if(!IsWindow(target.window)||current_pid!=target.pid){reason="window_closed";break;}
            if(self_test){auto elapsed=GetTickCount64()-begin;SetWindowPos(owned,nullptr,100,100,elapsed<2000?320:448,elapsed<2000?240:288,SWP_NOZORDER|SWP_NOACTIVATE);InvalidateRect(owned,nullptr,FALSE);}
            MSG message={};while(PeekMessageW(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageW(&message);}
            if(GetTickCount64()<next){Sleep(5);continue;}next=GetTickCount64()+1000/fps;
            if(GetTickCount64()>=next_memory){process_memory(timeline,process.get(),space);next_memory=GetTickCount64()+1000;}
            auto arrived=qpc();Direct3D11CaptureFrame frame{nullptr};
            for(unsigned n=0;n<4;++n){auto latest=pool.TryGetNextFrame();if(!latest)break;if(frame){frame.Close();++drained;}frame=std::move(latest);}
            DWORD foreground_pid=0;GetWindowThreadProcessId(GetForegroundWindow(),&foreground_pid);
            if(!frame){timeline<<"{\"event\":\"no_frame\",\"qpc\":"<<arrived<<",\"minimized\":"<<(IsIconic(target.window)?"true":"false")<<"}\n";timeline.flush();continue;}
            auto content=frame.ContentSize();require(valid_size(content.Width,content.Height),"resized window exceeds witness bound");
            if(content.Width>size.Width||content.Height>size.Height){frame.Close();size=content;pool.Recreate(direct,format,2,size);
                timeline<<"{\"event\":\"resize_gap\",\"qpc\":"<<arrived<<"}\n";timeline.flush();continue;}
            auto source_time=frame.SystemRelativeTime().count();require(source_time>=previous_source,"window compositor time moved backward");
            if(bytes>=disk_limit||fs::space(out).available<64ull*1024*1024){frame.Close();reason="disk_limit";break;}
            wchar_t name[64];swprintf_s(name,L"window-%06llu.jpg",static_cast<unsigned long long>(frames+1));auto temporary=out/L"pending.jpg";
            auto color=witness.save(frame,temporary);frame.Close();auto length=fs::file_size(temporary);
            if(length>disk_limit-bytes){fs::remove(temporary);reason="disk_limit";break;}
            fs::rename(temporary,out/name);bytes+=length;++frames;colors.insert(color);extents.emplace(content.Width,content.Height);
            timeline<<"{\"frame\":"<<frames<<",\"compositor_100ns\":"<<source_time<<",\"previous_compositor_100ns\":"<<previous_source
                <<",\"arrival_qpc\":"<<arrived<<",\"saved_qpc\":"<<qpc()<<",\"width\":"<<content.Width<<",\"height\":"<<content.Height
                <<",\"target_foreground\":"<<(foreground_pid==target.pid?"true":"false")<<",\"bytes\":"<<length<<"}\n";
            timeline.flush();require(bool(timeline),"witness timeline write failed");previous_source=source_time;
        }
        session.Close();pool.Close();complete=reason=="requested"||GetTickCount64()-begin>=seconds*1000ull;if(reason!="requested"&&complete)reason="duration";
        if(self_test)require(frames>=3&&colors.count(0xe6140a)&&colors.count(0x0a14e6)&&(seconds<3||extents.size()>=2),"window evidence failed motion/resize control");
        require(frames>0,"no window frames captured");
    }catch(hresult_error const& error){std::cerr<<"window capture HRESULT="<<std::hex<<unsigned(error.code())<<'\n';complete=false;reason="capture_error";}
    catch(std::exception const& error){std::cerr<<error.what()<<'\n';complete=false;reason="capture_error";}
    if(owned)DestroyWindow(owned);
    if(fixture_finished)SetEvent(fixture_finished.get());
    if(created&&fs::is_directory(out)){std::ofstream result(out/"finished.json");result<<"{\"complete\":"<<(complete?"true":"false")<<",\"reason\":\""<<reason
        <<"\",\"frames\":"<<frames<<",\"bytes\":"<<bytes<<",\"dequeued_unsaved_frames\":"<<drained<<",\"uncaptured_frames_unknown\":true}\n";result.flush();if(!result)return 1;}
    return complete?0:1;
}
