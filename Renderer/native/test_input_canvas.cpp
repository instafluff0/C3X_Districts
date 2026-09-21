#define NOMINMAX
#include "asset_content_hash.h"
#include "input_recording/canvas.h"
#include "input_recording/inspect.h"
#include <iostream>
#include <sstream>
using namespace c3x_inputs;
struct TestCanvas {
    HDC dc=nullptr;HBITMAP bitmap=nullptr;void* pixels=nullptr;
    explicit TestCanvas(HANDLE section=nullptr){
        BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=64;info.bmiHeader.biHeight=-32;
        info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;dc=CreateCompatibleDC(nullptr);
        bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&pixels,section,0);require(dc&&bitmap&&pixels,"test DIB allocation");
        SelectObject(dc,bitmap);std::memset(pixels,0,64*32*4);
    }
    ~TestCanvas(){DeleteDC(dc);DeleteObject(bitmap);}
};
int main(int argc,char** argv){try{
    require(argc==2,"test requires a new capture directory");
    require(SetEnvironmentVariableA("C3X_RENDERER_INPUT_RECORD_DIR",argv[1])!=0,"canvas test recording environment");
    auto& capture=canvas_capture();unsigned first=0,second=0;std::exception_ptr failure;
    TestCanvas a;
    {std::lock_guard<std::mutex> lock(capture.mutex);first=capture.describe(a.dc).id;}
    // Create the shared scratch on a short-lived thread, then use it after that
    // thread exits. A NULL compatibility DC would lose its owning thread here.
    std::thread caller([&]{try{TestCanvas b;std::lock_guard<std::mutex> lock(canvas_capture().mutex);
        auto& value=canvas_capture().describe(b.dc);second=value.id;canvas_capture().copy(value);
        require(value.word(0,0)==0,"initial CPU scratch contents");}catch(...){failure=std::current_exception();}});caller.join();
    if(failure)std::rethrow_exception(failure);
    require(first&&second&&first!=second,"CPU canvas identities collide across threads");
    {std::lock_guard<std::mutex> lock(capture.mutex);auto& value=capture.describe(a.dc);require(value.id==first,"CPU canvas identity changed across caller");
        static_cast<unsigned*>(a.pixels)[0]=0x123456;capture.copy(value);require(value.word(0,0)==0x123456,"CPU scratch lost with its creating thread");}
    auto mapping=CreateFileMappingW(INVALID_HANDLE_VALUE,nullptr,PAGE_READWRITE,0,65536,nullptr);require(mapping!=nullptr,"test shared DIB mapping");
    {TestCanvas mapped(mapping);bool rejected=false;try{capture.describe(mapped.dc);}catch(std::runtime_error const&){rejected=true;}
        require(rejected,"mapped CPU DIB silently treated as independent storage");}CloseHandle(mapping);
    Canvas left,right;unsigned storage[32]={};left.native_bits=reinterpret_cast<unsigned char*>(storage);left.native_bytes=64;
    right.native_bits=left.native_bits+32;right.native_bytes=64;require(aliases(left,right)&&aliases(right,left),"overlapping CPU canvas ranges accepted");
    right.native_bits=left.native_bits+64;require(!aliases(left,right),"adjacent CPU canvases confused with aliases");
    int calls=0,returned=0;
    {std::lock_guard<std::mutex> lock(capture.mutex);std::thread concurrent([&]{c3x_renderer_unit_v1 unit={};
        returned=cpu_unit(1,unit,nullptr,nullptr,nullptr,0,[&]{++calls;return 7;});});concurrent.join();}
    require(calls==1&&returned==7&&!runtime().active(),"concurrent capture blocked or changed original draw");
    runtime().finish();reset_canvas_capture();
    InputInspection inspection;std::ostringstream timeline;inspection.read(argv[1],timeline);
    require(inspection.verified&&!inspection.complete&&inspection.stop==unsigned(Stop::unsupported),"concurrent CPU capture incorrectly certified complete");
    std::cout<<"PASS CPU input recorder: shared identities/scratch, exited caller, mapped/overlapping storage, concurrent fail-closed with original draw preserved\n";return 0;
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}}
