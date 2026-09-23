#define NOMINMAX
#include <windows.h>
#include <atomic>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "../input_recording/journal.h"
#include "../remote_scene_output.h"
#include "../remote_renderer_backend.h"

using namespace c3x_inputs;
using namespace c3x_helper_trial;

int wmain(int argc,wchar_t** argv){
    HWND window=nullptr;
    try{
        require(argc==4,"usage: live_roundtrip.exe HELPER DLL CAPTURE");
        SegmentReader journal(argv[3]);Event event;
        require(journal.next(event)&&event.kind==Kind::manifest,"recording manifest missing");
        SetEnvironmentVariableA("C3X_RENDERER_MANUAL_VISUAL","1");
        c3x_remote_scene::Backend helper(argv[1],argv[2]);
        bool definitions=false,submitted=false;unsigned polls=0;
        c3x_remote_scene::CameraOutput adopted;
        while(journal.next(event)){
            if(event.kind!=Kind::native_bridge&&event.kind!=Kind::scene)continue;
            Reader recorded{event.payload};recorded.u64();recorded.u64();
            if(event.kind==Kind::native_bridge&&event.flags==8){
                bool present[4]={};std::string path[4];
                for(unsigned n=0;n<4;++n)path[n]=recorded.string(32768,&present[n]);recorded.done();
                require(helper.definitions(present[0]?path[0].c_str():nullptr,
                    present[1]?path[1].c_str():nullptr,present[2]?path[2].c_str():nullptr,
                    present[3]?path[3].c_str():nullptr)==C3X_RENDERER_RESULT_OK,"definition publication failed");
                definitions=true;
            }else if(event.kind==Kind::scene&&event.flags==3&&definitions){
                Bytes values(event.payload.begin()+recorded.at,event.payload.end());Reader input{values};
                c3x_renderer_camera_identity_v1 identity={};c3x_renderer_camera_identity_v1_fields(input,identity);
                Frame frame_value;frame(input,frame_value);input.done();
                c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame_value.value,identity};
                c3x_renderer_i64 ticket=0;
                require(helper.camera_begin(request,ticket)==C3X_RENDERER_RESULT_PENDING&&ticket>0,"camera begin failed");
                for(;polls<120000;++polls){
                    auto code=helper.camera_poll(ticket,adopted.value);
                    if(code==C3X_RENDERER_RESULT_PENDING){Sleep(1);continue;}
                    require(code==C3X_RENDERER_RESULT_OK,"camera poll failed");
                    require(adopted.value.camera.ticket==ticket&&adopted.value.image.ticket>0&&
                        adopted.value.image.map_image>0&&adopted.value.camera.frame.tile_count==frame_value.value.tile_count,
                        "adopted camera identity or occurrences changed");
                    break;
                }
                require(polls<120000,"camera readiness timed out");submitted=true;break;
            }
        }
        require(submitted,"recording has no GPU scene following definitions");
        c3x_renderer_world_page_v1 page={};bool page_ready=false;
        for(unsigned attempt=0;attempt<2000;++attempt){
            auto code=helper.world_query(page);
            if(code==C3X_RENDERER_RESULT_PENDING){Sleep(1);continue;}
            require(code==C3X_RENDERER_RESULT_OK&&page.first==0&&page.capacity==128,
                "x64 world page query failed");
            page_ready=true;break;
        }
        require(page_ready,"x64 world page was never available");
        c3x_renderer_tile_v1 page_storage[128]={};page.tiles=page_storage;
        auto rejected=helper.world_submit(page,C3X_RENDERER_RESULT_PENDING);
        if(rejected!=C3X_RENDERER_RESULT_PENDING){char detail[160];
            std::snprintf(detail,sizeof(detail),"rejected native page result=%d first=%u count=%u",rejected,page.first,page.count);
            throw std::runtime_error(detail);}
        c3x_renderer_world_page_v1 retry={};
        require(helper.world_query(retry)==C3X_RENDERER_RESULT_OK&&retry.first==page.first,
            "rejected native page advanced the world cursor");
        auto const& view=adopted.value;auto width=view.image.width,height=view.image.height;
        require(width>0&&height>0&&width<=2240&&height<=1260,"invalid adopted extent");
        window=CreateWindowExW(0,L"STATIC",L"C3X live helper roundtrip",WS_POPUP,0,0,width,height,
            nullptr,nullptr,GetModuleHandleW(nullptr),nullptr);
        require(window!=nullptr,"x86 presentation window failed");
        c3x_renderer_gpu_present_v1 offer={sizeof(offer)};offer.ticket=view.image.ticket;
        offer.image=view.image.map_image;offer.window=window;offer.width=width;offer.height=height;
        offer.area[2]=width;offer.area[3]=height;
        require(helper.present(offer)==C3X_RENDERER_RESULT_OK,"x64 final frame was not presented by x86");
        require(adopted.value.camera.output.visible_animation_count>0,
            "fixture has no visible animation to test independent presentation");
        std::vector<unsigned> initial_pixels,final_pixels;unsigned initial_width=0,initial_height=0,final_width=0,final_height=0;
        require(helper.replay_display(initial_pixels,initial_width,initial_height),"initial displayed frame readback failed");
        std::atomic<unsigned> independent_frames{0};
        std::exception_ptr independent_error;
        std::thread independent([&]{
            try{for(unsigned n=0;n<12;++n){
                    LARGE_INTEGER ticks={},frequency={};QueryPerformanceCounter(&ticks);QueryPerformanceFrequency(&frequency);
                    if(helper.visual(ticks.QuadPart,frequency.QuadPart)==C3X_RENDERER_RESULT_OK)
                        independent_frames.fetch_add(1,std::memory_order_relaxed);
                    Sleep(25);
                }}catch(...){independent_error=std::current_exception();}
        });
        Sleep(450); // The x86 window owner deliberately does no work or message pumping.
        independent.join();
        if(independent_error)std::rethrow_exception(independent_error);
        require(independent_frames.load(std::memory_order_relaxed)>=2,
            "visible ambient frames stopped while the x86 owner was blocked");
        require(helper.replay_display(final_pixels,final_width,final_height)&&
            initial_width==final_width&&initial_height==final_height&&initial_pixels!=final_pixels,
            "ambient presentation did not change the displayed map while the owner was blocked");
        offer={sizeof(offer)};offer.action=2;
        require(helper.present(offer)==C3X_RENDERER_RESULT_OK,"x86 native handoff failed");
        c3x_native_images::ScreenSnapshot native;
        native.window=window;native.width=width;native.height=height;
        native.area={0,0,width,height};native.pixels.resize(std::size_t((width+1)&~1)*height,0x4210);
        require(helper.screen(&native)==C3X_RENDERER_RESULT_OK&&
            helper.screen(nullptr)==C3X_RENDERER_RESULT_OK,"native CPU fallback presentation failed");
        require(helper.reset()==C3X_RENDERER_RESULT_OK,"helper reset failed");
        DestroyWindow(window);window=nullptr;
        std::printf("PASS live x64 camera to x86 native presenter: %u polls, %dx%d frame, %u visible animations, %u frames under blocked owner\n",
            polls+1,width,height,adopted.value.camera.output.visible_animation_count,
            independent_frames.load(std::memory_order_relaxed));
        return 0;
    }catch(std::exception const& error){
        if(window)DestroyWindow(window);
        std::fprintf(stderr,"FAIL live helper roundtrip: %s\n",error.what());return 1;
    }
}
