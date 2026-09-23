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
        require(argc==4||(argc==5&&std::wstring(argv[4])==L"--direct"),
            "usage: live_roundtrip.exe HELPER DLL CAPTURE [--direct]");
        bool direct=argc==5;
        SegmentReader journal(argv[3]);Event event;
        require(journal.next(event)&&event.kind==Kind::manifest,"recording manifest missing");
        // Shared presentation is driven by the fixture. The direct surface
        // must keep animating with no visual request from the window owner.
        SetEnvironmentVariableA("C3X_RENDERER_MANUAL_VISUAL",direct?"0":"1");
        if(direct)SetEnvironmentVariableA("C3X_RENDERER_DIRECT_SURFACE_TRIAL","1");
        c3x_remote_scene::Backend helper(argv[1],argv[2]);
        bool definitions=false,submitted=false;unsigned polls=0;
        c3x_remote_scene::CameraOutput adopted;
        while(journal.next(event)){
            if(event.kind==Kind::footer)break;
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
        c3x_renderer_world_page_v1 delta={};
        require(helper.world_delta_scope(delta)==C3X_RENDERER_RESULT_OK&&
            delta.first==UINT_MAX&&delta.capacity==128,"x64 move delta scope failed");
        c3x_renderer_tile_v1 changed_tile={};changed_tile.tile_x=0;changed_tile.tile_y=0;
        changed_tile.tile_flags=C3X_RENDERER_TILE_PREFETCH|C3X_RENDERER_TILE_TOPOLOGY_HALO|
            C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;
        delta.count=1;delta.tiles=&changed_tile;
        auto stale=delta;++stale.identity.viewer_epoch;
        require(helper.world_delta_submit(stale,C3X_RENDERER_RESULT_OK)==C3X_RENDERER_RESULT_SUPERSEDED,
            "stale viewer move delta was accepted");
        auto unchanged=delta;unchanged.count=0;
        require(helper.world_delta_submit(unchanged,C3X_RENDERER_RESULT_OK)==C3X_RENDERER_RESULT_OK,
            "unchanged move delta was not a no-op");
        require(helper.world_delta_submit(delta,C3X_RENDERER_RESULT_OK)==C3X_RENDERER_RESULT_OK,
            "accepted move delta did not reach Renderer64");
        c3x_renderer_unit_spawn_v1 birth={sizeof(birth)};
        birth.unit_id=9001;birth.tile_x=0;birth.tile_y=0;birth.unit_type_id=1;
        birth.owner_id=0;birth.visible=1;
        birth.map_epoch=delta.identity.map_epoch;birth.viewer_epoch=delta.identity.viewer_epoch;
        birth.presentation_time_ticks=1000;birth.presentation_frequency=1000000;
        require(helper.unit_spawn(birth)==C3X_RENDERER_RESULT_OK,"scoped unit birth did not reach Renderer64");
        c3x_renderer_unit_move_v1 move={sizeof(move)};
        move.unit_id=birth.unit_id;move.old_x=0;move.old_y=0;move.new_x=2;move.new_y=0;
        move.action=2;move.target_visible=1;move.source_visible=1;
        move.map_epoch=birth.map_epoch;move.viewer_epoch=birth.viewer_epoch;
        move.presentation_time_ticks=1100;move.presentation_frequency=birth.presentation_frequency;
        require(helper.unit_move(move)==C3X_RENDERER_RESULT_OK,"ordered unit move did not reach Renderer64");
        c3x_renderer_unit_state_v1 state={sizeof(state)};
        state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;state.unit_id=birth.unit_id;
        state.tile_x=2;state.tile_y=0;state.unit_type_id=1;state.owner_id=0;
        state.action=13;state.damage=1;state.max_hp=3;state.visible=1;
        state.map_epoch=birth.map_epoch;state.viewer_epoch=birth.viewer_epoch;
        state.presentation_time_ticks=1200;state.presentation_frequency=birth.presentation_frequency;
        require(helper.unit_state(state)==C3X_RENDERER_RESULT_OK,"action and damage did not reach Renderer64");
        auto sent=helper.replay_stats().sequence;
        state.presentation_time_ticks=1250;
        require(helper.unit_state(state)==C3X_RENDERER_RESULT_OK&&helper.replay_stats().sequence==sent,
            "unchanged unit state caused another x64 IPC roundtrip");
        auto wrong_viewer=state;++wrong_viewer.viewer_epoch;
        require(helper.unit_state(wrong_viewer)==C3X_RENDERER_RESULT_SUPERSEDED,
            "stale-viewer unit state was accepted");
        state.kind=C3X_RENDERER_UNIT_STATE_RETIRE;state.action=-1;state.max_hp=0;
        state.presentation_time_ticks=1300;
        require(helper.unit_state(state)==C3X_RENDERER_RESULT_OK,"unit retirement did not reach Renderer64");
        state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;state.action=13;state.max_hp=3;
        state.presentation_time_ticks=1400;
        require(helper.unit_state(state)==C3X_RENDERER_RESULT_BAD_ARGUMENT,
            "retired unit identity revived without a new birth");
        birth.tile_x=2;birth.presentation_time_ticks=1500;
        require(helper.unit_spawn(birth)==C3X_RENDERER_RESULT_OK,
            "new accepted birth could not reuse a retired ID");
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
        std::atomic<unsigned> independent_frames{0},pending_frames{0},failed_frames{0};
        std::atomic<int> first_failure{0};
        std::exception_ptr independent_error;
        std::thread independent([&]{
            if(direct)return;
            try{for(unsigned n=0;n<12;++n){
                    LARGE_INTEGER ticks={},frequency={};QueryPerformanceCounter(&ticks);QueryPerformanceFrequency(&frequency);
                    int code=helper.visual(ticks.QuadPart,frequency.QuadPart);
                    if(code==C3X_RENDERER_RESULT_OK)independent_frames.fetch_add(1,std::memory_order_relaxed);
                    else if(code==C3X_RENDERER_RESULT_PENDING)pending_frames.fetch_add(1,std::memory_order_relaxed);
                    else{failed_frames.fetch_add(1,std::memory_order_relaxed);if(!first_failure.load())first_failure.store(code);}
                    Sleep(25);
                }}catch(...){independent_error=std::current_exception();}
        });
        Sleep(450); // The x86 window owner deliberately does no work or message pumping.
        independent.join();
        if(independent_error)std::rethrow_exception(independent_error);
        if(!direct&&independent_frames.load(std::memory_order_relaxed)<2){char detail[180];
            std::snprintf(detail,sizeof(detail),"visible ambient frames stopped: ok=%u pending=%u failed=%u first=%d",
                independent_frames.load(),pending_frames.load(),failed_frames.load(),first_failure.load());
            throw std::runtime_error(detail);}
        require(helper.replay_display(final_pixels,final_width,final_height)&&
            initial_width==final_width&&initial_height==final_height&&initial_pixels!=final_pixels,
            "ambient presentation did not change the displayed map while the owner was blocked");
        if(direct){
            c3x_native_images::ScreenSnapshot patch;
            patch.window=window;patch.width=width;patch.height=height;
            patch.area={width/2-8,height/2-8,width/2+8,height/2+8};
            patch.pixels.resize(std::size_t((width+1)&~1)*height,0x7fff);
            require(helper.screen(&patch)==C3X_RENDERER_RESULT_OK,
                "direct-to-native partial transfer failed");
            std::vector<unsigned> patched;unsigned pw=0,ph=0;
            require(helper.replay_display(patched,pw,ph)&&pw==final_width&&ph==final_height&&
                patched[0]==final_pixels[0]&&
                patched[std::size_t(height/2)*width+width/2]!=final_pixels[std::size_t(height/2)*width+width/2],
                "native partial transfer lost retained direct pixels");
            require(helper.screen(nullptr)==C3X_RENDERER_RESULT_OK,
                "native screen release after direct transfer failed");
            require(helper.present(offer)==C3X_RENDERER_RESULT_OK,
                "direct surface did not resume after native transfer");
        }
        offer={sizeof(offer)};offer.action=2;
        require(helper.present(offer)==C3X_RENDERER_RESULT_OK,"x86 native handoff failed");
        c3x_native_images::ScreenSnapshot native;
        native.window=window;native.width=width;native.height=height;
        native.area={0,0,width,height};native.pixels.resize(std::size_t((width+1)&~1)*height,0x4210);
        require(helper.screen(&native)==C3X_RENDERER_RESULT_OK&&
            helper.screen(nullptr)==C3X_RENDERER_RESULT_OK,"native CPU fallback presentation failed");
        require(helper.reset()==C3X_RENDERER_RESULT_OK,"helper reset failed");
        DestroyWindow(window);window=nullptr;
        std::printf("PASS live x64 camera to %s: %u polls, %dx%d frame, %u visible animations, %u window-owner visual requests during block\n",
            direct?"x64 direct surface":"x86 native presenter",polls+1,width,height,adopted.value.camera.output.visible_animation_count,
            direct?0u:independent_frames.load(std::memory_order_relaxed));
        return 0;
    }catch(std::exception const& error){
        if(window)DestroyWindow(window);
        std::fprintf(stderr,"FAIL live helper roundtrip: %s\n",error.what());return 1;
    }
}
