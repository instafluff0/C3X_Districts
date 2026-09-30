#pragma once
#include "capture_model.h"
#include "pass_counts.h"

// Bounded source-grounded capture/adoption pilot. Native tile anchors stay128/64.
// SCROLL_CALL is the opt-in complete fresh-call trace; stage counts below name
// the last executed fresh draw at each boundary, not all worker submissions.
inline int sandbox_scroll_witness(HMODULE module,HWND window,c3x_renderer_frame_v1 const& original) {
    using Begin=int(*)(c3x_renderer_camera_request_v1 const*,c3x_renderer_i64*);
    using Poll=int(*)(c3x_renderer_i64,c3x_renderer_gpu_camera_view_v1*);
    using Draw=int(*)(c3x_renderer_frame_v1 const*,char const*,int,int,int,int,int,int,int,float);
    auto begin=reinterpret_cast<Begin>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
    auto ready=reinterpret_cast<Poll>(GetProcAddress(module,"c3x_renderer_trial_camera_ready"));
    auto adopt=reinterpret_cast<Poll>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll_view"));
    auto draw=reinterpret_cast<Draw>(GetProcAddress(module,"c3x_sandbox_draw_fresh"));
    auto present=reinterpret_cast<int(*)(HWND,c3x_renderer_frame_v1 const*,int,int,int,int,int,int,int)>(GetProcAddress(module,"c3x_sandbox_present"));
    auto invalidate=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_sandbox_scroll_cache_invalidate"));
    auto metrics=reinterpret_cast<void(*)(unsigned*,unsigned*,unsigned*,unsigned*,unsigned*,unsigned*)>(GetProcAddress(module,"c3x_sandbox_scroll_cache_metrics"));
    auto reasons=reinterpret_cast<void(*)(unsigned*)>(GetProcAddress(module,"c3x_sandbox_scroll_reasons"));
    auto capture=reinterpret_cast<int(*)(char const*)>(GetProcAddress(module,"c3x_sandbox_witness_capture"));
    auto content=reinterpret_cast<int(*)(char const*)>(GetProcAddress(module,"c3x_sandbox_world_content"));
    auto counts=reinterpret_cast<void(*)(SandboxPassCounts*)>(GetProcAddress(module,"c3x_sandbox_pass_counts"));
    auto flush=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_trial_trace_flush"));
    auto enabled=[](char const* key){char v[16]={};return GetEnvironmentVariableA(key,v,sizeof(v))&&v[0]=='1';};
    char value[64]={},directory[4*MAX_PATH]={};
    float zoom=GetEnvironmentVariableA("C3X_SANDBOX_SCROLL_ZOOM",value,sizeof(value))?float(std::atof(value)):1.25f;
    bool force=enabled("C3X_SANDBOX_SCROLL_FORCE_FULL"),quality=enabled("C3X_SANDBOX_SCROLL_QUALITY");
    bool scripted=enabled("C3X_SANDBOX_SCROLL_SCRIPTED_TIME"),diagnostic=enabled("C3X_SANDBOX_PASS_COUNTS");
    if(diagnostic)std::setvbuf(stdout,nullptr,_IONBF,0);
    if(!begin||!ready||!adopt||!draw||!present||!invalidate||!metrics||!reasons||
       original.tile_width!=128||original.tile_height!=64||original.target_width!=2240||original.target_height!=1260||
       !std::isfinite(zoom)||zoom<1.f||zoom>3.f)return 60;
    if(quality&&(!capture||!GetEnvironmentVariableA("C3X_SANDBOX_SCROLL_OUTPUT",directory,sizeof(directory))))return 61;
    SandboxCameraWitness witness(original);if(witness.canonical.empty())return 62;
    int span=original.world_width_tiles*64;
    // Main_Screen_Form::FUN_004de2a0 gives slow scrollspeed32/16. Its edge
    // path FUN_004de430 uses2*(step/(edge_coordinate+1)): coordinate15=>4/2,5=>10/4.
    // Arrows use32/16 directly. These are copied-native fixture inputs, not
    // an injected native-game witness or smaller substitute tile geometry.
    int slow_x=32,slow_y=16,edge_distance=15;
    int ex=2*(slow_x/(edge_distance+1)),ey=2*(slow_y/(edge_distance+1));
    std::vector<SandboxCameraWitness::View> views={{0,0,zoom,"origin"},
        {ex,ey,zoom,"edge_positive"},{2*ex,2*ey,zoom,"edge_continue"},
        {-ex,-ey,zoom,"edge_negative"},{128,64,zoom,"new_strip"},
        {0,0,zoom,"resident_return"}};
    GetEnvironmentVariableA("C3X_SANDBOX_SCROLL_TRACE",value,sizeof(value));
    bool regression=!std::strcmp(value,"regression"),motion=!std::strcmp(value,"motion");
    if(regression){
        views.push_back({slow_x,0,zoom,"arrow_right"});views.push_back({-slow_x,0,zoom,"arrow_left"});
        views.push_back({0,slow_y,zoom,"arrow_down"});views.push_back({0,-slow_y,zoom,"arrow_up"});
        views.push_back({1,1,zoom,"fraction_positive"});views.push_back({-1,-1,zoom,"fraction_negative"});
        views.push_back({512,320,zoom,"recenter"});views.push_back({192,96,zoom,"equivalent"});
        views.push_back({span+192,96,zoom,"wrap"});views.push_back({-span+192,96,zoom,"negative_wrap"});
        views.push_back({-640,-256,zoom,"jump"});views.push_back({0,0,zoom,"jump_return"});
        views.push_back({0,0,zoom,"content_change",0,1});views.push_back({0,0,zoom,"visibility_change",0,2});
        views.push_back({0,0,zoom,"lighting_change",0,5});
    }
    if(motion){
        int steps=GetEnvironmentVariableA("C3X_SANDBOX_SCROLL_STEPS",value,sizeof(value))?std::clamp(std::atoi(value),8,90):32;
        views.clear();for(int n=0;n<steps;++n)views.push_back({n*ex,n*ey,zoom,"edge_forward"});
        for(int n=steps-2;n>=0;--n)views.push_back({n*ex,n*ey,zoom,"edge_reverse"});
    }
    std::vector<unsigned> world;if(original.world_topology&&original.world_topology_count)
        world.assign(original.world_topology,original.world_topology+original.world_topology_count);
    if(world.empty())return 63;
    auto revision=original.world_topology_revision;
    c3x_renderer_camera_identity_v1 identity={1,2,3,4};
    LARGE_INTEGER frequency={},start={},previous_done={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&start);
    auto ms=[&](LONGLONG ticks){return 1000.*double(ticks)/double(frequency.QuadPart);};
    auto snapshot=[&](unsigned* m,unsigned* r){metrics(&m[0],&m[1],&m[2],&m[3],&m[4],&m[5]);reasons(r);};
    auto report=[&](char const* stage,std::size_t index,unsigned const* a,unsigned const* b,unsigned const* ra,unsigned const* rb){
        std::printf("SCROLL_CACHE frame=%zu stage=%s depth_copies=%u scrolls=%u full_draws=%u reflection_reuses=%u reflection_draws=%u pose_builds=%u fractional=%u guard=%u signature=%u environment=%u lights=%u shadow=%u depth_origin=%u anchors=%u strip_fills=%u explicit_reset=%u projection=%u\n",
            index,stage,b[0]-a[0],b[1]-a[1],b[2]-a[2],b[3]-a[3],b[4]-a[4],b[5]-a[5],
            rb[0]-ra[0],rb[1]-ra[1],rb[2]-ra[2],rb[3]-ra[3],rb[4]-ra[4],rb[5]-ra[5],rb[6]-ra[6],rb[7]-ra[7],rb[8]-ra[8],rb[9]-ra[9],rb[10]-ra[10]);
    };
    std::set<std::pair<int,int>> previous;
    int hour=original.hour;
    std::printf("SCROLL_CONFIG zoom=%.9f native_width=128 native_height=64 target=2240,1260 views=%zu trace=%s clock=%s force_full=%u quality=%u input_step=%d,%d reasons_overlap=1\n",
        zoom,views.size(),regression?"regression":motion?"motion":"pilot",scripted?"scripted":"wall",unsigned(force),unsigned(quality),ex,ey);
    for(std::size_t index=0;index<views.size();++index){
        auto view=views[index];
        if(motion&&!quality){LARGE_INTEGER now={};QueryPerformanceCounter(&now);while(ms(now.QuadPart-start.QuadPart)<double(index*17)){
            MSG message={};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}Sleep(1);QueryPerformanceCounter(&now);}}
        LARGE_INTEGER entered={},captured={},submitted={},prepared={},adopted={},display_begin={},drawn={},done={};QueryPerformanceCounter(&entered);
        if(view.change==1){auto changed=std::find_if(witness.canonical.begin(),witness.canonical.end(),[&](auto const& t){return t.terrain_type<5&&t.real_terrain_type<5&&t.city_id<0&&std::abs(t.anchor_x-original.target_width/2)<256&&std::abs(t.anchor_y-original.target_height/2)<256;});
            if(changed==witness.canonical.end())return 64;changed->terrain_type=changed->real_terrain_type=changed->terrain_type==2?3:2;
            world[unsigned(changed->tile_y*original.world_width_tiles+changed->tile_x)/2]=unsigned(changed->terrain_type)|(unsigned(changed->real_terrain_type)<<8)|(changed->river_code<<16);++identity.scene_epoch;++revision;}
        if(view.change==2){++identity.visibility_epoch;for(auto& t:witness.canonical)if(std::abs(t.anchor_x-original.target_width/2)<256)t.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;}
        if(view.change==5)hour=hour==12?0:12;
        auto tiles=witness.capture(view);if(tiles.empty())return 65;
        auto frame=original;frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());frame.hour=hour;
        frame.world_topology=world.data();frame.world_topology_count=unsigned(world.size());frame.world_topology_revision=revision;
        frame.presentation_frequency=1000;frame.presentation_time_ticks=29500+(scripted?c3x_renderer_i64(index*33):c3x_renderer_i64(ms(entered.QuadPart-start.QuadPart)));
        frame.dirty_flags=index&&!view.change?0:C3X_RENDERER_DIRTY_ALL;
        std::set<std::pair<int,int>> occurrences;unsigned added=0,removed=0,rendered=0,prefetched=0,cities=0;
        auto const& anchor=witness.canonical.front();
        for(auto const& t:tiles){if(t.anchor_x-t.tile_x*64!=anchor.anchor_x-anchor.tile_x*64+view.x||t.anchor_y-t.tile_y*32!=anchor.anchor_y-anchor.tile_y*32+view.y)return 66;
            if(!occurrences.emplace(t.tile_x,t.tile_y).second)return 67;bool render=(t.tile_flags&C3X_RENDERER_TILE_RENDER)!=0;rendered+=unsigned(render);cities+=unsigned(render&&t.city_id>=0);prefetched+=unsigned((t.tile_flags&C3X_RENDERER_TILE_PREFETCH)!=0);}
        for(auto t:occurrences)added+=unsigned(!previous.count(t));for(auto t:previous)removed+=unsigned(!occurrences.count(t));previous=occurrences;
        unsigned before[6]={},after_prep[6]={},after_display[6]={},rb[11]={},rp[11]={},rd[11]={};snapshot(before,rb);
        if(diagnostic&&counts){SandboxPassCounts work;counts(&work);std::printf("SCROLL_WORK_STAGE frame=%zu stage=seed last_fresh_call=1\n",index);sandbox_report_pass_counts(work,index,"scroll_seed");}
        QueryPerformanceCounter(&captured);c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,identity};
        c3x_renderer_i64 ticket=0;int result=begin(&request,&ticket);QueryPerformanceCounter(&submitted);if(result!=C3X_RENDERER_RESULT_PENDING)return 68;
        c3x_renderer_gpu_camera_view_v1 shown={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(shown)};auto deadline=GetTickCount64()+120000;unsigned polls=0;
        do{result=ready(ticket,&shown);++polls;if(result==C3X_RENDERER_RESULT_PENDING)Sleep(1);}while(result==C3X_RENDERER_RESULT_PENDING&&GetTickCount64()<deadline);
        QueryPerformanceCounter(&prepared);if(result!=C3X_RENDERER_RESULT_OK){if(flush)flush();std::printf("SCROLL_FAILED frame=%zu stage=ready ticket=%lld result=%d\n",index,ticket,result);return 69;}
        do{result=adopt(ticket,&shown);if(result==C3X_RENDERER_RESULT_PENDING)Sleep(1);}while(result==C3X_RENDERER_RESULT_PENDING&&GetTickCount64()<deadline);
        QueryPerformanceCounter(&adopted);if(result!=C3X_RENDERER_RESULT_OK)return 70;
        auto expected=frame;expected.tiles=shown.camera.frame.tiles;expected.world_topology=nullptr;expected.world_topology_count=0;
        if(std::memcmp(&shown.camera.frame,&expected,sizeof(expected))||shown.camera.ticket!=ticket||std::memcmp(&shown.camera.identity,&identity,sizeof(identity))||!shown.camera.frame.tiles||std::memcmp(shown.camera.frame.tiles,tiles.data(),tiles.size()*sizeof(tiles[0])))return 71;
        // Match the production adapter: cache coordinates are the absolute
        // anchor origin, while view.x/y are only input deltas.
        int render_camera_x=tiles[0].anchor_x-tiles[0].tile_x*frame.tile_width/2;
        int render_camera_y=tiles[0].anchor_y-tiles[0].tile_y*frame.tile_height/2;
        snapshot(after_prep,rp);report("prep",index,before,after_prep,rb,rp);
        if(diagnostic&&counts){SandboxPassCounts work;counts(&work);std::printf("SCROLL_WORK_STAGE frame=%zu stage=prep last_fresh_call=1\n",index);sandbox_report_pass_counts(work,index,"scroll_prep");}
        QueryPerformanceCounter(&display_begin);if(force)invalidate();result=draw(&frame,nullptr,render_camera_x,render_camera_y,0,0,0,0,0,zoom);QueryPerformanceCounter(&drawn);
        if(!result)result=present(window,&frame,0,0,0,0,0,render_camera_x,render_camera_y);QueryPerformanceCounter(&done);if(result)return 73;
        snapshot(after_display,rd);report("display",index,after_prep,after_display,rp,rd);
        std::printf("SCROLL_FRAME frame=%zu view=%s camera=%d,%d render_camera=%d,%d zoom=%.9f ticks=%lld ticket=%lld tiles=%u added=%u removed=%u render=%u prefetch=%u cities=%u capture_ms=%.6f submission_ms=%.6f preparation_ms=%.6f adoption_ms=%.6f draw_ms=%.6f present_ms=%.6f first_correct_ms=%.6f interval_ms=%.6f done_ms=%.6f geometry_ticks=%lld worker_draw_ticks=%lld upload_bytes=%u polls=%u exact_anchors=1\n",
            index,view.name,view.x,view.y,render_camera_x,render_camera_y,zoom,frame.presentation_time_ticks,ticket,frame.tile_count,added,removed,rendered,prefetched,cities,
            ms(captured.QuadPart-entered.QuadPart),ms(submitted.QuadPart-captured.QuadPart),ms(prepared.QuadPart-submitted.QuadPart),ms(adopted.QuadPart-prepared.QuadPart),ms(drawn.QuadPart-display_begin.QuadPart),ms(done.QuadPart-drawn.QuadPart),ms(done.QuadPart-entered.QuadPart),previous_done.QuadPart?ms(done.QuadPart-previous_done.QuadPart):0.,ms(done.QuadPart-start.QuadPart),shown.camera.output.geometry_ticks,shown.camera.output.draw_ticks,shown.camera.output.geometry_upload_bytes,polls);
        previous_done=done;
        if(diagnostic&&counts){SandboxPassCounts work;counts(&work);std::printf("SCROLL_WORK_STAGE frame=%zu stage=display last_fresh_call=1\n",index);sandbox_report_pass_counts(work,index,"scroll_display");}
        if(quality){char prefix[4*MAX_PATH]={};sprintf_s(prefix,"%s\\frame_%03zu_retained",directory,index);if(capture(prefix))return 74;
            if(content){char path[4*MAX_PATH]={};sprintf_s(path,"%s\\frame_%03zu_owners.csv",directory,index);if(content(path))return 75;}
            invalidate();if(draw(&frame,nullptr,render_camera_x,render_camera_y,0,0,0,0,0,zoom))return 76;
            sprintf_s(prefix,"%s\\frame_%03zu_full",directory,index);if(capture(prefix))return 77;
            std::printf("SCROLL_ORACLE frame=%zu view=%s ticks=%lld independent_raster=1 timed=0\n",index,view.name,frame.presentation_time_ticks);}
        std::fflush(stdout);
    }
    std::printf("SCROLL_WITNESS pass views=%zu trace=%s quality=%u scope=standalone_copied_native_preparation_adoption\n",views.size(),regression?"regression":motion?"motion":"pilot",unsigned(quality));return 0;
}
