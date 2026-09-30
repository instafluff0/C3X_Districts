#pragma once
#include "capture_model.h"

int sandbox_camera_witness(HMODULE module,HWND window,c3x_renderer_frame_v1 const& original) {
    using Begin=int(*)(c3x_renderer_camera_request_v1 const*,c3x_renderer_i64*);
    using Poll=int(*)(c3x_renderer_i64,c3x_renderer_gpu_camera_view_v1*);
    auto begin=reinterpret_cast<Begin>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
    auto ready=reinterpret_cast<Poll>(GetProcAddress(module,"c3x_renderer_trial_camera_ready"));
    auto adopt=reinterpret_cast<Poll>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll_view"));
    auto draw=reinterpret_cast<int(*)(c3x_renderer_frame_v1 const*,char const*,int,int,int,int,int,int,int,float)>(GetProcAddress(module,"c3x_sandbox_draw_fresh"));
    auto present=reinterpret_cast<int(*)(HWND,c3x_renderer_frame_v1 const*,int,int,int,int,int,int,int)>(GetProcAddress(module,"c3x_sandbox_present"));
    auto capture=reinterpret_cast<int(*)(char const*)>(GetProcAddress(module,"c3x_sandbox_witness_capture"));
    auto content=reinterpret_cast<int(*)(char const*)>(GetProcAddress(module,"c3x_sandbox_world_content"));
    auto counts=reinterpret_cast<void(*)(SandboxPassCounts*)>(GetProcAddress(module,"c3x_sandbox_pass_counts"));
    if(!begin||!ready||!adopt||!draw||!present)return 20;
    SandboxCameraWitness witness(original);
    int span=original.world_width_tiles*original.tile_width/2;
    std::vector<SandboxCameraWitness::View> views={{0,0,1,"origin"},{96,48,1,"diagonal"},
        {288,144,1,"strip"},{192,96,1,"equivalent"},{span+192,96,1,"wrap"},{-span+192,96,1,"negative_wrap"},{-640,-256,1,"jump"},
        {0,0,1.25f,"zoom_settle"},{96,48,1.25f,"zoom_pan"}};
    char scenario[32]={};GetEnvironmentVariableA("C3X_SANDBOX_WITNESS_SCENARIO",scenario,sizeof(scenario));
    bool pressure=!std::strcmp(scenario,"pressure"),seam=!std::strcmp(scenario,"seam");
    bool zoom_sequence=!std::strcmp(scenario,"zoom");
    if(zoom_sequence)views={{0,0,1,"capture_zoom"}};
    if(!std::strcmp(scenario,"resident") || !std::strcmp(scenario,"resident128") || pressure){
        views.push_back({0,0,1,"return_origin"});views.push_back({288,144,1,"resident_strip"});
        views.push_back({span+192,96,1,"resident_wrap"});views.push_back({-640,-256,1,"resident_jump"});
        if(pressure){views.push_back({-1664,0,1,"far_east"});views.push_back({1664,0,1,"far_west"});
            views.push_back({0,0,1,"pressure_return"});}
        if(!std::strcmp(scenario,"resident")){views.push_back({0,0,1,"native64",64});views.push_back({0,0,1,"native128",128});}
    }
    if(!std::strcmp(scenario,"reverse"))views={{span+192,96,1,"wrap"},{192,96,1,"equivalent"},
        {-span+192,96,1,"negative_wrap"},{0,0,1,"return_origin"}};
    if(!std::strcmp(scenario,"mutations"))views={{0,0,1,"origin"},{0,0,1,"edit",0,1},
        {0,0,1,"visibility",0,2},{0,0,1,"viewer",0,3},{0,0,1,"world",0,4}};
    std::vector<std::string> seam_names;
    if(seam){views.clear();seam_names.reserve(25);for(int i=0;i<25;++i){
        seam_names.push_back("seam_"+std::to_string(i));
        views.push_back({1424+(i<=12?i:24-i)*16,96,1,seam_names.back().c_str()});}}
    c3x_renderer_camera_identity_v1 identity={1,2,3,4};
    std::vector<unsigned> world;
    if(original.world_topology && original.world_topology_count)world.assign(original.world_topology,original.world_topology+original.world_topology_count);
    auto world_revision=original.world_topology_revision;
    char cancellation[8]={};GetEnvironmentVariableA("C3X_SANDBOX_WITNESS_CANCEL",cancellation,sizeof(cancellation));
    char captures[4*MAX_PATH]={},value[32]={};
    if(!GetEnvironmentVariableA("C3X_SANDBOX_PASS_COUNTS",value,sizeof(value)) || value[0]!='1')counts=nullptr;
    GetEnvironmentVariableA("C3X_SANDBOX_WITNESS_CAPTURE_DIR",captures,sizeof(captures));
    int frames=GetEnvironmentVariableA("C3X_SANDBOX_WITNESS_FRAMES",value,sizeof(value))?
        std::clamp(std::atoi(value),1,600):30;
    // A single named endpoint allows independent cold/warm image comparison.
    char endpoint[32]={};
    if(GetEnvironmentVariableA("C3X_SANDBOX_WITNESS_ENDPOINT",endpoint,sizeof(endpoint))) {
        auto found=std::find_if(views.begin(),views.end(),[&](auto const& v){return std::strcmp(v.name,endpoint)==0;});
        if(found==views.end())return 21;auto selected=*found;views={selected};
    }
    LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
    auto ms=[&](LONGLONG ticks){return 1000.*double(ticks)/frequency.QuadPart;};
    std::set<std::pair<int,int>> previous;
    for(std::size_t index=0;index<views.size();++index){
        auto view=views[index];
        if(view.change==1){
            auto changed=std::min_element(witness.canonical.begin(),witness.canonical.end(),[&](auto const& a,auto const& b){
                auto score=[&](auto const& t){return t.terrain_type<5 && t.real_terrain_type<5 && t.city_id<0?
                    std::abs(t.anchor_x-original.target_width/2)+std::abs(t.anchor_y-original.target_height/2):INT_MAX;};
                return score(a)<score(b);
            });
            if(changed==witness.canonical.end() || changed->terrain_type>=5)return 32;
            changed->terrain_type=changed->real_terrain_type=changed->terrain_type==2?3:2;
            if(!world.empty())world[unsigned(changed->tile_y*original.world_width_tiles+changed->tile_x)/2]=
                unsigned(changed->terrain_type)|(unsigned(changed->real_terrain_type)<<8)|(changed->river_code<<16);
            ++identity.scene_epoch;++world_revision;
            std::printf("CAMERA_MUTATION tile=%d,%d terrain=%d scene_epoch=%lld world_revision=%lld\n",
                changed->tile_x,changed->tile_y,changed->terrain_type,identity.scene_epoch,world_revision);
        }
        if(view.change==2){++identity.visibility_epoch;
            for(auto& t:witness.canonical)if(t.anchor_x>original.target_width/3 && t.anchor_x<original.target_width*2/3)
                t.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;}
        if(view.change==3){++identity.viewer_epoch;for(auto& t:witness.canonical)t.tile_flags|=C3X_RENDERER_TILE_VISIBLE;}
        if(view.change==4){++identity.map_epoch;++identity.scene_epoch;++world_revision;}
        auto tiles=witness.capture(view);
        if(tiles.empty())return 22;
        auto frame=original;frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        if(view.native_width){frame.tile_width=view.native_width;frame.tile_height=view.native_width/2;}
        frame.world_topology=world.empty()?nullptr:world.data();frame.world_topology_revision=world_revision;
        frame.presentation_time_ticks=29500+(seam?int(index)*33:0);frame.presentation_frequency=1000;
        frame.dirty_flags=index && !view.change?0:C3X_RENDERER_DIRTY_ALL;
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,identity};
        c3x_renderer_i64 ticket=0;LARGE_INTEGER entered={},submitted={},prepared={},adopted={},first={};
        std::fflush(stdout);QueryPerformanceCounter(&entered);int result;
        c3x_renderer_i64 abandoned=0;
        if(cancellation[0]=='1' && index){auto superseded=view;superseded.x+=64;
            auto obsolete=witness.capture(superseded);auto prior=frame;prior.tiles=obsolete.data();prior.tile_count=unsigned(obsolete.size());
            auto old=request;old.frame=&prior;result=begin(&old,&abandoned);if(result!=C3X_RENDERER_RESULT_PENDING)return 33;}
        result=begin(&request,&ticket);QueryPerformanceCounter(&submitted);
        if(result!=C3X_RENDERER_RESULT_PENDING)return 23;
        c3x_renderer_gpu_camera_view_v1 shown={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(shown)};
        auto deadline=GetTickCount64()+120000;unsigned polls=0;
        do {result=ready(ticket,&shown);++polls;if(result==C3X_RENDERER_RESULT_PENDING)Sleep(1);}
        while(result==C3X_RENDERER_RESULT_PENDING && GetTickCount64()<deadline);
        QueryPerformanceCounter(&prepared);if(result!=C3X_RENDERER_RESULT_OK){
            // Error returns can leave the DLL's buffered file tail unwritten.
            // Use its existing explicit diagnostic flush before process exit.
            auto flush=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_trial_trace_flush"));
            if(flush)flush();
            std::printf("CAMERA_FAILED view=%s ticket=%lld result=%d wait_ms=%.6f polls=%u deadline_expired=%u\n",
                view.name,ticket,result,ms(prepared.QuadPart-submitted.QuadPart),polls,unsigned(GetTickCount64()>=deadline));
            std::fflush(stdout);return 24;
        }
        do {result=adopt(ticket,&shown);if(result==C3X_RENDERER_RESULT_PENDING)Sleep(1);}
        while(result==C3X_RENDERER_RESULT_PENDING && GetTickCount64()<deadline);
        QueryPerformanceCounter(&adopted);if(result!=C3X_RENDERER_RESULT_OK)return 25;
        if(abandoned){c3x_renderer_gpu_camera_view_v1 stale={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(stale)};
            auto cancelled=adopt(abandoned,&stale);if(cancelled==C3X_RENDERER_RESULT_OK)return 34;
            std::printf("CAMERA_CANCEL old_ticket=%lld selected_ticket=%lld stale_result=%d\n",abandoned,ticket,cancelled);}
        auto const& selected=shown.camera.frame;
        auto expected=frame;expected.tiles=selected.tiles;expected.world_topology=nullptr;expected.world_topology_count=0;
        if(std::memcmp(&selected,&expected,sizeof(expected)) || shown.camera.ticket!=ticket || std::memcmp(&shown.camera.identity,&request.identity,sizeof(request.identity)) ||
           selected.tile_count!=frame.tile_count || selected.target_width!=frame.target_width ||
           selected.tile_width!=frame.tile_width || selected.presentation_time_ticks!=frame.presentation_time_ticks ||
           !selected.tiles || std::memcmp(selected.tiles,tiles.data(),tiles.size()*sizeof(tiles[0])))return 26;
        std::set<std::pair<int,int>> occurrences;
        for(auto const& t:tiles)occurrences.emplace(t.tile_x,t.tile_y);
        unsigned added=0,removed=0,rendered=0,prefetched=0,topology=0;
        for(auto const& t:tiles){rendered+=(t.tile_flags&C3X_RENDERER_TILE_RENDER)!=0;
            prefetched+=(t.tile_flags&C3X_RENDERER_TILE_PREFETCH)!=0;
            topology+=(t.tile_flags&C3X_RENDERER_TILE_TOPOLOGY_HALO)!=0 && !(t.tile_flags&C3X_RENDERER_TILE_PREFETCH);}
        for(auto t:occurrences)if(!previous.count(t))++added;
        for(auto t:previous)if(!occurrences.count(t))++removed;
        previous=occurrences;
        if(draw(&frame,nullptr,view.x,view.y,24,56,1,47,1,view.zoom) ||
           present(window,&frame,24,56,1,47,1,view.x,view.y))return 27;
        QueryPerformanceCounter(&first);
        std::printf("CAMERA_ADOPTION view=%s ticket=%lld identity=%lld,%lld,%lld,%lld native_width=%d tiles=%u added=%u removed=%u render=%u prefetch=%u topology_only=%u anchor_origin=%d,%d zoom=%.3f submission_ms=%.6f preparation_wait_ms=%.6f adoption_ms=%.6f first_correct_view_ms=%.6f geometry_ms=%.6f draw_ms=%.6f upload_bytes=%u polls=%u exact_anchors=1\n",
            view.name,ticket,identity.map_epoch,identity.viewer_epoch,identity.visibility_epoch,identity.scene_epoch,frame.tile_width,frame.tile_count,added,removed,rendered,prefetched,topology,
            tiles[0].anchor_x-tiles[0].tile_x*frame.tile_width/2,
            tiles[0].anchor_y-tiles[0].tile_y*frame.tile_height/2,view.zoom,
            ms(submitted.QuadPart-entered.QuadPart),ms(prepared.QuadPart-submitted.QuadPart),
            ms(adopted.QuadPart-prepared.QuadPart),ms(first.QuadPart-entered.QuadPart),
            ms(shown.camera.output.geometry_ticks),ms(shown.camera.output.draw_ticks),shown.camera.output.geometry_upload_bytes,polls);
        std::fflush(stdout);
        char receipts[4*MAX_PATH]={};
        if(GetEnvironmentVariableA("C3X_SANDBOX_WORLD_CONTENT_DIR",receipts,sizeof(receipts))){
            char path[4*MAX_PATH]={};sprintf_s(path,"%s\\%s.csv",receipts,view.name);
            if(!content || content(path))return 31;
        }
        std::vector<std::array<double,3>> times;std::vector<SandboxPassCounts> work;
        for(int step=0;step<frames;++step){
            frame.presentation_time_ticks=29533+(seam?int(index)*33:0)+step*33;
            float zoom=zoom_sequence?sandbox_capture_zoom(step):view.zoom;
            LARGE_INTEGER start={},drawn={},done={};QueryPerformanceCounter(&start);
            result=draw(&frame,nullptr,view.x,view.y,24,56,1,47,1,zoom);QueryPerformanceCounter(&drawn);
            if(!result)result=present(window,&frame,24,56,1,47,1,view.x,view.y);QueryPerformanceCounter(&done);
            if(result)return 28;
            times.push_back({ms(done.QuadPart-start.QuadPart),ms(drawn.QuadPart-start.QuadPart),ms(done.QuadPart-drawn.QuadPart)});
            if(counts){work.emplace_back();counts(&work.back());}
            char checkpoints[8]={};
            if(captures[0] && zoom_sequence && GetEnvironmentVariableA("C3X_SANDBOX_ZOOM_CHECKPOINTS",checkpoints,sizeof(checkpoints)) && checkpoints[0]=='1' &&
               (step==0 || step==22 || step==45 || step==89 || step==134 || step==179)){
                char prefix[4*MAX_PATH]={};sprintf_s(prefix,"%s\\zoom_%d",captures,step);
                if(!capture || capture(prefix))return 30;
            }
        }
        for(std::size_t step=0;step<times.size();++step)
            std::printf("CAMERA_INPUT view=%s frame=%zu time_ms=%lld camera=%d,%d zoom=%.9f\n",
                view.name,step,c3x_renderer_i64(29533+(seam?int(index)*33:0)+int(step)*33),view.x,view.y,
                zoom_sequence?sandbox_capture_zoom(int(step)):view.zoom);
        for(std::size_t step=0;step<times.size();++step)
            std::printf("CAMERA_FRAME view=%s frame=%zu total_ms=%.6f draw_ms=%.6f present_ms=%.6f\n",view.name,step,times[step][0],times[step][1],times[step][2]);
        for(std::size_t step=0;step<work.size();++step)sandbox_report_pass_counts(work[step],step,view.name);
        std::fflush(stdout);
        if(captures[0]){
            if(!capture)return 29;
            char prefix[4*MAX_PATH]={};sprintf_s(prefix,"%s\\%s",captures,view.name);
            if(capture(prefix))return 30;
        }
    }
    std::puts("CAMERA_WITNESS pass scope=standalone_production_preparation_and_gpu_adoption");
    return 0;
}
