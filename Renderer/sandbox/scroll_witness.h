#pragma once
#include "capture_model.h"
#include "pass_counts.h"
#include "static_raster_state.h"

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
    using StaticMetrics=c3x_renderer::render_core::StaticRasterMetrics;
    auto static_metrics=reinterpret_cast<int(*)(unsigned,StaticMetrics*)>(GetProcAddress(module,"c3x_sandbox_static_state_metrics"));
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
    bool phase=!std::strcmp(value,"phase"),states=!std::strcmp(value,"static");
    char const* trace=regression?"regression":motion?"motion":phase?"phase":states?"static":"pilot";
    // Slow edge coordinate7 gives8/4; both axes have integer projected phase
    // at settled1.25x. Coordinate15 remains the ordinary4/2 practical control.
    int phase_x=2*(slow_x/8),phase_y=2*(slow_y/8);
    if(phase){
        ex=phase_x;ey=phase_y;
        views={{0,0,zoom,"origin"},{ex,ey,zoom,"edge7_positive"},
            {2*ex,2*ey,zoom,"edge7_continue"},{-ex,-ey,zoom,"edge7_negative"},
            {128,64,zoom,"new_strip"},{0,0,zoom,"resident_return"}};
    }
    if(states){
        float replacement=zoom==1.5f?1.75f:1.5f;
        views={{0,0,zoom,"origin"},{0,0,zoom,"stationary_repeat"},
            {ex,ey,zoom,"edge15_positive"},{0,0,zoom,"edge15_return"},
            {phase_x,phase_y,zoom,"edge7_positive"},{2*phase_x,2*phase_y,zoom,"edge7_continue"},
            {0,0,zoom,"edge7_return"},{128,64,zoom,"new_strip"},{0,0,zoom,"resident_return"},
            {0,0,replacement,"zoom_replace"},{0,0,replacement,"zoom_repeat"},{0,0,zoom,"zoom_return"},
            {0,0,zoom,"world_content_change",0,1},{0,0,zoom,"world_content_repeat"},
            {0,0,zoom,"lighting_change",0,5},{0,0,zoom,"lighting_repeat"}};
        // Cross the nearest real4096-pixel world-row depth basis boundary.
        // This changes source anchors through capture, without a diagnostic setter.
        auto const& anchor=witness.canonical.front();
        auto center=std::int64_t(original.target_height)/2-
            (std::int64_t(anchor.anchor_y)-std::int64_t(anchor.tile_y)*32);
        auto rounded=center+2048,remainder=rounded%4096;if(remainder<0)remainder+=4096;
        auto origin=rounded-remainder;
        int higher=int(center-(origin+2048)-32),lower=int(center-(origin-2048)+32);
        int candidates[2]={higher,lower};if(std::abs(lower)<std::abs(higher))std::swap(candidates[0],candidates[1]);
        bool found=false;for(int shift:candidates){auto probe=witness.capture({0,shift,zoom,"depth_origin_cross"});
            unsigned visible=0;for(auto const& tile:probe)visible+=unsigned((tile.tile_flags&C3X_RENDERER_TILE_RENDER)!=0);
            if(visible<32)continue;views.push_back({0,shift,zoom,"depth_origin_cross"});
            views.push_back({0,0,zoom,"depth_origin_return"});found=true;break;}
        if(!found){std::printf("SCROLL_FAILED stage=depth_fixture reason=no_visible_boundary_crossing\n");return 78;}
    }
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
    // Preallocate the common records before starting the sequence. Quiet runs
    // collect identical aggregate counters/timestamps in both arms, then format
    // them only after the last Present; optional slot/pass work is diagnostic.
    struct FrameRecord {
        SandboxCameraWitness::View view{};
        int render_camera_x=0,render_camera_y=0,hour=0;
        c3x_renderer_i64 ticks=0,scene_epoch=0,revision=0,ticket=0,geometry_ticks=0,worker_draw_ticks=0;
        unsigned tile_count=0,added=0,removed=0,rendered=0,prefetched=0,cities=0,unit_records=0,upload_bytes=0,polls=0;
        LARGE_INTEGER entered={},captured={},submitted={},prepared={},adopted={},display_begin={},drawn={},done={},previous_done={};
        std::array<std::array<unsigned,6>,3> cache{};
        std::array<std::array<unsigned,14>,3> reasons{};
    };
    std::vector<FrameRecord> records(views.size());
    LARGE_INTEGER frequency={},start={},previous_done={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&start);
    auto ms=[&](LONGLONG ticks){return 1000.*double(ticks)/double(frequency.QuadPart);};
    auto snapshot=[&](unsigned* m,unsigned* r){metrics(&m[0],&m[1],&m[2],&m[3],&m[4],&m[5]);reasons(r);};
    auto report=[&](char const* stage,std::size_t index,unsigned const* a,unsigned const* b,unsigned const* ra,unsigned const* rb){
        std::printf("SCROLL_CACHE frame=%zu stage=%s depth_copies=%u scrolls=%u full_draws=%u reflection_reuses=%u reflection_draws=%u pose_builds=%u fractional=%u guard=%u signature=%u environment=%u lights=%u shadow=%u depth_origin=%u anchors=%u strip_fills=%u explicit_reset=%u projection=%u\n",
            index,stage,b[0]-a[0],b[1]-a[1],b[2]-a[2],b[3]-a[3],b[4]-a[4],b[5]-a[5],
            rb[0]-ra[0],rb[1]-ra[1],rb[2]-ra[2],rb[3]-ra[3],rb[4]-ra[4],rb[5]-ra[5],rb[6]-ra[6],rb[7]-ra[7],rb[8]-ra[8],rb[9]-ra[9],rb[10]-ra[10]);
    };
    struct StaticSnapshot {std::array<StaticMetrics,2> values{};std::array<bool,2> available{};};
    auto static_snapshot=[&](){StaticSnapshot state;
        for(unsigned slot=0;slot<2;++slot)state.available[slot]=static_metrics&&
            static_metrics(slot,&state.values[slot])==1&&state.values[slot].version==1&&state.values[slot].slot==slot;
        return state;};
    auto report_static=[&](char const* stage,std::size_t index,StaticSnapshot const& before,StaticSnapshot const& after){
        char const* names[2]={"canonical","display"};
        char const* reason_names[]={"fractional","guard","signature","environment","lights","shadow","depth_origin","anchors","strip_execution","explicit_reset","projection_reason","layout","error","classification"};
        for(unsigned slot=0;slot<2;++slot){
            std::printf("SCROLL_STATIC frame=%zu stage=%s slot=%u state=%s available=%u",index,stage,slot,names[slot],unsigned(after.available[slot]));
            if(!after.available[slot]){std::printf("\n");continue;}
            auto const& a=before.values[slot];auto const& b=after.values[slot];
            bool reset=b.full_draws<a.full_draws||b.restores<a.restores||b.reuses<a.reuses||b.strip_fills<a.strip_fills;
            auto delta=[](unsigned old,unsigned now){return now>=old?now-old:now;};
            std::printf(" version=%u valid=%u projection=%.9f full_draws=%u restores=%u reuses=%u strip_fills=%u full_total=%u restores_total=%u reuses_total=%u strips_total=%u region_revision=%llu region_bytes=%llu total_region_bytes=%llu shared_viewport_bytes=%llu total_gpu_bytes=%llu sample_count=%u counters_reset=%u",
                b.version,b.valid,b.projection,delta(a.full_draws,b.full_draws),delta(a.restores,b.restores),delta(a.reuses,b.reuses),delta(a.strip_fills,b.strip_fills),
                b.full_draws,b.restores,b.reuses,b.strip_fills,static_cast<unsigned long long>(b.region_revision),static_cast<unsigned long long>(b.region_bytes),
                static_cast<unsigned long long>(b.total_region_bytes),static_cast<unsigned long long>(b.shared_viewport_bytes),static_cast<unsigned long long>(b.total_gpu_bytes),b.sample_count,unsigned(reset));
            for(unsigned reason=0;reason<b.reasons.size();++reason)std::printf(" %s=%u",reason_names[reason],delta(a.reasons[reason],b.reasons[reason]));
            std::printf("\n");
        }
    };
    auto report_work=[&](char const* stage,std::size_t index){
        if(!diagnostic||!counts)return;
        SandboxPassCounts work;counts(&work);
        std::printf("SCROLL_WORK_STAGE frame=%zu stage=%s last_fresh_call=1\n",index,stage);
        char view[64]={};sprintf_s(view,"scroll_%s",stage);sandbox_report_pass_counts(work,index,view);
        unsigned long long main_instances=0,reflection_instances=0,main_triangles=0,reflection_triangles=0;
        for(unsigned layer=0;layer<SandboxPassCounts::layers;++layer){
            main_instances+=work.counts[SandboxPassCounts::units][layer].submitted_instances;
            reflection_instances+=work.counts[SandboxPassCounts::reflected_units][layer].submitted_instances;
            main_triangles+=work.counts[SandboxPassCounts::units][layer].triangles;
            reflection_triangles+=work.counts[SandboxPassCounts::reflected_units][layer].triangles;}
        std::printf("SCROLL_ACTORS frame=%zu stage=%s available=1 main_submitted_instances=%llu reflection_submitted_instances=%llu main_triangles=%llu reflection_triangles=%llu synthetic_actors=0\n",
            index,stage,main_instances,reflection_instances,main_triangles,reflection_triangles);
    };
    auto report_frame=[&](std::size_t index,FrameRecord const& record){
        auto const& view=record.view;
        std::printf("SCROLL_FRAME frame=%zu view=%s camera=%d,%d render_camera=%d,%d zoom=%.9f ticks=%lld hour=%d scene_epoch=%lld topology_revision=%lld ticket=%lld tiles=%u added=%u removed=%u render=%u prefetch=%u cities=%u unit_records=%u synthetic_actors=0 capture_ms=%.6f submission_ms=%.6f preparation_ms=%.6f adoption_ms=%.6f draw_ms=%.6f present_ms=%.6f first_correct_ms=%.6f interval_ms=%.6f done_ms=%.6f geometry_ticks=%lld worker_draw_ticks=%lld upload_bytes=%u polls=%u exact_anchors=1",
            index,view.name,view.x,view.y,record.render_camera_x,record.render_camera_y,view.zoom,record.ticks,record.hour,record.scene_epoch,record.revision,record.ticket,
            record.tile_count,record.added,record.removed,record.rendered,record.prefetched,record.cities,record.unit_records,
            ms(record.captured.QuadPart-record.entered.QuadPart),ms(record.submitted.QuadPart-record.captured.QuadPart),ms(record.prepared.QuadPart-record.submitted.QuadPart),
            ms(record.adopted.QuadPart-record.prepared.QuadPart),ms(record.drawn.QuadPart-record.display_begin.QuadPart),ms(record.done.QuadPart-record.drawn.QuadPart),
            ms(record.done.QuadPart-record.entered.QuadPart),record.previous_done.QuadPart?ms(record.done.QuadPart-record.previous_done.QuadPart):0.,
            ms(record.done.QuadPart-start.QuadPart),record.geometry_ticks,record.worker_draw_ticks,record.upload_bytes,record.polls);
        char const* counter_names[]={"depth_copies","scrolls","full_draws","reflection_reuses","reflection_draws","pose_builds"};
        char const* reason_names[]={"fractional","guard","signature","environment","lights","shadow","depth_origin","anchors","strip_fills","explicit_reset","projection"};
        char const* stages[]={"prep","display"};
        for(unsigned stage=0;stage<2;++stage){
            for(unsigned counter=0;counter<6;++counter)std::printf(" %s_%s=%u",stages[stage],counter_names[counter],record.cache[stage+1][counter]-record.cache[stage][counter]);
            for(unsigned reason=0;reason<11;++reason)std::printf(" %s_%s=%u",stages[stage],reason_names[reason],record.reasons[stage+1][reason]-record.reasons[stage][reason]);}
        std::printf("\n");
    };
    StaticSnapshot static_before,static_prep;
    std::set<std::pair<int,int>> previous;
    int hour=original.hour;
    std::printf("SCROLL_CONFIG zoom=%.9f native_width=128 native_height=64 target=2240,1260 views=%zu trace=%s clock=%s force_full=%u quality=%u input_step=%d,%d phase_step=8,4 native_edge_timer_ms=66 motion_schedule_ms=17 quality_invalidates_states=%u actors=none reasons_overlap=1 diagnostic=%u frame_records_buffered=%u\n",
        zoom,views.size(),trace,scripted?"scripted":"wall",unsigned(force),unsigned(quality),ex,ey,unsigned(quality),unsigned(diagnostic),unsigned(!diagnostic));
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
        std::set<std::pair<int,int>> occurrences;unsigned added=0,removed=0,rendered=0,prefetched=0,cities=0,unit_records=0;
        auto const& anchor=witness.canonical.front();
        for(auto const& t:tiles){if(t.anchor_x-t.tile_x*64!=anchor.anchor_x-anchor.tile_x*64+view.x||t.anchor_y-t.tile_y*32!=anchor.anchor_y-anchor.tile_y*32+view.y)return 66;
            if(!occurrences.emplace(t.tile_x,t.tile_y).second)return 67;bool render=(t.tile_flags&C3X_RENDERER_TILE_RENDER)!=0;rendered+=unsigned(render);cities+=unsigned(render&&t.city_id>=0);unit_records+=unsigned(render&&t.unit_type_id>=0);prefetched+=unsigned((t.tile_flags&C3X_RENDERER_TILE_PREFETCH)!=0);}
        for(auto t:occurrences)added+=unsigned(!previous.count(t));for(auto t:previous)removed+=unsigned(!occurrences.count(t));previous=occurrences;
        auto& record=records[index];
        auto& before=record.cache[0];auto& after_prep=record.cache[1];auto& after_display=record.cache[2];
        auto& rb=record.reasons[0];auto& rp=record.reasons[1];auto& rd=record.reasons[2];
        snapshot(before.data(),rb.data());
        if(diagnostic){static_before=static_snapshot();report_static("seed",index,static_before,static_before);report_work("seed",index);}
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
        snapshot(after_prep.data(),rp.data());
        if(diagnostic){static_prep=static_snapshot();report("prep",index,before.data(),after_prep.data(),rb.data(),rp.data());
            report_static("prep",index,static_before,static_prep);report_work("prep",index);}
        QueryPerformanceCounter(&display_begin);if(force)invalidate();result=draw(&frame,nullptr,render_camera_x,render_camera_y,0,0,0,0,0,view.zoom);QueryPerformanceCounter(&drawn);
        if(!result)result=present(window,&frame,0,0,0,0,0,render_camera_x,render_camera_y);QueryPerformanceCounter(&done);if(result)return 73;
        snapshot(after_display.data(),rd.data());
        if(diagnostic){auto static_display=static_snapshot();report("display",index,after_prep.data(),after_display.data(),rp.data(),rd.data());
            report_static("display",index,static_prep,static_display);}
        record.view=view;record.render_camera_x=render_camera_x;record.render_camera_y=render_camera_y;record.hour=frame.hour;
        record.ticks=frame.presentation_time_ticks;record.scene_epoch=identity.scene_epoch;record.revision=revision;record.ticket=ticket;
        record.tile_count=frame.tile_count;record.added=added;record.removed=removed;record.rendered=rendered;record.prefetched=prefetched;record.cities=cities;record.unit_records=unit_records;
        record.entered=entered;record.captured=captured;record.submitted=submitted;record.prepared=prepared;record.adopted=adopted;record.display_begin=display_begin;record.drawn=drawn;record.done=done;record.previous_done=previous_done;
        record.geometry_ticks=shown.camera.output.geometry_ticks;record.worker_draw_ticks=shown.camera.output.draw_ticks;record.upload_bytes=shown.camera.output.geometry_upload_bytes;record.polls=polls;
        if(diagnostic){report_frame(index,record);report_work("display",index);}
        previous_done=done;
        if(quality){char prefix[4*MAX_PATH]={};sprintf_s(prefix,"%s\\frame_%03zu_retained",directory,index);if(capture(prefix))return 74;
            if(content){char path[4*MAX_PATH]={};sprintf_s(path,"%s\\frame_%03zu_owners.csv",directory,index);if(content(path))return 75;}
            invalidate();if(draw(&frame,nullptr,render_camera_x,render_camera_y,0,0,0,0,0,view.zoom))return 76;
            sprintf_s(prefix,"%s\\frame_%03zu_full",directory,index);if(capture(prefix))return 77;
            if(diagnostic)std::printf("SCROLL_ORACLE frame=%zu view=%s ticks=%lld independent_raster=1 timed=0\n",index,view.name,frame.presentation_time_ticks);}
        if(diagnostic)std::fflush(stdout);
    }
    if(!diagnostic)for(std::size_t index=0;index<records.size();++index){
        report_frame(index,records[index]);
        if(quality)std::printf("SCROLL_ORACLE frame=%zu view=%s ticks=%lld independent_raster=1 timed=0\n",index,records[index].view.name,records[index].ticks);
    }
    std::printf("SCROLL_WITNESS pass views=%zu trace=%s quality=%u scope=standalone_copied_native_preparation_adoption\n",views.size(),trace,unsigned(quality));std::fflush(stdout);return 0;
}
