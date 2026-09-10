// Included by biq_preview.cpp after the production capture/render helpers exist.
if(ok && retained_replay) {
    using c3x_renderer::BusyReplayRequest;
    using c3x_renderer::BusySessionPlan;
    BusySessionPlan replay_plan{center_x,center_y,map_width,map_height};
    auto replay_schedule=c3x_renderer::fixed_busy_replay(replay_plan,unsigned(replay_samples));
    struct ReplayActor {int x,y,id;};
    std::vector<ReplayActor> replay_actors;
    for(long long location:{0ll,28000000ll,40000000ll}) {
        auto destination=replay_plan.at(location);
        std::vector<CsvTile const*> eligible;
        for(auto const& tile:source_tiles)if(!tile.topology_halo && tile.real>=0 && tile.real<=4 &&
            tile.x>=0 && tile.x<map_width && tile.y>=0 && tile.y<map_height &&
            !(tile.x%12==3 && tile.y%12==3))eligible.push_back(&tile);
        auto distance=[&](CsvTile const* tile){int dx=std::abs(tile->x-destination.x);dx=(std::min)(dx,map_width-dx);
            return dx+std::abs(tile->y-destination.y);};
        std::sort(eligible.begin(),eligible.end(),[&](auto a,auto b){
            int da=distance(a),db=distance(b);return da==db?preview_seed(a->x,a->y)<preview_seed(b->x,b->y):da<db;});
        int placed=0;
        for(auto tile:eligible) {
            if(std::any_of(replay_actors.begin(),replay_actors.end(),[&](auto const& actor){return actor.x==tile->x && actor.y==tile->y;}))continue;
            replay_actors.push_back({tile->x,tile->y,10000+int(replay_actors.size())});
            if(++placed==idle_unit_count)break;
        }
        if(placed!=idle_unit_count)ok=false;
    }
    auto replay_unit_draw=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_expanded"));
    auto replay_trim=reinterpret_cast<c3x_renderer_benchmark_trim_to_prepared_v1_fn>(
        GetProcAddress(module,"c3x_renderer_benchmark_trim_to_prepared_v1"));
    HDC replay_canvas=CreateCompatibleDC(nullptr);HBITMAP replay_bitmap=nullptr;HGDIOBJ replay_old_bitmap=nullptr;
    void* replay_composed_pixels=nullptr;
    BITMAPINFO replay_info={};replay_info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);replay_info.bmiHeader.biWidth=target_width;
    replay_info.bmiHeader.biHeight=-target_height;replay_info.bmiHeader.biPlanes=1;replay_info.bmiHeader.biBitCount=32;
    if(replay_canvas)replay_bitmap=CreateDIBSection(replay_canvas,&replay_info,DIB_RGB_COLORS,&replay_composed_pixels,nullptr,0);
    if(replay_bitmap)replay_old_bitmap=SelectObject(replay_canvas,replay_bitmap);
    if(!replay_canvas || !replay_bitmap || !replay_composed_pixels || !replay_unit_draw || !idle_unit_count ||
       (oracle_preparation && !replay_trim))ok=false;

    auto replay_units=[&](c3x_renderer_frame_v1 const& captured,std::vector<c3x_renderer_tile_v1> const& captured_tiles){
        std::vector<c3x_renderer_unit_v1> requests;
        char const* names[]={"Archer","Swordsman","Infantry","Warrior","Scout","Settler","Worker"};
        for(auto const& actor:replay_actors)for(auto const& tile:captured_tiles) {
            if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER) || tile.city_id>=0 ||
               ((tile.tile_x%map_width)+map_width)%map_width!=actor.x || tile.tile_y!=actor.y)continue;
            int index=actor.id-10000;
            c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);unit.unit_id=actor.id;
            unit.action=1;unit.direction=1+index%8;unit.frame_count=16;
            unit.presentation_frequency=captured.presentation_frequency;unit.presentation_time_ticks=captured.presentation_time_ticks;
            unit.sprite_width=unit.sprite_height=191;unit.projection_scale_milli=captured.tile_width*1000/128;
            unit.body_x=tile.anchor_x+captured.tile_width/2-191*unit.projection_scale_milli/2000;
            unit.body_y=tile.anchor_y+captured.tile_height/2-191*unit.projection_scale_milli/2000;
            int timeline=int(((captured.presentation_time_ticks-captured.presentation_frequency)*15/captured.presentation_frequency+index*7)%80);
            int phase=timeline%16,travel=0;
            if(timeline<16){unit.action=2;unit.direction=3;travel=phase+1;}
            else if(timeline<32){unit.action=index%7<5?3:8;unit.direction=3;travel=16;}
            else if(timeline<48){unit.action=2;unit.direction=7;travel=15-phase;}
            else if(timeline<64){unit.action=7;unit.direction=7;}
            unit.action_cursor=phase;unit.body_x+=travel*captured.tile_width/32;unit.body_y+=travel*captured.tile_height/32;
            unit.hour=captured.hour;unit.season=captured.season;unit.display_color_rgb=0x205bdd;
            sprintf_s(unit.unit_key,"PRTO_%s",names[index%7]);
            int projected_size=unit.sprite_width*unit.projection_scale_milli/1000;
            if(unit.body_x+projected_size<=0 || unit.body_y+projected_size<=0 ||
               unit.body_x>=target_width || unit.body_y>=target_height)continue;
            requests.push_back(unit);
        }
        std::sort(requests.begin(),requests.end(),[](auto const& a,auto const& b){
            return a.body_y==b.body_y?(a.body_x==b.body_x?a.unit_id<b.unit_id:a.body_x<b.body_x):a.body_y<b.body_y;});
        return requests;
    };
    auto replay_check_objects=[&](c3x_renderer_frame_v1 const& captured,c3x_renderer_output_v1 const& rendered){
        if(!preview_ownership(captured,rendered))return false;
        for(unsigned i=0;i<captured.tile_count;++i) {
            auto const& tile=captured.tiles[i];if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            unsigned required=0;
            if(tile.city_id>=0)required|=C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED;
            if(tile.road_mask)required|=C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED;
            if(tile.railroad_mask)required|=C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED;
            if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)required|=C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED;
            if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)required|=C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED;
            if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)required|=C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED;
            if(tile.resource_id>=0)required|=C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
            if((rendered.replacement_tile_flags[i]&required)!=required)return false;
        }
        return true;
    };
    auto replay_draw_units=[&](std::vector<c3x_renderer_unit_v1> const& requests){
        for(auto const& unit:requests){int bounds[4]={};
            if(replay_unit_draw(&unit,replay_canvas,replay_canvas,bounds)!=C3X_RENDERER_RESULT_OK ||
               bounds[2]<=bounds[0] || bounds[3]<=bounds[1] || bounds[2]<=0 || bounds[3]<=0 ||
               bounds[0]>=target_width || bounds[1]>=target_height)return false;
        }
        return GdiFlush()!=0;
    };
    auto replay_semantic_hash=[&](c3x_renderer_frame_v1 const& captured,
                                  std::vector<c3x_renderer_tile_v1> const& captured_tiles,
                                  std::vector<c3x_renderer_unit_v1> const& requests){
        unsigned long long value=14695981039346656037ull;
        auto mix=[&](unsigned long long part){value=(value^part)*1099511628211ull;};
        mix(captured.tile_width);mix(captured.tile_height);mix(captured.presentation_frequency);
        mix(captured.presentation_time_ticks);mix(captured.hour);mix(captured.season);mix(captured_tiles.size());
        for(auto const& tile:captured_tiles)for(auto part:{tile.tile_x,tile.tile_y,tile.anchor_x,tile.anchor_y,
            int(tile.tile_flags),tile.city_id,int(tile.road_mask),int(tile.railroad_mask),
            int(tile.improvement_flags),tile.resource_id})mix(static_cast<unsigned>(part));
        mix(requests.size());
        for(auto const& unit:requests) {
            for(auto part:{unit.unit_id,unit.action,unit.direction,unit.action_cursor,unit.frame_count,
                unit.body_x,unit.body_y,unit.sprite_width,unit.sprite_height,unit.projection_scale_milli,
                unit.hour,unit.season,int(unit.display_color_rgb)})mix(static_cast<unsigned>(part));
            for(auto c:unit.unit_key){mix(static_cast<unsigned char>(c));if(!c)break;}
        }
        return value;
    };
    struct ReplayStored {c3x_renderer_frame_v1 frame;std::vector<c3x_renderer_tile_v1> tiles;
        std::vector<c3x_renderer_unit_v1> units;BusyReplayRequest request;unsigned long long semantic_hash;};
    std::vector<ReplayStored> replay_requests;replay_requests.reserve(replay_schedule.size());
    std::set<unsigned long long> replay_unique_views,replay_unique_poses;
    for(auto const& request:replay_schedule) {
        center_x=request.view.x;center_y=request.view.y;tile_width=request.view.width;tile_height=tile_width/2;
        auto captured_tiles=capture_view();auto captured=frame;
        captured.tile_width=tile_width;captured.tile_height=tile_height;
        captured.presentation_time_ticks=captured.presentation_frequency+
            request.logical_us*captured.presentation_frequency/1000000;
        captured.tiles=captured_tiles.data();captured.tile_count=unsigned(captured_tiles.size());
        auto units=replay_units(captured,captured_tiles);
        unsigned long long view_hash=14695981039346656037ull;
        for(auto value:{request.view.x,request.view.y,request.view.width})view_hash=(view_hash^static_cast<unsigned>(value))*1099511628211ull;
        replay_unique_views.insert(view_hash);
        for(auto const& unit:units) {
            unsigned long long pose=14695981039346656037ull;
            for(auto value:{unit.action,unit.direction,unit.action_cursor,unit.frame_count,unit.projection_scale_milli,
                unit.hour,unit.season,int(unit.display_color_rgb)})pose=(pose^static_cast<unsigned>(value))*1099511628211ull;
            for(auto c:unit.unit_key){pose=(pose^static_cast<unsigned char>(c))*1099511628211ull;if(!c)break;}
            replay_unique_poses.insert(pose);
        }
        auto semantic=replay_semantic_hash(captured,captured_tiles,units);
        replay_requests.push_back({captured,std::move(captured_tiles),std::move(units),request,semantic});
    }
    for(auto& request:replay_requests)request.frame.tiles=request.tiles.data();

    reset();
    ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK;
    LARGE_INTEGER replay_frequency={},prepare_begin={},prepare_end={};QueryPerformanceFrequency(&replay_frequency);
    c3x_renderer_benchmark_oracle_trim_v1 trim={C3X_RENDERER_BENCHMARK_ORACLE_VERSION,sizeof(trim)};
    unsigned long long prepare_units=0,prepare_builds=0,prepare_evictions=0,prepare_uploads=0;
    unsigned long long cleared_viewport=0,cleared_regions=0,cleared_blocks=0,cleared_backdrops=0,cleared_publication=0;
    unsigned long long capacity_geometry_evictions=0,capacity_pose_evictions=0;
    std::size_t prepare_examined=0;bool prepare_capacity_limited=false;
    QueryPerformanceCounter(&prepare_begin);
    std::printf("REPLAY_PREPARE_BEGIN mode=%s requests=%zu samples_per_phase=%d\n",
        oracle_preparation?"oracle":"baseline",oracle_preparation?replay_requests.size():0,replay_samples);
    if(oracle_preparation)for(auto const& stored:replay_requests) {
        frame=stored.frame;tiles=stored.tiles;frame.tiles=tiles.data();
        ok=ok && replay_semantic_hash(frame,tiles,stored.units)==stored.semantic_hash;
        int code=render_checked(&frame,&output);
        ok=ok && code==C3X_RENDERER_RESULT_OK && replay_check_objects(frame,output);
        if(ok)std::memcpy(replay_composed_pixels,output.bgra_pixels,std::size_t(output.stride_bytes)*output.height);
        if(ok)ok=replay_draw_units(stored.units);
        prepare_units+=stored.units.size();
        prepare_builds+=output.geometry_tiles_built;prepare_evictions+=output.geometry_tiles_evicted;
        prepare_uploads+=output.geometry_upload_bytes;
        ++prepare_examined;
        if(ok && prepare_examined%25==0) {
            c3x_renderer_benchmark_oracle_trim_v1 interim={C3X_RENDERER_BENCHMARK_ORACLE_VERSION,sizeof(interim)};
            ok=replay_trim(&interim)==C3X_RENDERER_RESULT_OK;
            cleared_viewport+=interim.cleared_viewport_bytes;cleared_regions+=interim.cleared_region_bytes;
            cleared_blocks+=interim.cleared_pixel_block_bytes;cleared_backdrops+=interim.cleared_backdrop_bytes;
            cleared_publication+=interim.cleared_publication_bytes;trim=interim;
            capacity_geometry_evictions+=interim.capacity_geometry_evictions;
            capacity_pose_evictions+=interim.capacity_pose_evictions;
            auto memory=camera_memory_values();
            // Keep headroom above the hard 512 MiB acceptance floor so the
            // next render cannot cross it before another checkpoint.
            if(memory.second<640ull*1024*1024){prepare_capacity_limited=true;break;}
        }
        if(!ok)break;
    }
    if(ok && oracle_preparation) {
        c3x_renderer_benchmark_oracle_trim_v1 final_trim={C3X_RENDERER_BENCHMARK_ORACLE_VERSION,sizeof(final_trim)};
        ok=replay_trim(&final_trim)==C3X_RENDERER_RESULT_OK;
        cleared_viewport+=final_trim.cleared_viewport_bytes;cleared_regions+=final_trim.cleared_region_bytes;
        cleared_blocks+=final_trim.cleared_pixel_block_bytes;cleared_backdrops+=final_trim.cleared_backdrop_bytes;
        cleared_publication+=final_trim.cleared_publication_bytes;trim=final_trim;
        capacity_geometry_evictions+=final_trim.capacity_geometry_evictions;
        capacity_pose_evictions+=final_trim.capacity_pose_evictions;
    }
    QueryPerformanceCounter(&prepare_end);
    auto prepared_memory=camera_memory_values();
    bool prepared_memory_safe=!oracle_preparation || prepared_memory.second>=512ull*1024*1024;
    ok=ok && prepared_memory_safe;
    std::printf("REPLAY_PREPARE_END status=%s mode=%s requests=%zu requested=%zu capacity_limited=%d memory_safe=%d unique_views=%zu unique_poses=%zu unit_requests=%llu ms=%.3f geometry_admissions=%llu geometry_evictions=%llu capacity_geometry_evictions=%llu capacity_pose_evictions=%llu upload_bytes=%llu cleared_viewport=%llu cleared_regions=%llu cleared_blocks=%llu cleared_backdrops=%llu cleared_publication=%llu retained_geometry=%llu retained_natural=%llu retained_ground=%llu retained_waves=%llu retained_pose=%llu retained_payload=%llu retained_shadow=%llu retained_other=%llu geometry_entries=%u pose_entries=%u wave_entries=%u\n",
        ok?"pass":"FAIL",oracle_preparation?"oracle":"baseline",oracle_preparation?prepare_examined:0,
        replay_requests.size(),int(prepare_capacity_limited),int(prepared_memory_safe),
        replay_unique_views.size(),replay_unique_poses.size(),prepare_units,
        double(prepare_end.QuadPart-prepare_begin.QuadPart)*1000/replay_frequency.QuadPart,
        prepare_builds,prepare_evictions,capacity_geometry_evictions,capacity_pose_evictions,prepare_uploads,
        cleared_viewport,cleared_regions,cleared_blocks,cleared_backdrops,cleared_publication,
        trim.retained_geometry_bytes,trim.retained_natural_bytes,trim.retained_ground_bytes,
        trim.retained_wave_bytes,trim.retained_unit_pose_bytes,trim.retained_unit_payload_bytes,trim.retained_shadow_bytes,
        trim.retained_other_bytes,
        trim.retained_geometry_entries,trim.retained_unit_pose_entries,trim.retained_wave_entries);
    camera_memory();std::fflush(stdout);

    struct ReplaySnapshot {ReplayStored stored;std::vector<unsigned char> pixels;std::string label;};
    std::vector<ReplaySnapshot> replay_snapshots;unsigned saved_phases=0,saved_events=0;
    unsigned long long total_builds=0,total_evictions=0,total_uploads=0,total_units=0;unsigned recoveries=0;
    std::printf("REPLAY_BEGIN mode=%s clock=logical requests=%zu samples_per_phase=%d units_per_zone=%d dense=1 native_presented=0 final_map_cache=cleared\n",
        oracle_preparation?"oracle":"baseline",replay_requests.size(),replay_samples,idle_unit_count);
    for(unsigned index=0;index<replay_requests.size() && ok;++index) {
        auto const& stored=replay_requests[index];
        LARGE_INTEGER begin={},captured={},map_done={},copy_done={},finished={};QueryPerformanceCounter(&begin);
        frame=stored.frame;tiles=stored.tiles;frame.tiles=tiles.data();QueryPerformanceCounter(&captured);
        ok=ok && replay_semantic_hash(frame,tiles,stored.units)==stored.semantic_hash;
        int code=render_checked(&frame,&output);QueryPerformanceCounter(&map_done);
        ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 && replay_check_objects(frame,output);
        if(ok)std::memcpy(replay_composed_pixels,output.bgra_pixels,std::size_t(output.stride_bytes)*output.height);
        QueryPerformanceCounter(&copy_done);if(ok)ok=replay_draw_units(stored.units);QueryPerformanceCounter(&finished);
        std::size_t bytes=std::size_t(output.stride_bytes)*output.height;
        unsigned long long hash=14695981039346656037ull;
        if(ok)for(std::size_t i=0;i<bytes;++i)hash=(hash^static_cast<unsigned char const*>(replay_composed_pixels)[i])*1099511628211ull;
        double scale=1000.0/replay_frequency.QuadPart;
        std::printf("REPLAY_FRAME step=%u phase=%d label=%s event=%d logical_us=%lld x=%d y=%d tile_width=%d units=%zu request_hash=%llu result=%d ms=%.3f capture_ms=%.3f map_ms=%.3f copy_ms=%.3f units_ms=%.3f geometry_ms=%.3f draw_ms=%.3f readback_ms=%.3f built=%u reused=%u evicted=%u upload_bytes=%u raster_cached_pixels=%u fallback=%u recoveries=%u fnv64=%llu\n",
            index,stored.request.view.phase,stored.request.view.name,stored.request.event,stored.request.logical_us,
            stored.request.view.x,stored.request.view.y,stored.request.view.width,stored.units.size(),stored.semantic_hash,code,
            (finished.QuadPart-begin.QuadPart)*scale,(captured.QuadPart-begin.QuadPart)*scale,
            (map_done.QuadPart-captured.QuadPart)*scale,(copy_done.QuadPart-map_done.QuadPart)*scale,
            (finished.QuadPart-copy_done.QuadPart)*scale,double(output.geometry_ticks)*scale,double(output.draw_ticks)*scale,
            double(output.readback_ticks)*scale,output.geometry_tiles_built,output.geometry_tiles_reused,output.geometry_tiles_evicted,
            output.geometry_upload_bytes,output.raster_cached_pixels,output.fallback_tile_count,output.device_recoveries,hash);
        total_builds+=output.geometry_tiles_built;total_evictions+=output.geometry_tiles_evicted;
        total_uploads+=output.geometry_upload_bytes;total_units+=stored.units.size();
        recoveries=(std::max)(recoveries,output.device_recoveries);
        bool phase_sample=!(saved_phases&(1u<<stored.request.view.phase));
        bool event_sample=stored.request.event>=0 && !(saved_events&(1u<<stored.request.event));
        if(ok && (phase_sample || event_sample)) {
            std::string label=event_sample?"event-"+std::to_string(stored.request.event):"phase-"+std::to_string(stored.request.view.phase);
            auto data=static_cast<unsigned char const*>(replay_composed_pixels);
            ReplayStored snapshot=stored;snapshot.frame.tiles=nullptr;
            replay_snapshots.push_back({std::move(snapshot),std::vector<unsigned char>(data,data+bytes),label});
            if(phase_sample)saved_phases|=1u<<stored.request.view.phase;
            if(event_sample)saved_events|=1u<<stored.request.event;
        }
        if(index%25==0)camera_memory();std::fflush(stdout);
    }
    bool structural_complete=oracle_preparation && total_builds==0 && total_uploads==0;
    std::printf("REPLAY_TIMED_END status=%s mode=%s frames=%zu phase_mask=%u event_mask=%u zoom_mask=7 built=%llu evicted=%llu upload_bytes=%llu unit_requests=%llu recoveries=%u structural_complete=%d snapshots=%zu\n",
        ok?"pass":"FAIL",oracle_preparation?"oracle":"baseline",replay_requests.size(),saved_phases,saved_events,
        total_builds,total_evictions,total_uploads,total_units,recoveries,int(structural_complete),replay_snapshots.size());
    unsigned verified=0;
    for(auto const& saved:replay_snapshots) {
        if(!ok)break;
        auto picture=output;picture.width=target_width;picture.height=target_height;picture.stride_bytes=target_width*4;
        picture.bgra_pixels=saved.pixels.data();picture.clip_left=picture.clip_top=0;
        picture.clip_right=target_width;picture.clip_bottom=target_height;
        std::string path=std::string(argv[5])+".replay-"+saved.label+"-"+std::to_string(saved.stored.request.view.width)+".bmp";
        ok=write_bmp(path.c_str(),picture);reset();
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK;
        frame=saved.stored.frame;tiles=saved.stored.tiles;frame.tiles=tiles.data();
        ok=ok && replay_semantic_hash(frame,tiles,saved.stored.units)==saved.stored.semantic_hash;
        ok=ok && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && replay_check_objects(frame,output);
        if(ok)std::memcpy(replay_composed_pixels,output.bgra_pixels,std::size_t(output.stride_bytes)*output.height);
        ok=ok && replay_draw_units(saved.stored.units) &&
            std::memcmp(replay_composed_pixels,saved.pixels.data(),saved.pixels.size())==0;
        std::printf("REPLAY_PARITY label=%s phase=%d event=%d tile_width=%d status=%s units=%zu bytes=%zu\n",
            saved.label.c_str(),saved.stored.request.view.phase,saved.stored.request.event,saved.stored.request.view.width,
            ok?"pass":"FAIL",saved.stored.units.size(),saved.pixels.size());
        if(ok)++verified;std::fflush(stdout);
    }
    if(replay_canvas && replay_old_bitmap)SelectObject(replay_canvas,replay_old_bitmap);
    if(replay_bitmap)DeleteObject(replay_bitmap);if(replay_canvas)DeleteDC(replay_canvas);
    std::printf("REPLAY_END status=%s mode=%s frames=%zu snapshots=%zu verified=%u coverage_complete=%d\n",
        ok?"pass":"FAIL",oracle_preparation?"oracle":"baseline",replay_requests.size(),replay_snapshots.size(),verified,
        int(saved_phases==255 && saved_events==127));
}
