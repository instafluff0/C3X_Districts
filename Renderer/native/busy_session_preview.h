// Included in the standalone preview after capture/render helpers are available.
if(ok && busy_session) {
    using c3x_renderer::BusySessionPlan;
    using c3x_renderer::BusySessionView;
    BusySessionPlan plan{center_x,center_y,map_width,map_height};
    struct Actor {int x,y,id;};
    std::vector<Actor> actors;
    for(long long location:{0ll,28000000ll,40000000ll}) {
        auto destination=plan.at(location);
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
            if(std::any_of(actors.begin(),actors.end(),[&](auto const& actor){return actor.x==tile->x && actor.y==tile->y;}))continue;
            actors.push_back({tile->x,tile->y,10000+int(actors.size())});
            if(++placed==idle_unit_count)break;
        }
        if(placed!=idle_unit_count)ok=false;
    }
    auto unit_draw=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_expanded"));
    HDC canvas=CreateCompatibleDC(nullptr);HBITMAP bitmap=nullptr;HGDIOBJ old_bitmap=nullptr;void* composed_pixels=nullptr;
    BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=target_width;
    info.bmiHeader.biHeight=-target_height;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
    if(canvas)bitmap=CreateDIBSection(canvas,&info,DIB_RGB_COLORS,&composed_pixels,nullptr,0);
    if(bitmap)old_bitmap=SelectObject(canvas,bitmap);
    if(!canvas || !bitmap || !composed_pixels || !unit_draw || !idle_unit_count)ok=false;
    auto check_objects=[&](){
        for(unsigned i=0;i<frame.tile_count;++i) {
            auto const& tile=frame.tiles[i];if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER))continue;
            unsigned required=0;
            if(tile.city_id>=0)required|=C3X_RENDERER_TILE_CUSTOM_CITY_REPLACED;
            if(tile.road_mask)required|=C3X_RENDERER_TILE_CUSTOM_ROAD_REPLACED;
            if(tile.railroad_mask)required|=C3X_RENDERER_TILE_CUSTOM_RAILROAD_REPLACED;
            if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE)required|=C3X_RENDERER_TILE_CUSTOM_MINE_REPLACED;
            if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_IRRIGATION)required|=C3X_RENDERER_TILE_CUSTOM_FARM_REPLACED;
            if(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_BARBARIAN_CAMP)required|=C3X_RENDERER_TILE_CUSTOM_CAMP_REPLACED;
            if(tile.resource_id>=0)required|=C3X_RENDERER_TILE_CUSTOM_RESOURCE_REPLACED;
            if(i>=output.replacement_tile_count || (output.replacement_tile_flags[i]&required)!=required)return false;
        }
        return true;
    };
    auto body_requests=[&](){
        std::vector<c3x_renderer_unit_v1> requests;
        char const* names[]={"Archer","Swordsman","Infantry","Warrior","Scout","Settler","Worker"};
        for(auto const& actor:actors)for(auto const& tile:tiles) {
            if(!(tile.tile_flags&C3X_RENDERER_TILE_RENDER) || tile.city_id>=0 ||
               ((tile.tile_x%map_width)+map_width)%map_width!=actor.x || tile.tile_y!=actor.y)continue;
            int index=actor.id-10000;
            c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);unit.unit_id=actor.id;
            unit.action=1;unit.direction=1+index%8;unit.frame_count=16;
            unit.presentation_frequency=frame.presentation_frequency;unit.presentation_time_ticks=frame.presentation_time_ticks;
            unit.sprite_width=unit.sprite_height=191;unit.projection_scale_milli=tile_width*1000/128;
            unit.body_x=tile.anchor_x+tile_width/2-191*unit.projection_scale_milli/2000;
            unit.body_y=tile.anchor_y+tile_height/2-191*unit.projection_scale_milli/2000;
            int timeline=int(((frame.presentation_time_ticks-frame.presentation_frequency)*15/frame.presentation_frequency+index*7)%80);
            int phase=timeline%16,travel=0;
            if(timeline<16){unit.action=2;unit.direction=3;travel=phase+1;}
            else if(timeline<32){unit.action=index%7<5?3:8;unit.direction=3;travel=16;}
            else if(timeline<48){unit.action=2;unit.direction=7;travel=15-phase;}
            else if(timeline<64){unit.action=7;unit.direction=7;}
            unit.action_cursor=phase;unit.body_x+=travel*tile_width/32;unit.body_y+=travel*tile_height/32;
            unit.hour=frame.hour;unit.season=frame.season;unit.display_color_rgb=0x205bdd;
            sprintf_s(unit.unit_key,"PRTO_%s",names[index%7]);requests.push_back(unit);
        }
        std::sort(requests.begin(),requests.end(),[](auto const& a,auto const& b){
            return a.body_y==b.body_y?(a.body_x==b.body_x?a.unit_id<b.unit_id:a.body_x<b.body_x):a.body_y<b.body_y;});
        return requests;
    };
    auto draw_bodies=[&](std::vector<c3x_renderer_unit_v1> const& requests){
        for(auto const& unit:requests){int bounds[4]={};if(unit_draw(&unit,canvas,canvas,bounds)!=C3X_RENDERER_RESULT_OK)return false;}
        return GdiFlush()!=0;
    };
    struct Snapshot {c3x_renderer_frame_v1 frame;std::vector<c3x_renderer_tile_v1> tiles;
        std::vector<c3x_renderer_unit_v1> units;std::vector<unsigned char> pixels;int phase,width;};
    std::vector<Snapshot> snapshots;std::size_t snapshot_bytes=0;
    constexpr std::size_t snapshot_limit=96u*1024u*1024u;
    unsigned snapshot_mask=0,phase_mask=0,zoom_mask=0,frames=0,late_cameras=0;
    long long previous_slot=-1,skipped_slots=0;
    double evidence_ms=0;
    LARGE_INTEGER frequency={},session_started={},now={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&session_started);
    auto microseconds=[&](LARGE_INTEGER value){return (value.QuadPart-session_started.QuadPart)*1000000/frequency.QuadPart;};
    BusySessionView current=plan.at(0);
    std::printf("SESSION_BEGIN duration_us=%lld input_slot_us=%lld clock=wall mode=synchronous_completed_render native_presented=0 unit_warmup=0 initial_render_ms=%.3f units_per_zone=%d world_units=%zu dense=%d snapshots_limit=%zu\n",
        BusySessionPlan::duration_us,BusySessionPlan::slot_us,initial_render_ms,idle_unit_count,actors.size(),int(dense_scene),snapshot_limit);
    while(ok) {
        QueryPerformanceCounter(&now);long long elapsed=microseconds(now);
        if(elapsed>=BusySessionPlan::duration_us)break;
        long long slot=elapsed/BusySessionPlan::slot_us;
        if(slot==previous_slot){Sleep(1);continue;}
        long long skipped=slot-previous_slot-1;skipped_slots+=skipped;previous_slot=slot;
        auto requested=plan.at(elapsed);phase_mask|=1u<<requested.phase;
        zoom_mask|=requested.width==128?1u:requested.width==160?2u:4u;
        LARGE_INTEGER begin={},captured={},map_done={},copy_done={},finished={},evidence_done={};QueryPerformanceCounter(&begin);
        if(!requested.same_camera(current)) {
            center_x=requested.x;center_y=requested.y;tile_width=requested.width;tile_height=tile_width/2;
            tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
            frame.tile_width=tile_width;frame.tile_height=tile_height;
        }
        current=requested;frame.presentation_time_ticks=frame.presentation_frequency+elapsed*frame.presentation_frequency/1000000;
        auto units=body_requests();QueryPerformanceCounter(&captured);
        int code=render_checked(&frame,&output);QueryPerformanceCounter(&map_done);
        ok=code==C3X_RENDERER_RESULT_OK && output.fallback_tile_count==0 && check_objects();
        if(ok)std::memcpy(composed_pixels,output.bgra_pixels,std::size_t(output.stride_bytes)*output.height);
        QueryPerformanceCounter(&copy_done);
        if(ok)ok=draw_bodies(units);QueryPerformanceCounter(&finished);
        long long done=microseconds(finished);
        bool superseded=!requested.same_camera(plan.at((std::min)(done,BusySessionPlan::duration_us-1)));
        late_cameras+=superseded;
        unsigned moving=0,attacking=0,fortifying=0,idling=0;
        for(auto const& unit:units){moving+=unit.action==2;attacking+=unit.action==3;fortifying+=unit.action==7;idling+=unit.action==1;}
        double scale=1000.0/frequency.QuadPart;
        std::printf("SESSION_FRAME frame=%u phase=%d label=%s requested_us=%lld dispatch_us=%lld done_us=%lld skipped_slots=%lld superseded=%d x=%d y=%d tile_width=%d units=%zu moving=%u attacking=%u fortifying=%u idling=%u result=%d ms=%.3f capture_ms=%.3f map_ms=%.3f copy_ms=%.3f units_ms=%.3f built=%u reused=%u upload_bytes=%u recoveries=%u\n",
            frames,requested.phase,requested.name,slot*BusySessionPlan::slot_us,elapsed,done,skipped,int(superseded),center_x,center_y,tile_width,units.size(),moving,attacking,fortifying,idling,code,
            (finished.QuadPart-begin.QuadPart)*scale,(captured.QuadPart-begin.QuadPart)*scale,(map_done.QuadPart-captured.QuadPart)*scale,
            (copy_done.QuadPart-map_done.QuadPart)*scale,(finished.QuadPart-copy_done.QuadPart)*scale,
            output.geometry_tiles_built,output.geometry_tiles_reused,output.geometry_upload_bytes,output.device_recoveries);
        bool sample_phase=requested.phase==0 || requested.phase==1 || (requested.phase==2 && tile_width==192) || requested.phase==3 || requested.phase==5 || requested.phase==7;
        std::size_t bytes=std::size_t(output.stride_bytes)*output.height;
        if(ok && sample_phase && !(snapshot_mask&(1u<<requested.phase)) && snapshot_bytes+bytes<=snapshot_limit) {
            auto data=static_cast<unsigned char const*>(composed_pixels);
            snapshots.push_back({frame,tiles,units,std::vector<unsigned char>(data,data+bytes),requested.phase,tile_width});
            snapshot_bytes+=bytes;snapshot_mask|=1u<<requested.phase;
        }
        if(frames%30==0)camera_memory();
        ++frames;std::fflush(stdout);QueryPerformanceCounter(&evidence_done);
        evidence_ms+=(evidence_done.QuadPart-finished.QuadPart)*scale;
    }
    QueryPerformanceCounter(&now);
    std::printf("SESSION_TIMED_END status=%s frames=%u wall_ms=%.3f skipped_slots=%lld late_cameras=%u phase_mask=%u zoom_mask=%u coverage_complete=%d evidence_ms=%.3f snapshot_bytes=%zu\n",
        ok?"pass":"FAIL",frames,double(microseconds(now))/1000,skipped_slots,late_cameras,phase_mask,zoom_mask,
        int(phase_mask==255 && zoom_mask==7),evidence_ms,snapshot_bytes);
    // Replay recorded snapshots only after timing. No verification resets can
    // warm, evict or stall the measured continuous session.
    unsigned verified=0;
    for(auto const& saved:snapshots) {
        if(!ok)break;
        auto picture=output;picture.bgra_pixels=saved.pixels.data();picture.stride_bytes=saved.frame.target_width*4;
        picture.width=saved.frame.target_width;picture.height=saved.frame.target_height;
        std::string path=std::string(argv[5])+".session-"+std::to_string(saved.phase)+"-"+std::to_string(saved.width)+".bmp";
        ok=write_bmp(path.c_str(),picture);reset();
        frame=saved.frame;tiles=saved.tiles;frame.tiles=tiles.data();tile_width=frame.tile_width;tile_height=frame.tile_height;
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK &&
            render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK && check_objects();
        if(ok)std::memcpy(composed_pixels,output.bgra_pixels,std::size_t(output.stride_bytes)*output.height);
        ok=ok && draw_bodies(saved.units) && std::memcmp(composed_pixels,saved.pixels.data(),saved.pixels.size())==0;
        std::printf("SESSION_PARITY phase=%d tile_width=%d status=%s units=%zu bytes=%zu\n",saved.phase,saved.width,ok?"pass":"FAIL",saved.units.size(),saved.pixels.size());
        if(ok)++verified;std::fflush(stdout);
    }
    if(canvas && old_bitmap)SelectObject(canvas,old_bitmap);
    if(bitmap)DeleteObject(bitmap);if(canvas)DeleteDC(canvas);
    std::printf("SESSION_END status=%s frames=%u snapshots=%zu verified=%u coverage_complete=%d\n",ok?"pass":"FAIL",frames,snapshots.size(),verified,int(phase_mask==255 && zoom_mask==7));
}
