// Current-camera acceptance, including work spent preparing, followed by a
// separate full-detail reference run. Included in the existing replay harness.
if(ok && prepared_view_fixture) {
    struct AreaSample {std::vector<c3x_renderer_tile_v1> tiles;std::vector<unsigned char> pixels;
        std::vector<c3x_renderer_u32> ownership;};
    std::vector<AreaSample> samples;
    int origin_x=center_x,origin_y=center_y;
    LARGE_INTEGER frequency={},begin={},end={};QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&begin);
    bool ready=!prepare_nearby_view;
    if(prepare_nearby_view) {
        center_x=origin_x+1;auto probe_tiles=capture_view();center_x=origin_x;
        auto probe=frame;probe.tiles=probe_tiles.data();probe.tile_count=unsigned(probe_tiles.size());
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&probe,{1,2,3,frame.world_topology_revision}};
        auto deadline=GetTickCount64()+30000;
        do {c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
            ready=camera_present_view(&request,&view)==C3X_RENDERER_RESULT_OK;
            if(!ready)Sleep(10);
        }while(!ready && GetTickCount64()<deadline);
    }
    QueryPerformanceCounter(&end);
    std::printf("PREPARED_VIEW opportunity_ms=%.3f supported=%u ready=%u\n",double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,unsigned(prepare_nearby_view!=nullptr),unsigned(ready));
    ok=ok && ready;
    for(auto offset:std::vector<std::pair<int,int>>{{0,0},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1},{0,0}}) {
        QueryPerformanceCounter(&begin);
        center_x=origin_x+offset.first;center_y=origin_y+offset.second;
        tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        int code=render_checked(&frame,&output);
        AreaSample sample;sample.tiles=tiles;
        if(code==C3X_RENDERER_RESULT_OK && preview_ownership(frame,output)) {
            auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            sample.pixels.assign(p,p+std::size_t(output.stride_bytes)*output.height);
            sample.ownership.assign(output.replacement_tile_flags,output.replacement_tile_flags+output.replacement_tile_count);
        } else ok=false;
        QueryPerformanceCounter(&end);
        std::printf("PREPARED_VIEW sample=%zu x=%d y=%d total_ms=%.3f built=%u upload=%u draw_ticks=%lld\n",samples.size(),offset.first,offset.second,
            double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,output.geometry_tiles_built,output.geometry_upload_bytes,output.draw_ticks);
        if(code==C3X_RENDERER_RESULT_OK)write_bmp((std::string(argv[5])+".area-"+std::to_string(samples.size())+".bmp").c_str(),output);
        samples.push_back(std::move(sample));
    }
    reset();handoff_ticket=0;
    ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK;
    unsigned exact=0;
    for(unsigned i=0;i<samples.size();++i) {
        auto& sample=samples[i];frame.tiles=sample.tiles.data();frame.tile_count=unsigned(sample.tiles.size());
        int code=render(&frame,&output);
        unsigned long long differing=0,absolute_error=0;unsigned maximum=0;
        if(code==C3X_RENDERER_RESULT_OK && sample.pixels.size()==std::size_t(output.stride_bytes)*output.height) {
            auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            for(unsigned j=0;j<sample.pixels.size();++j){auto d=unsigned(std::abs(int(p[j])-int(sample.pixels[j])));differing+=d!=0;absolute_error+=d;maximum=(std::max)(maximum,d);}
        } else differing=~0ull;
        bool ownership=code==C3X_RENDERER_RESULT_OK && preview_ownership(frame,output) && sample.ownership.size()==output.replacement_tile_count &&
            std::equal(sample.ownership.begin(),sample.ownership.end(),output.replacement_tile_flags);
        exact+=differing==0 && ownership;ok=ok && ownership;
        std::printf("PREPARED_VIEW reference=%u differing_bytes=%llu absolute_error=%llu maximum=%u ownership=%u\n",i,differing,absolute_error,maximum,unsigned(ownership));
        if(code==C3X_RENDERER_RESULT_OK)write_bmp((std::string(argv[5])+".area-reference-"+std::to_string(i)+".bmp").c_str(),output);
    }
    center_x=origin_x;center_y=origin_y;tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
    unsigned prepared_exact=0;
    if(prepare_nearby_view) {
        reset();handoff_ticket=0;
        ok=ok && set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK;
        ok=ok && render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
        auto probe=frame;probe.tiles=samples[1].tiles.data();probe.tile_count=unsigned(samples[1].tiles.size());
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&probe,{1,2,3,frame.world_topology_revision}};
        auto deadline=GetTickCount64()+30000;bool fresh_ready=false;
        do {c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
            fresh_ready=camera_present_view(&request,&view)==C3X_RENDERER_RESULT_OK;
            if(!fresh_ready)Sleep(10);
        }while(!fresh_ready && GetTickCount64()<deadline);
        ok=ok && fresh_ready;
        for(unsigned i=0;i<samples.size();++i){
            probe.tiles=samples[i].tiles.data();probe.tile_count=unsigned(samples[i].tiles.size());
            c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
            bool same=camera_present_view(&request,&view)==C3X_RENDERER_RESULT_OK && preview_ownership(probe,view.output);
            if(same)same=samples[i].pixels.size()==std::size_t(view.output.stride_bytes)*view.output.height &&
                !std::memcmp(samples[i].pixels.data(),view.output.bgra_pixels,samples[i].pixels.size()) &&
                std::equal(samples[i].ownership.begin(),samples[i].ownership.end(),view.output.replacement_tile_flags);
            prepared_exact+=same;ok=ok && same;
        }
    } else ok=ok && exact==samples.size();
    std::printf("PREPARED_VIEW_END status=%s exact_prepared=%u control_exact=%u samples=%zu visual_review=%s\n",
        ok?"pass":"fail",prepared_exact,exact,samples.size(),exact==samples.size()?"unchanged":"required");
    // Separately timed, real elapsed animation clock. No file writes or oracle
    // preparation waits occur between these current-camera requests.
    auto offsets=std::vector<std::pair<int,int>>{{0,0},{1,0},{1,1},{0,1},{-1,1},{-1,0},{-1,-1},{0,-1}};
    LARGE_INTEGER paced_start={};QueryPerformanceCounter(&paced_start);
    auto start_clock=frame.presentation_time_ticks;
    for(unsigned i=0;i<48;++i){
        if(i)Sleep(67);
        QueryPerformanceCounter(&begin);
        frame.presentation_time_ticks=start_clock+(begin.QuadPart-paced_start.QuadPart)*frame.presentation_frequency/frequency.QuadPart;
        auto offset=offsets[i%offsets.size()];center_x=origin_x+offset.first;center_y=origin_y+offset.second;
        tiles=capture_view();frame.tiles=tiles.data();frame.tile_count=unsigned(tiles.size());
        int code=render_checked(&frame,&output);
        std::vector<unsigned char> copied;
        if(code==C3X_RENDERER_RESULT_OK){auto p=static_cast<unsigned char const*>(output.bgra_pixels);
            copied.assign(p,p+std::size_t(output.stride_bytes)*output.height);}
        ok=ok && code==C3X_RENDERER_RESULT_OK && preview_ownership(frame,output);
        QueryPerformanceCounter(&end);
        std::printf("PREPARED_PACED sample=%u total_ms=%.3f built=%u upload=%u draw_ticks=%lld\n",i,
            double(end.QuadPart-begin.QuadPart)*1000/frequency.QuadPart,output.geometry_tiles_built,output.geometry_upload_bytes,output.draw_ticks);
    }
    std::printf("PREPARED_PACED_END status=%s samples=48\n",ok?"pass":"fail");
    camera_memory();
}
