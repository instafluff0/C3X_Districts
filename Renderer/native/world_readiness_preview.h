// Included in the existing full-world CSV capture harness. This proves renderer
// preparation/request work, not live Civ III input or desktop presentation.
char world_test_option[8]={};
if(GetEnvironmentVariableA("C3X_RENDERER_WORLD_READINESS_TEST",world_test_option,sizeof(world_test_option)) && ok){
    auto set_world=reinterpret_cast<c3x_renderer_set_world_capture_fn>(GetProcAddress(module,"c3x_renderer_set_world_capture"));
    auto get_world=reinterpret_cast<c3x_renderer_world_status_fn>(GetProcAddress(module,"c3x_renderer_world_status"));
    auto gpu_world=reinterpret_cast<c3x_renderer_gpu_render_fn>(GetProcAddress(module,"c3x_renderer_gpu_render"));
    if(!set_world || !get_world || !gpu_world)return 1;
    static std::vector<c3x_renderer_tile_v1> world_records;
    static DWORD world_thread=0;
    world_thread=GetCurrentThreadId();capture_whole_world=true;world_records=capture_view();capture_whole_world=false;
    std::sort(world_records.begin(),world_records.end(),[](auto const& a,auto const& b){
        return a.tile_y!=b.tile_y?a.tile_y<b.tile_y:a.tile_x<b.tile_x;});
    if(world_records.size()!=world.size())return 1;
    auto producer=+[](c3x_renderer_world_page_v1* page)->int{
        if(GetCurrentThreadId()!=world_thread || page->first>=world_records.size())return C3X_RENDERER_RESULT_ERROR;
        page->count=unsigned((std::min)(std::size_t(page->capacity),world_records.size()-page->first));
        for(unsigned i=0;i<page->count;++i){page->tiles[i]=world_records[page->first+i];
            auto& tile=page->tiles[i];tile.anchor_x=tile.anchor_y=0;
            tile.tile_flags=(tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_BITS)|C3X_RENDERER_TILE_TOPOLOGY_HALO|C3X_RENDERER_TILE_PREFETCH;}
        return C3X_RENDERER_RESULT_OK;
    };
    c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{1,1,1,1}};
    c3x_renderer_gpu_frame_v1 image={sizeof(image)};
    c3x_renderer_output_v1 world_output={C3X_RENDERER_API_VERSION,sizeof(world_output)};
    if(gpu_world(&request,&image,&world_output)!=C3X_RENDERER_RESULT_OK || set_world(producer)!=C3X_RENDERER_RESULT_OK)return 1;
    ULONGLONG start=GetTickCount64();
    c3x_renderer_world_status_v1 state={sizeof(state)};
    while(GetTickCount64()-start<180000){
        MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
        if(get_world(&state)==C3X_RENDERER_RESULT_OK && state.capture_passes>0 && state.authoritative==state.total &&
           state.preparation_sequence==state.appearance_sequence && state.prepared_regions==state.regions)break;
        MsgWaitForMultipleObjectsEx(0,nullptr,16,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
    }
    auto memory=camera_memory_values();
    std::printf("WORLD_READINESS total=%u authoritative=%u passes=%lld regions=%u attempted=%u unavailable=%u preparation_ms=%llu largest_free=%zu\n",
        state.total,state.authoritative,state.capture_passes,state.regions,state.prepared_regions,state.unavailable_regions,
        GetTickCount64()-start,std::size_t(memory.second));
    if(!state.capture_passes || state.authoritative!=state.total){set_world(nullptr);return 1;}
    // No route-dependent warmup: the source paging/region policy above cannot
    // observe this seed or the destination sequence. Every first visit counts.
    int home_x=center_x,home_y=center_y;std::uint32_t random=0x38c3;
    LARGE_INTEGER frequency{};QueryPerformanceFrequency(&frequency);
    for(unsigned n=0;n<100;++n){
        random=random*1664525u+1013904223u;center_x=int(random%unsigned(map_width));
        random=random*1664525u+1013904223u;center_y=int(random%unsigned(map_height));
        center_x=(center_x&~1)|(center_y&1);
        auto next_tiles=capture_view();auto next=frame;next.tiles=next_tiles.data();next.tile_count=unsigned(next_tiles.size());
        next.presentation_time_ticks+=c3x_renderer_i64(n+1)*next.presentation_frequency/30;
        request.frame=&next;world_output={C3X_RENDERER_API_VERSION,sizeof(world_output)};image={sizeof(image)};
        LARGE_INTEGER begin{},end{};QueryPerformanceCounter(&begin);
        int code=gpu_world(&request,&image,&world_output);QueryPerformanceCounter(&end);
        std::printf("WORLD_JUMP sample=%u x=%d y=%d result=%d request_ms=%.3f built=%u reused=%u uploads=%u\n",
            n,center_x,center_y,code,1000.*double(end.QuadPart-begin.QuadPart)/double(frequency.QuadPart),
            world_output.geometry_tiles_built,world_output.geometry_tiles_reused,world_output.geometry_upload_bytes);
        if(code!=C3X_RENDERER_RESULT_OK){ok=false;break;}
    }
    center_x=home_x;center_y=home_y;set_world(nullptr);world_records.clear();
    std::printf("%s world readiness workload: samples=100 live_input=unmeasured desktop=unmeasured\n",ok?"PASS":"FAIL");
}
