// Included in the existing full-world CSV capture harness. This proves renderer
// preparation through desktop presentation, excluding live Civ III input.
char world_test_option[8]={};
if(GetEnvironmentVariableA("C3X_RENDERER_WORLD_READINESS_TEST",world_test_option,sizeof(world_test_option)) && ok){
    auto set_world=reinterpret_cast<c3x_renderer_set_world_capture_fn>(GetProcAddress(module,"c3x_renderer_set_world_capture"));
    auto get_world=reinterpret_cast<c3x_renderer_world_status_fn>(GetProcAddress(module,"c3x_renderer_world_status"));
    auto gpu_world=reinterpret_cast<c3x_renderer_gpu_render_fn>(GetProcAddress(module,"c3x_renderer_gpu_render"));
    auto world_images=reinterpret_cast<c3x_renderer_gpu_images_fn>(GetProcAddress(module,"c3x_renderer_gpu_images"));
    auto world_present=reinterpret_cast<c3x_renderer_gpu_present_fn>(GetProcAddress(module,"c3x_renderer_gpu_present"));
    auto world_reset=reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(module,"c3x_renderer_reset"));
    if(!set_world || !get_world || !gpu_world || !world_images || !world_present || !world_reset)return 1;
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
            tile.tile_flags=(tile.tile_flags&C3X_RENDERER_TILE_VISIBILITY_BITS)|C3X_RENDERER_TILE_TOPOLOGY_HALO|
                ((tile.tile_flags&C3X_RENDERER_TILE_EXPLORED)?C3X_RENDERER_TILE_PREFETCH:0u);}
        return C3X_RENDERER_RESULT_OK;
    };
    c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&frame,{1,1,1,1}};
    c3x_renderer_gpu_frame_v1 image={sizeof(image)};
    c3x_renderer_output_v1 world_output={C3X_RENDERER_API_VERSION,sizeof(world_output)};
    // Retire the preview's earlier legacy/CPU baseline. This campaign starts
    // with the same authoritative GPU publication path as native integration.
    world_reset();ULONGLONG initial_start=GetTickCount64();
    if(gpu_world(&request,&image,&world_output)!=C3X_RENDERER_RESULT_OK || set_world(producer)!=C3X_RENDERER_RESULT_OK)return 1;
    auto initial_ms=GetTickCount64()-initial_start;
    ULONGLONG start=GetTickCount64();
    c3x_renderer_world_status_v1 state={sizeof(state)};
    while(GetTickCount64()-start<180000){
        MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
        if(get_world(&state)==C3X_RENDERER_RESULT_OK && state.capture_passes>0 && state.authoritative<=state.total &&
           state.preparation_sequence==state.appearance_sequence && state.prepared_regions==state.regions)break;
        if(state.unavailable_regions)break;
        MsgWaitForMultipleObjectsEx(0,nullptr,16,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
    }
    auto memory=camera_memory_values();
    std::printf("WORLD_READINESS total=%u authoritative=%u passes=%lld regions=%u attempted=%u unavailable=%u first_request_ms=%llu preparation_ms=%llu largest_free=%zu\n",
        state.total,state.authoritative,state.capture_passes,state.regions,state.prepared_regions,state.unavailable_regions,
        initial_ms,GetTickCount64()-start,std::size_t(memory.second));
    if(!state.capture_passes || state.authoritative>state.total ||
       state.preparation_sequence!=state.appearance_sequence || state.prepared_regions!=state.regions ||
       state.unavailable_regions){set_world(nullptr);return 1;}
    // No route-dependent warmup: the source paging/region policy above cannot
    // observe this seed or the destination sequence. Every first visit counts.
    int home_x=center_x,home_y=center_y,first_x=0,first_y=0;std::uint32_t random=0x38c3;
    LARGE_INTEGER frequency{};QueryPerformanceFrequency(&frequency);
    // Complete every sample through the existing presenter. Submission alone
    // allows queued GPU work to migrate into later requests and hides latency.
    WNDCLASSA window_class{};window_class.lpfnWndProc=DefWindowProcA;
    window_class.hInstance=GetModuleHandleA(nullptr);window_class.lpszClassName="C3XWorldReadiness";
    if(!RegisterClassA(&window_class))return 1;
    HWND world_window=CreateWindowExA(WS_EX_TOPMOST|WS_EX_TOOLWINDOW,window_class.lpszClassName,
        "World readiness contract",WS_POPUP,20,20,frame.target_width,frame.target_height,
        nullptr,nullptr,window_class.hInstance,nullptr);
    if(!world_window)return 1;
    ShowWindow(world_window,SW_SHOWNOACTIVATE);UpdateWindow(world_window);
    auto desktop_library=LoadLibraryA("dwmapi.dll");
    auto desktop_complete=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(desktop_library,"DwmFlush"));
    if(!desktop_complete)return 1;
    struct WorldOracle {int x,y;c3x_renderer_i64 clock;std::vector<unsigned> pixels;};
    std::vector<WorldOracle> oracles;
    auto read_world=[&](std::vector<unsigned>& pixels){
        pixels.resize(std::size_t(image.width)*image.height);
        c3x_renderer_gpu_images_v1 read{};read.struct_size=sizeof(read);read.action=C3X_GPU_READBACK;
        read.ticket=image.ticket;read.image=image.map_image;read.pixel_count=unsigned(pixels.size());
        c3x_renderer_gpu_result_v1 result{sizeof(result)};
        return world_images(&read,&result,pixels.data(),unsigned(pixels.size()))==C3X_RENDERER_RESULT_OK;
    };
    char short_workload_option[16]={};
    unsigned samples=GetEnvironmentVariableA("C3X_RENDERER_WORLD_READINESS_SAMPLES",
        short_workload_option,sizeof(short_workload_option))?
        unsigned(std::clamp(std::atoi(short_workload_option),1,100)):100u;
    bool bounded=samples>=12 && samples<=16;
    // BEGIN bounded readiness route contract (also exercised on the host).
    struct WorldReadinessRoute {
        enum {first_visits=6,sweep_size=3};
        std::vector<std::pair<int,int>> origins;
        std::pair<int,int> destination(unsigned n)const{return origins[n<first_visits?n:(n-first_visits)%sweep_size];}
        bool oracle(unsigned n)const{return n==0 || n==2 || n==4 || (n>=first_visits && n<first_visits+sweep_size);}
        bool sweep_end(unsigned n)const{return n>=first_visits && (n-first_visits+1)%sweep_size==0;}
        char const* phase(unsigned n)const{return n<first_visits?"first":n<first_visits+sweep_size?"restore":"repeat";}
    };
    // END bounded readiness route contract.
    WorldReadinessRoute route;
    auto evict_world=reinterpret_cast<c3x_renderer_benchmark_session_reset_v1_fn>(
        GetProcAddress(module,"c3x_renderer_benchmark_session_reset_v1"));
    unsigned evictions=0,sweeps=0,restored_requests=0;
    std::uint64_t plateau_geometry=0;std::size_t plateau_available=0;
    if(bounded){
        // Pick six distinct explored destinations only after initialization.
        // The producer and preparation policy never see this route or seed.
        for(std::size_t attempt=0;route.origins.size()<WorldReadinessRoute::first_visits &&
            attempt<world_records.size()*8;++attempt){
            random=random*1664525u+1013904223u;auto const& candidate=world_records[random%world_records.size()];
            if(!(candidate.tile_flags&C3X_RENDERER_TILE_EXPLORED))continue;
            std::pair<int,int> at{candidate.tile_x,candidate.tile_y};
            if(at==std::make_pair(home_x,home_y) || std::find(route.origins.begin(),route.origins.end(),at)!=route.origins.end())continue;
            route.origins.push_back(at);
        }
        if(route.origins.size()!=WorldReadinessRoute::first_visits || !evict_world){
            std::printf("FAIL bounded world readiness: explored destinations or ordered eviction endpoint unavailable\n");ok=false;
        }
        std::printf("WORLD_ROUTE fixture=bounded samples=%u first_visits=6 revisit_routes=3 oracle_samples=0,2,4,6,7,8\n",samples);
    }
    auto compiler_checkpoint=[&](){
        c3x_renderer_world_status_v1 snapshot{sizeof(snapshot)};auto deadline=GetTickCount64()+1000;
        do{if(get_world(&snapshot)==C3X_RENDERER_RESULT_OK)return true;Sleep(1);}while(GetTickCount64()<deadline);
        return false;
    };
    for(unsigned n=0;ok && n<samples;++n){
        if(bounded){
            if(n==WorldReadinessRoute::first_visits){
                c3x_renderer_benchmark_oracle_trim_v1 retired={C3X_RENDERER_BENCHMARK_ORACLE_VERSION,sizeof(retired)};
                LARGE_INTEGER begin{},end{};QueryPerformanceCounter(&begin);
                int code=evict_world(3,&retired);QueryPerformanceCounter(&end);
                std::printf("WORLD_EVICTION sample=%u result=%d evicted=%u geometry_bytes=%llu begin_qpc=%lld end_qpc=%lld\n",
                    n,code,retired.capacity_geometry_evictions,retired.cleared_viewport_bytes,begin.QuadPart,end.QuadPart);
                if(code!=C3X_RENDERER_RESULT_OK || !retired.capacity_geometry_evictions || !retired.cleared_viewport_bytes){ok=false;break;}
                ++evictions;
            }
            auto destination=route.destination(n);center_x=destination.first;center_y=destination.second;
        }else{
            random=random*1664525u+1013904223u;center_x=int(random%unsigned(map_width));
            random=random*1664525u+1013904223u;center_y=int(random%unsigned(map_height));
            center_x=(center_x&~1)|(center_y&1);
            if(!n){first_x=center_x;first_y=center_y;}
            if(samples>=3 && n==samples-2){center_x=first_x;center_y=first_y;}
            if(samples>=3 && n==samples-1){center_x=home_x;center_y=home_y;}
        }
        auto next_tiles=capture_view();auto next=frame;next.tiles=next_tiles.data();next.tile_count=unsigned(next_tiles.size());
        next.presentation_time_ticks+=c3x_renderer_i64(n+1)*next.presentation_frequency/30;
        request.frame=&next;world_output={C3X_RENDERER_API_VERSION,sizeof(world_output)};image={sizeof(image)};
        if(bounded && !compiler_checkpoint()){ok=false;break;}
        LARGE_INTEGER begin{},end{};QueryPerformanceCounter(&begin);
        int code=gpu_world(&request,&image,&world_output);QueryPerformanceCounter(&end);
        c3x_renderer_gpu_present_v1 show{};show.struct_size=sizeof(show);show.ticket=image.ticket;
        show.image=image.map_image;show.window=world_window;show.width=image.width;show.height=image.height;
        show.area[2]=image.width;show.area[3]=image.height;
        int presented_code=code==C3X_RENDERER_RESULT_OK?world_present(&show):code;
        LARGE_INTEGER presented{};QueryPerformanceCounter(&presented);
        if(code!=C3X_RENDERER_RESULT_OK || presented_code!=C3X_RENDERER_RESULT_OK || FAILED(desktop_complete())){ok=false;break;}
        LARGE_INTEGER displayed{};QueryPerformanceCounter(&displayed);auto memory_now=camera_memory_values();
        if(bounded && !compiler_checkpoint()){ok=false;break;}
        std::printf("WORLD_JUMP sample=%u x=%d y=%d result=%d request_ms=%.3f present_ms=%.3f desktop_wait_ms=%.3f desktop_ms=%.3f built=%u reused=%u uploads=%u readbacks=%u largest_free=%zu available_va=%zu geometry_bytes=%u begin_qpc=%lld end_qpc=%lld phase=%s\n",
            n,center_x,center_y,code,1000.*double(end.QuadPart-begin.QuadPart)/double(frequency.QuadPart),
            1000.*double(presented.QuadPart-end.QuadPart)/double(frequency.QuadPart),
            1000.*double(displayed.QuadPart-presented.QuadPart)/double(frequency.QuadPart),
            1000.*double(displayed.QuadPart-begin.QuadPart)/double(frequency.QuadPart),
            world_output.geometry_tiles_built,world_output.geometry_tiles_reused,world_output.geometry_upload_bytes,
            image.map_readbacks,std::size_t(memory_now.second),std::size_t(memory_now.first),world_output.geometry_cache_bytes,begin.QuadPart,displayed.QuadPart,
            bounded?route.phase(n):"ordinary");
        if(bounded){
            // The parser separately requires zero compiler calls in every
            // interval and backing restore calls during the restoration sweep.
            if(n==WorldReadinessRoute::first_visits){
                if((!world_output.geometry_tiles_reused && !world_output.geometry_tiles_built) ||
                   !world_output.geometry_upload_bytes){ok=false;break;}
                ++restored_requests;
            }
            if(n>=WorldReadinessRoute::first_visits+WorldReadinessRoute::sweep_size &&
               (world_output.geometry_tiles_built || world_output.geometry_upload_bytes)){ok=false;break;}
        }
        if((bounded && route.oracle(n)) ||
           (!bounded && (n==0 || n==2 || n==4 || n==9 || n==14 || n==25))){
            WorldOracle oracle{center_x,center_y,next.presentation_time_ticks,{}};
            if(!read_world(oracle.pixels)){ok=false;break;}oracles.push_back(std::move(oracle));
        }
        // Account diagnostic snapshots before the plateau baseline; their
        // retained pixels must not appear as renderer growth on the next sweep.
        if(bounded && route.sweep_end(n)){
            auto sweep_memory=camera_memory_values();
            ++sweeps;auto geometry=std::uint64_t(world_output.geometry_cache_bytes);
            auto geometry_growth=geometry>plateau_geometry?geometry-plateau_geometry:0;
            auto va_growth=plateau_available>std::size_t(sweep_memory.first)?plateau_available-std::size_t(sweep_memory.first):0;
            bool plateau=sweeps==1 || (geometry_growth<=1024u*1024u && va_growth<=8u*1024u*1024u);
            std::printf("WORLD_SWEEP sweep=%u sample=%u geometry_bytes=%llu available_va=%zu largest_free=%zu geometry_growth=%llu va_growth=%zu plateau=%u\n",
                sweeps,n,geometry,std::size_t(sweep_memory.first),std::size_t(sweep_memory.second),geometry_growth,va_growth,unsigned(plateau));
            if(sweeps==1){plateau_geometry=geometry;plateau_available=std::size_t(sweep_memory.first);}
            if(!plateau){ok=false;break;}
        }
    }
    c3x_renderer_gpu_present_v1 discard{};discard.struct_size=sizeof(discard);discard.action=1;discard.window=world_window;
    world_present(&discard);DestroyWindow(world_window);UnregisterClassA(window_class.lpszClassName,window_class.hInstance);FreeLibrary(desktop_library);
    set_world(nullptr);
    // Independent reference checks use a fresh world/device lifetime, but the
    // same authoritative world pages as the prepared path. Comparing a scoped
    // world with a view-only cold render changes cross-tile art at its edges.
    // Readback remains an explicit diagnostic oracle outside the timed run.
    unsigned oracle_index=0;
    for(auto const& oracle:oracles){
        world_reset();center_x=oracle.x;center_y=oracle.y;
        auto selected=capture_view();auto cold=frame;cold.tiles=selected.data();cold.tile_count=unsigned(selected.size());
        cold.presentation_time_ticks=oracle.clock;request.frame=&cold;image={sizeof(image)};
        world_output={C3X_RENDERER_API_VERSION,sizeof(world_output)};std::vector<unsigned> pixels;
        if(gpu_world(&request,&image,&world_output)!=C3X_RENDERER_RESULT_OK ||
           set_world(producer)!=C3X_RENDERER_RESULT_OK){ok=false;break;}
        ULONGLONG reference_start=GetTickCount64();c3x_renderer_world_status_v1 reference={sizeof(reference)};
        while(GetTickCount64()-reference_start<30000){
            MSG message{};while(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
            if(get_world(&reference)==C3X_RENDERER_RESULT_OK &&
               reference.authoritative==(bounded?state.authoritative:reference.total) &&
               reference.total==world_records.size() && reference.capture_passes>0 &&
               (bounded || (reference.preparation_sequence==reference.appearance_sequence &&
               reference.prepared_regions==reference.regions)))break;
            MsgWaitForMultipleObjectsEx(0,nullptr,16,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
        }
        image={sizeof(image)};world_output={C3X_RENDERER_API_VERSION,sizeof(world_output)};
        if(!reference.capture_passes || reference.authoritative!=(bounded?state.authoritative:reference.total) ||
           reference.total!=world_records.size() ||
           (!bounded && (reference.preparation_sequence!=reference.appearance_sequence ||
           reference.prepared_regions!=reference.regions || reference.unavailable_regions)) ||
           gpu_world(&request,&image,&world_output)!=C3X_RENDERER_RESULT_OK || !read_world(pixels)){ok=false;break;}
        set_world(nullptr);
        unsigned differences=0,maximum=0;
        for(unsigned i=0;i<pixels.size();++i)for(unsigned shift:{0u,8u,16u}){
            unsigned delta=unsigned(std::abs(int((pixels[i]>>shift)&255)-int((oracle.pixels[i]>>shift)&255)));
            differences+=delta!=0;maximum=(std::max)(maximum,delta);
        }
        std::printf("WORLD_ORACLE x=%d y=%d differing_channels=%u max_channel_delta=%u independent=1 clock=%lld\n",
            center_x,center_y,differences,maximum,oracle.clock);
        world_output.width=image.width;world_output.height=image.height;world_output.stride_bytes=image.width*4;
        world_output.bgra_pixels=pixels.data();
        write_bmp((std::string(argv[5])+".world-"+std::to_string(oracle_index++)+".bmp").c_str(),world_output);
        if(maximum>1){ok=false;
            world_output.width=cold.target_width;world_output.height=cold.target_height;world_output.stride_bytes=cold.target_width*4;
            world_output.bgra_pixels=oracle.pixels.data();write_bmp((std::string(argv[5])+".world-prepared.bmp").c_str(),world_output);
            world_output.bgra_pixels=pixels.data();write_bmp((std::string(argv[5])+".world-cold.bmp").c_str(),world_output);break;}
    }
    set_world(nullptr);world_records.clear();
    center_x=home_x;center_y=home_y;
    if(bounded && (evictions!=1 || !restored_requests || sweeps<2 || oracles.size()!=6))ok=false;
    std::printf("%s world readiness workload: samples=%u live_input=unmeasured desktop=measured oracles=%zu fixture=%s evictions=%u sweeps=%u\n",
        ok?"PASS":"FAIL",samples,oracles.size(),bounded?"bounded":"ordinary",evictions,sweeps);
    char only[8]={};if(GetEnvironmentVariableA("C3X_RENDERER_WORLD_READINESS_ONLY",only,sizeof(only))){
        world_reset();
        auto finish_inputs=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_input_recording_finish"));
        if(finish_inputs)finish_inputs();
        if(!shared_module)FreeLibrary(module);return ok?0:1;
    }
}
