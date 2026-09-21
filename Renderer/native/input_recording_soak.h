// Included in the connected JGL fixture, before its recovery/config-off cases.
// Exercise real native ownership for wall-clock duration. This is a recorder
// endurance workload, not a gameplay or physical-scanout performance baseline.
char soak_option[16]={};
if(GetEnvironmentVariableA("C3X_RENDERER_INPUT_SOAK_SECONDS",soak_option,sizeof(soak_option))){
    auto seconds=std::atoi(soak_option);verify(seconds==30||seconds==600,"bounded input soak duration");
    verify(live_frame.world_topology_count==5000,"input soak uses Standard world topology");
    auto visual=reinterpret_cast<int(*)()>(GetProcAddress(renderer_module,"c3x_renderer_gpu_visual_frame"));
    auto status=reinterpret_cast<int(*)(c3x_renderer_visual_status_v1*)>(GetProcAddress(renderer_module,"c3x_renderer_gpu_visual_status"));
    auto clock=reinterpret_cast<c3x_renderer_visual_clock_fn>(GetProcAddress(renderer_module,"c3x_renderer_visual_clock"));
    verify(visual&&status&&clock,"input soak exports");
    auto saved_tiles=live_tiles;auto saved_frame=live_frame;auto saved_unit=unit;
    struct WindowWitnessInterval {
        HANDLE ready=nullptr,done=nullptr;
        ~WindowWitnessInterval(){if(ready)CloseHandle(ready);if(done)CloseHandle(done);}
    } witness;
    wchar_t witness_name[128]={};
    auto witness_length=GetEnvironmentVariableW(L"C3X_RENDERER_INPUT_WINDOW_EVENT",witness_name,128);
    if(witness_length){
        verify(witness_length<128,"window witness event length");
        witness.ready=CreateEventW(nullptr,TRUE,FALSE,witness_name);
        auto done_name=std::wstring(witness_name)+L"-done";witness.done=CreateEventW(nullptr,TRUE,FALSE,done_name.c_str());
        verify(witness.ready&&witness.done&&SetEvent(witness.ready),"begin external window evidence interval");
    }
    LARGE_INTEGER soak_frequency={};QueryPerformanceFrequency(&soak_frequency);
    auto began=GetTickCount64();unsigned iteration=0,accepted=0,pending=0,native_frames=0,cameras=0,last_second=UINT_MAX;
    while(GetTickCount64()-began<unsigned(seconds)*1000ull){
        auto elapsed=GetTickCount64()-began;unsigned second=unsigned(elapsed/1000),phase=(second/10)%3;
        LARGE_INTEGER before={},after={};QueryPerformanceCounter(&before);
        // Idle, directed action, and camera changes alternate. The same retained
        // mixed units and animated water are present throughout all three.
        bool native_frame=iteration==0||phase!=0;
        if(native_frame){
            if(iteration==0||(phase==2&&second!=last_second)){
                live_tiles=saved_tiles;int offset=phase==2?int(second%8)*4:0;
                for(auto& tile:live_tiles){tile.anchor_x-=offset;tile.anchor_y+=offset/2;}
                live_frame.tiles=live_tiles.data();live_frame.presentation_time_ticks=clock();
                c3x_renderer_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
                verify(exact_native_map(C3X_NATIVE_MAP_PREPARE,live_images[0],&live_request,&view)==1,"soak authoritative map prepare");
                verify(exact_native_map(C3X_NATIVE_MAP_COMMIT,live_images[0],nullptr,nullptr)==1,"soak authoritative map commit");++cameras;
            }
            copy(live_images[0],screen_surface,full);
            auto ticks=clock();
            for(unsigned actor=0;actor<8;++actor){
                auto body=saved_unit;body.unit_id=5000+int(actor);body.presentation_time_ticks=ticks;
                body.presentation_frequency=soak_frequency.QuadPart;body.body_x=w/3+int(actor%4)*100;
                body.body_y=h/3+int(actor/4)*140;unsigned playback=C3X_RENDERER_UNIT_STATE_CAPTURED|(actor==0?C3X_RENDERER_UNIT_SELECTED:0u);
                if(actor==1){strcpy_s(body.unit_key,"PRTO_Worker");body.action=13;}
                if(actor==2){body.action=2;body.action_cursor=int(iteration%16);body.body_x+=int(iteration%40);}
                int bounds[4]={};verify(live(C3X_NATIVE_UNIT_DRAW,screen_surface,screen_surface,&body,bounds,playback)==1,"soak mixed native units");
            }
            RECT stripe={8,h-32,96+int(iteration%64),h-8};
            verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&stripe,int(0x800003e0u+(iteration%31)))==0,"soak native HUD churn");
            verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&full,nullptr,0)==1,"soak native presentation");++native_frames;
        }
        verify(live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,1)==1,"soak visible animation policy");
        int code=visual();verify(code==1||code==C3X_RENDERER_RESULT_PENDING,"soak independent ambient frame");
        accepted+=code==1;pending+=code==C3X_RENDERER_RESULT_PENDING;QueryPerformanceCounter(&after);
        std::printf("INPUT_SOAK_SAMPLE step=%u phase=%u elapsed_ms=%llu native=%u result=%d call_ms=%.3f begin_qpc=%lld end_qpc=%lld\n",
            iteration,phase,elapsed,unsigned(native_frame),code,1000.*double(after.QuadPart-before.QuadPart)/soak_frequency.QuadPart,before.QuadPart,after.QuadPart);
        if(second!=last_second){
            MEMORYSTATUSEX memory={};memory.dwLength=sizeof(memory);verify(GlobalMemoryStatusEx(&memory)!=FALSE,"soak virtual memory");
            std::size_t address=0,largest=0,free_bytes=0;MEMORY_BASIC_INFORMATION block={};
            while(VirtualQuery(reinterpret_cast<void*>(address),&block,sizeof(block))){
                if(block.State==MEM_FREE){free_bytes+=block.RegionSize;largest=std::max(largest,std::size_t(block.RegionSize));}
                auto next=reinterpret_cast<std::size_t>(block.BaseAddress)+block.RegionSize;if(next<=address)break;address=next;
            }
            c3x_renderer_visual_status_v1 current={sizeof(current)};verify(status(&current)==1,"soak retained state");
            std::printf("INPUT_SOAK_MEMORY second=%u available_virtual=%llu free_bytes=%zu largest_free=%zu retained_bytes=%lld frames=%lld\n",
                second,memory.ullAvailVirtual,free_bytes,largest,current.retained_bytes,current.frames);
            std::fflush(stdout);last_second=second;
        }
        ++iteration;
        // External think time is separate from measured production work.
        Sleep(50);
    }
    verify(accepted>0&&native_frames>0&&cameras>0,"soak exercised all owners");
    // Windows' mandatory capture border changes desktop edge pixels. Observe
    // only this interval and close capture before resuming exact screen oracles.
    // No pixels are masked and no production visual comparison is weakened.
    if(witness.done)verify(WaitForSingleObject(witness.done,30000)==WAIT_OBJECT_0,"external window observer closed before exact desktop checks");
    live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,0);
    live_tiles=saved_tiles;live_frame=saved_frame;live_frame.tiles=live_tiles.data();unit=saved_unit;
    c3x_renderer_camera_view_v1 restored={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(restored)};
    verify(exact_native_map(C3X_NATIVE_MAP_PREPARE,live_images[0],&live_request,&restored)==1&&
        exact_native_map(C3X_NATIVE_MAP_COMMIT,live_images[0],nullptr,nullptr)==1,"restore pre-soak map");
    copy(live_images[0],screen_surface,full);
    // Regenerate the fixture's independent screen oracle before reset recovery.
    c3x_renderer_output_v1 control={C3X_RENDERER_API_VERSION,sizeof(control)};
    verify(render_view(&live_request,&control)==1,"restored CPU oracle");auto pixels=static_cast<unsigned const*>(control.bgra_pixels);
    expected.assign(pixels,pixels+std::size_t(w)*h);
    verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&full,nullptr,0)==1,"restored native presentation");capture_display(expected);
    std::printf("PASS input native soak: duration_ms=%llu requested_seconds=%d units=8 topology=5000 native_frames=%u cameras=%u ambient=%u pending=%u\n",
        GetTickCount64()-began,seconds,native_frames,cameras,accepted,pending);
}
