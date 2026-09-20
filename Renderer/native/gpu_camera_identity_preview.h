// Included in the native GPU harness with its existing capture/oracle helpers.
char identity_test[8]={};GetEnvironmentVariableA("C3X_RENDERER_GPU_CAMERA_IDENTITY_TEST",identity_test,sizeof(identity_test));
if(phase==0 && ok && !std::strcmp(identity_test,"1")){
    auto begin=reinterpret_cast<c3x_renderer_gpu_camera_begin_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
    auto poll=reinterpret_cast<c3x_renderer_gpu_camera_poll_view_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll_view"));
    if(!verify_gpu(begin && poll,"atomic identity exports"))return 1;
    struct Witness {c3x_renderer_frame_v1 frame;std::vector<c3x_renderer_tile_v1> tiles;std::vector<unsigned> pixels,coverage;};
    std::vector<Witness> witnesses;
    c3x_renderer_gpu_camera_view_v1 selected={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(selected)};
    auto epochs=request.identity;unsigned prepared_count=0;
    for(int step=0;step<16 && ok;++step){
        auto input=test_frame;auto captured=test_tiles;input.tiles=captured.data();
        if(step==1 || step==2){input.clip_left=7;input.clip_top=9;input.clip_right-=11;input.clip_bottom-=13;
            input.visible_animation_count=17;input.dirty_flags=3;
            if(step==2)input.presentation_time_ticks+=input.presentation_frequency/2;}
        if(step==3)std::reverse(captured.begin(),captured.end());
        if(step==4){input.tile_width=test_frame.tile_width*3/2;input.tile_height=test_frame.tile_height*3/2;
            for(auto& tile:captured){tile.anchor_x=tile.anchor_x*3/2;tile.anchor_y=tile.anchor_y*3/2;}}
        if(step==5){input.target_width-=32;input.target_height-=16;input.clip_right=input.target_width;input.clip_bottom=input.target_height;}
        if(step==6)for(auto& tile:captured){tile.anchor_x-=37;tile.anchor_y+=19;}
        if(step==7){input.world_wrap_x=1;for(auto& tile:captured)tile.tile_x+=input.world_width_tiles;}
        if(step==8){++epochs.visibility_epoch;for(auto& tile:captured){tile.tile_flags|=C3X_RENDERER_TILE_VISIBILITY_KNOWN|C3X_RENDERER_TILE_EXPLORED;tile.tile_flags&=~C3X_RENDERER_TILE_VISIBLE;}}
        if(step==9)++epochs.map_epoch;
        if(step==10)++epochs.viewer_epoch;
        if(step==11)++epochs.scene_epoch;
        if(step==12)++input.world_topology_revision;
        if(step==13){input.hour=0;input.season=(input.season+1)%4;}
        auto old=selected;std::vector<c3x_renderer_tile_v1> old_occurrences;
        if(step && step<14)old_occurrences.assign(old.camera.frame.tiles,old.camera.frame.tiles+old.camera.frame.tile_count);
        if(step==14 || step==15){
            if(step==14)verify_gpu(set_definitions(argv[2],argv[3],nullptr,custom_path)==C3X_RENDERER_RESULT_OK,"configuration retirement");
            else gpu_reset();
            auto untouched=selected;
            verify_gpu(poll(old.camera.ticket,&selected)==C3X_RENDERER_RESULT_SUPERSEDED && !std::memcmp(&untouched,&selected,sizeof(selected)),"retirement rejects old atomic identity");
        }
        c3x_renderer_camera_request_v1 demand={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(demand),&input,epochs};
        c3x_renderer_i64 ticket=0;verify_gpu(begin(&demand,&ticket)==C3X_RENDERER_RESULT_PENDING && ticket>old.camera.ticket,"identity transition never aliases retired ticket");
        if(step && step<14){
            auto untouched=selected;
            verify_gpu(poll(old.camera.ticket,&selected)==C3X_RENDERER_RESULT_SUPERSEDED && !std::memcmp(&untouched,&selected,sizeof(selected)),"old identity cannot adopt");
            verify_gpu(!std::memcmp(old.camera.frame.tiles,old_occurrences.data(),old_occurrences.size()*sizeof(old_occurrences[0])),"pending demand preserves adopted occurrences");
        }
        auto untouched=selected;int code=C3X_RENDERER_RESULT_PENDING;auto deadline=GetTickCount64()+30000;
        while(ok && code==C3X_RENDERER_RESULT_PENDING && GetTickCount64()<deadline){
            code=poll(ticket,&selected);
            if(code==C3X_RENDERER_RESULT_PENDING){verify_gpu(!std::memcmp(&untouched,&selected,sizeof(selected)),"pending identity is atomic");Sleep(1);}
        }
        if(!verify_gpu(code==C3X_RENDERER_RESULT_OK,"identity transition adopted"))break;
        auto expected_frame=input;expected_frame.tiles=selected.camera.frame.tiles;
        expected_frame.world_topology=nullptr;expected_frame.world_topology_count=0;
        expected_frame.presentation_time_ticks=selected.image.presentation_time_ticks;
        auto const& metadata=selected.camera.output;
        verify_gpu(selected.camera.ticket==ticket && !std::memcmp(&selected.camera.identity,&epochs,sizeof(epochs)) &&
            !std::memcmp(&selected.camera.frame,&expected_frame,sizeof(expected_frame)) &&
            !std::memcmp(selected.camera.frame.tiles,captured.data(),captured.size()*sizeof(captured[0])) &&
            metadata.clip_left==input.clip_left && metadata.clip_top==input.clip_top && metadata.clip_right==input.clip_right && metadata.clip_bottom==input.clip_bottom &&
            metadata.width==input.target_width && metadata.height==input.target_height && metadata.replacement_tile_count==captured.size() &&
            metadata.device_generation==selected.image.device_generation && metadata.content_revision==selected.image.content_revision &&
            selected.image.width==input.target_width && selected.image.height==input.target_height && selected.image.session>0 &&
            selected.image.presentation_time_ticks<=input.presentation_time_ticks && !metadata.bgra_pixels && !selected.image.map_readbacks,
            "complete adopted identity and coverage");
        if(step==1 || step==2)verify_gpu(selected.image.prepared && !metadata.geometry_tiles_built && !metadata.geometry_upload_bytes &&
            selected.image.presentation_time_ticks==test_frame.presentation_time_ticks,"metadata-only reuse preserves honest sample without builds");
        if(step==3 || step==7)verify_gpu(!selected.image.prepared,"changed occurrence order/basis cannot relabel old pixels");
        if(step==14 || step==15)verify_gpu(selected.image.session!=old.image.session || selected.image.content_revision!=old.image.content_revision ||
            selected.image.device_generation!=old.image.device_generation,"retirement advances lifecycle identity");
        prepared_count+=selected.image.prepared!=0;
        view=selected.image;actual.resize(std::size_t(input.target_width)*input.target_height);
        verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK,"atomic image oracle readback");
        auto requested_ticks=input.presentation_time_ticks;input.presentation_time_ticks=selected.image.presentation_time_ticks;
        witnesses.push_back({input,std::move(captured),actual,std::vector<unsigned>(metadata.replacement_tile_flags,metadata.replacement_tile_flags+metadata.replacement_tile_count)});
        std::printf("ATOMIC_IDENTITY step=%d prepared=%u sample=%lld requested=%lld result=%u\n",step,selected.image.prepared,input.presentation_time_ticks,requested_ticks,unsigned(ok));std::fflush(stdout);
    }
    // Independent CPU production views use each image's actual sampled clock.
    // Run after the transition sequence so the oracle cannot retire its donors.
    for(unsigned step=0;step<witnesses.size() && ok;++step){auto& witness=witnesses[step];witness.frame.tiles=witness.tiles.data();
        gpu_reset();std::vector<unsigned> oracle;
        bool rendered=capture_reference(witness.frame,oracle);
        if(rendered && oracle!=witness.pixels){
            unsigned changed=0,maximum=0;for(std::size_t n=0;n<oracle.size();++n)if(oracle[n]!=witness.pixels[n]){
                ++changed;for(unsigned channel=0;channel<3;++channel){
                    int a=int((oracle[n]>>(channel*8))&255),b=int((witness.pixels[n]>>(channel*8))&255);
                    maximum=(std::max)(maximum,unsigned(std::abs(a-b)));
                }
            }
            std::printf("ATOMIC_DIFFERENCE step=%u changed=%u maximum=%u\n",step,changed,maximum);
        }
        verify_gpu(rendered && oracle==witness.pixels,"atomic transition exact cold pixels");
        auto cpu_render=reinterpret_cast<c3x_renderer_render_view_fn>(GetProcAddress(module,"c3x_renderer_render_view"));
        c3x_renderer_camera_request_v1 demand={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(demand),&witness.frame,{11,12,13,14}};
        c3x_renderer_output_v1 coverage={C3X_RENDERER_API_VERSION,sizeof(coverage)};
        verify_gpu(cpu_render && cpu_render(&demand,&coverage)==C3X_RENDERER_RESULT_OK && !coverage.fallback_tile_count &&
            coverage.replacement_tile_count==witness.coverage.size() && std::equal(witness.coverage.begin(),witness.coverage.end(),coverage.replacement_tile_flags),"atomic transition exact cold replacement coverage");
        std::printf("ATOMIC_PIXELS step=%u exact=%u\n",step,unsigned(ok));std::fflush(stdout);
    }
    if(!ok)return 1;
    std::printf("PASS atomic GPU identity transitions: cases=%zu prepared=%u exact_pixels=1 clip=1 zoom=1 wrap=1 order=1 epochs=1 configuration=1 reset=1\n",witnesses.size(),prepared_count);
    actual.resize(std::size_t(test_frame.target_width)*test_frame.target_height);gpu_reset();
    verify_gpu(render(&test_frame,&control)==C3X_RENDERER_RESULT_OK && capture_reference(test_frame,expected) &&
        gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK,"restore phase after identity matrix");map_expected=expected;
}
