static_assert(sizeof(c3x_renderer_gpu_frame_v1)==52 && sizeof(c3x_renderer_gpu_command_v1)==64 &&
              sizeof(c3x_renderer_gpu_images_v1)==60 && sizeof(c3x_renderer_gpu_result_v1)==48,"native C/C++ GPU API layout");
// Included in the production capture harness after its ordinary CPU control.
char gpu_frame_test[8]={};GetEnvironmentVariableA("C3X_RENDERER_GPU_FRAME_TEST",gpu_frame_test,sizeof(gpu_frame_test));
if(ok && !std::strcmp(gpu_frame_test,"1")) {
    auto gpu_render=reinterpret_cast<c3x_renderer_gpu_render_fn>(GetProcAddress(module,"c3x_renderer_gpu_render"));
    auto gpu_images=reinterpret_cast<c3x_renderer_gpu_images_fn>(GetProcAddress(module,"c3x_renderer_gpu_images"));
    auto verify_gpu=[&](bool value,char const* label){if(!value){std::printf("GPU_FRAME_FAIL %s\n",label);ok=false;}return value;};
    auto gpu_present=reinterpret_cast<c3x_renderer_gpu_present_fn>(GetProcAddress(module,"c3x_renderer_gpu_present"));
    auto gpu_reset=reinterpret_cast<c3x_renderer_reset_fn>(GetProcAddress(module,"c3x_renderer_reset"));
    if(!gpu_render||!gpu_images||!gpu_present||!gpu_reset)return 1;
    auto test_frame=frame;auto test_tiles=tiles;test_frame.tiles=test_tiles.data();
    c3x_renderer_gpu_frame_v1 view={sizeof(view)};
    c3x_renderer_gpu_result_v1 status={sizeof(status)};
    std::vector<unsigned> expected,map_expected,actual(std::size_t(frame.target_width)*frame.target_height);
    auto image_request=[&](int action,c3x_renderer_i64 image=0){c3x_renderer_gpu_images_v1 r={};r.struct_size=sizeof(r);r.action=action;r.ticket=view.ticket;r.image=image;return r;};
    auto execute=[&](c3x_renderer_gpu_images_v1 const& r){return gpu_images(&r,&status,nullptr,0);};
    auto read=[&](c3x_renderer_i64 image){auto r=image_request(C3X_GPU_READBACK,image);r.pixel_count=unsigned(actual.size());return gpu_images(&r,&status,actual.data(),unsigned(actual.size()));};
    c3x_renderer_i64 old_ticket=0;
    for(int phase=0;phase<4 && ok;++phase){
        if(phase==1)test_frame.presentation_time_ticks+=test_frame.presentation_frequency/4;
        if(phase==2)for(auto& tile:test_tiles){tile.anchor_x-=23;tile.anchor_y+=11;}
        if(phase==3)for(auto& tile:test_tiles)if(tile.tile_flags&C3X_RENDERER_TILE_RENDER){tile.terrain_type=(tile.terrain_type+1)%3;break;}
        gpu_reset(); // independent producer counters, not implicit CPU-mode retirement
        c3x_renderer_output_v1 control={C3X_RENDERER_API_VERSION,sizeof(control)};
        if(!verify_gpu(render(&test_frame,&control)==C3X_RENDERER_RESULT_OK,"CPU control render"))break;
        auto pixels=static_cast<unsigned const*>(control.bgra_pixels);expected.assign(pixels,pixels+actual.size());map_expected=expected;
        std::vector<unsigned> ownership;
        if(control.replacement_tile_count)ownership.assign(control.replacement_tile_flags,control.replacement_tile_flags+control.replacement_tile_count);
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&test_frame,{1,2,3,phase+1}};
        c3x_renderer_output_v1 meta={C3X_RENDERER_API_VERSION,sizeof(meta)};
        if(!verify_gpu(gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK,"GPU render")||
           !verify_gpu(!meta.bgra_pixels&&!view.map_readbacks&&view.width==test_frame.target_width&&view.height==test_frame.target_height,"GPU-only output")||
           !verify_gpu(meta.replacement_tile_count==ownership.size()&&std::equal(ownership.begin(),ownership.end(),meta.replacement_tile_flags),"native ownership parity"))break;
#ifdef C3X_GPU_NATIVE_CONTRACT
        if(phase==0){char jgl_path[2048]={};GetEnvironmentVariableA("C3X_RENDERER_GPU_JGL_TEST",jgl_path,sizeof(jgl_path));
            if(!verify_gpu(jgl_path[0]&&native_worker_contract(jgl_path,gpu_images,view,gpu_render,gpu_present,request,expected.data(),test_tiles[0].anchor_x,test_tiles[0].anchor_y),"actual native hooks on renderer worker"))break;
            // The independent native oracle intentionally reads back. Start a
            // fresh session for the existing producer/readback counters below.
            gpu_reset();
            if(!verify_gpu(render(&test_frame,&control)==C3X_RENDERER_RESULT_OK&&gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK,"fresh session after native oracle"))break;
        }
#endif
        if(old_ticket){auto stale=image_request(C3X_GPU_CREATE);stale.ticket=old_ticket;stale.width=stale.height=2;verify_gpu(execute(stale)==C3X_RENDERER_RESULT_SUPERSEDED,"old ticket rejected");}
        old_ticket=view.ticket;
        auto create=image_request(C3X_GPU_CREATE);create.width=view.width;create.height=view.height;
        if(!verify_gpu(execute(create)==C3X_RENDERER_RESULT_OK&&status.readbacks==0&&status.uploads==0,"resident map has no CPU upload/readback"))break;
        auto canvas=status.image;
        c3x_renderer_gpu_command_v1 draw={0,canvas,view.map_image,{0,0,view.width,view.height},{0,0,view.width,view.height},0,0,0};
        auto submit=image_request(C3X_GPU_SUBMIT);submit.commands=&draw;submit.command_count=1;
        if(!verify_gpu(execute(submit)==C3X_RENDERER_RESULT_OK&&status.readbacks==0,"GPU map-to-composition copy"))break;
        if(!verify_gpu(read(canvas)==C3X_RENDERER_RESULT_OK&&actual==expected,"exact resident-map pixels"))break;
        auto ui_create=image_request(C3X_GPU_CREATE);ui_create.width=ui_create.height=2;
        if(!verify_gpu(execute(ui_create)==C3X_RENDERER_RESULT_OK,"CPU UI surface"))break;
        auto ui=status.image;unsigned ui_pixels[4]={0xffff00ff,0xff012345,0xff6789ab,0xffff00ff};
        auto upload=image_request(C3X_GPU_UPLOAD,ui);upload.revision=1;upload.pixels=ui_pixels;upload.pixel_count=4;
        if(!verify_gpu(execute(upload)==C3X_RENDERER_RESULT_OK&&status.uploads==1,"owned CPU UI upload")||
           !verify_gpu(execute(upload)==C3X_RENDERER_RESULT_OK&&status.uploads==1,"unchanged UI revision reused"))break;
        c3x_renderer_gpu_command_v1 key={2,canvas,ui,{40,40,42,42},{0,0,view.width,view.height},0,0,0xffff00ff};
        submit.commands=&key;
        if(!verify_gpu(execute(submit)==C3X_RENDERER_RESULT_OK,"GPU UI composition"))break;
        expected[std::size_t(40)*view.width+41]=ui_pixels[1];expected[std::size_t(41)*view.width+40]=ui_pixels[2];
        submit.commands=&draw;
        // A native overlay-like fill and an invalid second command prove ordered
        // execution and complete preflight without altering the immutable map.
        draw.kind=1;draw.area[0]=7;draw.area[1]=9;draw.area[2]=31;draw.area[3]=27;draw.color=0xff123456;
        if(!verify_gpu(execute(submit)==C3X_RENDERER_RESULT_OK,"selected GPU fill"))break;
        for(int y=9;y<27;++y)for(int x=7;x<31;++x)expected[std::size_t(y)*view.width+x]=draw.color;
        c3x_renderer_gpu_command_v1 bad[2]={draw,draw};bad[0].color=0;bad[1].destination=view.map_image;
        submit.commands=bad;submit.command_count=2;
        verify_gpu(execute(submit)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"immutable map/whole transaction reject");
        if(!verify_gpu(read(canvas)==C3X_RENDERER_RESULT_OK&&actual==expected,"no partial rejected composition"))break;
        // Repeated GPU frames keep UI allocations alive, but retire the old map
        // identity. Commands from the preceding ticket cannot touch the new frame.
        auto before=view;
        if(!verify_gpu(gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK&&view.ticket!=before.ticket&&view.map_image!=before.map_image,"new map lifetime"))break;
        if(!verify_gpu(read(canvas)==C3X_RENDERER_RESULT_OK&&actual==expected,"composition survives next map render"))break;
        auto destroy=image_request(C3X_GPU_DESTROY,ui);verify_gpu(execute(destroy)==C3X_RENDERER_RESULT_OK,"release CPU UI source");
        destroy=image_request(C3X_GPU_DESTROY,canvas);verify_gpu(execute(destroy)==C3X_RENDERER_RESULT_OK,"release owned composition image");
        auto memory=camera_memory_values();
        std::printf("GPU_FRAME memory available_virtual=%llu largest_free_region=%zu\n",memory.first,memory.second);
        std::printf("GPU_FRAME phase=%d exact=1 tile_count=%u gpu_input_uploads=0 execution_readbacks=0 explicit_oracle_readbacks=%lld resident_bytes=%lld\n",phase,test_frame.tile_count,status.readbacks,status.resident_bytes);
    }
    // A CPU request must rebuild current CPU pixels rather than expose the stale
    // mirror left by GPU-only rendering. Native composition survives that work;
    // only the explicit lifecycle reset retires its handles.
    if(ok){c3x_renderer_output_v1 cpu={C3X_RENDERER_API_VERSION,sizeof(cpu)};
        verify_gpu(render(&test_frame,&cpu)==C3X_RENDERER_RESULT_OK&&cpu.bgra_pixels&&std::equal(map_expected.begin(),map_expected.end(),static_cast<unsigned const*>(cpu.bgra_pixels)),"exact CPU fallback after GPU output");
        auto stale=image_request(C3X_GPU_CREATE);stale.width=stale.height=2;
        verify_gpu(execute(stale)==C3X_RENDERER_RESULT_OK,"CPU request preserves GPU image session");
        auto destroy=image_request(C3X_GPU_DESTROY,status.image);verify_gpu(execute(destroy)==C3X_RENDERER_RESULT_OK,"release interleaved image");
        gpu_reset();verify_gpu(execute(stale)==C3X_RENDERER_RESULT_SUPERSEDED,"explicit reset retires GPU ticket");}
    std::puts(ok?"PASS resident map GPU worker: stationary, animation, scrolling, local content, exact pixels/ownership, immutable map, ticket lifetime, GPU composition and CPU fallback":"FAIL resident map GPU worker");
    // Return to the harness's original frame/output before its ordinary checks.
    if(ok)ok=render_checked(&frame,&output)==C3X_RENDERER_RESULT_OK;
}
