static_assert(sizeof(c3x_renderer_gpu_frame_v1)==64 && sizeof(c3x_renderer_gpu_command_v1)==104 &&
              sizeof(c3x_renderer_gpu_images_v1)==64 && sizeof(c3x_renderer_gpu_result_v1)==48,"native C/C++ GPU API layout");
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
    // One actual ambient resource makes the independent visual-frame fixture
    // exercise map animation, not just repeated static terrain submission.
    auto animal=std::min_element(test_tiles.begin(),test_tiles.end(),[&](auto const& a,auto const& b){
        auto distance=[&](auto const& t){return (t.tile_flags&C3X_RENDERER_TILE_RENDER)&&t.real_terrain_type<=4?
            std::abs(t.anchor_x-frame.target_width/2)+std::abs(t.anchor_y-frame.target_height/2):INT_MAX;};return distance(a)<distance(b);});
    if(animal!=test_tiles.end()){animal->resource_id=101;animal->resource_class=0;strcpy_s(animal->resource_name,"Cattle");}
    c3x_renderer_gpu_frame_v1 view={sizeof(view)};
    c3x_renderer_gpu_result_v1 status={sizeof(status)};
    std::vector<unsigned> expected,map_expected,actual(std::size_t(frame.target_width)*frame.target_height);
    auto image_request=[&](int action,c3x_renderer_i64 image=0){c3x_renderer_gpu_images_v1 r={};r.struct_size=sizeof(r);r.action=action;r.ticket=view.ticket;r.image=image;return r;};
    auto execute=[&](c3x_renderer_gpu_images_v1 const& r){return gpu_images(&r,&status,nullptr,0);};
    auto read=[&](c3x_renderer_i64 image){auto r=image_request(C3X_GPU_READBACK,image);r.pixel_count=unsigned(actual.size());return gpu_images(&r,&status,actual.data(),unsigned(actual.size()));};
    auto capture_reference=[&](c3x_renderer_frame_v1 const& input,std::vector<unsigned>& image){
        auto present_view=reinterpret_cast<c3x_renderer_camera_present_view_fn>(GetProcAddress(module,"c3x_renderer_camera_present_view"));
        auto render_view=reinterpret_cast<c3x_renderer_render_view_fn>(GetProcAddress(module,"c3x_renderer_render_view"));
        if(!present_view || !render_view)return false;
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&input,{11,12,13,14}};
        c3x_renderer_camera_view_v1 lease={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(lease)};
        present_view(&request,&lease); // enable the native CPU publication policy
        c3x_renderer_output_v1 cpu={C3X_RENDERER_API_VERSION,sizeof(cpu)};
        if(render_view(&request,&cpu)!=C3X_RENDERER_RESULT_OK || !cpu.bgra_pixels ||
           cpu.width!=input.target_width || cpu.height!=input.target_height)return false;
        auto pixels=static_cast<unsigned const*>(cpu.bgra_pixels);image.assign(pixels,pixels+std::size_t(cpu.width)*cpu.height);return true;
    };
    c3x_renderer_i64 old_ticket=0;
    char coverage_test[8]={};GetEnvironmentVariableA("C3X_RENDERER_SCROLL_COVERAGE_TEST",coverage_test,sizeof(coverage_test));
    if(!std::strcmp(coverage_test,"1")){
        auto longest_black=[](c3x_renderer_output_v1 const& image){int longest=0;
            for(int y=16;y<image.height-16;y+=8){auto row=reinterpret_cast<unsigned const*>(static_cast<unsigned char const*>(image.bgra_pixels)+std::size_t(y)*image.stride_bytes);int run=0;
                for(int x=16;x<image.width-16;++x){run=(row[x]&0xffffff)?0:run+1;longest=(std::max)(longest,run);}}
            return longest;
        };
        for(int step=0;step<40;++step){
            tile_width=step<24?128:192;tile_height=tile_width/2;
            int position=step-step%2; // Repeat the committed camera/clock after cancellation too.
            int ox=-3616-(position<24?position:position-24)*84,oy=-940-(position>=12?(position%12)*64:0);
            center_x=(target_width/2-tile_width/2-ox)/(tile_width/2);
            center_y=(target_height/2-tile_height/2-oy)/(tile_height/2);
            auto selected=capture_view();auto input=test_frame;input.tile_width=tile_width;input.tile_height=tile_height;
            int dx=ox-(selected[0].anchor_x-selected[0].tile_x*tile_width/2),dy=oy-(selected[0].anchor_y-selected[0].tile_y*tile_height/2);
            for(auto& tile:selected){tile.anchor_x+=dx;tile.anchor_y+=dy;}
            input.tiles=selected.data();input.tile_count=unsigned(selected.size());input.presentation_time_ticks=1000000;input.presentation_frequency=1000000;
            if(step){
                auto begin_camera=reinterpret_cast<c3x_renderer_camera_begin_fn>(GetProcAddress(module,"c3x_renderer_camera_begin"));
                auto cancel_camera=reinterpret_cast<c3x_renderer_camera_cancel_fn>(GetProcAddress(module,"c3x_renderer_camera_cancel"));
                auto future=input;auto future_tiles=selected;for(auto& tile:future_tiles)tile.anchor_y+=64;
                future.tiles=future_tiles.data();future.presentation_time_ticks+=step*100000;
                c3x_renderer_i64 ticket=0;if(!begin_camera||!cancel_camera||begin_camera(&future,&ticket)!=C3X_RENDERER_RESULT_PENDING)return 1;
                Sleep(5+step%4*25);cancel_camera(ticket);Sleep(250);
            }
            input.presentation_time_ticks+=position*100000;
            c3x_renderer_output_v1 warm={C3X_RENDERER_API_VERSION,sizeof(warm)};
            std::vector<unsigned> warm_pixels;if(!capture_reference(input,warm_pixels))return 1;
            warm.width=input.target_width;warm.height=input.target_height;warm.stride_bytes=warm.width*4;warm.bgra_pixels=warm_pixels.data();
            int black=longest_black(warm);std::printf("SCROLL_COVERAGE step=%d origin=%d,%d tile_width=%d black_span=%d\n",step,ox,oy,tile_width,black);std::fflush(stdout);
            if(black>=32){
                write_bmp((std::string(argv[5])+".scroll-warm.bmp").c_str(),warm);
                gpu_reset();SetEnvironmentVariableA("C3X_RENDERER_WORLD_PREPARATION","0");
                c3x_renderer_output_v1 cold={C3X_RENDERER_API_VERSION,sizeof(cold)};
                if(render(&input,&cold)!=C3X_RENDERER_RESULT_OK)return 1;
                write_bmp((std::string(argv[5])+".scroll-cold.bmp").c_str(),cold);
                std::printf("SCROLL_COVERAGE cold_black_span=%d warm_black_span=%d\n",longest_black(cold),black);
                return 1;
            }
            Sleep(100); // Let independent visual work run between camera demands.
        }
        std::puts("PASS scroll coverage: fine pans, two axes, zoom and independent visual work; no missing map strips");return 0;
    }
    if(GetEnvironmentVariableA("C3X_RENDERER_WAVE_VISIBILITY_TEST",nullptr,0)){
        // Same geometry, projection and clock: only authoritative visibility
        // changes. Compare reused occurrences with an independently reset scene.
        auto capture_map=[&](c3x_renderer_frame_v1 const& source,std::vector<unsigned>& pixels){
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&source,{11,12,13,14}};
            c3x_renderer_output_v1 metadata={C3X_RENDERER_API_VERSION,sizeof(metadata)};
            if(gpu_render(&request,&view,&metadata)!=C3X_RENDERER_RESULT_OK || read(view.map_image)!=C3X_RENDERER_RESULT_OK)return false;
            pixels=actual;return true;
        };
        auto input=test_frame;auto records=test_tiles;input.tiles=records.data();
        input.presentation_frequency=1000000;input.presentation_time_ticks=1000000;
        // Keep selected contributors identical under the retirement control.
        // The copied topology/visibility halo remains available.
        for(auto& tile:records)tile.tile_flags&=~C3X_RENDERER_TILE_PREFETCH;
        std::vector<unsigned> warm,cold;
        if(!capture_map(input,warm))return 1;
        gpu_reset();if(!capture_map(input,cold))return 1;
        if(!verify_gpu(warm==cold,"same visible clock before and after reset"))return 1;
        auto begin_camera=reinterpret_cast<c3x_renderer_gpu_camera_begin_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
        auto cancel_camera=reinterpret_cast<c3x_renderer_camera_cancel_fn>(GetProcAddress(module,"c3x_renderer_camera_cancel"));
        for(unsigned delay:{0u,5u,30u,100u}){
            auto next=input;auto shifted=records;next.tiles=shifted.data();
            for(auto& tile:shifted){tile.anchor_x-=23;tile.anchor_y+=11;}
            next.presentation_time_ticks+=1000000;
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&next,{11,12,13,14}};
            c3x_renderer_i64 ticket=0;
            if(!begin_camera||!cancel_camera||begin_camera(&request,&ticket)!=C3X_RENDERER_RESULT_PENDING)return 1;
            Sleep(delay);cancel_camera(ticket);
            if(!capture_map(input,warm)||!verify_gpu(warm==cold,"same wave sample after cancelled camera"))return 1;
        }
        auto moved=input;auto translated=records;moved.tiles=translated.data();
        for(auto& tile:translated){tile.anchor_x-=52;tile.anchor_y+=28;}
        if(!capture_map(moved,warm))return 1;
        // The existing contributor-selection control deterministically retires
        // occurrences without changing the requested frame identity, just as an
        // interrupted view does. Immutable coast cells remain reusable.
        SetEnvironmentVariableA("C3X_RENDERER_PREFETCH_FOREGROUND_CONTROL","1");
        if(!capture_map(moved,warm))return 1;
        gpu_reset();if(!capture_map(moved,cold))return 1;
        std::size_t shifted_pixels=0;for(std::size_t i=0;i<warm.size();++i)shifted_pixels+=warm[i]!=cold[i];
        std::printf("WAVE_REBASE changed_pixels=%zu\n",shifted_pixels);std::fflush(stdout);
        if(shifted_pixels){c3x_renderer_output_v1 image={C3X_RENDERER_API_VERSION,sizeof(image)};
            image.width=moved.target_width;image.height=moved.target_height;image.stride_bytes=image.width*4;
            image.bgra_pixels=warm.data();write_bmp((std::string(argv[5])+".warm.bmp").c_str(),image);
            image.bgra_pixels=cold.data();write_bmp((std::string(argv[5])+".cold.bmp").c_str(),image);}
        if(!verify_gpu(!shifted_pixels,"rebased shoreline occurrences match cold scene"))return 1;
        for(unsigned visibility:{1u,2u,0u,2u}){
            for(auto& tile:records){tile.tile_flags&=~C3X_RENDERER_TILE_VISIBILITY_BITS;
                tile.tile_flags|=C3X_RENDERER_TILE_VISIBILITY_KNOWN;
                if(visibility)tile.tile_flags|=C3X_RENDERER_TILE_EXPLORED;
                if(visibility==2)tile.tile_flags|=C3X_RENDERER_TILE_VISIBLE;}
            if(!capture_map(input,warm))return 1;
            gpu_reset();if(!capture_map(input,cold))return 1;
            std::size_t different=0;for(std::size_t i=0;i<warm.size();++i)different+=warm[i]!=cold[i];
            std::printf("WAVE_VISIBILITY state=%u changed_pixels=%zu\n",visibility,different);std::fflush(stdout);
            if(!verify_gpu(!different,"warm visibility matches independent cold scene"))return 1;
            if(visibility<2){auto later=input;later.presentation_time_ticks+=2000000;
                if(!capture_map(later,warm)||!verify_gpu(warm==cold,"fogged and unseen scenery stays frozen"))return 1;}
        }
        std::puts("PASS wave visibility: visible/fog/unseen/reveal, same-clock warm/cold parity, frozen fog");return 0;
    }
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
        if(!verify_gpu(capture_reference(test_frame,expected),"CPU working-area reference"))break;map_expected=expected;
        c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&test_frame,{1,2,3,phase+1}};
        c3x_renderer_output_v1 meta={C3X_RENDERER_API_VERSION,sizeof(meta)};
        if(!verify_gpu(gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK,"GPU render")||
           !verify_gpu(!meta.bgra_pixels&&!view.map_readbacks&&view.width==test_frame.target_width&&view.height==test_frame.target_height,"GPU-only output")||
           !verify_gpu(meta.replacement_tile_count==ownership.size()&&std::equal(ownership.begin(),ownership.end(),meta.replacement_tile_flags),"native ownership parity"))break;
        char fresh_witness[8]={};
        if(phase==0 && GetEnvironmentVariableA("C3X_RENDERER_FRESH_WITNESS",
                fresh_witness,sizeof(fresh_witness)) && !std::strcmp(fresh_witness,"1")) {
            if(!verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK,
                    "fresh map readback witness"))break;
            auto witness=meta;witness.bgra_pixels=actual.data();
            if(!verify_gpu(write_bmp((std::string(argv[5])+".fresh.bmp").c_str(),witness),
                    "fresh map image witness"))break;
            std::printf("GPU_FRAME_FRESH_WITNESS width=%d height=%d built=%u reused=%u map_readbacks=%u\n",
                view.width,view.height,meta.geometry_tiles_built,
                meta.geometry_tiles_reused,view.map_readbacks);
            auto spawn_unit=reinterpret_cast<c3x_renderer_unit_spawn_fn>(GetProcAddress(module,"c3x_renderer_unit_spawn"));
            auto state_unit=reinterpret_cast<c3x_renderer_unit_state_fn>(GetProcAddress(module,"c3x_renderer_unit_state"));
            auto capture_unit=reinterpret_cast<c3x_renderer_gpu_unit_fn>(GetProcAddress(module,"c3x_renderer_gpu_unit"));
            auto land=std::min_element(test_tiles.begin(),test_tiles.end(),[&](auto const& a,auto const& b){
                auto distance=[&](auto const& t){return (t.tile_flags&C3X_RENDERER_TILE_RENDER)&&
                    (t.tile_flags&C3X_RENDERER_TILE_VISIBLE)&&t.real_terrain_type<=4&&t.city_id<0&&
                    t.tile_x>=0&&t.tile_x+2<test_frame.world_width_tiles&&t.tile_y>=0&&
                    t.anchor_y>test_frame.target_height/2?
                    std::abs(t.anchor_x-test_frame.target_width/2)+std::abs(t.anchor_y-test_frame.target_height*3/4):INT_MAX;};
                return distance(a)<distance(b);
            });
            if(!verify_gpu(spawn_unit&&state_unit&&capture_unit&&land!=test_tiles.end()&&
                land->real_terrain_type<=4,"fresh unit fixture"))break;
            c3x_renderer_unit_spawn_v1 birth={sizeof(birth)};
            birth.unit_id=28173;birth.tile_x=land->tile_x;birth.tile_y=land->tile_y;
            birth.unit_type_id=1;birth.owner_id=0;birth.visible=1;
            birth.map_epoch=request.identity.map_epoch;birth.viewer_epoch=request.identity.viewer_epoch;
            birth.presentation_frequency=test_frame.presentation_frequency;
            birth.presentation_time_ticks=test_frame.presentation_time_ticks-3;
            c3x_renderer_unit_state_v1 unit_state={sizeof(unit_state)};
            unit_state.kind=C3X_RENDERER_UNIT_STATE_OBSERVE;unit_state.unit_id=birth.unit_id;
            unit_state.tile_x=birth.tile_x;unit_state.tile_y=birth.tile_y;
            unit_state.unit_type_id=birth.unit_type_id;unit_state.owner_id=birth.owner_id;
            unit_state.action=1;unit_state.max_hp=3;unit_state.visible=1;
            unit_state.map_epoch=birth.map_epoch;unit_state.viewer_epoch=birth.viewer_epoch;
            unit_state.presentation_frequency=birth.presentation_frequency;
            unit_state.presentation_time_ticks=birth.presentation_time_ticks+1;
            c3x_renderer_unit_v1 body={};body.struct_size=sizeof(body);
            strcpy_s(body.unit_key,"PRTO_Warrior");body.unit_id=birth.unit_id;
            body.action=1;body.direction=3;body.frame_count=16;
            body.sprite_width=body.sprite_height=191;
            body.projection_scale_milli=test_frame.tile_width*1000/128;
            body.body_x=land->anchor_x+test_frame.tile_width/2-191*body.projection_scale_milli/2000;
            body.body_y=land->anchor_y+test_frame.tile_height/2-191*body.projection_scale_milli/2000;
            body.presentation_frequency=birth.presentation_frequency;
            body.presentation_time_ticks=birth.presentation_time_ticks+2;
            body.hour=test_frame.hour;body.season=test_frame.season;
            body.display_color_rgb=0x205bdd;
            c3x_renderer_gpu_unit_v1 unit_target={sizeof(unit_target)};
            unit_target.ticket=view.ticket;unit_target.destination=view.map_image;
            unit_target.background=view.map_image;
            unit_target.clip[2]=test_frame.target_width;unit_target.clip[3]=test_frame.target_height;
            unit_target.playback_flags=C3X_RENDERER_UNIT_STATE_CAPTURED;
            int body_bounds[4]={};auto before_unit=actual;
            if(!verify_gpu(spawn_unit(&birth)==C3X_RENDERER_RESULT_OK&&
                state_unit(&unit_state)==C3X_RENDERER_RESULT_OK&&
                capture_unit(&body,&unit_target,body_bounds)==C3X_RENDERER_RESULT_OK,
                "fresh captured unit accepted"))break;
            auto unit_request=request;++unit_request.identity.scene_epoch;
            c3x_renderer_output_v1 unit_meta={C3X_RENDERER_API_VERSION,sizeof(unit_meta)};
            if(!verify_gpu(gpu_render(&unit_request,&view,&unit_meta)==C3X_RENDERER_RESULT_OK&&
                !view.map_readbacks&&unit_meta.raster_draw_pixels==0,
                "fresh unit map rendered without raster work")||
               !verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK,
                "fresh unit image witness"))break;
            std::size_t unit_pixels=0;
            for(std::size_t i=0;i<actual.size();++i)unit_pixels+=actual[i]!=before_unit[i];
            std::printf("GPU_FRAME_FRESH_UNIT changed_pixels=%zu map_readbacks=%u raster_pixels=%u\n",
                unit_pixels,view.map_readbacks,unit_meta.raster_draw_pixels);
            if(!verify_gpu(unit_pixels>0,"fresh unit visible in map scene"))break;
            witness=unit_meta;witness.bgra_pixels=actual.data();
            if(!verify_gpu(write_bmp((std::string(argv[5])+".unit.bmp").c_str(),witness),
                "fresh unit screenshot"))break;
            auto unit_move=reinterpret_cast<c3x_renderer_unit_move_fn>(GetProcAddress(module,"c3x_renderer_unit_move"));
            auto observe_unit=reinterpret_cast<c3x_renderer_unit_visual_fn>(GetProcAddress(module,"c3x_renderer_unit_visual"));
            auto destination_tile=std::find_if(test_tiles.begin(),test_tiles.end(),[&](auto const& tile){
                return tile.tile_x==birth.tile_x+2&&tile.tile_y==birth.tile_y&&
                    (tile.tile_flags&C3X_RENDERER_TILE_VISIBLE);});
            if(!verify_gpu(unit_move&&observe_unit&&destination_tile!=test_tiles.end(),
                "one-tile move destination visible"))break;
            c3x_renderer_unit_move_v1 movement={sizeof(movement)};
            movement.unit_id=birth.unit_id;movement.old_x=birth.tile_x;movement.old_y=birth.tile_y;
            movement.new_x=destination_tile->tile_x;movement.new_y=destination_tile->tile_y;
            movement.action=2;movement.source_visible=movement.target_visible=1;
            movement.map_epoch=birth.map_epoch;movement.viewer_epoch=birth.viewer_epoch;
            movement.presentation_frequency=birth.presentation_frequency;
            movement.presentation_time_ticks=body.presentation_time_ticks+1;
            unit_state.tile_x=movement.new_x;unit_state.tile_y=movement.new_y;
            unit_state.action=2;unit_state.presentation_time_ticks=movement.presentation_time_ticks+1;
            body.action=2;body.action_cursor=4;body.body_x+=test_frame.tile_width/2;
            body.presentation_time_ticks=unit_state.presentation_time_ticks+1;
            c3x_renderer_unit_visual_v1 motion={sizeof(motion)};
            motion.unit_id=body.unit_id;motion.action=body.action;
            motion.pixel_x=body.body_x;motion.pixel_y=body.body_y;
            motion.target_x=body.body_x+test_frame.tile_width/2;motion.target_y=body.body_y;
            motion.body_x=body.body_x;motion.body_y=body.body_y;
            motion.projection_scale_milli=body.projection_scale_milli;
            motion.max_hp=3;motion.flags=C3X_RENDERER_UNIT_STATE_CAPTURED;
            motion.presentation_frequency=body.presentation_frequency;
            motion.presentation_time_ticks=body.presentation_time_ticks;
            if(!verify_gpu(unit_move(&movement)==C3X_RENDERER_RESULT_OK&&
                state_unit(&unit_state)==C3X_RENDERER_RESULT_OK&&
                observe_unit(&motion)==C3X_RENDERER_RESULT_OK&&
                capture_unit(&body,&unit_target,body_bounds)==C3X_RENDERER_RESULT_OK,
                "ordered native one-tile move captured"))break;
            auto moving_frame=test_frame;moving_frame.presentation_time_ticks+=100000;
            auto moving_request=unit_request;moving_request.frame=&moving_frame;
            ++moving_request.identity.scene_epoch;
            c3x_renderer_output_v1 moving_meta={C3X_RENDERER_API_VERSION,sizeof(moving_meta)};
            if(!verify_gpu(gpu_render(&moving_request,&view,&moving_meta)==C3X_RENDERER_RESULT_OK&&
                !view.map_readbacks&&moving_meta.raster_draw_pixels==0&&
                read(view.map_image)==C3X_RENDERER_RESULT_OK,
                "moving native unit in fresh map"))break;
            witness=moving_meta;witness.bgra_pixels=actual.data();
            if(!verify_gpu(write_bmp((std::string(argv[5])+".moving.bmp").c_str(),witness),
                "moving unit screenshot"))break;
            std::printf("GPU_FRAME_FRESH_MOVE id=%d source=%d,%d target=%d,%d map_readbacks=%u raster_pixels=%u\n",
                body.unit_id,movement.old_x,movement.old_y,movement.new_x,movement.new_y,
                view.map_readbacks,moving_meta.raster_draw_pixels);
            auto shifted_tiles=test_tiles;
            for(auto& tile:shifted_tiles){tile.anchor_x+=96;tile.anchor_y+=48;}
            auto shifted=test_frame;shifted.tiles=shifted_tiles.data();
            shifted.presentation_time_ticks+=shifted.presentation_frequency/2;
            auto shifted_request=request;shifted_request.frame=&shifted;
            ++shifted_request.identity.scene_epoch;
            c3x_renderer_output_v1 shifted_meta={C3X_RENDERER_API_VERSION,sizeof(shifted_meta)};
            int shifted_code=gpu_render(&shifted_request,&view,&shifted_meta);
            std::printf("GPU_FRAME_FRESH_SCROLL_CODE code=%d\n",shifted_code);
            if(!verify_gpu(shifted_code==C3X_RENDERER_RESULT_OK,
                    "fresh scrolled GPU render"))break;
            if(!verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK,
                    "fresh scrolled readback witness"))break;
            witness=shifted_meta;witness.bgra_pixels=actual.data();
            if(!verify_gpu(write_bmp((std::string(argv[5])+".scrolled.bmp").c_str(),witness),
                    "fresh scrolled image witness"))break;
            std::printf("GPU_FRAME_FRESH_SCROLL built=%u reused=%u invalidations=%u map_readbacks=%u\n",
                shifted_meta.geometry_tiles_built,shifted_meta.geometry_tiles_reused,
                shifted_meta.frame_invalidation_flags,view.map_readbacks);
            auto edited_tiles=shifted_tiles;
            auto edited=std::find_if(edited_tiles.begin(),edited_tiles.end(),[](auto const& tile){
                return (tile.tile_flags&C3X_RENDERER_TILE_RENDER)!=0 &&
                    tile.terrain_type<=4 && !(tile.improvement_flags&C3X_RENDERER_IMPROVEMENT_MINE);
            });
            if(edited==edited_tiles.end())break;
            edited->improvement_flags|=C3X_RENDERER_IMPROVEMENT_MINE;
            auto local=shifted;local.tiles=edited_tiles.data();
            auto local_request=shifted_request;local_request.frame=&local;
            ++local_request.identity.scene_epoch;
            c3x_renderer_output_v1 local_meta={C3X_RENDERER_API_VERSION,sizeof(local_meta)};
            if(!verify_gpu(gpu_render(&local_request,&view,&local_meta)==C3X_RENDERER_RESULT_OK,
                    "fresh local GPU render"))break;
            if(!verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK,
                    "fresh local readback witness"))break;
            witness=local_meta;witness.bgra_pixels=actual.data();
            if(!verify_gpu(write_bmp((std::string(argv[5])+".edited.bmp").c_str(),witness),
                    "fresh local image witness"))break;
            std::printf("GPU_FRAME_FRESH_EDIT built=%u reused=%u invalidations=%u map_readbacks=%u\n",
                local_meta.geometry_tiles_built,local_meta.geometry_tiles_reused,
                local_meta.frame_invalidation_flags,view.map_readbacks);
            int home_x=center_x,home_y=center_y,home_width=tile_width,home_height=tile_height;
            for(int step=0;step<4 && ok;++step){
                center_x=step==0?home_x+18:step==1?home_x+18:step==2?home_x+map_width:home_x;
                center_y=step<2?home_y+8:home_y;
                tile_width=step==1?160:home_width;tile_height=tile_width/2;
                auto navigation_tiles=capture_view();
                auto navigation=test_frame;navigation.tiles=navigation_tiles.data();
                navigation.tile_count=unsigned(navigation_tiles.size());
                navigation.tile_width=tile_width;navigation.tile_height=tile_height;
                navigation.presentation_time_ticks+=c3x_renderer_i64(step+1)*200000;
                auto navigation_request=request;navigation_request.frame=&navigation;
                navigation_request.identity.scene_epoch+=10+step;
                c3x_renderer_output_v1 navigation_meta={C3X_RENDERER_API_VERSION,sizeof(navigation_meta)};
                if(!verify_gpu(gpu_render(&navigation_request,&view,&navigation_meta)==C3X_RENDERER_RESULT_OK&&
                    !view.map_readbacks&&navigation_meta.raster_draw_pixels==0,
                    "fresh jump zoom wrap return"))break;
                std::printf("GPU_FRAME_FRESH_NAV step=%d center=%d,%d zoom=%d built=%u reused=%u map_readbacks=%u raster_pixels=%u\n",
                    step,center_x,center_y,tile_width,navigation_meta.geometry_tiles_built,
                    navigation_meta.geometry_tiles_reused,view.map_readbacks,navigation_meta.raster_draw_pixels);
                {
                    if(!verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK,
                        "fresh navigation screenshot"))break;
                    std::size_t occupied=0;for(auto pixel:actual)occupied+=pixel!=0;
                    std::printf("GPU_FRAME_FRESH_NAV_PIXELS step=%d occupied=%zu\n",step,occupied);
                    if(!verify_gpu(occupied>0,"fresh navigation map has visible pixels"))break;
                    witness=navigation_meta;witness.bgra_pixels=actual.data();
                    auto name=std::string(argv[5])+(step==0?".jump.bmp":step==1?".zoom.bmp":
                        step==2?".wrap.bmp":".return.bmp");
                    if(!verify_gpu(write_bmp(name.c_str(),witness),
                        "fresh navigation image"))break;
                }
            }
            center_x=home_x;center_y=home_y;tile_width=home_width;tile_height=home_height;
            if(!ok)break;
            // The game uses the copied, pending camera API. A cold camera may
            // take much longer to build than either native submission or a
            // concurrent unit fact; neither caller may wait for that frame.
            auto async_begin=reinterpret_cast<c3x_renderer_gpu_camera_begin_fn>(
                GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
            auto async_poll=reinterpret_cast<c3x_renderer_gpu_camera_poll_view_fn>(
                GetProcAddress(module,"c3x_renderer_gpu_camera_poll_view"));
            if(!verify_gpu(async_begin&&async_poll,"fresh asynchronous camera exports"))break;
            center_x=home_x+42;center_y=home_y-5;
            auto pending_tiles=capture_view();
            auto pending_frame=test_frame;pending_frame.tiles=pending_tiles.data();
            pending_frame.tile_count=unsigned(pending_tiles.size());
            pending_frame.presentation_time_ticks+=pending_frame.presentation_frequency*3;
            auto pending_request=request;pending_request.frame=&pending_frame;
            pending_request.identity.scene_epoch+=40;
            LARGE_INTEGER clock_rate={},begin_time={},begin_done={},unit_done={};
            QueryPerformanceFrequency(&clock_rate);
            c3x_renderer_i64 pending_ticket=0;
            QueryPerformanceCounter(&begin_time);
            int begin_code=async_begin(&pending_request,&pending_ticket);
            QueryPerformanceCounter(&begin_done);
            if(!verify_gpu(begin_code==C3X_RENDERER_RESULT_PENDING&&pending_ticket>0,
                "fresh cold camera accepted without rendering"))break;
            int pending_bounds[4]={};
            unit_target.ticket=view.ticket;unit_target.destination=view.map_image;
            unit_target.background=view.map_image;
            int unit_code=capture_unit(&body,&unit_target,pending_bounds);
            QueryPerformanceCounter(&unit_done);
            if(!verify_gpu(unit_code==C3X_RENDERER_RESULT_OK,
                "fresh unit capture while cold camera prepares"))break;
            double begin_ms=1000.0*double(begin_done.QuadPart-begin_time.QuadPart)/clock_rate.QuadPart;
            double unit_ms=1000.0*double(unit_done.QuadPart-begin_done.QuadPart)/clock_rate.QuadPart;
            c3x_renderer_gpu_camera_view_v1 adopted={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(adopted)};
            int poll_code=C3X_RENDERER_RESULT_PENDING;
            auto deadline=GetTickCount64()+15000;
            while(poll_code==C3X_RENDERER_RESULT_PENDING&&GetTickCount64()<deadline){
                poll_code=async_poll(pending_ticket,&adopted);
                if(poll_code==C3X_RENDERER_RESULT_PENDING)Sleep(1);
            }
            if(!verify_gpu(poll_code==C3X_RENDERER_RESULT_OK&&
                adopted.image.map_readbacks==0&&adopted.camera.output.raster_draw_pixels==0,
                "fresh asynchronous camera publication"))break;
            double render_ms=1000.0*double(adopted.camera.output.renderer_cpu_ticks)/clock_rate.QuadPart;
            std::printf("GPU_FRAME_FRESH_ASYNC begin_ms=%.3f unit_capture_ms=%.3f render_ms=%.3f built=%u map_readbacks=%u\n",
                begin_ms,unit_ms,render_ms,adopted.camera.output.geometry_tiles_built,
                adopted.image.map_readbacks);
            if(!verify_gpu(render_ms>100.0&&begin_ms<render_ms/3&&unit_ms<render_ms/3,
                "native fact submission independent of cold Renderer64 frame"))break;
            center_x=home_x;center_y=home_y;
            gpu_reset();
            c3x_renderer_output_v1 reset_meta={C3X_RENDERER_API_VERSION,sizeof(reset_meta)};
            if(!verify_gpu(gpu_render(&request,&view,&reset_meta)==C3X_RENDERER_RESULT_OK,
                    "fresh post-reset GPU render"))break;
            std::printf("GPU_FRAME_FRESH_RESET built=%u reused=%u map_readbacks=%u\n",
                reset_meta.geometry_tiles_built,reset_meta.geometry_tiles_reused,
                view.map_readbacks);
            gpu_reset();return 0;
        }
        char camera_test[8]={};GetEnvironmentVariableA("C3X_RENDERER_GPU_CAMERA_TEST",camera_test,sizeof(camera_test));
        if(phase==0 && !std::strcmp(camera_test,"1")){
            auto begin=reinterpret_cast<c3x_renderer_gpu_camera_begin_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
            auto poll=reinterpret_cast<c3x_renderer_gpu_camera_poll_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll"));
            auto poll_view=reinterpret_cast<c3x_renderer_gpu_camera_poll_view_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll_view"));
            auto cancel=reinterpret_cast<c3x_renderer_camera_cancel_fn>(GetProcAddress(module,"c3x_renderer_camera_cancel"));
            if(!verify_gpu(begin && poll && cancel,"GPU camera exports"))break;
            auto future=test_frame;auto future_tiles=test_tiles;future.tiles=future_tiles.data();
            auto next=request;next.frame=&future;
            c3x_renderer_i64 ticket=0,prior=0,duplicate=0;
            LARGE_INTEGER frequency={},started={},ended={};QueryPerformanceFrequency(&frequency);
            std::vector<double> submission_ms;
            auto pending=view;pending.ticket=-77;
            auto pending_meta=meta;pending_meta.width=-77;
            auto invalid_meta=pending_meta;invalid_meta.api_version=0;
            verify_gpu(poll(0,&pending,&invalid_meta)==C3X_RENDERER_RESULT_BAD_ARGUMENT && pending.ticket==-77,"GPU poll validates output ABI");
            c3x_renderer_gpu_camera_view_v1 atomic_view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(atomic_view)};
            atomic_view.image.ticket=-77;auto untouched=atomic_view;
            if(poll_view){
                auto invalid=atomic_view;invalid.version=0;
                verify_gpu(poll_view(0,&invalid)==C3X_RENDERER_RESULT_BAD_ARGUMENT && invalid.image.ticket==-77,"atomic GPU view validates version");
                invalid=atomic_view;invalid.struct_size--;
                verify_gpu(poll_view(0,&invalid)==C3X_RENDERER_RESULT_BAD_ARGUMENT && invalid.image.ticket==-77,"atomic GPU view validates size");
            }
            // Exact view/time/visibility identity, durable local edits and copied
            // input lifetime all participate in the production request queue.
            for(int step=0;step<64 && ok;++step){
                for(std::size_t n=0;n<future_tiles.size();++n){future_tiles[n].anchor_x=test_tiles[n].anchor_x-step*3;future_tiles[n].anchor_y=test_tiles[n].anchor_y+step;}
                future.presentation_time_ticks=test_frame.presentation_time_ticks+step*future.presentation_frequency/60;
                next.identity.visibility_epoch=100+step;
                QueryPerformanceCounter(&started);int code=begin(&next,&ticket);QueryPerformanceCounter(&ended);
                submission_ms.push_back(double(ended.QuadPart-started.QuadPart)*1000/frequency.QuadPart);
                verify_gpu(code==C3X_RENDERER_RESULT_PENDING && ticket>prior,"replace GPU camera demand");
                if(prior)verify_gpu(poll(prior,&pending,&pending_meta)==C3X_RENDERER_RESULT_SUPERSEDED && pending.ticket==-77 && pending_meta.width==-77,"stale poll leaves output untouched");
                if(prior && poll_view)verify_gpu(poll_view(prior,&atomic_view)==C3X_RENDERER_RESULT_SUPERSEDED &&
                    !std::memcmp(&atomic_view,&untouched,sizeof(atomic_view)),"stale atomic view leaves output untouched");
                verify_gpu(begin(&next,&duplicate)==C3X_RENDERER_RESULT_PENDING && duplicate==ticket,"identical GPU request retains ticket");
                prior=ticket;
            }
            // Reading the adopted front is a real GPU operation and may pause
            // assembly, but must not retire the pending request or change pixels.
            verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK && actual==map_expected,"adopted front survives replacement demand");
            auto frozen_tiles=future_tiles;future_tiles.clear();future_tiles.shrink_to_fit();future.tiles=nullptr;
            int camera_result=C3X_RENDERER_RESULT_PENDING;auto deadline=GetTickCount64()+15000;
            while(ok && camera_result==C3X_RENDERER_RESULT_PENDING && GetTickCount64()<deadline){
                camera_result=poll_view?poll_view(ticket,&atomic_view):poll(ticket,&pending,&pending_meta);
                if(poll_view && camera_result==C3X_RENDERER_RESULT_OK){pending=atomic_view.image;pending_meta=atomic_view.camera.output;}
                if(camera_result==C3X_RENDERER_RESULT_PENDING){
                    if(poll_view)verify_gpu(!std::memcmp(&atomic_view,&untouched,sizeof(atomic_view)),"pending atomic view leaves output untouched");
                    verify_gpu(pending.ticket==-77 && pending_meta.width==-77,"pending poll leaves output untouched");Sleep(1);
                }
            }
            verify_gpu(camera_result==C3X_RENDERER_RESULT_OK && pending.ticket!=view.ticket && !pending.map_readbacks && !pending_meta.bgra_pixels,"latest copied camera adopted resident");
            if(poll_view && ok){
                auto const& captured=atomic_view.camera.frame;auto expected_frame=future;
                expected_frame.tiles=captured.tiles;expected_frame.world_topology=nullptr;expected_frame.world_topology_count=0;
                verify_gpu(atomic_view.camera.ticket==ticket && !std::memcmp(&atomic_view.camera.identity,&next.identity,sizeof(next.identity)) &&
                    !std::memcmp(&captured,&expected_frame,sizeof(captured)) && captured.tile_count==frozen_tiles.size() &&
                    !std::memcmp(captured.tiles,frozen_tiles.data(),frozen_tiles.size()*sizeof(frozen_tiles[0])) &&
                    atomic_view.image.presentation_time_ticks==captured.presentation_time_ticks &&
                    atomic_view.image.width==captured.target_width && atomic_view.image.height==captured.target_height &&
                    atomic_view.image.device_generation==atomic_view.camera.output.device_generation &&
                    atomic_view.image.content_revision==atomic_view.camera.output.content_revision &&
                    atomic_view.camera.output.replacement_tile_count==ownership.size() &&
                    std::equal(ownership.begin(),ownership.end(),atomic_view.camera.output.replacement_tile_flags),
                    "atomic GPU view owns exact sampled frame, occurrence order, coverage and complete identity");
                auto adopted=atomic_view;
                verify_gpu(poll_view(ticket,&atomic_view)==C3X_RENDERER_RESULT_OK && !std::memcmp(&atomic_view,&adopted,sizeof(adopted)),"repeated atomic GPU view stable");
            }
            auto selected=pending;verify_gpu(poll(ticket,&pending,&pending_meta)==C3X_RENDERER_RESULT_OK && pending.ticket==selected.ticket && pending.map_image==selected.map_image,"repeated adoption stable");
            view=selected;verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK,"latest camera pixel oracle");auto selected_pixels=actual;
            future.tiles=frozen_tiles.data();std::vector<unsigned> reference;
            verify_gpu(capture_reference(future,reference) && selected_pixels==reference,"copied latest camera equals independent render");
            verify_gpu(gpu_render(&next,&view,&meta)==C3X_RENDERER_RESULT_OK,"restore resident camera after oracle");
            auto kept=view;future.presentation_time_ticks+=future.presentation_frequency;next.identity.visibility_epoch++;
            verify_gpu(begin(&next,&ticket)==C3X_RENDERER_RESULT_PENDING,"begin cancelled GPU camera");cancel(ticket);
            verify_gpu(poll(ticket,&pending,&pending_meta)==C3X_RENDERER_RESULT_SUPERSEDED,"cancelled GPU camera cannot publish");
            verify_gpu(read(kept.map_image)==C3X_RENDERER_RESULT_OK && actual==reference,"cancel preserves adopted front");
            verify_gpu(begin(&next,&ticket)==C3X_RENDERER_RESULT_PENDING &&
                gpu_render(&next,&view,&meta)==C3X_RENDERER_RESULT_OK &&
                poll(ticket,&pending,&pending_meta)==C3X_RENDERER_RESULT_OK && pending.ticket==view.ticket && pending.map_image==view.map_image,
                "synchronous GPU caller joins the queued request");
            future.presentation_time_ticks+=future.presentation_frequency;
            verify_gpu(begin(&next,&ticket)==C3X_RENDERER_RESULT_PENDING,"begin reset GPU camera");gpu_reset();
            verify_gpu(poll(ticket,&pending,&pending_meta)==C3X_RENDERER_RESULT_SUPERSEDED,"reset retires GPU camera ticket");
            if(poll_view){auto before=atomic_view;
                verify_gpu(poll_view(ticket,&atomic_view)==C3X_RENDERER_RESULT_SUPERSEDED && !std::memcmp(&before,&atomic_view,sizeof(before)),"reset cannot publish atomic GPU view");
                if(ok)std::puts("PASS atomic GPU camera view: copied_occurrences=1 complete_identity=1 stable_adoption=1 stale_writes=0 reset_retirement=1");
            }
            std::sort(submission_ms.begin(),submission_ms.end());double total=0;for(double ms:submission_ms)total+=ms;
            if(ok)std::printf("PASS replaceable GPU camera: requests=64 begin_mean_ms=%.3f begin_p95_ms=%.3f begin_max_ms=%.3f copied_inputs=1 stale_adoptions=0 retained_front=1\n",total/submission_ms.size(),submission_ms[60],submission_ms.back());
            // Oracle readback is explicit; restore clean producer counters for
            // the ordinary native contract and whole-workload measurements.
            if(!verify_gpu(render(&test_frame,&control)==C3X_RENDERER_RESULT_OK && capture_reference(test_frame,expected) &&
                gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK,"fresh session after camera oracle"))break;map_expected=expected;
        }
#include "gpu_camera_identity_preview.h"
#ifdef C3X_GPU_NATIVE_CONTRACT
        if(phase==0){char jgl_path[2048]={};GetEnvironmentVariableA("C3X_RENDERER_GPU_JGL_TEST",jgl_path,sizeof(jgl_path));
            std::vector<NativeFrameSample> performance_frames;
            char benchmark_option[8]={};GetEnvironmentVariableA("C3X_RENDERER_NATIVE_FRAME_BENCHMARK",benchmark_option,sizeof(benchmark_option));
            if(!std::strcmp(benchmark_option,"1")){
                int home_x=center_x,home_y=center_y;
                for(int workload=0;workload<3;++workload)for(int step=0;step<40;++step){
                    NativeFrameSample sample;sample.workload=workload;sample.step=step;sample.frame=test_frame;
                    center_x=home_x+(workload==1&&step>=8?(step-8)*2:0);center_y=home_y;
                    sample.tiles=capture_view();sample.frame.tile_count=unsigned(sample.tiles.size());
                    sample.frame.presentation_frequency=60000;sample.frame.presentation_time_ticks=600000+step*1000;
                    if(workload==2&&step>=16){auto changed=std::min_element(sample.tiles.begin(),sample.tiles.end(),[&](auto const& a,auto const& b){
                        auto distance=[&](auto const& t){return (t.tile_flags&C3X_RENDERER_TILE_RENDER)?std::abs(t.anchor_x-frame.target_width/2)+std::abs(t.anchor_y-frame.target_height/2):INT_MAX;};
                        return distance(a)<distance(b);});
                        if(changed!=sample.tiles.end())changed->terrain_type=(changed->terrain_type+1)%3;
                    }
                    performance_frames.push_back(std::move(sample));
                }
                center_x=home_x;center_y=home_y;
            }
            if(!verify_gpu(jgl_path[0]&&native_worker_contract(jgl_path,gpu_images,view,gpu_render,gpu_present,request,expected.data(),test_tiles[0].anchor_x,test_tiles[0].anchor_y,performance_frames),"actual native hooks on renderer worker"))break;
            // The independent native oracle intentionally reads back. Start a
            // fresh session for the existing producer/readback counters below.
            gpu_reset();
            if(!verify_gpu(render(&test_frame,&control)==C3X_RENDERER_RESULT_OK && capture_reference(test_frame,expected) &&
                gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK,"fresh session after native oracle"))break;map_expected=expected;
        }
#endif
        if(phase==0 && ok){
            // Every extent renders the demanded retained scene. No padded donor
            // or alternate zoom image changes the clock or coverage contract.
            auto refresh=test_frame;auto refresh_tiles=test_tiles;
            for(auto& tile:refresh_tiles)tile.anchor_x-=80;
            refresh.tiles=refresh_tiles.data();refresh.presentation_time_ticks+=refresh.presentation_frequency/15;
            auto refresh_request=request;refresh_request.frame=&refresh;
            auto prepare=reinterpret_cast<c3x_renderer_prepare_nearby_view_fn>(GetProcAddress(module,"c3x_renderer_prepare_nearby_view"));
            verify_gpu(capture_reference(refresh,expected) && gpu_render(&refresh_request,&view,&meta)==C3X_RENDERER_RESULT_OK &&
                !view.prepared && !view.map_readbacks && !meta.bgra_pixels && view.presentation_time_ticks==refresh.presentation_time_ticks,
                "exact resident camera demand");
            verify_gpu(prepare && prepare(&refresh_request)==C3X_RENDERER_RESULT_ERROR,"legacy optional image preparation declines");
            verify_gpu(read(view.map_image)==C3X_RENDERER_RESULT_OK && actual==expected,"exact camera pixels");
            if(ok)std::puts("PASS bounded GPU map demand: exact current camera and clock; no wider preparation at any extent");
            gpu_reset();verify_gpu(render(&test_frame,&control)==C3X_RENDERER_RESULT_OK && capture_reference(test_frame,expected) &&
                gpu_render(&request,&view,&meta)==C3X_RENDERER_RESULT_OK,"restore exact camera phase control");map_expected=expected;
        }
        if(old_ticket){auto stale=image_request(C3X_GPU_CREATE);stale.ticket=old_ticket;stale.width=stale.height=2;verify_gpu(execute(stale)==C3X_RENDERER_RESULT_SUPERSEDED,"old ticket rejected");}
        old_ticket=view.ticket;
        auto create=image_request(C3X_GPU_CREATE);create.width=view.width;create.height=view.height;
        if(!verify_gpu(execute(create)==C3X_RENDERER_RESULT_OK&&status.readbacks==0&&status.uploads==0,"resident map has no CPU upload/readback"))break;
        auto canvas=status.image;
        c3x_renderer_gpu_command_v1 draw={0,canvas,view.map_image,{0,0,view.width,view.height},{0,0,view.width,view.height},0,0,0};
        auto submit=image_request(C3X_GPU_SUBMIT);submit.commands=&draw;submit.command_count=1;submit.command_struct_size=sizeof(draw);
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
