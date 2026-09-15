// Included inside the connected native-screen fixture, after its pixel oracles.
// These are real captured map demands and actual JGL drawing/presentation hooks.
if(!performance_frames.empty()){
    auto cpu_blit=reinterpret_cast<c3x_renderer_blit_fn>(GetProcAddress(renderer_module,"c3x_renderer_blit"));
    auto cpu_present=reinterpret_cast<c3x_renderer_camera_present_view_fn>(GetProcAddress(renderer_module,"c3x_renderer_camera_present_view"));
    auto prepare_view=reinterpret_cast<c3x_renderer_prepare_nearby_view_fn>(GetProcAddress(renderer_module,"c3x_renderer_prepare_nearby_view"));
    verify(cpu_blit&&cpu_present&&prepare_view,"whole-frame publication exports");
    auto desktop_module=LoadLibraryA("dwmapi.dll");auto finish_desktop=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(desktop_module,"DwmFlush"));
    verify(finish_desktop!=nullptr,"whole-frame desktop completion boundary");
    LARGE_INTEGER counter_frequency={};QueryPerformanceFrequency(&counter_frequency);
    auto milliseconds=[&](long long ticks){return 1000.*double(ticks)/double(counter_frequency.QuadPart);};
    verify(SetWindowPos(window,HWND_TOPMOST,0,0,0,0,SWP_NOSIZE|SWP_NOACTIVATE)!=FALSE,"position whole-frame test window");
    std::printf("FRAME_SCOPE width=%d height=%d desktop_width=%d desktop_height=%d capture=outside_timing detail=full control=same_DLL_CPU map_units_UI_final_transfer=inside_timing\n",w,h,GetSystemMetrics(SM_CXSCREEN),GetSystemMetrics(SM_CYSCREEN));
    JGLSprite hud_color={},hud_alpha={},lookup_sprite={},shadow_sprite={};auto jgl_base=reinterpret_cast<char*>(jgl);
    for(auto sprite:{&hud_color,&hud_alpha,&lookup_sprite,&shadow_sprite}){reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(jgl_base+0x7e80)(sprite,nullptr);
        sprite->bit_count=8;sprite->width=sprite->stride=64;sprite->height=16;}
    std::vector<unsigned char> hud_indices(64*16),hud_weights(64*16);
    for(unsigned n=0;n<hud_indices.size();++n){unsigned hud_codes[4]={0,1,254,255},weights[5]={0,64,128,192,255};
        hud_indices[n]=static_cast<unsigned char>(hud_codes[n%4]);hud_weights[n]=static_cast<unsigned char>(weights[(n/4)%5]);}
    hud_color.bits=hud_indices.data();hud_alpha.bits=hud_weights.data();
    auto hud_palette=palette_create(graph,nullptr);verify(hud_palette!=nullptr,"whole-frame HUD palette");
    auto hud_table=*static_cast<void***>(hud_palette);
    auto hud_words=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(hud_table[6])(hud_palette);
    for(unsigned n=0;n<256;++n){hud_words[n]=0x0c85;
        auto rgb=reinterpret_cast<unsigned char*(__thiscall*)(void*,unsigned)>(hud_table[8])(hud_palette,n);rgb[0]=24;rgb[1]=32;rgb[2]=40;}
    std::vector<unsigned short> lookup_table(31*32768);
    for(unsigned block=0;block<31;++block)for(unsigned word=0;word<32768;++word){
        unsigned b=(word&31)*std::min(block+1,16u)/16,g=((word>>5)&31)*std::min(block+1,16u)/16,r=((word>>10)&31)*std::min(block+1,16u)/16;
        lookup_table[block*32768+word]=static_cast<unsigned short>(b|(g<<5)|(r<<10));}
    std::vector<unsigned char> lookup_indices(64*16);for(unsigned n=0;n<lookup_indices.size();++n)lookup_indices[n]=static_cast<unsigned char>(n%17);
    lookup_sprite.bits=lookup_indices.data();
    std::vector<unsigned char> shadow_indices(64*16);for(unsigned n=0;n<shadow_indices.size();++n)shadow_indices[n]=static_cast<unsigned char>(248+n%4);
    shadow_sprite.bits=shadow_indices.data();std::vector<unsigned short> map_shadow_table(4*32768);
    for(unsigned block=0;block<4;++block)for(unsigned word=0;word<32768;++word){
        unsigned b=(word&31)*(block+1)/4,g=((word>>5)&31)*(block+1)/4,r=((word>>10)&31)*(block+1)/4;
        map_shadow_table[block*32768+word]=static_cast<unsigned short>(b|(g<<5)|(r<<10));}

    std::puts("FRAME_SCOPE native_HUD_alpha_slots=20,21,22 source_preparation=outside_timing composition=inside_timing native_label_panels=8 native_label_borders=8 native_lookup_panel=1 native_lookup_sprite=1 native_FLC_lookup=1 native_scaled_FLC_cursor=1 native_single_key=1 native_solid_mask=1 native_map_shadow=1 native_opacity_transition=1");
    for(int workload=0;workload<3;++workload)for(int block=0;block<4;++block){
        bool resident=block==1||block==2; // CPU / GPU / GPU / CPU balances order.
        reset();state.custom_renderer_native_image=resident?live:nullptr;
        verify(screen_surface->Bits_Data_Links==0,"whole-frame entry has no outstanding native bits");
        SelectObject(label_dc,previous_font);
        for(auto image:live_images){
            verify(reinterpret_cast<Init>(image->vtable[1])(image,w,h,16,1)==0,"fresh whole-frame surface lifetime");
            verify(reinterpret_cast<Fill>(image->vtable[17])(image,&full,int(0x80000000u))==0,"whole-frame initial surface");
        }
        label_dc=*reinterpret_cast<HDC*>(reinterpret_cast<char*>(screen_surface)+0x4bc);previous_font=SelectObject(label_dc,label_font);
        SetTextColor(label_dc,RGB(237,171,55));SetBkMode(label_dc,TRANSPARENT);SetTextAlign(label_dc,TA_LEFT|TA_TOP);
        for(auto const& captured:performance_frames)if(captured.workload==workload){
            auto requested=captured.frame;requested.tiles=captured.tiles.data();
            c3x_renderer_camera_request_v1 packet={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(packet),&requested,{21,22,23,24}};
            c3x_renderer_camera_view_v1 displayed={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(displayed)};
            LARGE_INTEGER begin={},map_done={},units_done={},submit_done={},end={};QueryPerformanceCounter(&begin);
            if(resident){
                verify(native_map_view(C3X_NATIVE_MAP_PREPARE,live_images[0],&packet,&displayed)==C3X_RENDERER_RESULT_OK,"timed resident map demand");
                verify(native_map_view(C3X_NATIVE_MAP_COMMIT,live_images[0],nullptr,nullptr)==C3X_RENDERER_RESULT_OK,"timed resident map commit");
            }else{
                if(cpu_present(&packet,&displayed)!=C3X_RENDERER_RESULT_OK){
                    displayed.frame=requested;displayed.identity=packet.identity;displayed.output={C3X_RENDERER_API_VERSION,sizeof(displayed.output)};
                    verify(render_view(&packet,&displayed.output)==C3X_RENDERER_RESULT_OK,"timed CPU map demand");
                }
                auto map_dc=reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(live_images[0]->vtable[10])(live_images[0]);
                verify(map_dc&&cpu_blit(&displayed.output,map_dc)==C3X_RENDERER_RESULT_OK,"timed CPU map insertion");
                reinterpret_cast<Release>(live_images[0]->vtable[11])(live_images[0],1);
            }
            copy(live_images[0],screen_surface,full);QueryPerformanceCounter(&map_done);
            HDC body_dc=nullptr;
            if(!resident)body_dc=reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(screen_surface->vtable[10])(screen_surface);
            for(int n=0;n<8;++n){auto actor=unit;actor.unit_id=800+n;actor.direction=1+n;actor.action_cursor=(captured.step/4+n)%16;
                actor.presentation_frequency=requested.presentation_frequency;actor.presentation_time_ticks=requested.presentation_time_ticks;
                actor.body_x=(n%4)*w/4;actor.body_y=(n/4)*h/2;int bounds[4]={};
                if(resident){int result=live(C3X_NATIVE_UNIT_DRAW,screen_surface,screen_surface,&actor,bounds,0);
                    if(result!=1)std::fprintf(stderr,"FRAME_UNIT_FAILURE workload=%d block=%d step=%d actor=%d result=%d lifetime=%d leases=%d,%d\n",workload,block,captured.step,n,result,
                        state.custom_renderer_native_lifetime(C3X_NATIVE_MAP,screen_surface,0),
                        *reinterpret_cast<int*>(reinterpret_cast<char*>(screen_surface)+0x4c4),*reinterpret_cast<int*>(reinterpret_cast<char*>(screen_surface)+0x4c8));
                    verify(result==1,"timed resident unit");}
                else verify(unit_cpu(&actor,body_dc,body_dc,bounds)==C3X_RENDERER_RESULT_OK,"timed CPU unit");
            }
            if(!resident)reinterpret_cast<Release>(screen_surface->vtable[11])(screen_surface,1);
            verify(screen_surface->Bits_Data_Links==0,"whole-frame units release native bits");
            QueryPerformanceCounter(&units_done);
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*)>(hud_color.vtable[20])(&hud_color,&hud_alpha,live_images[0],screen_surface,5,h-70,hud_palette)==0,"timed HUD explicit-background blend");
            for(int slot:{21,22})verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGLSprite*,JGL_Image*,int,int,void*)>(hud_color.vtable[slot])(&hud_color,&hud_alpha,screen_surface,80+(slot-21)*75,h-70,hud_palette)==0,"timed HUD destination blend");
            verify(screen_surface->Bits_Data_Links==0,"whole-frame HUD releases native bits");
            RECT hud={0,h-38,w,h};verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&hud,int(0x80001234u))==0,"timed native HUD fill");
            for(int n=0;n<8;++n){int x=n*w/8;RECT label_area={x,h-112,x+w/8-2,h-88};
                verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,RECT*,int,int)>(screen_surface->vtable[18])(screen_surface,&label_area,int(0x80000000u),50)==0,"timed translucent map label panel");
                auto line=reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,int,int,int,int)>(screen_surface->vtable[25]);
                int edge=int(0x80002defu),right=label_area.right-1,bottom=label_area.bottom-1;
                verify(line(screen_surface,x,label_area.top,right,label_area.top,edge,1)==0&&line(screen_surface,x,bottom,right,bottom,edge,1)==0&&
                    line(screen_surface,x,label_area.top,x,bottom,edge,1)==0&&line(screen_surface,right,label_area.top,right,bottom,edge,1)==0,"timed map label border");
                verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,char const*,int)>(screen_surface->vtable[46])(screen_surface,x+3,h-109,"Berlin: 6",9)==0,"timed native labels");
            }
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*)>(hud_color.vtable[23])(&hud_color,screen_surface,480,h-70,hud_palette)==0,"timed native single-key UI");
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,int,void*)>(hud_color.vtable[29])(&hud_color,screen_surface,550,h-70,17,hud_palette)==0,"timed native solid sprite mask");
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*,void*)>(shadow_sprite.vtable[31])(&shadow_sprite,screen_surface,w-140,h-200,map_shadow_table.data(),hud_palette)==0,"timed native map shadow");
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,float,void*,int)>(hud_color.vtable[37])(&hud_color,screen_surface,480,h-90,0.625f,hud_palette,0)==0,"timed command-panel opacity transition");
            RECT lookup_area={w-160,h-180,w-16,h-120};
            verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,RECT*,JGL_Image*,int,void*)>(screen_surface->vtable[21])(screen_surface,&lookup_area,live_images[0],40,lookup_table.data())==0,"timed native lookup panel");
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*,void*)>(lookup_sprite.vtable[33])(&lookup_sprite,screen_surface,260,h-70,lookup_table.data(),hud_palette)==0,"timed native lookup sprite");
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,JGL_Image*,int,int,void*,void*)>(hud_color.vtable[35])(&hud_color,live_images[0],screen_surface,335,h-70,lookup_table.data(),hud_palette)==0,"timed native FLC lookup body and shadow");
            auto native_scales=reinterpret_cast<int*>(jgl_base+0x6c0fc);int saved_lookup_scales[3]={native_scales[0],native_scales[1],native_scales[2]};
            native_scales[0]=native_scales[1]=1;native_scales[2]=2;
            verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,JGL_Image*,int,int,int,int,int,void*,void*)>(hud_color.vtable[34])(&hud_color,live_images[0],screen_surface,410,h-70,1,1,2,lookup_table.data(),hud_palette)==0,"timed native scaled FLC cursor and shadow");
            for(unsigned i=0;i<3;++i)native_scales[i]=saved_lookup_scales[i];
            verify(screen_surface->Bits_Data_Links==0&&live_images[0]->Bits_Data_Links==0,"whole-frame lookup releases native bits");
            last_transfer=full;final_ui_drawn=false;patch_JGL_present_screen(&full);GdiFlush();
            verify(screen_surface->Bits_Data_Links==0,"whole-frame final transfer releases native bits");
            int preparation=prepare_view(&packet);verify(preparation==C3X_RENDERER_RESULT_OK||preparation==C3X_RENDERER_RESULT_BAD_ARGUMENT,"caller-driven preparation offer");
            QueryPerformanceCounter(&submit_done);verify(SUCCEEDED(finish_desktop()),"whole-frame desktop completion");QueryPerformanceCounter(&end);
            if(captured.step>=8){auto const& output=displayed.output;
                double age=1000.*double(requested.presentation_time_ticks-displayed.frame.presentation_time_ticks)/double(requested.presentation_frequency);
                std::printf("FRAME_SAMPLE workload=%d block=%d route=%s step=%d request_ms=%.3f desktop_ms=%.3f map_ms=%.3f units_ms=%.3f UI_present_prepare_ms=%.3f sample_age_ms=%.3f builds=%u reused=%u upload_bytes=%u tiles=%u\n",
                    workload,block,resident?"GPU":"CPU",captured.step,milliseconds(submit_done.QuadPart-begin.QuadPart),milliseconds(end.QuadPart-begin.QuadPart),
                    milliseconds(map_done.QuadPart-begin.QuadPart),milliseconds(units_done.QuadPart-map_done.QuadPart),milliseconds(submit_done.QuadPart-units_done.QuadPart),
                    age,output.geometry_tiles_built,output.geometry_tiles_reused,output.geometry_upload_bytes,requested.tile_count);
            }
        }
        if(resident){
            for(auto image:live_images){auto words=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(original_bits)(image);
                int pitch=*reinterpret_cast<int*>(reinterpret_cast<char*>(image)+0x40);bool untouched=words!=nullptr;
                if(words)for(int y=0;y<h;++y)for(int x=0;x<w;++x)untouched=untouched&&words[y*pitch+x]==0;
                reinterpret_cast<Release>(original_release)(image,1);verify(untouched,"timed resident map/screen never restored CPU pixels");}
            std::printf("FRAME_RESIDENCY workload=%d block=%d native_CPU_map_and_screen_untouched=1\n",workload,block);
        }
    }
    reset();state.custom_renderer_native_image=live;live_active=false;FreeLibrary(desktop_module);
    for(auto sprite:{&hud_color,&hud_alpha,&lookup_sprite,&shadow_sprite}){sprite->bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(jgl_base+0x7ed0)(sprite);}
    reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(jgl_base+0x3cf10)(hud_palette,1);
    std::puts("PASS whole native frame comparison: stationary animation, fresh dense scrolling, local change; 8 warmup demands and 32 measured demands per arm/block; caller and desktop boundary plus sample age");
}
