// Included by the existing scene harness. This records renderer inputs and
// independently reads output witnesses; no renderer pixels are fed into replay.
char input_contract[16]={};
if(GetEnvironmentVariableA("C3X_RENDERER_INPUT_CONTRACT",input_contract,sizeof(input_contract))){
    auto gpu_render=reinterpret_cast<c3x_renderer_gpu_render_fn>(GetProcAddress(module,"c3x_renderer_gpu_render"));
    auto gpu_images=reinterpret_cast<c3x_renderer_gpu_images_fn>(GetProcAddress(module,"c3x_renderer_gpu_images"));
    auto gpu_unit=reinterpret_cast<c3x_renderer_gpu_unit_fn>(GetProcAddress(module,"c3x_renderer_gpu_unit"));
    auto gpu_present=reinterpret_cast<c3x_renderer_gpu_present_fn>(GetProcAddress(module,"c3x_renderer_gpu_present"));
    auto input_camera_begin=reinterpret_cast<c3x_renderer_gpu_camera_begin_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_begin"));
    auto input_camera_poll=reinterpret_cast<c3x_renderer_gpu_camera_poll_fn>(GetProcAddress(module,"c3x_renderer_gpu_camera_poll"));
    auto visual=reinterpret_cast<int(*)()>(GetProcAddress(module,"c3x_renderer_gpu_visual_frame"));
    auto clock=reinterpret_cast<c3x_renderer_visual_clock_fn>(GetProcAddress(module,"c3x_renderer_visual_clock"));
    auto finish=reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_input_recording_finish"));
    if(!gpu_render||!gpu_images||!gpu_present||!gpu_unit||!finish||!visual||!clock||!input_camera_begin||!input_camera_poll)return 1;
    auto check=[](bool good,char const* label){if(!good){std::fprintf(stderr,"INPUT_CONTRACT_FAIL %s\n",label);throw std::runtime_error(label);}};
    WNDCLASSW wc={};wc.lpfnWndProc=DefWindowProcW;wc.hInstance=GetModuleHandleW(nullptr);wc.lpszClassName=L"C3XInputCaptureContract";RegisterClassW(&wc);
    HWND window=CreateWindowW(wc.lpszClassName,L"C3X input capture contract",WS_POPUP,0,0,frame.target_width,frame.target_height,nullptr,nullptr,wc.hInstance,nullptr);
    try{
        check(window!=nullptr,"window");int steps=std::clamp(std::atoi(input_contract),1,1200);
        auto input=frame;auto owned_tiles=tiles;input.tiles=owned_tiles.data();
        c3x_renderer_gpu_frame_v1 view={sizeof(view)};c3x_renderer_i64 packed=0,detail=0;
        c3x_renderer_gpu_result_v1 input_status={sizeof(input_status)};
        auto run=[&](c3x_renderer_gpu_images_v1 const& command){check(gpu_images(&command,&input_status,nullptr,0)==1,"image input");};
        auto make=[&](int format){c3x_renderer_gpu_images_v1 value={sizeof(value)};value.action=C3X_GPU_CREATE;value.ticket=view.ticket;value.width=input.target_width;value.height=input.target_height;value.format=format;run(value);return input_status.image;};
        std::vector<unsigned> pixels(std::size_t(input.target_width)*input.target_height);
        for(int step=0;step<steps;++step){
            check(clock()>=0,"exported live visual clock");
            input.presentation_time_ticks=frame.presentation_time_ticks+std::int64_t(step)*frame.presentation_frequency/10;
            if(step&&step%10==0)for(auto& tile:owned_tiles)tile.anchor_x-=8;
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&input,{1,2,3,4}};
            c3x_renderer_output_v1 meta={C3X_RENDERER_API_VERSION,sizeof(meta)};
            if(step%3==1){c3x_renderer_i64 ticket=0;check(input_camera_begin(&request,&ticket)==4,"camera input");auto deadline=GetTickCount64()+60000;int code=4;
                while(code==4&&GetTickCount64()<deadline){code=input_camera_poll(ticket,&view,&meta);if(code==4)Sleep(1);}check(code==1,"camera adoption");}
            else check(gpu_render(&request,&view,&meta)==1,"map input");
            if(!packed){packed=make(C3X_GPU_RGB555);detail=make(C3X_GPU_BGRA32);}
            c3x_renderer_gpu_command_v1 commands[3]={};
            for(auto& cmd:commands){cmd.area[2]=cmd.clip[2]=input.target_width;cmd.area[3]=cmd.clip[3]=input.target_height;}
            commands[0].kind=4;commands[0].destination=packed;commands[0].source=view.map_image;
            commands[1].kind=0;commands[1].destination=detail;commands[1].source=view.map_image;
            commands[2].kind=1;commands[2].destination=detail;commands[2].area[1]=input.target_height-12;commands[2].color=0xff204060u+unsigned(step);
            c3x_renderer_gpu_images_v1 submit={sizeof(submit)};submit.action=C3X_GPU_SUBMIT;submit.ticket=view.ticket;submit.commands=commands;submit.command_count=3;submit.command_struct_size=sizeof(commands[0]);run(submit);
            c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);unit.unit_id=41;unit.action=1;unit.direction=2;unit.frame_count=16;
            unit.body_x=input.target_width/2-95;unit.body_y=input.target_height/2-95;unit.sprite_width=unit.sprite_height=191;unit.projection_scale_milli=1000;
            unit.hour=input.hour;unit.season=input.season;unit.presentation_time_ticks=input.presentation_time_ticks;unit.presentation_frequency=input.presentation_frequency;
            std::memcpy(unit.unit_key,"PRTO_Warrior",13);unit.display_color_rgb=0x205bdd;
            c3x_renderer_gpu_unit_v1 dest={sizeof(dest),view.ticket,packed,packed,detail,detail,{0,0,input.target_width,input.target_height},3};int bounds[4]={};
            check(gpu_unit(&unit,&dest,bounds)==1,"unit input");
            if(step==0){
                auto cpu_draw=reinterpret_cast<c3x_renderer_unit_draw_playback_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_playback"));check(cpu_draw!=nullptr,"CPU unit export");
                struct InputCanvas {
                    HDC dc=nullptr;HBITMAP bitmap=nullptr;void* data=nullptr;int pitch=0,depth=0;
                    InputCanvas(int w,int h,int bits):depth(bits){struct Info{BITMAPINFOHEADER head;DWORD masks[3];} info={};
                        info.head.biSize=sizeof(info.head);info.head.biWidth=w;info.head.biHeight=-h;info.head.biPlanes=1;info.head.biBitCount=WORD(bits);info.head.biCompression=bits==16?BI_BITFIELDS:BI_RGB;
                        info.masks[0]=0x7c00;info.masks[1]=0x3e0;info.masks[2]=0x1f;dc=CreateCompatibleDC(nullptr);
                        bitmap=CreateDIBSection(dc,reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS,&data,nullptr,0);pitch=((w*bits+31)/32)*4;
                        if(bitmap&&data){SelectObject(dc,bitmap);std::memset(data,0,std::size_t(pitch)*unsigned(h));}}
                    ~InputCanvas(){if(dc)DeleteDC(dc);if(bitmap)DeleteObject(bitmap);}
                };
                for(int depth:{16,32}){InputCanvas cpu_destination(input.target_width,input.target_height,depth),cpu_background(input.target_width,input.target_height,depth);
                    check(cpu_destination.data&&cpu_background.data,"CPU canvas allocation");IntersectClipRect(cpu_destination.dc,7,9,input.target_width-11,input.target_height-13);
                    for(unsigned iteration=0;iteration<3;++iteration){
                        RECT stripe={30+int(iteration)*7,20,80+int(iteration)*7,50};auto brush=CreateSolidBrush(RGB(20,90+iteration*20,120));FillRect(cpu_background.dc,&stripe,brush);DeleteObject(brush);
                        int cpu_bounds[4]={};auto actor=unit;actor.unit_id=91+depth;actor.presentation_time_ticks+=iteration*actor.presentation_frequency/10;
                        // Serial callers on different threads share one canvas
                        // identity domain. Flush the actual GDI writer first.
                        GdiFlush();int drawn=0;
                        if(iteration==1||(depth==32&&iteration==0)){std::thread other([&]{drawn=cpu_draw(&actor,cpu_destination.dc,cpu_background.dc,cpu_bounds,3);});other.join();}
                        else drawn=cpu_draw(&actor,cpu_destination.dc,cpu_background.dc,cpu_bounds,3);
                        check(drawn==1,"CPU compatibility unit and native edits across caller threads");}
                }
            }
            c3x_renderer_gpu_present_v1 present={sizeof(present),0,view.ticket,detail,window,input.target_width,input.target_height,{0,0,input.target_width,input.target_height}};
            check(gpu_present(&present)==1,"presentation input");
            Sleep(20);auto code=visual();check(code==1||code==4,"ambient input");
            c3x_renderer_gpu_images_v1 read={sizeof(read)};read.action=C3X_GPU_READBACK;read.ticket=view.ticket;read.image=detail;read.pixel_count=unsigned(pixels.size());
            check(gpu_images(&read,&input_status,pixels.data(),unsigned(pixels.size()))==1,"independent frame witness");
        }
        // The production presenter rejects calls from a nonowning thread.
        // Replay must execute that ownership check, not assume every caller owns it.
        int wrong_thread=0;std::thread nonowner([&]{c3x_renderer_gpu_present_v1 release={sizeof(release)};release.action=1;wrong_thread=gpu_present(&release);});nonowner.join();
        check(wrong_thread==C3X_RENDERER_RESULT_BAD_ARGUMENT,"nonowner presentation rejection");
        // CPU camera fallback is a production input path as well. Exercise both
        // legacy pixels and the atomic description without changing GPU exports.
        auto cpu_begin=reinterpret_cast<c3x_renderer_camera_begin_fn>(GetProcAddress(module,"c3x_renderer_camera_begin"));
        auto cpu_poll=reinterpret_cast<c3x_renderer_camera_poll_fn>(GetProcAddress(module,"c3x_renderer_camera_poll"));
        auto cpu_begin_view=reinterpret_cast<c3x_renderer_camera_begin_view_fn>(GetProcAddress(module,"c3x_renderer_camera_begin_view"));
        auto cpu_poll_view=reinterpret_cast<c3x_renderer_camera_poll_view_fn>(GetProcAddress(module,"c3x_renderer_camera_poll_view"));
        auto world_capture=reinterpret_cast<c3x_renderer_set_world_capture_fn>(GetProcAddress(module,"c3x_renderer_set_world_capture"));
        check(cpu_begin&&cpu_poll&&cpu_begin_view&&cpu_poll_view&&world_capture,"CPU camera and world lifecycle exports");
        check(world_capture(+[](c3x_renderer_world_page_v1*)->int{return C3X_RENDERER_RESULT_PENDING;})==1,"enable world input");
        for(unsigned atomic=0;atomic<2;++atomic){
            c3x_renderer_i64 ticket=0;input.presentation_time_ticks+=input.presentation_frequency/10;
            c3x_renderer_camera_request_v1 request={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(request),&input,{1,2,3,4}};
            check((atomic?cpu_begin_view(&request,&ticket):cpu_begin(&input,&ticket))==4,"CPU camera begin");
            int code=4;auto deadline=GetTickCount64()+60000;
            while(code==4&&GetTickCount64()<deadline){
                c3x_renderer_camera_view_v1 cpu_view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(cpu_view)};
                c3x_renderer_output_v1 cpu_pixels={C3X_RENDERER_API_VERSION,sizeof(cpu_pixels)};
                code=atomic?cpu_poll_view(ticket,&cpu_view):cpu_poll(ticket,&cpu_pixels);if(code==4)Sleep(1);
            }check(code==1,"CPU camera adoption");
        }
        check(world_capture(nullptr)==1,"disable world input");
        reset();check(clock()==0,"retired visual clock after reset");finish();DestroyWindow(window);std::puts("PASS production input capture: map, unit, composition, clocks, present, output witnesses and reset");return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"%s\n",e.what());reset();finish();if(window)DestroyWindow(window);return 1;}
}
