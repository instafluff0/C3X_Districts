// Included only in the real-renderer/native-hook fixture.
#include <thread>
#include "native_screen_bridge.h"
WorkerClient* screen_client=nullptr;
c3x_native_images::Adapter<WorkerClient>* screen_adapter=nullptr;
c3x_renderer_gpu_present_fn screen_present=nullptr;
c3x_renderer_gpu_frame_v1 screen_frame={};
void* screen_graph=nullptr;
JGL_Image* screen_image=nullptr;
unsigned screen_transfers=0;
bool final_ui_drawn=false;

int present_native_image(void* image,void* graph,void const* rect){
    c3x_renderer_gpu_present_v1 request={sizeof(request)};
    auto id=screen_adapter->display_image(image);
    if(!id){request.action=1;verify(screen_present(&request)==C3X_RENDERER_RESULT_OK,"release GPU window before native fallback");return 0;}
    verify(final_ui_drawn,"final native UI precedes GPU transfer");
    auto dc=*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138);
    request.window=WindowFromDC(dc);request.ticket=screen_frame.ticket;request.image=std::int64_t(id);
    request.width=screen_frame.width;request.height=screen_frame.height;
    RECT area=rect?*static_cast<RECT const*>(rect):RECT{0,0,request.width,request.height};
    request.area[0]=area.left;request.area[1]=area.top;request.area[2]=area.right;request.area[3]=area.bottom;
    screen_client->flush();verify(screen_present(&request)==C3X_RENDERER_RESULT_OK,"native GPU final transfer");++screen_transfers;return 1;
}
void __cdecl complete_native_ui(RECT* area){
    RECT button={7,9,31,27};
    verify(reinterpret_cast<Fill>(screen_image->vtable[17])(screen_image,&button,int(0x800003e0u))==0,"final native button");
    final_ui_drawn=true;
    auto table=*static_cast<void***>(screen_graph);
    verify(reinterpret_cast<int(__thiscall*)(void*,RECT*)>(table[41])(screen_graph,area)==0,"native final transfer return");
}
LRESULT CALLBACK screen_window_proc(HWND window,UINT message,WPARAM w,LPARAM l){return DefWindowProcA(window,message,w,l);}
struct PreparedScreenPalette {void** vtable;void* native;};
int __fastcall prepared_screen_palette(PreparedScreenPalette*,int){return 0;}
unsigned short* __fastcall snapshot_test_bits(void* image,int){return *reinterpret_cast<unsigned short**>(static_cast<char*>(image)+0x4c0);}
void __fastcall snapshot_test_release(void*,int,int){}
bool native_screen_contract(char const* path,WorkerClient& gpu,c3x_renderer_gpu_frame_v1 const& frame,
                            c3x_renderer_gpu_present_fn present,unsigned const* map,int phase_x,int phase_y,c3x_renderer_native_image_fn live,c3x_renderer_render_view_fn render_view,c3x_renderer_camera_request_v1 const& demand,void (*reset)()){
    state={};capture={};events.clear();lines.clear();
    SetProcessDPIAware();WNDCLASSA wc={};wc.lpfnWndProc=screen_window_proc;wc.hInstance=GetModuleHandleA(nullptr);wc.lpszClassName="C3XNativeTransferContract";
    verify(RegisterClassA(&wc)!=0,"register native test window");
    HWND window=CreateWindowExA(WS_EX_TOPMOST|WS_EX_TOOLWINDOW,wc.lpszClassName,"Native transfer contract",WS_POPUP,20,20,frame.width,frame.height,nullptr,nullptr,wc.hInstance,nullptr);
    verify(window!=nullptr,"create native-owned test window");ShowWindow(window,SW_SHOWNOACTIVATE);UpdateWindow(window);
    auto dc=GetDC(window);HMODULE jgl=LoadLibraryA(path);verify(jgl&&dc,"native screen fixture");
    auto graph=reinterpret_cast<void*(__cdecl*)()>(GetProcAddress(jgl,"get_graphsy_object_ptr"))();auto gt=*reinterpret_cast<void***>(graph);
    // The isolated factory does not install the screen palettes. Use audited
    // native creation/activation, as the game does, before its original transfer.
    auto palette_create=reinterpret_cast<void*(__thiscall*)(void*,void*)>(gt[30]);
    void* palettes[2]={palette_create(graph,nullptr),palette_create(graph,nullptr)};
    // Graphsy takes Civ III palette-owner interfaces (native table at +4),
    // not bare JGL tables. Their initialization slot is already satisfied here.
    void* palette_vtable[1]={reinterpret_cast<void*>(prepared_screen_palette)};
    PreparedScreenPalette palette_owners[2]={{palette_vtable,palettes[0]},{palette_vtable,palettes[1]}};
    verify(palettes[0]&&palettes[1]&&reinterpret_cast<int(__thiscall*)(void*,void*,void*)>(gt[48])(graph,&palette_owners[0],&palette_owners[1])==0,"native screen palettes");
    auto create=reinterpret_cast<Create>(gt[31]);auto root=create(graph,nullptr,1);
    int w=frame.width,h=frame.height;RECT full={0,0,w,h};
    verify(reinterpret_cast<Init>(root->vtable[1])(root,w,h,16,1)==0,"screen root init");
    c3x_native_images::Adapter<WorkerClient> owner(gpu,root->vtable[4],root->vtable[9]);adapter=&owner;
    screen_adapter=&owner;screen_client=&gpu;screen_present=present;screen_frame=frame;screen_graph=graph;
    capture.write=log_line;state.custom_renderer_native_observe=observe;set_custom_renderer_native_probe(root);
    verify(state.custom_renderer_native_probe_active,"attach actual image and Graphsy final hooks");state.custom_renderer_native_image=translate;
    native_present_image=present_native_image;present_fn=complete_native_ui;
    JGL_Image* canvases[3];
    for(auto& canvas:canvases){canvas=create(graph,nullptr,1);verify(reinterpret_cast<Init>(canvas->vtable[1])(canvas,w,h,16,1)==0,"fresh native surface");
        verify(reinterpret_cast<Fill>(canvas->vtable[17])(canvas,&full,int(0x80000000u))==0,"clear native surface");}
    auto scene=canvases[0],screen_surface=canvases[1],save=canvases[2];screen_image=screen_surface;screen.JGL.Image=screen_surface;
    struct NativePcx {void* unused;JGL_Image* image;} pcx={nullptr,screen_surface};
    auto old_screen=*reinterpret_cast<void**>(static_cast<char*>(graph)+0x148);auto old_dc=*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138);
    *reinterpret_cast<void**>(static_cast<char*>(graph)+0x148)=&pcx;*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138)=dc;
    verify(owner.insert_map(scene,Id(frame.map_image),{0,0,w,h},0,0,phase_x,phase_y),"full-color map native insertion");
    auto copy=[&](JGL_Image* from,JGL_Image* to,RECT area){verify(reinterpret_cast<Copy>(from->vtable[16])(from,to,&area,&area)==0,"native family copy");};
    copy(scene,screen_surface,full);
    RECT popup={43,47,121,113};copy(screen_surface,save,popup);
    verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&popup,int(0x80007fffu))==0,"native popup draw");copy(save,screen_surface,popup);
    std::vector<unsigned> expected(map,map+std::size_t(w)*h),observed(expected.size());
    // Source-index transparency must preserve full-color map pixels. Matching
    // RGB at ordinary indices remains opaque, regardless of magenta/green keys.
    JGLSprite overlay={};auto base=reinterpret_cast<char*>(jgl);
    reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(base+0x7e80)(&overlay,nullptr);
    unsigned char indices[8]={0,254,1,255,255,1,254,0};
    overlay.bit_count=8;overlay.bits=indices;overlay.stride=4;overlay.width=4;overlay.height=2;
    auto palette_table=*static_cast<void***>(palettes[0]);
    auto colors=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(palette_table[6])(palettes[0]);
    verify(colors!=nullptr,"native overlay palette");colors[0]=0x7c1f;colors[1]=0x3e0;
    verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*)>(overlay.vtable[17])(&overlay,screen_surface,33,29,palettes[0])==0,"indexed native overlay on full-color map");
    for(int y=0;y<2;++y)for(int x=0;x<4;++x)if(indices[y*4+x]<254)expected[(y+29)*w+x+33]=indices[y*4+x]?0xff00ff00u:0xffff00ffu;
    overlay.bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base+0x7ed0)(&overlay);
    for(int y=9;y<27;++y)for(int x=7;x<31;++x)expected[y*w+x]=0xff00ff00u;
    final_ui_drawn=false;patch_JGL_present_screen(&full);
    verify(*reinterpret_cast<void**>(reinterpret_cast<char*>(screen_surface)+0x7c)==palettes[0],"GPU transfer preserves native screen palette binding");
    c3x_renderer_gpu_present_v1 foreign_release={sizeof(foreign_release)};foreign_release.action=1;int foreign_result=0;
    std::thread foreign_present([&]{foreign_result=present(&foreign_release);});foreign_present.join();
    verify(foreign_result==C3X_RENDERER_RESULT_BAD_ARGUMENT,"window ownership cannot be released by a foreign thread");
    verify(screen_transfers==1&&owner.stats().readbacks==0,"one complete GPU transfer with no native map readback");
    verify(gpu.readback(owner.display_image(screen_surface),observed.data(),observed.size())&&observed==expected,"map/screen/save-restore family preserves full color exactly");
    RECT last_transfer=full;bool live_active=false,preserve_gdi_display=false;unsigned capture_number=0;
    auto capture_display=[&](std::vector<unsigned> const& pixels){
        ++capture_number;
        auto dwm=LoadLibraryA("dwmapi.dll");verify(dwm!=nullptr,"desktop completion oracle");auto flush=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(dwm,"DwmFlush"));
        verify(flush!=nullptr,"desktop completion function");
        int sw=GetSystemMetrics(SM_CXSCREEN),sh=GetSystemMetrics(SM_CYSCREEN);verify(sw>0&&sh>0,"desktop extent");
        BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=w;info.bmiHeader.biHeight=-h;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
        auto desktop=GetDC(nullptr),capture_dc=CreateCompatibleDC(desktop);void* bits=nullptr;auto bitmap=CreateDIBSection(capture_dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
        verify(bitmap&&bits,"display oracle allocation");auto old=SelectObject(capture_dc,bitmap);
        std::vector<unsigned char> seen(pixels.size(),0);std::size_t differences=0;
        // Move only this test window when it exceeds the desktop. Explicitly
        // replay the last transfer after exposure; no timer/sleep substitutes
        // for a native request or compositor completion. Every pixel is checked.
        for(int y=0;y<h;y+=sh)for(int x=0;x<w;x+=sw){
            int sx=std::min(x,std::max(0,w-sw)),sy=std::min(y,std::max(0,h-sh));
            verify(SetWindowPos(window,HWND_TOPMOST,-sx,-sy,0,0,SWP_NOSIZE|SWP_NOACTIVATE)!=FALSE,"expose test-window region");
            if(live_active){
                verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&last_transfer,nullptr,0)==1,"live native-window exposure transfer");
            }else if(owner.owns(screen_surface)){
                c3x_renderer_gpu_present_v1 redraw={sizeof(redraw)};redraw.ticket=frame.ticket;redraw.image=std::int64_t(owner.display_image(screen_surface));
                redraw.window=window;redraw.width=w;redraw.height=h;redraw.area[0]=last_transfer.left;redraw.area[1]=last_transfer.top;redraw.area[2]=last_transfer.right;redraw.area[3]=last_transfer.bottom;
                verify(present(&redraw)==C3X_RENDERER_RESULT_OK,"native-window exposure redraw");
            }else if(!preserve_gdi_display)verify(reinterpret_cast<int(__thiscall*)(void*,RECT*)>(state.custom_renderer_jgl_present_original)(graph,&full)==0,"native GDI exposure redraw");
            verify(SUCCEEDED(flush()),"desktop composition completion");
            int cw=std::min(w-sx,sw),ch=std::min(h-sy,sh);
            verify(BitBlt(capture_dc,0,0,cw,ch,desktop,0,0,SRCCOPY)!=FALSE,"capture native final display");GdiFlush();
            for(int yy=0;yy<ch;++yy)for(int xx=0;xx<cw;++xx){auto i=std::size_t(yy+sy)*w+xx+sx;seen[i]=1;
                if((static_cast<unsigned*>(bits)[std::size_t(yy)*w+xx]&0xffffff)!=(pixels[i]&0xffffff)){
                    if(!differences)std::fprintf(stderr,"display first mismatch capture=%u live=%u x=%d y=%d expected=%08x actual=%08x\n",capture_number,unsigned(live_active),xx+sx,yy+sy,pixels[i],static_cast<unsigned*>(bits)[std::size_t(yy)*w+xx]);++differences;}}
        }
        SelectObject(capture_dc,old);DeleteObject(bitmap);DeleteDC(capture_dc);ReleaseDC(nullptr,desktop);FreeLibrary(dwm);
        if(differences)std::fprintf(stderr,"native display RGB differences=%zu client=%d,%d desktop=%d,%d\n",differences,w,h,sw,sh);
        verify(!differences&&std::all_of(seen.begin(),seen.end(),[](unsigned char value){return value==1;}),"every native final displayed pixel exact");
    };
    capture_display(expected);
    // Native partial transfer must retain the previous screen outside its rect,
    // even though its newly composed source has changed everywhere.
    verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&full,int(0x80007c00u))==0,"new screen contents");
    RECT partial={43,47,121,113};last_transfer=partial;final_ui_drawn=false;patch_JGL_present_screen(&partial);
    for(int y=partial.top;y<partial.bottom;++y)for(int x=partial.left;x<partial.right;++x)expected[y*w+x]=0xffff0000u;
    capture_display(expected);verify(screen_transfers==2&&owner.stats().readbacks==0,"partial transfer uses retained GPU display");
    // Re-creation requires a complete first transfer, never a partly initialized
    // screen. A retired image is rejected, then a full native request recovers.
    c3x_renderer_gpu_present_v1 retry={sizeof(retry)};retry.action=1;verify(present(&retry)==C3X_RENDERER_RESULT_OK,"release before re-creation");
    retry.action=0;retry.ticket=frame.ticket;retry.image=std::int64_t(owner.display_image(screen_surface));retry.window=window;retry.width=w;retry.height=h;
    retry.area[0]=partial.left;retry.area[1]=partial.top;retry.area[2]=partial.right;retry.area[3]=partial.bottom;
    verify(present(&retry)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"partial first frame rejected");
    retry.area[0]=retry.area[1]=0;retry.area[2]=w;retry.area[3]=h;
    auto retired=gpu.create(w,h,Format::bgra32);gpu.destroy(retired);retry.image=std::int64_t(retired);
    verify(present(&retry)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"retired display source rejected");
    last_transfer=full;final_ui_drawn=false;patch_JGL_present_screen(&full);
    expected.assign(expected.size(),0xffff0000u);for(int y=9;y<27;++y)for(int x=7;x<31;++x)expected[y*w+x]=0xff00ff00u;
    capture_display(expected);verify(owner.stats().readbacks==0,"full transfer recovers presentation without readback");
    // Explicit ownership release lets the audited original Graphsy/GDI transfer
    // run. Its actual HDC access restores the complete current native bitmap.
    state.current_config.enable_custom_rendering=false;
    final_ui_drawn=false;patch_JGL_present_screen(&full);
    verify(owner.stats().readbacks==3&&!owner.owns(screen_surface)&&state.custom_renderer_native_image==nullptr,"config-off releases GPU window and restores native family before original Graphsy transfer");
    expected.assign(expected.size(),0xffff0000u);for(int y=9;y<27;++y)for(int x=7;x<31;++x)expected[y*w+x]=0xff00ff00u;capture_display(expected);
    // Exercise the actual exported live callback on the preexisting, CPU-owned
    // screen. Retained raw writes must remain visible without getter events.
    verify(live&&render_view,"live screen exports");
    state.current_config.enable_custom_rendering=true;state.custom_renderer_native_image=live;
    auto retained=reinterpret_cast<Get>(screen_surface->vtable[7])(screen_surface,0,0);
    auto stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(screen_surface)+0x40);
    reinterpret_cast<Release>(screen_surface->vtable[9])(screen_surface,1);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)retained[y*stride+x]=std::uint16_t((x+y)&0x7fff);
    auto native_expected=[&](){std::vector<unsigned> rgb(std::size_t(w)*h);
        for(int y=0;y<h;++y)for(int x=0;x<w;++x){unsigned word=retained[y*stride+x],r=(word>>10)&31,g=(word>>5)&31,b=word&31;
            rgb[std::size_t(y)*w+x]=0xff000000u|((r<<3)|(r>>2))<<16|((g<<3)|(g>>2))<<8|((b<<3)|(b>>2));}return rgb;};
    last_transfer=full;final_ui_drawn=false;patch_JGL_present_screen(&full);live_active=true;
    expected=native_expected();capture_display(expected);
    // An ordinary map render must retain the display and its untouched pixels.
    // Conversely, a UI transfer must retain map publication bytes and metadata.
    c3x_renderer_output_v1 publication={C3X_RENDERER_API_VERSION,sizeof(publication)};
    verify(render_view(&demand,&publication)==C3X_RENDERER_RESULT_OK&&publication.bgra_pixels,"map render while native GPU presentation active");
    auto published=static_cast<unsigned const*>(publication.bgra_pixels);
    std::vector<unsigned> publication_pixels(published,published+std::size_t(publication.width)*publication.height);
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)retained[y*stride+x]=0x001f;
    last_transfer=partial;final_ui_drawn=false;patch_JGL_present_screen(&partial);
    for(int y=partial.top;y<partial.bottom;++y)for(int x=partial.left;x<partial.right;++x)expected[y*w+x]=0xff0000ffu;
    capture_display(expected);
    verify(std::equal(publication_pixels.begin(),publication_pixels.end(),published),"UI transfer preserves current map publication storage");
    // Explicit device/session reset preserves a complete native fallback and
    // permits recreation on the next map demand, with no stale native surfaces.
    reset();live_active=false;last_transfer=full;patch_JGL_present_screen(&full);
    expected=native_expected();capture_display(expected);
    verify(render_view(&demand,&publication)==C3X_RENDERER_RESULT_OK,"recreate renderer after reset");
    patch_JGL_present_screen(&full);live_active=true;capture_display(expected);
    // Independent RGB565 DIB input exercises the packed GPU expansion shader.
    // Its native lease fixture leaves the real JGL objects and palettes intact.
    struct Info565 {BITMAPINFOHEADER header;DWORD masks[3];} info565={};info565.header.biSize=sizeof(BITMAPINFOHEADER);
    info565.header.biWidth=w;info565.header.biHeight=-h;info565.header.biPlanes=1;info565.header.biBitCount=16;info565.header.biCompression=BI_BITFIELDS;
    info565.masks[0]=0xf800;info565.masks[1]=0x7e0;info565.masks[2]=0x1f;
    void* words565=nullptr;auto bitmap565=CreateDIBSection(dc,reinterpret_cast<BITMAPINFO*>(&info565),DIB_RGB_COLORS,&words565,nullptr,0);verify(bitmap565&&words565,"RGB565 source DIB");
    unsigned pitch565=(unsigned(w)+1)&~1u;std::vector<unsigned> metadata565(0x4d4/4);void* lease_table[10]={};
    lease_table[4]=reinterpret_cast<void*>(snapshot_test_bits);lease_table[9]=reinterpret_cast<void*>(snapshot_test_release);
    auto fake565=metadata565.data();*reinterpret_cast<void***>(fake565)=lease_table;
    fake565[0x24/4]=16;fake565[0x38/4]=w;fake565[0x3c/4]=h;fake565[0x40/4]=pitch565;
    *reinterpret_cast<HBITMAP*>(fake565+0x4b4/4)=bitmap565;*reinterpret_cast<void**>(fake565+0x4c0/4)=words565;
    for(int y=0;y<h;++y)for(int x=0;x<w;++x){unsigned r=x&31,g=y&63,b=(x+y)&31;static_cast<unsigned short*>(words565)[std::size_t(y)*pitch565+x]=std::uint16_t((r<<11)|(g<<5)|b);
        expected[std::size_t(y)*w+x]=0xff000000u|((r<<3)|(r>>2))<<16|((g<<2)|(g>>4))<<8|((b<<3)|(b>>2));}
    auto actual_screen=screen_surface;screen_surface=reinterpret_cast<JGL_Image*>(fake565);
    verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&full,nullptr,0)==1,"RGB565 GPU transfer");capture_display(expected);
    verify(SetWindowPos(window,HWND_TOPMOST,0,0,0,0,SWP_NOSIZE|SWP_NOACTIVATE)!=FALSE,"expose RGB565 fallback");
    live(C3X_NATIVE_IMAGE_DRAIN,nullptr,nullptr,nullptr,nullptr,0);live_active=false;preserve_gdi_display=true;
    if(w<=GetSystemMetrics(SM_CXSCREEN)&&h<=GetSystemMetrics(SM_CYSCREEN))capture_display(expected);
    preserve_gdi_display=false;live_active=true;screen_surface=actual_screen;DeleteObject(bitmap565);
    patch_JGL_present_screen(&full);expected=native_expected();capture_display(expected);
    // Isolated final-transfer comparison, including the same desktop completion
    // barrier on both arms. This is not whole-map or live-game latency evidence.
    auto dwm=LoadLibraryA("dwmapi.dll");auto completion=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(dwm,"DwmFlush"));
    LARGE_INTEGER frequency={};QueryPerformanceFrequency(&frequency);
    for(int block=0;block<6;++block){bool use_gpu=block%2!=0;
        live(C3X_NATIVE_IMAGE_DRAIN,nullptr,nullptr,nullptr,nullptr,0);
        state.custom_renderer_native_image=use_gpu?live:nullptr;
        patch_JGL_present_screen(&full);GdiFlush();completion();
        LARGE_INTEGER begin={},end={};QueryPerformanceCounter(&begin);long long callback_ticks=0;
        for(int n=0;n<8;++n){LARGE_INTEGER a={},b={};QueryPerformanceCounter(&a);patch_JGL_present_screen(&full);GdiFlush();QueryPerformanceCounter(&b);callback_ticks+=b.QuadPart-a.QuadPart;completion();}
        QueryPerformanceCounter(&end);
        std::printf("NATIVE_TRANSFER block=%d route=%s requests=8 callback_ms=%.3f request_and_desktop_ms=%.3f\n",block,use_gpu?"gpu-compatibility":"native-gdi",
            1000.*double(callback_ticks)/double(frequency.QuadPart)/8.,1000.*double(end.QuadPart-begin.QuadPart)/double(frequency.QuadPart)/8.);
    }
    LARGE_INTEGER snapshot_begin={},snapshot_end={};QueryPerformanceCounter(&snapshot_begin);
    for(int n=0;n<24;++n){c3x_native_images::ScreenSnapshot snapshot;verify(snapshot.capture(screen_surface,graph,&full),"native CPU snapshot diagnostic");}
    QueryPerformanceCounter(&snapshot_end);
    std::printf("NATIVE_SNAPSHOT requests=24 capture_ms=%.3f\n",1000.*double(snapshot_end.QuadPart-snapshot_begin.QuadPart)/double(frequency.QuadPart)/24.);
    FreeLibrary(dwm);state.custom_renderer_native_image=live;patch_JGL_present_screen(&full);
    // Observation can stop without disabling the live transfer owner.
    capture.presents=8191;capture.ended=false;last_transfer=full;patch_JGL_present_screen(&full);
    verify(capture.ended&&state.custom_renderer_native_probe_active&&state.custom_renderer_native_image==live,"bounded diagnostics do not detach live presentation");
    expected=native_expected();capture_display(expected);
    // Leave newer CPU pixels outside a partial transfer deliberately unshown.
    // Config-off must restore the displayed image, not expose those newer pixels.
    // Keep the dirty native rectangle on the desktop for GDI. A window DC
    // clips off-desktop drawing, unlike the retained GPU texture.
    verify(SetWindowPos(window,HWND_TOPMOST,0,0,0,0,SWP_NOSIZE|SWP_NOACTIVATE)!=FALSE,"expose native fallback rectangle");
    for(int y=0;y<h;++y)for(int x=0;x<w;++x)retained[y*stride+x]=0x7c00;
    state.current_config.enable_custom_rendering=false;patch_JGL_present_screen(&partial);live_active=false;
    for(int y=partial.top;y<partial.bottom;++y)for(int x=partial.left;x<partial.right;++x)expected[y*w+x]=0xffff0000u;
    preserve_gdi_display=true;
    verify(!state.custom_renderer_native_image,"live config-off drains before native GDI");capture_display(expected);
    preserve_gdi_display=false;
    std::puts("PASS live native screen: existing CPU surfaces, retained-pointer edits, exact full/partial display, interleaved map publication, observation expiry and config-off fallback");
    set_custom_renderer_native_probe(nullptr);native_present_image=nullptr;present_fn=native_present;screen.JGL.Image=nullptr;
    *reinterpret_cast<void**>(static_cast<char*>(graph)+0x148)=old_screen;*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138)=old_dc;
    for(auto canvas:canvases)reinterpret_cast<Destroy>(canvas->vtable[0])(canvas,1);reinterpret_cast<Destroy>(root->vtable[0])(root,1);
    reinterpret_cast<void(__thiscall*)(void*)>(gt[49])(graph);
    for(auto palette:palettes)reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(reinterpret_cast<char*>(jgl)+0x3cf10)(palette,1);
    reinterpret_cast<void(__thiscall*)(void*,unsigned)>(gt[0])(graph,1);FreeLibrary(jgl);ReleaseDC(window,dc);DestroyWindow(window);UnregisterClassA(wc.lpszClassName,wc.hInstance);
    std::puts("PASS native screen transfer: full-color map/copy/popup family, final UI ordering, exact displayed RGB, partial present, original GDI fallback; no execution readbacks before fallback");return true;
}
