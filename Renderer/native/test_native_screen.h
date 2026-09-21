#include "test_native_bootstrap.h"
// Included only in the real-renderer/native-hook fixture.
#include <thread>
#include "native_screen_bridge.h"
#include "test_native_line_bridge.h"
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
    if(!id){request.action=2;verify(screen_present(&request)==C3X_RENDERER_RESULT_OK,"preserve displayed GPU pixels before native fallback");return 0;}
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
int __fastcall unexpected_startup_native_transfer(void*,int,RECT*){return 917;}
bool native_screen_contract(char const* path,WorkerClient& gpu,c3x_renderer_gpu_frame_v1 frame,
                            c3x_renderer_gpu_present_fn present,unsigned const* map,int phase_x,int phase_y,c3x_renderer_native_image_fn live,c3x_renderer_render_view_fn render_view,c3x_renderer_camera_request_v1 const& demand,void (*reset)(),std::vector<NativeFrameSample> const& performance_frames){
    state={};capture={};events.clear();lines.clear();
    SetProcessDPIAware();WNDCLASSA wc={};wc.lpfnWndProc=screen_window_proc;wc.hInstance=GetModuleHandleA(nullptr);wc.lpszClassName="C3XNativeTransferContract";
    verify(RegisterClassA(&wc)!=0,"register native test window");
    HWND window=CreateWindowExA(WS_EX_TOPMOST|WS_EX_TOOLWINDOW,wc.lpszClassName,"Native transfer contract",WS_POPUP,20,20,frame.width,frame.height,nullptr,nullptr,wc.hInstance,nullptr);
    verify(window!=nullptr,"create native-owned test window");ShowWindow(window,SW_SHOWNOACTIVATE);UpdateWindow(window);
    HMODULE renderer_module=nullptr;verify(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<char const*>(live),&renderer_module)!=FALSE,"native renderer module");
    char candidate_path[MAX_PATH];verify(GetModuleFileNameA(renderer_module,candidate_path,MAX_PATH)>0,"candidate module path");tracking_candidate=candidate_path;
    auto dc=GetDC(window);auto graph=patch_load_jgl_lib(path);auto jgl=bootstrap_jgl;verify(graph&&dc,"native screen fixture post-load boundary");
    auto gt=*static_cast<void***>(graph);
    verify(state.custom_renderer_native_lifetime&&state.custom_renderer_native_probe_active,"production bootstrap before screen creation");
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
    void* original_bits=state.custom_renderer_jgl_original[4];void* original_release=state.custom_renderer_jgl_original[9];
    auto lifetime=state.custom_renderer_native_lifetime;
    verify(lifetime(C3X_NATIVE_MAP,root,0)!=0,"screen root lifetime came from actual bootstrap/init hook");
    c3x_renderer_native_observation observer_start={};observer_start.struct_size=sizeof(observer_start);
    observer_start.operation=C3X_NATIVE_VERIFY;observer_start.object=jgl;
    capture.write=log_line;verify(capture.observe(&observer_start)!=0,"initialize independent test observer");
    state.custom_renderer_native_observe=observe;
    JGL_Image* canvases[3];
    for(auto& canvas:canvases){canvas=create(graph,nullptr,1);verify(reinterpret_cast<Init>(canvas->vtable[1])(canvas,w,h,16,1)==0,"fresh native surface");
        verify(reinterpret_cast<Fill>(canvas->vtable[17])(canvas,&full,int(0x80000000u))==0,"clear native surface");}
    OpenGLRenderer startup_lines;PCX_Image startup_target;startup_target.JGL.Image=canvases[1];
    LoadedConfig startup_config={"(base)",nullptr};state.loaded_config_names=&startup_config;state.current_config.enable_custom_rendering=false;
    verify(patch_OpenGLRenderer_initialize(&startup_lines,0,&startup_target)==0&&!startup_lines.initialized&&
        lifetime(C3X_NATIVE_MAP,canvases[1],0),"pre-configuration main canvas line initialization preserves copy admission");
    state.loaded_config_names=&fixture_base_config;state.current_config.enable_custom_rendering=true;
    c3x_native_images::Adapter<WorkerClient> owner(gpu,original_bits,original_release,lifetime);adapter=&owner;state.custom_renderer_native_image=translate;
    if(w==2240&&h==1260){
        // The live HUD uses several separate full-size canvases. Their CPU
        // words are stale after GPU takeover and must not exhaust source cache.
        std::vector<JGL_Image*> family;
        for(unsigned n=0;n<6;++n){auto image=create(graph,nullptr,1);
            verify(reinterpret_cast<Init>(image->vtable[1])(image,w,h,16,1)==0,"full-resolution family init");
            verify(owner.admit(image),"six full-resolution native canvases fit without stale CPU mirrors");family.push_back(image);}
        for(auto image:family)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        std::puts("PASS six full-resolution native canvases: no stale CPU-mirror admission limit");
    }
    auto escaped_dc=reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(root->vtable[10])(root);
    verify(escaped_dc!=nullptr,"public root DC escape");reinterpret_cast<Release>(root->vtable[11])(root,1);
    verify(!owner.admit(root),"CPU-escaped surface rejected at GPU demand");
    for(auto canvas:canvases)verify(!owner.image(canvas),"observed unused surfaces have no GPU allocations");
    screen_adapter=&owner;screen_client=&gpu;screen_present=present;screen_frame=frame;screen_graph=graph;
    state.custom_renderer_native_image=translate;native_present_image=present_native_image;present_fn=complete_native_ui;
    verify(reinterpret_cast<Fill>(canvases[2]->vtable[17])(canvases[2],&full,int(0x80001234u))==0&&!owner.image(canvases[2]),"unrelated native fill does not acquire GPU residency");
    auto scene=canvases[0],screen_surface=canvases[1],save=canvases[2];screen_image=screen_surface;screen.JGL.Image=screen_surface;
    struct NativePcx {void* unused;JGL_Image* image;} pcx={nullptr,screen_surface};
    auto old_screen=*reinterpret_cast<void**>(static_cast<char*>(graph)+0x148);auto old_dc=*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138);
    *reinterpret_cast<void**>(static_cast<char*>(graph)+0x148)=&pcx;*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138)=dc;
    verify(owner.insert_map(scene,Id(frame.map_image),{0,0,w,h},0,0,phase_x,phase_y),"full-color map native insertion");
    auto copy=[&](JGL_Image* from,JGL_Image* to,RECT area){verify(reinterpret_cast<Copy>(from->vtable[16])(from,to,&area,&area)==0,"native family copy");};
    copy(scene,screen_surface,full);
    verify(owner.owns(scene)&&owner.owns(screen_surface)&&!owner.image(save),"startup evidence admits only demanded map and screen destinations");
    auto popup_bytes=gpu.stats().resident_bytes;
    RECT popup={43,47,121,113};copy(screen_surface,save,popup);
    if(owner.stats().readbacks)std::fprintf(stderr,"popup ownership loss: width=%d height=%d resident_before=%lld resident_after=%lld readbacks=%llu\n",
        w,h,popup_bytes,gpu.stats().resident_bytes,static_cast<unsigned long long>(owner.stats().readbacks));
    verify(owner.stats().readbacks==0,"native popup save preserves GPU ownership");
    verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&popup,int(0x80007fffu))==0,"native popup draw");copy(save,screen_surface,popup);
    std::vector<unsigned> expected(map,map+std::size_t(w)*h),observed(expected.size());
    // The native unit adapter consumes these actual JGL image identities; it
    // must not acquire either CPU DC before composing and presenting the unit.
    auto unit_gpu=reinterpret_cast<c3x_renderer_gpu_unit_fn>(GetProcAddress(renderer_module,"c3x_renderer_gpu_unit"));
    auto unit_cpu=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(GetProcAddress(renderer_module,"c3x_renderer_unit_draw_expanded"));
    c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);strcpy_s(unit.unit_key,"PRTO_Warrior");unit.unit_id=732;
    unit.action=1;unit.direction=3;unit.frame_count=16;unit.action_cursor=7;unit.sprite_width=unit.sprite_height=191;
    unit.projection_scale_milli=demand.frame->tile_width*1000/128;
    unit.body_x=120;unit.body_y=160;unit.hour=12;unit.display_color_rgb=0x205bdd;unit.presentation_frequency=1000000;unit.presentation_time_ticks=1000000;
    {
        // Native animation also draws into separate packed scratch surfaces,
        // before either surface has acquired an optional full-color layer.
        JGL_Image* scratch[2];constexpr unsigned extent=384;
        RECT scratch_area={0,0,extent,extent};
        for(auto& image:scratch){image=create(graph,nullptr,1);
            verify(reinterpret_cast<Init>(image->vtable[1])(image,extent,extent,16,1)==0,"unit scratch init");
            verify(reinterpret_cast<Fill>(image->vtable[17])(image,&scratch_area,int(0x800003e0u))==0,"unit scratch clear");}
        UnitOracleDib oracle(extent,extent,1);auto words=static_cast<unsigned short*>(oracle.pixels);
        std::fill(words,words+extent*extent,static_cast<unsigned short>(0x3e0));int expected_bounds[4]={},bounds[4]={};
        verify(unit_cpu(&unit,oracle.dc,oracle.dc,expected_bounds)==C3X_RENDERER_RESULT_OK,"packed scratch unit oracle");GdiFlush();
        auto reads=owner.stats().readbacks;
        verify(owner.draw_unit(unit_gpu,frame.ticket,unit,scratch[0],scratch[1],bounds,0),"separate packed unit scratch composition");
        verify(std::equal(bounds,bounds+4,expected_bounds)&&owner.owns(scratch[0])&&owner.owns(scene)&&
            lifetime(C3X_NATIVE_MAP,scene,0)&&lifetime(C3X_NATIVE_MAP,scratch[0],0)&&owner.stats().readbacks==reads,
            "unit scratch draw preserves native map lifetime without CPU escape");
        std::vector<unsigned> pixels(extent*extent);verify(gpu.readback(owner.image(scratch[0]),pixels.data(),pixels.size()),"scratch oracle readback");
        for(unsigned i=0;i<pixels.size();++i)verify(pixels[i]==words[i],"packed scratch exact native pixels");
        for(auto image:scratch)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        std::puts("PASS native unit scratch: separate packed surfaces, exact pixels/bounds, zero delivery readbacks, map residency preserved");
    }
    {
        UnitOracleDib oracle(w,h,0);std::copy(expected.begin(),expected.end(),static_cast<unsigned*>(oracle.pixels));int expected_bounds[4]={},bounds[4]={};
        verify(unit_cpu&&unit_cpu(&unit,oracle.dc,oracle.dc,expected_bounds)==C3X_RENDERER_RESULT_OK,"native unit display oracle");GdiFlush();
        verify(owner.draw_unit(unit_gpu,frame.ticket,unit,screen_surface,screen_surface,bounds,0),"native unit image adapter to GPU composition");
        verify(std::equal(bounds,bounds+4,expected_bounds)&&owner.owns(screen_surface)&&owner.stats().readbacks==0,"native unit preserves GPU ownership and erase bounds");
        for(unsigned i=0;i<expected.size();++i)expected[i]=static_cast<unsigned*>(oracle.pixels)[i]|0xff000000u;
    }
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
    auto scales=reinterpret_cast<int*>(base+0x6c0fc);int saved_scales[3]={scales[0],scales[1],scales[2]};
    scales[0]=2;scales[1]=1;scales[2]=1;
    verify(reinterpret_cast<int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*)>(overlay.vtable[17])(&overlay,screen_surface,41,31,palettes[0])==0,"scaled indexed overlay on resident map");
    for(int y=0;y<2;++y)for(int x=0;x<8;++x){auto index=indices[y*4+x/2];
        if(index<254)expected[(y+32)*w+x+43]=index?0xff00ff00u:0xffff00ffu;}
    for(unsigned i=0;i<3;++i)scales[i]=saved_scales[i];
    overlay.bits=nullptr;reinterpret_cast<void(__thiscall*)(JGLSprite*)>(base+0x7ed0)(&overlay);
    for(int y=9;y<27;++y)for(int x=7;x<31;++x)expected[y*w+x]=0xff00ff00u;
    final_ui_drawn=false;patch_JGL_present_screen(&full);
    verify(*reinterpret_cast<void**>(reinterpret_cast<char*>(screen_surface)+0x7c)==palettes[0],"GPU transfer preserves native screen palette binding");
    c3x_renderer_gpu_present_v1 foreign_release={sizeof(foreign_release)};foreign_release.action=1;int foreign_result=0;
    std::thread foreign_present([&]{foreign_result=present(&foreign_release);});foreign_present.join();
    verify(foreign_result==C3X_RENDERER_RESULT_BAD_ARGUMENT,"window ownership cannot be released by a foreign thread");
    verify(screen_transfers==1&&owner.stats().readbacks==0,"one complete GPU transfer with no native map readback");
    bool screen_read=gpu.readback(owner.display_image(screen_surface),observed.data(),observed.size());
    if(screen_read && observed!=expected){
        unsigned count=0,first=0;for(unsigned i=0;i<expected.size();++i)if(observed[i]!=expected[i]){if(!count)first=i;++count;}
        std::fprintf(stderr,"NATIVE_SCREEN_MISMATCH pixels=%u first=%u,%u actual=%08x expected=%08x\n",count,first%w,first/w,observed[first],expected[first]);
    }
    verify(screen_read&&observed==expected,"map/screen/save-restore family preserves full color exactly");
    RECT last_transfer=full;bool live_active=false,preserve_gdi_display=false;unsigned capture_number=0;
    auto capture_display=[&](std::vector<unsigned> const& pixels,std::vector<unsigned>* captured_pixels=nullptr){
        if(captured_pixels)captured_pixels->resize(pixels.size());
        ++capture_number;
        auto dwm=LoadLibraryA("dwmapi.dll");verify(dwm!=nullptr,"desktop completion oracle");auto flush=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(dwm,"DwmFlush"));
        verify(flush!=nullptr,"desktop completion function");
        // The desktop can have a persistent lower-right system watermark.
        // Expose every source pixel through the upper-left quarter instead;
        // this changes only test-window placement, never masks mismatches.
        int sw=GetSystemMetrics(SM_CXSCREEN)/2,sh=GetSystemMetrics(SM_CYSCREEN)/2;verify(sw>0&&sh>0,"desktop capture extent");
        BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=w;info.bmiHeader.biHeight=-h;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
        auto desktop=GetDC(nullptr),capture_dc=CreateCompatibleDC(desktop);void* bits=nullptr;auto bitmap=CreateDIBSection(capture_dc,&info,DIB_RGB_COLORS,&bits,nullptr,0);
        verify(bitmap&&bits,"display oracle allocation");auto old=SelectObject(capture_dc,bitmap);
        std::vector<unsigned char> seen(pixels.size(),0);std::size_t differences=0;RECT mismatch={w,h,0,0};
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
                if(captured_pixels)(*captured_pixels)[i]=0xff000000u|(static_cast<unsigned*>(bits)[std::size_t(yy)*w+xx]&0xffffff);
                else if((static_cast<unsigned*>(bits)[std::size_t(yy)*w+xx]&0xffffff)!=(pixels[i]&0xffffff)){
                    if(!differences)std::fprintf(stderr,"display first mismatch capture=%u live=%u x=%d y=%d expected=%08x actual=%08x\n",capture_number,unsigned(live_active),xx+sx,yy+sy,pixels[i],static_cast<unsigned*>(bits)[std::size_t(yy)*w+xx]);++differences;mismatch.left=std::min(mismatch.left,LONG(xx+sx));mismatch.top=std::min(mismatch.top,LONG(yy+sy));
                    mismatch.right=std::max(mismatch.right,LONG(xx+sx+1));mismatch.bottom=std::max(mismatch.bottom,LONG(yy+sy+1));}}
        }
        SelectObject(capture_dc,old);DeleteObject(bitmap);DeleteDC(capture_dc);ReleaseDC(nullptr,desktop);FreeLibrary(dwm);
        if(differences)std::fprintf(stderr,"native display RGB differences=%zu client=%d,%d desktop=%d,%d bounds=%ld,%ld,%ld,%ld\n",differences,w,h,sw,sh,mismatch.left,mismatch.top,mismatch.right,mismatch.bottom);
        verify(!differences&&std::all_of(seen.begin(),seen.end(),[](unsigned char value){return value==1;}),"every native final displayed pixel exact");
    };
    capture_display(expected);
    {
        auto visual=reinterpret_cast<int(*)()>(GetProcAddress(renderer_module,"c3x_renderer_gpu_visual_frame"));
        auto status=reinterpret_cast<int(*)(c3x_renderer_visual_status_v1*)>(GetProcAddress(renderer_module,"c3x_renderer_gpu_visual_status"));
        if(visual&&status){
        copy(screen_surface,save,full);copy(scene,screen_surface,full);
        // Repeated native save/restore transfers must retain shared versions,
        // not one fullscreen texture per copy, before independent playback.
        for(unsigned n=0;n<12;++n){copy(scene,screen_surface,full);copy(screen_surface,scene,full);}
        NativeUiAssets hud(graph,reinterpret_cast<char*>(jgl));
        auto hud_readbacks=owner.stats().readbacks;
        // Empty draws are common during hover/selection. Match native errors,
        // then compose the actual HUD before resource-only and unit animation.
        JGLSprite empty_sprite={};auto jgl_base=reinterpret_cast<char*>(jgl);
        reinterpret_cast<JGLSprite*(__thiscall*)(JGLSprite*,void*)>(jgl_base+0x7e80)(&empty_sprite,nullptr);
        using EmptyDraw=int(__thiscall*)(JGLSprite*,JGL_Image*,int,int,void*);
        for(unsigned n=0;n<24;++n)verify(reinterpret_cast<EmptyDraw>(empty_sprite.vtable[17])(&empty_sprite,screen_surface,int(n),int(n),nullptr)==7,"full-screen empty sprite preserves native error");
        reinterpret_cast<void(__thiscall*)(JGLSprite*)>(jgl_base+0x7ed0)(&empty_sprite);
        for(unsigned n=0;n<hud.pairs.size();++n){int x=n%2?0:std::max(0,w-hud.pairs[n]->color.width),y=std::max(30,h-hud.pairs[n]->color.height-8-int(n/2)*36);
            verify(hud.draw(n,scene,screen_surface,x,y)==0,"actual HUD over resident map");}
        verify(owner.owns(scene)&&owner.owns(screen_surface)&&owner.stats().readbacks==hud_readbacks,"HUD and hover keep animated map dependencies resident");
        // Resource-only idle must not depend on an animated/selected unit.
        final_ui_drawn=false;patch_JGL_present_screen(&full);
        live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,1);
        c3x_renderer_visual_status_v1 ambient_before={sizeof(ambient_before)},ambient_after={sizeof(ambient_after)};
        SetWindowPos(window,HWND_TOPMOST,0,0,w,h,SWP_NOACTIVATE|SWP_SHOWWINDOW);
        auto ambient_dwm=LoadLibraryA("dwmapi.dll");auto ambient_finish=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(ambient_dwm,"DwmFlush"));
        int ambient_w=std::min(w,GetSystemMetrics(SM_CXSCREEN)),ambient_h=std::min(h,GetSystemMetrics(SM_CYSCREEN));
        auto ambient_desktop=GetDC(nullptr),ambient_dc=CreateCompatibleDC(ambient_desktop);void* ambient_pixels=nullptr;
        BITMAPINFO ambient_info={};ambient_info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);ambient_info.bmiHeader.biWidth=ambient_w;ambient_info.bmiHeader.biHeight=-ambient_h;ambient_info.bmiHeader.biPlanes=1;ambient_info.bmiHeader.biBitCount=32;
        auto ambient_bitmap=CreateDIBSection(ambient_dc,&ambient_info,DIB_RGB_COLORS,&ambient_pixels,nullptr,0);verify(ambient_bitmap&&ambient_finish,"ambient display witness");
        auto ambient_previous=SelectObject(ambient_dc,ambient_bitmap);std::vector<unsigned> ambient_first;std::size_t ambient_changes=0;
        verify(status(&ambient_before)==1,"resource-only idle status before");
        for(unsigned n=0;n<8;++n){Sleep(70);int result=visual();
            verify(result==1||result==C3X_RENDERER_RESULT_PENDING,"resource-only idle visual frame");
            if(n==0||n==7){verify(SUCCEEDED(ambient_finish()),"ambient desktop completion");
                verify(BitBlt(ambient_dc,0,0,ambient_w,ambient_h,ambient_desktop,0,0,SRCCOPY)!=FALSE,"ambient desktop pixels");GdiFlush();
                auto words=static_cast<unsigned*>(ambient_pixels);
                if(!n)ambient_first.assign(words,words+ambient_w*ambient_h);
                else for(int y=32;y<ambient_h-80;++y)for(int x=32;x<ambient_w-32;++x)ambient_changes+=(words[y*ambient_w+x]&0xffffff)!=(ambient_first[y*ambient_w+x]&0xffffff);
            }
        }
        SelectObject(ambient_dc,ambient_previous);DeleteObject(ambient_bitmap);DeleteDC(ambient_dc);ReleaseDC(nullptr,ambient_desktop);FreeLibrary(ambient_dwm);
        verify(ambient_changes>0,"ambient map pixels visibly change without units or native redraws");
        std::printf("PASS ambient desktop motion: changed_pixels=%zu native_draw_calls=0\n",ambient_changes);

        verify(status(&ambient_after)==1&&ambient_after.map_samples>ambient_before.map_samples&&
            ambient_after.unit_samples==ambient_before.unit_samples,"resources animate without any animated unit");
        std::printf("PASS resource-only idle: map_samples=%lld unit_samples=0\n",ambient_after.map_samples-ambient_before.map_samples);
        // A new prepared map is not a native screen transfer. The committed
        // front (and any untouched rectangles) still owns its animation input.
        auto render_gpu=reinterpret_cast<c3x_renderer_gpu_render_fn>(GetProcAddress(renderer_module,"c3x_renderer_gpu_render"));
        c3x_renderer_gpu_frame_v1 prepared={sizeof(prepared)};
        c3x_renderer_output_v1 prepared_meta={C3X_RENDERER_API_VERSION,sizeof(prepared_meta)};
        gpu.flush();verify(render_gpu&&render_gpu(&demand,&prepared,&prepared_meta)==1,"prepare replacement before screen transfer");
        gpu.advance(prepared);frame=prepared;screen_frame=prepared;
        auto retained_before=ambient_after;
        for(unsigned n=0;n<4;++n){Sleep(70);visual();}
        verify(status(&ambient_after)==1&&ambient_after.map_samples>retained_before.map_samples&&
            ambient_after.unit_samples==retained_before.unit_samples,"unpublished replacement cannot freeze displayed resources");
        std::printf("PASS unpublished map replacement: displayed_map_samples=%lld\n",ambient_after.map_samples-retained_before.map_samples);
        char visual_count_option[16]={},visual_case[16]={},visual_frames_option[16]={};
        GetEnvironmentVariableA("C3X_RENDERER_VISUAL_FRAMES",visual_frames_option,sizeof(visual_frames_option));
        unsigned visual_frames=visual_frames_option[0]?unsigned(std::max(30,std::min(1200,std::atoi(visual_frames_option)))):30;
        GetEnvironmentVariableA("C3X_RENDERER_VISUAL_UNITS",visual_count_option,sizeof(visual_count_option));
        GetEnvironmentVariableA("C3X_RENDERER_VISUAL_UNIT_CASE",visual_case,sizeof(visual_case));
        unsigned visual_count=std::max(1,std::min(32,std::atoi(visual_count_option)));
        unsigned working=0,frozen=0,directed=0;
        for(unsigned n=0;n<visual_count;++n){
            auto selected=unit;selected.unit_id=901+int(n);selected.action_cursor=0;int bounds[4]={};
            if(visual_count>1){unsigned columns=visual_count<=8?4:8,rows=(visual_count+columns-1)/columns;
                selected.body_x=int(n%columns)*w/int(columns);selected.body_y=int(n/columns)*h/int(rows);
                selected.direction=1+int(n%8);}
            unsigned flags=C3X_RENDERER_UNIT_STATE_CAPTURED|(n==0?C3X_RENDERER_UNIT_SELECTED:0);
            if(n && (!std::strcmp(visual_case,"work") || (!std::strcmp(visual_case,"mixed") && n%3==1))){
                strcpy_s(selected.unit_key,"PRTO_Worker");selected.action=13;++working;
            }else if(n && !std::strcmp(visual_case,"mixed") && n%3==2){selected.action=2;selected.action_cursor=7;++directed;}
            else if(n)++frozen;
            verify(owner.draw_unit(unit_gpu,frame.ticket,selected,screen_surface,screen_surface,bounds,flags),"publish captured visual unit once");
        }
        std::printf("VISUAL_WORKLOAD units=%u selected_idle=1 work=%u frozen_idle=%u native_action=%u\n",visual_count,working,frozen,directed);
        final_ui_drawn=false;patch_JGL_present_screen(&full);
        verify(live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,1)==1,"retained complete frame ready");
        c3x_renderer_visual_status_v1 before={sizeof(before)},after={sizeof(after)};verify(status(&before)==1,"visual status before");
        auto native_events=events.size();auto transfers=screen_transfers;
        LARGE_INTEGER begin={},end={},frequency={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&begin);
        auto desktop_library=LoadLibraryA("dwmapi.dll");
        auto visual_desktop=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(desktop_library,"DwmFlush"));
        verify(visual_desktop!=nullptr,"independent visual desktop boundary");
        SetWindowPos(window,HWND_TOPMOST,20,20,w,h,SWP_NOACTIVATE|SWP_SHOWWINDOW);
        auto desktop_dc=GetDC(nullptr);double visual_request_ms=0,visual_desktop_ms=0;
        unsigned auxiliary_timers=0,held_visual_timers=0;
        for(unsigned n=0;n<visual_frames;++n){
            // A sustained standalone run must service its window messages.
            // Keep timer delivery for the separate transport assertion below;
            // these samples exercise exactly one direct visual opportunity.
            MSG message;
            while(PeekMessageA(&message,nullptr,0,WM_TIMER-1,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
            while(PeekMessageA(&message,nullptr,WM_TIMER+1,0xffff,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
            // D3D/window support can queue timers too. Service those during
            // a long direct-frame run; only withhold this DLL's visual callback
            // until the explicit transport check. Do not manufacture a timer
            // backlog that a normal game message pump would have consumed.
            for(unsigned drained=0;drained<64&&PeekMessageA(&message,nullptr,WM_TIMER,WM_TIMER,PM_REMOVE);++drained){
                HMODULE callback_module=nullptr;
                if(message.lParam&&GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<char const*>(message.lParam),&callback_module)&&callback_module==renderer_module){++held_visual_timers;break;}
                ++auxiliary_timers;DispatchMessageA(&message);
            }
            Sleep(33);LARGE_INTEGER a={},b={};QueryPerformanceCounter(&a);
            int result=visual();verify(result==1||result==C3X_RENDERER_RESULT_PENDING,"independent completed GPU visual frame");QueryPerformanceCounter(&b);
            verify(SUCCEEDED(visual_desktop()),"independent visual desktop completion");LARGE_INTEGER visible={};QueryPerformanceCounter(&visible);
            verify(GetPixel(desktop_dc,30,32)==RGB(0,255,0),"retained opaque UI unchanged over independent animation");
            double request_ms=1000.*double(b.QuadPart-a.QuadPart)/frequency.QuadPart,desktop_ms=1000.*double(visible.QuadPart-a.QuadPart)/frequency.QuadPart;
            visual_request_ms+=request_ms;visual_desktop_ms+=desktop_ms;
            std::printf("VISUAL_SAMPLE result=%d request_ms=%.3f desktop_ms=%.3f begin_qpc=%lld end_qpc=%lld\n",
                result,request_ms,desktop_ms,a.QuadPart,b.QuadPart);}
        ReleaseDC(nullptr,desktop_dc);FreeLibrary(desktop_library);
        std::printf("DIRECT_MESSAGE_PUMP auxiliary_timers=%u held_renderer_timers=%u\n",auxiliary_timers,held_visual_timers);
        QueryPerformanceCounter(&end);verify(status(&after)==1,"visual status after");
        verify(after.frames-before.frames>=visual_frames/2&&after.frames-before.frames<=visual_frames&&after.map_samples>before.map_samples&&after.unit_samples>before.unit_samples&&after.pose_changes>before.pose_changes,
            "authored unit poses advance with no new native selection");
        verify(events.size()==native_events&&screen_transfers==transfers,"visual frames never call native drawing/transfer hooks");
        verify(after.retained_bytes<=128ll*1024*1024,"retained visual memory bound");
        auto prior_window=GetForegroundWindow();SetWindowPos(window,HWND_TOPMOST,20,20,w,h,SWP_SHOWWINDOW);
        auto foreground_thread=GetWindowThreadProcessId(prior_window,nullptr),caller_thread=GetCurrentThreadId();
        bool attached=foreground_thread && foreground_thread!=caller_thread && AttachThreadInput(caller_thread,foreground_thread,TRUE);
        auto activated=SetForegroundWindow(window);SetFocus(window);
        if(attached)AttachThreadInput(caller_thread,foreground_thread,FALSE);
        // Activation can be delivered asynchronously across input queues.
        auto activation_deadline=GetTickCount64()+500;
        while(GetForegroundWindow()!=GetAncestor(window,GA_ROOT)&&GetTickCount64()<activation_deadline){
            MSG message;
            while(PeekMessageA(&message,nullptr,0,WM_TIMER-1,PM_REMOVE)){TranslateMessage(&message);DispatchMessageA(&message);}
            Sleep(1);
        }
        if(GetForegroundWindow()!=GetAncestor(window,GA_ROOT))std::fprintf(stderr,
            "FOREGROUND_DIAGNOSTIC activated=%d attached=%d visible=%d prior=%p actual=%p expected=%p caller_thread=%lu foreground_thread=%lu\n",
            int(activated),int(attached),int(IsWindowVisible(window)),static_cast<void*>(prior_window),static_cast<void*>(GetForegroundWindow()),
            static_cast<void*>(GetAncestor(window,GA_ROOT)),static_cast<unsigned long>(caller_thread),static_cast<unsigned long>(foreground_thread));
        verify(IsWindowVisible(window)&&GetForegroundWindow()==GetAncestor(window,GA_ROOT),"timer fixture owns visible foreground window");
        c3x_renderer_visual_status_v1 transported=after;
        LARGE_INTEGER deadline={};QueryPerformanceCounter(&deadline);deadline.QuadPart+=frequency.QuadPart*3;
        unsigned delivered_timers=0;
        do{
            // Check the deadline/progress after each callback. A frame longer
            // than 33 ms can keep WM_TIMER continuously due; draining all of
            // them here would starve the very status check that ends the test.
            MSG message;if(PeekMessageA(&message,nullptr,WM_TIMER,WM_TIMER,PM_REMOVE)){++delivered_timers;DispatchMessageA(&message);}
            verify(status(&transported)==1,"timer transport status");QueryPerformanceCounter(&end);
            if(transported.frames>=after.frames+3)break;Sleep(1);
        }while(end.QuadPart<deadline.QuadPart);
        std::printf("TIMER_DELIVERY callbacks=%u frames=%lld native_events=%zu transfers=%u visible=%d foreground=%d\n",delivered_timers,transported.frames-after.frames,events.size()-native_events,unsigned(screen_transfers-transfers),int(IsWindowVisible(window)),int(GetForegroundWindow()==GetAncestor(window,GA_ROOT)));
        if(prior_window)SetForegroundWindow(prior_window);
        verify(transported.frames>=after.frames+3&&events.size()==native_events&&screen_transfers==transfers,
            "renderer timer transports frames without native draw demand");
        std::printf("PASS visual timer transport: frames=%lld native_draw_calls=0\n",transported.frames-after.frames);
        live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,0);
        verify(visual()==C3X_RENDERER_RESULT_PENDING,"explicit modal policy pauses renderer clock");
        auto clock=reinterpret_cast<c3x_renderer_visual_clock_fn>(GetProcAddress(renderer_module,"c3x_renderer_visual_clock"));
        verify(clock!=nullptr,"native capture shares renderer visual clock");auto paused=clock();Sleep(20);
        verify(clock()==paused,"modal pause excludes wall time from shared clock");
        std::printf("PASS independent resident frames: frames=%lld map_samples=%lld unit_samples=%lld pose_changes=%lld bytes=%lld nodes=%lld average_request_ms=%.3f average_desktop_ms=%.3f native_draw_calls=0\n",
            after.frames-before.frames,after.map_samples-before.map_samples,after.unit_samples-before.unit_samples,
            after.pose_changes-before.pose_changes,after.retained_bytes,after.nodes,
            visual_request_ms/visual_frames,visual_desktop_ms/visual_frames);
        copy(save,screen_surface,full);final_ui_drawn=false;patch_JGL_present_screen(&full);
        capture_display(expected);
        live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,1);
        }
    }
    auto transfers_before_partial=screen_transfers;
    // Native partial transfer must retain the previous screen outside its rect,
    // even though its newly composed source has changed everywhere.
    verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&full,int(0x80007c00u))==0,"new screen contents");
    RECT partial={43,47,121,113};last_transfer=partial;final_ui_drawn=false;patch_JGL_present_screen(&partial);
    for(int y=partial.top;y<partial.bottom;++y)for(int x=partial.left;x<partial.right;++x)expected[y*w+x]=0xffff0000u;
    capture_display(expected);verify(screen_transfers==transfers_before_partial+1&&owner.stats().readbacks==0,"partial transfer uses retained GPU display");
    // Partial native handoff must retain full-color displayed pixels, including
    // those outside the native update that differ from the current red canvas.
    c3x_renderer_gpu_present_v1 handoff={sizeof(handoff)};handoff.action=2;
    verify(present(&handoff)==C3X_RENDERER_RESULT_OK,"preserve owned GPU display during GDI handoff");
    preserve_gdi_display=true;
    // The retained presentation texture is gone; prevent the capture oracle
    // from implicitly recreating it from the newer native working image.
    auto saved_native_image=screen_surface;screen_surface=nullptr;
    capture_display(expected);
    verify(SetWindowPos(window,HWND_TOPMOST,0,0,0,0,SWP_NOSIZE|SWP_NOACTIVATE)!=FALSE,"expose native handoff update");
    verify(PatBlt(dc,partial.left,partial.top,partial.right-partial.left,partial.bottom-partial.top,BLACKNESS)!=FALSE,"native partial drawing after GPU handoff");GdiFlush();
    for(int y=partial.top;y<partial.bottom;++y)for(int x=partial.left;x<partial.right;++x)expected[y*w+x]=0xff000000u;
    capture_display(expected);screen_surface=saved_native_image;preserve_gdi_display=false;
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
    verify(!owner.admit(screen_surface),"retained CPU pointer prevents readmission after GPU drain");
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
    // A CPU UI screen can start GPU presentation before any map demand or
    // material initialization, including after an explicit renderer reset.
    reset();last_transfer=full;
    // Enter through the production Graphsy hook with only the process-owned
    // module, as after configuration load / scene unload before any map call.
    // A distinct native sentinel proves substitution rather than a successful
    // native BitBlt hiding an unconnected GPU backend.
    auto saved_observer=state.custom_renderer_native_observe;
    auto saved_native_transfer=state.custom_renderer_jgl_present_original;
    state.custom_renderer_native_module=renderer_module;
    state.custom_renderer_native_image=nullptr;state.custom_renderer_native_observe=nullptr;
    state.custom_renderer_jgl_present_original=reinterpret_cast<void*>(unexpected_startup_native_transfer);
    state.current_config.enable_custom_rendering=false;
    verify(patch_JGL_Graphsy_present(graph,0,&full)==917&&!state.custom_renderer_native_image&&!state.custom_renderer_native_observe,"config-off UI does not bind GPU presentation");
    state.current_config.enable_custom_rendering=true;
    for(int phase=0;phase<3;++phase){
        verify(patch_JGL_Graphsy_present(graph,0,&full)==0&&state.custom_renderer_native_image==live&&state.custom_renderer_native_map==nullptr&&
            state.custom_renderer_native_observe==reinterpret_cast<c3x_renderer_native_observe_fn>(GetProcAddress(renderer_module,"c3x_renderer_native_observe")),"actual native UI entry binds the process-owned presenter without map initialization");
        live_active=true;expected=native_expected();capture_display(expected);
        if(phase==0){set_custom_renderer_native_probe(nullptr);state.custom_renderer_native_observe=nullptr;
            verify(!state.custom_renderer_native_image&&state.custom_renderer_native_probe_active,"scene unload keeps process hooks but releases presentation");}
        if(phase==1){state.current_config.enable_custom_rendering=false;
            verify(patch_JGL_Graphsy_present(graph,0,&full)==917&&!state.custom_renderer_native_image,"config-off drains GPU presentation before native UI transfer");
            state.current_config.enable_custom_rendering=true;}
    }
    state.custom_renderer_native_module=renderer_module;state.custom_renderer_native_observe=saved_observer;
    state.custom_renderer_jgl_present_original=saved_native_transfer;
    std::puts("PASS caller-owned UI presentation: configured pre-map entry, scene unload/rebind, config-off drain/native transfer and reenable; exact displayed RGB");
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
    state.custom_renderer_native_observe=observe; // Route this explicit observer-expiry fixture to its local counter.
    // Earlier rebinding deliberately switches observers inside a present. Start
    // this independent expiry case with a balanced local observation stack.
    capture.depth=0;capture.presents=8191;capture.ended=false;last_transfer=full;patch_JGL_present_screen(&full);
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
    // Exercise the production owner/export, replacing the fixture's adapter
    // callback. Native surfaces precede map demand; prepare/cancel leaves their
    // CPU pixels unchanged and commit shares ownership with actual JGL copies.
    {
        auto exact_native_map=reinterpret_cast<c3x_renderer_native_map_view_fn>(GetProcAddress(renderer_module,"c3x_renderer_native_map_view"));
        auto request_native_camera=reinterpret_cast<c3x_renderer_native_camera_request_fn>(GetProcAddress(renderer_module,"c3x_renderer_native_camera_request"));
        auto poll_native_camera=reinterpret_cast<c3x_renderer_native_camera_poll_fn>(GetProcAddress(renderer_module,"c3x_renderer_native_camera_poll"));
        auto native_camera_message=reinterpret_cast<c3x_renderer_native_camera_message_fn>(GetProcAddress(renderer_module,"c3x_renderer_native_camera_message"));
        unsigned ready_message=native_camera_message?native_camera_message():0;
        auto navigate=reinterpret_cast<c3x_renderer_native_navigation_fn>(GetProcAddress(renderer_module,"c3x_renderer_native_navigation"));
        bool navigation_test=GetEnvironmentVariableA("C3X_RENDERER_NATIVE_NAVIGATION_TEST",nullptr,0)!=0;
        verify(!navigation_test||navigate,"native navigation export");
        bool native_camera_test=GetEnvironmentVariableA("C3X_RENDERER_NATIVE_CAMERA_TEST",nullptr,0)!=0;
        verify(exact_native_map && (!native_camera_test || (request_native_camera && poll_native_camera && ready_message)),"native camera exports");
        unsigned native_camera_requests=0,native_camera_pending=0,native_camera_messages=0,native_camera_wakes=0;
        auto native_map_view=[&](int action,void* image,c3x_renderer_camera_request_v1 const* request,c3x_renderer_camera_view_v1* output){
            if(!native_camera_test || action!=C3X_NATIVE_MAP_PREPARE)return exact_native_map(action,image,request,output);
            LARGE_INTEGER frequency={},started={},enqueued={},finished={};QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&started);
            custom_renderer_native_view current={};current.width=request->frame->target_width;current.height=request->frame->target_height;
            current.tile_width=current.native_width=request->frame->tile_width;
            auto target=current;target.camera_x=17;target.camera_y=-9;target.min_x=4;target.max_x=20;
            c3x_renderer_i64 ticket=0;int result=navigation_test?navigate(C3X_NAV_REQUEST,image,&target,request):request_native_camera(image,request,&ticket);QueryPerformanceCounter(&enqueued);
            if(result!=C3X_RENDERER_RESULT_PENDING)return result;
            ++native_camera_requests;PostMessageA(window,WM_APP+61,0,0);
            c3x_renderer_gpu_camera_view_v1 view={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(view)};
            auto untouched=view;double maximum_poll=0,pending_maximum=0,ready_ms=0;unsigned pending_polls=0;auto deadline=GetTickCount64()+30000;
            while(result==C3X_RENDERER_RESULT_PENDING && GetTickCount64()<deadline){
                LARGE_INTEGER before={},after={};QueryPerformanceCounter(&before);
                if(navigation_test){
                    result=navigate(C3X_NAV_POLL,image,&current,nullptr);
                    if(result==C3X_RENDERER_RESULT_PENDING)verify(!current.camera_x&&!current.camera_y,"pending native camera and picking remain displayed");
                    if(result==C3X_RENDERER_RESULT_OK){
                        verify(current.camera_x==17&&current.camera_y==-9&&current.min_x==4,"ready native view advances before overlays");
                        view.camera.version=C3X_RENDERER_CAMERA_VIEW_VERSION;view.camera.struct_size=sizeof(view.camera);
                        result=exact_native_map(C3X_NATIVE_MAP_PREPARE,image,request,&view.camera);
                    }
                }else result=poll_native_camera(image,ticket,&view);
                QueryPerformanceCounter(&after);
                double poll_ms=1000.*double(after.QuadPart-before.QuadPart)/double(frequency.QuadPart);
                maximum_poll=std::max(maximum_poll,poll_ms);
                if(result==C3X_RENDERER_RESULT_OK)ready_ms=poll_ms;
                if(result==C3X_RENDERER_RESULT_PENDING){
                    ++pending_polls;++native_camera_pending;pending_maximum=std::max(pending_maximum,poll_ms);
                    verify(!std::memcmp(&view,&untouched,sizeof(view)),"native pending has no new coverage or output");
                    verify(exact_native_map(C3X_NATIVE_MAP_COMMIT,image,nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"pending native map cannot commit");
                    // Exercise a real message-pump opportunity between polls,
                    // including ambient timers while camera work owns the GPU.
                    bool woke=false;MSG message={};if(PeekMessageA(&message,nullptr,0,0,PM_REMOVE)){
                        if(message.message==WM_APP+61)++native_camera_messages;
                        if(message.message==ready_message){++native_camera_wakes;woke=true;}
                        TranslateMessage(&message);DispatchMessageA(&message);
                    }
                    // Wait for completion or other UI work, with a bounded
                    // retry if Windows cannot deliver a wake. No spin or timer
                    // resolution change; production itself returns pending.
                    if(!woke)MsgWaitForMultipleObjectsEx(0,nullptr,16,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
                }
            }
            QueryPerformanceCounter(&finished);
            if(result==C3X_RENDERER_RESULT_OK)*output=view.camera;
            std::printf("NATIVE_CAMERA_SAMPLE result=%d begin_ms=%.3f poll_max_ms=%.3f pending_max_ms=%.3f ready_ms=%.3f pending=%u complete_ms=%.3f\n",result,
                1000.*double(enqueued.QuadPart-started.QuadPart)/double(frequency.QuadPart),maximum_poll,pending_maximum,ready_ms,pending_polls,
                1000.*double(finished.QuadPart-started.QuadPart)/double(frequency.QuadPart));
            return result;
        };
        auto native_map=[&](int action,void* image,c3x_renderer_camera_request_v1 const* request,c3x_renderer_output_v1* output){
            c3x_renderer_camera_view_v1 sample={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(sample)};
            int code=native_map_view(action,image,request,output?&sample:nullptr);
            if(code==C3X_RENDERER_RESULT_OK && output){*output=sample.output;verify(sample.frame.presentation_time_ticks<=request->frame->presentation_time_ticks,"native map returns actual sample clock");}
            return code;
        };
        verify(exact_native_map!=nullptr,"production resident native map export");
        state.current_config.enable_custom_rendering=true;state.custom_renderer_native_image=live;
        JGL_Image* live_images[2];
        for(auto& image:live_images){image=create(graph,nullptr,1);verify(reinterpret_cast<Init>(image->vtable[1])(image,w,h,16,1)==0,"production native image init");
            verify(reinterpret_cast<Fill>(image->vtable[17])(image,&full,int(0x80001234u))==0,"production native pre-map contents");}
        // Live startup presents CPU menus on the eventual game screen before
        // any GPU map exists. The old private snapshot called the public bits
        // hook here, permanently excluding that same screen from later copies.
        screen_surface=live_images[1];screen_image=screen_surface;screen.JGL.Image=screen_surface;pcx.image=screen_surface;
        for(int startup=0;startup<24;++startup){
            // Before configuration loads, this is the original Graphsy GDI
            // transfer; afterwards the same screen uses CPU/GPU compatibility.
            state.current_config.enable_custom_rendering=startup>=12;
            auto links=screen_surface->Bits_Data_Links;events.clear();
            verify(patch_JGL_Graphsy_present(graph,0,&full)==0,"pre-map CPU menu presentation");
            verify(lifetime(C3X_NATIVE_MAP,screen_surface,0)&&screen_surface->Bits_Data_Links==links,
                "native/configured startup preserves eventual GPU screen admission and caller leases");
            verify(std::none_of(events.begin(),events.end(),[](auto const& e){return e.operation==C3X_NATIVE_BITS||e.operation==C3X_NATIVE_PIXEL;}),
                "private screen snapshot never becomes a public pixel escape");
            verify(state.custom_renderer_native_operation==0,"startup transfer restores native operation scope");
        }
        std::puts("PASS pre-map screen transfers: native=12 configured=12 lifetime_preserved=24 public_pixel_access=0");
        auto raw_unchanged=[&](){auto words=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(original_bits)(live_images[0]);
            auto stride_words=*reinterpret_cast<int*>(reinterpret_cast<char*>(live_images[0])+0x40);bool same=words!=nullptr;
            if(words)for(int y=0;y<h;++y)for(int x=0;x<w;++x)same=same&&words[y*stride_words+x]==0x1234;
            reinterpret_cast<Release>(original_release)(live_images[0],1);return same;};
        c3x_renderer_output_v1 meta={C3X_RENDERER_API_VERSION,sizeof(meta)};
        auto tiny=create(graph,nullptr,1);verify(reinterpret_cast<Init>(tiny->vtable[1])(tiny,4,4,16,1)==0,"unsupported native GPU extent");
        auto tiny_frame=*demand.frame;tiny_frame.target_width=tiny_frame.target_height=tiny_frame.clip_right=tiny_frame.clip_bottom=4;tiny_frame.clip_left=tiny_frame.clip_top=0;
        auto tiny_request=demand;tiny_request.frame=&tiny_frame;
        verify(native_map(C3X_NATIVE_MAP_PREPARE,tiny,&tiny_request,&meta)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"unsupported GPU surface returns admission rejection for CPU fallback");
        reinterpret_cast<Destroy>(tiny->vtable[0])(tiny,1);
        verify(native_map(C3X_NATIVE_MAP_PREPARE,root,&demand,&meta)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"CPU-escaped map target declines before preparation");
        verify(native_map(C3X_NATIVE_MAP_PREPARE,live_images[0],&demand,&meta)==C3X_RENDERER_RESULT_OK&&!meta.bgra_pixels&&raw_unchanged(),"resident prepare returns metadata without native pixel writes");
        verify(native_map(C3X_NATIVE_MAP_COMMIT,live_images[1],nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"wrong destination cannot commit prepared map");
        verify(native_map(C3X_NATIVE_MAP_CANCEL,live_images[0],nullptr,nullptr)==C3X_RENDERER_RESULT_OK&&raw_unchanged(),"rejected ownership can cancel without publishing");
        verify(native_map(C3X_NATIVE_MAP_COMMIT,live_images[0],nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"cancelled map cannot commit");
        screen_surface=live_images[1];screen_image=screen_surface;screen.JGL.Image=screen_surface;pcx.image=screen_surface;
        auto live_frame=*demand.frame;auto live_request=demand;live_request.frame=&live_frame;
        std::vector<c3x_renderer_tile_v1> live_tiles(live_frame.tiles,live_frame.tiles+live_frame.tile_count);live_frame.tiles=live_tiles.data();
        LOGFONTA text_font={};text_font.lfHeight=-17;text_font.lfWeight=700;text_font.lfOutPrecision=7;strcpy_s(text_font.lfFaceName,"Arial");
        auto label_font=CreateFontIndirectA(&text_font);verify(label_font!=nullptr,"native label font");
        auto label_dc=*reinterpret_cast<HDC*>(reinterpret_cast<char*>(screen_surface)+0x4bc);auto previous_font=SelectObject(label_dc,label_font);
        SetTextColor(label_dc,RGB(237,171,55));SetBkMode(label_dc,TRANSPARENT);SetTextAlign(label_dc,TA_LEFT|TA_TOP);
        c3x_native_text::State label_state; c3x_native_text::Raster label;
        verify(c3x_native_text::capture(label_dc,label_state)&&c3x_native_text::compile(label_dc,label_state,"Berlin: 6",9,label),"independent compiled label fixture");
        char const* placements[]={"mountain-behind","mountain-own","mountain-front","forest-behind","forest-own","forest-front","building-behind","building-own","building-front"};
        for(int step=0;step<9;++step){
            if(step)for(auto& tile:live_tiles){tile.anchor_x-=13;tile.anchor_y+=7;}
            int anchor_x=w/2+(step<3?-65:step<6?-192:45)*live_frame.tile_width/128-13*step;
            int anchor_y=h/2+(step<6?(step%3==0?15:step%3==1?32:90):(step%3==0?-60:step%3==1?0:80))*live_frame.tile_width/128+7*step;
            unit.body_x=anchor_x-unit.sprite_width*unit.projection_scale_milli/2000;
            unit.body_y=anchor_y-unit.sprite_height*unit.projection_scale_milli/2000;
            c3x_renderer_output_v1 control={C3X_RENDERER_API_VERSION,sizeof(control)};
            verify(render_view(&live_request,&control)==C3X_RENDERER_RESULT_OK,"production native map independent CPU oracle");
            auto pixels=static_cast<unsigned const*>(control.bgra_pixels);expected.assign(pixels,pixels+std::size_t(w)*h);
            std::vector<unsigned> ownership(control.replacement_tile_flags,control.replacement_tile_flags+control.replacement_tile_count);
            meta={C3X_RENDERER_API_VERSION,sizeof(meta)};
            verify(native_map(C3X_NATIVE_MAP_PREPARE,live_images[0],&live_request,&meta)==C3X_RENDERER_RESULT_OK&&!meta.bgra_pixels&&
                meta.replacement_tile_count==ownership.size()&&std::equal(ownership.begin(),ownership.end(),meta.replacement_tile_flags),"production prepare ownership matches current capture");
            verify(native_map(C3X_NATIVE_MAP_COMMIT,live_images[0],nullptr,nullptr)==C3X_RENDERER_RESULT_OK&&raw_unchanged(),"production native commit keeps map CPU storage untouched");
            OpenGLRenderer line_context;PCX_Image line_target;line_target.JGL.Image=live_images[0];
            for(int repeat=0;repeat<16;++repeat){
                verify(patch_OpenGLRenderer_initialize(&line_context,0,&line_target)==0,"native map-tail line initialization");
                patch_OpenGLRenderer_set_opacity(&line_context,0,255);patch_OpenGLRenderer_set_color(&line_context,0,0x80000000u);
                patch_OpenGLRenderer_set_line_width(&line_context,0,1);
                verify(!line_context.initialized&&!line_context.style_calls&&lifetime(C3X_NATIVE_MAP,live_images[0],0)&&raw_unchanged(),
                    "empty map-tail OpenGL setup retains GPU map without public DC or CPU readback");
            }
            copy(live_images[0],screen_surface,full);
            int bounds[4]={};
            UnitOracleDib unit_oracle(w,h,0);std::copy(expected.begin(),expected.end(),static_cast<unsigned*>(unit_oracle.pixels));int expected_bounds[4]={};
            verify(unit_cpu(&unit,unit_oracle.dc,unit_oracle.dc,expected_bounds)==C3X_RENDERER_RESULT_OK,"production owned unit CPU oracle");GdiFlush();
            verify(live(C3X_NATIVE_UNIT_DRAW,screen_surface,screen_surface,&unit,bounds,0)==1&&std::equal(bounds,bounds+4,expected_bounds),"production unit callback uses same map owner");
            for(unsigned n=0;n<expected.size();++n)expected[n]=static_cast<unsigned*>(unit_oracle.pixels)[n]|0xff000000u;
            write_unit_layer_witness(expected,w,h,placements[step]);
            int label_x=63+step*17,label_y=39;
            verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,char const*,int)>(screen_surface->vtable[46])(screen_surface,label_x,label_y,"Berlin: 6",9)==0,"actual native text hook");
            for(unsigned y=0;y<label.height;++y)for(unsigned x=0;x<label.width;++x){int dx=label_x+label.left+int(x),dy=label_y+label.top+int(y);
                if(dx>=0&&dy>=0&&dx<w&&dy<h)expected[dy*w+dx]=c3x_native_text::apply(label,y*label.width+x,expected[dy*w+dx],false,true);}
            auto cpu_screen=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(original_bits)(screen_surface);
            int cpu_stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(screen_surface)+0x40);bool untouched=true;
            for(int y=0;y<h;++y)for(int x=0;x<w;++x)untouched=untouched&&cpu_screen[y*cpu_stride+x]==0x1234;
            reinterpret_cast<Release>(original_release)(screen_surface,1);verify(untouched,"native map/copy/unit/text leaves CPU screen untouched");
            for(int y=9;y<27;++y)for(int x=7;x<31;++x)expected[y*w+x]=0xff00ff00u;
            last_transfer=full;final_ui_drawn=false;patch_JGL_present_screen(&full);live_active=true;capture_display(expected);
        }
        // Real outline calls after the previously empty native initialization.
        // Verify displayed pixels independently, then restore the exact scene.
        if(w>=320&&h>=240){
            OpenGLRenderer context;PCX_Image target;target.JGL.Image=screen_surface;
            auto saved_lines=create(graph,nullptr,1);verify(reinterpret_cast<Init>(saved_lines->vtable[1])(saved_lines,w,h,16,1)==0,"outline GPU save image");
            copy(screen_surface,saved_lines,full);
            RECT panel={96,96,240,200},line_clip={100,100,200,190};
            verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&panel,int(0x80000000u))==0,"GPU outline backdrop");
            verify(reinterpret_cast<Clip>(screen_surface->vtable[13])(screen_surface,&line_clip)==0,"GPU outline clipping");
            auto stroke=[&](int y,unsigned color,unsigned alpha,int width,int dash){
                verify(patch_OpenGLRenderer_initialize(&context,0,&target)==0,"GPU outline begin");
                patch_OpenGLRenderer_set_color(&context,0,color);patch_OpenGLRenderer_set_opacity(&context,0,alpha);
                patch_OpenGLRenderer_set_line_width(&context,0,width);
                if(dash)patch_OpenGLRenderer_enable_line_dashing(&context);else patch_OpenGLRenderer_disable_line_dashing(&context);
                patch_OpenGLRenderer_draw_line(&context,0,110,y,220,y);
            };
            stroke(110,0x80007fffu,255,1,0);stroke(130,0x80007c00u,128,3,0);stroke(150,0x800003e0u,255,2,1);
            state.current_config.draw_lines_using_gdi_plus=LDO_ALWAYS;stroke(170,0x8000001fu,255,2,1);
            state.current_config.draw_lines_using_gdi_plus=LDO_NEVER;
            verify(!context.initialized&&!context.drawn&&!context.style_calls&&!native_gdi_initializations&&raw_unchanged()&&
                lifetime(C3X_NATIVE_MAP,screen_surface,0),"native/GDI+ outlines stay GPU resident with no map DC escape");
            verify(reinterpret_cast<Clip>(screen_surface->vtable[13])(screen_surface,&full)==0,"restore outline clip");
            last_transfer=full;verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&full,nullptr,0)==1,"GPU outline display");
            std::vector<unsigned> shown;capture_display(expected,&shown);
            auto at=[&](int x,int y){return shown[std::size_t(y)*w+x]&0xffffff;};
            verify(at(120,110)==0xf8f8f8&&at(120,130)==0x7c0000&&at(120,129)==0x7c0000&&at(120,132)==0,
                "displayed native color, alpha and line width");
            verify(at(112,150)==0&&at(117,150)==0x00f800&&at(112,170)==0x0000f8&&at(117,170)==0,
                "GL factor-five and GDI+ width-scaled dash patterns");
            for(int y=100;y<190;++y)for(int x=200;x<240;++x)verify(at(x,y)==0,"native outline scissor survives GPU routing");
            // The displayed-stroke oracle is complete. Retire its full-screen
            // save before the independent escape test needs another target;
            // retaining both would exceed the bounded live family at fullscreen.
            copy(saved_lines,screen_surface,full);reinterpret_cast<Destroy>(saved_lines->vtable[0])(saved_lines,1);
            // CPU UI retargeting still invokes the original initializer. A
            // separate admitted scratch proves a genuine escape mid-scope
            // lazily initializes that backend and replays its style.
            target.JGL.Image=root;verify(patch_OpenGLRenderer_initialize(&context,0,&target)==0&&context.initialized==0,"CPU UI defers its line initializer until a stroke");
            patch_OpenGLRenderer_draw_line(&context,0,3,3,20,3);
            verify(context.initialized==1&&context.drawn==1,"CPU UI actual stroke initializes native backend");
            auto line_scratch=create(graph,nullptr,1);verify(reinterpret_cast<Init>(line_scratch->vtable[1])(line_scratch,w,h,16,1)==0,"outline scratch lifecycle");
            copy(live_images[0],line_scratch,full);target.JGL.Image=line_scratch;
            verify(patch_OpenGLRenderer_initialize(&context,0,&target)==0&&context.initialized==1,"new outline target admitted by GPU copy");
            patch_OpenGLRenderer_set_color(&context,0,0x80007c00u);patch_OpenGLRenderer_set_opacity(&context,0,128);patch_OpenGLRenderer_set_line_width(&context,0,3);
            auto escaped=reinterpret_cast<HDC(__thiscall*)(JGL_Image*)>(line_scratch->vtable[10])(line_scratch);verify(escaped!=nullptr,"outline-scope real CPU escape");
            reinterpret_cast<Release>(line_scratch->vtable[11])(line_scratch,1);
            patch_OpenGLRenderer_draw_line(&context,0,3,3,20,3);
            verify(context.initialized==2&&context.drawn==2&&!lifetime(C3X_NATIVE_MAP,line_scratch,0)&&!state.custom_renderer_line_owner,
                "real CPU escape retains native line fallback without re-admission");
            state.current_config.draw_lines_using_gdi_plus=LDO_ALWAYS;
            verify(patch_OpenGLRenderer_initialize(&context,0,&target)==0&&native_gdi_initializations==0,"CPU GDI+ initialization remains deferred");
            patch_OpenGLRenderer_draw_line(&context,0,3,3,20,3);
            verify(native_gdi_initializations==1&&native_gdi_draws==1&&native_gdi_argb==0x80f80000&&native_gdi_width==3,"GDI+ fallback preserves captured style");
            state.current_config.draw_lines_using_gdi_plus=LDO_NEVER;reinterpret_cast<Destroy>(line_scratch->vtable[0])(line_scratch,1);
            verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&full,nullptr,0)==1,"restore GPU outline save");capture_display(expected);
            std::puts("PASS native outline bridge: empty_initializations=144 resident=144 GPU_strokes=4 native_DC=0 actual_clip_color_alpha_width_dash=1 CPU_escape_fallback=1");
        }
        // An unrelated CPU-owned UI source uses the existing GPU presenter,
        // retaining full-color displayed pixels outside a partial transfer. The
        // map/screen GPU family stays resident throughout this ownership mix.
        auto resident_screen=screen_surface;auto resident_expected=expected;
        screen_surface=canvases[1];last_transfer=partial;
        verify(reinterpret_cast<Fill>(screen_surface->vtable[17])(screen_surface,&full,int(0x8000001fu))==0,"CPU-owned UI while map owner active");
        verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&partial,nullptr,0)==1,"CPU UI shares resident final presenter");
        for(int y=partial.top;y<partial.bottom;++y)for(int x=partial.left;x<partial.right;++x)expected[y*w+x]=0xff0000ffu;
        capture_display(expected);verify(raw_unchanged(),"CPU UI transfer does not restore GPU map storage");
        auto resident_bits=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(original_bits)(resident_screen);
        int resident_stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(resident_screen)+0x40);
        bool screen_untouched=true;for(int y=0;y<h;++y)for(int x=0;x<w;++x)screen_untouched=screen_untouched&&resident_bits[y*resident_stride+x]==0x1234;
        reinterpret_cast<Release>(original_release)(resident_screen,1);verify(screen_untouched,"CPU UI transfer leaves resident screen CPU bytes untouched");
        screen_surface=resident_screen;last_transfer=full;expected=resident_expected;
        verify(live(C3X_NATIVE_IMAGE_PRESENT,screen_surface,graph,&full,nullptr,0)==1,"return to resident screen without rebuilding map");capture_display(expected);
        char tactical_output[MAX_PATH]={};GetEnvironmentVariableA("C3X_RENDERER_TACTICAL_PREVIEW",tactical_output,sizeof(tactical_output));
        if(tactical_output[0]){
            verify(live(C3X_NATIVE_TACTICAL_CAPABLE,nullptr,nullptr,nullptr,nullptr,0)==1,"tactical capability on active native owner");
            auto original_unit=unit;int cx=w*3/4,cy=h/2,tx=w/4,ty=h*2/3;
            unit.body_x=cx-unit.sprite_width*unit.projection_scale_milli/2000;
            unit.body_y=cy-unit.sprite_height*unit.projection_scale_milli/2000;
            auto draw_body=[&]{int bounds[4]={};verify(live(C3X_NATIVE_UNIT_DRAW,screen_surface,screen_surface,&unit,bounds,1)==1,"tactical under-unit ordering");};
            auto show=[&]{last_transfer=full;final_ui_drawn=false;patch_JGL_present_screen(&full);};
            auto save_tactical=[&](char const* suffix,std::vector<unsigned> const& pixels){
                auto name=std::string(tactical_output)+suffix;FILE* f=nullptr;verify(!fopen_s(&f,name.c_str(),"wb"),"tactical preview file");
                BITMAPFILEHEADER a={};a.bfType=0x4d42;a.bfOffBits=sizeof(a)+sizeof(BITMAPINFOHEADER);a.bfSize=a.bfOffBits+w*h*4;
                BITMAPINFOHEADER b={};b.biSize=sizeof(b);b.biWidth=w;b.biHeight=-h;b.biPlanes=1;b.biBitCount=32;
                fwrite(&a,sizeof(a),1,f);fwrite(&b,sizeof(b),1,f);fwrite(pixels.data(),4,pixels.size(),f);fclose(f);
            };
            copy(live_images[0],screen_surface,full);draw_body();show();std::vector<unsigned> baseline;capture_display(expected,&baseline);
            auto ring=[&]{int p[4]={cx,cy,live_frame.tile_width,1};verify(live(C3X_NATIVE_TACTICAL_RING,screen_surface,nullptr,p,nullptr,0)==1,"copied selected marker");};
            copy(live_images[0],screen_surface,full);ring();draw_body();
            c3x_renderer_tactical_view_v1 view={live_frame.tile_width,live_frame.tile_width,0,0};
            verify(live(C3X_NATIVE_TACTICAL_ROUTE_BEGIN,screen_surface,nullptr,&view,nullptr,0)==1,"route lexical capture begin");
            // These are the same actual native line/text hooks used by Civ III.
            auto line=reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,int,int,int,int)>(screen_surface->vtable[25]);
            verify(line(screen_surface,cx,cy,w/2,ty,int(0x80007c00u),1)==0&&line(screen_surface,w/2,ty,tx,ty,int(0x80007c00u),1)==0,"native route line capture");
            int destination[2]={tx,ty};verify(live(C3X_NATIVE_TACTICAL_TARGET,screen_surface,nullptr,destination,nullptr,0)==1,"native destination capture");
            verify(reinterpret_cast<int(__thiscall*)(JGL_Image*,int,int,char const*,int)>(screen_surface->vtable[46])(screen_surface,tx,ty,"2",1)==0,"native authoritative turn label capture");
            verify(live(C3X_NATIVE_TACTICAL_ROUTE_END,screen_surface,nullptr,nullptr,nullptr,0)==1,"route batch publish");
            show();std::vector<unsigned> marked;capture_display(expected,&marked);verify(marked!=baseline,"tactical GPU marks change display");save_tactical("-route.bmp",marked);
            auto visual=reinterpret_cast<int(*)()>(GetProcAddress(renderer_module,"c3x_renderer_gpu_visual_frame"));
            auto status=reinterpret_cast<int(*)(c3x_renderer_visual_status_v1*)>(GetProcAddress(renderer_module,"c3x_renderer_gpu_visual_status"));
            c3x_renderer_visual_status_v1 before={sizeof(before)},after={sizeof(after)};verify(status(&before)==1,"tactical retained state");
            live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,1);
            auto initial_native_events=events.size();
            SetWindowPos(window,HWND_TOPMOST,0,0,w,h,SWP_NOACTIVATE|SWP_SHOWWINDOW);
            auto tactical_dwm=LoadLibraryA("dwmapi.dll");auto finish=reinterpret_cast<HRESULT(WINAPI*)()>(GetProcAddress(tactical_dwm,"DwmFlush"));
            HDC desktop=GetDC(nullptr),capture_dc=CreateCompatibleDC(desktop);void* pixels=nullptr;
            BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=w;info.bmiHeader.biHeight=-h;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
            auto bitmap=CreateDIBSection(capture_dc,&info,DIB_RGB_COLORS,&pixels,nullptr,0);auto previous=SelectObject(capture_dc,bitmap);
            std::vector<unsigned> first_motion;bool moved=false;
            LARGE_INTEGER tactical_frequency={};QueryPerformanceFrequency(&tactical_frequency);
            for(unsigned n=0;n<12;++n){Sleep(80);LARGE_INTEGER start={},submitted={},done={};QueryPerformanceCounter(&start);
                verify(visual()==1,"independent tactical frame");QueryPerformanceCounter(&submitted);verify(SUCCEEDED(finish()),"tactical desktop completion");QueryPerformanceCounter(&done);
                std::printf("TACTICAL_VISUAL_SAMPLE request_ms=%.3f desktop_ms=%.3f begin_qpc=%lld end_qpc=%lld\n",
                    1000.*double(submitted.QuadPart-start.QuadPart)/tactical_frequency.QuadPart,1000.*double(done.QuadPart-start.QuadPart)/tactical_frequency.QuadPart,start.QuadPart,submitted.QuadPart);
                if(w<=GetSystemMetrics(SM_CXSCREEN)&&h<=GetSystemMetrics(SM_CYSCREEN)){
                    verify(BitBlt(capture_dc,0,0,w,h,desktop,0,0,SRCCOPY)!=FALSE,"capture completed tactical frame");GdiFlush();
                    std::vector<unsigned> movie(static_cast<unsigned*>(pixels),static_cast<unsigned*>(pixels)+w*h);
                    if(first_motion.empty())first_motion=movie;
                    // Compare just the selected marker's bottom arc/markers;
                    // the frozen unit and map content cannot certify this motion.
                    int left=std::max(0,cx-live_frame.tile_width/3),right=std::min(w,cx+live_frame.tile_width/3);
                    int top=std::max(0,cy+live_frame.tile_height/4),bottom=std::min(h,cy+live_frame.tile_height/2);
                    for(int y=top;y<bottom;++y)for(int x=left;x<right;++x)moved|=(first_motion[y*w+x]&0xffffff)!=(movie[y*w+x]&0xffffff);
                    char suffix[40];sprintf_s(suffix,"-motion-%02u.bmp",n);save_tactical(suffix,movie);
                }
            }
            SelectObject(capture_dc,previous);DeleteObject(bitmap);DeleteDC(capture_dc);ReleaseDC(nullptr,desktop);FreeLibrary(tactical_dwm);
            verify(status(&after)==1&&after.frames-before.frames==12&&after.retained_bytes<=128ll*1024*1024,"bounded independent tactical history");
            verify(events.size()==initial_native_events,"tactical animation makes no native draw calls");
            if(w<=GetSystemMetrics(SM_CXSCREEN)&&h<=GetSystemMetrics(SM_CYSCREEN))verify(moved,"selected marker visibly rotates");
            live(C3X_NATIVE_VISUAL_POLICY,nullptr,nullptr,nullptr,nullptr,0);
            copy(live_images[0],screen_surface,full);draw_body();show();capture_display(baseline); // cancellation removes both histories exactly
            copy(live_images[0],screen_surface,full);
            verify(live(C3X_NATIVE_TACTICAL_GRID,screen_surface,nullptr,&live_frame,nullptr,1)==1,"native-setting grid draw");draw_body();show();
            std::vector<unsigned> grid;capture_display(expected,&grid);verify(grid!=baseline,"grid visible");save_tactical("-grid.bmp",grid);
            copy(live_images[0],screen_surface,full);verify(live(C3X_NATIVE_TACTICAL_GRID,screen_surface,nullptr,&live_frame,nullptr,0)==1,"grid off");draw_body();show();capture_display(baseline);
            expected=baseline;unit=original_unit;
            std::puts("PASS tactical native composition: scoped native route/turn capture, under-unit marker, clipped grid, exact cancellation/grid-off, no terrain or unit content rebuild");
        }
        if(GetEnvironmentVariableA("C3X_RENDERER_NATIVE_RECOVERY_TEST",nullptr,0)){
            custom_renderer_native_view current={};current.width=w;current.height=h;
            current.tile_width=current.native_width=live_frame.tile_width;
            auto target=current;target.camera_x=57;target.min_x=3;
            for(int recovery=0;recovery<4;++recovery){
                current.camera_x=0;current.min_x=0;
                verify(navigate(C3X_NAV_REQUEST,live_images[0],&target,&live_request)==C3X_RENDERER_RESULT_PENDING,"recovery begins copied native demand");
                if(recovery==0){
                    verify(navigate(C3X_NAV_DISCARD,live_images[0],&current,nullptr)==C3X_RENDERER_RESULT_SUPERSEDED&&!current.camera_x,"explicit cancellation keeps displayed camera");
                }else if(recovery==1){
                    verify(navigate(C3X_NAV_BARRIER,live_images[0],&current,nullptr)==C3X_RENDERER_RESULT_OK&&current.camera_x==57,"config-off barrier returns requested camera without coverage");
                }else{
                    if(recovery==3){
                        int code=C3X_RENDERER_RESULT_PENDING;auto deadline=GetTickCount64()+30000;
                        while(code==C3X_RENDERER_RESULT_PENDING&&GetTickCount64()<deadline){code=navigate(C3X_NAV_POLL,live_images[0],&current,nullptr);if(code==C3X_RENDERER_RESULT_PENDING)Sleep(1);}
                        verify(code==C3X_RENDERER_RESULT_OK&&current.camera_x==57,"ready-but-unvalidated recovery point");
                    }
                    auto before=current;reset();
                    verify(navigate(C3X_NAV_POLL,live_images[0],&current,nullptr)==C3X_RENDERER_RESULT_SUPERSEDED&&!std::memcmp(&before,&current,sizeof(current)),"reset cannot revive pending or ready camera");
                    live_active=false;preserve_gdi_display=true;capture_display(expected);preserve_gdi_display=false;live_active=true;
                }
                verify(exact_native_map(C3X_NATIVE_MAP_COMMIT,live_images[0],nullptr,nullptr)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"retired native transaction cannot commit");
                // A fresh authoritative request recovers using the same native
                // surfaces; worker tickets/sessions cannot alias the old view.
                c3x_renderer_camera_view_v1 rebuilt={C3X_RENDERER_CAMERA_VIEW_VERSION,sizeof(rebuilt)};
                verify(exact_native_map(C3X_NATIVE_MAP_PREPARE,live_images[0],&live_request,&rebuilt)==C3X_RENDERER_RESULT_OK,"fresh capture recovers after retirement");
                verify(exact_native_map(C3X_NATIVE_MAP_COMMIT,live_images[0],nullptr,nullptr)==C3X_RENDERER_RESULT_OK,"fresh recovery commit");
            }
            std::puts("PASS native async recovery: cases=4 cancellation=1 config_off_barrier=1 pending_reset=1 ready_reset=1 recreated=1 stale_commit=0");
        }
        {
            // Recovery may have legitimately retired the old screen; the map
            // was freshly committed above and is the current resident target.
            OpenGLRenderer context;PCX_Image target;target.JGL.Image=live_images[0];
            verify(patch_OpenGLRenderer_initialize(&context,0,&target)==0&&!context.initialized,"GPU line scope before configuration-off");
            patch_OpenGLRenderer_set_line_width(&context,0,2);
            state.current_config.enable_custom_rendering=false;
            patch_OpenGLRenderer_draw_line(&context,0,3,3,23,3);
            verify(context.initialized==1&&context.drawn==1&&!state.custom_renderer_line_owner&&!state.custom_renderer_native_image,
                "configuration-off inside line scope drains before original native initialization/draw");
            state.current_config.enable_custom_rendering=true;state.custom_renderer_native_image=live;
            std::puts("PASS native outline configuration-off: mid_scope_drain=1 native_initialize=1 native_draw=1");
        }
        // Reset must drain before the renderer retires its image session, while
        // the native surfaces and final window still exist.
        reset();live_active=false;preserve_gdi_display=true;capture_display(expected);preserve_gdi_display=false;
        #include "native_frame_benchmark.h"
        SelectObject(label_dc,previous_font);DeleteObject(label_font);
        for(auto image:live_images)reinterpret_cast<Destroy>(image->vtable[0])(image,1);
        screen_surface=canvases[1];screen_image=screen_surface;screen.JGL.Image=screen_surface;pcx.image=screen_surface;
        if(navigation_test)std::printf("PASS native navigation: fresh_capture=1 deferred_camera=1 native_composition=1\n");
        if(native_camera_test){verify(native_camera_requests>0 && native_camera_pending>0 && native_camera_messages>0 && native_camera_wakes>0,"native async opportunities exercised");
            std::printf("PASS nonblocking native camera: requests=%u pending=%u messages=%u wakes=%u exact_composition=1 pending_coverage=0\n",native_camera_requests,native_camera_pending,native_camera_messages,native_camera_wakes);}
        std::puts("PASS production native map owner: prepare/validate/commit, cancelled ownership, unchanged CPU map, native copies, units, cached text, exact compiled final display, next-view session and reset handoff");
    }
    set_custom_renderer_native_probe(nullptr);native_present_image=nullptr;present_fn=native_present;screen.JGL.Image=nullptr;
    *reinterpret_cast<void**>(static_cast<char*>(graph)+0x148)=old_screen;*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138)=old_dc;
    for(auto canvas:canvases)reinterpret_cast<Destroy>(canvas->vtable[0])(canvas,1);reinterpret_cast<Destroy>(root->vtable[0])(root,1);
    reinterpret_cast<void(__thiscall*)(void*)>(gt[49])(graph);
    for(auto palette:palettes)reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(reinterpret_cast<char*>(jgl)+0x3cf10)(palette,1);
    patch_unload_jgl_lib();ReleaseDC(window,dc);DestroyWindow(window);UnregisterClassA(wc.lpszClassName,wc.hInstance);
    std::puts("PASS native screen transfer: full-color map/copy/popup family, final UI ordering, exact displayed RGB, partial present, original GDI fallback; no execution readbacks before fallback");return true;
}
