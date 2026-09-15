#define C3X_NATIVE_WORKER_TEST
#include "test_native_image_adapter.cpp"
#include "test_gpu_unit_composition.h"
#include "native_frame_workload.h"
#include "test_native_screen.h"

bool native_worker_contract(char const* path,c3x_renderer_gpu_images_fn images,c3x_renderer_gpu_frame_v1& view,c3x_renderer_gpu_render_fn render,c3x_renderer_gpu_present_fn present,c3x_renderer_camera_request_v1 const& request,
                            unsigned const* pixels,int phase_x,int phase_y,std::vector<NativeFrameSample> const& performance_frames){
    try{
        state={};capture={};events.clear();lines.clear();
        HMODULE module=nullptr;verify(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<char const*>(images),&module)!=FALSE,"worker module");
        diagnostic_dispatch=reinterpret_cast<c3x_renderer_native_image_fn>(GetProcAddress(module,"c3x_renderer_native_image"));
        verify(diagnostic_dispatch!=nullptr,"live CPU diagnostic dispatch export");
        WorkerClient gpu(images,view);
        if(native_adapter_contract(path,gpu,Id(view.map_image),pixels,view.width,phase_x,phase_y))return false;
        verify(gpu.stats().resident_bytes==std::int64_t(view.width)*view.height*4+64*48*8,"native drain leaves only immutable map and paired overlap scratch");
        auto canvas=gpu.create(16,16,Format::bgra32);
        auto before=gpu.submitted_batches();auto calls=gpu.worker_calls();
        Command fill={Kind::fill,canvas,0,{0,0,16,16},{0,0,16,16},0,0,0};
        for(unsigned n=0;n<100;++n){fill.color=0xff000000u+n;verify(gpu.submit(&fill,1),"queue draw");}
        verify(gpu.worker_calls()==calls,"100 consecutive draws do not wake worker individually");
        gpu.flush();verify(gpu.submitted_batches()==before+1,"100 consecutive draws form one worker packet");
        std::vector<unsigned> observed(256);verify(gpu.readback(canvas,observed.data(),observed.size()),"batch oracle");
        verify(std::all_of(observed.begin(),observed.end(),[&](unsigned c){return c==fill.color;}),"ordered batch result");
        // Saturated ownership rejects the next map before retiring the old one.
        std::vector<Id> saturated;
        for(unsigned n=0;n<128;++n){auto id=gpu.create(2,2,Format::bgra32);if(!id)break;saturated.push_back(id);}
        verify(!saturated.empty()&&saturated.size()<128,"bounded GPU image admission");
        auto unavailable=view;c3x_renderer_output_v1 rejected_map={C3X_RENDERER_API_VERSION,sizeof(rejected_map)};
        verify(render(&request,&unavailable,&rejected_map)==C3X_RENDERER_RESULT_BAD_ARGUMENT,"map image limit selects CPU fallback without retiring the old ticket");
        verify(gpu.readback(canvas,observed.data(),observed.size())&&std::all_of(observed.begin(),observed.end(),[&](unsigned c){return c==fill.color;}),"failed publication preserves preceding native pixels and ticket");
        for(auto id:saturated)gpu.destroy(id);
        auto next=view;next.ticket++;next.session++;bool rejected=false;
        try{gpu.advance(next);}catch(std::exception const&){rejected=true;}verify(rejected,"cannot adopt handles into another session");
        c3x_renderer_output_v1 metadata={C3X_RENDERER_API_VERSION,sizeof(metadata)};
        verify(render(&request,&next,&metadata)==C3X_RENDERER_RESULT_OK&&next.session==view.session&&next.ticket!=view.ticket,"next map preserves image session");
        gpu.advance(next);view=next;gpu.readback(canvas,observed.data(),observed.size());
        verify(std::all_of(observed.begin(),observed.end(),[&](unsigned c){return c==fill.color;}),"worker client retains UI image across map frames");
        // The production unit path submits an independent body job. Its CPU
        // compatibility output must not destroy retained GPU native backgrounds.
        auto unit_draw=reinterpret_cast<c3x_renderer_unit_draw_background_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_background"));
        verify(unit_draw!=nullptr,"production unit draw");
        BITMAPINFO unit_info={};unit_info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);unit_info.bmiHeader.biWidth=384;unit_info.bmiHeader.biHeight=-384;
        unit_info.bmiHeader.biPlanes=1;unit_info.bmiHeader.biBitCount=32;void* unit_pixels=nullptr;
        auto unit_dc=CreateCompatibleDC(nullptr);auto unit_bitmap=CreateDIBSection(unit_dc,&unit_info,DIB_RGB_COLORS,&unit_pixels,nullptr,0);
        verify(unit_dc&&unit_bitmap&&unit_pixels,"unit interleave destination");auto previous_unit=SelectObject(unit_dc,unit_bitmap);
        std::fill_n(static_cast<unsigned*>(unit_pixels),384*384,0xff565b62u);
        c3x_renderer_unit_v1 unit={};unit.struct_size=sizeof(unit);strcpy_s(unit.unit_key,"PRTO_Warrior");
        unit.unit_id=731;unit.action=1;unit.direction=3;unit.frame_count=16;unit.action_cursor=7;
        unit.sprite_width=unit.sprite_height=191;unit.body_x=unit.body_y=32;unit.hour=12;unit.display_color_rgb=0x205bdd;
        unit.presentation_frequency=1000000;unit.presentation_time_ticks=1000000;
        verify(unit_draw(&unit,unit_dc,unit_dc)==C3X_RENDERER_RESULT_OK,"actual unit job between GPU native operations");
        verify(gpu.readback(canvas,observed.data(),observed.size())&&std::all_of(observed.begin(),observed.end(),[&](unsigned c){return c==fill.color;}),"native GPU pixels and session survive unit job");
        gpu_unit_contract(gpu,module,view,unit);
        SelectObject(unit_dc,previous_unit);DeleteObject(unit_bitmap);DeleteDC(unit_dc);
        auto cpu_render=reinterpret_cast<c3x_renderer_render_view_fn>(GetProcAddress(module,"c3x_renderer_render_view"));
        c3x_renderer_output_v1 cpu_output={C3X_RENDERER_API_VERSION,sizeof(cpu_output)};
        verify(cpu_render&&cpu_render(&request,&cpu_output)==C3X_RENDERER_RESULT_OK,"interleaved CPU map publication");
        verify(gpu.readback(canvas,observed.data(),observed.size())&&std::all_of(observed.begin(),observed.end(),[&](unsigned c){return c==fill.color;}),"native GPU pixels and session survive CPU map publication");
        gpu.destroy(canvas);
        // Every input channel/8x8 threshold in both destination formats. This
        // compatibility seam uses the existing CPU blitter's approved rounding.
        std::vector<unsigned> gradient(2048*8),actual(gradient.size());
        for(unsigned y=0;y<8;++y)for(unsigned x=0;x<2048;++x){auto c=x/8;gradient[y*2048+x]=0xff000000u|c|(c<<8)|(c<<16);}
        auto source=gpu.create(2048,8,Format::bgra32);gpu.upload(source,1,gradient.data(),gradient.size());
        auto r5=c3x_renderer::color_rounding_table(31),r6=c3x_renderer::color_rounding_table(63);
        for(auto format:{Format::rgb555,Format::rgb565}){
            auto target=gpu.create(2048,8,format);
            Command convert={Kind::quantize,target,source,{0,0,2048,8},{0,0,2048,8},0,0,7u|(5u<<3)};
            gpu.submit(&convert,1);gpu.readback(target,actual.data(),actual.size());
            for(unsigned y=0;y<8;++y)for(unsigned x=0;x<2048;++x){auto t=c3x_renderer::color_threshold(x-7,y-5),c=x/8;
                unsigned expected=(r5[t][c]>>3)|(((format==Format::rgb565?r6[t][c]>>2:r5[t][c]>>3))<<5)|((r5[t][c]>>3)<<(format==Format::rgb565?11:10));
                verify(actual[y*2048+x]==expected,"all native map rounding thresholds exact");}
            // Native GDI is the independent 16 -> 32 color oracle. Preserve
            // these native bytes for the UI; never quantize the full-color map.
            struct NativeInfo {BITMAPINFOHEADER header;DWORD masks[3];} native={};
            native.header.biSize=sizeof(BITMAPINFOHEADER);native.header.biWidth=2048;native.header.biHeight=-8;
            native.header.biPlanes=1;native.header.biBitCount=16;native.header.biCompression=BI_BITFIELDS;
            native.masks[0]=format==Format::rgb565?0xf800:0x7c00;native.masks[1]=format==Format::rgb565?0x7e0:0x3e0;native.masks[2]=0x1f;
            auto dc16=CreateCompatibleDC(nullptr),dc32=CreateCompatibleDC(nullptr);void* bits16=nullptr;void* bits32=nullptr;
            auto b16=CreateDIBSection(dc16,reinterpret_cast<BITMAPINFO*>(&native),DIB_RGB_COLORS,&bits16,nullptr,0);
            BITMAPINFO full={};full.bmiHeader=native.header;full.bmiHeader.biBitCount=32;full.bmiHeader.biCompression=BI_RGB;
            auto b32=CreateDIBSection(dc32,&full,DIB_RGB_COLORS,&bits32,nullptr,0);verify(dc16&&dc32&&b16&&b32,"native color oracle surfaces");
            auto previous16=SelectObject(dc16,b16),previous32=SelectObject(dc32,b32);
            for(unsigned i=0;i<actual.size();++i)static_cast<unsigned short*>(bits16)[i]=static_cast<unsigned short>(actual[i]);
            verify(BitBlt(dc32,0,0,2048,8,dc16,0,0,SRCCOPY)!=FALSE,"native color expansion");GdiFlush();
            auto expanded=gpu.create(2048,8,Format::bgra32);Command expand={Kind::expand,expanded,target,{0,0,2048,8},{0,0,2048,8},0,0,65536};
            gpu.submit(&expand,1);gpu.readback(expanded,actual.data(),actual.size());
            for(unsigned i=0;i<actual.size();++i)if(actual[i]!=(static_cast<unsigned*>(bits32)[i]|0xff000000u)){
                std::fprintf(stderr,"native expansion mismatch format=%d packed=%04x expected=%08x actual=%08x\n",int(format),unsigned(static_cast<unsigned short*>(bits16)[i]),static_cast<unsigned*>(bits32)[i]|0xff000000u,actual[i]);
                verify(false,"555/565 expansion matches native GDI");}
            std::vector<unsigned> covered(actual.size());
            for(unsigned i=0;i<covered.size();++i)covered[i]=i%17?65536u|static_cast<unsigned short*>(bits16)[i]:0u;
            auto decoded=gpu.create(2048,8,Format::bgra32);gpu.upload(decoded,1,covered.data(),covered.size());
            Command clear={Kind::fill,expanded,0,{0,0,2048,8},{0,0,2048,8},0,0,0xff123456u};gpu.submit(&clear,1);
            Command sprite={Kind::native_sprite,expanded,decoded,{0,0,2048,8},{0,0,2048,8},0,0,format==Format::rgb565?2u:1u};
            gpu.submit(&sprite,1);gpu.readback(expanded,actual.data(),actual.size());
            for(unsigned i=0;i<actual.size();++i)verify(actual[i]==(i%17?(static_cast<unsigned*>(bits32)[i]|0xff000000u):0xff123456u),"native sprite coverage and 555/565 full-color expansion");
            gpu.destroy(decoded);
            SelectObject(dc16,previous16);SelectObject(dc32,previous32);DeleteObject(b16);DeleteObject(b32);DeleteDC(dc16);DeleteDC(dc32);
            gpu.destroy(expanded);gpu.destroy(target);
        }
        gpu.destroy(source);

        auto live=reinterpret_cast<c3x_renderer_native_image_fn>(GetProcAddress(module,"c3x_renderer_native_image"));
        auto render_view=reinterpret_cast<c3x_renderer_render_view_fn>(GetProcAddress(module,"c3x_renderer_render_view"));
        verify(native_screen_contract(path,gpu,view,present,pixels,phase_x,phase_y,live,render_view,request,reinterpret_cast<void(*)()>(GetProcAddress(module,"c3x_renderer_reset")),performance_frames),"native final screen contract");
        std::printf("PASS native GPU worker transport: actual hooks, full-color map/native UI composition, exact 555/565 compatibility/expansion, CPU barriers, bounded packet reuse; batches=%llu worker_calls=%llu\n",gpu.submitted_batches(),gpu.worker_calls());
        return true;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL native GPU worker transport: %s\n",e.what());return false;}
}
