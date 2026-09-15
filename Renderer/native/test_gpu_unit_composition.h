// Uses the production CPU blitter as the native-word/coverage oracle.
#include "build/native_unit_blitter.h"
struct UnitOracleDib {
    HDC dc=nullptr;HBITMAP bitmap=nullptr;HGDIOBJ previous=nullptr;void* pixels=nullptr;
    UnitOracleDib(unsigned w,unsigned h,unsigned format){
        struct Info {BITMAPINFOHEADER h;DWORD masks[3];} i={};i.h.biSize=sizeof(i.h);i.h.biWidth=w;i.h.biHeight=-int(h);
        i.h.biPlanes=1;i.h.biBitCount=format?16:32;i.h.biCompression=format?BI_BITFIELDS:BI_RGB;
        i.masks[0]=format==2?0xf800:0x7c00;i.masks[1]=format==2?0x7e0:0x3e0;i.masks[2]=0x1f;
        dc=CreateCompatibleDC(nullptr);bitmap=CreateDIBSection(dc,reinterpret_cast<BITMAPINFO*>(&i),DIB_RGB_COLORS,&pixels,nullptr,0);
        verify(dc&&bitmap&&pixels,"unit oracle DIB");previous=SelectObject(dc,bitmap);
    }
    ~UnitOracleDib(){if(previous)SelectObject(dc,previous);if(bitmap)DeleteObject(bitmap);if(dc)DeleteDC(dc);}
};
void gpu_unit_contract(WorkerClient& gpu,HMODULE module,c3x_renderer_gpu_frame_v1 const& view,c3x_renderer_unit_v1 unit){
    auto draw=reinterpret_cast<c3x_renderer_gpu_unit_fn>(GetProcAddress(module,"c3x_renderer_gpu_unit"));
    auto native=reinterpret_cast<c3x_renderer_unit_draw_expanded_fn>(GetProcAddress(module,"c3x_renderer_unit_draw_expanded"));
    verify(draw&&native,"production GPU/native unit exports");
    constexpr unsigned w=384,h=384,n=w*h;
    for(unsigned format=1;format<=2;++format){
        auto f=format==2?Format::rgb565:Format::rgb555;unsigned key=format==2?0xf81f:0x7c1f;
        UnitOracleDib packed(w,h,format),under(w,h,format),full(w,h,0),full_under(w,h,0);
        std::vector<unsigned> words(n),ground(n),colors(n),ground_colors(n),actual(n),body(n);
        for(unsigned i=0;i<n;++i){words[i]=i%7?(i*113)&(format==2?65535:32767):key;ground[i]=i%19?(i*719)&(format==2?65535:32767):key;
            static_cast<unsigned short*>(packed.pixels)[i]=static_cast<unsigned short>(words[i]);static_cast<unsigned short*>(under.pixels)[i]=static_cast<unsigned short>(ground[i]);}
        verify(BitBlt(full.dc,0,0,w,h,packed.dc,0,0,SRCCOPY)&&BitBlt(full_under.dc,0,0,w,h,under.dc,0,0,SRCCOPY),"native unit background expansion");GdiFlush();
        for(unsigned i=0;i<n;++i){colors[i]=(static_cast<unsigned*>(full.pixels)[i]&0xfefefe)|0xff010101u;
            ground_colors[i]=(static_cast<unsigned*>(full_under.pixels)[i]&0xfefefe)|0xff010101u;
            unsigned a=i%256;body[i]=(a<<24)|((i%251*a/255)<<16)|((i%137*a/255)<<8)|(i%73*a/255);}
        auto d=gpu.create(w,h,f),b=gpu.create(w,h,f),detail=gpu.create(w,h,Format::bgra32),bd=gpu.create(w,h,Format::bgra32),source=gpu.create(w,h,Format::bgra32);
        verify(d&&b&&detail&&bd&&source,"GPU unit pair admission");gpu.upload(source,1,body.data(),body.size());
        for(unsigned phase=0;phase<8;++phase){
            // Fresh native words and fuller-color map inputs, independent or
            // aliased underlay. The first two cases cover every alpha value.
            gpu.upload(d,phase+1,words.data(),n);gpu.upload(b,phase+1,ground.data(),n);
            gpu.upload(detail,phase+1,colors.data(),n);gpu.upload(bd,phase+1,ground_colors.data(),n);
            std::copy(colors.begin(),colors.end(),static_cast<unsigned*>(full.pixels));std::copy(ground_colors.begin(),ground_colors.end(),static_cast<unsigned*>(full_under.pixels));
            for(unsigned i=0;i<n;++i){static_cast<unsigned short*>(packed.pixels)[i]=static_cast<unsigned short>(words[i]);static_cast<unsigned short*>(under.pixels)[i]=static_cast<unsigned short>(ground[i]);}
            RECT clip={9,7,361,373};IntersectClipRect(packed.dc,clip.left,clip.top,clip.right,clip.bottom);IntersectClipRect(full.dc,clip.left,clip.top,clip.right,clip.bottom);
            auto calls=gpu.stats().readbacks;
            if(phase<2){
                NativeUnitBlitter oracle;unsigned keyed=0;int x=phase?-13:0,y=phase?-9:0;
                verify(oracle.blit_pixels(body,w,h,packed.dc,x,y,phase?packed.dc:under.dc,keyed),"native alpha/key oracle");
                verify(oracle.blit_pixels(body,w,h,full.dc,x,y,phase?full.dc:full_under.dc,keyed),"full-color alpha/key oracle");
                Command over={Kind::unit_over,d,source,{x,y,x+int(w),y+int(h)},{clip.left,clip.top,clip.right,clip.bottom},0,0,0,phase?d:b,detail,phase?detail:bd};
                verify(gpu.submit(&over,1),"GPU alpha/key composition");gpu.flush();
            }else{
                auto pose=unit;pose.frame_count=4;pose.action_cursor=phase>=5?1:0;
                if(phase==3||phase==7){pose.body_x-=7;pose.body_y+=11;} // same content at another native anchor
                if(phase>=4)pose.direction=2;
                if(phase>=5){pose.hour=0;pose.season=1;}
                if(phase>=6)pose.projection_scale_milli=1250;
                int expected[4]={},bounds[4]={};
                verify(native(&pose,packed.dc,under.dc,expected)==C3X_RENDERER_RESULT_OK,"actual native unit word oracle");
                verify(native(&pose,full.dc,full_under.dc,expected)==C3X_RENDERER_RESULT_OK,"actual full-color unit oracle");
                c3x_renderer_gpu_unit_v1 target={sizeof(target),view.ticket,std::int64_t(d),std::int64_t(b),std::int64_t(detail),std::int64_t(bd),{clip.left,clip.top,clip.right,clip.bottom},0};
                verify(draw(&pose,&target,bounds)==C3X_RENDERER_RESULT_OK,"production resident unit to GPU backgrounds");
                verify(std::equal(bounds,bounds+4,expected),"GPU unit native erase bounds");
                // Refresh packet counters without performing a readback.
                auto sentinel=gpu.create(1,1,Format::bgra32);verify(sentinel!=0,"unit counter query");gpu.destroy(sentinel);
            }
            verify(gpu.stats().readbacks==calls,"unit composition never reads destination or underlay to CPU");GdiFlush();
            gpu.readback(d,actual.data(),actual.size());
            for(unsigned i=0;i<n;++i)if(actual[i]!=static_cast<unsigned short*>(packed.pixels)[i]){std::fprintf(stderr,"unit packed mismatch format=%u phase=%u i=%u expected=%x actual=%x\n",format,phase,i,unsigned(static_cast<unsigned short*>(packed.pixels)[i]),actual[i]);verify(false,"GPU unit exact native words");}
            gpu.readback(detail,actual.data(),actual.size());
            for(unsigned i=0;i<n;++i)if((actual[i]&0xffffff)!=(static_cast<unsigned*>(full.pixels)[i]&0xffffff)){std::fprintf(stderr,"unit detail mismatch format=%u phase=%u i=%u expected=%x actual=%x\n",format,phase,i,static_cast<unsigned*>(full.pixels)[i],actual[i]);verify(false,"GPU unit exact full-color composition");}
        }
        for(auto id:{d,b,detail,bd,source})gpu.destroy(id);
    }
    std::puts("PASS GPU unit composition: native 555/565, full-color map, all alpha values, keyed/aliased underlays, clipping, cold/warm resident units across anchors/direction/time/zoom; zero background readback");
}
