#pragma once
// Cache GDI text response at 17 background levels per channel. This preserves
// native shaping and smoothing with bounded interpolation error at glyph edges.
// Preparation uses synthetic backgrounds only; native/map pixels are never read.
#include <windows.h>
#include <array>
#include <vector>
#include <string>
#include <map>
#include <cstring>
#include <algorithm>
#include <cstdint>
namespace c3x_native_text {
enum class Refusal { none, arguments, dc_mapping, dc_transform, font, colors,
    alignment, clip, extent, raster_bounds, dib, text_out, curve_count,
    cache_bytes, gpu_admission, gpu_upload, anchor_range, submission };
struct Diagnostic { Refusal reason=Refusal::none;unsigned width=0,height=0,curves=0; };
inline char const* refusal_name(Refusal reason){
    switch(reason){
    case Refusal::none:return "none";case Refusal::arguments:return "arguments";
    case Refusal::dc_mapping:return "dc_mapping";case Refusal::dc_transform:return "dc_transform";
    case Refusal::font:return "font";case Refusal::colors:return "colors";
    case Refusal::alignment:return "alignment";case Refusal::clip:return "clip";
    case Refusal::extent:return "extent";case Refusal::raster_bounds:return "raster_bounds";
    case Refusal::dib:return "dib";case Refusal::text_out:return "text_out";
    case Refusal::curve_count:return "curve_count";case Refusal::cache_bytes:return "cache_bytes";
    case Refusal::gpu_admission:return "gpu_admission";case Refusal::gpu_upload:return "gpu_upload";
    case Refusal::anchor_range:return "anchor_range";case Refusal::submission:return "submission";
    }return "unknown";
}
inline bool refuse(Diagnostic* diagnostic,Refusal reason){if(diagnostic)diagnostic->reason=reason;return false;}
struct Raster {
    unsigned width=0,height=0;int left=0,top=0;
    std::vector<unsigned> pixels,curves; // pixel: three 10-bit curve IDs, bit30 marks glyph/background coverage
};
struct Dib {
    HDC dc=nullptr;HBITMAP bitmap=nullptr;HGDIOBJ previous=nullptr;void* pixels=nullptr;
    Dib(unsigned width,unsigned height){
        dc=CreateCompatibleDC(nullptr);BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
        info.bmiHeader.biWidth=LONG(width);info.bmiHeader.biHeight=-LONG(height);info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;
        if(dc)bitmap=CreateDIBSection(dc,&info,DIB_RGB_COLORS,&pixels,nullptr,0);
        if(bitmap)previous=SelectObject(dc,bitmap);
    }
    ~Dib(){if(previous)SelectObject(dc,previous);if(bitmap)DeleteObject(bitmap);if(dc)DeleteDC(dc);}
    Dib(Dib const&)=delete;Dib& operator=(Dib const&)=delete;
};
struct State {
    LOGFONTA font={};COLORREF foreground=0,background=0;unsigned align=0;int mode=0;
    bool operator==(State const& b)const{return !std::memcmp(&font,&b.font,sizeof font)&&foreground==b.foreground&&background==b.background&&align==b.align&&mode==b.mode;}
};
inline bool capture(HDC dc,State& font_state,Diagnostic* diagnostic=nullptr){
    POINT a={},b={};XFORM transform={};
    if(!dc||GetMapMode(dc)!=MM_TEXT||GetLayout(dc)!=0||GetTextCharacterExtra(dc)!=0||
       !GetViewportOrgEx(dc,&a)||!GetWindowOrgEx(dc,&b)||a.x!=b.x||a.y!=b.y)return refuse(diagnostic,Refusal::dc_mapping);
    if(GetGraphicsMode(dc)==GM_ADVANCED&&(!GetWorldTransform(dc,&transform)||transform.eM11!=1||transform.eM22!=1||transform.eM12||transform.eM21||transform.eDx||transform.eDy))return refuse(diagnostic,Refusal::dc_transform);
    if(GetObjectA(GetCurrentObject(dc,OBJ_FONT),sizeof font_state.font,&font_state.font)!=sizeof font_state.font||font_state.font.lfEscapement||font_state.font.lfOrientation)return refuse(diagnostic,Refusal::font);
    font_state.foreground=GetTextColor(dc);font_state.background=GetBkColor(dc);font_state.align=GetTextAlign(dc);font_state.mode=GetBkMode(dc);
    if(font_state.foreground==CLR_INVALID||font_state.background==CLR_INVALID||font_state.align==GDI_ERROR||(font_state.mode!=OPAQUE&&font_state.mode!=TRANSPARENT))return refuse(diagnostic,Refusal::colors);
    return true;
}
inline void setup(HDC dc,HGDIOBJ font,State const& s){SelectObject(dc,font);SetTextColor(dc,s.foreground);SetBkColor(dc,s.background);SetBkMode(dc,s.mode);SetTextAlign(dc,TA_LEFT|TA_TOP|TA_NOUPDATECP);}
inline unsigned channel(unsigned word,unsigned c,bool green6){return (word>>(c==0?0:c==1?5:green6?11:10))&((c==1&&green6)?63:31);}
inline unsigned expand(unsigned word,bool green6){unsigned b=channel(word,0,green6),g=channel(word,1,green6),r=channel(word,2,green6);
    return ((b<<3)|(b>>2))|((green6?((g<<2)|(g>>4)):((g<<3)|(g>>2)))<<8)|(((r<<3)|(r>>2))<<16);}
// A complete shaped string may exceed one response tile. Keep temporary GDI
// and 17-sample response storage at 16K pixels while admitting a 32K-pixel
// packed result (<=128 KiB plus <=68 KiB curve payload). Each strip shapes the
// whole string, shifted by an integer row offset, so kerning and glyph order
// remain native. No string fragmentation and no destination pixels are used.
inline constexpr unsigned response_tile_pixels=16384,maximum_raster_pixels=32768;
inline bool compile(HDC source,State const& font_state,char const* text,unsigned count,Raster& out,Diagnostic* diagnostic=nullptr){
    if(!count){out={};return true;}if(!text||count>1024)return refuse(diagnostic,Refusal::arguments);
    SIZE extent={};TEXTMETRICA metrics={};
    if(!GetTextExtentPoint32A(source,text,int(count),&extent)||!GetTextMetricsA(source,&metrics)||extent.cx<0||metrics.tmHeight<1)return refuse(diagnostic,Refusal::extent);
    auto margin64=std::int64_t(metrics.tmHeight)+std::abs(std::int64_t(metrics.tmOverhang));
    auto width64=std::int64_t(extent.cx)+2*margin64,height64=std::int64_t(metrics.tmHeight)+2*margin64;
    if(diagnostic){diagnostic->width=unsigned(std::min(width64,std::int64_t(UINT32_MAX)));diagnostic->height=unsigned(std::min(height64,std::int64_t(UINT32_MAX)));}
    if(width64<1||width64>2240||height64<1||height64>1260||width64*height64>maximum_raster_pixels)return refuse(diagnostic,Refusal::raster_bounds);
    unsigned w=unsigned(width64),h=unsigned(height64);int margin=int(margin64);
    using Curve=std::array<unsigned char,17>;
    std::map<Curve,unsigned> unique;out={};out.width=w;out.height=h;out.left=-margin;out.top=-margin;out.pixels.resize(w*h);
    auto font=GetCurrentObject(source,OBJ_FONT);unsigned rows=std::min(h,response_tile_pixels/w);
    for(unsigned top=0;top<h;top+=rows){
        unsigned tile_height=std::min(rows,h-top),size=w*tile_height;
        Dib tile(w,tile_height);if(!tile.pixels)return refuse(diagnostic,Refusal::dib);
        setup(tile.dc,font,font_state);std::vector<Curve> response(std::size_t(size)*3);
        for(unsigned sample=0;sample<17;++sample){
            unsigned level=std::min(sample*16u,255u);auto wide=static_cast<unsigned*>(tile.pixels);
            std::fill(wide,wide+size,level*0x010101u);
            if(!TextOutA(tile.dc,margin,margin-int(top),text,int(count)))return refuse(diagnostic,Refusal::text_out);GdiFlush();
            for(unsigned n=0;n<size;++n)for(unsigned c=0;c<3;++c)response[n*3+c][sample]=static_cast<unsigned char>((wide[n]>>(c*8))&255);
        }
        for(unsigned n=0;n<size;++n){unsigned pixel=0;
            for(unsigned c=0;c<3;++c){auto const& curve=response[n*3+c];auto found=unique.find(curve);unsigned id;
                // Coverage is independent of the current map color. Preserve unused
                // native word bits only outside the glyph/opaque-background footprint.
                for(unsigned sample=0;sample<17;++sample)if(curve[sample]!=std::min(sample*16u,255u)){pixel|=1u<<30;break;}
                if(found==unique.end()){id=unsigned(unique.size());if(diagnostic)diagnostic->curves=id+1;if(id>=1024)return refuse(diagnostic,Refusal::curve_count);unique.emplace(curve,id);
                    for(auto value:curve)out.curves.push_back(value);
                }else id=found->second;pixel|=id<<(c*10);
            }out.pixels[top*w+n]=pixel;
        }
    }
    return true;
}
// Native TA_* placement is applied after compilation/cache lookup. Widen before
// subtracting the advance/ascent as well as before adding the raster margin.
inline bool place(unsigned align,int advance,int ascent,int height,int x,int y,
                  int left,int top,unsigned width,unsigned rows,RECT& area){
    auto origin_x=std::int64_t(x),origin_y=std::int64_t(y);
    // GDI centers an odd advance by rounding the half-width upward.
    if((align&TA_CENTER)==TA_CENTER)origin_x-=(std::int64_t(advance)+1)/2;else if(align&TA_RIGHT)origin_x-=advance;
    if((align&TA_BASELINE)==TA_BASELINE)origin_y-=ascent;else if(align&TA_BOTTOM)origin_y-=height;
    auto l=origin_x+left,t=origin_y+top,r=l+width,b=t+rows;
    if(l<INT32_MIN||r>INT32_MAX||t<INT32_MIN||b>INT32_MAX)return false;
    area={LONG(l),LONG(t),LONG(r),LONG(b)};return true;
}

inline unsigned apply(Raster const& raster,unsigned pixel,unsigned below,bool green6,bool detail){
    unsigned ids=raster.pixels[pixel],rgb=detail?below:expand(below,green6),result=0;
    for(unsigned c=0;c<3;++c){unsigned id=(ids>>(c*10))&1023,level=(rgb>>(c*8))&255,index=std::min(level/16,15u),fraction=level-index*16,span=index==15?15:16;
        unsigned a=raster.curves[id*17+index],b=raster.curves[id*17+index+1];
        result|=((a*(span-fraction)+b*fraction+span/2)/span)<<(8*c);
    }
    if(detail)return result|0xff000000u;
    unsigned packed=(result>>3&31)|((result>>(green6?10:11)&(green6?63:31))<<5)|((result>>19&31)<<(green6?11:10));
    return (!green6&&!(ids&(1u<<30)))?(packed|(below&32768)):packed;
}
}
