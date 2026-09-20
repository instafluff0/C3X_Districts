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
inline bool capture(HDC dc,State& font_state){
    POINT a={},b={};XFORM transform={};
    if(!dc||GetMapMode(dc)!=MM_TEXT||GetLayout(dc)!=0||GetTextCharacterExtra(dc)!=0||
       !GetViewportOrgEx(dc,&a)||!GetWindowOrgEx(dc,&b)||a.x!=b.x||a.y!=b.y)return false;
    if(GetGraphicsMode(dc)==GM_ADVANCED&&(!GetWorldTransform(dc,&transform)||transform.eM11!=1||transform.eM22!=1||transform.eM12||transform.eM21||transform.eDx||transform.eDy))return false;
    if(GetObjectA(GetCurrentObject(dc,OBJ_FONT),sizeof font_state.font,&font_state.font)!=sizeof font_state.font||font_state.font.lfEscapement||font_state.font.lfOrientation)return false;
    font_state.foreground=GetTextColor(dc);font_state.background=GetBkColor(dc);font_state.align=GetTextAlign(dc);font_state.mode=GetBkMode(dc);
    return font_state.foreground!=CLR_INVALID&&font_state.background!=CLR_INVALID&&font_state.align!=GDI_ERROR&&(font_state.mode==OPAQUE||font_state.mode==TRANSPARENT);
}
inline void setup(HDC dc,HGDIOBJ font,State const& s){SelectObject(dc,font);SetTextColor(dc,s.foreground);SetBkColor(dc,s.background);SetBkMode(dc,s.mode);SetTextAlign(dc,TA_LEFT|TA_TOP|TA_NOUPDATECP);}
inline unsigned channel(unsigned word,unsigned c,bool green6){return (word>>(c==0?0:c==1?5:green6?11:10))&((c==1&&green6)?63:31);}
inline unsigned expand(unsigned word,bool green6){unsigned b=channel(word,0,green6),g=channel(word,1,green6),r=channel(word,2,green6);
    return ((b<<3)|(b>>2))|((green6?((g<<2)|(g>>4)):((g<<3)|(g>>2)))<<8)|(((r<<3)|(r>>2))<<16);}
// Preparation is bounded to 16K pixels (<2 MiB scratch), runs only on cache
// misses and uses the actual selected font. Full-color response also drives the
// native compatibility image; packed edge colors may differ from 16-bit GDI.
inline bool compile(HDC source,State const& font_state,char const* text,unsigned count,Raster& out){
    if(!count){out={};return true;}if(!text||count>1024)return false;
    SIZE extent={};TEXTMETRICA metrics={};
    if(!GetTextExtentPoint32A(source,text,int(count),&extent)||!GetTextMetricsA(source,&metrics)||extent.cx<0||metrics.tmHeight<1)return false;
    int margin=metrics.tmHeight+std::abs(metrics.tmOverhang);
    unsigned w=unsigned(extent.cx+2*margin),h=unsigned(metrics.tmHeight+2*margin);
    if(!w||w>2240||!h||h>1260||std::uint64_t(w)*h>16384)return false;
    Dib full(w,h);if(!full.pixels)return false;
    auto font=GetCurrentObject(source,OBJ_FONT);setup(full.dc,font,font_state);
    unsigned size=w*h;
    using Curve=std::array<unsigned char,17>;
    std::vector<Curve> response(std::size_t(size)*3);
    auto draw=[&](HDC dc){return TextOutA(dc,margin,margin,text,int(count))!=FALSE;};
    for(unsigned sample=0;sample<17;++sample){
        unsigned level=std::min(sample*16u,255u);auto wide=static_cast<unsigned*>(full.pixels);
        std::fill(wide,wide+size,level*0x010101u);
        if(!draw(full.dc))return false;GdiFlush();
        for(unsigned n=0;n<size;++n)for(unsigned c=0;c<3;++c)response[n*3+c][sample]=static_cast<unsigned char>((wide[n]>>(c*8))&255);
    }
    std::map<Curve,unsigned> unique;out={};out.width=w;out.height=h;out.left=-margin;out.top=-margin;out.pixels.resize(size);
    for(unsigned n=0;n<size;++n){unsigned pixel=0;
        for(unsigned c=0;c<3;++c){auto const& curve=response[n*3+c];auto found=unique.find(curve);unsigned id;
            // Coverage is independent of the current map color. Preserve unused
            // native word bits only outside the glyph/opaque-background footprint.
            for(unsigned sample=0;sample<17;++sample)if(curve[sample]!=std::min(sample*16u,255u)){pixel|=1u<<30;break;}
            if(found==unique.end()){id=unsigned(unique.size());if(id>=1024)return false;unique.emplace(curve,id);
                for(auto value:curve)out.curves.push_back(value);
            }else id=found->second;pixel|=id<<(c*10);
        }out.pixels[n]=pixel;
    }
    return true;
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
