#pragma once
#include "gpu_frame_api.h"
#include <vector>
namespace c3x_native_images {
// Compatibility admission for images created before our hooks were installed.
// Civ III remains their pixel owner. Snapshot the completed native screen at its
// transfer boundary; never infer damage from pixel equality or retain a lease.
struct ScreenSnapshot {
    HWND window=nullptr;int width=0,height=0;RECT area={};
    unsigned native_format=1;
    std::vector<unsigned short> pixels;
    bool capture(void* image,void* graph,void const* requested){
        if(!image||!graph)return false;
        auto field=[&](unsigned offset){return *reinterpret_cast<int*>(static_cast<char*>(image)+offset);};
        if(field(0x24)!=16)return false;
        width=field(0x38);height=field(0x3c);
        if(width<=0||height<=0||width>2240||height>1192)return false;
        auto dc=*reinterpret_cast<HDC*>(static_cast<char*>(graph)+0x138);
        window=WindowFromDC(dc);DWORD process=0;
        if(!window||GetWindowThreadProcessId(window,&process)!=GetCurrentThreadId()||process!=GetCurrentProcessId())return false;
        RECT client={};if(!GetClientRect(window,&client)||client.right!=width||client.bottom!=height)return false;
        area=requested?*static_cast<RECT const*>(requested):RECT{0,0,width,height};
        area.left=std::max(0L,area.left);area.top=std::max(0L,area.top);
        area.right=std::min(LONG(width),area.right);area.bottom=std::min(LONG(height),area.bottom);
        if(area.left>=area.right||area.top>=area.bottom)return false;
        DIBSECTION dib={};auto bitmap=*reinterpret_cast<HBITMAP*>(static_cast<char*>(image)+0x4b4);
        if(GetObject(bitmap,sizeof(dib),&dib)!=sizeof(dib)||!dib.dsBm.bmBits||dib.dsBm.bmBitsPixel!=16||
           dib.dsBm.bmWidth!=width||dib.dsBm.bmHeight!=height||dib.dsBm.bmWidthBytes<width*2)return false;
        bool rgb565=false;
        if(dib.dsBmih.biCompression==BI_BITFIELDS){
            if(dib.dsBitfields[0]==0xf800&&dib.dsBitfields[1]==0x7e0&&dib.dsBitfields[2]==0x1f)rgb565=true;
            else if(dib.dsBitfields[0]!=0x7c00||dib.dsBitfields[1]!=0x3e0||dib.dsBitfields[2]!=0x1f)return false;
        }else if(dib.dsBmih.biCompression!=BI_RGB)return false;
        native_format=rgb565?2:1;
        unsigned pitch=(unsigned(width)+1)&~1u;pixels.resize(std::size_t(pitch)*height);GdiFlush();
        // GetObject does not preserve JGL's logical row orientation. Borrow its
        // native bits/stride instead, exactly as the existing image adapter does.
        auto table=*static_cast<void***>(image);
        auto bits=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(table[4])(image);
        if(!bits)return false;
        int stride=field(0x40);
        if(stride<width){reinterpret_cast<void(__thiscall*)(void*,int)>(table[9])(image,1);return false;}
        for(int y=area.top;y<area.bottom;++y)
            std::memcpy(pixels.data()+std::size_t(y)*pitch+area.left,bits+std::size_t(y)*stride+area.left,(area.right-area.left)*2);
        reinterpret_cast<void(__thiscall*)(void*,int)>(table[9])(image,1);
        return true;
    }
};
}
