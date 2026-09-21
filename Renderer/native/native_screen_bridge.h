#pragma once
#include "gpu_frame_api.h"
#include "native_access.h"
#include <vector>
namespace c3x_native_images {
// Snapshot a CPU-owned screen before map admission or after an ownership barrier.
// A private copy must not turn a freshly observed screen into a public pointer
// escape: it may join the GPU map family later. The caller returns any GPU-owned
// image to CPU ownership first. No pointer or lease survives this copy.
struct ScreenSnapshot {
    HWND window=nullptr;int width=0,height=0;RECT area={};
    unsigned native_format=1;
    std::vector<unsigned short> pixels;
    bool capture(void* image,void* graph,void const* requested){
        if(!image||!graph)return false;
        auto field=[&](unsigned offset){return c3x_native_access::field(image,offset);};
        if(field(0x24)!=16)return false;
        width=field(0x38);height=field(0x3c);
        if(width<=0||height<=0||width>2240||height>1260)return false;
        window=c3x_native_access::window(graph);DWORD process=0;
        if(!window||GetWindowThreadProcessId(window,&process)!=GetCurrentThreadId()||process!=GetCurrentProcessId())return false;
        RECT client={};if(!GetClientRect(window,&client)||client.right!=width||client.bottom!=height)return false;
        area=requested?*static_cast<RECT const*>(requested):RECT{0,0,width,height};
        area.left=std::max(0L,area.left);area.top=std::max(0L,area.top);
        area.right=std::min(LONG(width),area.right);area.bottom=std::min(LONG(height),area.bottom);
        if(area.left>=area.right||area.top>=area.bottom)return false;
        DIBSECTION dib={};
        if(!c3x_native_access::dib(image,dib)||!dib.dsBm.bmBits||dib.dsBm.bmBitsPixel!=16||
           dib.dsBm.bmWidth!=width||dib.dsBm.bmHeight!=height||dib.dsBm.bmWidthBytes<width*2)return false;
        bool rgb565=false;
        if(dib.dsBmih.biCompression==BI_BITFIELDS){
            if(dib.dsBitfields[0]==0xf800&&dib.dsBitfields[1]==0x7e0&&dib.dsBitfields[2]==0x1f)rgb565=true;
            else if(dib.dsBitfields[0]!=0x7c00||dib.dsBitfields[1]!=0x3e0||dib.dsBitfields[2]!=0x1f)return false;
        }else if(dib.dsBmih.biCompression!=BI_RGB)return false;
        native_format=rgb565?2:1;
        unsigned pitch=(unsigned(width)+1)&~1u;pixels.resize(std::size_t(pitch)*height);GdiFlush();
        // Audited JGL getter 0x1b70 returns exactly Bits_Data (+0x4c0), while
        // incrementing a lease counter. Read that same logical base directly;
        // invoking the public hook would incorrectly revoke lifetime evidence.
        // GetObject's DIB base alone does not preserve native row orientation.
        auto bits=c3x_native_access::words(image,nullptr);
        if(!bits)return false;
        int stride=field(0x40);
        if(stride<width)return false;
        for(int y=area.top;y<area.bottom;++y)
            std::memcpy(pixels.data()+std::size_t(y)*pitch+area.left,bits+std::size_t(y)*stride+area.left,(area.right-area.left)*2);
        c3x_native_access::release_words(image,nullptr);
        return true;
    }
};
}
