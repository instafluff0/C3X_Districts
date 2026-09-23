#pragma once
#include "runtime.h"
#include <array>
namespace c3x_inputs {
// CPU compatibility units consume a native GDI canvas. Only DIB values and the
// effective pixel clip cross the boundary; HDC/HBITMAP values never do. Native
// changes are input patches. Renderer writes are independently hashed outputs.
struct Canvas {
    unsigned id=0;HDC dc=nullptr;HBITMAP bitmap=nullptr;unsigned char* bits=nullptr;
    int width=0,height=0,stride=0,depth=0,top_down=0,format=0;
    POINT viewport={},origin={};RECT clip={};
    std::vector<std::array<std::uint32_t,4>> hashes;
    bool owns=false;std::uint64_t last_seen=0;unsigned char* native_bits=nullptr;std::size_t native_bytes=0;
    Canvas()=default;Canvas(Canvas const&)=delete;Canvas& operator=(Canvas const&)=delete;
    ~Canvas(){if(owns){DeleteDC(dc);DeleteObject(bitmap);}}
    unsigned word(unsigned row,unsigned column)const{auto pixel=bits+std::size_t(row)*unsigned(stride)+std::size_t(column)*unsigned(depth/8);
        if(depth==16){unsigned short value;std::memcpy(&value,pixel,2);return value;}unsigned value;std::memcpy(&value,pixel,4);return value;}
    void set_word(unsigned row,unsigned column,unsigned value){auto pixel=bits+std::size_t(row)*unsigned(stride)+std::size_t(column)*unsigned(depth/8);std::memcpy(pixel,&value,std::size_t(depth/8));}
    std::array<std::uint32_t,4> block(unsigned first,unsigned rows)const{
        // DIB row padding is neither consumed pixel input nor a stable witness.
        Writer packed;packed.reserve(std::size_t(rows)*unsigned(width)*unsigned(depth/8));
        for(unsigned y=first;y<first+rows;++y){auto row=bits+std::size_t(y)*unsigned(stride);packed.bytes.insert(packed.bytes.end(),row,row+std::size_t(width)*unsigned(depth/8));}
        return c3x_renderer::asset_content_hash(packed.bytes.data(),packed.bytes.size());
    }
    unsigned rows_per_block()const{return std::max(1u,65536u/(unsigned(width)*4));}
    void witness(Writer& out){out.u32(id);for(unsigned y=0;y<unsigned(height);y+=rows_per_block()){
        auto hash=block(y,std::min(rows_per_block(),unsigned(height)-y));for(auto part:hash)out(part);}}
    void remember(){hashes.clear();for(unsigned y=0;y<unsigned(height);y+=rows_per_block())hashes.push_back(block(y,std::min(rows_per_block(),unsigned(height)-y)));}
};
template<class IO>void canvas_fields(IO& out,Canvas& value){
    out(value.id);out(value.width);out(value.height);out(value.depth);out(value.top_down);out(value.format);
    // Windows LONG differs from int32_t on some hosts; the wire always uses 32 bits.
    std::int32_t fields[]={value.viewport.x,value.viewport.y,value.origin.x,value.origin.y,value.clip.left,value.clip.top,value.clip.right,value.clip.bottom};
    for(auto& field:fields)out(field);value.viewport={fields[0],fields[1]};value.origin={fields[2],fields[3]};value.clip={fields[4],fields[5],fields[6],fields[7]};
}
class CanvasCapture {
    std::map<HDC,std::unique_ptr<Canvas>> canvases;unsigned next=0;std::uint64_t uses=0;
    std::unique_ptr<Canvas> scratch;
public:
    std::mutex mutex;
    void copy(Canvas& value){
        if(!scratch||scratch->width!=value.width||scratch->height!=value.height||scratch->depth!=value.depth||scratch->format!=value.format){
            scratch.reset();auto next_surface=std::make_unique<Canvas>();next_surface->owns=true;next_surface->width=value.width;next_surface->height=value.height;next_surface->depth=value.depth;next_surface->format=value.format;
            struct Info{BITMAPINFOHEADER head;DWORD masks[3];} info={};info.head.biSize=sizeof(info.head);info.head.biWidth=value.width;info.head.biHeight=-value.height;
            info.head.biPlanes=1;info.head.biBitCount=WORD(value.depth);info.head.biCompression=value.depth==16?BI_BITFIELDS:BI_RGB;
            info.masks[0]=value.format==C3X_GPU_RGB565?0xf800:0x7c00;info.masks[1]=value.format==C3X_GPU_RGB565?0x7e0:0x3e0;info.masks[2]=0x1f;
            // A NULL compatibility DC dies with its creating thread. Use the
            // actual device so serial callers can safely retain one scratch DIB.
            next_surface->dc=CreateCompatibleDC(value.dc);void* data=nullptr;next_surface->bitmap=CreateDIBSection(next_surface->dc,reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS,&data,nullptr,0);
            require(next_surface->dc&&next_surface->bitmap&&data,"CPU input copy allocation failed");SetICMMode(next_surface->dc,ICM_OFF);SelectObject(next_surface->dc,next_surface->bitmap);next_surface->bits=static_cast<unsigned char*>(data);next_surface->stride=((value.width*value.depth+31)/32)*4;scratch=std::move(next_surface);
        }
        // GetObject normalizes DIB height and cannot prove row orientation.
        // The native GDI copy supplies canonical top-down pixels without writing
        // the game canvas. Only this one bounded scratch bitmap is retained.
        require(BitBlt(scratch->dc,0,0,value.width,value.height,value.dc,value.origin.x-value.viewport.x,value.origin.y-value.viewport.y,SRCCOPY)!=0&&GdiFlush(),"CPU input copy failed");
        value.bits=scratch->bits;value.stride=scratch->stride;value.top_down=1;
    }
    void remember(Canvas& value){copy(value);value.remember();}
    void witness(Writer& out,Canvas& value){copy(value);value.witness(out);}

    Canvas& describe(HDC dc){
        require(dc!=nullptr,"missing CPU canvas");auto bitmap=static_cast<HBITMAP>(GetCurrentObject(dc,OBJ_BITMAP));DIBSECTION dib={};
        require(bitmap&&GetObject(bitmap,sizeof(dib),&dib)==sizeof(dib)&&dib.dsBm.bmBits,"CPU canvas is not a DIB");
        // Two views of a mapping may alias despite different virtual addresses
        // and bitmap handles. Until storage identity is in the protocol, reject
        // section-backed DIBs rather than replaying them as independent images.
        require(!dib.dshSection,"mapped CPU canvas needs a shared-storage replay contract");
        require(GetMapMode(dc)==MM_TEXT&&GetGraphicsMode(dc)==GM_COMPATIBLE&&GetLayout(dc)==0,"unsupported CPU canvas transform");
        require(dib.dsBm.bmWidth>0&&dib.dsBm.bmWidth<=2240&&dib.dsBm.bmHeight>0&&dib.dsBm.bmHeight<=1260&&
            (dib.dsBm.bmBitsPixel==16||dib.dsBm.bmBitsPixel==32),"CPU canvas extent/format unsupported");
        int format=C3X_GPU_BGRA32;if(dib.dsBm.bmBitsPixel==16){format=C3X_GPU_RGB555;
            if(dib.dsBmih.biCompression==BI_BITFIELDS){
                if(dib.dsBitfields[0]==0xf800&&dib.dsBitfields[1]==0x7e0&&dib.dsBitfields[2]==0x1f)format=C3X_GPU_RGB565;
                else require(dib.dsBitfields[0]==0x7c00&&dib.dsBitfields[1]==0x3e0&&dib.dsBitfields[2]==0x1f,"unsupported CPU canvas masks");
            }else require(dib.dsBmih.biCompression==BI_RGB,"unsupported CPU canvas compression");}
        else require(dib.dsBmih.biCompression==BI_RGB,"unsupported CPU BGRA canvas compression");
        auto found=canvases.find(dc);
        bool changed=found==canvases.end()||found->second->bitmap!=bitmap||found->second->width!=dib.dsBm.bmWidth||found->second->height!=dib.dsBm.bmHeight||found->second->depth!=dib.dsBm.bmBitsPixel||found->second->format!=format;
        if(changed){
            if(found!=canvases.end()){retire(found->second->id);canvases.erase(found);}
            if(canvases.size()>=8){auto oldest=std::min_element(canvases.begin(),canvases.end(),[](auto const& a,auto const& b){return a.second->last_seen<b.second->last_seen;});retire(oldest->second->id);canvases.erase(oldest);}
            require(next<UINT32_MAX,"CPU canvas identity exhausted");auto value=std::make_unique<Canvas>();value->id=++next;value->dc=dc;value->bitmap=bitmap;
            found=canvases.emplace(dc,std::move(value)).first;
        }
        auto& value=*found->second;value.last_seen=++uses;value.bits=static_cast<unsigned char*>(dib.dsBm.bmBits);value.width=dib.dsBm.bmWidth;value.height=dib.dsBm.bmHeight;
        value.depth=dib.dsBm.bmBitsPixel;value.stride=dib.dsBm.bmWidthBytes;value.top_down=1;value.format=format;
        require(value.stride>=value.width*value.depth/8,"invalid CPU canvas stride");
        value.native_bits=value.bits;value.native_bytes=std::size_t(value.stride)*unsigned(value.height);
        require(GetViewportOrgEx(dc,&value.viewport)&&GetWindowOrgEx(dc,&value.origin),"CPU canvas origin unavailable");
        int clip=GetClipBox(dc,&value.clip);require(clip==SIMPLEREGION||clip==NULLREGION,"unsupported complex CPU canvas clip");
        if(clip==NULLREGION)value.clip={0,0,0,0};return value;
    }
    void retire(unsigned id){Call input(Kind::native_snapshot,4,[&](Writer& out){out(id);});input.result(1);}
    void capture(Canvas& value){
        copy(value);
        {Call metadata(Kind::native_snapshot,2,[&](Writer& out){canvas_fields(out,value);});metadata.result(1);}
        unsigned index=0;for(unsigned y=0;y<unsigned(value.height);y+=value.rows_per_block(),++index){
            auto rows=std::min(value.rows_per_block(),unsigned(value.height)-y);auto hash=value.block(y,rows);
            if(index<value.hashes.size()&&hash==value.hashes[index])continue;
            Call pixels(Kind::native_snapshot,3,[&](Writer& out){out(value.id);out(y);out(rows);out.reserve(std::size_t(rows)*unsigned(value.width)*4);
                for(unsigned row=y;row<y+rows;++row)for(unsigned x=0;x<unsigned(value.width);++x)out(value.word(row,x));});pixels.result(1);
        }
        value.remember();
    }
    void reset(){canvases.clear();scratch.reset();}
};
// One process-wide identity domain and scratch budget. Thread-local registries
// reused IDs starting at one and silently multiplied the stated memory bound.
// Concurrent producers stop capture instead of blocking the game's draw calls.
inline CanvasCapture& canvas_capture(){static CanvasCapture value;return value;}
inline void reset_canvas_capture(){auto& capture=canvas_capture();std::unique_lock<std::mutex> lock(capture.mutex,std::try_to_lock);
    if(lock.owns_lock())capture.reset();else runtime().stop(Stop::unsupported);}
inline bool aliases(Canvas const& a,Canvas const& b){
    auto first=reinterpret_cast<std::uintptr_t>(a.native_bits),second=reinterpret_cast<std::uintptr_t>(b.native_bits);
    return first<=second?second-first<a.native_bytes:first-second<b.native_bytes;
}
template<class Draw>int cpu_unit(unsigned variant,c3x_renderer_unit_v1 const& unit,HDC destination,HDC background,int* bounds,unsigned flags,Draw draw){
    if(!runtime().active())return draw();
    auto& capture=canvas_capture();std::unique_lock<std::mutex> lock(capture.mutex,std::try_to_lock);
    if(!lock.owns_lock()){runtime().stop(Stop::unsupported);return draw();}
    struct Cleanup {CanvasCapture& capture;~Cleanup(){if(!runtime().active())capture.reset();}} cleanup{capture};
    Canvas* dest=nullptr;Canvas* back=nullptr;
    try{GdiFlush();dest=&capture.describe(destination);if(background)back=&capture.describe(background);
        require(!back||back==dest||!aliases(*dest,*back),"aliased CPU unit canvases need a shared-storage replay contract");}
    catch(...){runtime().stop(Stop::unsupported);return draw();}
    Call input(Kind::unit,2,[&](Writer& out){out(variant);c3x_inputs::unit(out,unit);out(flags);out(dest->id);out.u32(back?back->id:0);});
    try{capture.capture(*dest);if(back&&back!=dest)capture.capture(*back);}
    catch(...){runtime().stop(Stop::unsupported);}
    int result=draw();GdiFlush();
    input.result(result,[&](Writer& out){out.u32(bounds?1:0);if(bounds)for(unsigned n=0;n<4;++n)out(std::int32_t(result==1?bounds[n]:0));
        capture.witness(out,*dest);if(back&&back!=dest)capture.witness(out,*back);});
    try{capture.remember(*dest);if(back&&back!=dest)capture.remember(*back);}catch(...){runtime().stop(Stop::allocation_failure);}return result;
}
class CanvasReplay {
    std::map<unsigned,std::unique_ptr<Canvas>> values;
public:
    Canvas& get(unsigned id){auto found=values.find(id);require(found!=values.end(),"missing replay CPU canvas");return *found->second;}
    void event(unsigned kind,Reader& in){
        if(kind==4){require(values.erase(in.u32())==1,"unknown retired CPU canvas");return;}
        if(kind==2){Canvas layout;canvas_fields(in,layout);require(layout.id&&layout.width>0&&layout.width<=2240&&layout.height>0&&layout.height<=1260&&
            (layout.depth==16||layout.depth==32)&&(layout.top_down==0||layout.top_down==1)&&layout.format>=0&&layout.format<=2,"invalid replay CPU canvas");
            auto found=values.find(layout.id);if(found==values.end()){
                require(values.size()<8,"replay CPU canvas budget");auto value=std::make_unique<Canvas>();value->id=layout.id;value->width=layout.width;value->height=layout.height;value->depth=layout.depth;value->format=layout.format;value->top_down=layout.top_down;value->owns=true;
                struct Info{BITMAPINFOHEADER header;DWORD masks[3];} info={};info.header.biSize=sizeof(info.header);info.header.biWidth=value->width;info.header.biHeight=value->top_down?-value->height:value->height;
                info.header.biPlanes=1;info.header.biBitCount=WORD(value->depth);info.header.biCompression=value->depth==16?BI_BITFIELDS:BI_RGB;
                info.masks[0]=value->format==C3X_GPU_RGB565?0xf800:0x7c00;info.masks[1]=value->format==C3X_GPU_RGB565?0x7e0:0x3e0;info.masks[2]=0x1f;
                value->dc=CreateCompatibleDC(nullptr);void* pixels=nullptr;value->bitmap=CreateDIBSection(value->dc,reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS,&pixels,nullptr,0);
                require(value->dc&&value->bitmap&&pixels,"replay CPU DIB allocation failed");SelectObject(value->dc,value->bitmap);value->bits=static_cast<unsigned char*>(pixels);
                value->stride=((value->width*value->depth+31)/32)*4;std::memset(value->bits,0,std::size_t(value->stride)*unsigned(value->height));found=values.emplace(value->id,std::move(value)).first;
            }
            auto& value=*found->second;require(value.width==layout.width&&value.height==layout.height&&value.depth==layout.depth&&value.format==layout.format&&value.top_down==layout.top_down,"CPU canvas identity changed without retirement");
            value.viewport=layout.viewport;value.origin=layout.origin;value.clip=layout.clip;
            SetViewportOrgEx(value.dc,value.viewport.x,value.viewport.y,nullptr);SetWindowOrgEx(value.dc,value.origin.x,value.origin.y,nullptr);SelectClipRgn(value.dc,nullptr);
            IntersectClipRect(value.dc,value.clip.left,value.clip.top,value.clip.right,value.clip.bottom);return;
        }
        require(kind==3,"unknown CPU canvas input");auto& value=get(in.u32());auto row=in.u32(),rows=in.u32();require(rows&&row<unsigned(value.height)&&rows<=unsigned(value.height)-row,"CPU canvas patch bounds");
        for(unsigned y=row;y<row+rows;++y)for(unsigned x=0;x<unsigned(value.width);++x){auto pixel=in.u32();require(value.depth==32||pixel<=65535,"invalid CPU canvas word");value.set_word(y,x,pixel);}
    }
    bool check(Reader& expected,Canvas& value){Writer actual;value.witness(actual);expected.available(actual.bytes.size());
        bool exact=std::equal(actual.bytes.begin(),actual.bytes.end(),expected.bytes.begin()+expected.at);
        char audit[4]={};bool diagnostic=GetEnvironmentVariableA("C3X_CPU_UNIT_AUDIT",audit,sizeof(audit))==1&&audit[0]=='1';
        if(!realtime_replay().enabled&&!diagnostic)require(exact,"replay CPU unit pixels differ");
        expected.at+=actual.bytes.size();return exact;}
    void reset(){values.clear();}
};
}
