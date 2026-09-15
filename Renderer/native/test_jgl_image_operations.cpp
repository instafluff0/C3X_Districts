// Standalone execution of verified JGL image methods; never loads Civ III.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <cstdint>

struct Image { void** table; };
struct Observation {
    Image* image=nullptr;void** original=nullptr;std::array<void*,60> table={};unsigned dc_calls=0,pixel_calls=0;
};
std::array<Observation,2> observations;
Observation& owner(Image* image){for(auto& o:observations)if(o.image==image)return o;throw std::runtime_error("unowned image");}
HDC __fastcall acquire(Image* image,void*){
    auto& o=owner(image);++o.dc_calls;return reinterpret_cast<HDC(__thiscall*)(Image*)>(o.original[10])(image);
}
void* __fastcall pixel(Image* image,void*,int x,int y){
    auto& o=owner(image);++o.pixel_calls;return reinterpret_cast<void*(__thiscall*)(Image*,int,int)>(o.original[7])(image,x,y);
}
void verify(bool ok,char const* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    HMODULE module=nullptr;void* graph=nullptr;
    try{
        module=LoadLibraryA(argv[1]);verify(module!=nullptr,"LoadLibrary JGL");
        // The runner pins the installed binary by hash before this ABI witness.
        FARPROC entry=GetProcAddress(module,"get_graphsy_object_ptr");verify(entry!=nullptr,"JGL export");
        using Factory=void*(__cdecl*)();Factory factory=nullptr;static_assert(sizeof(factory)==sizeof(entry));std::memcpy(&factory,&entry,sizeof(entry));
        graph=factory();verify(graph!=nullptr,"JGL graph factory");auto gt=*reinterpret_cast<void***>(graph);
        using Create=Image*(__thiscall*)(void*,void*,int);using Init=int(__thiscall*)(Image*,int,int,int,int);
        using Fill=int(__thiscall*)(Image*,RECT*,int);using Clip=int(__thiscall*)(Image*,RECT*);
        using Copy=int(__thiscall*)(Image*,Image*,RECT*,RECT*);
        for(auto& o:observations){o.image=reinterpret_cast<Create>(gt[31])(graph,nullptr,1);verify(o.image!=nullptr,"create image");
            o.original=o.image->table;verify(reinterpret_cast<Init>(o.original[1])(o.image,64,48,16,1)==0,"initialize 16-bit image");
            std::memcpy(o.table.data(),o.original,sizeof(o.table));o.table[10]=reinterpret_cast<void*>(&acquire);o.table[7]=reinterpret_cast<void*>(&pixel);o.image->table=o.table.data();}
        auto& a=observations[0];auto& b=observations[1];RECT full={0,0,64,48},clip={7,5,40,33},area={0,0,56,40};
        auto fill=reinterpret_cast<Fill>(a.original[17]);verify(fill(a.image,&full,int(0x80000000u))==0,"clear image");
        verify(reinterpret_cast<Clip>(a.original[13])(a.image,&clip)==0,"set clip");a.dc_calls=a.pixel_calls=0;
        verify(fill(a.image,&area,int(0x800003e0u))==0,"fill image");
        unsigned fill_dc=a.dc_calls,fill_pixels=a.pixel_calls;
        verify(fill_dc==0&&fill_pixels>0,"native fill must demonstrate direct pixels without HDC");
        // Original pixel accessor holds a native pixel lease; release once per access.
        using Get=std::uint16_t*(__thiscall*)(Image*,int,int);using Release=void(__thiscall*)(Image*,int);
        auto get=reinterpret_cast<Get>(a.original[7]);auto release=reinterpret_cast<Release>(a.original[9]);
        for(int y=0;y<48;++y)for(int x=0;x<64;++x){auto p=get(a.image,x,y);verify(p!=nullptr,"pixel lease");unsigned value=*p;release(a.image,1);
            verify(value==unsigned(x>=7&&x<40&&y>=5&&y<33?0x03e0:0),"fill clipping pixels");}
        verify(reinterpret_cast<Clip>(a.original[13])(a.image,&full)==0,"reset clip");a.dc_calls=b.dc_calls=0;
        verify(reinterpret_cast<Copy>(a.original[16])(a.image,b.image,&full,&full)==0,"native image copy");GdiFlush();
        verify(a.dc_calls==1&&b.dc_calls==1,"copy must acquire both image DCs");
        auto get_b=reinterpret_cast<Get>(b.original[7]);auto release_b=reinterpret_cast<Release>(b.original[9]);
        for(int y=0;y<48;++y)for(int x=0;x<64;++x){auto p=get_b(b.image,x,y);verify(p!=nullptr,"copied pixel lease");unsigned value=*p;release_b(b.image,1);
            verify(value==unsigned(x>=7&&x<40&&y>=5&&y<33?0x03e0:0),"copy pixels");}
        std::printf("PASS actual JGL 16-bit clipped fill: dc_calls=%u pixel_calls=%u; copy: source_dc=%u destination_dc=%u; exact 3072 pixels\n",fill_dc,fill_pixels,a.dc_calls,b.dc_calls);
        auto source_words=get(a.image,0,0);auto source_stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(a.image)+0x40);
        for(int y=0;y<48;++y)for(int x=0;x<64;++x)source_words[y*source_stride+x]=std::uint16_t((y*64+x)*193u+79u)&0x7fff;
        release(a.image,1);
        auto source_dc=reinterpret_cast<HDC(__thiscall*)(Image*)>(a.original[10])(a.image);
        int stretch_mode=GetStretchBltMode(source_dc);reinterpret_cast<Release>(a.original[11])(a.image,1);
        for(auto extent:std::array<std::array<int,2>,8>{{{{32,24}},{{43,31}},{{61,43}},{{91,73}},{{7,5}},{{13,17}},{{97,19}},{{23,79}}}}){
            RECT scaled={-3,-2,extent[0]-3,extent[1]-2};
            verify(reinterpret_cast<Copy>(a.original[16])(a.image,b.image,&full,&scaled)==0,"native scaled image");GdiFlush();
            auto words=get_b(b.image,0,0);auto stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(b.image)+0x40);unsigned mismatches=0;
            auto interval=[](int at,int src,int dst){int end=((2*at+1)*src)/(2*dst)+1;
                return std::array<int,2>{{src<=dst?end-1:at==0?0:((2*at-1)*src)/(2*dst)+1,end}};};
            for(int y=0;y<48&&y<scaled.bottom;++y)for(int x=0;x<64&&x<scaled.right;++x){
                auto xr=interval(x+3,64,extent[0]),yr=interval(y+2,48,extent[1]);unsigned expected=0x7fff;
                for(int yy=yr[0];yy<yr[1];++yy)for(int xx=xr[0];xx<xr[1];++xx)expected&=((yy*64+xx)*193u+79u)&0x7fff;
                mismatches+=words[y*stride+x]!=expected;
            }
            release_b(b.image,1);verify(stretch_mode==BLACKONWHITE&&mismatches==0,"actual native center-end shrink mapping");
            std::printf("PASS JGL stretch width=%d height=%d exact clipped native pixels\n",extent[0],extent[1]);
        }
        for(auto& o:observations){o.image->table=o.original;reinterpret_cast<void(__thiscall*)(Image*,unsigned)>(o.original[0])(o.image,1);o.image=nullptr;}
        reinterpret_cast<void(__thiscall*)(void*,unsigned)>(gt[0])(graph,1);graph=nullptr;FreeLibrary(module);module=nullptr;return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
