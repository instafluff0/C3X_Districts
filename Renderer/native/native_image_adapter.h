#pragma once
// Native ownership stays on the caller thread; Backend owns ordered image
// commands and explicit CPU barriers. No native pointers cross to the GPU worker.
// Live surface/presentation admission is separate from this tested adapter.
#include "c3x_renderer_api.h"
#include "gpu_image_commands.h"
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <climits>

namespace c3x_native_images {
using namespace c3x_gpu_images;
struct Counts {std::uint64_t translated=0,fallbacks=0,readbacks=0,readback_bytes=0,source_checks=0;};
template<class Backend> class Adapter {
    struct Image {void* native=nullptr;Id gpu=0;unsigned width=0,height=0;Format format=Format::rgb555;
        bool owned=false,dirty=false,cpu_uploaded=false;std::uint64_t revision=0;std::vector<std::uint32_t> cpu;};
    Backend& gpu;
    void* get_bits;void* release_bits;DWORD thread=GetCurrentThreadId();
    std::array<Image,32> images={};std::uint64_t cpu_bytes=0;
    static constexpr std::uint64_t cpu_budget=64u*1024u*1024u;
    Counts counters;
    using Get=std::uint16_t*(__thiscall*)(void*);
    using Release=void(__thiscall*)(void*,int);
    static int field(void* p,unsigned offset){return *reinterpret_cast<int*>(static_cast<char*>(p)+offset);}
    static Rect rect(void const* p){auto r=static_cast<RECT const*>(p);return {r->left,r->top,r->right,r->bottom};}
    Image* find(void* p){for(auto& image:images)if(p&&image.native==p)return &image;return nullptr;}
    void forget(Image& image){gpu.destroy(image.gpu);cpu_bytes-=image.cpu.size()*4;image={};}
    Image* create(void* p,bool owned){
        if(!p||field(p,0x24)!=16)return nullptr;
        int w=field(p,0x38),h=field(p,0x3c),stride=field(p,0x40);
        if(w<=0||h<=0||w>2240||h>1192||stride<w)return nullptr;
        // Derive format from the actual DIB; 16-bit alone does not distinguish 555/565.
        DIBSECTION dib={};auto bitmap=*reinterpret_cast<HBITMAP*>(static_cast<char*>(p)+0x4b4);
        if(GetObject(bitmap,sizeof dib,&dib)!=sizeof dib||dib.dsBm.bmBitsPixel!=16)return nullptr;
        Format format;
        if(dib.dsBmih.biCompression==BI_RGB)format=Format::rgb555;
        else if(dib.dsBmih.biCompression==BI_BITFIELDS&&dib.dsBitfields[0]==0x7c00&&dib.dsBitfields[1]==0x3e0&&dib.dsBitfields[2]==0x1f)format=Format::rgb555;
        else if(dib.dsBmih.biCompression==BI_BITFIELDS&&dib.dsBitfields[0]==0xf800&&dib.dsBitfields[1]==0x7e0&&dib.dsBitfields[2]==0x1f)format=Format::rgb565;
        else return nullptr;
        if(std::uint64_t(w)*h*4>cpu_budget-cpu_bytes)return nullptr;
        for(auto& image:images)if(!image.native){
            std::vector<std::uint32_t> bytes(std::size_t(w)*h);
            auto id=gpu.create(w,h,format);if(!id)return nullptr;
            image.native=p;image.gpu=id;image.width=w;image.height=h;image.format=format;image.owned=owned;
            image.cpu=std::move(bytes);cpu_bytes+=image.cpu.size()*4;return &image;
        }return nullptr;
    }
    std::vector<std::uint32_t> read_cpu(Image const& image){
        // Complete queued GDI writes before accessing the DIB. Use original core
        // methods, so private adapter leases neither recurse nor imply CPU escape.
        std::vector<std::uint32_t> result(std::size_t(image.width)*image.height);
        GdiFlush();auto bits=reinterpret_cast<Get>(get_bits)(image.native);
        if(!bits)throw std::runtime_error("native image lease failed");
        auto stride=field(image.native,0x40);
        for(unsigned y=0;y<image.height;++y)for(unsigned x=0;x<image.width;++x)result[y*image.width+x]=bits[y*stride+x];
        reinterpret_cast<Release>(release_bits)(image.native,1);return result;
    }
    bool refresh(Image& image){
        ++counters.source_checks;auto content=read_cpu(image);
        // Pointer equality and getter/release counts are not content revisions.
        // CPU-owned sources may retain pointers: compare complete words on use.
        if(image.cpu_uploaded&&content==image.cpu)return true;
        if(!gpu.upload(image.gpu,image.revision+1,content.data(),content.size()))return false;
        ++image.revision;image.cpu_uploaded=true;image.cpu=std::move(content);return true;
    }
    void cpu_ownership(Image& image){
        if(image.dirty){
            if(!gpu.readback(image.gpu,image.cpu.data(),image.cpu.size()))
                throw std::runtime_error("cannot read current GPU image");
            GdiFlush();auto bits=reinterpret_cast<Get>(get_bits)(image.native);
            if(!bits)throw std::runtime_error("cannot restore native image ownership");
            auto stride=field(image.native,0x40);
            for(unsigned y=0;y<image.height;++y)for(unsigned x=0;x<image.width;++x)bits[y*stride+x]=std::uint16_t(image.cpu[y*image.width+x]);
            reinterpret_cast<Release>(release_bits)(image.native,1);
            ++counters.readbacks;counters.readback_bytes+=image.cpu.size()*4;image.dirty=false;
            // Reestablish CPU-upload revision validity only on its next source use.
            image.cpu_uploaded=false;
        }
        // Never reacquire GPU destination ownership within this native lifetime:
        // a previously returned pointer or DC may be retained after release.
        image.owned=false;
    }
public:
    Adapter(Backend& g,void* bits,void* release):gpu(g),get_bits(bits),release_bits(release){}
    ~Adapter(){for(auto& image:images)if(image.native)forget(image);}
    Adapter(Adapter const&)=delete;Adapter& operator=(Adapter const&)=delete;
    // Call drain while native objects/device still exist. A synchronization/device
    // failure is terminal for this isolated backend, never a stale-pixel fallback.
    void drain(){for(auto& image:images)if(image.native){cpu_ownership(image);forget(image);}}
    int operation(int op,void* object,void* source,void const* source_rect,void const* target_rect,unsigned color){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native GPU adapter thread changed");
        auto destination=find(object);
        if(op==C3X_NATIVE_IMAGE_DRAIN){drain();return 0;}
        if(op==C3X_NATIVE_DESTROY){if(destination)forget(*destination);return 0;}
        if(op==C3X_NATIVE_IMAGE_REINIT){if(destination){cpu_ownership(*destination);forget(*destination);}return 0;}
        if(op==C3X_NATIVE_INIT){
            // Admission is only for a successful fresh lifetime observed after
            // attachment. Preexisting images always enter as CPU-owned sources.
            if(destination)forget(*destination);
            destination=create(object,true);if(destination&&!refresh(*destination))forget(*destination);return 0;
        }
        if(op==C3X_NATIVE_PIXEL||op==C3X_NATIVE_BITS||op==C3X_NATIVE_DC){if(destination)cpu_ownership(*destination);return 0;}
        if(op!=C3X_NATIVE_COPY&&op!=C3X_NATIVE_FILL&&op!=C3X_NATIVE_IMAGE_DRAW)return 0;
        auto fallback=[&](){if(destination)cpu_ownership(*destination);auto s=find(source);if(s&&s!=destination)cpu_ownership(*s);++counters.fallbacks;return 0;};
        if(!destination||!destination->owned)return fallback();
        Command command={Kind::fill,destination->gpu,0,{},rect(static_cast<char*>(object)+0x44),0,0,color&0xffff};
        command.area=target_rect?rect(target_rect):Rect{0,0,int(destination->width),int(destination->height)};
        if(op==C3X_NATIVE_FILL){
            // Null rectangle uses a different native whole-image clear helper.
            if(!target_rect||(color&0xffff0000)!=0x80000000)return fallback();
        }else{
            if(!target_rect)return fallback();
            auto s=find(source);if(!s)s=create(source,false);
            if(!s||s->format!=destination->format)return fallback();
            if(op==C3X_NATIVE_IMAGE_DRAW){
                // Audited ordinary 16-bit transparent image mode. Palette/shadow
                // variants retain native execution until their semantics pass.
                if(unsigned(field(source,0x4d0))!=0x80007c1fu)return fallback();
                auto right=std::int64_t(command.area.left)+s->width,bottom=std::int64_t(command.area.top)+s->height;
                if(right>INT_MAX||bottom>INT_MAX)return fallback();
                command.area.right=int(right);command.area.bottom=int(bottom);
                if(std::max(command.area.left,command.clip.left)>=std::min(command.area.right,command.clip.right)||
                   std::max(command.area.top,command.clip.top)>=std::min(command.area.bottom,command.clip.bottom))return fallback();
                command.kind=Kind::color_key;command.color=0x7c1f;
            }else{
                if(!source_rect)return fallback();auto r=rect(source_rect);
                if(r.left>=r.right||r.top>=r.bottom)return fallback();
                if(std::int64_t(r.right)-r.left!=std::int64_t(command.area.right)-command.area.left||
                   std::int64_t(r.bottom)-r.top!=std::int64_t(command.area.bottom)-command.area.top)return fallback();
                command.kind=Kind::copy;command.source_x=r.left;command.source_y=r.top;
            }
            if(!s->owned&&!refresh(*s))return fallback();
            command.source=s->gpu;
        }
        if(!gpu.submit(&command,1))return fallback();
        destination->dirty=true;++counters.translated;return 1;
    }
    // The map is an immutable resident source. Quantize only at the native
    // destination, using the same world-anchored rounding as the CPU blitter.
    bool insert_map(void* p,Id map,Rect area,int source_x,int source_y,int phase_x,int phase_y){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native map adapter thread changed");
        auto d=find(p);if(!d||!d->owned)return false;
        Command c={Kind::quantize,d->gpu,map,area,rect(static_cast<char*>(p)+0x44),source_x,source_y,
            (unsigned(phase_x)&7u)|((unsigned(phase_y)&7u)<<3)};
        if(!gpu.submit(&c,1))return false;d->dirty=true;return true;
    }
    Id image(void* p){auto i=find(p);return i?i->gpu:0;}
    bool owns(void* p){auto i=find(p);return i&&i->owned;}
    Counts stats()const{return counters;}
};
} // namespace c3x_native_images
