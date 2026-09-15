#pragma once
// Native ownership stays on the caller thread; Backend owns ordered image
// commands and explicit CPU barriers. No native pointers cross to the GPU worker.
// Live surface/presentation admission is separate from this tested adapter.
#include "gpu_frame_api.h"
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
    struct Image {void* native=nullptr;Id gpu=0,detail=0;unsigned width=0,height=0;Format format=Format::rgb555;
        bool owned=false,dirty=false,cpu_uploaded=false;std::uint64_t revision=0;std::vector<std::uint32_t> cpu;};
    Backend& gpu;
    void* get_bits;void* release_bits;DWORD thread=GetCurrentThreadId();
    std::array<Image,32> images={};std::uint64_t cpu_bytes=0;
    static constexpr std::uint64_t cpu_budget=64u*1024u*1024u;
    Counts counters;
    Id sprite_image=0;unsigned sprite_width=0,sprite_height=0;
    std::uint64_t sprite_revision=0;std::vector<std::uint32_t> sprite_pixels;
    using Get=std::uint16_t*(__thiscall*)(void*);
    using Release=void(__thiscall*)(void*,int);
    static int field(void* p,unsigned offset){return *reinterpret_cast<int*>(static_cast<char*>(p)+offset);}
    static Rect rect(void const* p){auto r=static_cast<RECT const*>(p);return {r->left,r->top,r->right,r->bottom};}
    Image* find(void* p){for(auto& image:images)if(p&&image.native==p)return &image;return nullptr;}
    void forget(Image& image){if(image.detail)gpu.destroy(image.detail);gpu.destroy(image.gpu);cpu_bytes-=image.cpu.size()*4;image={};}
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
    // A native-format GPU mirror preserves exact CPU fallback words; only the
    // map/screen copy family also retains full-color pixels. Both change in one
    // validated submission, including native save/restore and partial copies.
    bool full_color(Image& image){
        if(image.detail)return true;
        auto id=gpu.create(image.width,image.height,Format::bgra32);if(!id)return false;
        Command expand={Kind::expand,id,image.gpu,{0,0,int(image.width),int(image.height)},
            {0,0,int(image.width),int(image.height)},0,0,65536};
        if(!gpu.submit(&expand,1)){gpu.destroy(id);return false;}image.detail=id;return true;
    }
    static unsigned expanded(unsigned word,Format format){
        unsigned b=word&31,g=(word>>5)&(format==Format::rgb565?63:31),r=(word>>(format==Format::rgb565?11:10))&31;
        return ((b<<3)|(b>>2))|((format==Format::rgb565?((g<<2)|(g>>4)):((g<<3)|(g>>2)))<<8)|(((r<<3)|(r>>2))<<16)|0xff000000u;
    }
    bool draw_sprite(Image& destination,void* source,void const* palette,void const* target){
        if(!source||!target)return false;
        auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
        if(!module||*static_cast<void***>(source)!=reinterpret_cast<void**>(module+0x68440))return false;
        // Audited slot 17 ordinary and row-trimmed 8-bit sources, plus ordinary
        // 16-bit sources. Other scaling retains its native program.
        int bits=field(source,0x20),w=field(source,0x30),h=field(source,0x34),stride=field(source,0x2c);
        bool trimmed=(field(source,0x18)&1)!=0;
        if((bits!=8&&bits!=16)||(bits==16&&trimmed)||w<1||h<1||w>1024||h>1024||(!trimmed&&stride<w))return false;
        auto rows=trimmed?*reinterpret_cast<unsigned char**>(static_cast<char*>(source)+0x1c):nullptr;
        if(trimmed&&!rows)return false;
        if(trimmed)for(int y=0;y<h;++y)if(unsigned(rows[y*4])+rows[y*4+1]>unsigned(w))return false;
        int denominator=*reinterpret_cast<int*>(module+0x6c104);
        if(denominator<=0||*reinterpret_cast<int*>(module+0x6c0fc)!=denominator||
           *reinterpret_cast<int*>(module+0x6c100)!=denominator)return false;
        unsigned key=unsigned(field(source,0x28));
        if(bits==16&&!(key&0x80000000u))return false; // native palette-key mutation needs its own contract
        auto area=rect(target);
        if(std::int64_t(area.left)+w>INT_MAX||std::int64_t(area.top)+h>INT_MAX)return false;
        area.right=area.left+w;area.bottom=area.top+h;
        auto clip=rect(static_cast<char*>(destination.native)+0x44);
        if(std::max({0,area.left,clip.left})>=std::min({int(destination.width),area.right,clip.right})||
           std::max({0,area.top,clip.top})>=std::min({int(destination.height),area.bottom,clip.bottom}))return true;
        unsigned short const* colors=nullptr;
        if(bits==8){
            if(!palette)palette=*reinterpret_cast<void**>(static_cast<char*>(source)+0x10);
            if(!palette){auto owner=*reinterpret_cast<void**>(module+0x70f48);if(owner)palette=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
            if(!palette)return false;
            auto table=*static_cast<void* const* const*>(palette);
            colors=reinterpret_cast<unsigned short const*(__thiscall*)(void const*)>(table[destination.format==Format::rgb565?7:6])(palette);
            if(!colors)return false;
        }
        std::vector<std::uint32_t> decoded(std::size_t(w)*h);
        auto table=*static_cast<void***>(source);
        auto pixels=reinterpret_cast<unsigned char*(__thiscall*)(void*)>(table[8])(source);
        if(!pixels)return false;
        for(int y=0;y<h;++y){
            unsigned left=trimmed?rows[y*4]:0,count=trimmed?rows[y*4+1]:unsigned(w);
            unsigned offset=trimmed?unsigned(rows[y*4+2])|(unsigned(rows[y*4+3])<<8):unsigned(y)*stride;
            for(unsigned x=0;x<count;++x){
                unsigned c=bits==8?pixels[offset+x]:reinterpret_cast<unsigned short*>(pixels)[offset+x];
                decoded[std::size_t(y)*w+left+x]=bits==8?(c<254?65536u|colors[c]:0u):(c!=(key&65535)?65536u|c:0u);
            }
        }
        if(trimmed)*reinterpret_cast<unsigned*>(static_cast<char*>(source)+0x28)=key&255; // native source metadata side effect
        reinterpret_cast<void(__thiscall*)(void*,int)>(table[9])(source,1);
        if(!sprite_image||sprite_width!=unsigned(w)||sprite_height!=unsigned(h)){
            if(sprite_image)gpu.destroy(sprite_image);
            sprite_image=0;sprite_pixels.clear();sprite_revision=0;
            sprite_image=gpu.create(w,h,Format::bgra32);if(!sprite_image)return false;
            sprite_width=w;sprite_height=h;
        }
        // Compare actual source pixels and palette expansion, including writes
        // through retained pointers. A source address is not a content revision.
        if(decoded!=sprite_pixels){
            if(!gpu.upload(sprite_image,++sprite_revision,decoded.data(),decoded.size()))return false;
            sprite_pixels=std::move(decoded);
        }
        Command sprite_command={Kind::native_sprite,destination.gpu,sprite_image,area,clip};
        Command commands[2]={sprite_command,sprite_command};
        static_assert(int(Kind::native_sprite)==6);
        unsigned count=1;
        if(destination.detail){commands[1]=commands[0];commands[1].destination=destination.detail;
            commands[1].color=destination.format==Format::rgb565?2:1;count=2;}
        if(!gpu.submit(commands,count))return false;
        destination.dirty=true;++counters.translated;return true;
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
        if(image.detail){gpu.destroy(image.detail);image.detail=0;}
        image.owned=false;
    }
public:
    Adapter(Backend& g,void* bits,void* release):gpu(g),get_bits(bits),release_bits(release){}
    ~Adapter(){for(auto& image:images)if(image.native)forget(image);if(sprite_image)gpu.destroy(sprite_image);}
    Adapter(Adapter const&)=delete;Adapter& operator=(Adapter const&)=delete;
    // Call drain while native objects/device still exist. A synchronization/device
    // failure is terminal for this isolated backend, never a stale-pixel fallback.
    void drain(){for(auto& image:images)if(image.native){cpu_ownership(image);forget(image);}if(sprite_image)gpu.destroy(sprite_image);sprite_image=0;sprite_pixels.clear();}
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
        if(op==C3X_NATIVE_SPRITE){
            if(destination&&destination->owned&&draw_sprite(*destination,source,source_rect,target_rect))return 1;
            if(destination)cpu_ownership(*destination);++counters.fallbacks;return 0;
        }
        if(op!=C3X_NATIVE_COPY&&op!=C3X_NATIVE_FILL&&op!=C3X_NATIVE_IMAGE_DRAW)return 0;
        auto fallback=[&](){if(destination)cpu_ownership(*destination);auto s=find(source);if(s&&s!=destination)cpu_ownership(*s);++counters.fallbacks;return 0;};
        if(!destination||!destination->owned)return fallback();
        Image* input=nullptr;
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
            if(command.kind==Kind::color_key&&s->detail)return fallback(); // native key compares packed words, not full-color RGB
            if(!s->owned&&!refresh(*s))return fallback();
            input=s;
            command.source=s->gpu;
        }
        // Decide unsupported native geometry before acknowledging a queued
        // operation. A later worker rejection cannot replay a skipped JGL call.
        if(command.area.left>command.area.right||command.area.top>command.area.bottom||
           command.clip.left>command.clip.right||command.clip.top>command.clip.bottom)return fallback();
        Rect selected={std::max({0,command.area.left,command.clip.left}),std::max({0,command.area.top,command.clip.top}),
            std::min({int(destination->width),command.area.right,command.clip.right}),std::min({int(destination->height),command.area.bottom,command.clip.bottom})};
        if(input&&selected.left<selected.right&&selected.top<selected.bottom){
            auto x=std::int64_t(command.source_x)+selected.left-command.area.left,y=std::int64_t(command.source_y)+selected.top-command.area.top;
            if(x<0||y<0||x+selected.right-selected.left>input->width||y+selected.bottom-selected.top>input->height)return fallback();
        }
        Command commands[2]={command,command};unsigned count=1;
        if(input&&input->detail&&!full_color(*destination))return fallback();
        if(destination->detail){
            auto& detail=commands[1];detail.destination=destination->detail;
            if(command.kind==Kind::fill)detail.color=expanded(command.color,destination->format);
            else if(input->detail)detail.source=input->detail;
            else {detail.kind=Kind::expand;detail.source=input->gpu;detail.color=command.kind==Kind::copy?65536:command.color;}
            count=2;
        }
        if(!gpu.submit(commands,count))return fallback();
        destination->dirty=true;++counters.translated;return 1;
    }
    // The map is an immutable resident source. Quantize only at the native
    // destination, using the same world-anchored rounding as the CPU blitter.
    bool insert_map(void* p,Id map,Rect area,int source_x,int source_y,int phase_x,int phase_y){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native map adapter thread changed");
        auto d=find(p);if(!d||!d->owned)return false;
        Command c={Kind::quantize,d->gpu,map,area,rect(static_cast<char*>(p)+0x44),source_x,source_y,
            (unsigned(phase_x)&7u)|((unsigned(phase_y)&7u)<<3)};
        if(!full_color(*d))return false;
        Command commands[2]={c,c};commands[1].kind=Kind::copy;commands[1].destination=d->detail;commands[1].color=0;
        if(!gpu.submit(commands,2))return false;d->dirty=true;return true;
    }
    bool draw_unit(c3x_renderer_gpu_unit_fn draw,std::int64_t ticket,c3x_renderer_unit_v1 const& unit,void* target,void* background,int* bounds,unsigned flags){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native unit adapter thread changed");
        auto d=find(target);if(!draw||!bounds||!d||!d->owned)return false;
        auto b=find(background);if(!b)b=create(background,false);
        if(!b||b->format!=d->format)return false;
        if(!b->owned&&!refresh(*b))return false;
        if(b->detail&&!full_color(*d))return false;
        auto clip=rect(static_cast<char*>(target)+0x44);
        c3x_renderer_gpu_unit_v1 request={sizeof(request),ticket,std::int64_t(d->gpu),std::int64_t(b->gpu),std::int64_t(d->detail),std::int64_t(b->detail),{clip.left,clip.top,clip.right,clip.bottom},flags};
        gpu.flush();int result=draw(&unit,&request,bounds);
        if(result==C3X_RENDERER_RESULT_OK){d->dirty=true;++counters.translated;return true;}
        if(result!=C3X_RENDERER_RESULT_BAD_ARGUMENT)throw std::runtime_error("GPU unit composition failed; native pixels cannot be published");
        cpu_ownership(*d);if(b!=d)cpu_ownership(*b);return false;
    }
    Id display_image(void* p){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native display adapter thread changed");
        auto d=find(p);return d&&d->owned&&full_color(*d)?d->detail:0;
    }
    Id image(void* p){auto i=find(p);return i?i->gpu:0;}
    bool owns(void* p){auto i=find(p);return i&&i->owned;}
    Counts stats()const{return counters;}
};
} // namespace c3x_native_images
