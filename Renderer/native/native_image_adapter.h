#pragma once
// Native ownership stays on the caller thread; Backend owns ordered image
// commands and explicit CPU barriers. No native pointers cross to the GPU worker.
// Live surface/presentation admission is separate from this tested adapter.
#include "gpu_frame_api.h"
#include "gpu_image_commands.h"
#include "native_text_raster.h"
#include <array>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <climits>

namespace c3x_native_images {
using namespace c3x_gpu_images;
struct Counts {std::uint64_t translated=0,fallbacks=0,readbacks=0,readback_bytes=0,source_checks=0,source_reuses=0,source_expanded_bytes=0,text_builds=0,text_hits=0;};
template<class Backend> class Adapter {
    struct Image {void* native=nullptr;Id gpu=0,detail=0;unsigned width=0,height=0;Format format=Format::rgb555;
        bool owned=false,dirty=false,cpu_uploaded=false;std::uint64_t revision=0,used=0;std::vector<std::uint16_t> cpu;};
    Backend& gpu;
    c3x_renderer_native_lifetime_fn lifetime;
    void* get_bits;void* release_bits;DWORD thread=GetCurrentThreadId();
    std::array<Image,32> images={};std::uint64_t cpu_bytes=0,source_age=0;
    static constexpr std::uint64_t cpu_budget=64u*1024u*1024u;
    Counts counters;unsigned large_cpu_barrier_reports=0,copy_rejection_reports=0,sprite_rejection_reports=0,blend_rejection_reports=0;
    Id sprite_image=0;unsigned sprite_width=0,sprite_height=0;
    std::uint64_t sprite_revision=0;std::vector<std::uint32_t> sprite_pixels;
    struct Lookup {Id image=0;std::uint64_t revision=0;std::vector<std::uint16_t> words;};
    std::array<Lookup,2> lookups; // Full effects and the small native shadow table coexist.
    struct Text {c3x_native_text::State font_state;std::string text;
        Id pixels=0,curves=0;Rect area={};int advance=0,ascent=0,height=0;std::uint64_t age=0,bytes=0;};
    std::array<Text,32> texts={};std::uint64_t text_age=0,text_bytes=0;
    void retire_text(Text& text){if(text.pixels)gpu.destroy(text.pixels);if(text.curves)gpu.destroy(text.curves);text_bytes-=text.bytes;text={};}
    bool draw_text(Image& destination,void* text,void const* target,unsigned count){
        if(!target||(!text&&count)||count>1024)return false;if(!count)return true;
        auto dc=*reinterpret_cast<HDC*>(static_cast<char*>(destination.native)+0x4bc);
        c3x_native_text::State font_state;if(!c3x_native_text::capture(dc,font_state)||(font_state.align&(TA_UPDATECP|TA_RTLREADING)))return false;
        RECT clip={};auto clip_kind=GetClipBox(dc,&clip);if(clip_kind==ERROR||clip_kind==COMPLEXREGION)return false;if(clip_kind==NULLREGION)return true;
        Text* cached=nullptr;for(auto& entry:texts)if(entry.age&&entry.font_state==font_state&&entry.text.size()==count&&!std::memcmp(entry.text.data(),text,count)){cached=&entry;break;}
        if(cached)++counters.text_hits;
        else {
            ++counters.text_builds;c3x_native_text::Raster raster;
            if(!c3x_native_text::compile(dc,font_state,static_cast<char const*>(text),count,raster))return false;
            auto bytes=(raster.pixels.size()+raster.curves.size())*4;
            if(bytes>8u*1024u*1024u)return false;
            while(text_bytes+bytes>8u*1024u*1024u){auto oldest=std::min_element(texts.begin(),texts.end(),[](Text const& a,Text const& b){return (a.age?a.age:UINT64_MAX)<(b.age?b.age:UINT64_MAX);});retire_text(*oldest);}
            cached=&*std::min_element(texts.begin(),texts.end(),[](Text const& a,Text const& b){return a.age<b.age;});retire_text(*cached);
            auto pixels=gpu.create(raster.width,raster.height,Format::bgra32);if(!pixels)return false;
            auto curves=gpu.create(17,unsigned(raster.curves.size()/17),Format::bgra32);if(!curves){gpu.destroy(pixels);return false;}
            if(!gpu.upload(pixels,1,raster.pixels.data(),raster.pixels.size())||!gpu.upload(curves,1,raster.curves.data(),raster.curves.size())){gpu.destroy(pixels);gpu.destroy(curves);return false;}
            SIZE extent={};TEXTMETRICA metrics={};GetTextExtentPoint32A(dc,static_cast<char const*>(text),int(count),&extent);GetTextMetricsA(dc,&metrics);
            cached->font_state=font_state;cached->text.assign(static_cast<char const*>(text),count);
            cached->pixels=pixels;cached->curves=curves;cached->area={raster.left,raster.top,raster.left+int(raster.width),raster.top+int(raster.height)};
            cached->advance=extent.cx;cached->ascent=metrics.tmAscent;cached->height=metrics.tmHeight;cached->bytes=bytes;text_bytes+=bytes;
        }
        cached->age=++text_age;auto anchor=rect(target);int x=anchor.left,y=anchor.top;
        if((font_state.align&TA_CENTER)==TA_CENTER)x-=cached->advance/2;else if(font_state.align&TA_RIGHT)x-=cached->advance;
        if((font_state.align&TA_BASELINE)==TA_BASELINE)y-=cached->ascent;else if(font_state.align&TA_BOTTOM)y-=cached->height;
        auto area=cached->area;
        if(std::int64_t(x)+area.left<INT_MIN||std::int64_t(x)+area.right>INT_MAX||std::int64_t(y)+area.top<INT_MIN||std::int64_t(y)+area.bottom>INT_MAX)return false;
        area={area.left+x,area.top+y,area.right+x,area.bottom+y};
        Command command={Kind::native_text,destination.gpu,cached->pixels,area,rect(&clip),0,0,0,cached->curves};
        Command commands[2]={command,command};unsigned n=1;
        if(destination.detail){commands[1].destination=destination.detail;n=2;}
        if(!gpu.submit(commands,n))return false;destination.dirty=true;++counters.translated;return true;
    }
    using Get=std::uint16_t*(__thiscall*)(void*);
    using Release=void(__thiscall*)(void*,int);
    static int field(void* p,unsigned offset){return *reinterpret_cast<int*>(static_cast<char*>(p)+offset);}
    bool native_sprite(void* p)const{
        auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
        return module&&p&&*static_cast<void***>(p)==reinterpret_cast<void**>(module+0x68440);
    }
    static Rect rect(void const* p){auto r=static_cast<RECT const*>(p);return {r->left,r->top,r->right,r->bottom};}
    Image* find(void* p){for(auto& image:images)if(p&&image.native==p){image.used=++source_age;return &image;}return nullptr;}
    void forget(Image& image){if(image.detail)gpu.destroy(image.detail);gpu.destroy(image.gpu);cpu_bytes-=image.cpu.size()*2;image={};}
    void trim_cpu_sources(){
        // Trim only between native operations, before borrowing any Image*.
        // Retained recipes own immutable versions; CPU pixels authorize these
        // unowned mirrors again on demand. Never evict authoritative GPU images.
        auto count=std::size_t(std::count_if(images.begin(),images.end(),[](auto const& i){return i.native!=nullptr;}));
        if(count<images.size()-2 && cpu_bytes<=48u*1024u*1024u)return;
        while(count>images.size()-8 || cpu_bytes>48u*1024u*1024u){
            Image* oldest=nullptr;
            for(auto& i:images)if(i.native&&!i.owned&&!i.dirty&&(!oldest||i.used<oldest->used))oldest=&i;
            if(!oldest)break;forget(*oldest);--count;
        }
    }
    Image* create(void* p,bool owned){
        if(!p||field(p,0x24)!=16)return nullptr;
        int w=field(p,0x38),h=field(p,0x3c),stride=field(p,0x40);
        if(w<=0||h<=0||w>2240||h>1260||stride<w)return nullptr;
        // Derive format from the actual DIB; 16-bit alone does not distinguish 555/565.
        DIBSECTION dib={};auto bitmap=*reinterpret_cast<HBITMAP*>(static_cast<char*>(p)+0x4b4);
        if(GetObject(bitmap,sizeof dib,&dib)!=sizeof dib||dib.dsBm.bmBitsPixel!=16)return nullptr;
        Format format;
        if(dib.dsBmih.biCompression==BI_RGB)format=Format::rgb555;
        else if(dib.dsBmih.biCompression==BI_BITFIELDS&&dib.dsBitfields[0]==0x7c00&&dib.dsBitfields[1]==0x3e0&&dib.dsBitfields[2]==0x1f)format=Format::rgb555;
        else if(dib.dsBmih.biCompression==BI_BITFIELDS&&dib.dsBitfields[0]==0xf800&&dib.dsBitfields[1]==0x7e0&&dib.dsBitfields[2]==0x1f)format=Format::rgb565;
        else return nullptr;
        if(std::uint64_t(w)*h*2>cpu_budget-cpu_bytes)return nullptr;
        for(auto& image:images)if(!image.native){
            std::vector<std::uint16_t> bytes(std::size_t(w)*h);
            auto id=gpu.create(w,h,format);if(!id)return nullptr;
            image.native=p;image.used=++source_age;image.gpu=id;image.width=w;image.height=h;image.format=format;image.owned=owned;
            image.cpu=std::move(bytes);cpu_bytes+=image.cpu.size()*2;return &image;
        }return nullptr;
    }
    bool refresh(Image& image){
        ++counters.source_checks;
        std::vector<std::uint32_t> content;
        std::vector<std::uint16_t> captured;
        {
            // Retained native pointers can change without another getter call.
            // Compare every visible word, in its native representation, before
            // allocating or widening an upload. Padding is not image content.
            GdiFlush();auto bits=reinterpret_cast<Get>(get_bits)(image.native);
            if(!bits)throw std::runtime_error("native image lease failed");
            struct Lease {void* image;Release release;~Lease(){release(image,1);}} lease{image.native,reinterpret_cast<Release>(release_bits)};
            auto stride=field(image.native,0x40);
            auto count=std::size_t(image.width)*image.height;
            bool same=image.cpu_uploaded&&image.cpu.size()==count;
            for(unsigned y=0;same&&y<image.height;++y)
                same=std::memcmp(image.cpu.data()+std::size_t(y)*image.width,bits+std::size_t(y)*stride,image.width*2)==0;
            if(same){++counters.source_reuses;return true;}
            content.resize(count);if(!image.owned)captured.resize(count);
            for(unsigned y=0;y<image.height;++y){
                auto row=bits+std::size_t(y)*stride;auto offset=std::size_t(y)*image.width;
                std::copy_n(row,image.width,content.data()+offset);
                if(!image.owned)std::memcpy(captured.data()+offset,row,image.width*2);
            }
            counters.source_expanded_bytes+=content.size()*4;
        } // Release the private native lease before dispatching to the worker.
        if(!gpu.upload(image.gpu,image.revision+1,content.data(),content.size()))return false;
        ++image.revision;image.cpu_uploaded=true;
        cpu_bytes-=image.cpu.size()*2;image.cpu=std::move(captured);cpu_bytes+=image.cpu.size()*2;
        return true;
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
    bool upload_sprite(std::vector<unsigned>& decoded,unsigned width,unsigned height){
        if(!sprite_image||sprite_width!=width||sprite_height!=height){
            if(sprite_image)gpu.destroy(sprite_image);
            sprite_image=0;sprite_pixels.clear();sprite_revision=0;
            sprite_image=gpu.create(width,height,Format::bgra32);if(!sprite_image)return false;
            sprite_width=width;sprite_height=height;
        }
        if(decoded!=sprite_pixels){
            if(!gpu.upload(sprite_image,++sprite_revision,decoded.data(),decoded.size()))return false;
            sprite_pixels=std::move(decoded);
        }
        return true;
    }
    Id upload_lookup(void const* table,unsigned blocks=16){
        if(!table||!blocks||blocks>31)return 0;
        auto& cache=lookups[blocks<=4?1:0];auto words=static_cast<std::uint16_t const*>(table);unsigned count=blocks*32768u;
        if(cache.image&&cache.words.size()>=count&&!std::memcmp(cache.words.data(),words,std::size_t(count)*2))return cache.image;
        // Content, not pointer identity, validates caller-owned mutable tables.
        // Shadow and FLC tables are simultaneously resident, so alternating
        // native draws do not replace/upload each other's shared lookup asset.
        std::vector<std::uint16_t> captured(words,words+count);
        unsigned height=blocks<=4?128:1024;
        std::vector<std::uint32_t> pixels(std::size_t(1024)*height);std::copy(captured.begin(),captured.end(),pixels.begin());
        if(!cache.image){cache.image=gpu.create(1024,height,Format::bgra32);if(!cache.image)return 0;}
        if(!gpu.upload(cache.image,++cache.revision,pixels.data(),pixels.size()))return 0;
        cache.words=std::move(captured);return cache.image;
    }
    bool draw_lookup(Image& destination,void* source,void const* packet,void const* target){
        if(!source||!packet||!target)return false;
        auto const& inputs=*static_cast<c3x_renderer_native_lookup const*>(packet);
        if(inputs.percent<0||inputs.percent>100)return false;
        auto background=find(source);if(!background)background=create(source,false);
        if(!background||background->format!=destination.format||background->width!=destination.width||background->height!=destination.height||
           field(source,0x40)!=field(destination.native,0x40))return false;
        auto area=rect(target),clip=Rect{0,0,int(destination.width),int(destination.height)};
        if(std::max(0,area.left)>=std::min(int(destination.width),area.right)||std::max(0,area.top)>=std::min(int(destination.height),area.bottom))return false; // Native return 3.
        if(!background->owned&&!refresh(*background))return false;
        if(background->detail&&!full_color(destination))return false;
        auto lookup_image=upload_lookup(inputs.table);if(!lookup_image)return false;
        // JGL promotes its float constant before x87 multiplication and truncates.
        // In particular 40 percent selects block 10, not block 9.
        unsigned block=15-unsigned(double(inputs.percent)*double(0.01f)*15.0);
        Command command={Kind::native_lookup,destination.gpu,lookup_image,area,clip,0,0,block,background->gpu,destination.detail,background->detail};
        if(!gpu.submit(&command,1))return false;destination.dirty=true;++counters.translated;return true;
    }
    bool draw_blend(Image& destination,void* source,void const* packet,void const* target,unsigned mode){
        if(!source||!packet||!target||mode>1)return false;
        auto const& inputs=*static_cast<c3x_renderer_native_sprite_blend const*>(packet);
        auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
        if(!module||!inputs.alpha||!inputs.background)return false;
        for(auto sprite:{source,inputs.alpha})if(*static_cast<void***>(sprite)!=reinterpret_cast<void**>(module+0x68440)||field(sprite,0x20)!=8||(field(sprite,0x18)&1))return false;
        int width=field(source,0x30),height=field(source,0x34),stride=field(source,0x2c);
        if(width<=0||height<=0||width>1024||height>1024||stride<width||field(inputs.alpha,0x30)<width||field(inputs.alpha,0x34)<height||field(inputs.alpha,0x2c)!=stride)return false;
        // The straight-alpha native loop traverses contiguous source bytes and
        // does not apply the image clip. Admit only its bounded rectangle form.
        if(mode&&stride!=width)return false;
        auto area=rect(target);
        if(std::int64_t(area.left)+width>INT_MAX||std::int64_t(area.top)+height>INT_MAX)return false;
        area.right=area.left+width;area.bottom=area.top+height;
        auto background=find(inputs.background);if(!background)background=create(inputs.background,false);
        if(!background||background->format!=destination.format||background->width!=destination.width||background->height!=destination.height||field(inputs.background,0x40)!=field(destination.native,0x40))return false;
        auto clip=mode?Rect{0,0,int(destination.width),int(destination.height)}:rect(static_cast<char*>(inputs.background)+0x44);
        if(mode&&(area.left<0||area.top<0||area.right>int(destination.width)||area.bottom>int(destination.height)))return false;
        if(std::max({0,clip.left,area.left})>=std::min({int(destination.width),clip.right,area.right})||std::max({0,clip.top,area.top})>=std::min({int(destination.height),clip.bottom,area.bottom}))return true;
        auto palette=inputs.palette?inputs.palette:*reinterpret_cast<void**>(static_cast<char*>(source)+0x10);
        if(!palette){auto owner=*reinterpret_cast<void**>(module+0x70f48);if(owner)palette=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
        if(!palette)return false;
        auto table=*static_cast<void***>(palette);unsigned colors[256]={};
        if(mode){
            for(unsigned n=0;n<256;++n){auto rgb=reinterpret_cast<unsigned char*(__thiscall*)(void*,unsigned)>(table[8])(palette,n);
                if(!rgb)return false;colors[n]=(unsigned(rgb[0])<<16)|(unsigned(rgb[1])<<8)|rgb[2];}
        }else{
            auto words=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(table[destination.format==Format::rgb565?7:6])(palette);
            if(!words)return false;for(unsigned n=0;n<256;++n)colors[n]=words[n];
        }
        std::vector<unsigned> decoded(std::size_t(width)*height); // Allocate before borrowing either native source.
        auto source_table=*static_cast<void***>(source),alpha_table=*static_cast<void***>(inputs.alpha);
        auto pixels=reinterpret_cast<unsigned char*(__thiscall*)(void*)>(source_table[8])(source);
        if(!pixels)return false;
        auto alpha=reinterpret_cast<unsigned char*(__thiscall*)(void*)>(alpha_table[8])(inputs.alpha);
        if(!alpha){reinterpret_cast<Release>(source_table[9])(source,1);return false;}
        for(int y=0;y<height;++y)for(int x=0;x<width;++x){unsigned i=y*stride+x,index=pixels[i],weight=alpha[i];
            // These alpha programs key only 255, unlike ordinary sprite draw.
            decoded[std::size_t(y)*width+x]=index==255||weight==255?0xff000000u:colors[index]|(weight<<24);}
        reinterpret_cast<Release>(alpha_table[9])(inputs.alpha,1);reinterpret_cast<Release>(source_table[9])(source,1);
        if(!background->owned&&!refresh(*background))return false;
        if(background->detail&&!full_color(destination))return false;
        if(!upload_sprite(decoded,unsigned(width),unsigned(height)))return false;
        Command command={Kind::native_blend,destination.gpu,sprite_image,area,clip,0,0,mode,background->gpu,destination.detail,background->detail};
        if(!gpu.submit(&command,1))return false;
        destination.dirty=true;++counters.translated;return true;
    }
    bool draw_diagonal(Image& destination,Rect endpoints,unsigned color){
        // JGL normalizes X, includes both endpoints, and advances the minor
        // coordinate only when its doubled Bresenham error is strictly positive.
        int x=endpoints.left,y=endpoints.top,x1=endpoints.right,y1=endpoints.bottom;
        if(x>x1){std::swap(x,x1);std::swap(y,y1);}
        auto clip=rect(static_cast<char*>(destination.native)+0x44);
        Rect area={std::max({0,x,clip.left}),std::max({0,std::min(y,y1),clip.top}),
            std::min({int(destination.width),x1+1,clip.right}),std::min({int(destination.height),std::max(y,y1)+1,clip.bottom})};
        if(area.left>=area.right||area.top>=area.bottom)return true;
        unsigned w=unsigned(area.right-area.left),h=unsigned(area.bottom-area.top);
        std::vector<std::uint32_t> decoded(std::size_t(w)*h);
        int dx=x1-x,dy=std::abs(y1-y),step=y1>y?1:-1,major=std::max(dx,dy),minor=std::min(dx,dy),error=2*minor-major;
        for(int n=0;n<=major;++n){
            if(x>=area.left&&x<area.right&&y>=area.top&&y<area.bottom)decoded[std::size_t(y-area.top)*w+x-area.left]=65536|color;
            if(error>0){if(dx>=dy)y+=step;else ++x;error-=2*major;}
            if(dx>=dy)++x;else y+=step;error+=2*minor;
        }
        if(!upload_sprite(decoded,w,h))return false;
        Command commands[2]={{Kind::native_sprite,destination.gpu,sprite_image,area,clip},{}};unsigned count=1;
        if(destination.detail){commands[1]=commands[0];commands[1].destination=destination.detail;
            commands[1].color=destination.format==Format::rgb565?2:1;count=2;}
        if(!gpu.submit(commands,count))return false;
        destination.dirty=true;++counters.translated;return true;
    }
    bool draw_sprite(Image& destination,void* source,void const* palette,void const* target,void const* lookup=nullptr,Image* background=nullptr,int const* native_scale=nullptr,int style=0,unsigned style_color=0,float opacity=1){
        if(!source||!target)return false;
        auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
        if(!module||*static_cast<void***>(source)!=reinterpret_cast<void**>(module+0x68440))return false;
        // Audited slot 17 ordinary and row-trimmed 8-bit sources, plus ordinary
        // 16-bit sources. Positive scaling follows each native source program;
        // mirrored sources and unsafe trimmed edges retain native ownership.
        int bits=field(source,0x20),w=field(source,0x30),h=field(source,0x34),stride=field(source,0x2c);
        bool trimmed=(field(source,0x18)&1)!=0;
        if(lookup&&!background&&style!=3&&(trimmed||bits==16)){++counters.translated;return true;} // Audited native lookup no-op.
        if(style&&(style<1||style>4||bits!=8))return false;
        if(style&&!field(source,0x14))return false;
        if((style==1||style==3||style==4)&&trimmed){++counters.translated;return true;} // Native no-op for these source layouts.
        if(style==4){
            if(!(opacity>=0&&opacity<=1))return false;
            if(opacity<=0.03125f){++counters.translated;return true;}
            // The pinned native opaque threshold is 100 - 1/32 despite
            // accepting only [0,1]; even opacity 1 uses its 15/16 helper.
            // JGL selects eighths inclusively within +/- 1/32, otherwise
            // the intervening sixteenth. Keep its exact endpoint choices.
            unsigned step=1;
            for(unsigned even=2;even<=14;even+=2){
                if(opacity<float(even)/16-0.03125f)break;
                step=even;if(opacity<=float(even)/16+0.03125f)break;step=even+1;
            }
            style_color=step|((style_color&255)?256u:0u);
        }
        if(style==3&&destination.format!=Format::rgb555)return false;
        if(background&&(bits!=8||trimmed||destination.format!=Format::rgb555))return false;
        if((bits!=8&&bits!=16)||(bits==16&&trimmed)||w<1||h<1||w>1024||h>1024||(!trimmed&&stride<w))return false;
        auto rows=trimmed?*reinterpret_cast<unsigned char**>(static_cast<char*>(source)+0x1c):nullptr;
        if(trimmed&&!rows)return false;
        if(trimmed)for(int y=0;y<h;++y)if(unsigned(rows[y*4])+rows[y*4+1]>unsigned(w))return false;
        int denominator=*reinterpret_cast<int*>(module+0x6c104);
        int scale_x=*reinterpret_cast<int*>(module+0x6c0fc),scale_y=*reinterpret_cast<int*>(module+0x6c100);
        if(background&&!native_scale){
            if(denominator<=0||denominator>32767||scale_x==INT_MIN||scale_y==INT_MIN)return false;
            scale_x=std::abs(scale_x);scale_y=std::abs(scale_y);
            if(scale_x!=denominator||scale_y!=denominator){++counters.translated;return true;} // Native FLC form does not scale.
        }
        if(native_scale){
            if(native_scale[0]!=native_scale[1]||native_scale[0]==0||native_scale[0]==INT_MIN||native_scale[2]<=0||native_scale[2]>32767||scale_x==INT_MIN||scale_y==INT_MIN)return false;
            if(native_scale[0]<0){++counters.translated;return true;} // Native scaled FLC mirror is a no-op.
            denominator=native_scale[2];scale_x=std::abs(scale_x);scale_y=std::abs(scale_y);
        }
        if(denominator<=0||denominator>32767||scale_x<=0||scale_y<=0)return false;
        bool scaled=scale_x!=denominator||scale_y!=denominator;
        unsigned key=unsigned(field(source,0x28));
        if(bits==16&&!(key&0x80000000u)){
            // JGL resolves this key through the destination palette and mutates
            // the source descriptor before drawing, even for its scaled no-op.
            auto selected=*reinterpret_cast<void**>(static_cast<char*>(destination.native)+0x7c);
            if(!selected){auto owner=*reinterpret_cast<void**>(module+0x70f48);if(owner)selected=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
            if(selected){auto table=*static_cast<void***>(selected);
                auto words=reinterpret_cast<unsigned short const*(__thiscall*)(void*)>(table[destination.format==Format::rgb565?7:6])(selected);
                if(!words)return false;key=words[key&255];}
        }
        // The verified native 16-bit helper deliberately does not draw scaled
        // sprites. Preserve that behavior without reading a resident destination.
        if(bits==16&&scaled){*reinterpret_cast<unsigned*>(static_cast<char*>(source)+0x28)=key;++counters.translated;return true;}
        int output_width=w,output_height=h,offset_x=0,offset_y=0;
        std::uint64_t step_x=65536,step_y=65536;
        if(scaled||native_scale){
            float x=float(double(scale_x)/denominator),y=float(double(scale_y)/denominator);
            if(double(w)*x>2240||double(h)*y>1260)return false;
            output_width=int(double(w)*x);output_height=int(double(h)*y);
            // Ordinary JGL indexed scaling starts one scaled source-pixel beyond
            // the supplied anchor and samples with truncated 16.16 increments.
            if(!trimmed){offset_x=int(x);offset_y=int(y);}
            step_x=(std::uint64_t(denominator)<<16)/unsigned(scale_x);
            step_y=(std::uint64_t(denominator)<<16)/unsigned(scale_y);
            if(!output_width||!output_height)return true;
            if(((std::uint64_t(output_width-1)*step_x)>>16)>=unsigned(w)||((std::uint64_t(output_height-1)*step_y)>>16)>=unsigned(h))return false;
        }
        auto area=rect(target);
        if(std::int64_t(area.left)+offset_x+output_width>INT_MAX||std::int64_t(area.top)+offset_y+output_height>INT_MAX)return false;
        area.left+=offset_x;area.top+=offset_y;
        area.right=area.left+output_width;area.bottom=area.top+output_height;
        auto clip=rect(static_cast<char*>(background?background->native:destination.native)+0x44);
        std::uint64_t native_row_step=0;
        if(native_scale){
            // Slot 34 clips the destination first, then restarts at source byte
            // zero. Its native zoom loop consumes every second indexed byte;
            // the row increment also depends on the clipped width. Preserve
            // this program instead of substituting generic nearest sampling.
            area={std::max({0,area.left,clip.left}),std::max({0,area.top,clip.top}),
                  std::min({int(destination.width),area.right,clip.right}),std::min({int(destination.height),area.bottom,clip.bottom})};
            if(area.left>=area.right||area.top>=area.bottom)return true;
            output_width=area.right-area.left;output_height=area.bottom-area.top;
            auto step=(std::uint64_t(denominator)<<16)/unsigned(native_scale[0]);
            auto row=std::int64_t(step>>16)*stride+2LL*output_width-std::int64_t((std::uint64_t(output_width)*step)>>16);
            if(row<0||std::uint64_t(output_height-1)*row+2u*(output_width-1)>=std::uint64_t(stride)*h)return false;
            native_row_step=std::uint64_t(row);
        }
        if(std::max({0,area.left,clip.left})>=std::min({int(destination.width),area.right,clip.right})||
           std::max({0,area.top,clip.top})>=std::min({int(destination.height),area.bottom,clip.bottom})){
            if(bits==16||trimmed)*reinterpret_cast<unsigned*>(static_cast<char*>(source)+0x28)=trimmed?key&255:key;return true;}
        unsigned short const* colors=nullptr;
        if(bits==8){
            if(!palette)palette=*reinterpret_cast<void**>(static_cast<char*>(source)+0x10);
            if(!palette){auto owner=*reinterpret_cast<void**>(module+0x70f48);if(owner)palette=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
            if(!palette)return false;
            auto table=*static_cast<void* const* const*>(palette);
            colors=reinterpret_cast<unsigned short const*(__thiscall*)(void const*)>(table[destination.format==Format::rgb565?7:6])(palette);
            if(!colors)return false;
        }
        if(style==2)style_color=!trimmed&&(style_color&0x80000000u)?style_color&65535:colors[style_color&255];
        // This is CPU source preparation, as for palette expansion/decompression;
        // composition still targets native/full-color GPU images exclusively.
        std::vector<std::uint32_t> decoded(std::size_t(output_width)*output_height);
        auto table=*static_cast<void***>(source);
        auto pixels=reinterpret_cast<unsigned char*(__thiscall*)(void*)>(table[8])(source);
        if(!pixels)return false;
        unsigned stream_end=0;
        if(trimmed)for(int y=0;y<h;++y)stream_end=std::max(stream_end,(unsigned(rows[y*4+2])|(unsigned(rows[y*4+3])<<8))+rows[y*4+1]);
        int first_x=0,last_x=output_width,first_y=0,last_y=output_height;
        if(trimmed&&scaled){
            first_x=std::max({0,clip.left,area.left})-area.left;last_x=std::min({int(destination.width),clip.right,area.right})-area.left;
            first_y=std::max({0,clip.top,area.top})-area.top;last_y=std::min({int(destination.height),clip.bottom,area.bottom})-area.top;
        }
        bool valid_source=true;
        for(int y=first_y;y<last_y&&valid_source;++y){
            unsigned source_y=native_scale?0:unsigned((std::uint64_t(y)*step_y)>>16);
            unsigned left=trimmed?rows[source_y*4]:0,count=trimmed?rows[source_y*4+1]:unsigned(w);
            unsigned offset=trimmed?unsigned(rows[source_y*4+2])|(unsigned(rows[source_y*4+3])<<8):source_y*stride;
            // JGL's scaled trimmed program stops advancing source/destination
            // rows at an empty descriptor and includes the byte after its count.
            // Preserve that observable traversal, but never read beyond the
            // source stream proven by its descriptors. Unsupported edges fall back.
            if(trimmed&&scaled&&!count)break;
            for(int x=first_x;x<last_x;++x){
                unsigned source_x=unsigned((std::uint64_t(x)*step_x)>>16);
                if(native_scale){auto c=pixels[std::uint64_t(y)*native_row_step+2u*x];
                    decoded[std::size_t(y)*output_width+x]=c<224?65536u|colors[c]:c==255?0xffffffffu:c<240?c-209:c-240;continue;}
                if(source_x<left)continue;source_x-=left;
                if(source_x>count||(!(trimmed&&scaled)&&source_x==count))continue;
                if(trimmed&&offset+source_x>=stream_end){valid_source=false;break;}
                unsigned c=bits==8?pixels[offset+source_x]:reinterpret_cast<unsigned short*>(pixels)[offset+source_x];
                if(style==4){
                    decoded[std::size_t(y)*output_width+x]=c<((style_color&256)?248u:255u)?colors[c]|((style_color&31)<<24):0xff000000u;continue;
                }
                if(style){
                    if(style==3&&c<254&&(c<248||c>251)){valid_source=false;break;} // The native map shadow allocation has four blocks.
                    decoded[std::size_t(y)*output_width+x]=style==3?(c<254?c-248:0xffffffffu):
                        (c<unsigned(style==1||trimmed?255:254)?65536u|(style==2?style_color:colors[c]):0u);continue;
                }
                decoded[std::size_t(y)*output_width+x]=background?(c<224?65536u|colors[c]:c==255?0xffffffffu:c<240?c-209:c-240):lookup?c:bits==8?(c<254?65536u|colors[c]:0u):(c!=(key&65535)?65536u|c:0u);
            }
        }
        reinterpret_cast<void(__thiscall*)(void*,int)>(table[9])(source,1);
        if(!valid_source)return false;
        // Source/palette bytes, including retained-pointer edits, prove reuse.
        if(!upload_sprite(decoded,unsigned(output_width),unsigned(output_height)))return false;
        Command sprite_command={Kind::native_sprite,destination.gpu,sprite_image,area,clip};
        Command commands[2]={sprite_command,sprite_command};
        static_assert(int(Kind::native_sprite)==6);
        unsigned count=1;
        if(destination.detail){commands[1]=commands[0];commands[1].destination=destination.detail;
            commands[1].color=destination.format==Format::rgb565?2:1;count=2;}
        if(style==4){
            Command command={Kind::native_blend,destination.gpu,sprite_image,area,clip,0,0,3,destination.gpu,destination.detail,destination.detail};
            if(!gpu.submit(&command,1))return false;
        }else if(lookup){
            auto lookup_image=upload_lookup(lookup,style==3?4:background?31:16);if(!lookup_image)return false;
            Command command={Kind::native_lookup,destination.gpu,lookup_image,area,clip,0,0,background||style==3?32u:0u,background?background->gpu:destination.gpu,destination.detail,background?background->detail:destination.detail};
            command.program=sprite_image;if(!gpu.submit(&command,1))return false;
        }else if(!gpu.submit(commands,count))return false;
        if(bits==16||trimmed)*reinterpret_cast<unsigned*>(static_cast<char*>(source)+0x28)=trimmed?key&255:key;
        destination.dirty=true;++counters.translated;return true;
    }
    void cpu_ownership(Image& image,int operation=0){
        if(image.dirty){
            if(image.width>=640 && image.height>=480 && large_cpu_barrier_reports++<16){
                // Guest x86 stack walking can stop in this DLL. Record the
                // actual adapter operation and stable slot instead of guessing
                // the native caller from an incomplete stack.
                char line[256];std::snprintf(line,sizeof(line),
                    "[C3X renderer] stage=native-cpu-barrier operation=%d slot=%u width=%u height=%u detail=%u owned=%u\n",
                    operation,unsigned(&image-images.data()),image.width,image.height,unsigned(image.detail!=0),unsigned(image.owned));
                OutputDebugStringA(line);
            }
            // GPU ownership makes the old CPU mirror stale. Materialize only
            // this explicit fallback, then release its temporary storage.
            std::vector<std::uint32_t> words(std::size_t(image.width)*image.height);
            if(!gpu.readback(image.gpu,words.data(),words.size()))
                throw std::runtime_error("cannot read current GPU image");
            GdiFlush();auto bits=reinterpret_cast<Get>(get_bits)(image.native);
            if(!bits)throw std::runtime_error("cannot restore native image ownership");
            auto stride=field(image.native,0x40);
            for(unsigned y=0;y<image.height;++y)for(unsigned x=0;x<image.width;++x)bits[y*stride+x]=std::uint16_t(words[y*image.width+x]);
            reinterpret_cast<Release>(release_bits)(image.native,1);
            ++counters.readbacks;counters.readback_bytes+=words.size()*4;image.dirty=false;
            // Reestablish CPU-upload revision validity only on its next source use.
            image.cpu_uploaded=false;
        }
        // Public access revokes startup evidence permanently until reinit. An
        // audited private native fallback can be admitted again after it finishes;
        // refresh then uploads its actual CPU result before any GPU drawing.
        if(image.detail){gpu.destroy(image.detail);image.detail=0;}
        image.owned=false;
    }
public:
    Adapter(Backend& g,void* bits,void* release,c3x_renderer_native_lifetime_fn evidence=nullptr):gpu(g),lifetime(evidence),get_bits(bits),release_bits(release){}
    ~Adapter(){for(auto& text:texts)retire_text(text);for(auto& image:images)if(image.native)forget(image);if(sprite_image)gpu.destroy(sprite_image);for(auto& lookup:lookups)if(lookup.image)gpu.destroy(lookup.image);}
    Adapter(Adapter const&)=delete;Adapter& operator=(Adapter const&)=delete;
    // Call drain while native objects/device still exist. A synchronization/device
    // failure is terminal for this isolated backend, never a stale-pixel fallback.
    void drain(){for(auto& text:texts)retire_text(text);for(auto& image:images)if(image.native){cpu_ownership(image,C3X_NATIVE_IMAGE_DRAIN);forget(image);}if(sprite_image)gpu.destroy(sprite_image);sprite_image=0;sprite_pixels.clear();for(auto& lookup:lookups)if(lookup.image)gpu.destroy(lookup.image);lookups={};}
    // Admission follows an actual destination demand. Startup observation proves
    // the entire lifetime even when its INIT preceded this GPU session. Sources
    // and unused canvases do not allocate resident destination pairs at startup.
    bool admit(void* object,char const** rejection=nullptr){
        auto reject=[&](char const* reason){if(rejection)*rejection=reason;return false;};
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native admission thread changed");
        auto d=find(object);if(d&&d->owned)return true;
        if(!lifetime||!lifetime(C3X_NATIVE_MAP,object,0))return reject("lifetime");
        // No takeover during even a private outstanding native lease.
        if(field(object,0x4c4)||field(object,0x4c8))return reject("outstanding-lease");
        if(!d)d=create(object,false);
        if(!d)return reject("surface-admission");
        if(!refresh(*d))return reject("upload");
        d->owned=true;
        cpu_bytes-=d->cpu.size()*2;std::vector<std::uint16_t>().swap(d->cpu);
        return true;
    }
    int operation(int op,void* object,void* source,void const* source_rect,void const* target_rect,unsigned color){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native GPU adapter thread changed");
        trim_cpu_sources();
        auto destination=find(object);
        if(op==C3X_NATIVE_IMAGE_DRAIN){drain();return 0;}
        if(op==C3X_NATIVE_DESTROY){if(destination)forget(*destination);return 0;}
        if(op==C3X_NATIVE_IMAGE_REINIT){if(destination){cpu_ownership(*destination,op);forget(*destination);}return 0;}
        if(op==C3X_NATIVE_INIT){
            // The startup service records INIT independently. The legacy fixture
            // without that service can admit only fresh observed native images.
            if(destination)forget(*destination);
            if(!lifetime){destination=create(object,true);if(destination&&!refresh(*destination))forget(*destination);}return 0;
        }
        if(op==C3X_NATIVE_PIXEL||op==C3X_NATIVE_BITS||op==C3X_NATIVE_DC){if(destination)cpu_ownership(*destination,op);return 0;}
        // Extend the map/save/display family only along an owned transfer.
        // Unrelated UI fills/sprites remain CPU-generated; they enter as upload
        // sources if subsequently drawn onto the resident map family.
        if(op==C3X_NATIVE_COPY||op==C3X_NATIVE_IMAGE_DRAW){auto input=find(source);if(input&&input->owned){
            char const* rejection=nullptr;
            if(!admit(object,&rejection)&&copy_rejection_reports++<16){
                char line[384];auto from=source_rect?rect(source_rect):Rect{};auto to=target_rect?rect(target_rect):Rect{};
                std::snprintf(line,sizeof(line),"[C3X renderer] stage=native-copy-admission operation=%d reason=%s source_slot=%u destination_slot=%d source=%ux%u destination=%dx%d bits=%d from=%d,%d,%d,%d to=%d,%d,%d,%d\n",
                    op,rejection?rejection:"unknown",unsigned(input-images.data()),destination?int(destination-images.data()):-1,
                    input->width,input->height,object?field(object,0x38):0,object?field(object,0x3c):0,object?field(object,0x24):0,
                    from.left,from.top,from.right,from.bottom,to.left,to.top,to.right,to.bottom);OutputDebugStringA(line);
            }
            destination=find(object);
        }}
        if(op==C3X_NATIVE_TEXT){
            if(destination&&destination->owned&&draw_text(*destination,source,target_rect,color))return 1;
            if(destination)cpu_ownership(*destination,op);++counters.fallbacks;return 0;
        }
        if(op==C3X_NATIVE_LOOKUP||op==C3X_NATIVE_SPRITE_LOOKUP||op==C3X_NATIVE_SPRITE_LOOKUP_OVER||op==C3X_NATIVE_SPRITE_LOOKUP_SCALED){
            if(op==C3X_NATIVE_SPRITE_LOOKUP&&native_sprite(source)&&!field(source,0x14))return 0;
            auto inputs=static_cast<c3x_renderer_native_lookup const*>(source_rect);
            Image* background=nullptr;bool compatible=true;
            if(op==C3X_NATIVE_SPRITE_LOOKUP_OVER||op==C3X_NATIVE_SPRITE_LOOKUP_SCALED){
                compatible=false;
                if(inputs&&inputs->background&&destination&&destination->owned){
                    background=find(inputs->background);if(!background)background=create(inputs->background,false);
                    compatible=background&&background->format==destination->format&&background->width==destination->width&&background->height==destination->height&&
                        field(background->native,0x40)==field(destination->native,0x40);
                    if(compatible&&!background->owned)compatible=refresh(*background);
                    if(compatible&&background->detail)compatible=full_color(*destination);
                }
            }
            if(destination&&destination->owned&&inputs&&inputs->table&&compatible&&
               (op==C3X_NATIVE_LOOKUP?draw_lookup(*destination,source,inputs,target_rect):draw_sprite(*destination,source,inputs->palette,target_rect,inputs->table,background,op==C3X_NATIVE_SPRITE_LOOKUP_SCALED?inputs->scale:nullptr)))return 1;
            if(destination)cpu_ownership(*destination,op);
            if(background&&background!=destination)cpu_ownership(*background,op);
            if(op==C3X_NATIVE_LOOKUP){auto input=find(source);if(input&&input!=destination)cpu_ownership(*input,op);}
            ++counters.fallbacks;return 0;
        }
        if(op==C3X_NATIVE_SPRITE_BLEND){
            // Pinned JGL slots 20/21/22 reject non-indexed sprite formats before
            // borrowing either image. Let native return 23 without destroying
            // the resident underlay (slot 22 only queries image metadata first).
            auto inputs=static_cast<c3x_renderer_native_sprite_blend const*>(source_rect);
            if(color<=1&&inputs&&native_sprite(source)&&native_sprite(inputs->alpha)&&
               (field(source,0x20)!=8||field(inputs->alpha,0x20)!=8))return 0;
            // A native blend reads a separate background just like a copy.
            // HUD scratch images start CPU-owned; extend the admitted family
            // from that background before deciding to fall back and read it.
            auto owned_background=inputs?find(inputs->background):nullptr;
            if(owned_background&&owned_background->owned&&(!destination||!destination->owned)){
                char const* reason=nullptr;
                if(admit(object,&reason))destination=find(object);
                else if(blend_rejection_reports++<8){char line[192];std::snprintf(line,sizeof(line),
                    "[C3X renderer] stage=native-blend-admission reason=%s background_owned=1\n",reason?reason:"unknown");OutputDebugStringA(line);}
            }
            if(destination&&destination->owned&&draw_blend(*destination,source,source_rect,target_rect,color))return 1;
            if(destination&&destination->owned&&blend_rejection_reports++<8){
                bool known=native_sprite(source),alpha=inputs&&native_sprite(inputs->alpha);auto bg=inputs?inputs->background:nullptr;
                char line[512];std::snprintf(line,sizeof(line),
                    "[C3X renderer] stage=native-blend-fallback mode=%u source=%d,%d,%d,%d,%d,%d alpha=%d,%d,%d,%d,%d,%d background=%d,%d,%d destination=%u,%u,%d\n",
                    color,known?field(source,0x20):-1,known?field(source,0x30):0,known?field(source,0x34):0,known?field(source,0x2c):0,known?field(source,0x18):0,known&&field(source,0x14)!=0,
                    alpha?field(inputs->alpha,0x20):-1,alpha?field(inputs->alpha,0x30):0,alpha?field(inputs->alpha,0x34):0,alpha?field(inputs->alpha,0x2c):0,alpha?field(inputs->alpha,0x18):0,alpha&&field(inputs->alpha,0x14)!=0,
                    bg?field(bg,0x38):0,bg?field(bg,0x3c):0,bg?field(bg,0x40):0,destination->width,destination->height,field(object,0x40));OutputDebugStringA(line);
            }
            if(destination)cpu_ownership(*destination,op);
            if(source_rect){auto background=find(static_cast<c3x_renderer_native_sprite_blend const*>(source_rect)->background);if(background&&background!=destination)cpu_ownership(*background,op);}
            ++counters.fallbacks;return 0;
        }
        if(op==C3X_NATIVE_SPRITE_STYLE){
            auto input=static_cast<c3x_renderer_native_sprite_style const*>(source_rect);
            // Slots 23/29/31/37 also query source bits before image pixels.
            if(input&&input->mode>=1&&input->mode<=4&&native_sprite(source)&&!field(source,0x14))return 0;
            if(destination&&destination->owned&&input&&input->mode>=1&&input->mode<=4&&
               (input->mode!=3||input->table)&&draw_sprite(*destination,source,input->palette,target_rect,input->table,nullptr,nullptr,input->mode,input->color,input->mode==4?input->opacity:1.f))return 1;
            if(destination)cpu_ownership(*destination,op);++counters.fallbacks;return 0;
        }
        if(op==C3X_NATIVE_SPRITE){
            // JGL 0x8180 calls the pure bits getter (0x98a0) first. A null
            // source returns 7 without touching the destination. Preserve that
            // native result via pass-through, with no GPU ownership barrier.
            if(native_sprite(source)&&!field(source,0x14))return 0;
            if(destination&&destination->owned&&draw_sprite(*destination,source,source_rect,target_rect))return 1;
            if(destination&&destination->owned&&destination->width>=640&&destination->height>=480&&sprite_rejection_reports++<8){
                auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
                bool known=module&&source&&*static_cast<void***>(source)==reinterpret_cast<void**>(module+0x68440);
                char line[320];std::snprintf(line,sizeof(line),
                    "[C3X renderer] stage=native-sprite-fallback known=%u bits=%d size=%dx%d stride=%d flags=%d pixels=%u scale=%d,%d,%d\n",
                    unsigned(known),known?field(source,0x20):0,known?field(source,0x30):0,known?field(source,0x34):0,
                    known?field(source,0x2c):0,known?field(source,0x18):0,unsigned(known&&field(source,0x14)!=0),
                    module?*reinterpret_cast<int*>(module+0x6c0fc):0,module?*reinterpret_cast<int*>(module+0x6c100):0,module?*reinterpret_cast<int*>(module+0x6c104):0);
                OutputDebugStringA(line);
            }
            if(destination)cpu_ownership(*destination,op);++counters.fallbacks;return 0;
        }
        if(op!=C3X_NATIVE_COPY&&op!=C3X_NATIVE_FILL&&op!=C3X_NATIVE_IMAGE_DRAW&&op!=C3X_NATIVE_TINT&&op!=C3X_NATIVE_LINE)return 0;
        auto fallback=[&](){if(destination)cpu_ownership(*destination,op);auto s=find(source);if(s&&s!=destination)cpu_ownership(*s,op);++counters.fallbacks;return 0;};
        if(!destination||!destination->owned)return fallback();
        Image* input=nullptr;
        Command command={Kind::fill,destination->gpu,0,{},rect(static_cast<char*>(object)+0x44),0,0,color&0xffff};
        command.area=target_rect?rect(target_rect):Rect{0,0,int(destination->width),int(destination->height)};
        if(op==C3X_NATIVE_FILL||op==C3X_NATIVE_TINT||op==C3X_NATIVE_LINE){
            // Both native fill helpers resolve a nonnegative index through the
            // current palette. Negative values are already packed native words.
            if(!(color&0x80000000u)){
                auto palette=*reinterpret_cast<void**>(static_cast<char*>(object)+0x7c);
                if(!palette){auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
                    if(!module)return fallback();auto owner=*reinterpret_cast<void**>(module+0x70f48);
                    if(owner)palette=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
                if(palette){auto table=*static_cast<void***>(palette);
                    auto colors=reinterpret_cast<unsigned short const*(__thiscall*)(void*)>(table[destination->format==Format::rgb565?7:6])(palette);
                    if(!colors)return fallback();command.color=colors[color&255];}
            }
            if(op==C3X_NATIVE_TINT){
                if(!source_rect)return fallback();
                int weight=std::clamp(*static_cast<int const*>(source_rect),0,100)*256/100;
                command.kind=Kind::native_blend;command.source=command.background=destination->gpu;
                command.source_width=int(command.color);command.source_height=weight;command.color=2;
                command.detail=command.background_detail=destination->detail;
            }else if(op==C3X_NATIVE_LINE){
                if(!source_rect)return fallback();auto endpoints=rect(source_rect);
                for(int n:{endpoints.left,endpoints.top,endpoints.right,endpoints.bottom})if(n<-32768||n>32767)return fallback();
                if(endpoints.left==endpoints.right&&endpoints.top==endpoints.bottom){++counters.translated;return 1;}
                if(endpoints.left!=endpoints.right&&endpoints.top!=endpoints.bottom)
                    return draw_diagonal(*destination,endpoints,command.color)?1:fallback();
                command.area={std::min(endpoints.left,endpoints.right),std::min(endpoints.top,endpoints.bottom),
                    std::max(endpoints.left,endpoints.right)+1,std::max(endpoints.top,endpoints.bottom)+1};
            }
        }else{
            if(!target_rect)return fallback();
            auto s=find(source);if(!s)s=create(source,false);
            if(!s||s->format!=destination->format)return fallback();
            if(op==C3X_NATIVE_IMAGE_DRAW){
                // JGL's sprite traversal is not overlap-safe. Self-draw retains
                // its ordered native program; StretchBlt overlap is separate.
                if(s==destination)return fallback();
                unsigned key=unsigned(field(source,0x4d0));
                auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));if(!module)return fallback();
                if(!(key&0x80000000u)){
                    auto palette=*reinterpret_cast<void**>(static_cast<char*>(object)+0x7c);
                    if(!palette){auto owner=*reinterpret_cast<void**>(module+0x70f48);if(owner)palette=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
                    if(palette){auto table=*static_cast<void***>(palette);
                        auto colors=reinterpret_cast<unsigned short const*(__thiscall*)(void*)>(table[destination->format==Format::rgb565?7:6])(palette);
                        if(!colors)return fallback();key=colors[key&255];}
                    // Image::draw uses a temporary sprite descriptor; the
                    // palette-key mutation must not change the source Image.
                }
                int scale=*reinterpret_cast<int*>(module+0x6c104),sx=*reinterpret_cast<int*>(module+0x6c0fc),sy=*reinterpret_cast<int*>(module+0x6c100);
                if(scale<=0||sx<=0||sy<=0)return fallback();
                if(sx!=scale||sy!=scale){++counters.translated;return 1;} // Native 16-bit scaled no-op.
                auto right=std::int64_t(command.area.left)+s->width,bottom=std::int64_t(command.area.top)+s->height;
                if(right>INT_MAX||bottom>INT_MAX)return fallback();
                command.area.right=int(right);command.area.bottom=int(bottom);
                if(std::max(command.area.left,command.clip.left)>=std::min(command.area.right,command.clip.right)||
                   std::max(command.area.top,command.clip.top)>=std::min(command.area.bottom,command.clip.bottom))return fallback();
                command.kind=Kind::color_key;command.color=key&65535;
            }else{
                if(!source_rect)return fallback();auto r=rect(source_rect);
                if(r.left>=r.right||r.top>=r.bottom)return fallback();
                command.kind=Kind::copy;command.source_x=r.left;command.source_y=r.top;
                if(std::int64_t(r.right)-r.left!=std::int64_t(command.area.right)-command.area.left||
                   std::int64_t(r.bottom)-r.top!=std::int64_t(command.area.bottom)-command.area.top){
                    auto dc=*reinterpret_cast<HDC*>(static_cast<char*>(object)+0x4bc);
                    if(!dc||GetStretchBltMode(dc)!=BLACKONWHITE||GetMapMode(dc)!=MM_TEXT||GetGraphicsMode(dc)!=GM_COMPATIBLE||
                       r.left<0||r.top<0||r.right>int(s->width)||r.bottom>int(s->height)||
                       command.area.right<=command.area.left||command.area.bottom<=command.area.top||
                       std::int64_t(command.area.right)-command.area.left>65535||std::int64_t(command.area.bottom)-command.area.top>65535)return fallback();
                    command.kind=Kind::native_image;command.color=65536;
                    command.source_width=r.right-r.left;command.source_height=r.bottom-r.top;
                }
            }
            if(command.kind==Kind::color_key&&s->detail){
                command.kind=Kind::native_image;command.source_width=int(s->width);command.source_height=int(s->height);
            }
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
        if(input&&command.kind!=Kind::native_image&&selected.left<selected.right&&selected.top<selected.bottom){
            auto x=std::int64_t(command.source_x)+selected.left-command.area.left,y=std::int64_t(command.source_y)+selected.top-command.area.top;
            if(x<0||y<0||x+selected.right-selected.left>input->width||y+selected.bottom-selected.top>input->height)return fallback();
        }
        Command commands[2]={command,command};unsigned count=1;
        if(input&&input->detail&&!full_color(*destination))return fallback();
        if(command.kind==Kind::native_image){
            commands[0].detail=destination->detail;commands[0].background_detail=input->detail;
        }else if(command.kind!=Kind::native_blend&&destination->detail){
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
        if(!admit(p))return false;auto d=find(p);
        Command c={Kind::quantize,d->gpu,map,area,rect(static_cast<char*>(p)+0x44),source_x,source_y,
            (unsigned(phase_x)&7u)|((unsigned(phase_y)&7u)<<3)};
        if(!full_color(*d))return false;
        Command commands[2]={c,c};commands[1].kind=Kind::copy;commands[1].destination=d->detail;commands[1].color=0;
        if(!gpu.submit(commands,2))return false;d->dirty=true;return true;
    }
    bool draw_unit(c3x_renderer_gpu_unit_fn draw,std::int64_t ticket,c3x_renderer_unit_v1 const& unit,void* target,void* background,int* bounds,unsigned flags){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("native unit adapter thread changed");
        if(!draw||!bounds||!admit(target))return false;auto d=find(target);
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
    template<class Draw> bool draw_tactical(Draw draw,std::int64_t ticket,void* target){
        if(GetCurrentThreadId()!=thread)throw std::runtime_error("tactical adapter thread changed");
        if(!admit(target))return false;auto d=find(target);
        if(!full_color(*d))return false;
        auto clip=rect(static_cast<char*>(target)+0x44);
        c3x_renderer_gpu_unit_v1 request={sizeof(request),ticket,std::int64_t(d->gpu),std::int64_t(d->gpu),
            std::int64_t(d->detail),std::int64_t(d->detail),{clip.left,clip.top,clip.right,clip.bottom},0};
        gpu.flush();int result=draw(request);
        if(result!=C3X_RENDERER_RESULT_OK)throw std::runtime_error("tactical composition failed");
        d->dirty=true;++counters.translated;return true;
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
