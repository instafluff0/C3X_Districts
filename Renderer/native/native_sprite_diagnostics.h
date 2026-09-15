#pragma once
// Temporary, bounded game-evaluation oracle. No timing heuristics or pixel
// substitutions: compare a sampled GPU draw with JGL on an isolated CPU image.
namespace c3x_native_images {
using namespace c3x_gpu_images;
class SpriteDiagnostics {
    std::array<std::uint64_t,96> seen={};unsigned used=0;
    // Separate keyed/UI and ordinary budgets so map sprites cannot consume all
    // samples before the user opens an Advisor. Fingerprints include content,
    // since native allocators reuse sprite and palette addresses.
    std::array<std::uint64_t,192> cpu_seen={};unsigned cpu_used[2]={};
    std::array<std::uint64_t,256> completed_seen={};unsigned completed_used=0;
    static unsigned hash(std::vector<unsigned> const& words){unsigned h=2166136261u;for(auto v:words)h=(h^v)*16777619u;return h;}
public:
    unsigned samples=0,mismatches=0;
    void (*write)(char const*)=[](char const* line){OutputDebugStringA(line);};
    unsigned cpu_samples()const{return cpu_used[0]+cpu_used[1];}
    struct Sample {std::vector<unsigned> before,expected,detail;unsigned width=0,height=0;int style=0;};
    void cpu_operation(int operation,void* source,void const* from){
        if(operation==C3X_NATIVE_SPRITE)cpu_source(source,0,from);
        else if(operation==C3X_NATIVE_SPRITE_STYLE&&from){
            auto const& style=*static_cast<c3x_renderer_native_sprite_style const*>(from);
            cpu_source(source,style.mode,style.palette);
        }
    }
    void cpu_source(void* source,int style,void const* selected){
        unsigned bucket=style==1?1:0,base=bucket*96;
        if(!source||cpu_used[bucket]==96)return;
        auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
        if(!module||*static_cast<void***>(source)!=reinterpret_cast<void**>(module+0x68440))return;
        auto f=static_cast<int*>(source);
        if(f[8]!=8||f[12]<=0||f[13]<=0||f[12]>768||f[13]>256)return;
        auto palette=selected?selected:*reinterpret_cast<void**>(static_cast<char*>(source)+0x10);
        if(!palette){auto owner=*reinterpret_cast<void**>(module+0x70f48);if(owner)palette=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
        unsigned palette_hash=2166136261u,gray=0;unsigned short const* colors=nullptr;
        if(palette){auto table=*static_cast<void* const* const*>(palette);
            colors=reinterpret_cast<unsigned short const*(__thiscall*)(void const*)>(table[6])(palette);
            if(colors)for(unsigned i=0;i<256;++i){unsigned c=colors[i];palette_hash=(palette_hash^c)*16777619u;gray+=((c&31)==(c>>5&31)&&(c&31)==(c>>10&31));}}
        unsigned source_hash=2166136261u;bool raw=!(f[6]&1)&&f[5]&&f[11]>=f[12];
        if(raw){auto pixels=reinterpret_cast<unsigned char const*>(std::uintptr_t(unsigned(f[5])));
            for(int y=0;y<f[13];++y)for(int x=0;x<f[12];++x)source_hash=(source_hash^pixels[y*f[11]+x])*16777619u;}
        std::uint64_t key=(std::uint64_t(palette_hash)<<32)|source_hash;
        key^=std::uint64_t(unsigned(f[12]+f[13]*769+style*197633));
        auto begin=cpu_seen.begin()+base;
        if(std::find(begin,begin+cpu_used[bucket],key)!=begin+cpu_used[bucket])return;
        cpu_seen[base+cpu_used[bucket]++]=key;char line[512];
        std::snprintf(line,sizeof(line),"[C3X renderer] stage=ui-sprite-source route=cpu sample=%u style=%d width=%d height=%d bits=%d packing=%d palette=%s palette_hash=%08x gray_entries=%u source_hash=%08x raw_hash_valid=%u\n",
            base+cpu_used[bucket],style,f[12],f[13],f[8],f[6],selected?"explicit":f[4]?"source":"default",palette_hash,gray,source_hash,unsigned(raw));write(line);
        // Full palette words allow offline comparison with the supplied PCX
        // assets. No game text, filenames, addresses or screen pixels are logged.
        if(colors)for(unsigned start=0;start<256;start+=64){
            int n=std::snprintf(line,sizeof(line),"[C3X renderer] stage=ui-sprite-palette sample=%u start=%u words=",base+cpu_used[bucket],start);
            for(unsigned i=start;i<start+64;++i)n+=std::snprintf(line+n,sizeof(line)-n,"%04x",unsigned(colors[i]));
            line[n++]='\n';line[n]=0;write(line);
        }
    }
    unsigned completed_samples()const{return completed_used;}
    unsigned completed_mismatches=0;
    void cpu_result(void* destination,void* source,void const* selected,void const* target,unsigned result){
        if(!destination||!source||!target||completed_used==completed_seen.size())return;
        auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
        if(!module||*static_cast<void***>(source)!=reinterpret_cast<void**>(module+0x68440)||
           *static_cast<void***>(destination)!=reinterpret_cast<void**>(module+0x68238))return;
        auto f=static_cast<int*>(source),d=static_cast<int*>(destination);
        if(f[8]!=8||(f[6]&1)||!f[5]||f[12]<=0||f[13]<=0||f[12]>768||f[13]>256||f[11]<f[12])return;
        auto palette=selected?selected:*reinterpret_cast<void**>(static_cast<char*>(source)+0x10);
        if(!palette){auto owner=*reinterpret_cast<void**>(module+0x70f48);if(owner)palette=*reinterpret_cast<void**>(static_cast<char*>(owner)+4);}
        unsigned short const* colors=nullptr;unsigned palette_hash=2166136261u;
        if(palette&&d[9]==16&&(d[10]==0||d[10]==1)){
            auto table=*static_cast<void* const* const*>(palette);
            colors=reinterpret_cast<unsigned short const*(__thiscall*)(void const*)>(table[d[10]?7:6])(palette);
            if(colors)for(unsigned i=0;i<256;++i)palette_hash=(palette_hash^colors[i])*16777619u;
        }
        auto pixels=reinterpret_cast<unsigned char const*>(std::uintptr_t(unsigned(f[5])));unsigned source_hash=2166136261u;
        for(int y=0;y<f[13];++y)for(int x=0;x<f[12];++x)source_hash=(source_hash^pixels[y*f[11]+x])*16777619u;
        std::uint64_t key=(std::uint64_t(palette_hash)<<32)|source_hash;
        key^=unsigned(f[12]+f[13]*769+d[9]*197633+d[10]*401177);
        auto scales=reinterpret_cast<int*>(module+0x6c0fc);auto at=static_cast<RECT const*>(target);
        auto clip=reinterpret_cast<RECT const*>(static_cast<char*>(destination)+0x44);
        auto bits=*reinterpret_cast<unsigned char**>(static_cast<char*>(destination)+0x4c0);
        bool check=result==0&&bits&&d[16]>=d[14]&&(d[9]==8||colors)&&scales[0]>0&&scales[0]==scales[1]&&scales[0]==scales[2];
        unsigned checked=0,bad=0,first_expected=0,first_actual=0;int first_x=0,first_y=0;
        GdiFlush();
        if(check)for(int y=0;y<f[13];++y)for(int x=0;x<f[12];++x){
            int dx=at->left+x,dy=at->top+y;auto index=pixels[y*f[11]+x];
            if(index>=254||dx<0||dy<0||dx>=d[14]||dy>=d[15]||dx<clip->left||dx>=clip->right||dy<clip->top||dy>=clip->bottom)continue;
            unsigned expected=d[9]==8?index:colors[index];
            unsigned actual=d[9]==8?bits[dy*d[16]+dx]:reinterpret_cast<unsigned short*>(bits)[dy*d[16]+dx];
            ++checked;if(expected!=actual){if(!bad){first_x=dx;first_y=dy;first_expected=expected;first_actual=actual;}++bad;}
        }
        // A later bad draw of the same source must not be hidden by its first,
        // correct use. Clipping alone does not consume more successful samples.
        key^=std::uint64_t(bad)*0x9e3779b1u+first_actual+std::uint64_t(result)*0x100000001ull;
        if(std::find(completed_seen.begin(),completed_seen.begin()+completed_used,key)!=completed_seen.begin()+completed_used)return;
        completed_seen[completed_used++]=key;completed_mismatches+=bad!=0;
        unsigned dib_palette_hash=2166136261u,dib_colors=0;
        if(d[9]==8){RGBQUAD palette_colors[256];auto dc=*reinterpret_cast<HDC*>(static_cast<char*>(destination)+0x4bc);
            dib_colors=GetDIBColorTable(dc,0,256,palette_colors);
            for(unsigned i=0;i<dib_colors;++i){auto c=palette_colors[i];unsigned word=(c.rgbRed>>3)<<10|(c.rgbGreen>>3)<<5|(c.rgbBlue>>3);
                dib_palette_hash=(dib_palette_hash^word)*16777619u;}}
        char line[512];std::snprintf(line,sizeof(line),"[C3X renderer] stage=ui-native-draw sample=%u source_hash=%08x palette_hash=%08x width=%d height=%d destination_bits=%d format=%d scale=%d,%d,%d result=%u checked=%u mismatches=%u first=%d,%d expected=%04x actual=%04x dib_colors=%u dib_palette_hash=%08x\n",
            completed_used,source_hash,palette_hash,f[12],f[13],d[9],d[10],scales[0],scales[1],scales[2],result,checked,bad,first_x,first_y,first_expected,first_actual,dib_colors,dib_palette_hash);write(line);
    }
    template<class Backend> Sample begin(Backend& gpu,Id image,Id detail,void* destination,void* sprite,void const* palette,void const* target,int style,std::vector<unsigned> const& decoded){
        Sample sample;
        if(used==seen.size()||(style!=0&&style!=1))return sample;
        auto f=static_cast<int*>(sprite);
        if(f[12]>768||f[13]>256)return sample;
        std::uint64_t key=hash(decoded)|(std::uint64_t(unsigned(style+f[12]*7+f[13]*107))<<32);
        if(std::find(seen.begin(),seen.begin()+used,key)!=seen.begin()+used)return sample;
        seen[used++]=key;
        auto module=reinterpret_cast<char*>(GetModuleHandleA("jgl.dll"));
        // Native image init creates RGB555. Do not manufacture a different
        // descriptor for a 565 destination merely to admit a diagnostic.
        auto d=static_cast<int*>(destination);if(d[10]!=0)return sample;
        sample.width=unsigned(d[14]);sample.height=unsigned(d[15]);sample.style=style;
        sample.before.resize(std::size_t(sample.width)*sample.height);
        if(!gpu.readback(image,sample.before.data(),sample.before.size()))return {};
        if(detail)sample.detail.resize(sample.before.size());
        // The export constructs/replaces the global owner; diagnostics must borrow it.
        auto graph=*reinterpret_cast<void**>(module+0x70d30);
        if(!graph)return {};
        auto table=*static_cast<void***>(graph);
        auto clone=reinterpret_cast<void*(__thiscall*)(void*,void*,int)>(table[31])(graph,nullptr,1);
        if(!clone)return {};
        struct Cleanup {void* p;char* module;~Cleanup(){reinterpret_cast<void*(__thiscall*)(void*,unsigned)>(module+0x15d0)(p,1);}} cleanup={clone,module};
        if(reinterpret_cast<int(__thiscall*)(void*,int,int,int,int)>(module+0x1800)(clone,int(sample.width),int(sample.height),16,1)!=0)return {};
        auto bits=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(module+0x1b70)(clone);
        if(!bits)return {};int stride=static_cast<int*>(clone)[16];
        for(unsigned y=0;y<sample.height;++y)for(unsigned x=0;x<sample.width;++x)bits[y*stride+x]=static_cast<unsigned short>(sample.before[y*sample.width+x]);
        reinterpret_cast<void(__thiscall*)(void*,int)>(module+0x1b90)(clone,1);
        reinterpret_cast<void(__thiscall*)(void*,void*)>(module+0x1ca0)(clone,*reinterpret_cast<void**>(static_cast<char*>(destination)+0x7c));
        reinterpret_cast<int(__thiscall*)(void*,void*)>(module+0x1a40)(clone,static_cast<char*>(destination)+0x44);
        auto anchor=static_cast<RECT const*>(target);int saved_key=f[10];
        int result=reinterpret_cast<int(__thiscall*)(void*,void*,int,int,void const*)>(module+(style?0x8050:0x8180))(sprite,clone,anchor->left,anchor->top,palette);
        f[10]=saved_key;GdiFlush();
        bits=reinterpret_cast<unsigned short*(__thiscall*)(void*)>(module+0x1b70)(clone);
        if(!bits)return {};sample.expected.resize(sample.before.size());
        for(unsigned y=0;y<sample.height;++y)for(unsigned x=0;x<sample.width;++x)sample.expected[y*sample.width+x]=bits[y*stride+x];
        reinterpret_cast<void(__thiscall*)(void*,int)>(module+0x1b90)(clone,1);
        unsigned opaque=0,gray=0;for(auto code:decoded)if(code&65536){++opaque;unsigned c=code&65535;gray+=((c&31)==(c>>5&31)&&(c&31)==(c>>10&31));}
        char line[384];std::snprintf(line,sizeof(line),"[C3X renderer] stage=ui-sprite-source route=gpu sample=%u style=%d width=%d height=%d bits=%d packing=%d palette=%s native_result=%d decoded=%08x native=%08x opaque=%u gray=%u\n",
            samples+1,style,f[12],f[13],f[8],f[6],palette?"explicit":f[4]?"source":"default",result,hash(decoded),hash(sample.expected),opaque,gray);OutputDebugStringA(line);
        return sample;
    }
    template<class Backend> void finish(Backend& gpu,Id image,Id detail,Sample& sample){
        if(sample.expected.empty())return;
        std::vector<unsigned> actual(sample.before.size());if(!gpu.readback(image,actual.data(),actual.size()))return;
        unsigned bad=0,first=0;for(unsigned i=0;i<actual.size();++i)if(actual[i]!=sample.expected[i]){if(!bad)first=i;++bad;}
        unsigned detail_bad=0,detail_checked=0;
        if(detail&&!sample.detail.empty()&&gpu.readback(detail,sample.detail.data(),sample.detail.size())){
            // Only changed native words prove an opaque write independently of
            // our decoder. Unchanged pixels can retain full-color map precision.
            for(unsigned i=0;i<actual.size();++i)if(sample.expected[i]!=sample.before[i]){
                unsigned c=sample.expected[i],b=c&31,g=c>>5&31,r=c>>10&31;
                unsigned rgb=0xff000000u|((b<<3)|(b>>2))|(((g<<3)|(g>>2))<<8)|(((r<<3)|(r>>2))<<16);
                ++detail_checked;detail_bad+=sample.detail[i]!=rgb;
            }
        }
        ++samples;mismatches+=bool(bad||detail_bad);char line[384];
        std::snprintf(line,sizeof(line),"[C3X renderer] stage=ui-sprite-oracle sample=%u native_mismatches=%u detail_mismatches=%u detail_checked=%u first=%u,%u expected=%04x actual=%04x gpu=%08x remaining=%u diagnostic_readbacks=%u\n",
            samples,bad,detail_bad,detail_checked,first%sample.width,first/sample.width,sample.expected[first],actual[first],hash(actual),unsigned(seen.size()-used),sample.detail.empty()?2u:3u);OutputDebugStringA(line);
    }
};
}
