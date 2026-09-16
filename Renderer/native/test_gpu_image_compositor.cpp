#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include "gpu_image_compositor.h"
#include "native_observation.h"
using namespace c3x_gpu_images;
void require(bool value,char const* message){if(!value)throw std::runtime_error(message);}
struct Native {void** table;};
using Init=int(__thiscall*)(Native*,int,int,int,int);
using Fill=int(__thiscall*)(Native*,RECT*,int);
using Clip=int(__thiscall*)(Native*,RECT*);
using Copy=int(__thiscall*)(Native*,Native*,RECT*,RECT*);
using Get=std::uint16_t*(__thiscall*)(Native*,int,int);
using Release=void(__thiscall*)(Native*,int);
std::vector<std::uint32_t> read_native(Native* image,unsigned w,unsigned h){
    auto data=reinterpret_cast<Get>(image->table[7])(image,0,0);require(data!=nullptr,"native bits");
    auto stride=*reinterpret_cast<int*>(reinterpret_cast<char*>(image)+0x40);
    std::vector<std::uint32_t> result(w*h);for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x)result[y*w+x]=data[y*stride+x];
    reinterpret_cast<Release>(image->table[9])(image,1);return result;
}
std::vector<std::uint32_t> read_gpu(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
    D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> stage;checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
    D3D11_MAPPED_SUBRESOURCE m={};checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));std::vector<std::uint32_t> values(d.Width*d.Height);
    for(unsigned y=0;y<d.Height;++y)std::memcpy(values.data()+y*d.Width,static_cast<char*>(m.pData)+y*m.RowPitch,d.Width*4);
    context->Unmap(stage.Get(),0);return values;
}
int main(int argc,char** argv){
    if(argc!=2)return 2;
    try {
        HMODULE jgl=LoadLibraryA(argv[1]);require(c3x_native_observation::verified_module(jgl),"audited JGL hash");
        auto graph=reinterpret_cast<void*(__cdecl*)()>(GetProcAddress(jgl,"get_graphsy_object_ptr"))();auto table=*reinterpret_cast<void***>(graph);
        auto create=reinterpret_cast<Native*(__thiscall*)(void*,void*,int)>(table[31]);
        Native* native[3]={create(graph,nullptr,1),create(graph,nullptr,1),create(graph,nullptr,1)};
        ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> context;
        checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context));
        {
            // A retained map plus packed/full-color native destination and
            // underlay fit the existing budget at the live map's dimensions.
            // One small unit must not require two additional full-size images.
            constexpr unsigned w=2240,h=1192;Rect full_area={0,0,w,h},body_area={1000,476,1064,524};
            Compositor bounded(device.Get(),context.Get());
            auto map=bounded.create(w,h,Format::bgra32),d=bounded.create(w,h,Format::rgb555),b=bounded.create(w,h,Format::rgb555);
            auto detail=bounded.create(w,h,Format::bgra32),bd=bounded.create(w,h,Format::bgra32),body=bounded.create(64,48,Format::bgra32);
            require(map&&d&&b&&detail&&bd&&body,"live-size map/native pair admission");
            Command seed[]={{Kind::fill,d,0,full_area,full_area,0,0,0x3e0},{Kind::fill,b,0,full_area,full_area,0,0,0x3e0},
                {Kind::fill,detail,0,full_area,full_area,0,0,0xff00ff00u},{Kind::fill,bd,0,full_area,full_area,0,0,0xff00ff00u},
                {Kind::fill,body,0,{0,0,64,48},{0,0,64,48},0,0,0xffff0000u}};
            require(bounded.submit(seed,5),"live-size composition seed");auto bytes=bounded.stats().resident_bytes;
            Command unit={Kind::unit_over,d,body,body_area,full_area,0,0,0,b,detail,bd};
            require(bounded.submit(&unit,1),"live-size unit uses selected scratch within unchanged budget");
            require(bounded.stats().resident_bytes-bytes==2u*64u*48u*4u,"unit scratch tracks the selected rectangle");
            auto words=read_gpu(device.Get(),context.Get(),bounded.texture(d));
            auto colors=read_gpu(device.Get(),context.Get(),bounded.texture(detail));
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){bool hit=x>=1000&&x<1064&&y>=476&&y<524;
                require(words[y*w+x]==(hit?0x7c00u:0x3e0u)&&colors[y*w+x]==(hit?0xffff0000u:0xff00ff00u),"live-size unit exact selected pixels");}
            std::puts("PASS live-size unit: 2240x1192 map and native color pairs; 24576 scratch bytes, unchanged 64 MiB budget, exact pixels");
        }
        Compositor gpu(device.Get(),context.Get());Id images[3]={};constexpr unsigned width=64,height=48;
        RECT full={0,0,width,height};Rect area={0,0,width,height};
        for(unsigned n=0;n<3;++n){require(reinterpret_cast<Init>(native[n]->table[1])(native[n],width,height,16,1)==0,"native init");
            require(reinterpret_cast<Fill>(native[n]->table[17])(native[n],&full,int(0x80000000u))==0,"native clear");
            images[n]=gpu.create(width,height,Format::rgb555);require(images[n]!=0,"GPU image");}
        auto compare=[&](unsigned n,char const* label="copy/fill"){GdiFlush();auto a=read_native(native[n],width,height),b=read_gpu(device.Get(),context.Get(),gpu.texture(images[n]));
            for(unsigned i=0;i<a.size();++i)if(a[i]!=b[i]){std::fprintf(stderr,"Mismatch %s pixel=%u native=%04x gpu=%04x\n",label,i,a[i],b[i]);throw std::runtime_error("exact native/GPU packed pixels");}};
        // Seed CPU UI once. Its local revision suppresses subsequent uploads.
        RECT stripe={4,5,38,26};require(reinterpret_cast<Fill>(native[1]->table[17])(native[1],&stripe,int(0x80003e0u))==0,"native source fill");
        auto source=read_native(native[1],width,height);require(gpu.upload(images[1],1,source.data(),source.size()),"first source upload");
        require(gpu.upload(images[1],1,source.data(),source.size())&&gpu.stats().uploads==1,"unchanged source upload skipped");
        for(unsigned phase=0;phase<6;++phase){
            RECT clip={7,8,59,44};require(reinterpret_cast<Clip>(native[0]->table[13])(native[0],&clip)==0,"native clip");
            RECT fill={-4,3,44,39};unsigned color=0x1800+phase;
            require(reinterpret_cast<Fill>(native[0]->table[17])(native[0],&fill,int(0x80000000u|color))==0,"native clipped fill");
            Command command={Kind::fill,images[0],0,{-4,3,44,39},{7,8,59,44},0,0,color};require(gpu.submit(&command,1),"GPU clipped fill");compare(0);
            RECT source_rect={0,0,48,32},destination={-3,2,45,34};
            require(reinterpret_cast<Copy>(native[1]->table[16])(native[1],native[0],&source_rect,&destination)==0,"native copy");
            command={Kind::copy,images[0],images[1],{-3,2,45,34},{7,8,59,44}};require(gpu.submit(&command,1),"GPU clipped source copy");compare(0);
            require(reinterpret_cast<Clip>(native[0]->table[13])(native[0],&full)==0,"native unclip");
            RECT popup={10,12,40,30};
            require(reinterpret_cast<Copy>(native[0]->table[16])(native[0],native[2],&popup,&popup)==0,"native background save");
            require(reinterpret_cast<Fill>(native[0]->table[17])(native[0],&popup,int(0x80007fffu))==0,"native popup");
            require(reinterpret_cast<Copy>(native[2]->table[16])(native[2],native[0],&popup,&popup)==0,"native background restore");
            Command transaction[]={{Kind::copy,images[2],images[0],{10,12,40,30},area,10,12},
                {Kind::fill,images[0],0,{10,12,40,30},area,0,0,0x7fff},
                {Kind::copy,images[0],images[2],{10,12,40,30},area,10,12}};
            require(gpu.submit(transaction,3),"GPU save/popup/restore transaction");compare(0);compare(2);
            RECT move_source={3,5,45,35},move_dest={9,11,51,41};
            require(reinterpret_cast<Copy>(native[0]->table[16])(native[0],native[0],&move_source,&move_dest)==0,"native overlapping copy");
            command={Kind::copy,images[0],images[0],{9,11,51,41},area,3,5};require(gpu.submit(&command,1),"GPU overlapping copy");compare(0);
        }
        // Native 16-bit keyed image drawing, with a locally changed CPU UI source.
        require(reinterpret_cast<Fill>(native[1]->table[17])(native[1],&full,int(0x80007c1fu))==0,"keyed source background");
        require(reinterpret_cast<Fill>(native[1]->table[17])(native[1],&stripe,int(0x800003e0u))==0,"keyed source content");
        source=read_native(native[1],width,height);require(gpu.upload(images[1],2,source.data(),source.size()),"local source revision upload");
        using Draw=int(__thiscall*)(Native*,Native*,int,int);
        require(reinterpret_cast<Draw>(native[1]->table[33])(native[1],native[0],0,0)==0,"native keyed image draw");
        Command keyed={Kind::color_key,images[0],images[1],area,area,0,0,0x7c1f};require(gpu.submit(&keyed,1),"GPU native color key");compare(0,"keyed image");
        using Acquire=HDC(__thiscall*)(Native*);using ReleaseDc=void(__thiscall*)(Native*,int);
        auto dc=reinterpret_cast<Acquire>(native[0]->table[10])(native[0]);require(dc!=nullptr,"native DC for invert");
        require(PatBlt(dc,10,12,30,18,DSTINVERT)!=FALSE,"native destination invert");GdiFlush();
        reinterpret_cast<ReleaseDc>(native[0]->table[11])(native[0],1);
        Command inverse={Kind::invert,images[0],0,{10,12,40,30},area,0,0,0xffff};require(gpu.submit(&inverse,1),"GPU native 16-bit invert");compare(0,"invert");
        auto before=read_gpu(device.Get(),context.Get(),gpu.texture(images[0]));
        Command bad[]={{Kind::fill,images[0],0,area,area,0,0,5},{Kind::copy,images[0],99999,area,area}};
        require(!gpu.submit(bad,2)&&before==read_gpu(device.Get(),context.Get(),gpu.texture(images[0])),"invalid transaction cannot partially execute");
        auto old=images[1];require(gpu.destroy(old),"destroy source");images[1]=gpu.create(width,height,Format::rgb555);require(images[1]!=old&&!gpu.texture(old),"retired handle cannot alias replacement");
        require(gpu.upload(images[1],5,source.data(),source.size()),"replacement upload");
        Command dirty={Kind::fill,images[1],0,area,area,0,0,1};require(gpu.submit(&dirty,1),"GPU mutation");
        require(!gpu.upload(images[1],5,source.data(),source.size())&&!gpu.upload(images[1],4,source.data(),source.size()),"stale CPU revision rejected after GPU mutation");
        require(gpu.upload(images[1],6,source.data(),source.size()),"new CPU revision admitted");
        auto rgba_source=gpu.create(2,1,Format::bgra32),rgba_dest=gpu.create(2,1,Format::bgra32);
        std::uint32_t rgba[2]={0x00ff00ff,0x00010203};require(gpu.upload(rgba_source,1,rgba,2),"RGB source upload");
        Command rgb_key={Kind::color_key,rgba_dest,rgba_source,{0,0,2,1},{0,0,2,1},0,0,0xffff00ff};require(gpu.submit(&rgb_key,1),"RGB key ignores alpha");
        auto keyed_pixels=read_gpu(device.Get(),context.Get(),gpu.texture(rgba_dest));require(keyed_pixels[0]==0&&keyed_pixels[1]==0x00010203,"RGB color-key semantics");
        auto other_encoding=gpu.create(width,height,Format::rgb565);
        Command conversion={Kind::copy,images[0],other_encoding,area,area};require(!gpu.submit(&conversion,1),"different native pixel encodings cannot silently copy");
        require(gpu.destroy(other_encoding),"release encoding witness");
        // Validate both native encodings against real GDI DIBs. The full-color
        // output uses GDI's independent 16 -> 32 expansion as its oracle.
        for(auto format:{Format::rgb555,Format::rgb565}){
            struct BitmapInfo {BITMAPINFOHEADER header;DWORD masks[3];} info={};
            info.header.biSize=sizeof(BITMAPINFOHEADER);info.header.biWidth=64;info.header.biHeight=-48;info.header.biPlanes=1;info.header.biBitCount=16;info.header.biCompression=BI_BITFIELDS;
            info.masks[0]=format==Format::rgb565?0xf800:0x7c00;info.masks[1]=format==Format::rgb565?0x7e0:0x3e0;info.masks[2]=31;
            HDC dib_dc[3]={CreateCompatibleDC(nullptr),CreateCompatibleDC(nullptr),CreateCompatibleDC(nullptr)};HBITMAP bitmap[3]={};HGDIOBJ previous[3]={};void* bits[3]={};
            for(int n=0;n<3;++n){if(n==2){info.header.biBitCount=32;info.header.biCompression=BI_RGB;}
                bitmap[n]=CreateDIBSection(dib_dc[n],reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS,&bits[n],nullptr,0);
                require(dib_dc[n]&&bitmap[n]&&bits[n],"native encoding oracle DIB");previous[n]=SelectObject(dib_dc[n],bitmap[n]);}
            auto input=gpu.create(64,48,format),output=gpu.create(64,48,format),detail=gpu.create(64,48,Format::bgra32);
            std::vector<unsigned> words(64*48);for(unsigned n=0;n<words.size();++n)words[n]=(n*193+79)&(format==Format::rgb565?65535:32767);
            require(gpu.upload(input,1,words.data(),words.size()),"native transfer format source");
            for(unsigned n=0;n<words.size();++n)static_cast<unsigned short*>(bits[0])[n]=static_cast<unsigned short>(words[n]);
            for(auto extent:std::array<std::array<int,2>,3>{{{{43,31}},{{91,73}},{{23,79}}}}){
                require(StretchBlt(dib_dc[1],-3,-2,extent[0],extent[1],dib_dc[0],5,3,57,43,SRCCOPY)!=FALSE,"GDI packed transfer oracle");GdiFlush();
                require(BitBlt(dib_dc[2],0,0,64,48,dib_dc[1],0,0,SRCCOPY)!=FALSE,"GDI independent native color expansion");GdiFlush();
                Command transfer={Kind::native_image,output,input,{-3,-2,extent[0]-3,extent[1]-2},area,5,3,65536,0,detail,0,57,43};
                require(gpu.submit(&transfer,1),"paired native transfer");
                auto actual=read_gpu(device.Get(),context.Get(),gpu.texture(output)),rgb=read_gpu(device.Get(),context.Get(),gpu.texture(detail));
                for(int y=0;y<48&&y<extent[1]-2;++y)for(int x=0;x<64&&x<extent[0]-3;++x){auto n=y*64+x;
                    require(actual[n]==static_cast<unsigned short*>(bits[1])[n],"555/565 scaled native words exact GDI");
                    require((rgb[n]&0xffffff)==(static_cast<unsigned*>(bits[2])[n]&0xffffff),"555/565 scaled expansion exact GDI");}
                auto invalid=transfer;invalid.source_width=65;
                Command transaction[2]={{Kind::fill,output,0,area,area,0,0,5},invalid};
                require(!gpu.submit(transaction,2)&&actual==read_gpu(device.Get(),context.Get(),gpu.texture(output)),"invalid paired transfer rejects the entire transaction");
            }
            gpu.destroy(input);gpu.destroy(output);gpu.destroy(detail);
            for(int n=0;n<3;++n){SelectObject(dib_dc[n],previous[n]);DeleteObject(bitmap[n]);DeleteDC(dib_dc[n]);}
        }
        Compositor bounded(device.Get(),context.Get(),width*height*4);auto tight=bounded.create(width,height,Format::rgb555);require(tight!=0&&bounded.create(1,1,Format::rgb555)==0,"resident budget enforced");
        Command overlap={Kind::copy,tight,tight,area,area};require(!bounded.submit(&overlap,1)&&bounded.stats().commands==0,"scratch budget rejection precedes execution");
        for(auto image:native)reinterpret_cast<void(__thiscall*)(Native*,unsigned)>(image->table[0])(image,1);
        reinterpret_cast<void(__thiscall*)(void*,unsigned)>(table[0])(graph,1);FreeLibrary(jgl);
        auto counts=gpu.stats();std::printf("PASS GPU native operations: 6 phases exact JGL pixels; commands=%llu snapshots=%llu uploads=%llu bytes=%llu; no execution readback (oracle readback separate); lifetime/revision/budget/rejection pass\n",
            counts.commands,counts.snapshots,counts.uploads,counts.resident_bytes);return 0;
    }catch(std::exception const& e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
