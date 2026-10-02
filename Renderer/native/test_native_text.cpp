// Bounded GDI text precision diagnostic. Native font quality is never replaced.
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <fstream>
#include <sstream>
#include "native_text_raster.h"
#include "gpu_image_compositor.h"
struct Dib {
    HDC dc=CreateCompatibleDC(nullptr);HBITMAP bitmap=nullptr;HGDIOBJ old=nullptr;void* pixels=nullptr;
    Dib(int depth,bool green6,int width=320,int height=48){struct Info {BITMAPINFOHEADER h;DWORD masks[3];} info={};
        info.h.biSize=sizeof info.h;info.h.biWidth=width;info.h.biHeight=-height;info.h.biPlanes=1;info.h.biBitCount=WORD(depth);
        if(depth==16){info.h.biCompression=BI_BITFIELDS;info.masks[0]=green6?0xf800:0x7c00;info.masks[1]=green6?0x7e0:0x3e0;info.masks[2]=31;}
        bitmap=CreateDIBSection(dc,reinterpret_cast<BITMAPINFO*>(&info),DIB_RGB_COLORS,&pixels,nullptr,0);
        if(!bitmap||!pixels)throw std::runtime_error("DIB allocation");old=SelectObject(dc,bitmap);
    }
    ~Dib(){SelectObject(dc,old);DeleteObject(bitmap);DeleteDC(dc);}
};
std::map<std::string,std::string> refusal_capture(char const* path,bool candidate){
    std::ifstream input(path);if(!input)throw std::runtime_error("text refusal log unavailable");std::string line;
    while(std::getline(input,line)){
        auto start=line.find(candidate?"stage=native-text-candidate ":"stage=native-text-refused ");if(start==std::string::npos)continue;
        std::istringstream fields(line.substr(start));std::map<std::string,std::string> record;std::string field;
        while(fields>>field){auto equal=field.find('=');if(equal==std::string::npos||!record.emplace(field.substr(0,equal),field.substr(equal+1)).second)throw std::runtime_error("invalid text refusal fields");}
        if(record.at("capture")=="1")return record;
    }throw std::runtime_error("no complete owner-thread text refusal capture");
}
std::vector<unsigned char> refusal_hex(std::string const& text){
    if(text.size()%2||text.size()>2048)throw std::runtime_error("text refusal hex bound");std::vector<unsigned char> bytes;
    auto digit=[](char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("invalid refusal hex");};
    for(std::size_t n=0;n<text.size();n+=2)bytes.push_back(static_cast<unsigned char>(digit(text[n])*16+digit(text[n+1])));return bytes;
}
void qualify_gpu_text(HDC source,c3x_native_text::Raster const& raster,char const* text,unsigned count,RECT anchor,RECT clip);
void reproduce_refusal(char const* path,bool candidate=false,bool qualify=false){
    auto record=refusal_capture(path,candidate);auto require=[&](bool value,char const* message){if(!value)throw std::runtime_error(message);};
    require(record.at("version")=="1"&&record.at("dc")=="1"&&record.at("font_valid")=="1"&&record.at("text_valid")=="1"&&record.at("text_complete")=="1"&&record.at("origins_valid")=="1"&&record.at("transform_valid")=="1","incomplete text refusal owner facts");
    auto font_bytes=refusal_hex(record.at("font")),text=refusal_hex(record.at("text"));LOGFONTA font_state={};
    require(font_bytes.size()==sizeof(font_state)&&text.size()==std::stoul(record.at("count")),"text refusal byte extent");std::memcpy(&font_state,font_bytes.data(),sizeof(font_state));
    auto numbers=[&](char const* name){std::string value=record.at(name);std::replace(value.begin(),value.end(),',',' ');return std::istringstream(value);};
    int width=0,height=0;{auto in=numbers("destination");require(bool(in>>width>>height)&&width>0&&width<=2240&&height>0&&height<=1260,"refusal destination extent");}
    Dib source(32,false,width,height);auto font=CreateFontIndirectA(&font_state);require(font!=nullptr,"reproduction font unavailable");auto previous=SelectObject(source.dc,font);
    struct FontLease {HDC dc;HGDIOBJ previous;HFONT font;~FontLease(){SelectObject(dc,previous);DeleteObject(font);}} lease{source.dc,previous,font};
    SetMapMode(source.dc,std::stoi(record.at("mapping")));SetLayout(source.dc,DWORD(std::stoul(record.at("layout"))));SetTextCharacterExtra(source.dc,std::stoi(record.at("extra")));
    SetTextColor(source.dc,COLORREF(std::stoul(record.at("foreground"))));SetBkColor(source.dc,COLORREF(std::stoul(record.at("background"))));SetBkMode(source.dc,std::stoi(record.at("mode")));SetTextAlign(source.dc,UINT(std::stoul(record.at("align"))));
    int x=0,y=0;{auto in=numbers("viewport");require(bool(in>>x>>y),"refusal viewport");SetViewportOrgEx(source.dc,x,y,nullptr);}{auto in=numbers("window");require(bool(in>>x>>y),"refusal window");SetWindowOrgEx(source.dc,x,y,nullptr);}
    if(std::stoi(record.at("graphics"))==GM_ADVANCED){SetGraphicsMode(source.dc,GM_ADVANCED);XFORM transform={};auto in=numbers("transform");require(bool(in>>transform.eM11>>transform.eM12>>transform.eM21>>transform.eM22>>transform.eDx>>transform.eDy),"refusal transform");require(SetWorldTransform(source.dc,&transform)!=FALSE,"reproduction transform unavailable");}
    RECT clip={};{auto in=numbers("clip");require(bool(in>>clip.left>>clip.top>>clip.right>>clip.bottom),"refusal clip");}
    int clip_kind=std::stoi(record.at("clip_kind"));
    if(clip_kind==SIMPLEREGION){auto region=CreateRectRgn(clip.left,clip.top,clip.right,clip.bottom);require(region!=nullptr,"reproduction clip unavailable");SelectClipRgn(source.dc,region);DeleteObject(region);}
    c3x_native_text::State state;c3x_native_text::Diagnostic diagnostic;c3x_native_text::Raster raster;
    bool accepted=c3x_native_text::capture(source.dc,state,&diagnostic);
    if(accepted&&(state.align&(TA_UPDATECP|TA_RTLREADING)))accepted=c3x_native_text::refuse(&diagnostic,c3x_native_text::Refusal::alignment);
    if(accepted&&(clip_kind==ERROR||clip_kind==COMPLEXREGION))accepted=c3x_native_text::refuse(&diagnostic,c3x_native_text::Refusal::clip);
    require(clip_kind!=NULLREGION,"captured null clip is an accepted native no-op, not a compiler operation");
    if(accepted)accepted=c3x_native_text::compile(source.dc,state,reinterpret_cast<char const*>(text.data()),unsigned(text.size()),raster,&diagnostic);
    if(qualify){
        require(accepted,"copied native candidate still refused");
        RECT anchor={};{auto in=numbers("anchor");require(bool(in>>anchor.left>>anchor.top),"captured anchor missing");}
        qualify_gpu_text(source.dc,raster,reinterpret_cast<char const*>(text.data()),unsigned(text.size()),anchor,clip);return;
    }
    if(candidate){std::printf("PASS new candidate text classification count=%zu known=%s owned=%s dirty=%s accepted=%u reason=%s raster=%u,%u curves=%u original_failure_identity=unproved\n",text.size(),record.at("known").c_str(),record.at("owned").c_str(),record.at("dirty").c_str(),unsigned(accepted),c3x_native_text::refusal_name(diagnostic.reason),diagnostic.width,diagnostic.height,diagnostic.curves);return;}
    require(!accepted&&record.at("reason")==c3x_native_text::refusal_name(diagnostic.reason),"captured text refusal not reproduced");
    if(diagnostic.reason==c3x_native_text::Refusal::raster_bounds){unsigned captured_width=0,captured_height=0;auto in=numbers("raster");require(bool(in>>captured_width>>captured_height)&&diagnostic.width==captured_width&&diagnostic.height==captured_height,"captured raster dimensions differ");}
    std::printf("PASS actual owner-thread text refusal reproduction count=%zu reason=%s raster=%u,%u curves=%u\n",text.size(),c3x_native_text::refusal_name(diagnostic.reason),diagnostic.width,diagnostic.height,diagnostic.curves);
}
void text_extent_refusal_contract(){
    Dib source(32,false);LOGFONTA description={};description.lfHeight=-17;description.lfWeight=700;strcpy_s(description.lfFaceName,"Arial");auto font=CreateFontIndirectA(&description);if(!font)throw std::runtime_error("extent font");auto previous=SelectObject(source.dc,font);
    std::string text(1024,'W');c3x_native_text::State state;c3x_native_text::Diagnostic diagnostic;c3x_native_text::Raster raster;
    bool captured=c3x_native_text::capture(source.dc,state,&diagnostic),compiled=captured&&c3x_native_text::compile(source.dc,state,text.data(),unsigned(text.size()),raster,&diagnostic);
    SelectObject(source.dc,previous);DeleteObject(font);
    if(!captured||compiled||diagnostic.reason!=c3x_native_text::Refusal::raster_bounds||std::uint64_t(diagnostic.width)*diagnostic.height<=c3x_native_text::maximum_raster_pixels)throw std::runtime_error("oversized shaped text raster refusal contract");
    std::printf("PASS text extent refusal contract count=1024 raster=%u,%u pixels=%llu\n",diagnostic.width,diagnostic.height,static_cast<unsigned long long>(std::uint64_t(diagnostic.width)*diagnostic.height));
}
unsigned expand(unsigned c,bool g6){unsigned b=c&31,g=(c>>5)&(g6?63:31),r=(c>>(g6?11:10))&31;
    return ((b<<3)|(b>>2))|((g6?((g<<2)|(g>>4)):((g<<3)|(g>>2)))<<8)|(((r<<3)|(r>>2))<<16);}
unsigned pack(unsigned c,bool g6){return (c>>3&31)|((c>>(g6?10:11)&(g6?63:31))<<5)|((c>>19&31)<<(g6?11:10));}
std::vector<unsigned> read_gpu(ID3D11Device* device,ID3D11DeviceContext* context,ID3D11Texture2D* texture){
    D3D11_TEXTURE2D_DESC d={};texture->GetDesc(&d);d.Usage=D3D11_USAGE_STAGING;d.BindFlags=0;d.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    Microsoft::WRL::ComPtr<ID3D11Texture2D> stage;c3x_gpu_images::checked(device->CreateTexture2D(&d,nullptr,&stage));context->CopyResource(stage.Get(),texture);
    D3D11_MAPPED_SUBRESOURCE m={};c3x_gpu_images::checked(context->Map(stage.Get(),0,D3D11_MAP_READ,0,&m));std::vector<unsigned> pixels(d.Width*d.Height);
    for(unsigned y=0;y<d.Height;++y)std::memcpy(pixels.data()+y*d.Width,static_cast<char*>(m.pData)+y*m.RowPitch,d.Width*4);
    context->Unmap(stage.Get(),0);return pixels;
}
// Only this executable reads GPU pixels, from a cropped text-sized target.
// Production compiles synthetic GDI backgrounds and submits immutable inputs.
void qualify_gpu_text(HDC source,c3x_native_text::Raster const& raster,char const* text,unsigned count,RECT anchor,RECT clip){
    using namespace c3x_gpu_images;
    auto require=[](bool value,char const* message){if(!value)throw std::runtime_error(message);};
    c3x_native_text::State state;SIZE extent={};TEXTMETRICA metrics={};
    require(c3x_native_text::capture(source,state)&&GetTextExtentPoint32A(source,text,int(count),&extent)&&GetTextMetricsA(source,&metrics),"captured native metrics");
    RECT original={};require(c3x_native_text::place(state.align,extent.cx,metrics.tmAscent,metrics.tmHeight,anchor.left,anchor.top,raster.left,raster.top,raster.width,raster.height,original),"captured native placement");
    // Integer translation crops a small owned test region around the actual
    // attachment point; the recorded clipping/align/font remain meaningful.
    int crop_x=original.left-8,crop_y=original.top-8;
    anchor.left-=crop_x;anchor.top-=crop_y;clip.left-=crop_x;clip.right-=crop_x;clip.top-=crop_y;clip.bottom-=crop_y;
    unsigned width=raster.width+16,height=raster.height+16;
    require(std::uint64_t(width)*height<=65536,"cropped text test bound");
    Dib native(32,false,int(width),int(height));auto selected=SelectObject(native.dc,GetCurrentObject(source,OBJ_FONT));
    SetTextColor(native.dc,state.foreground);SetBkColor(native.dc,state.background);SetBkMode(native.dc,state.mode);SetTextAlign(native.dc,state.align);
    Microsoft::WRL::ComPtr<ID3D11Device> device;Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL feature;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context));
    Compositor gpu(device.Get(),context.Get());
    auto glyph=gpu.create(raster.width,raster.height,Format::bgra32),curves=gpu.create(17,unsigned(raster.curves.size()/17),Format::bgra32),target=gpu.create(width,height,Format::bgra32);
    require(glyph&&curves&&target&&gpu.upload(glyph,1,raster.pixels.data(),raster.pixels.size())&&gpu.upload(curves,1,raster.curves.data(),raster.curves.size()),"captured glyph GPU upload");
    c3x_native_text::State shadow_state=state;shadow_state.foreground=RGB(21,57,83);
    c3x_native_text::Raster shadow;require(c3x_native_text::compile(source,shadow_state,text,count,shadow),"ordered shadow text compilation");
    auto shadow_glyph=gpu.create(shadow.width,shadow.height,Format::bgra32),shadow_curves=gpu.create(17,unsigned(shadow.curves.size()/17),Format::bgra32);
    require(shadow_glyph&&shadow_curves&&gpu.upload(shadow_glyph,1,shadow.pixels.data(),shadow.pixels.size())&&gpu.upload(shadow_curves,1,shadow.curves.data(),shadow.curves.size()),"ordered shadow GPU upload");
    unsigned error_max=0,samples_exact=0,cases=0;std::uint64_t revision=0;
    for(unsigned alignment:{state.align,unsigned(TA_CENTER|TA_BASELINE),unsigned(TA_RIGHT|TA_BOTTOM),unsigned(TA_LEFT|TA_TOP)})for(bool clipped:{false,true}){
        int x=8-raster.left,y=8-raster.top;
        if((alignment&TA_CENTER)==TA_CENTER)x+=int((std::int64_t(extent.cx)+1)/2);else if(alignment&TA_RIGHT)x+=extent.cx;
        if((alignment&TA_BASELINE)==TA_BASELINE)y+=metrics.tmAscent;else if(alignment&TA_BOTTOM)y+=metrics.tmHeight;
        if(alignment==state.align){x=anchor.left;y=anchor.top;}
        RECT placed={};require(c3x_native_text::place(alignment,extent.cx,metrics.tmAscent,metrics.tmHeight,x,y,raster.left,raster.top,raster.width,raster.height,placed),"native alignment placement");
        RECT current_clip=clipped?RECT{LONG(8+raster.width/3),10,LONG(width-12),LONG(height-11)}:clip;
        auto region=CreateRectRgn(current_clip.left,current_clip.top,current_clip.right,current_clip.bottom);require(region!=nullptr,"native clipping region");SelectClipRgn(native.dc,region);DeleteObject(region);
        SetTextAlign(native.dc,alignment);
        for(unsigned sample=0;sample<19;++sample){
            unsigned level=std::min(sample*16u,255u);std::vector<unsigned> before(std::size_t(width)*height);
            for(unsigned n=0;n<before.size();++n)before[n]=sample<17?(0xff000000u|level*0x010101u):(0xff000000u|((n*3137u+sample*1777u)&0xffffffu));
            std::copy(before.begin(),before.end(),static_cast<unsigned*>(native.pixels));
            require(TextOutA(native.dc,x,y,text,int(count))!=FALSE,"native complete-string text output");GdiFlush();
            std::vector<unsigned> expected=before;
            for(unsigned row=0;row<height;++row)for(unsigned col=0;col<width;++col){
                auto n=row*width+col;int sx=int(col)-placed.left,sy=int(row)-placed.top;
                if(int(col)>=current_clip.left&&int(col)<current_clip.right&&int(row)>=current_clip.top&&int(row)<current_clip.bottom&&sx>=0&&sy>=0&&sx<int(raster.width)&&sy<int(raster.height))
                    expected[n]=c3x_native_text::apply(raster,unsigned(sy)*raster.width+unsigned(sx),before[n],false,true);
                auto actual=static_cast<unsigned*>(native.pixels)[n];unsigned error=0;
                for(unsigned c=0;c<3;++c)error=std::max(error,unsigned(std::abs(int(expected[n]>>(8*c)&255)-int(actual>>(8*c)&255))));
                if(sample<17&&error){
                    std::printf("TEXT_SAMPLE_MISMATCH count=%u font=%s font_height=%ld quality=%u mode=%d foreground=%08x background=%08x align=%u clipped=%u sample=%u level=%u pixel=%u,%u source=%d,%d anchor=%d,%d placed=%ld,%ld,%ld,%ld clip=%ld,%ld,%ld,%ld metrics=%ld,%ld,%ld raster=%u,%u expected=%08x native=%08x curve_ids=%08x\n",
                        count,state.font.lfFaceName,state.font.lfHeight,unsigned(state.font.lfQuality),state.mode,unsigned(state.foreground),unsigned(state.background),alignment,unsigned(clipped),sample,level,col,row,sx,sy,x,y,placed.left,placed.top,placed.right,placed.bottom,current_clip.left,current_clip.top,current_clip.right,current_clip.bottom,extent.cx,metrics.tmHeight,metrics.tmAscent,raster.width,raster.height,expected[n],actual,(sx>=0&&sy>=0&&sx<int(raster.width)&&sy<int(raster.height))?raster.pixels[unsigned(sy)*raster.width+unsigned(sx)]:0u);
                    require(false,"strip shaping differs from monolithic native GDI sample");
                }
                error_max=std::max(error_max,error);
            }
            if(sample<17){++samples_exact;continue;}
            require(gpu.upload(target,++revision,before.data(),before.size()),"test destination upload");
            Rect area={placed.left,placed.top,placed.right,placed.bottom},scissor={current_clip.left,current_clip.top,current_clip.right,current_clip.bottom};
            // A different-color shadow precedes a full fill, then the actual
            // foreground. Reordering either text around the fill changes output.
            unsigned sentinel=0xff335577u;
            Command commands[3]={{Kind::native_text,target,shadow_glyph,area,scissor,0,0,0,shadow_curves},{Kind::fill,target,0,{0,0,int(width),int(height)},{0,0,int(width),int(height)},0,0,sentinel},{Kind::native_text,target,glyph,area,scissor,0,0,0,curves}};
            require(gpu.submit(commands,3),"ordered copied GPU text submission");
            auto got=read_gpu(device.Get(),context.Get(),gpu.texture(target));
            for(unsigned row=0;row<height;++row)for(unsigned col=0;col<width;++col){
                auto n=row*width+col;int sx=int(col)-placed.left,sy=int(row)-placed.top;unsigned wanted=sentinel;
                if(int(col)>=current_clip.left&&int(col)<current_clip.right&&int(row)>=current_clip.top&&int(row)<current_clip.bottom&&sx>=0&&sy>=0&&sx<int(raster.width)&&sy<int(raster.height))
                    wanted=c3x_native_text::apply(raster,unsigned(sy)*raster.width+unsigned(sx),wanted,false,true);
                require(got[n]==wanted,"actual GPU text response/order differs from copied raster");
            }++cases;
        }
    }
    require(error_max<=3,"native smoothing exceeds existing three-level interpolation bound");
    gpu.destroy(shadow_glyph);gpu.destroy(shadow_curves);gpu.destroy(glyph);gpu.destroy(curves);gpu.destroy(target);SelectObject(native.dc,selected);
    std::printf("PASS captured native GPU text count=%u raster=%u,%u exact_GDI_samples=%u GPU_order_cases=%u edge_max=%u original_failure_identity=unproved\n",count,raster.width,raster.height,samples_exact,cases,error_max);
}
void actual_native_message_contract(){
    char const text[]="Our recent breakthrough in technology has proven the brilliance of a great scientific leader, Aida Yasuki!";
    static_assert(sizeof(text)-1==106,"recorded native message length");
    Dib source(32,false);LOGFONTA description={};description.lfHeight=-10;description.lfOutPrecision=7;strcpy_s(description.lfFaceName,"Lucida Sans");
    for(int quality:{DEFAULT_QUALITY,ANTIALIASED_QUALITY,CLEARTYPE_QUALITY})for(int mode:{TRANSPARENT,OPAQUE}){
        description.lfQuality=BYTE(quality);auto font=CreateFontIndirectA(&description);if(!font)throw std::runtime_error("recorded native font");auto old=SelectObject(source.dc,font);
        SetTextColor(source.dc,16316664);SetBkColor(source.dc,16777215);SetTextAlign(source.dc,TA_BASELINE);SetBkMode(source.dc,mode);
        c3x_native_text::State state;c3x_native_text::Raster raster;c3x_native_text::Diagnostic diagnostic;
        if(!c3x_native_text::capture(source.dc,state,&diagnostic)||!c3x_native_text::compile(source.dc,state,text,106,raster,&diagnostic)||raster.width!=543||raster.height!=42)throw std::runtime_error("recorded native text extent or admission differs");
        qualify_gpu_text(source.dc,raster,text,106,{861,592,0,0},{0,0,2240,1260});
        SelectObject(source.dc,old);DeleteObject(font);
    }
}

void native_strip_seam_contract(){
    Dib source(32,false);LOGFONTA description={};description.lfHeight=-32;description.lfQuality=ANTIALIASED_QUALITY;strcpy_s(description.lfFaceName,"Arial");
    auto font=CreateFontIndirectA(&description);if(!font)throw std::runtime_error("strip seam native font");auto old=SelectObject(source.dc,font);
    SetTextColor(source.dc,RGB(237,171,55));SetBkColor(source.dc,RGB(47,68,211));SetTextAlign(source.dc,TA_BASELINE);SetBkMode(source.dc,TRANSPARENT);
    c3x_native_text::State state;c3x_native_text::Raster raster;std::string text="AgjpQRST";bool found=false;
    for(unsigned attempt=0;attempt<24;++attempt){
        SIZE extent={};TEXTMETRICA metrics={};if(!GetTextExtentPoint32A(source.dc,text.data(),int(text.size()),&extent)||!GetTextMetricsA(source.dc,&metrics))throw std::runtime_error("strip seam metrics");
        int margin=metrics.tmHeight+std::abs(metrics.tmOverhang);unsigned width=unsigned(extent.cx+2*margin),height=unsigned(metrics.tmHeight+2*margin);
        unsigned rows=c3x_native_text::response_tile_pixels/width;
        if(std::uint64_t(width)*height<=c3x_native_text::maximum_raster_pixels&&rows>unsigned(margin)&&rows<unsigned(margin+metrics.tmHeight)){
            if(!c3x_native_text::capture(source.dc,state)||!c3x_native_text::compile(source.dc,state,text.data(),unsigned(text.size()),raster))throw std::runtime_error("strip seam compilation");
            qualify_gpu_text(source.dc,raster,text.data(),unsigned(text.size()),{300,80,0,0},{0,0,640,240});found=true;break;
        }text+='W';
    }
    SelectObject(source.dc,old);DeleteObject(font);if(!found)throw std::runtime_error("native glyph seam fixture unavailable");
}

void save_comparison(unsigned short const* native,unsigned const* wide,std::vector<unsigned> const& gpu){
    std::vector<unsigned> pixels(320*48*3);
    for(unsigned n=0;n<320*48;++n){pixels[n]=expand(native[n],false);pixels[320*48+n]=wide[n];pixels[2*320*48+n]=gpu[n];}
    BITMAPFILEHEADER file={};file.bfType=0x4d42;file.bfOffBits=sizeof(file)+sizeof(BITMAPINFOHEADER);file.bfSize=file.bfOffBits+DWORD(pixels.size()*4);
    BITMAPINFOHEADER info={};info.biSize=sizeof(info);info.biWidth=320;info.biHeight=-144;info.biPlanes=1;info.biBitCount=32;
    FILE* output=nullptr;if(fopen_s(&output,"build/gpu-composition/native-text-comparison.bmp","wb")||!output)throw std::runtime_error("comparison output");
    std::fwrite(&file,sizeof(file),1,output);std::fwrite(&info,sizeof(info),1,output);std::fwrite(pixels.data(),4,pixels.size(),output);std::fclose(output);
}
int main(int argc,char** argv){try {
    if(argc==3&&!std::strcmp(argv[1],"--reproduce-refusal")){reproduce_refusal(argv[2]);return 0;}
    if(argc==3&&!std::strcmp(argv[1],"--reproduce-candidate")){reproduce_refusal(argv[2],true);return 0;}
    if(argc==3&&!std::strcmp(argv[1],"--qualify-candidate")){reproduce_refusal(argv[2],true,true);return 0;}
    if(argc!=1)throw std::runtime_error("usage: test_native_text [--reproduce-refusal|--reproduce-candidate|--qualify-candidate renderer.log]");
    text_extent_refusal_contract();
    actual_native_message_contract();
    native_strip_seam_contract();
    using namespace c3x_gpu_images;Microsoft::WRL::ComPtr<ID3D11Device> device;Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;D3D_FEATURE_LEVEL feature;
    checked(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&feature,&context));
    Compositor gpu(device.Get(),context.Get());
    for(bool g6:{false,true})for(int quality:{DEFAULT_QUALITY,ANTIALIASED_QUALITY,CLEARTYPE_QUALITY})for(int bk:{TRANSPARENT,OPAQUE}){
        Dib native(16,g6),wide(32,g6);LOGFONTA lf={};lf.lfHeight=-17;lf.lfWeight=700;lf.lfQuality=BYTE(quality);lf.lfOutPrecision=7;
        strcpy_s(lf.lfFaceName,"Arial");auto font=CreateFontIndirectA(&lf);
        auto a=SelectObject(native.dc,font),b=SelectObject(wide.dc,font);
        c3x_native_text::State state;SetBkMode(native.dc,bk);SetTextColor(native.dc,RGB(237,171,55));SetBkColor(native.dc,RGB(47,68,211));
        c3x_native_text::Raster raster;LARGE_INTEGER started,ended,frequency;QueryPerformanceFrequency(&frequency);QueryPerformanceCounter(&started);
        bool compiled=c3x_native_text::capture(native.dc,state)&&c3x_native_text::compile(native.dc,state,"Berlin: 6",9,raster);
        QueryPerformanceCounter(&ended);std::printf("compiled=%d curves=%zu area=%ux%u ms=%.3f\n",int(compiled),raster.curves.size()/17,raster.width,raster.height,1000.*(ended.QuadPart-started.QuadPart)/frequency.QuadPart);
        if(!compiled)throw std::runtime_error("native glyph compilation rejected");
        auto glyph=gpu.create(raster.width,raster.height,Format::bgra32),curves=gpu.create(17,unsigned(raster.curves.size()/17),Format::bgra32);
        auto destination=gpu.create(320,48,g6?Format::rgb565:Format::rgb555),detail=gpu.create(320,48,Format::bgra32);
        if(!glyph||!curves||!destination||!detail||!gpu.upload(glyph,1,raster.pixels.data(),raster.pixels.size())||!gpu.upload(curves,1,raster.curves.data(),raster.curves.size()))throw std::runtime_error("glyph GPU upload");
        unsigned mismatch=0,partial=0,changed_high=0,response_error=0,response_pixels=0;
        for(int background=0;background<4;++background){
            std::vector<unsigned short> before(320*48);
            for(unsigned n=0;n<before.size();++n){unsigned c=background==0?0:background==1?0x4210:background==2?0xc210:(n*3137)&65535;
                before[n]=static_cast<unsigned short>(c);static_cast<unsigned short*>(native.pixels)[n]=before[n];static_cast<unsigned*>(wide.pixels)[n]=expand(c,g6);}
            std::vector<unsigned> native_before(before.begin(),before.end()),wide_before(320*48);
            for(unsigned n=0;n<wide_before.size();++n)wide_before[n]=expand(before[n],g6)|0xff000000u;
            gpu.upload(destination,unsigned(background+1),native_before.data(),native_before.size());gpu.upload(detail,unsigned(background+1),wide_before.data(),wide_before.size());
            Rect area={7+raster.left,9+raster.top,7+raster.left+int(raster.width),9+raster.top+int(raster.height)};
            Command commands[2]={{Kind::native_text,destination,glyph,area,{0,0,320,48},0,0,0,curves},{Kind::native_text,detail,glyph,area,{0,0,320,48},0,0,0,curves}};
            if(!gpu.submit(commands,2))throw std::runtime_error("GPU glyph submission");
            auto native_gpu=read_gpu(device.Get(),context.Get(),gpu.texture(destination)),detail_gpu=read_gpu(device.Get(),context.Get(),gpu.texture(detail));
            for(auto dc:{native.dc,wide.dc}){SetBkMode(dc,bk);SetTextColor(dc,RGB(237,171,55));SetBkColor(dc,RGB(47,68,211));TextOutA(dc,7,9,"Berlin: 6",9);}GdiFlush();
            if(!g6&&quality==DEFAULT_QUALITY&&bk==TRANSPARENT&&background==1)save_comparison(static_cast<unsigned short*>(native.pixels),static_cast<unsigned*>(wide.pixels),detail_gpu);
            unsigned fg=pack(0xedab37,g6),bg=pack(0x2f44d3,g6);
            for(unsigned n=0;n<before.size();++n){unsigned got=static_cast<unsigned short*>(native.pixels)[n],rgb=static_cast<unsigned*>(wide.pixels)[n];
                if((got&(g6?65535:32767))!=pack(rgb,g6))++mismatch;
                if(compiled){int x=int(n%320)-7-raster.left,y=int(n/320)-9-raster.top;
                    unsigned response=expand(before[n],g6)|0xff000000u;
                    unsigned packed_response=before[n];
                    if(x>=0&&y>=0&&x<int(raster.width)&&y<int(raster.height)){
                        response=c3x_native_text::apply(raster,unsigned(y)*raster.width+unsigned(x),response,g6,true);
                        packed_response=c3x_native_text::apply(raster,unsigned(y)*raster.width+unsigned(x),before[n],g6,false);
                    }
                    if(response!=detail_gpu[n]||packed_response!=native_gpu[n]){std::printf("pixel=%u expected_native=%04x actual_native=%04x expected_full=%08x actual_full=%08x before=%04x\n",n,packed_response,native_gpu[n],response,detail_gpu[n],unsigned(before[n]));throw std::runtime_error("GPU text differs from compiled response");}
                    unsigned error=0;for(unsigned c=0;c<3;++c)error=std::max(error,unsigned(std::abs(int(response>>(8*c)&255)-int(rgb>>(8*c)&255))));
                    response_error=std::max(response_error,error);response_pixels+=error!=0;
                }
                if(got!=before[n]&&got!=fg&&got!=bg)++partial;
                if(!g6&&(got&32767)==(unsigned(before[n])&32767u)&&got!=before[n])++changed_high;
            }
        }
        std::printf("text format=%s quality=%d background=%d native_vs_bgra=%u partial=%u high_only=%u response_max=%u response_pixels=%u\n",g6?"565":"555",quality,bk,mismatch,partial,changed_high,response_error,response_pixels);
        if(response_error>3)throw std::runtime_error("text exceeds three-level interpolation tolerance");
        gpu.destroy(glyph);gpu.destroy(curves);gpu.destroy(destination);gpu.destroy(detail);
        SelectObject(native.dc,a);SelectObject(wide.dc,b);DeleteObject(font);
    }std::puts("PASS native GPU text: 555/565 and full color, actual GDI smoothing, transparent/opaque, bounded edge error, exact GPU response");return 0;
}catch(std::exception const& e){std::printf("FAIL %s\n",e.what());return 1;}}
