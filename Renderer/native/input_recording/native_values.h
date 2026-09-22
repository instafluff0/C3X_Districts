#pragma once
#include "runtime.h"
#include "../native_access.h"
#include <tuple>
#include <set>
namespace c3x_inputs {
using NativeKey=std::tuple<unsigned,unsigned,unsigned>;
using NativeValues=std::map<NativeKey,Bytes>;
// One invocation's external values, independent of GPU handles or native layouts.
// Only content hashes persist in the recorder; replay owns the actual values.
struct NativeValueStream {
    std::map<NativeKey,std::array<unsigned,4>> hashes;
    NativeValues values;
    void retire(unsigned object){for(auto it=hashes.begin();it!=hashes.end();)if(std::get<1>(it->first)==object)it=hashes.erase(it);else ++it;for(auto it=values.begin();it!=values.end();)if(std::get<1>(it->first)==object)it=values.erase(it);else ++it;}
    void encode(Writer& out,NativeValues const& incoming){
        require(incoming.size()<=4096,"native dependency count limit");out.u32(unsigned(incoming.size()));
        for(auto const& entry:incoming){auto const& key=entry.first;auto const& bytes=entry.second;
            auto hash=c3x_renderer::asset_content_hash(bytes.data(),bytes.size());auto found=hashes.find(key);
            bool changed=found==hashes.end()||found->second!=hash;
            out.u32(std::get<0>(key));out.u32(std::get<1>(key));out.u32(std::get<2>(key));out.u32(changed?1:0);
            for(auto x:hash)out.u32(x);
            if(changed){out.u32(unsigned(bytes.size()));out.reserve(bytes.size());out.bytes.insert(out.bytes.end(),bytes.begin(),bytes.end());hashes[key]=hash;require(hashes.size()<=32768,"native dependency hash budget");}
        }
    }
    NativeValues decode(Reader& in){NativeValues current;auto count=in.u32();require(count<=4096,"native dependency count limit");
        for(unsigned n=0;n<count;++n){auto kind=in.u32(),object=in.u32(),field=in.u32();NativeKey key{kind,object,field};auto changed=in.u32();require(changed<=1,"native dependency encoding");std::array<unsigned,4> hash;for(auto& x:hash)x=in.u32();
            if(changed){auto size=in.u32();in.available(size);require(size<=payload_limit,"native value limit");values[key]=Bytes(in.bytes.begin()+in.at,in.bytes.begin()+in.at+size);in.at+=size;}
            auto found=values.find(key);require(found!=values.end(),"missing native dependency predecessor");require(c3x_renderer::asset_content_hash(found->second.data(),found->second.size())==hash,"native dependency content mismatch");
            require(current.emplace(key,found->second).second,"duplicate native dependency");}
        std::size_t bytes=0;for(auto const& entry:values)bytes+=entry.second.size();require(bytes<=96u*1024u*1024u&&values.size()<=32768,"native dependency replay budget");return current;
    }
};
struct NativeValuesProvider: c3x_native_access::Provider {
    enum Query:unsigned {scalar=1,reference,is_sprite,scaling,default_palette,bitmap,context,graph,read_words,write_words,sprite_pixels,row_table,palette_words,palette_rgb,written_field,written_words,camera_readiness,lookup_words};
    bool replay=false;NativeValues values;HWND target=nullptr;std::size_t bytes=0;bool failed=false;
    std::map<unsigned,unsigned short*> writes;
    std::map<unsigned,HDC> contexts;std::vector<HGDIOBJ> objects;
    bool recording_failed()noexcept override{if(replay)return false;failed=true;runtime().stop(Stop::unsupported);return true;}
    unsigned identity(void const* p){return replay?unsigned(reinterpret_cast<std::uintptr_t>(p)):runtime().object(const_cast<void*>(p));}
    void* object(unsigned id){return reinterpret_cast<void*>(std::uintptr_t(id));}
    NativeKey key(unsigned type,void const* p,unsigned index=0){return {type,identity(p),index};}
    Bytes& get(NativeKey const& id){auto found=values.find(id);if(found==values.end())throw std::runtime_error("missing native external input: kind="+std::to_string(std::get<0>(id))+" object="+std::to_string(std::get<1>(id))+" field="+std::to_string(std::get<2>(id)));return found->second;}
    void save(NativeKey const& id,Writer& out)noexcept{try{if(failed)return;if(values.count(id))return;require(out.bytes.size()<=payload_limit-bytes,"native invocation input budget");bytes+=out.bytes.size();values.emplace(id,std::move(out.bytes));}catch(...){failed=true;runtime().stop(Stop::unsupported);}}
    template<class F>int number(unsigned type,void* p,unsigned index,F native){auto id=key(type,p,index);if(replay){Reader in{get(id)};int result=0;in(result);in.done();return result;}int result;{c3x_native_access::NativeScope scope;result=native();}Writer out;out(result);save(id,out);return result;}
    int field(void* p,unsigned offset)override{return number(scalar,p,offset,[&]{auto n=c3x_native_access::field(p,offset);return offset==0x14?int(n!=0):n;});}
    void set_field(void* p,unsigned offset,unsigned value)override{auto id=key(written_field,p,offset);if(replay){Reader in{get(id)};require(in.u32()==value,"native field output differs");Writer next;next.u32(value);values[key(scalar,p,offset)]=std::move(next.bytes);return;} {c3x_native_access::NativeScope scope;c3x_native_access::set_field(p,offset,value);}Writer out;out(value);save(id,out);}
    void* pointer(void* p,unsigned offset)override{auto id=key(reference,p,offset);if(replay){Reader in{get(id)};return object(in.u32());}void* result;{c3x_native_access::NativeScope scope;result=c3x_native_access::pointer(p,offset);}Writer out;out.u32(identity(result));save(id,out);return result;}
    bool sprite(void* p)override{return number(is_sprite,p,0,[&]{return int(c3x_native_access::sprite(p));})!=0;}
    int scale(unsigned axis)override{return number(scaling,nullptr,axis,[&]{return c3x_native_access::scale(axis);});}
    void* palette()override{auto id=key(default_palette,nullptr);if(replay){Reader in{get(id)};return object(in.u32());}void* result;{c3x_native_access::NativeScope scope;result=c3x_native_access::palette();}Writer out;out.u32(identity(result));save(id,out);return result;}
    bool dib(void* p,DIBSECTION& d)override{auto id=key(bitmap,p);bool ok=false;
        if(replay){Reader in{get(id)};ok=in.u32()!=0;d={};if(ok){d.dsBm.bmWidth=int(in.u32());d.dsBm.bmHeight=int(in.u32());d.dsBm.bmWidthBytes=int(in.u32());d.dsBm.bmBitsPixel=WORD(in.u32());d.dsBmih.biCompression=in.u32();for(auto& x:d.dsBitfields)x=in.u32();d.dsBm.bmBits=in.u32()?this:nullptr;}in.done();return ok;}
        {c3x_native_access::NativeScope scope;ok=c3x_native_access::dib(p,d);}Writer out;out.u32(ok);if(ok){out.u32(d.dsBm.bmWidth);out.u32(d.dsBm.bmHeight);out.u32(d.dsBm.bmWidthBytes);out.u32(d.dsBm.bmBitsPixel);out.u32(d.dsBmih.biCompression);for(auto x:d.dsBitfields)out.u32(x);out.u32(d.dsBm.bmBits!=nullptr);}save(id,out);return ok;}
    HWND window(void* p)override{auto id=key(graph,p);if(replay){Reader in{get(id)};return in.u32()?target:nullptr;}HWND result;{c3x_native_access::NativeScope scope;result=c3x_native_access::window(p);}Writer out;out.u32(result!=nullptr);save(id,out);return result;}
    int readiness(int value)override{auto id=key(camera_readiness,nullptr);if(replay){auto found=values.find(id);if(found==values.end())return value;Reader in{found->second};int recorded=int(in.u32());
        // Native CPU values needed for adoption first exist at the recorded
        // consumption boundary. Keep earlier offers pending; at consumption,
        // performance mode waits only for actual work, never recorded latency.
        if(replay_execution().performance&&recorded!=C3X_RENDERER_RESULT_PENDING)return -999;
        return recorded;}if(value!=-999){Writer out;out(value);save(id,out);}return value;}
    HDC dc(void* p)override;
    unsigned short* words(void* p,void* getter,bool write)override{
        auto id=key(write?write_words:read_words,p);if(replay){auto& data=get(id);Reader in{data};auto present=in.u32();if(!present)return nullptr;auto count=in.u32();require(count<=2240u*1260u+1260u,"native words extent");if(write&&data.size()==8)data.resize(8+std::size_t(count)*2);require(data.size()==8+std::size_t(count)*2,"native words payload");auto result=reinterpret_cast<unsigned short*>(data.data()+8);if(write)writes[identity(p)]=result;return result;}
        unsigned short* result;{c3x_native_access::NativeScope scope;result=c3x_native_access::words(p,getter,write);}borrowed_words=true;last_words=result;Writer out;out.u32(result!=nullptr);
        if(result){int w=field(p,0x38),h=field(p,0x3c),stride=field(p,0x40);require(w>0&&w<=2240&&h>0&&h<=1260&&stride>=w&&stride<=2241,"native words extent");auto count=unsigned(stride)*unsigned(h);out.u32(count);if(!write)out.reserve(std::size_t(count)*2);
            if(!write)for(int y=0;y<h;++y)for(int x=0;x<stride;++x){unsigned word=x<w?result[y*stride+x]:0;out.bytes.push_back(static_cast<unsigned char>(word));out.bytes.push_back(static_cast<unsigned char>(word>>8));}
            if(write)writes[identity(p)]=result;}
        save(id,out);return result;
    }
    void release_words(void* p,void* release)override{auto found=writes.find(identity(p));if(found!=writes.end()){
            if(replay&&replay_execution().performance){writes.erase(found);return;}
            auto w=field(p,0x38),h=field(p,0x3c),stride=field(p,0x40);Sha256 hash;for(int y=0;y<h;++y)hash.add(found->second+y*stride,std::size_t(w)*2);auto result=hash.finish();auto id=key(written_words,p);
            if(replay){Reader in{get(id)};for(auto x:result){auto expected=in.u32();if(!replay_execution().performance&&expected!=x)throw std::runtime_error("native CPU ownership output differs: image="+std::to_string(identity(p))+" extent="+std::to_string(w)+"x"+std::to_string(h)+" expected="+std::to_string(expected)+" actual="+std::to_string(x));}in.done();}
            else {Writer out;for(auto x:result)out.u32(x);save(id,out);}writes.erase(found);}
        if(!replay){c3x_native_access::NativeScope scope;c3x_native_access::release_words(p,release);}}
    unsigned char const* buffer(unsigned type,void* p,unsigned size,unsigned char const* data){auto id=key(type,p);if(replay){auto& b=get(id);Reader in{b};if(!in.u32())return nullptr;auto n=in.u32();require(n==size&&b.size()==8+size,"native byte input extent");return b.data()+8;}Writer out;out.u32(data!=nullptr);if(data){out.u32(size);out.reserve(size);out.bytes.insert(out.bytes.end(),data,data+size);}save(id,out);return data;}
    unsigned char const* rows(void* p)override{auto height=field(p,0x34);require(height>0&&height<=1024,"native row table height");unsigned char const* result=nullptr;if(!replay){c3x_native_access::NativeScope scope;result=c3x_native_access::rows(p);}return buffer(row_table,p,unsigned(height)*4,result);}
    unsigned char* sprite_bytes(void* p)override{unsigned size=0;int h=field(p,0x34),stride=field(p,0x2c),bits=field(p,0x20);require(h>0&&h<=1024&&(bits==8||bits==16),"native sprite dimensions");
        if(field(p,0x18)&1){auto table=rows(p);require(table!=nullptr,"native sprite rows missing");for(int y=0;y<h;++y)size=std::max(size,(unsigned(table[y*4+2])|(unsigned(table[y*4+3])<<8))+table[y*4+1]);}
        else {require(stride>0&&stride<=4096,"native sprite stride");size=unsigned(stride)*unsigned(h)*unsigned(bits/8);}
        unsigned char* result=nullptr;if(!replay){c3x_native_access::NativeScope scope;result=c3x_native_access::sprite_bytes(p);}if(!replay){borrowed_sprite=true;last_sprite=result;}return const_cast<unsigned char*>(buffer(sprite_pixels,p,size,result));}
    void release_sprite(void* p)override{if(!replay){c3x_native_access::NativeScope scope;c3x_native_access::release_sprite(p);}}
    unsigned short const* colors(void const* p,bool green6)override{auto id=key(palette_words,p,green6);if(replay){auto& b=get(id);Reader in{b};return in.u32()?reinterpret_cast<unsigned short const*>(b.data()+4):nullptr;}
        unsigned short const* result;{c3x_native_access::NativeScope scope;result=c3x_native_access::colors(p,green6);}Writer out;out.u32(result!=nullptr);if(result){out.reserve(512);for(unsigned n=0;n<256;++n){out.bytes.push_back(static_cast<unsigned char>(result[n]));out.bytes.push_back(static_cast<unsigned char>(result[n]>>8));}}save(id,out);return result;}
    unsigned short const* lookup(void const* p,unsigned blocks)override{require(blocks&&blocks<=31,"native lookup blocks");auto id=key(lookup_words,p,blocks);
        if(replay){auto& data=get(id);require(data.size()==std::size_t(blocks)*65536,"native lookup input extent");return reinterpret_cast<unsigned short const*>(data.data());}
        Writer out;auto data=static_cast<unsigned char const*>(p);auto size=std::size_t(blocks)*65536;out.reserve(size);out.bytes.insert(out.bytes.end(),data,data+size);save(id,out);return static_cast<unsigned short const*>(p);
    }
    unsigned rgb(void* p,unsigned index)override{return unsigned(number(palette_rgb,p,index,[&]{return int(c3x_native_access::rgb(p,index));}));}
    ~NativeValuesProvider(){for(auto const& entry:contexts)DeleteDC(entry.second);for(auto value:objects)DeleteObject(value);}
};
inline HDC NativeValuesProvider::dc(void* p){
    auto id=key(context,p);auto object_id=identity(p);if(replay&&contexts.count(object_id))return contexts.at(object_id);
    HDC result=nullptr;Writer out;
    if(!replay){c3x_native_access::NativeScope scope;result=c3x_native_access::dc(p);out.u32(result!=nullptr);if(result){
        LOGFONTA f={};require(GetObjectA(GetCurrentObject(result,OBJ_FONT),sizeof f,&f)==sizeof f,"native font metadata missing");
        for(auto n:{f.lfHeight,f.lfWidth,f.lfEscapement,f.lfOrientation,f.lfWeight})out(std::int32_t(n));
        for(auto n:{f.lfItalic,f.lfUnderline,f.lfStrikeOut,f.lfCharSet,f.lfOutPrecision,f.lfClipPrecision,f.lfQuality,f.lfPitchAndFamily})out.u32(n);out.string(f.lfFaceName,LF_FACESIZE);
        for(auto n:{GetMapMode(result),GetGraphicsMode(result),GetStretchBltMode(result),GetBkMode(result),GetTextCharacterExtra(result)})out(std::int32_t(n));
        out.u32(GetLayout(result));out.u32(GetTextAlign(result));out.u32(GetTextColor(result));out.u32(GetBkColor(result));
        POINT a={},b={};SIZE w={},v={};GetViewportOrgEx(result,&a);GetWindowOrgEx(result,&b);GetWindowExtEx(result,&w);GetViewportExtEx(result,&v);
        for(auto n:{a.x,a.y,b.x,b.y,w.cx,w.cy,v.cx,v.cy})out(std::int32_t(n));XFORM x={1,0,0,1,0,0};if(GetGraphicsMode(result)==GM_ADVANCED)require(GetWorldTransform(result,&x)!=FALSE,"native world transform missing");for(auto n:{x.eM11,x.eM12,x.eM21,x.eM22,x.eDx,x.eDy})out(n);
        HRGN clip=CreateRectRgn(0,0,0,0);require(clip!=nullptr,"native clip allocation");int has=GetClipRgn(result,clip);out(std::int32_t(has));
        if(has==1){auto length=GetRegionData(clip,0,nullptr);require(length&&length<=65536,"native clip extent");std::vector<unsigned char> data(length);require(GetRegionData(clip,length,reinterpret_cast<RGNDATA*>(data.data()))==length,"native clip read");auto region=reinterpret_cast<RGNDATA*>(data.data());require(region->rdh.iType==RDH_RECTANGLES,"native clip type");out.u32(region->rdh.nCount);auto rects=reinterpret_cast<RECT*>(region->Buffer);for(unsigned n=0;n<region->rdh.nCount;++n)for(auto value:{rects[n].left,rects[n].top,rects[n].right,rects[n].bottom})out(std::int32_t(value));}DeleteObject(clip);
    }save(id,out);return result;}
    Reader in{get(id)};if(!in.u32())return nullptr;
    LOGFONTA f={};f.lfHeight=LONG(in.u32());f.lfWidth=LONG(in.u32());f.lfEscapement=LONG(in.u32());f.lfOrientation=LONG(in.u32());f.lfWeight=LONG(in.u32());
    f.lfItalic=BYTE(in.u32());f.lfUnderline=BYTE(in.u32());f.lfStrikeOut=BYTE(in.u32());f.lfCharSet=BYTE(in.u32());f.lfOutPrecision=BYTE(in.u32());f.lfClipPrecision=BYTE(in.u32());f.lfQuality=BYTE(in.u32());f.lfPitchAndFamily=BYTE(in.u32());auto face=in.string(LF_FACESIZE);std::memcpy(f.lfFaceName,face.c_str(),face.size()+1);
    auto map=int(in.u32()),graphics=int(in.u32()),stretch=int(in.u32()),mode=int(in.u32()),extra=int(in.u32());auto layout=in.u32(),align=in.u32(),foreground=in.u32(),background=in.u32();int params[8];for(auto& n:params)n=int(in.u32());XFORM x;in(x.eM11);in(x.eM12);in(x.eM21);in(x.eM22);in(x.eDx);in(x.eDy);
    result=CreateCompatibleDC(nullptr);require(result!=nullptr,"replay native DC allocation");contexts[object_id]=result;
    BITMAPINFO info={};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);info.bmiHeader.biWidth=2240;info.bmiHeader.biHeight=-1260;info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;void* pixels=nullptr;auto surface=CreateDIBSection(result,&info,DIB_RGB_COLORS,&pixels,nullptr,0);require(surface!=nullptr,"replay native DC surface");objects.push_back(surface);SelectObject(result,surface);
    auto font=CreateFontIndirectA(&f);require(font!=nullptr,"replay native font allocation");objects.push_back(font);SelectObject(result,font);
    SetMapMode(result,map);SetGraphicsMode(result,graphics);SetStretchBltMode(result,stretch);SetBkMode(result,mode);SetTextCharacterExtra(result,extra);SetLayout(result,layout);SetTextAlign(result,align);SetTextColor(result,foreground);SetBkColor(result,background);
    SetViewportOrgEx(result,params[0],params[1],nullptr);SetWindowOrgEx(result,params[2],params[3],nullptr);SetWindowExtEx(result,params[4],params[5],nullptr);SetViewportExtEx(result,params[6],params[7],nullptr);if(graphics==GM_ADVANCED)SetWorldTransform(result,&x);
    int has=int(in.u32());if(has==1){auto count=in.u32();require(count<=4096,"replay clip count");HRGN clip=CreateRectRgn(0,0,0,0);require(clip!=nullptr,"replay clip allocation");for(unsigned n=0;n<count;++n){int r[4];for(auto& a:r)a=int(in.u32());auto part=CreateRectRgn(r[0],r[1],r[2],r[3]);require(part!=nullptr,"replay clip rectangle");CombineRgn(clip,clip,part,RGN_OR);DeleteObject(part);}SelectClipRgn(result,clip);DeleteObject(clip);}require(has>=0,"native clip acquisition failed");in.done();return result;
}
}
