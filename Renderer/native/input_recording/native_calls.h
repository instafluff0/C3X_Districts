#pragma once
#include "native_values.h"
namespace c3x_inputs {
inline NativeValueStream& recorded_native_values(){static NativeValueStream stream;return stream;}
struct NativeCall {
    Call call;NativeValuesProvider values; c3x_native_access::Provider* prior=nullptr;bool active=false;
    template<class Encode>NativeCall(unsigned kind,Encode encode):call(Kind::native_bridge,kind,encode){
        if(call.id){require(!native_root(),"nested native bridge recording");native_root()=true;prior=c3x_native_access::provider();c3x_native_access::provider()=&values;active=true;}}
    ~NativeCall(){if(active){c3x_native_access::provider()=prior;native_root()=false;}}
    template<class Encode>int result(int code,Encode encode){return call.result(code,[&](Writer& out){recorded_native_values().encode(out,values.values);encode(out);});}
    int result(int code){return result(code,[](Writer&){});}
};
inline unsigned native_id(void const* p){return runtime().object(const_cast<void*>(p));}
inline void native_rect(Writer& out,void const* p,unsigned count=4){out.u32(p?1:0);if(p){auto values=static_cast<int const*>(p);for(unsigned n=0;n<count;++n)out(std::int32_t(values[n]));}}
inline void native_table(Writer& out,void const* p,unsigned){out.u32(native_id(p));}
inline void native_view(Writer& out,custom_renderer_native_view const& v){
    out(v.camera_x);out(v.camera_y);out(v.min_x);out(v.max_x);out(v.min_y);out(v.max_y);
    out(v.width);out(v.height);out(v.tile_width);out(v.native_width);out(v.translate_x);out(v.translate_y);
}
inline void native_view(Reader& in,custom_renderer_native_view& v){in(v.camera_x);in(v.camera_y);in(v.min_x);in(v.max_x);in(v.min_y);in(v.max_y);in(v.width);in(v.height);in(v.tile_width);in(v.native_width);in(v.translate_x);in(v.translate_y);}
inline void native_request(Writer& out,c3x_renderer_camera_request_v1 const* p){out.u32(p?1:0);if(p){c3x_renderer_camera_identity_v1_fields(out,p->identity);frame(out,*p->frame);}}
inline void native_operation_input(Writer& out,int op,void* image,void* source,void const* from,void const* to,unsigned color){
    out(std::int32_t(op));out.u32(native_id(image));out.u32(op==C3X_NATIVE_TEXT?0:native_id(source));out(color);
    if(op==C3X_NATIVE_TEXT){out.u32(source?1:0);if(source){require(color<=1024,"native text length");out.u32(color);out.reserve(color);auto text=static_cast<unsigned char const*>(source);out.bytes.insert(out.bytes.end(),text,text+color);}native_rect(out,to);}
    else if(op==C3X_NATIVE_UNIT_DRAW){out.u32(from?1:0);if(from)unit(out,*static_cast<c3x_renderer_unit_v1 const*>(from));out.u32(to?1:0);}
    else if(op==C3X_NATIVE_TACTICAL_GRID){out.u32(from?1:0);if(from)frame(out,*static_cast<c3x_renderer_frame_v1 const*>(from));}
    else if(op==C3X_NATIVE_TACTICAL_ROUTE_BEGIN){out.u32(from?1:0);if(from){auto& v=*static_cast<c3x_renderer_tactical_view_v1 const*>(from);out(v.tile_width);out(v.native_tile_width);out(v.translate_x_fp);out(v.translate_y_fp);}}
    else if(op==C3X_NATIVE_LOOKUP||op==C3X_NATIVE_SPRITE_LOOKUP||op==C3X_NATIVE_SPRITE_LOOKUP_OVER||op==C3X_NATIVE_SPRITE_LOOKUP_SCALED){out.u32(from?1:0);if(from){auto& p=*static_cast<c3x_renderer_native_lookup const*>(from);out.u32(native_id(p.palette));out(p.percent);out.u32(native_id(p.background));for(auto x:p.scale)out(x);native_table(out,p.table,op>=C3X_NATIVE_SPRITE_LOOKUP_OVER?31:16);}native_rect(out,to);}
    else if(op==C3X_NATIVE_SPRITE_STYLE){out.u32(from?1:0);if(from){auto& p=*static_cast<c3x_renderer_native_sprite_style const*>(from);out.u32(native_id(p.palette));out(p.color);out(p.mode);out(p.opacity);native_table(out,p.table,p.mode==3?4:16);}native_rect(out,to);}
    else if(op==C3X_NATIVE_SPRITE_BLEND){out.u32(from?1:0);if(from){auto& p=*static_cast<c3x_renderer_native_sprite_blend const*>(from);out.u32(native_id(p.alpha));out.u32(native_id(p.background));out.u32(native_id(p.palette));}native_rect(out,to);}
    else if(op==C3X_NATIVE_SPRITE){out.u32(native_id(from));native_rect(out,to);}
    else if(op==C3X_NATIVE_COPY||op==C3X_NATIVE_FILL||op==C3X_NATIVE_IMAGE_DRAW||op==C3X_NATIVE_IMAGE_PRESENT||op==C3X_NATIVE_LINE||op==C3X_NATIVE_TINT||op==C3X_NATIVE_TACTICAL_RING||op==C3X_NATIVE_TACTICAL_TARGET||op==C3X_NATIVE_STROKE){
        native_rect(out,from,op==C3X_NATIVE_TINT?1:op==C3X_NATIVE_TACTICAL_TARGET?2:op==C3X_NATIVE_STROKE?7:4);native_rect(out,to);}
    // The remaining notifications consume identity/op/color only.
}
struct NativeOperationInput {
    int operation=0;unsigned color=0;void* image=nullptr;void* source=nullptr;void const* from=nullptr;void const* to=nullptr;
    std::array<int,7> a{};std::array<int,4> b{};std::string text;std::vector<unsigned short> table;
    c3x_renderer_unit_v1 actor{};Frame scene;c3x_renderer_tactical_view_v1 view{};c3x_renderer_native_lookup lookup{};c3x_renderer_native_sprite_style style{};c3x_renderer_native_sprite_blend blend{};
    static void* object(unsigned id){return reinterpret_cast<void*>(std::uintptr_t(id));}
    void rect(Reader& in,void const*& p,int* values,unsigned count=4){if(in.u32()){for(unsigned n=0;n<count;++n)in(values[n]);p=values;}}
    void words(Reader& in,void*& p){p=object(in.u32());}
    void decode(Reader& in){in(operation);image=object(in.u32());source=object(in.u32());in(color);auto op=operation;
        if(op==C3X_NATIVE_TEXT){if(in.u32()){auto n=in.u32();require(n==color&&n<=1024,"native text size");in.available(n);text.assign(reinterpret_cast<char const*>(in.bytes.data()+in.at),n);in.at+=n;source=text.data();}rect(in,to,b.data());}
        else if(op==C3X_NATIVE_UNIT_DRAW){if(in.u32()){unit(in,actor);from=&actor;}if(in.u32())to=b.data();}
        else if(op==C3X_NATIVE_TACTICAL_GRID){if(in.u32()){frame(in,scene);from=&scene.value;}}
        else if(op==C3X_NATIVE_TACTICAL_ROUTE_BEGIN){if(in.u32()){in(view.tile_width);in(view.native_tile_width);in(view.translate_x_fp);in(view.translate_y_fp);from=&view;}}
        else if(op==C3X_NATIVE_LOOKUP||op==C3X_NATIVE_SPRITE_LOOKUP||op==C3X_NATIVE_SPRITE_LOOKUP_OVER||op==C3X_NATIVE_SPRITE_LOOKUP_SCALED){if(in.u32()){lookup.palette=object(in.u32());in(lookup.percent);lookup.background=object(in.u32());for(auto& x:lookup.scale)in(x);words(in,lookup.table);from=&lookup;}rect(in,to,b.data());}
        else if(op==C3X_NATIVE_SPRITE_STYLE){if(in.u32()){style.palette=object(in.u32());in(style.color);in(style.mode);in(style.opacity);words(in,style.table);from=&style;}rect(in,to,b.data());}
        else if(op==C3X_NATIVE_SPRITE_BLEND){if(in.u32()){blend.alpha=object(in.u32());blend.background=object(in.u32());blend.palette=object(in.u32());from=&blend;}rect(in,to,b.data());}
        else if(op==C3X_NATIVE_SPRITE){from=object(in.u32());rect(in,to,b.data());}
        else if(op==C3X_NATIVE_COPY||op==C3X_NATIVE_FILL||op==C3X_NATIVE_IMAGE_DRAW||op==C3X_NATIVE_IMAGE_PRESENT||op==C3X_NATIVE_LINE||op==C3X_NATIVE_TINT||op==C3X_NATIVE_TACTICAL_RING||op==C3X_NATIVE_TACTICAL_TARGET||op==C3X_NATIVE_STROKE){rect(in,from,a.data(),op==C3X_NATIVE_TINT?1:op==C3X_NATIVE_TACTICAL_TARGET?2:op==C3X_NATIVE_STROKE?7:4);rect(in,to,b.data());}
    }
};
}
