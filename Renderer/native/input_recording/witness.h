#pragma once
#include "codec.h"
#include "../asset_content_hash.h"
namespace c3x_inputs {
// Output witnesses are compared after production work. They are never decoded
// into scene/image input and omit execution-dependent cache/timing counters.
inline void output_witness(Writer& out,c3x_renderer_output_v1 const& value,int result){
    out.u32(result==C3X_RENDERER_RESULT_OK?1:0);if(result!=C3X_RENDERER_RESULT_OK)return;
    out(value.width);out(value.height);out(value.clip_left);out(value.clip_top);out(value.clip_right);out(value.clip_bottom);
    out(value.rendered_tile_count);out(value.fallback_tile_count);out(value.visible_animation_count);out(value.replacement_tile_count);
    require(value.replacement_tile_count<=8192&&(!value.replacement_tile_count||value.replacement_tile_flags),"invalid output ownership witness");
    for(unsigned n=0;n<value.replacement_tile_count;++n)out(value.replacement_tile_flags[n]);
    bool pixels=value.bgra_pixels!=nullptr;out.u32(pixels?1:0);
    if(pixels){require(value.width>0&&value.height>0&&value.width<=8192&&value.height<=8192&&value.stride_bytes==value.width*4,"invalid output pixel witness");
        auto hash=c3x_renderer::asset_content_hash(static_cast<unsigned char const*>(value.bgra_pixels),std::size_t(value.width)*unsigned(value.height)*4);for(auto part:hash)out(part);}
}
inline void check_output(Reader& expected,c3x_renderer_output_v1 const& value,int result){
    Writer actual;output_witness(actual,value,result);expected.available(actual.bytes.size());
    Reader generated{actual.bytes};auto recorded=expected;
    char const* names[]={"result","width","height","clip_left","clip_top","clip_right","clip_bottom",
        "rendered_tile_count","fallback_tile_count","visible_animation_count","replacement_tile_count"};
    char audit[4]={};bool pixel_audit=GetEnvironmentVariableA("C3X_MAP_PIXEL_AUDIT",audit,sizeof(audit))==1&&audit[0]=='1';
    bool pixel_drift=false;
    while(generated.at<actual.bytes.size()){
        auto word=generated.at/4;auto wanted=recorded.u32(),got=generated.u32();
        bool pixel_hash=value.bgra_pixels&&word>=actual.bytes.size()/4-4;
        if(wanted!=got&&pixel_audit&&pixel_hash)pixel_drift=true;
        bool implementation_count=pixel_audit&&(word==7||word==9);
        if(wanted!=got&&!implementation_count&&!((realtime_replay().enabled||pixel_audit)&&pixel_hash))throw std::runtime_error(std::string("replay map output differs: ")+
            (word<11?names[word]:"ownership/pixel witness")+" word="+std::to_string(word)+
            " expected="+std::to_string(wanted)+" actual="+std::to_string(got));
    }
    if(pixel_drift)++replay_execution().map_pixel_mismatches;
    expected.at+=actual.bytes.size();
}
inline void adoption_witness(Writer& out,c3x_renderer_camera_view_v1 const& value,int result){
    output_witness(out,value.output,result);if(result!=C3X_RENDERER_RESULT_OK)return;
    c3x_renderer_camera_identity_v1_fields(out,value.identity);
    Writer owned;frame(owned,value.frame);
    auto hash=c3x_renderer::asset_content_hash(owned.bytes.data(),owned.bytes.size());for(auto part:hash)out(part);
}
inline void check_adoption(Reader& expected,c3x_renderer_camera_view_v1 const& value,int result){
    char audit[4]={};bool pixel_audit=GetEnvironmentVariableA("C3X_MAP_PIXEL_AUDIT",audit,sizeof(audit))==1&&audit[0]=='1';
    if(realtime_replay().enabled||pixel_audit){check_output(expected,value.output,result);if(result==C3X_RENDERER_RESULT_OK){
        Writer identity;c3x_renderer_camera_identity_v1_fields(identity,value.identity);expected.available(identity.bytes.size()+16);
        require(std::equal(identity.bytes.begin(),identity.bytes.end(),expected.bytes.begin()+expected.at),"realtime adopted identity differs");expected.at+=identity.bytes.size()+16;}return;}
    Writer actual;adoption_witness(actual,value,result);expected.available(actual.bytes.size());
    require(std::equal(actual.bytes.begin(),actual.bytes.end(),expected.bytes.begin()+expected.at),"replay adopted camera differs");expected.at+=actual.bytes.size();
}

inline void gpu_witness(Writer& out,c3x_renderer_gpu_frame_v1 const& image,c3x_renderer_output_v1 const& metadata,int result){
    out.u32(result==C3X_RENDERER_RESULT_OK?1:0);if(result!=C3X_RENDERER_RESULT_OK)return;
    out(image.width);out(image.height);out(image.presentation_time_ticks);output_witness(out,metadata,result);
}
inline void check_gpu(Reader& expected,c3x_renderer_gpu_frame_v1 const& image,c3x_renderer_output_v1 const& metadata,int result){
    auto good=expected.u32();require(good==(result==C3X_RENDERER_RESULT_OK?1u:0u),"GPU adoption result differs");if(!good)return;
    int width=0,height=0;expected(width);expected(height);auto ticks=std::int64_t(expected.u64());
    require(width==image.width&&height==image.height,"GPU adoption extent differs");
    if(!realtime_replay().enabled&&ticks!=image.presentation_time_ticks)throw std::runtime_error("GPU adopted sample differs: expected="+std::to_string(ticks)+" actual="+std::to_string(image.presentation_time_ticks));
    check_output(expected,metadata,result);
}

}
