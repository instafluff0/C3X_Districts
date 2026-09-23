#pragma once
#include "input_recording/codec.h"

namespace c3x_remote_scene {
// This is the process-boundary form of the complete map result. No ABI pointer,
// allocation address or x64 object lifetime crosses into the x86 bridge.
template<class IO,class Value>void output_fields(IO& io,Value& value){
    io(value.width);io(value.height);io(value.stride_bytes);
    io(value.clip_left);io(value.clip_top);io(value.clip_right);io(value.clip_bottom);
    io(value.rendered_tile_count);io(value.fallback_tile_count);
    io(value.visible_animation_count);io(value.request_continuous_redraw);io(value.renderer_cpu_ticks);
    io(value.textured_tile_count);io(value.replacement_tile_count);io(value.frame_invalidation_flags);
    io(value.cache_hits);io(value.cache_misses);io(value.cache_evictions);io(value.cache_stale_rejections);
    io(value.cache_entries);io(value.cache_capacity);io(value.device_generation);io(value.device_recoveries);
    io(value.content_revision);io(value.geometry_tiles_built);io(value.geometry_tiles_reused);
    io(value.geometry_tiles_evicted);io(value.geometry_cache_bytes);io(value.geometry_upload_bytes);
    io(value.geometry_ticks);io(value.draw_ticks);io(value.readback_ticks);
    io(value.raster_reused_pixels);io(value.raster_draw_pixels);
    io(value.prefetch_tiles_pending);io(value.prefetch_tiles_built);
    io(value.prefetch_tiles_unavailable);io(value.prefetch_tiles_cancelled);
    io(value.prefetch_cache_bytes);io(value.prefetch_ticks);io(value.raster_cached_pixels);
    io(value.prefetch_blocks_pending);io(value.prefetch_blocks_built);io(value.pixel_block_cache_bytes);
}
template<class IO,class Value>void gpu_fields(IO& io,Value& value){
    io(value.ticket);io(value.map_image);io(value.width);io(value.height);
    io(value.device_generation);io(value.map_readbacks);io(value.content_revision);
    io(value.session);io(value.prepared);io(value.presentation_time_ticks);
}
struct Output {
    c3x_renderer_output_v1 value={C3X_RENDERER_API_VERSION,sizeof(value)};
    c3x_renderer_gpu_frame_v1 gpu={sizeof(gpu)};
    std::vector<unsigned> fallbacks,replacements,pixels;
    void bind(){
        value.fallback_tile_count=unsigned(fallbacks.size());
        value.fallback_tile_indices=fallbacks.empty()?nullptr:fallbacks.data();
        value.replacement_tile_count=unsigned(replacements.size());
        value.replacement_tile_flags=replacements.empty()?nullptr:replacements.data();
        value.bgra_pixels=pixels.empty()?nullptr:pixels.data();
    }
};
inline void encode(c3x_inputs::Writer& out,c3x_renderer_gpu_frame_v1 const& gpu,
                   c3x_renderer_output_v1 const& source){
    c3x_inputs::require(source.api_version==C3X_RENDERER_API_VERSION&&source.struct_size==sizeof(source)&&
        gpu.struct_size==sizeof(gpu),"remote scene output ABI mismatch");
    c3x_inputs::require(source.width>=0&&source.height>=0&&source.width<=2240&&source.height<=1260&&
        source.fallback_tile_count<=8192&&source.replacement_tile_count<=8192&&
        (!source.fallback_tile_count||source.fallback_tile_indices)&&
        (!source.replacement_tile_count||source.replacement_tile_flags),"remote scene output bounds");
    auto value=source;auto frame=gpu;gpu_fields(out,frame);output_fields(out,value);
    for(unsigned n=0;n<source.fallback_tile_count;++n)out(source.fallback_tile_indices[n]);
    for(unsigned n=0;n<source.replacement_tile_count;++n)out(source.replacement_tile_flags[n]);
    unsigned pixel_count=source.bgra_pixels?unsigned(source.width)*unsigned(source.height):0;
    c3x_inputs::require(!pixel_count||source.stride_bytes==source.width*4,"remote scene output stride");
    out(pixel_count);
    auto pixels=static_cast<unsigned const*>(source.bgra_pixels);
    for(unsigned n=0;n<pixel_count;++n)out(pixels[n]);
}
inline void decode(c3x_inputs::Reader& in,Output& target){
    target={};target.value.api_version=C3X_RENDERER_API_VERSION;
    target.value.struct_size=sizeof(target.value);target.gpu.struct_size=sizeof(target.gpu);
    gpu_fields(in,target.gpu);output_fields(in,target.value);
    c3x_inputs::require(target.value.width>=0&&target.value.height>=0&&
        target.value.width<=2240&&target.value.height<=1260&&
        target.value.fallback_tile_count<=8192&&target.value.replacement_tile_count<=8192,
        "remote scene result bounds");
    target.fallbacks.resize(target.value.fallback_tile_count);
    for(auto& item:target.fallbacks)in(item);
    target.replacements.resize(target.value.replacement_tile_count);
    for(auto& item:target.replacements)in(item);
    auto pixel_count=in.u32();
    c3x_inputs::require(pixel_count==0||
        (pixel_count==unsigned(target.value.width)*unsigned(target.value.height)&&
         target.value.stride_bytes==target.value.width*4),"remote scene result pixel extent");
    target.pixels.resize(pixel_count);for(auto& item:target.pixels)in(item);
    in.done();target.bind();
}
}
