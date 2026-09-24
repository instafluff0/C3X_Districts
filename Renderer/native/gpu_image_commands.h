#pragma once
#include <cstdint>
#include <algorithm>
namespace c3x_gpu_images {
using Id=std::uint64_t;
enum class Format { rgb555, rgb565, bgra32 };
enum class Kind { copy, fill, color_key, invert, quantize, expand, native_sprite, unit_over, native_text, native_image, native_blend, native_lookup };
struct Rect { int left,top,right,bottom; };
inline Rect intersection(Rect a,Rect b){return {std::max(a.left,b.left),std::max(a.top,b.top),std::min(a.right,b.right),std::min(a.bottom,b.bottom)};}
// Retained replay renders a command into a rectangle-local scratch image.
// Rebase a current screen-space occurrence by the same offset as its original
// command envelope; the immediate submission has an offset of zero.
inline Rect rebase_direct_rect(Rect occurrence,Rect original_envelope,Rect command_envelope){
    int dx=command_envelope.left-original_envelope.left,dy=command_envelope.top-original_envelope.top;
    return {occurrence.left+dx,occurrence.top+dy,occurrence.right+dx,occurrence.bottom+dy};
}
struct Command {Kind kind;Id destination,source;Rect area,clip;int source_x=0,source_y=0;std::uint32_t color=0;Id background=0,detail=0,background_detail=0;int source_width=0,source_height=0;Id program=0;};
}
