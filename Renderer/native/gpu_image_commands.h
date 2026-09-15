#pragma once
#include <cstdint>
namespace c3x_gpu_images {
using Id=std::uint64_t;
enum class Format { rgb555, rgb565, bgra32 };
enum class Kind { copy, fill, color_key, invert, quantize, expand, native_sprite, unit_over, native_text, native_image, native_blend, native_lookup };
struct Rect { int left,top,right,bottom; };
struct Command {Kind kind;Id destination,source;Rect area,clip;int source_x=0,source_y=0;std::uint32_t color=0;Id background=0,detail=0,background_detail=0;int source_width=0,source_height=0;Id program=0;};
}
