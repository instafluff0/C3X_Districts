#pragma once
#include <cstdint>
namespace c3x_gpu_images {
using Id=std::uint64_t;
enum class Format { rgb555, rgb565, bgra32 };
enum class Kind { copy, fill, color_key, invert, quantize };
struct Rect { int left,top,right,bottom; };
struct Command {Kind kind;Id destination,source;Rect area,clip;int source_x=0,source_y=0;std::uint32_t color=0;};
}
