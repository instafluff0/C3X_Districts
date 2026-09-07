#pragma once
#include <array>
#include <cstdint>

namespace c3x_renderer {
// Ordered rounding after display transfer, solely for a 16-bit destination.
// Full-color rendering/caches remain untouched. An 8-pixel period divides both
// Civ III tile bases, preserving the pattern across fixed tile scroll jumps.
constexpr unsigned color_threshold(unsigned x,unsigned y) {
    unsigned value=0;
    for(unsigned bit=0;bit<3;++bit) {
        unsigned a=(x>>bit)&1u,b=(y>>bit)&1u;
        value=(value<<2)|((a^b)<<1)|b;
    }
    return value;
}
using ColorRoundingTable=std::array<std::array<std::uint8_t,256>,64>;
inline ColorRoundingTable color_rounding_table(unsigned levels) {
    ColorRoundingTable table{};
    for(unsigned t=0;t<64;++t)for(unsigned v=0;v<256;++v) {
        unsigned scaled=v*levels, q=scaled/255, residue=scaled%255;
        if(residue*128>(t*2+1)*255)++q;
        table[t][v]=std::uint8_t((q*255+levels/2)/levels);
    }
    return table;
}
}
