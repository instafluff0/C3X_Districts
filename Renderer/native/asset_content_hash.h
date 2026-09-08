#pragma once
#include <array>
#include <cstdint>
#include <cstring>

namespace c3x_renderer {
// MurmurHash3 x86_128, Austin Appleby's public-domain algorithm:
// https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
// Alignment-safe block loads and bounded tail assembly. Cache identity only;
// provenance/security verification continues to use SHA-256 independently.
inline std::array<std::uint32_t,4> asset_content_hash(
    unsigned char const* data,std::size_t size,std::uint32_t seed=0) {
    auto rot=[](std::uint32_t x,unsigned r){return (x<<r)|(x>>(32-r));};
    auto finish=[](std::uint32_t x){x^=x>>16;x*=0x85ebca6b;x^=x>>13;x*=0xc2b2ae35;x^=x>>16;return x;};
    constexpr std::uint32_t c1=0x239b961b,c2=0xab0e9789,c3=0x38b34ae5,c4=0xa1e38b93;
    std::uint32_t h1=seed,h2=seed,h3=seed,h4=seed;
    std::size_t at=0;
    for(;size-at>=16;at+=16){
        std::uint32_t k[4];std::memcpy(k,data+at,16);
        h1^=rot(k[0]*c1,15)*c2;h1=rot(h1,19)+h2;h1=h1*5+0x561ccd1b;
        h2^=rot(k[1]*c2,16)*c3;h2=rot(h2,17)+h3;h2=h2*5+0x0bcaa747;
        h3^=rot(k[2]*c3,17)*c4;h3=rot(h3,15)+h4;h3=h3*5+0x96cd1c35;
        h4^=rot(k[3]*c4,18)*c1;h4=rot(h4,13)+h1;h4=h4*5+0x32ac3b17;
    }
    std::uint32_t k[4]={};auto tail=size-at;
    for(std::size_t i=0;i<tail;i++)k[i/4]|=std::uint32_t(data[at+i])<<((i%4)*8);
    if(tail>12)h4^=rot(k[3]*c4,18)*c1;
    if(tail>8)h3^=rot(k[2]*c3,17)*c4;
    if(tail>4)h2^=rot(k[1]*c2,16)*c3;
    if(tail>0)h1^=rot(k[0]*c1,15)*c2;
    auto length=std::uint32_t(size);
    h1^=length;h2^=length;h3^=length;h4^=length;
    h1+=h2+h3+h4;h2+=h1;h3+=h1;h4+=h1;
    h1=finish(h1);h2=finish(h2);h3=finish(h3);h4=finish(h4);
    h1+=h2+h3+h4;h2+=h1;h3+=h1;h4+=h1;
    return {h1,h2,h3,h4};
}
}
