#pragma once
#include "codec.h"
#include "../asset_content_hash.h"
#include <array>
namespace c3x_inputs {
// Native screen inputs only. Keep block fingerprints in the game, not another
// full-screen shadow. Every delta names the previous content and resulting
// content; omitted, reordered or changed patches fail before production use.
using PixelHash=std::array<std::uint32_t,4>;
inline PixelHash pixel_hash(std::vector<unsigned short> const& pixels){
    return c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(pixels.data()),pixels.size()*2);
}
struct ScreenInput {
    static constexpr unsigned block_words=8192,limit=2240u*1260u;
    std::vector<PixelHash> blocks;PixelHash previous={};unsigned previous_count=0;
    void encode(Writer& out,std::vector<unsigned short> const& pixels){
        require(pixels.size()<=limit,"native screen input limit");auto count=unsigned(pixels.size());out(count);
        for(auto word:previous)out(word);auto after=pixel_hash(pixels);for(auto word:after)out(word);
        std::vector<PixelHash> next;std::vector<unsigned> changed;
        for(unsigned first=0;first<count;first+=block_words){auto n=std::min(block_words,count-first);
            auto hash=c3x_renderer::asset_content_hash(reinterpret_cast<unsigned char const*>(pixels.data()+first),std::size_t(n)*2);
            auto block=unsigned(next.size());next.push_back(hash);
            if(count!=previous_count||block>=blocks.size()||hash!=blocks[block])changed.push_back(block);
        }
        out.u32(unsigned(changed.size()));for(auto block:changed){out(block);auto first=block*block_words,n=std::min(block_words,count-first);
            out.reserve(std::size_t(n)*2);for(unsigned i=first;i<first+n;++i){out.bytes.push_back(static_cast<unsigned char>(pixels[i]));out.bytes.push_back(static_cast<unsigned char>(pixels[i]>>8));}}
        blocks.swap(next);previous=after;previous_count=count;
    }
};
struct ScreenReplay {
    std::vector<unsigned short> pixels;PixelHash previous={};
    void decode(Reader& in){
        auto count=in.u32();require(count<=ScreenInput::limit,"native screen replay limit");PixelHash before,after;
        for(auto& word:before)word=in.u32();for(auto& word:after)word=in.u32();require(before==previous,"native screen delta predecessor mismatch");
        auto blocks=(count+ScreenInput::block_words-1)/ScreenInput::block_words,changed=in.u32();require(changed<=blocks,"native screen delta count");
        if(count!=pixels.size())require(changed==blocks,"native screen resize requires all pixels");pixels.resize(count);
        unsigned last=0;for(unsigned i=0;i<changed;++i){auto block=in.u32();require(block<blocks&&(!i||block>last),"native screen delta order");last=block;
            auto first=block*ScreenInput::block_words,n=std::min(ScreenInput::block_words,count-first);in.available(std::size_t(n)*2);
            for(unsigned j=first;j<first+n;++j){pixels[j]=static_cast<unsigned short>(unsigned(in.bytes[in.at])|(unsigned(in.bytes[in.at+1])<<8));in.at+=2;}}
        require(pixel_hash(pixels)==after,"native screen delta content mismatch");previous=after;
    }
};
}
