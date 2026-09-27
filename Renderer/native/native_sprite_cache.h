#pragma once
#include "gpu_image_commands.h"
#include <array>
#include <cstdint>
#include <vector>

namespace c3x_native_images {
// Content identifies a decoded source, including its current palette, style and
// dimensions. Native pointers never prove freshness. Eviction goes through the
// backend's ordered destroy, after every draw that used the old image.
template<class Backend, unsigned Capacity=256> class SpriteCache {
    struct Entry {
        c3x_gpu_images::Id image=0;
        unsigned width=0,height=0;
        std::uint64_t hash=0,age=0;
        std::vector<unsigned> words;
    };
    std::array<Entry,Capacity> entries{};
    std::uint64_t clock=0;
    std::size_t resident=0;
    static constexpr std::size_t limit=16u*1024u*1024u;
    void retire(Backend& gpu,Entry& entry){
        if(entry.image)gpu.destroy(entry.image);
        resident-=entry.words.size()*sizeof(unsigned);entry={};
    }
public:
    std::uint64_t hits=0,uploads=0,uploaded_bytes=0;
    std::size_t bytes()const{return resident;}
    c3x_gpu_images::Id select(Backend& gpu,std::vector<unsigned>& words,unsigned width,unsigned height){
        if(!width||!height||words.size()!=std::size_t(width)*height||words.size()*sizeof(unsigned)>limit)return 0;
        std::uint64_t hash=14695981039346656037ull;
        for(auto word:words){hash^=word;hash*=1099511628211ull;}
        Entry* chosen=nullptr;
        for(auto& entry:entries){
            if(entry.image&&entry.width==width&&entry.height==height&&entry.hash==hash&&entry.words==words){
                entry.age=++clock;++hits;return entry.image;
            }
            if(!chosen||entry.age<chosen->age)chosen=&entry;
        }
        retire(gpu,*chosen);
        auto size=words.size()*sizeof(unsigned);
        while(resident>limit-size){
            Entry* oldest=nullptr;
            for(auto& entry:entries)if(entry.image&&(!oldest||entry.age<oldest->age))oldest=&entry;
            retire(gpu,*oldest);
        }
        auto image=gpu.create(width,height,c3x_gpu_images::Format::bgra32);
        if(!image)return 0;
        if(!gpu.upload(image,1,words.data(),words.size())){gpu.destroy(image);return 0;}
        chosen->image=image;chosen->width=width;chosen->height=height;
        chosen->hash=hash;chosen->age=++clock;chosen->words=std::move(words);
        resident+=size;++uploads;uploaded_bytes+=size;return image;
    }
    void clear(Backend& gpu){for(auto& entry:entries)retire(gpu,entry);}
    void abandon(){entries={};resident=0;}
};
}
