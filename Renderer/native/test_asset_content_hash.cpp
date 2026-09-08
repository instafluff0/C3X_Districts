#include "asset_content_hash.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
int main(){
    using c3x_renderer::asset_content_hash;
    // SMHasher's published x86_128 verification vector, including every tail.
    unsigned char key[256]={},digests[4096]={};
    for(unsigned i=0;i<256;i++){
        key[i]=static_cast<unsigned char>(i);
        auto h=asset_content_hash(key,i,256-i);std::memcpy(digests+i*16,h.data(),16);
    }
    assert(asset_content_hash(digests,sizeof(digests))[0]==0xb3ece62a);
    // Every input byte participates, including unaligned tails and zero suffixes.
    for(unsigned length=0;length<=256;length++){
        auto expected=asset_content_hash(key,length);
        std::vector<unsigned char> unaligned(length+17,0);
        for(unsigned offset=1;offset<16;offset++){
            std::memcpy(unaligned.data()+offset,key,length);
            assert(asset_content_hash(unaligned.data()+offset,length)==expected);
        }
        for(unsigned i=0;i<length;i++){
            key[i]^=1;assert(asset_content_hash(key,length)!=expected);key[i]^=1;
        }
        if(length){auto with_zero=std::vector<unsigned char>(key,key+length);with_zero.push_back(0);
            assert(asset_content_hash(with_zero.data(),with_zero.size())!=expected);}
    }
    std::cout<<"PASS asset hash: upstream verification, every byte, all tails and unaligned inputs\n";
    std::vector<unsigned char> payload(8u*1024u*1024u);
    for(std::size_t i=0;i<payload.size();i++)payload[i]=static_cast<unsigned char>(i*73);
    volatile std::uint64_t sink=0;
    auto before=std::chrono::steady_clock::now();
    for(unsigned run=0;run<3;run++){
        std::uint64_t h=1469598103934665603ull;
        for(auto byte:payload){h^=byte;h*=1099511628211ull;}
        sink=h;
    }
    auto middle=std::chrono::steady_clock::now();
    for(unsigned run=0;run<3;run++)sink=asset_content_hash(payload.data(),payload.size())[0];
    auto after=std::chrono::steady_clock::now();
    std::cout<<"HASH benchmark bytes="<<payload.size()*3<<" old_ms="
             <<std::chrono::duration<double,std::milli>(middle-before).count()<<" new_ms="
             <<std::chrono::duration<double,std::milli>(after-middle).count()<<" sink="<<sink<<"\n";
}
