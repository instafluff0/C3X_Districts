#pragma once
#include "coastal_waves.h"
#include <array>
#include <cstring>

namespace c3x_renderer { namespace render_core {
// Recover the existing ribbon's grid without changing a single triangle or
// vertex channel. Keep a literal fallback if a future producer has different
// topology or supplies unequal copies of what used to be a shared grid point.
struct IndexedWaveRibbon {
    std::vector<WavePoint> vertices;
    std::vector<unsigned> indices;
    bool empty()const{return indices.empty();}
};
inline IndexedWaveRibbon indexed_wave_ribbon(std::vector<WavePoint> const& source){
    constexpr unsigned rows=32,columns=6,points=(rows+1)*(columns+1),elements=rows*columns*6;
    IndexedWaveRibbon result;
    auto literal=[&](){
        result.vertices=source;result.indices.resize(source.size());
        for(unsigned i=0;i<result.indices.size();++i)result.indices[i]=i;
    };
    if(source.size()!=elements){literal();return result;}
    auto equal=[](WavePoint const& a,WavePoint const& b){
        // Compare fields individually; WavePoint has implementation padding.
        return !std::memcmp(&a.position.x,&b.position.x,sizeof(a.position.x)) &&
            !std::memcmp(&a.position.y,&b.position.y,sizeof(a.position.y)) &&
            !std::memcmp(&a.distance,&b.distance,sizeof(a.distance)) &&
            !std::memcmp(&a.along,&b.along,sizeof(a.along)) &&
            !std::memcmp(&a.coverage,&b.coverage,sizeof(a.coverage));
    };
    result.vertices.resize(points);result.indices.reserve(elements);
    std::array<bool,points> assigned{};unsigned cursor=0;
    for(unsigned y=0;y<rows;++y)for(unsigned x=0;x<columns;++x){
        unsigned a=y*(columns+1)+x,b=a+1,d=a+columns+1,e=d+1;
        for(auto index:{a,b,e,a,e,d}){
            auto const& point=source[cursor++];
            if(assigned[index] && !equal(result.vertices[index],point)){literal();return result;}
            if(!assigned[index]){result.vertices[index]=point;assigned[index]=true;}
            result.indices.push_back(index);
        }
    }
    return result;
}
}}
