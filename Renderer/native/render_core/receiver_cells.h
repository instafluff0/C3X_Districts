#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Conservative coverage of many receiver rectangles on a coarse cell grid.
// A query passes when its rectangle shares a cell with any receiver, so every
// rectangle that intersects a receiver passes. The mirror uses it to keep only
// records whose reflections can reach a water receiver: on a coast-heavy view
// the union of all water spans the whole screen and keeps inland objects too.
struct ReceiverCells {
    int left=0,top=0,columns=0,rows=0,cell=32;
    std::vector<unsigned> sums; // (columns+1)*(rows+1) inclusive prefix sums
    void clear(){columns=rows=0;sums.clear();}
    // Bounds and receivers are closed integer rectangles {left,top,right,bottom}.
    void build(std::array<int,4> const& bounds,std::vector<std::array<int,4>> const& receivers,int cell_size=32){
        clear();cell=std::max(1,cell_size);left=bounds[0];top=bounds[1];
        if(bounds[2]<bounds[0] || bounds[3]<bounds[1])return;
        columns=(bounds[2]-bounds[0])/cell+1;rows=(bounds[3]-bounds[1])/cell+1;
        sums.assign(std::size_t(columns+1)*std::size_t(rows+1),0);
        auto at=[&](int x,int y)->unsigned&{return sums[std::size_t(y+1)*std::size_t(columns+1)+std::size_t(x+1)];};
        for(auto const& r:receivers){int c0,r0,c1,r1;if(!span(r,c0,r0,c1,r1))continue;
            for(int y=r0;y<=r1;++y)for(int x=c0;x<=c1;++x)at(x,y)=1;}
        for(int y=0;y<rows;++y)for(int x=0;x<columns;++x)
            at(x,y)+=at(x-1,y)+at(x,y-1)-at(x-1,y-1);
    }
    bool any(std::array<int,4> const& r)const{
        int c0,r0,c1,r1;if(!span(r,c0,r0,c1,r1))return false;
        auto at=[&](int x,int y){return sums[std::size_t(y+1)*std::size_t(columns+1)+std::size_t(x+1)];};
        return at(c1,r1)-at(c0-1,r1)-at(c1,r0-1)+at(c0-1,r0-1)!=0;
    }
private:
    bool span(std::array<int,4> const& r,int& c0,int& r0,int& c1,int& r1)const{
        if(!columns || !rows || r[2]<r[0] || r[3]<r[1])return false;
        auto floor_div=[&](long long value){return value>=0?value/cell:-((-value+cell-1)/cell);};
        long long x0=floor_div((long long)r[0]-left),x1=floor_div((long long)r[2]-left);
        long long y0=floor_div((long long)r[1]-top),y1=floor_div((long long)r[3]-top);
        if(x1<0 || y1<0 || x0>=columns || y0>=rows)return false;
        c0=int(std::max(0LL,x0));r0=int(std::max(0LL,y0));
        c1=int(std::min<long long>(columns-1,x1));r1=int(std::min<long long>(rows-1,y1));return true;
    }
};
}}
