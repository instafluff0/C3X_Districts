#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Candidate selection only: callers still apply their exact intersection test.
// Indices borrow the assembled geometry's ordering, never own scene/GPU pointers.
struct RegionContributorIndex {
    using Item=std::pair<unsigned,unsigned>;
    using Cell=std::pair<int,int>;
    std::array<std::map<Cell,std::vector<Item>>,2> cells;
    static constexpr std::size_t budget=16u*1024u*1024u;
    static constexpr std::size_t cell_limit=4096;
    std::size_t bytes=0,count=0;
    bool ready=false;
    int tile_width=0;
    float reflection_height=0;
    void clear(){for(auto& pass:cells)pass.clear();bytes=count=0;ready=false;}
    bool add(unsigned pass,Item item,double left,double top,double right,double bottom){
        if(pass>=2 || !std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) || !std::isfinite(bottom) ||
           left>right || top>bottom || std::abs(left)>1e8 || std::abs(top)>1e8 || std::abs(right)>1e8 || std::abs(bottom)>1e8)return false;
        int x0=int(std::floor(left/128)),y0=int(std::floor(top/128));
        int x1=int(std::floor(right/128)),y1=int(std::floor(bottom/128));
        if(std::uint64_t(x1-x0+1)*std::uint64_t(y1-y0+1)>cell_limit)return false;
        // Inclusive extra edge cells are conservative, including negative origins.
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){
            auto found=cells[pass].find({x,y});
            if(found==cells[pass].end()){
                if(count>=cell_limit || bytes>budget-128)return false;
                found=cells[pass].emplace(Cell{x,y},std::vector<Item>{}).first;
                ++count;bytes+=128; // Map node, key, vector and allocator allowance.
            }
            auto& items=found->second;
            if(items.size()==items.capacity()){
                auto capacity=std::max<std::size_t>(4,items.capacity()*2);
                auto growth=(capacity-items.capacity())*sizeof(Item);
                if(growth>budget-bytes)return false;
                auto old=items.capacity();items.reserve(capacity);
                bytes+=(items.capacity()-old)*sizeof(Item);
                if(bytes>budget)return false;
            }
            items.push_back(item);
        }
        return true;
    }
    bool query(unsigned pass,int x,int y,int extent,std::vector<Item>& output)const {
        output.clear();if(!ready || pass>=2 || extent<0 || extent>512)return false;
        int x0=int(std::floor(double(x)/128)),y0=int(std::floor(double(y)/128));
        int x1=int(std::floor((double(x)+extent)/128)),y1=int(std::floor((double(y)+extent)/128));
        for(int cy=y0;cy<=y1;++cy)for(int cx=x0;cx<=x1;++cx){
            auto found=cells[pass].find({cx,cy});if(found==cells[pass].end())continue;
            if(output.size()+found->second.size()>budget/(2*sizeof(Item)))return false;
            output.insert(output.end(),found->second.begin(),found->second.end());
        }
        std::sort(output.begin(),output.end());
        output.erase(std::unique(output.begin(),output.end()),output.end());
        return true;
    }
};
} }
