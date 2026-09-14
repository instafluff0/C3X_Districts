#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

namespace c3x_renderer { namespace render_core {
// Persistent normalized isometric bounds. Residency owns membership; a view
// supplies a separate eligibility map. Entries never authorize an unseen draw.
class WorldPassIndex {
public:
    using Key=std::uintptr_t;
    using Cell=std::pair<int,int>;
    static constexpr std::size_t budget=16u*1024u*1024u;
private:
    struct Entry {std::uint64_t owner;std::vector<Cell> cells;};
    std::map<Cell,std::vector<Key>> cells;
    std::unordered_map<Key,Entry> entries;
    std::unordered_map<std::uint64_t,std::vector<Key>> owners;
    std::size_t bytes_=0;
public:
    void clear(){cells.clear();entries.clear();owners.clear();bytes_=0;}
    bool contains(Key key)const{return entries.find(key)!=entries.end();}
    std::size_t bytes()const{return bytes_;}
    bool add(std::uint64_t owner,Key key,double left,double top,double right,double bottom){
        if(contains(key))return true;
        if(!std::isfinite(left+top+right+bottom) || left>right || top>bottom ||
            std::abs(left)>1e8 || std::abs(right)>1e8 || std::abs(top)>1e8 || std::abs(bottom)>1e8)return false;
        int x0=int(std::floor(left)),y0=int(std::floor(top)),x1=int(std::floor(right)),y1=int(std::floor(bottom));
        auto count=std::uint64_t(x1-x0+1)*std::uint64_t(y1-y0+1);
        // Conservative charge includes nodes, hash buckets, spare vector
        // capacity and owner membership. No allocation grows without admission.
        auto charge=256+count*160;
        if(count>4096 || charge>budget-bytes_)return false;
        try {
            Entry entry{owner,{}};entry.cells.reserve(std::size_t(count));
            for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x)entry.cells.push_back({x,y});
            entries.emplace(key,std::move(entry));owners[owner].push_back(key);
            for(auto cell:entries.at(key).cells)cells[cell].push_back(key);
            bytes_+=std::size_t(charge);return true;
        }catch(...){clear();return false;} // Views fall back to ordinary bounds selection.
    }
    void erase(std::uint64_t owner){
        auto found=owners.find(owner);if(found==owners.end())return;
        for(auto key:found->second){auto entry=entries.find(key);if(entry==entries.end())continue;
            for(auto cell:entry->second.cells){auto where=cells.find(cell);if(where==cells.end())continue;
                auto& values=where->second;values.erase(std::remove(values.begin(),values.end(),key),values.end());
                if(values.empty())cells.erase(where);
            }
            bytes_-=256+entry->second.cells.size()*160;entries.erase(entry);
        }
        owners.erase(found);
    }
    bool query(double left,double top,double right,double bottom,std::vector<Key>& out)const{
        out.clear();
        if(!std::isfinite(left+top+right+bottom) || left>right || top>bottom ||
            std::abs(left)>1e8 || std::abs(right)>1e8 || std::abs(top)>1e8 || std::abs(bottom)>1e8)return false;
        int x0=int(std::floor(left)),y0=int(std::floor(top)),x1=int(std::floor(right)),y1=int(std::floor(bottom));
        if(std::uint64_t(x1-x0+1)*std::uint64_t(y1-y0+1)>65536)return false;
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){auto found=cells.find({x,y});
            if(found==cells.end())continue;
            if(out.size()+found->second.size()>budget/sizeof(Key))return false;
            out.insert(out.end(),found->second.begin(),found->second.end());
        }
        std::sort(out.begin(),out.end());out.erase(std::unique(out.begin(),out.end()),out.end());return true;
    }
};
} }
