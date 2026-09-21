#pragma once
#include <map>
#include <memory>
// Only animated damage needs a static underlay. Keep original-format MSAA
// samples in small tiles, rather than allocating another whole-view target.
namespace c3x_renderer { namespace render_core {
struct LinearBackup {
    struct Tile {std::unique_ptr<LinearTarget> target;bool used=false;};
    std::map<std::pair<LONG,LONG>,Tile> tiles;
    LinearRestore program;
    ID3D11Device* device=nullptr; // borrowed from the enclosing renderer
    UINT width=0,height=0;
    void reset(){tiles.clear();program.reset();device=nullptr;width=height=0;}
    bool ensure(ID3D11Device* d,UINT w,UINT h){
        if(device!=d||width!=w||height!=h){reset();device=d;width=w;height=h;}
        return w&&h&&!(w%2)&&!(h%2)&&program.ensure(d);
    }
    template<class Visit>bool visit(D3D11_RECT region,Visit callback){
        region={std::max<LONG>(0,region.left),std::max<LONG>(0,region.top),
            std::min<LONG>(LONG(width/2),region.right),std::min<LONG>(LONG(height/2),region.bottom)};
        for(LONG y=region.top/128*128;y<region.bottom;y+=128)
            for(LONG x=region.left/128*128;x<region.right;x+=128){
                D3D11_RECT part={std::max(x,region.left),std::max(y,region.top),std::min(x+128,region.right),std::min(y+128,region.bottom)};
                if(part.left<part.right&&part.top<part.bottom&&!callback(x,y,part))return false;
            }
        return true;
    }
    bool capture(ID3D11DeviceContext* context,LinearTarget const& source,std::vector<D3D11_RECT> const& regions){
        if(source.width!=width||source.height!=height||!device)return false;
        for(auto& tile:tiles)tile.second.used=false;
        for(auto region:regions)if(!visit(region,[&](LONG x,LONG y,D3D11_RECT part){
            auto& tile=tiles[{x,y}];if(!tile.target)tile.target=std::make_unique<LinearTarget>();
            UINT w=std::min<UINT>(256,width-UINT(x)*2),h=std::min<UINT>(256,height-UINT(y)*2);
            if(!tile.target->ensure(device,w,h,true,false))return false;
            tile.used=true;std::vector<D3D11_RECT> local={{part.left-x,part.top-y,part.right-x,part.bottom-y}};
            return program.draw(context,*tile.target,source.samples,source.depth_samples,-x,-y,{},&local,width,height);
        }))return false;
        // Previous damage was restored before this capture. Tiles outside the
        // new damage have no remaining reader; do not accumulate old views.
        for(auto it=tiles.begin();it!=tiles.end();)if(!it->second.used)it=tiles.erase(it);else ++it;
        return true;
    }
    bool restore(ID3D11DeviceContext* context,LinearTarget const& target,std::vector<D3D11_RECT> const& regions,bool clear=false){
        if(target.width!=width||target.height!=height||!device)return false;
        for(auto region:regions){
            if(clear){std::vector<D3D11_RECT> one={region};if(!program.draw(context,target,nullptr,nullptr,0,0,one,&one))return false;}
            else if(!visit(region,[&](LONG x,LONG y,D3D11_RECT part){
                auto it=tiles.find({x,y});if(it==tiles.end()||!it->second.target)return false;
                auto& source=*it->second.target;std::vector<D3D11_RECT> one={part};
                return program.draw(context,target,source.samples,source.depth_samples,x,y,{},&one,source.width,source.height);
            }))return false;
        }
        return true;
    }
    std::size_t bytes()const{std::size_t total=0;for(auto const& tile:tiles)if(tile.second.target)total+=tile.second.target->bytes();return total;}
};
}}
