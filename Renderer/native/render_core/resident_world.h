#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "resident_content.h"

namespace c3x_renderer { namespace render_core {
// The explored world's draw records, by 8x8 world block (camera decoupling,
// stage 6). Records are in a fixed world pixel basis: a tile's occurrence
// anchor is its canonical coordinate times half a tile, so a camera step never
// changes them; only world edits do. Blocks are copy-on-write and a generation
// is the block map, so publishing or forking a view costs O(blocks), and an
// edit copies only the blocks it touches. Each tile keeps the owners of the
// meshes its records borrow; a generation keeps those owners alive.
//
// Queries visit the records of blocks meeting a rectangle in record space,
// with x-wrap turns, in the window's draw order: layer, then occurrence row,
// column and position within the tile (as `order_occurrences` sorted the
// window's flat arrays), then turn. Block culling is conservative; callers
// keep their exact intersection test.
template<class Record,std::size_t Layers> class ResidentWorld {
public:
    static constexpr int block=8;
    // A record's pass-0 extent in record space (bounds plus translation, plus
    // any lighting radius) and its world height range for mirrored extents.
    struct Extent {double left=0,top=0,right=0,bottom=0;float low_z=0,high_z=0;};
    // One flat record array per tile (two vectors per layer cost about
    // 1.9 KB per tile before any record): layer L holds [first[L], first[L+1]).
    struct Tile {
        int tile_x=0,tile_y=0;
        std::vector<Record> records;
        std::vector<Extent> extents; // parallel to records
        std::array<std::uint16_t,Layers+1> first{};
        std::vector<std::pair<ContentHandle,std::shared_ptr<void>>> owners;
        std::uint32_t flags=0;
        std::uint64_t content=0; // the caller's key of the tile content the records were built from
        // Appends to a layer; layers may arrive in any order.
        void add(unsigned layer,Record const& record,Extent const& extent){
            if(records.size()>=0xffff)throw std::length_error("resident world tile");
            auto at=std::ptrdiff_t(first[layer+1]);
            records.insert(records.begin()+at,record);extents.insert(extents.begin()+at,extent);
            for(unsigned l=layer+1;l<=Layers;++l)++first[l];
        }
        std::size_t begin_of(std::size_t layer)const{return first[layer];}
        std::size_t end_of(std::size_t layer)const{return first[layer+1];}
        std::size_t count(std::size_t layer)const{return std::size_t(first[layer+1]-first[layer]);}
        std::shared_ptr<void const> owner(ContentHandle handle)const{
            for(auto const& item:owners)if(item.first.slot==handle.slot && item.first.generation==handle.generation)return item.second;
            return nullptr;
        }
        std::size_t bytes()const{
            return sizeof(Tile)+owners.capacity()*sizeof(owners[0])+records.capacity()*sizeof(Record)+extents.capacity()*sizeof(Extent);
        }
    };
    using TilePtr=std::shared_ptr<Tile const>;
    struct Block {
        int x=0,y=0;
        std::vector<TilePtr> tiles; // occurrence order: row, then column
        Extent extent; // union of every record's extent; low_z/high_z as min/max
        bool empty=true;
        std::uint64_t revision=0;
    };
    using BlockPtr=std::shared_ptr<Block const>;
    using BlockKey=std::pair<int,int>; // (row, column) of the block, so map order is occurrence order
    struct Generation {
        std::map<BlockKey,BlockPtr> blocks;
        std::uint64_t revision=0;
        std::size_t tiles=0,records=0,bytes=0;
        Tile const* tile(int x,int y)const{
            auto found=blocks.find(key_of(x,y));if(found==blocks.end())return nullptr;
            auto const& tiles_=found->second->tiles;
            auto at=std::lower_bound(tiles_.begin(),tiles_.end(),std::make_pair(y,x),
                [](TilePtr const& a,std::pair<int,int> const& b){return std::make_pair(a->tile_y,a->tile_x)<b;});
            return at!=tiles_.end() && (*at)->tile_x==x && (*at)->tile_y==y?at->get():nullptr;
        }
        std::shared_ptr<void const> owner(int x,int y,ContentHandle handle)const{
            auto found=tile(x,y);return found?found->owner(handle):nullptr;
        }
    };
    using Lease=std::shared_ptr<Generation const>;
    static int floor_div(int a,int b){return a>=0?a/b:-((-a+b-1)/b);}
    static BlockKey key_of(int x,int y){return {floor_div(y,block),floor_div(x,block)};}

    ResidentWorld(){current=std::make_shared<Generation>();}
    // Replaces (or inserts) a tile's records; the tile's previous owners are
    // released when no generation borrows them. Extents parallel the records.
    void set(Tile tile){
        bool valid=tile.extents.size()==tile.records.size() && tile.first[0]==0 && tile.first[Layers]==tile.records.size();
        for(unsigned layer=0;valid && layer<Layers;++layer)valid=tile.first[layer]<=tile.first[layer+1];
        if(!valid)throw std::invalid_argument("resident world tile layout");
        tile.records.shrink_to_fit();tile.extents.shrink_to_fit();tile.owners.shrink_to_fit();
        auto& generation=writable();auto key=key_of(tile.tile_x,tile.tile_y);
        auto& slot=generation.blocks[key];
        auto next=slot?std::make_shared<Block>(*slot):std::make_shared<Block>();
        next->y=key.first;next->x=key.second;
        auto shared=std::make_shared<Tile const>(std::move(tile));
        auto at=locate(*next,shared->tile_x,shared->tile_y);
        if(at!=next->tiles.end() && (*at)->tile_x==shared->tile_x && (*at)->tile_y==shared->tile_y){
            account(generation,**at,-1);*at=shared;
        }else{next->tiles.insert(at,shared);++generation.tiles;}
        account(generation,*shared,1);
        finish(generation,slot,std::move(next));
    }
    bool erase(int x,int y){
        auto& generation=writable();auto key=key_of(x,y);
        auto slot=generation.blocks.find(key);if(slot==generation.blocks.end())return false;
        auto next=std::make_shared<Block>(*slot->second);
        auto at=locate(*next,x,y);
        if(at==next->tiles.end() || (*at)->tile_x!=x || (*at)->tile_y!=y)return false;
        account(generation,**at,-1);next->tiles.erase(at);--generation.tiles;
        if(next->tiles.empty()){generation.blocks.erase(slot);return true;}
        finish(generation,slot->second,std::move(next));return true;
    }
    void clear(){current=std::make_shared<Generation>();current->revision=++serial;}
    Lease publish()const{return current;}
    Generation const& view()const{return *current;}
    std::uint64_t revision()const{return current->revision;}

    // Visits candidate records of `generation` whose pass extent meets the
    // record-space rectangle [left,right) x [top,bottom) at any wrap turn
    // (turn 0: as stored, 1: -wrap_pixels, 2: +wrap_pixels). `mirror(z)` is
    // the mirrored pass's extent per world height (0 for the main pass).
    // visit(layer, record, turn, x_shift) in draw order. Returns false only
    // when the rectangle is not finite.
    template<class Mirror,class Visit>
    static bool query(Generation const& generation,double left,double top,double right,double bottom,
            int wrap_pixels,int wrap_tiles,Mirror mirror,Visit visit){
        if(!std::isfinite(left+top+right+bottom) || left>right || top>bottom)return false;
        struct Candidate {unsigned layer;int y,x;std::uint32_t sequence;unsigned turn;Record const* record;};
        std::vector<Candidate> found;
        for(unsigned turn=0;turn<(wrap_pixels?3u:1u);++turn){
            int shift=turn==0?0:turn==1?-wrap_pixels:wrap_pixels;
            int tile_shift=turn==0?0:turn==1?-wrap_tiles:wrap_tiles;
            double l=left-shift,r=right-shift;
            for(auto const& entry:generation.blocks){auto const& b=*entry.second;
                if(b.empty || !meets(b.extent,l,top,r,bottom,mirror))continue;
                for(auto const& tile:b.tiles)for(unsigned layer=0;layer<Layers;++layer){
                    auto const begin=tile->begin_of(layer),end=tile->end_of(layer);
                    for(std::size_t i=begin;i<end;++i)
                        if(meets(tile->extents[i],l,top,r,bottom,mirror))
                            found.push_back({layer,tile->tile_y,tile->tile_x+tile_shift,std::uint32_t(i-begin),turn,&tile->records[i]});
                }
            }
        }
        std::sort(found.begin(),found.end(),[](Candidate const& a,Candidate const& b){
            if(a.layer!=b.layer)return a.layer<b.layer;
            if(a.y!=b.y)return a.y<b.y;
            if(a.x!=b.x)return a.x<b.x;
            if(a.sequence!=b.sequence)return a.sequence<b.sequence;
            return a.turn<b.turn;});
        for(auto const& c:found){int shift=c.turn==0?0:c.turn==1?-wrap_pixels:wrap_pixels;visit(c.layer,*c.record,c.turn,shift);}
        return true;
    }
    // Every record of `generation` in draw order, without wrap copies.
    template<class Visit>static void each(Generation const& generation,unsigned layer,Visit visit){
        struct Item {int y,x;std::uint32_t sequence;Record const* record;};
        std::vector<Item> items;
        for(auto const& entry:generation.blocks)for(auto const& tile:entry.second->tiles){
            auto const begin=tile->begin_of(layer),end=tile->end_of(layer);
            for(std::size_t i=begin;i<end;++i)items.push_back({tile->tile_y,tile->tile_x,std::uint32_t(i-begin),&tile->records[i]});
        }
        std::sort(items.begin(),items.end(),[](Item const& a,Item const& b){
            return a.y!=b.y?a.y<b.y:a.x!=b.x?a.x<b.x:a.sequence<b.sequence;});
        for(auto const& item:items)visit(*item.record);
    }
private:
    std::shared_ptr<Generation> current;
    std::uint64_t serial=0;
    template<class Mirror>static bool meets(Extent const& e,double l,double t,double r,double b,Mirror const& mirror){
        double low=mirror(e.low_z),high=mirror(e.high_z);
        return !(e.right<=l || e.left>=r || e.bottom+high<=t || e.top+low>=b);
    }
    static typename std::vector<TilePtr>::iterator locate(Block& b,int x,int y){
        return std::lower_bound(b.tiles.begin(),b.tiles.end(),std::make_pair(y,x),
            [](TilePtr const& a,std::pair<int,int> const& k){return std::make_pair(a->tile_y,a->tile_x)<k;});
    }
    // An edit after a publication forks the block map (not the blocks).
    Generation& writable(){
        if(current.use_count()>1){auto next=std::make_shared<Generation>(*current);current=std::move(next);}
        current->revision=++serial;
        return *current;
    }
    static void account(Generation& g,Tile const& tile,int sign){
        std::size_t records=tile.records.size();
        if(sign>0){g.records+=records;g.bytes+=tile.bytes();}else{g.records-=records;g.bytes-=tile.bytes();}
    }
    void finish(Generation& generation,BlockPtr& slot,std::shared_ptr<Block> next){
        Extent e{1e300,1e300,-1e300,-1e300,1e30f,-1e30f};bool empty=true;
        for(auto const& tile:next->tiles)for(auto const& x:tile->extents){
            e.left=std::min(e.left,x.left);e.top=std::min(e.top,x.top);
            e.right=std::max(e.right,x.right);e.bottom=std::max(e.bottom,x.bottom);
            e.low_z=std::min(e.low_z,x.low_z);e.high_z=std::max(e.high_z,x.high_z);empty=false;
        }
        next->extent=e;next->empty=empty;next->revision=generation.revision;
        slot=std::move(next);
    }
};
} }
