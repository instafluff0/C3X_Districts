#pragma once
#include "input_recording/codec.h"
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace c3x_remote_scene {
// Incremental camera requests between the bridge and the helper. Civ III
// re-captures its whole envelope at every camera step (all tiles, 512 bytes
// each), and the bridge used to encode and send all of it; with the resident
// world window's wider envelope (3,560 tiles on the busy save) Civ III's thread
// spent about 30 ms more per step waiting on the bridge's transport thread
// (performance review, section 29). The helper now keeps every tile it has
// received, by canonical coordinate; a request carries each occurrence's
// coordinates and anchor, full fields only for tiles whose content changed,
// and the topology array only when it changed. The helper rebuilds the exact
// frame, so everything after the wire is unchanged. A receiver that lacks a
// referenced tile (a restart, its bound) asks for a full base.
inline std::uint64_t camera_delta_hash(unsigned char const* data,std::size_t size){
    std::uint64_t hash=14695981039346656037ull;
    for(std::size_t i=0;i<size;++i){hash^=data[i];hash*=1099511628211ull;}
    return hash;
}
inline std::uint64_t camera_delta_key(std::int32_t x,std::int32_t y){
    return (std::uint64_t(std::uint32_t(x))<<32)|std::uint32_t(y);
}
// Both sides drop their copies past this many coordinates (a huge map); the
// sender then sends a new base.
constexpr std::size_t camera_delta_limit=65536;
constexpr std::uint32_t camera_delta_base_missing=0x42415345u; // reply marker: send a base

struct CameraDeltaSender {
    std::unordered_map<std::uint64_t,std::uint64_t> sent;
    std::vector<std::pair<std::uint64_t,std::uint64_t>> pending;
    std::uint32_t generation=0;
    std::uint64_t topology=0,pending_topology=0;
    bool base=true;
    std::size_t full_tiles=0,reused_tiles=0;
    void reset(){sent.clear();pending.clear();topology=0;base=true;++generation;}
    // Writes the frame's delta against what the receiver holds; commit()
    // once the receiver has decoded it.
    void encode(c3x_inputs::Writer& out,c3x_renderer_frame_v1 const& frame){
        c3x_inputs::require(frame.api_version==C3X_RENDERER_API_VERSION&&frame.struct_size==sizeof(frame),"camera delta frame ABI mismatch");
        c3x_inputs::require(frame.tile_count<=8192&&frame.world_topology_count<=12800,"camera delta occurrence limit");
        c3x_inputs::require((!frame.tile_count||frame.tiles)&&(!frame.world_topology_count||frame.world_topology),"camera delta frame arrays");
        if(sent.size()+frame.tile_count>camera_delta_limit)reset();
        pending.clear();full_tiles=reused_tiles=0;
        auto fields=frame;c3x_inputs::frame_fields(out,fields);
        out.u32(generation);out.u32(base?1u:0u);out.u32(frame.tile_count);
        c3x_inputs::Writer scratch;
        for(unsigned i=0;i<frame.tile_count;++i){auto tile=frame.tiles[i];
            out(tile.tile_x);out(tile.tile_y);out(tile.anchor_x);out(tile.anchor_y);
            tile.anchor_x=tile.anchor_y=0;scratch.bytes.clear();c3x_inputs::c3x_renderer_tile_v1_fields(scratch,tile);
            auto key=camera_delta_key(tile.tile_x,tile.tile_y);
            auto hash=camera_delta_hash(scratch.bytes.data(),scratch.bytes.size());
            auto found=base?sent.end():sent.find(key);
            bool full=found==sent.end()||found->second!=hash;
            out.u32(full?1u:0u);
            if(full){c3x_inputs::c3x_renderer_tile_v1_fields(out,tile);pending.push_back({key,hash});++full_tiles;}
            else ++reused_tiles;
        }
        pending_topology=frame.world_topology_count?camera_delta_hash(
            reinterpret_cast<unsigned char const*>(frame.world_topology),frame.world_topology_count*sizeof(c3x_renderer_u32)):0;
        pending_topology^=std::uint64_t(frame.world_topology_count)*0x9e3779b97f4a7c15ull+1;
        bool same=!base&&pending_topology==topology;
        out.u32(same?1u:0u);
        if(!same){out.u32(frame.world_topology_count);for(unsigned i=0;i<frame.world_topology_count;++i)out.u32(frame.world_topology[i]);}
    }
    void commit(){
        if(base){sent.clear();base=false;}
        for(auto const& item:pending)sent[item.first]=item.second;
        topology=pending_topology;pending.clear();
    }
};

struct CameraDeltaReceiver {
    std::unordered_map<std::uint64_t,c3x_renderer_tile_v1> tiles;
    std::vector<c3x_renderer_u32> topology;
    std::uint32_t generation=0;bool valid=false;
    // Rebuilds the full frame. False when the delta refers to tiles this
    // receiver does not hold; the input is then not fully read.
    bool decode(c3x_inputs::Reader& in,c3x_inputs::Frame& out){
        out.value={};c3x_inputs::frame_fields(in,out.value);
        auto next=in.u32();bool base=in.u32()!=0;auto count=in.u32();
        c3x_inputs::require(count<=8192,"camera delta occurrence limit");
        if(base){tiles.clear();topology.clear();generation=next;valid=true;}
        else if(!valid||next!=generation)return false;
        if(tiles.size()+count>camera_delta_limit){valid=false;return false;}
        out.tiles.resize(count);
        for(auto& tile:out.tiles){
            std::int32_t x=0,y=0,anchor_x=0,anchor_y=0;in(x);in(y);in(anchor_x);in(anchor_y);
            bool full=in.u32()!=0;auto key=camera_delta_key(x,y);
            if(full){c3x_renderer_tile_v1 value={};c3x_inputs::c3x_renderer_tile_v1_fields(in,value);tiles[key]=value;tile=value;}
            else{auto found=tiles.find(key);if(found==tiles.end()){valid=false;return false;}tile=found->second;}
            tile.anchor_x=anchor_x;tile.anchor_y=anchor_y;
        }
        if(in.u32()==0){
            auto words=in.u32();c3x_inputs::require(words<=12800,"camera delta world limit");
            topology.resize(words);for(auto& word:topology)in(word);
        }
        out.topology=topology;out.bind();
        return true;
    }
};
}
