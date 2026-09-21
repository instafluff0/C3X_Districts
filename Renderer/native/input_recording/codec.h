#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include "../c3x_renderer_api.h"
#include "../gpu_frame_api.h"
#include "../tactical_overlay.h"

namespace c3x_inputs {
using Bytes=std::vector<unsigned char>;
constexpr std::uint32_t protocol_version=8;
constexpr std::size_t payload_limit=16u*1024u*1024u;
inline void require(bool condition,char const* message){if(!condition)throw std::runtime_error(message);}
struct Writer {
    Bytes bytes;
    void reserve(std::size_t n){require(n<=payload_limit-bytes.size(),"input payload limit");
        auto needed=bytes.size()+n;if(needed>bytes.capacity())bytes.reserve(std::min(payload_limit,std::max(needed,std::max(std::size_t(256),bytes.capacity()*2))));}
    void u32(std::uint32_t n){reserve(4);auto at=bytes.size();bytes.resize(at+4);for(unsigned i=0;i<4;++i)bytes[at+i]=static_cast<unsigned char>(n>>(8*i));}
    void u64(std::uint64_t n){u32(std::uint32_t(n));u32(std::uint32_t(n>>32));}
    void operator()(std::uint32_t n){u32(n);}
    void operator()(std::int32_t n){u32(std::uint32_t(n));}
    void operator()(std::int64_t n){u64(std::uint64_t(n));}
    void operator()(float n){std::uint32_t bits;std::memcpy(&bits,&n,4);u32(bits);}
    void string(char const* value,std::size_t maximum){
        if(!value){u32(UINT32_MAX);return;}
        std::size_t n=0;while(n<maximum&&value[n])++n;require(n<maximum,"unterminated input string");
        u32(std::uint32_t(n));reserve(n);bytes.insert(bytes.end(),value,value+n);
    }
    template<std::size_t N>void operator()(char const(&value)[N]){string(value,N);}
};
struct Reader {
    Bytes const& bytes;std::size_t at=0;
    void available(std::size_t n)const{require(at<=bytes.size()&&n<=bytes.size()-at,"truncated input payload");}
    std::uint32_t u32(){available(4);std::uint32_t n=0;for(unsigned i=0;i<4;++i)n|=std::uint32_t(bytes[at++])<<(8*i);return n;}
    std::uint64_t u64(){auto lo=u32();return lo|(std::uint64_t(u32())<<32);}
    void operator()(std::uint32_t& n){n=u32();}
    void operator()(std::int32_t& n){auto bits=u32();std::memcpy(&n,&bits,4);}
    void operator()(std::int64_t& n){auto bits=u64();std::memcpy(&n,&bits,8);}
    void operator()(float& n){auto bits=u32();std::memcpy(&n,&bits,4);}
    std::string string(std::size_t maximum,bool* present=nullptr){
        auto n=u32();if(present)*present=n!=UINT32_MAX;if(n==UINT32_MAX){require(present!=nullptr,"null required string");return {};}
        require(n<maximum,"input string limit");available(n);
        std::string result(reinterpret_cast<char const*>(bytes.data()+at),n);at+=n;
        require(result.find('\0')==std::string::npos,"embedded input string terminator");return result;
    }
    template<std::size_t N>void operator()(char(&value)[N]){auto text=string(N);std::memset(value,0,N);std::memcpy(value,text.data(),text.size());}
    void done()const{require(at==bytes.size(),"trailing input payload");}
};
struct ClockOrigin {
    std::uint64_t qpc=0,utc_filetime=0,qpc_after_utc=0;
    std::uint32_t precise_utc=0;
};
inline void check_clock_origin(ClockOrigin const& clock){
    require(clock.qpc&&clock.utc_filetime&&clock.qpc_after_utc>=clock.qpc&&clock.precise_utc<=1,"invalid input clock correlation");
}
inline void clock_origin(Writer& io,ClockOrigin const& clock){
    check_clock_origin(clock);io.u64(clock.qpc);io.u64(clock.utc_filetime);io.u64(clock.qpc_after_utc);io.u32(clock.precise_utc);
}
inline void clock_origin(Reader& io,ClockOrigin& clock){
    clock.qpc=io.u64();clock.utc_filetime=io.u64();clock.qpc_after_utc=io.u64();clock.precise_utc=io.u32();check_clock_origin(clock);
}
// Explicit protocol fields; test coverage fails when the ABI adds a field.
template<class IO,class Value>void c3x_renderer_tile_v1_fields(IO& io,Value& v){
    io(v.tile_x);
    io(v.tile_y);
    io(v.anchor_x);
    io(v.anchor_y);
    io(v.terrain_type);
    io(v.square_parts);
    io(v.terrain_overlays);
    io(v.visibility_mask);
    io(v.variant_seed);
    io(v.tile_flags);
    io(v.real_terrain_type);
    io(v.resource_id);
    io(v.resource_class);
    io(v.tile_building_id);
    io(v.city_id);
    io(v.city_owner_id);
    io(v.city_population);
    io(v.city_size);
    io(v.city_culture_group);
    io(v.city_era);
    io(v.unit_type_id);
    io(v.unit_owner_id);
    io(v.unit_class);
    io(v.unit_state);
    io(v.unit_damage);
    io(v.unit_direction);
    io(v.river_code);
    io(v.road_mask);
    io(v.railroad_mask);
    io(v.route_style);
    io(v.feature_flags);
    io(v.improvement_flags);
    io(v.irrigation_mask);
    io(v.city_flags);
    io(v.has_effect);
    io(v.territory_owner_id);
    io(v.fog_status);
    io(v.tile_visibility);
    io(v.resource_name);
    io(v.city_owner);
    io(v.city_civilization);
    io(v.city_era_name);
    io(v.unit_owner);
    io(v.unit_civilization);
    io(v.unit_era_name);
    io(v.unit_type_name);
    io(v.barbarian_tribe_id);
}
// Explicit protocol fields; test coverage fails when the ABI adds a field.
template<class IO,class Value>void c3x_renderer_unit_v1_fields(IO& io,Value& v){
    io(v.unit_id);
    io(v.action);
    io(v.queued_action);
    io(v.direction);
    io(v.action_cursor);
    io(v.frame_count);
    io(v.body_x);
    io(v.body_y);
    io(v.sprite_width);
    io(v.sprite_height);
    io(v.reduced);
    io(v.projection_scale_milli);
    io(v.hour);
    io(v.season);
    io(v.display_color_rgb);
    io(v.presentation_time_ticks);
    io(v.presentation_frequency);
    io(v.unit_key);
}
// Explicit protocol fields; test coverage fails when the ABI adds a field.
template<class IO,class Value>void c3x_renderer_camera_identity_v1_fields(IO& io,Value& v){
    io(v.map_epoch);
    io(v.viewer_epoch);
    io(v.visibility_epoch);
    io(v.scene_epoch);
}
// Pointers and ABI sizes never enter the wire representation.
template<class IO,class Value>void frame_fields(IO& io,Value& v){
    io(v.target_width);io(v.target_height);io(v.clip_left);io(v.clip_top);io(v.clip_right);io(v.clip_bottom);
    io(v.tile_width);io(v.tile_height);io(v.hour);io(v.season);
    io(v.presentation_time_ticks);io(v.presentation_frequency);io(v.dirty_flags);io(v.visible_animation_count);
    io(v.world_width_tiles);io(v.world_height_tiles);io(v.world_wrap_x);io(v.world_wrap_y);io(v.world_topology_revision);
}
inline void frame(Writer& out,c3x_renderer_frame_v1 const& v){
    require(v.api_version==C3X_RENDERER_API_VERSION&&v.struct_size==sizeof(v),"input frame ABI mismatch");
    require(v.tile_count<=8192&&v.world_topology_count<=12800,"input world/occurrence limit");
    require((!v.tile_count||v.tiles)&&(!v.world_topology_count||v.world_topology),"missing input frame arrays");
    frame_fields(out,v);out.u32(v.tile_count);
    for(unsigned i=0;i<v.tile_count;++i)c3x_renderer_tile_v1_fields(out,v.tiles[i]);
    out.u32(v.world_topology_count);for(unsigned i=0;i<v.world_topology_count;++i)out.u32(v.world_topology[i]);
}
struct Frame {
    c3x_renderer_frame_v1 value={};std::vector<c3x_renderer_tile_v1> tiles;std::vector<c3x_renderer_u32> topology;
    void bind(){value.api_version=C3X_RENDERER_API_VERSION;value.struct_size=sizeof(value);
        value.tile_count=unsigned(tiles.size());value.tiles=tiles.empty()?nullptr:tiles.data();
        value.world_topology_count=unsigned(topology.size());value.world_topology=topology.empty()?nullptr:topology.data();}
};
inline void frame(Reader& in,Frame& out){
    out.value={};frame_fields(in,out.value);auto count=in.u32();require(count<=8192,"input occurrence limit");
    out.tiles.resize(count);for(auto& tile:out.tiles){tile={};c3x_renderer_tile_v1_fields(in,tile);}
    count=in.u32();require(count<=12800,"input world limit");out.topology.resize(count);for(auto& item:out.topology)in(item);out.bind();
}
inline void unit(Writer& out,c3x_renderer_unit_v1 const& v){require(v.struct_size==sizeof(v),"input unit ABI mismatch");c3x_renderer_unit_v1_fields(out,v);}
inline void unit(Reader& in,c3x_renderer_unit_v1& v){v={};v.struct_size=sizeof(v);c3x_renderer_unit_v1_fields(in,v);}

template<class IO,class Value>void target_fields(IO& io,Value& v){
    io(v.ticket);io(v.destination);io(v.background);io(v.detail);io(v.background_detail);
    for(auto& x:v.clip)io(x);io(v.playback_flags);
}
template<class IO,class Value>void command_fields(IO& io,Value& v){
    io(v.kind);io(v.destination);io(v.source);for(auto& x:v.area)io(x);for(auto& x:v.clip)io(x);
    io(v.source_x);io(v.source_y);io(v.color);io(v.background);io(v.detail);io(v.background_detail);
    io(v.source_width);io(v.source_height);io(v.program);
}
template<class IO,class Value>void image_fields(IO& io,Value& v){
    io(v.action);io(v.ticket);io(v.image);io(v.revision);io(v.width);io(v.height);io(v.format);io(v.pixel_count);
}
inline void images(Writer& out,c3x_renderer_gpu_images_v1 const& v){
    require(v.struct_size==sizeof(v)&&v.pixel_count<=2240u*1260u&&v.command_count<=2048,"input image bounds");
    image_fields(out,v);
    if(v.action==C3X_GPU_UPLOAD){require(v.pixels||!v.pixel_count,"missing CPU image input");
        out.reserve(std::size_t(v.pixel_count)*4);for(unsigned n=0;n<v.pixel_count;++n)out(v.pixels[n]);}
    out(v.command_count);require(v.commands||!v.command_count,"missing command input");
    for(unsigned n=0;n<v.command_count;++n)command_fields(out,v.commands[n]);
}
struct Images {
    c3x_renderer_gpu_images_v1 value={};std::vector<unsigned> pixels;std::vector<c3x_renderer_gpu_command_v1> commands;
    void bind(){value.struct_size=sizeof(value);value.pixels=pixels.empty()?nullptr:pixels.data();
        value.command_count=unsigned(commands.size());value.command_struct_size=commands.empty()?0:sizeof(commands[0]);value.commands=commands.empty()?nullptr:commands.data();}
};
inline void images(Reader& in,Images& out){
    out.value={};image_fields(in,out.value);require(out.value.pixel_count<=2240u*1260u,"input image limit");
    out.pixels.clear();if(out.value.action==C3X_GPU_UPLOAD){in.available(std::size_t(out.value.pixel_count)*4);
        out.pixels.resize(out.value.pixel_count);for(auto& x:out.pixels)in(x);}
    auto count=in.u32();require(count<=2048,"input command limit");out.commands.resize(count);
    for(auto& x:out.commands){x={};command_fields(in,x);}out.bind();
}
inline void tactical(Writer& out,c3x_renderer::tactical::Input const& v){
    require(v.primitives.size()<=16384,"input tactical limit");out.u32(v.animated?1:0);out.u32(unsigned(v.primitives.size()));
    for(auto& p:v.primitives)for(auto& a:{p.bounds,p.shape,p.color,p.style})for(float x:a)out(x);
}
inline void tactical(Reader& in,c3x_renderer::tactical::Input& v){
    auto animated=in.u32(),count=in.u32();require(animated<=1&&count<=16384,"input tactical bounds");v={};v.animated=animated!=0;
    for(unsigned n=0;n<count;++n){c3x_renderer::tactical::Primitive p;
        for(auto* a:{&p.bounds,&p.shape,&p.color,&p.style})for(auto& x:*a)in(x);v.append(p);}
}
}
