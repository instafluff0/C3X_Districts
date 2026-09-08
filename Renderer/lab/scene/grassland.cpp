// First production-scene connection for the Mac backend. Deliberately rejects
// unsupported scenes; a flat pilot must not masquerade as the gameplay fixture.
#include "../shared/natural/world.h"
#include "../shared/natural/queries.h"
#include "../shared/natural/ground.h"
#include "../shared/natural/mesh.h"
#include "../shared/natural/relief.h"
#include "../shared/natural/patterns.h"
#include "../../native/profile_v2/relief_query.h"
#include "../../native/source_fidelity/light_frame.h"
#include "../contracts/packet_v1.h"
#include <fstream>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <sstream>
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;

bool read_bytes(std::string const& path,std::vector<std::uint8_t>& bytes) {
    std::ifstream stream(path,std::ios::binary);
    if(!stream)return false;
    bytes.assign(std::istreambuf_iterator<char>(stream),{});return true;
}
labv2::Texture texture(std::vector<std::uint8_t> const& bytes) {
    if(bytes.size()<148 || std::memcmp(bytes.data(),"DDS ",4))throw std::runtime_error("Invalid DDS input");
    auto word=[&](unsigned at){std::uint32_t v;std::memcpy(&v,bytes.data()+at,4);return v;};
    if(word(4)!=124 || word(76)!=32 || word(84)!=0x30315844u)
        throw std::runtime_error("Expected DX10 DDS");
    if(word(132)!=3 || word(140)!=1 || (word(136)&4))
        throw std::runtime_error("Expected one 2D DDS texture, not an array, cube or volume");
    labv2::Texture t;t.width=word(16);t.height=word(12);t.format=word(128);
    unsigned levels=std::max(1u,word(28)),block=0,pixel=0;
    if(!t.width || !t.height || t.width>16384 || t.height>16384 || levels>15)
        throw std::runtime_error("Invalid DDS dimensions");
    unsigned maximum_levels=1;
    for(unsigned size=std::max(t.width,t.height);size>1;size/=2)++maximum_levels;
    if(levels>maximum_levels)throw std::runtime_error("Invalid DDS mip count");
    switch(t.format) {
        case 71:case 72:case 80:block=8;break;
        case 77:case 78:case 83:block=16;break;
        case 10:case 11:pixel=8;break;
        case 35:pixel=4;break;
        case 61:pixel=1;break;
        default:throw std::runtime_error("Unsupported DDS format");
    }
    std::size_t at=148;unsigned width=t.width,height=t.height;
    for(unsigned level=0;level<levels;level++) {
        unsigned pitch=pixel?width*pixel:std::max(1u,(width+3)/4)*block;
        std::size_t size=std::size_t(pitch)*(pixel?height:std::max(1u,(height+3)/4));
        if(size>bytes.size()-at)throw std::runtime_error("Truncated DDS mip");
        t.mips.push_back({pitch,{bytes.begin()+at,bytes.begin()+at+size}});at+=size;
        width=std::max(1u,width/2);height=std::max(1u,height/2);
    }
    if(at!=bytes.size())throw std::runtime_error("Unexpected trailing DDS data");
    return t;
}
struct DetailWorld {
    std::vector<std::array<int,2>> tiles;
    std::vector<std::uint32_t> bits=std::vector<std::uint32_t>(512,0xffffffffu);
};
DetailWorld read_detail(std::istream& csv,bool gameplay=false) {
    std::string line;
    if(!std::getline(csv,line) || line!="C3X_BIQ_TERRAIN_V3,32,32,512")
        throw std::runtime_error("Expected complete category detail fixture");
    DetailWorld result;
    while(std::getline(csv,line)) {
        if(std::count(line.begin(),line.end(),',')!=6)
            throw std::runtime_error("Expected seven scene fields");
        std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
        int x,y,base,real,bonus,overlay,river;
        if(!(row>>x>>y>>base>>real>>bonus>>overlay>>river) || x<0 || x>=32 || y<0 || y>=32 ||
            ((x+y)&1) || bonus || overlay || river ||
            result.bits[(y*32+x)/2]!=0xffffffffu)
            throw std::runtime_error("Expected an unmodified grassland fixture");
        int expected_base=gameplay && x<12?1:2,expected_real=expected_base;
        if(gameplay && x==18 && y==14){expected_base=2;expected_real=5;}
        if(gameplay && x==14 && y==12){expected_base=2;expected_real=7;}
        if(gameplay && x==20 && y==10){expected_base=2;expected_real=6;}
        if(base!=expected_base || real!=expected_real)
            throw std::runtime_error("Pilot accepts only the unmodified selected grassland fixture");
        row>>std::ws;
        if(!row.eof())throw std::runtime_error("Unexpected trailing scene fields");
        result.bits[(y*32+x)/2]=unsigned(base)|(unsigned(real)<<8);result.tiles.push_back({x,y});
    }
    if(csv.bad() || result.tiles.size()!=512)throw std::runtime_error("Incomplete world");
    return result;
}
template<class T>unsigned buffer(labv2::Packet& packet,T const* values,std::size_t count) {
    auto bytes=reinterpret_cast<std::uint8_t const*>(values);
    packet.buffers.emplace_back(bytes,bytes+count*sizeof(T));return unsigned(packet.buffers.size()-1);
}
int main(int argc,char**argv) {try {
    bool gameplay=argc==5 && std::string(argv[4])=="--context-geometry";
    if(argc!=4 && !gameplay)throw std::runtime_error("usage: grassland <repository> <scene.csv> <output.packet> [--context-geometry]");
    std::ifstream csv(argv[2]);auto world=read_detail(csv,gameplay);
    labv2::Packet packet;packet.width=1280;packet.height=960;packet.downsample=2;
    packet.color_branch=1;packet.geometry_contract=1;packet.binding_contract=3;packet.valid_rect={0,0,640,480};
    NaturalWorld natural;std::vector<unsigned> textures;
    if(!natural.load_data(textures,[&](auto const&path,auto&out){return read_bytes(std::string(argv[1])+"/"+path,out);},
        [&](auto const&bytes,unsigned&index){packet.textures.push_back(texture(bytes));index=unsigned(packet.textures.size());return true;}))
        throw std::runtime_error("Natural pack: "+natural.failure);
    auto environment=evaluate_environment(12,0);auto light=light_frame(environment);
    auto frames=natural.frame_settings(environment,light.data()+8);packet.exposure=environment.exposure;
    std::array<unsigned,3> settings{};
    for(unsigned i=0;i<(gameplay?3u:1u);i++)settings[i]=buffer(packet,&frames[i],1);
    packet.shader_count=gameplay?3:1;
    std::array<float,20> shadow{};std::copy(light.begin(),light.end(),shadow.begin());
    shadow[16]=shadow[17]=1;unsigned shadow_settings=buffer(packet,shadow.data(),shadow.size());
    std::array<float,256> pages{};unsigned page_settings=buffer(packet,pages.data(),pages.size());
    std::vector<float> city_lights((3+1024*3+256*2)*4);unsigned local_lights=buffer(packet,city_lights.data(),city_lights.size());
    labv2::Texture atlas;atlas.width=atlas.height=1;atlas.format=41;atlas.array_layers=1;
    float far=-1e6f;std::vector<std::uint8_t> far_bytes(sizeof(far));std::memcpy(far_bytes.data(),&far,sizeof(far));
    atlas.mips.push_back({4,far_bytes});packet.textures.push_back(atlas);unsigned shadow_texture=unsigned(packet.textures.size());
    profile_v2::WorldCoast coast;coast.update({32,32,true,false},world.bits.data(),world.bits.size(),0);
    natural.update_rivers(coast.world(),0);
    profile_v2::ExactPointCache<profile_v2::ShoreSample> scratch;
    std::array<ReliefFields,14> relief_fields;
    std::array<std::vector<labv2::Draw>,25> draws;
    unsigned vertices=0;
    for(auto tile:world.tiles)for(int wrap=-1;wrap<=1;wrap++) {
        int x=tile[0]+wrap*32,y=tile[1],ax=x*64+320-64-16*64,ay=y*32+240-32-16*32;
        if(ax+128<0 || ax>640 || ay+64<0 || ay>480)continue;
        SurfaceQueries queries(coast,scratch,x,y,[](auto,auto){},[](auto,auto){});
        GroundProjection project{(x+y)/2,(x-y)/2,64,32,128.f/224*.82f,480};
        profile_v2::ExactPointCache<profile_v2::GroundSample> pickup_samples;std::size_t pickup_queries=0;
        ReliefSurface pickup_surface(coast.world().dimensions(),project.column,project.row,
            queries.shore(queries.center_u,queries.center_v).distance,
            [&](int c,int r){return queries.tile(c,r);},
            [&](int kind,unsigned variant,int channel,float u,float v){return relief_source(relief_fields,true,kind,variant,channel,u,v);},
            [&](float u,float v){return queries.shore(u,v);},
            [](int,int,float,float){return 1000.f;}, // Fixture validation excludes every river edge.
            [](float u,float v){return patterns::dune_height(u,v,1);},
            [](int,int){return 0.f;},pickup_samples,pickup_queries);
        auto height=[&](float u,float v,float*support=nullptr){return queries.height(natural,
            [&](float a,float b){return pickup_surface.height(a,b);},u,v,support);};
        std::array<std::vector<MapVertex>,25> meshes;
        emit_ground_grid(meshes[0],[&](float u,float v){return ground_surface(project,u,v,height,
            [&](float a,float b){return queries.shore(a,b);},[&](float a,float b){return queries.weights(a,b);});},[]{return false;});
        auto owner=queries.natural_tile(project.column,project.row);
        if(!emit_relief_meshes(natural,owner.real,owner,project,height,
            [&](float a,float b){return queries.shore(a,b);},[]{return false;},meshes[1],meshes[2]))
            throw std::runtime_error("Relief emission failed");
        if(owner.real==7 && !emit_forest(natural,owner,project,{},height,
            [&](float a,float b){return queries.shore(a,b);},
            [&](float a,float b){return natural.river_sample({a,b}).distance;},
            patterns::feature_hash,patterns::stable_random,[]{return false;},meshes))
            throw std::runtime_error("Forest emission failed");
        float viewport[]={float(ax),float(ay),-float(ay)/480,0,1.f/640,1.f/480,480,0};
        unsigned viewport_buffer=buffer(packet,viewport,8);
        for(unsigned layer=0;layer<25;layer++){
        auto const&mesh=meshes[layer];if(mesh.empty())continue;
        unsigned provider=layer<2?0:layer==2?1:2;
        labv2::Draw draw;draw.vertex_buffer=buffer(packet,mesh.data(),mesh.size());
        draw.constant_buffer=settings[provider];draw.frame_buffer=viewport_buffer;draw.shader_index=provider;
        draw.extra_constants[0]=shadow_settings;draw.extra_constants[2]=page_settings;draw.extra_constants[4]=local_lights;
        draw.count=unsigned(mesh.size());draw.stride=sizeof(MapVertex);draw.depth_mode=layer==1?1:2;
        // This validated inland fixture has opaque ground; its natural objects
        // and decals use the production premultiplied-over blend state.
        draw.blend_mode=layer?1:0;
        // Native uploads a packed 76-byte view of these fields. The packet can
        // address the shared CPU layout directly, without another packing copy.
        draw.attributes={{3,offsetof(MapVertex,x)},{3,offsetof(MapVertex,world_x)},
            {3,offsetof(MapVertex,normal_x)},{2,offsetof(MapVertex,u)},
            {4,offsetof(MapVertex,material_grass)},{2,offsetof(MapVertex,authored_relief_height)},
            {1,offsetof(MapVertex,base_terrain)}};
        if(provider==0)for(unsigned i=0;i<22;i++)if(i!=17)draw.textures[i]=textures[natural.terrain[i]];
        if(provider==1)for(unsigned i=0;i<13;i++)draw.textures[i]=textures[natural.mountain[i]];
        if(provider==2){auto const&m=natural.materials[natural.bodies[layer-3].material];
            for(unsigned i=0;i<7;i++)if(m.channels[i]!=0xffffffffu)draw.textures[i+3]=textures[m.channels[i]];}
        draw.textures[17]=shadow_texture;draws[layer].push_back(draw);vertices+=draw.count;
        }
    }
    for(auto const&layer:draws)packet.draws.insert(packet.draws.end(),layer.begin(),layer.end());
    if(!labv2::write_packet(argv[3],packet))throw std::runtime_error("Packet write failed");
    std::cout<<"Grassland "<<(gameplay?"context geometry (shadow pass pending)":"detail pilot")<<": "
             <<packet.draws.size()<<" draws, "<<vertices<<" production vertices; not a complete category parity claim\n";
    return 0;
}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 1;}}
