// Q7 presentation adapter: reuse Q0 loaders, packet ABI, shader and environment.
// No graphics backend is implemented here. The frozen entry is never called.
#define main q7_unused_frozen_entry
#include "../../shared/frozen_scene.cpp"
#undef main
#include "../../shared/environment_runtime.cpp"
#include "../lighting/shadow_field_v1.h"
#include <fstream>
#include <map>
#include <sstream>

// Q7 disk contract is independent of additions to Q0's migration sidecars.
struct Q7Vertex {
    float x,y,depth,u,v,normal_x,normal_y,normal_z,material_index;
};
static_assert(sizeof(Q7Vertex)==36,"Q7 geometry wire drift");

int main(int argc, char **argv) {
    if (argc != 7) return 2;
    try {
        std::ifstream descriptor(argv[6]);
        std::string json((std::istreambuf_iterator<char>(descriptor)), {});
        // The owned generator uses a mandatory scenarios.geometry string.
        auto k = json.find("\"geometry\"");
        bool pack_input=k==std::string::npos;
        if(pack_input)k=json.find("\"presentation_geometry\"");
        if(k==std::string::npos)throw std::runtime_error("Q7 geometry input missing");
        auto q = json.find('"', json.find(':', k) + 1);
        std::string path = json.substr(q + 1, json.find('"', q + 1) - q - 1);
        if(pack_input)path+="/geometry.bin";
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Q7 geometry missing");
        auto u32 = [&]() { uint32_t v; in.read((char*)&v, 4); return v; };
        auto str = [&]() { auto n=u32(); if(n>4096)throw std::runtime_error("path limit");
            std::string s(n, '\0'); in.read(s.data(), n); return s; };
        if (u32() != 0x37515043) throw std::runtime_error("Q7 payload version");
        recorded.width=atoi(argv[2]); recorded.height=atoi(argv[3]);
        recorded.downsample=atoi(argv[5]);
        auto e=c3x_renderer::evaluate_environment(atof(argv[4]),0);
        LabSettings s={}; s.exposure=1; s.l13a_layout=1; s.l17_layout=1;
        s.cities_enabled=1; s.environment_exposure=e.exposure;
        s.sun_intensity=e.sun_intensity; s.moon_intensity=e.moon_intensity;
        s.night_activation=e.night_activation; s.emissive_scale=e.emissive_scale;
        s.shadow_strength=e.shadow_strength; s.hour=atof(argv[4]);
        for(int i=0;i<3;++i){s.sun_direction[i]=e.sun_direction[i];s.sun_color[i]=e.sun_color[i];
            s.moon_direction[i]=e.moon_direction[i];s.moon_color[i]=e.moon_color[i];s.ambient_color[i]=e.ambient_color[i];}
        recorded.buffers.emplace_back((uint8_t*)&s,(uint8_t*)&s+sizeof(s));
        ID3D11Device device;
        std::map<std::string,unsigned> textures;
        auto texture=[&](std::string const& name){
            if(name.empty())return 0u;
            if(textures.count(name))return textures[name];
            ID3D11ShaderResourceView* v=nullptr; unsigned w,h;
            std::vector<uint8_t> bytes;
            if(!read_file(name,bytes)||bytes.size()<148)throw std::runtime_error("DDS missing");
            unsigned fmt=read_u32(bytes,128);if(fmt==71)fmt=72;if(fmt==77)fmt=78;
            if(!load_dds(&device,name,fmt,&v,w,h))
                throw std::runtime_error("Q7 texture failed: "+name);
            unsigned id=v->id;release(v);textures[name]=id;return id;
        };
        std::vector<q6::WorldTriangle> casters;
        float span=6;
        auto draws=u32(); if(draws>250)throw std::runtime_error("Q7 draw limit");
        for(unsigned i=0;i<draws;++i){
            auto base=str(),emissive=str();auto count=u32();
            if(count>3000000 || count%3)throw std::runtime_error("Q7 vertex limit");
            std::vector<Q7Vertex> v(count);
            in.read((char*)v.data(),count*sizeof(Q7Vertex));
            q6::WorldTriangle tri;
            for(unsigned j=0;j<count;++j){
                auto &vertex=v[j];
                float sx=(vertex.x+1)*recorded.width*.5f-recorded.width*.5f;
                float sy=(1-vertex.y)*recorded.height*.5f-recorded.height*.5f;
                float z=(.94f-vertex.depth-.375f-sy*.75f/recorded.height)/
                    (80.9543f*.75f/recorded.height+.20732f);
                float sum=(sy+80.9543f*z)/32, difference=sx/64;
                tri[j%3]={(sum+difference)*.5f,(sum-difference)*.5f,z};
                for(float f:tri[j%3])span=std::max(span,std::abs(f)*4+2);
                // Masked excavation decals cannot become opaque shadow quads.
                // Shared alpha-tested caster adoption is tracked separately.
                if(j%3==2 && vertex.material_index> -99)casters.push_back(tri);
                if(vertex.material_index>=0)vertex.material_index=-29.f;
            }
            labv2::Draw d;d.feature=1;d.depth=1;d.clear_depth=i==0;
            d.constant_buffer=0;d.vertex_buffer=recorded.buffers.size();
            d.stride=sizeof(Q7Vertex);d.count=count;
            d.attributes={{3,0},{2,12},{3,20},{1,32}};
            d.textures[124]=texture(base);d.textures[116]=texture(emissive);
            recorded.buffers.emplace_back((uint8_t*)v.data(),(uint8_t*)(v.data()+v.size()));
            recorded.draws.push_back(d);
        }
        auto shadow=q6::build_shadow_frame(casters,e,1024,span);
        recorded.textures.push_back(std::move(shadow.texture));
        for(auto&d:recorded.draws)d.textures[115]=recorded.textures.size();
        s.height_texel[0]=shadow.U[0];s.height_texel[1]=shadow.U[1];s.normal_strength=shadow.U[2];s.exposure=span;
        s.lab_mode=shadow.V[0];s.beauty_relief_enabled=shadow.V[1];s.beauty_water_enabled=shadow.V[2];s.shoreline_integrated=1024;
        s.promotion_tile_layout=shadow.L[0];s.scene_width=shadow.L[1];s.scene_height=shadow.L[2];
        s.roads_enabled=recorded.width;s.roads_only=recorded.height;s.dune_only=1;s.l10_layout=1;
        if(json.find("shadows-off")!=std::string::npos)s.dune_only=0;
        if(json.find("emissive-only")!=std::string::npos)s.biq_layout=1;
        if(json.find("component-id")!=std::string::npos)s.marsh_enabled=1;
        if(json.find("component-depth")!=std::string::npos)s.marsh_enabled=2;
        memcpy(recorded.buffers[0].data(),&s,sizeof(s));
        if(!in || in.peek()!=EOF)throw std::runtime_error("Q7 truncated/trailing geometry");
        return labv2::write_packet(argv[1],recorded)?0:1;
    } catch(std::exception const& e){fprintf(stderr,"Q7: %s\n",e.what());return 1;}
}
