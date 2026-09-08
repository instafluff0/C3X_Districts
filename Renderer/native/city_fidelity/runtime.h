#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Source-agnostic, immutable city composition. Source conversion, facade
// quadrature and layout search never run in the game or on a cache miss.
namespace c3x_renderer { namespace city_fidelity {
constexpr float source_z_metric = .648266978876f;
struct Material {
    unsigned address=0, channels=0, ground=0;
    std::string textures[7];
};
struct Vertex {
    float position[3],uv0[2],normal[3],uv1[2],tangent[3],bitangent[3],uv2[2];
};
static_assert(sizeof(Vertex)==72,"city vertex wire contract");
struct Part { unsigned material=0; std::vector<Vertex> vertices; std::vector<unsigned> indices; };
struct Point { float x=0,y=0; };
struct Model { float low[3]={},high[3]={}; std::vector<Point> hull; std::vector<Part> parts; };
struct Light { float position[3],range,color[3],intensity,direction[3],owner; };
static_assert(sizeof(Light)==48,"city light wire contract");
struct Instance {
    unsigned model=0,capital=0; float scale=1,yaw=0,offset[2]={},bounds[4]={};
    std::vector<Light> lights;
};
struct PavingVertex { float x,y,coverage; };
struct Paving {
    unsigned material=0; float period[2]={},atlas[4]={};
    std::vector<PavingVertex> vertices; std::vector<unsigned> indices;
};
struct Composition {
    unsigned culture=0,era=0,size=0,capital=0,environment=0;
    std::string authority;
    float clearance[4]={}; // dry shore, height range, vegetation margin, river pixels
    std::vector<Instance> instances;
    Paving paving;
};
struct WorldInstance {
    float x=0,y=0,z=0,scale=1,cosine=1,sine=0;
    void position(float const*source,float*world) const {
        world[0]=x+scale*(source[0]*cosine-source[1]*sine);
        world[1]=y-scale*(source[0]*sine+source[1]*cosine);
        world[2]=z+scale*source[2]/source_z_metric;
    }
    // Source material response uses the authored source frame; world-space
    // shadow bias uses its inverse-transpose through the common world basis.
    void source_direction(float const*source,float*out) const {
        out[0]=source[0]*cosine-source[1]*sine;
        out[1]=source[0]*sine+source[1]*cosine;out[2]=source[2];
    }
    void world_normal(float const*source,float*out) const {
        source_direction(source,out);out[1]=-out[1];out[2]*=source_z_metric;
        float length=std::sqrt(out[0]*out[0]+out[1]*out[1]+out[2]*out[2]);
        if(length>0)for(unsigned j=0;j<3;j++)out[j]/=length;
    }
    Light light(Light input,unsigned owner) const {
        // Quadrature already applied the instance rotation and uniform scale.
        input.position[0]+=x;input.position[1]-=y;
        input.position[2]+=z*source_z_metric;input.owner=float(owner);return input;
    }
};
inline WorldInstance place(Instance const&i,float column,float row,float ground_height) {
    return {column+i.offset[0],row-i.offset[1],ground_height/112.f,
        i.scale,std::cos(i.yaw),std::sin(i.yaw)};
}
struct Library {
    std::vector<Material> materials; std::vector<Model> models; std::vector<Composition> compositions;
    std::size_t byte_count=0;
    struct Reader {
        std::vector<std::uint8_t> const&data; std::size_t cursor=8; bool valid=true;
        bool bytes(void*out,std::size_t n) {
            if(!valid || cursor>data.size() || n>data.size()-cursor){valid=false;return false;}
            if(n)std::memcpy(out,data.data()+cursor,n);cursor+=n;return true;
        }
        unsigned number(unsigned limit=0xffffffffu) {
            unsigned value=0;if(!bytes(&value,4) || value>limit)valid=false;return valid?value:0;
        }
        bool floats(float*out,std::size_t count) {
            if(!bytes(out,count*4))return false;
            for(std::size_t i=0;i<count;i++)if(!std::isfinite(out[i]))valid=false;
            return valid;
        }
        std::string string(bool path=false) {
            unsigned n=number(1024);if(!valid || n>data.size()-cursor){valid=false;return {};}
            std::string s(reinterpret_cast<char const*>(data.data()+cursor),n);cursor+=n;
            if(s.find('\0')!=std::string::npos)valid=false;
            if(path && !s.empty() && (s.compare(0,9,"Renderer/") || s.find("..")!=std::string::npos ||
                s.find(':')!=std::string::npos || s.find('\\')!=std::string::npos))valid=false;
            return s;
        }
        template<class T>bool records(std::vector<T>&v,unsigned maximum) {
            unsigned count=number(maximum);
            if(!valid || count>(data.size()-cursor)/sizeof(T)){valid=false;return false;}
            v.resize(count);return floats(reinterpret_cast<float*>(v.data()),count*sizeof(T)/4);
        }
        bool indices(std::vector<unsigned>&out,unsigned n,unsigned count) {
            if(!valid || count>(data.size()-cursor)/4){valid=false;return false;}
            out.resize(count);if(!bytes(out.data(),count*4))return false;
            for(unsigned i:out)if(i>=n)valid=false;
            if(count%3)valid=false;return valid;
        }
    };
    bool decode(std::vector<std::uint8_t> const&bytes) {
        // Decode transactionally: an incomplete/stale pack cannot replace a
        // usable library or cause native city suppression.
        if(bytes.size()<20 || bytes.size()>32u*1024u*1024u || std::memcmp(bytes.data(),"C3XCITY2",8))return false;
        Library next;Reader r{bytes};
        unsigned nm=r.number(512),nb=r.number(512),nt=r.number(256);
        if(!r.valid || !nm || !nb || !nt)return false;
        next.materials.resize(nm);next.models.resize(nb);next.compositions.resize(nt);
        for(auto&m:next.materials){
            m.address=r.number(3);m.channels=r.number(63);m.ground=r.number(1);
            for(auto&p:m.textures)p=r.string(true);
            if(m.textures[0].empty() || (m.address!=0 && m.address!=3))return false;
        }
        for(auto&m:next.models){
            unsigned np=r.number(128);if(!np || !r.floats(m.low,3) || !r.floats(m.high,3))return false;
            for(unsigned j=0;j<3;j++)if(m.low[j]>m.high[j])return false;
            if(!r.records(m.hull,4096) || m.hull.size()<3)return false;
            m.parts.resize(np);
            for(auto&p:m.parts){
                p.material=r.number(nm-1);unsigned nv=r.number(1000000),ni=r.number(3000000);
                if(!r.valid || !nv || !ni || nv>(bytes.size()-r.cursor)/sizeof(Vertex))return false;
                p.vertices.resize(nv);
                if(!r.floats(reinterpret_cast<float*>(p.vertices.data()),nv*18) || !r.indices(p.indices,nv,ni))return false;
            }
        }
        for(auto&t:next.compositions){
            t.culture=r.number(4);t.era=r.number(3);t.size=r.number(2);t.capital=r.number(1);t.environment=r.number(1);
            t.authority=r.string();if(!r.floats(t.clearance,4))return false;
            for(float v:t.clearance)if(v<0 || v>20)return false;
            unsigned ni=r.number(32);if(!r.valid || !ni)return false;t.instances.resize(ni);
            unsigned light_count=0,capital_count=0;
            for(auto&i:t.instances){
                i.model=r.number(nb-1);i.capital=r.number(1);
                if(!r.floats(&i.scale,1) || !r.floats(&i.yaw,1) || !r.floats(i.offset,2) || !r.floats(i.bounds,4))return false;
                if(i.scale<=0 || i.scale>100 || i.bounds[0]>=i.bounds[2] || i.bounds[1]>=i.bounds[3] || !r.records(i.lights,4))return false;
                for(auto const&l:i.lights)if(l.range<=0 || l.range>1 || l.intensity<0)return false;
                light_count+=unsigned(i.lights.size());capital_count+=i.capital;
            }
            if(light_count>128 || capital_count!=t.capital)return false;
            if(r.number(1)){
                auto&p=t.paving;p.material=r.number(nm-1);
                if(!r.floats(p.period,2) || !r.floats(p.atlas,4) || p.period[0]<=0 || p.period[1]<=0)return false;
                unsigned nv=r.number(30000),ni2=r.number(180000);
                if(!r.valid || nv>(bytes.size()-r.cursor)/sizeof(PavingVertex))return false;
                p.vertices.resize(nv);
                if(!r.floats(reinterpret_cast<float*>(p.vertices.data()),nv*3) || !r.indices(p.indices,nv,ni2))return false;
            }
        }
        if(!r.valid || r.cursor!=bytes.size())return false;
        next.byte_count=bytes.size();*this=std::move(next);return true;
    }
};
} }
