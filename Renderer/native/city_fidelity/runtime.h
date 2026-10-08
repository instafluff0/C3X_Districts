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
struct Lighting {
    std::vector<Light> lights;
    struct Box {float low[4],high[4];};
    std::vector<Box> blockers;
};
// Version-five instance flags. A site-optional body may yield to water, a
// river channel, a mountain or steep relief at its footprint; earlier packs
// carry no flags and keep every authored body.
constexpr unsigned instance_site_optional=1u,instance_accent=2u,instance_tree=4u;
// Version-five attached effect: a flame, smoke source or night light at a
// model-space point. Width is in tile widths, height in world height units.
// Volcano plumes come from the terrain, not the pack (city_fidelity::volcano_plume).
enum EffectKind:unsigned {effect_flame=0,effect_smoke=1,effect_night_light=2,effect_volcano_plume=3,effect_kind_count};
struct Effect {float position[3],kind,width,height,seed,intensity;};
static_assert(sizeof(Effect)==32,"city effect wire contract");
struct Instance {
    unsigned model=0,capital=0; float scale=1,yaw=0,offset[2]={},bounds[4]={};
    unsigned flags=0;
    std::vector<Light> lights;
    std::vector<Effect> effects;
};
struct PavingVertex { float x,y,coverage; };
struct Paving {
    unsigned material=0; float period[2]={},atlas[4]={};
    std::vector<PavingVertex> vertices; std::vector<unsigned> indices;
};
struct Composition {
    unsigned culture=0,era=0,size=0,capital=0,environment=0;
    unsigned variant=0,walled=0,owns_walls=0,anchor_layout=0;
    bool site_aware=false; // any body carries instance_site_optional
    std::string authority;
    float clearance[4]={}; // dry shore, height range, vegetation margin, river pixels
    std::vector<Instance> instances;
    Paving paving;
    unsigned foundation_material=~0u;
    float foundation_uv[4]={};
    float foundation_step[2]={};
};
inline float metropolis_wall_radius(Composition const& city){
    if(city.size!=2)return 0.0f;
    float radius=.70f;
    for(auto const& building:city.instances)
        for(float x:{building.bounds[0],building.bounds[2]})
            for(float y:{building.bounds[1],building.bounds[3]}){
                float bx=std::abs(building.offset[0]+x);
                float by=std::abs(building.offset[1]+y);
                float reach=std::pow(std::pow(bx,6.0f)+std::pow(by,6.0f),1.0f/6.0f);
                radius=std::max(radius,reach+.03f);
            }
    return radius;
}
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
    // Version-five response offsets: body gain, contrast, saturation; window
    // shoulder and gain; pale-albedo damping (zero is the identity) and
    // reserved fields. Earlier packs leave them zero.
    float look[8]={};
    // Version-five ground-flagged material drawing attached effect quads.
    unsigned effect_material=~0u;
    std::size_t byte_count=0;
    bool complete_city_set()const{
        unsigned variants=0;
        for(auto const&t:compositions)variants=std::max(variants,t.variant+1);
        if(!variants || variants>8)return false;
        for(unsigned c=0;c<5;++c)for(unsigned e=0;e<4;++e)for(unsigned s=0;s<3;++s)
          for(unsigned capital=0;capital<2;++capital)for(unsigned wall=0;wall<(s==0?2u:1u);++wall)
            for(unsigned variant=0;variant<variants;++variant){
                unsigned matches=0;
                for(auto const&t:compositions)if(t.culture==c && t.era==e && t.size==s &&
                    t.capital==capital && t.walled==wall && t.variant==variant && t.owns_walls && t.anchor_layout)++matches;
                if(matches!=1)return false;
            }
        return true;
    }
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
        if(bytes.size()<20 || bytes.size()>128u*1024u*1024u ||
            (std::memcmp(bytes.data(),"C3XCITY2",8) && std::memcmp(bytes.data(),"C3XCITY3",8) &&
             std::memcmp(bytes.data(),"C3XCITY4",8) && std::memcmp(bytes.data(),"C3XCITY5",8)))return false;
        bool with_foundations=bytes[7]>='3',with_variants=bytes[7]>='4',with_flags=bytes[7]=='5';
        Library next;Reader r{bytes};
        unsigned nm=r.number(1024),nb=r.number(1024),nt=r.number(1024);
        if(!r.valid || !nm || !nb || !nt)return false;
        if(with_flags){
            if(!r.floats(next.look,8))return false;
            for(float value:next.look)if(value<-1.f || value>4.f)return false;
            next.effect_material=r.number();
            if(next.effect_material!=~0u && next.effect_material>=nm)return false;
        }
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
            if(with_variants){t.variant=r.number(255);t.walled=r.number(1);t.owns_walls=r.number(1);t.anchor_layout=r.number(1);}
            t.authority=r.string();if(!r.floats(t.clearance,4))return false;
            for(unsigned j=0;j<4;j++)
                if(t.clearance[j]<0 || t.clearance[j]>(j==1?112.f:20.f))return false;
            unsigned ni=r.number(128);if(!r.valid || !ni)return false;t.instances.resize(ni);
            unsigned light_count=0,capital_count=0;
            for(auto&i:t.instances){
                i.model=r.number(nb-1);i.capital=r.number(1);
                if(!r.floats(&i.scale,1) || !r.floats(&i.yaw,1) || !r.floats(i.offset,2) || !r.floats(i.bounds,4))return false;
                if(with_flags)i.flags=r.number(7);
                t.site_aware=t.site_aware || (i.flags&instance_site_optional)!=0;
                if(i.scale<=0 || i.scale>100 || i.bounds[0]>=i.bounds[2] || i.bounds[1]>=i.bounds[3] || !r.records(i.lights,4))return false;
                for(auto const&l:i.lights)if(l.range<=0 || l.range>1 || l.intensity<0)return false;
                if(with_flags){
                    if(!r.records(i.effects,16))return false;
                    for(auto const&e:i.effects)
                        if(e.kind<0 || e.kind>=float(effect_kind_count) || e.kind!=std::floor(e.kind) ||
                           e.width<=0 || e.width>1 || e.height<=0 || e.height>2 || e.intensity<0 || e.intensity>8 ||
                           next.effect_material==~0u)return false;
                }
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
            if(with_foundations && r.number(1)){
                t.foundation_material=r.number(nm-1);
                if(!r.floats(t.foundation_uv,4) || !r.floats(t.foundation_step,2) ||
                    t.foundation_step[0]<=0 || t.foundation_step[1]<=0 ||
                    t.foundation_step[0]>1 || t.foundation_step[1]>112 ||
                    t.foundation_uv[0]<0 ||
                    t.foundation_uv[1]<0 || t.foundation_uv[2]>1 ||
                    t.foundation_uv[3]>1 || t.foundation_uv[0]>=t.foundation_uv[2] ||
                    t.foundation_uv[1]>=t.foundation_uv[3])return false;
            }
        }
        if(!r.valid || r.cursor!=bytes.size())return false;
        next.byte_count=bytes.size();*this=std::move(next);return true;
    }
};
} }
