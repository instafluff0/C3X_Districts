#pragma once
// Current production asset decoding and CPU response; no graphics API dependency.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "../../../native/environment_runtime.h"
#include "../../../native/source_fidelity/kernels.h"
namespace c3x_renderer { namespace fidelity {
struct HeightField {
    unsigned width=0,height=0; float minimum=0,maximum=1;
    std::vector<std::uint8_t> pixels;
    float sample(float u,float v) const {
        if(pixels.empty())return 0;
        u-=std::floor(u);v-=std::floor(v);
        float px=u*width,py=v*height,tx=px-std::floor(px),ty=py-std::floor(py);
        unsigned x=unsigned(px)%width,y=unsigned(py)%height;
        auto at=[&](unsigned a,unsigned b){return (pixels[(b%height)*width+a%width]/255.f-minimum)/std::max(.0001f,maximum-minimum);};
        return (at(x,y)*(1-tx)+at(x+1,y)*tx)*(1-ty)+(at(x,y+1)*(1-tx)+at(x+1,y+1)*tx)*ty;
    }
};
struct BodyVertex {float position[3],normal[3],uv[2];};
struct Body {unsigned material=0;std::vector<BodyVertex> vertices;};
struct Material {unsigned channels[7]={},tint=0,repeat=0;};
struct Recipe {unsigned object;float scale,variation;unsigned count,min_count,priority,flags;float width,reduction;};
struct Frame {float sun[4],color[4],ambient[4],view[4],detail[4],quality[4];};
struct NaturalData {
    std::vector<HeightField> fields;
    std::vector<Material> materials;
    std::vector<Body> bodies;
    std::vector<Recipe> recipes;
    unsigned terrain[22]={},mountain[13]={},macro[5][2]={};
    std::string failure;
    template<class Texture,class Read,class Upload>
    bool load_data(std::vector<Texture>&textures,Read read,Upload upload){
        failure="catalog";
        std::vector<std::uint8_t>d;
        if(!read("Renderer/packs/NaturalFidelityRuntime/natural.bin",d)||d.size()<24||std::memcmp(d.data(),"C3XNAT1\0",8))return false;
        std::size_t pos=8;
        auto take=[&](void*out,std::size_t n){if(n>d.size()-pos)return false;std::memcpy(out,d.data()+pos,n);pos+=n;return true;};
        unsigned count[4]={};if(!take(count,16)||count[0]>128||count[1]>64||count[2]!=22||count[3]!=25)return false;
        textures.resize(count[0]);fields.resize(count[0]);
        for(unsigned i=0;i<count[0];i++){
            unsigned n=0;if(!take(&n,4)||n>128||n>d.size()-pos)return false;
            std::string path(reinterpret_cast<char const*>(d.data()+pos),n);pos+=n;
            if(path.find_first_not_of("0123456789abcdef.ds")!=std::string::npos)return false;
            failure="texture "+path;
            std::vector<std::uint8_t>bytes;
            if(!read("Renderer/packs/NaturalFidelityRuntime/"+path,bytes)||bytes.size()<148)return false;
            unsigned format=0;std::memcpy(&format,bytes.data()+128,4);
            if(format==61 /* DDS R8_UNORM */){auto&f=fields[i];std::memcpy(&f.height,bytes.data()+12,4);std::memcpy(&f.width,bytes.data()+16,4);
                if(!f.width||!f.height||f.width>4096||f.height>4096||148ull+std::uint64_t(f.width)*f.height>bytes.size())return false;
                f.pixels.assign(bytes.begin()+148,bytes.begin()+148+f.width*f.height);
                auto mm=std::minmax_element(f.pixels.begin(),f.pixels.end());f.minimum=*mm.first/255.f;f.maximum=*mm.second/255.f;}
            if(!upload(bytes,textures[i]))return false;
        }
        failure="bindings";
        if(!take(terrain,sizeof(terrain))||!take(mountain,sizeof(mountain))||!take(macro,sizeof(macro)))return false;
        for(auto i:terrain)if(i>=textures.size())return false;
        for(auto i:mountain)if(i>=textures.size())return false;
        for(auto const&r:macro)for(auto i:r)if(i>=fields.size()||fields[i].pixels.empty())return false;
        if(fields[terrain[14]].pixels.empty())return false;
        failure="materials";
        materials.resize(count[1]);for(auto&m:materials){if(!take(&m,sizeof(m)))return false;
            for(auto i:m.channels)if(i!=0xffffffffu && i>=textures.size())return false;
            if(m.tint>1||m.repeat>1)return false;}
        failure="bodies";
        bodies.resize(count[2]);for(auto&b:bodies){unsigned n=0;
            if(!take(&b.material,4)||!take(&n,4)||b.material>=materials.size()||n<3||n>300000||n%3)return false;
            b.vertices.resize(n);if(!take(b.vertices.data(),n*sizeof(BodyVertex)))return false;
            for(auto const&v:b.vertices){for(auto x:v.position)if(!std::isfinite(x))return false;for(auto x:v.normal)if(!std::isfinite(x))return false;for(auto x:v.uv)if(!std::isfinite(x))return false;}}
        failure="recipes";
        recipes.resize(count[3]);unsigned weight=0;
        for(auto&r:recipes){if(!take(&r,sizeof(r))||r.object>=bodies.size()||!std::isfinite(r.scale)||r.scale<=0||!std::isfinite(r.variation)||r.variation<0||r.variation>2||r.flags>7)return false;weight+=r.count;}
        if(pos!=d.size()||weight!=180)return false;
        return true;
    }
    std::array<Frame,3> frame_settings(EnvironmentState const&e,float const*light)const{
        // Source response coefficients retained; one authoritative phase and L.
        auto noon=evaluate_environment(12,0);
        float strength=(e.sun_intensity+e.moon_intensity)/(noon.sun_intensity+noon.moon_intensity);
        float const*color=e.sun_intensity>=e.moon_intensity?e.sun_color:e.moon_color;
        std::array<Frame,3> result={};
        for(unsigned i=0;i<3;i++){
            Frame f={};std::copy(light,light+3,f.sun);f.sun[3]=(i==0?2.08f:i==1?2.18f:2.05f)*strength;
            float source_color[]={1,i==2?4.5f/6.2f:.91f,i==2?3.5f/6.2f:.76f};
            for(unsigned j=0;j<3;j++){f.color[j]=source_color[j]*color[j]/std::max(.001f,noon.sun_color[j]);
                f.ambient[j]=(j==0?(i==2?.34f:.35f):j==1?(i==2?.45f:.46f):(i==2?.60f:.61f))*e.ambient_color[j]/std::max(.001f,noon.ambient_color[j]);}
            f.ambient[3]=i==0?.61f:i==1?.58f:.62f;
            f.view[0]=.490290f;f.view[1]=-.735435f;f.view[2]=.469979f;
            if(i==0){f.detail[0]=.43f;f.detail[1]=.075f;f.detail[2]=.70f;}
            if(i==1){f.detail[0]=fields[macro[1][0]].minimum;f.detail[1]=fields[macro[1][0]].maximum;f.detail[2]=1.72f;f.detail[3]=3.25f;
                f.quality[0]=1;f.quality[1]=.72f;f.quality[2]=.105f;f.quality[3]=.72f;}
            if(i==2){f.detail[0]=.09f;f.detail[1]=.82f;f.detail[2]=.16f;f.detail[3]=1;}
            result[i]=f;
        }
        return result;
    }
    template<class Lookup>float height(float x,float y,Lookup lookup,float*support_out=nullptr)const{
        float h=2.5f,support=0;
        for(int r=int(std::floor(y))-1;r<=int(std::floor(y))+1;r++)for(int c=int(std::floor(x))-1;c<=int(std::floor(x))+1;c++){
            Tile tile=lookup(c,r);if(tile.real!=5)continue;Hill hill=composed_hill(tile);
            if(std::abs(x-hill.x)>hill.radius_x||std::abs(y-hill.y)>hill.radius_y)continue;
            float authored=composed_source_macro(fields[terrain[14]],hill,x,y);
            float s=smooth01((hill_support(hill,x,y)+(authored-.5f)*.56f-.08f)/.88f);
            h=std::max(h,2.5f+hill.height*s*(.20f+authored*.94f));support=std::max(support,s);
        }
        if(support_out)*support_out=support;return h;
    }
};
} }
