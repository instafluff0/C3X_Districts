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
#include "../../../native/scene_lighting.h"
#include "../../../native/source_fidelity/kernels.h"
#include "low_relief.h"
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
struct SurfaceRecipe {unsigned biome;float scale,variation;unsigned weight;float width,height;unsigned first,vertex_count;};
struct SurfaceVertex {float x,y,u,v;};
struct Frame {float sun[4],color[4],ambient[4],view[4],detail[4],quality[4];};
// Forest recipe sets, as ranges of recipes: broadleaf, pine and snow-covered
// pine. The two pine sets come from an optional pack file; without it every
// forest keeps the broadleaf recipe.
enum ForestVariety : unsigned {broadleaf_forest,pine_forest,snow_pine_forest,forest_variety_count};
struct RecipeSet {unsigned first=0,end=0,weight=0;};
// Tree bodies, including the optional varieties' extra materials.
constexpr unsigned max_natural_bodies=40;
struct NaturalData {
    LowRelief low_relief;
    std::vector<HeightField> fields;
    std::vector<Material> materials;
    std::vector<Body> bodies;
    std::vector<Recipe> recipes;
    std::vector<SurfaceRecipe> surface_recipes;
    std::vector<SurfaceVertex> surface_vertices;
    unsigned terrain[31]={},floodplain[3]={},mountain[13]={},macro[5][2]={};
    // Principal ridge direction of each macro stamp (radians; u right, v up).
    float macro_axis[5]={};
    // Optional dedicated volcano element: height, footprint blend, and the
    // lava-channel mask from its region IDs (region 112). Without it a
    // volcano uses an ordinary mountain stamp. volcano_trend is the authored
    // height's mean at each distance from the element centre (64 steps over
    // half its width): MountainShape keeps the gullies and ridges about it.
    HeightField volcano_height,volcano_blend,volcano_channel;
    std::array<float,64> volcano_trend{};
    bool volcano_ready=false;
    // The broadleaf contract: the pack's first 25 recipes, weighing 180.
    std::array<RecipeSet,forest_variety_count> forest_sets{{{0,25,180},{},{}}};
    std::string failure;
    RecipeSet const& forest_set(unsigned variety)const{
        return forest_sets[variety<forest_variety_count && forest_sets[variety].weight?variety:broadleaf_forest];
    }
    // Optional pine and snow-pine forests: their extra textures, materials and
    // bodies (an existing body's vertices with another material) and recipes.
    // Any invalid entry leaves the broadleaf-only data unchanged.
    template<class Texture,class Read,class Upload>
    bool load_forest_varieties(std::vector<std::uint8_t> const&d,std::vector<Texture>&textures,Read read,Upload upload,char const*natural_pack){
        std::size_t pos=8;unsigned count[4]={};
        auto take=[&](void*out,std::size_t n){if(n>d.size()-pos)return false;std::memcpy(out,d.data()+pos,n);pos+=n;return true;};
        if(d.size()<24||std::memcmp(d.data(),"C3XFVAR1",8)||!take(count,16)||count[0]>16||count[1]>16||
           count[3]!=forest_variety_count-1||bodies.size()+count[2]>max_natural_bodies)return false;
        std::vector<std::string> paths(count[0]);
        for(auto&path:paths){unsigned n=0;if(!take(&n,4)||n>128||n>d.size()-pos)return false;
            path.assign(reinterpret_cast<char const*>(d.data()+pos),n);pos+=n;
            if(path.find_first_not_of("0123456789abcdef.ds")!=std::string::npos)return false;}
        std::size_t texture_total=textures.size()+count[0],material_total=materials.size()+count[1];
        std::vector<Material> extra_materials(count[1]);
        for(auto&m:extra_materials){if(!take(&m,sizeof(m))||m.tint>1||m.repeat>1)return false;
            for(auto i:m.channels)if(i!=0xffffffffu && i>=texture_total)return false;}
        std::vector<Body> extra_bodies(count[2]);
        for(auto&b:extra_bodies){unsigned source=0;
            if(!take(&source,4)||!take(&b.material,4)||source>=bodies.size()||b.material>=material_total)return false;
            b.vertices=bodies[source].vertices;}
        std::vector<Recipe> extra_recipes;std::array<RecipeSet,forest_variety_count> sets=forest_sets;
        for(unsigned set=1;set<forest_variety_count;set++){unsigned n=0;
            if(!take(&n,4)||!n||n>32)return false;
            sets[set].first=unsigned(recipes.size()+extra_recipes.size());sets[set].weight=0;
            for(unsigned i=0;i<n;i++){Recipe r{};
                if(!take(&r,sizeof(r))||r.object>=bodies.size()+count[2]||!std::isfinite(r.scale)||r.scale<=0||
                   !std::isfinite(r.variation)||r.variation<0||r.variation>2||r.flags>7||r.count>64)return false;
                sets[set].weight+=r.count;extra_recipes.push_back(r);}
            sets[set].end=unsigned(recipes.size()+extra_recipes.size());
            if(!sets[set].weight)return false;
        }
        if(pos!=d.size())return false;
        // Uploaded views join the owner's table, which releases them on reset
        // even if a later upload fails.
        std::size_t first=textures.size();textures.resize(texture_total);fields.resize(texture_total);
        for(unsigned i=0;i<count[0];i++){std::vector<std::uint8_t>bytes;
            if(!read(std::string(natural_pack)+paths[i],bytes)||bytes.size()<148||!upload(bytes,textures[first+i]))return false;}
        materials.insert(materials.end(),extra_materials.begin(),extra_materials.end());
        bodies.insert(bodies.end(),extra_bodies.begin(),extra_bodies.end());
        recipes.insert(recipes.end(),extra_recipes.begin(),extra_recipes.end());
        forest_sets=sets;
        return true;
    }
    template<class Texture,class Read,class Upload>
    bool load_data(std::vector<Texture>&textures,Read read,Upload upload){
        failure="catalog";
        std::vector<std::uint8_t>d;
        constexpr char const* natural_pack="Renderer/packs/NaturalFidelityRuntime/";
        if(!read(std::string(natural_pack)+"natural.bin",d)||d.size()<32||std::memcmp(d.data(),"C3XNAT4\0",8))return false;
        std::size_t pos=8;
        auto take=[&](void*out,std::size_t n){if(n>d.size()-pos)return false;std::memcpy(out,d.data()+pos,n);pos+=n;return true;};
        unsigned count[6]={};if(!take(count,24)||count[0]>128||count[1]>64||
                count[2]!=32||count[3]!=35||
                count[4]<4||count[4]>64||count[5]<3||count[5]>100000||count[5]%3)return false;
        // Jungle bodies and decals use recipes 25..34. An older forest-only
        // pack cannot satisfy the current terrain compiler's input contract.
        textures.resize(count[0]);fields.resize(count[0]);
        for(unsigned i=0;i<count[0];i++){
            unsigned n=0;if(!take(&n,4)||n>128||n>d.size()-pos)return false;
            std::string path(reinterpret_cast<char const*>(d.data()+pos),n);pos+=n;
            if(path.find_first_not_of("0123456789abcdef.ds")!=std::string::npos)return false;
            failure="texture "+path;
            std::vector<std::uint8_t>bytes;
            if(!read(std::string(natural_pack)+path,bytes)||bytes.size()<148)return false;
            unsigned format=0;std::memcpy(&format,bytes.data()+128,4);
            if(format==61 /* DDS R8_UNORM */){auto&f=fields[i];std::memcpy(&f.height,bytes.data()+12,4);std::memcpy(&f.width,bytes.data()+16,4);
                if(!f.width||!f.height||f.width>4096||f.height>4096||148ull+std::uint64_t(f.width)*f.height>bytes.size())return false;
                f.pixels.assign(bytes.begin()+148,bytes.begin()+148+f.width*f.height);
                auto mm=std::minmax_element(f.pixels.begin(),f.pixels.end());f.minimum=*mm.first/255.f;f.maximum=*mm.second/255.f;}
            if(!upload(bytes,textures[i]))return false;
        }
        failure="bindings";
        if(!take(terrain,sizeof(terrain))||!take(floodplain,sizeof(floodplain))||!take(mountain,sizeof(mountain))||!take(macro,sizeof(macro)))return false;
        for(auto i:terrain)if(i>=textures.size())return false;
        for(auto i:floodplain)if(i>=textures.size())return false;
        for(auto i:mountain)if(i>=textures.size())return false;
        for(auto const&r:macro)for(auto i:r)if(i>=fields.size()||fields[i].pixels.empty())return false;
        // From height-squared second moments of the authored field, so any
        // pack's stamps can be turned to follow a mountain range.
        for(unsigned i=0;i<5;i++){
            auto const& f=fields[macro[i][0]];double w=0,mx=0,my=0,xx=0,yy=0,xy=0;
            for(unsigned y=0;y<f.height;y++)for(unsigned x=0;x<f.width;x++){
                double h=f.pixels[std::size_t(y)*f.width+x]/255.,q=h*h;
                double px=(x+.5)/f.width-.5,py=.5-(y+.5)/f.height;
                w+=q;mx+=q*px;my+=q*py;xx+=q*px*px;yy+=q*py*py;xy+=q*px*py;
            }
            if(w<=0)continue;
            mx/=w;my/=w;
            macro_axis[i]=float(.5*std::atan2(2*(xy/w-mx*my),(xx/w-mx*mx)-(yy/w-my*my)));
        }
        if(fields[terrain[14]].pixels.empty()||fields[terrain[30]].pixels.empty())return false;
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
        if(weight!=301u)return false;
        forest_sets={{{0,25,0},{},{}}};
        for(unsigned i=0;i<25;i++)forest_sets[broadleaf_forest].weight+=recipes[i].count;
        surface_recipes.resize(count[4]);unsigned surface_weight[4]={};
        for(auto&r:surface_recipes){if(!take(&r,sizeof(r))||r.biome>3||!std::isfinite(r.scale)||r.scale<=0||r.scale>16||
                !std::isfinite(r.variation)||r.variation<0||r.variation>2||!r.weight||r.weight>64||
                !std::isfinite(r.width)||!std::isfinite(r.height)||r.width<=0||r.height<=0||r.width>4||r.height>4||
                r.vertex_count<3||r.vertex_count%3||r.first>count[5]||r.vertex_count>count[5]-r.first)return false;
            surface_weight[r.biome]+=r.weight;}
        surface_vertices.resize(count[5]);if(!take(surface_vertices.data(),surface_vertices.size()*sizeof(SurfaceVertex)))return false;
        for(auto const&v:surface_vertices)if(!std::isfinite(v.x)||!std::isfinite(v.y)||!std::isfinite(v.u)||!std::isfinite(v.v)||
                v.u<-.02f||v.u>1.05f||v.v<-.02f||v.v>1.05f)return false;
        if(pos!=d.size()||!surface_weight[0]||!surface_weight[1]||!surface_weight[2]||!surface_weight[3])return false;
        // Older/simple texture-only packs remain valid and exactly flat.
        d.clear();read(std::string(natural_pack)+"low-relief.bin",d);
        failure="low relief";if(!low_relief.load(d))return false;
        d.clear();
        if(read(std::string(natural_pack)+"forest-varieties.bin",d))
            load_forest_varieties(d,textures,read,upload,natural_pack);
        volcano_ready=load_volcano_element(read);
        return true;
    }
    template<class Read> bool load_volcano_element(Read read){
        constexpr char const* element="Renderer/packs/TerrainElementsNormalized/textures/terrain_elements/terrain_feature_volcano/";
        HeightField* targets[]={&volcano_height,&volcano_blend,&volcano_channel};
        char const* names[]={"height_lod0.dds","blend_lod0.dds","region_ids_lod0.dds"};
        for(unsigned i=0;i<3;i++){
            std::vector<std::uint8_t> bytes;
            if(!read(std::string(element)+names[i],bytes)||bytes.size()<148)return false;
            unsigned format=0;std::memcpy(&format,bytes.data()+128,4);
            if(format!=61 && format!=62)return false; // DDS R8_UNORM, R8_UINT
            auto& f=*targets[i];std::memcpy(&f.height,bytes.data()+12,4);std::memcpy(&f.width,bytes.data()+16,4);
            if(!f.width||f.width>1024||f.height!=f.width||148ull+std::uint64_t(f.width)*f.height>bytes.size())return false;
            f.pixels.assign(bytes.begin()+148,bytes.begin()+148+std::size_t(f.width)*f.height);
            if(i<2){auto mm=std::minmax_element(f.pixels.begin(),f.pixels.end());f.minimum=*mm.first/255.f;f.maximum=*mm.second/255.f;}
        }
        if(volcano_blend.width!=volcano_height.width || volcano_channel.width!=volcano_height.width)return false;
        for(auto& id:volcano_channel.pixels)id=id==112?255:0;
        volcano_channel.minimum=0;volcano_channel.maximum=1;
        measure_volcano_trend();
        return true;
    }
    void measure_volcano_trend(){
        std::array<float,64> count{};volcano_trend.fill(0);
        unsigned const w=volcano_height.width;
        for(unsigned y=0;y<w;y++)for(unsigned x=0;x<w;x++){
            float dx=(x+.5f)/w-.5f,dy=(y+.5f)/w-.5f;
            unsigned bin=unsigned(std::sqrt(dx*dx+dy*dy)*128);if(bin>=64)continue;
            volcano_trend[bin]+=(volcano_height.pixels[y*w+x]/255.f-volcano_height.minimum)/
                std::max(.0001f,volcano_height.maximum-volcano_height.minimum);
            count[bin]++;
        }
        for(unsigned i=0;i<64;i++)volcano_trend[i]=count[i]>0?volcano_trend[i]/count[i]:0;
    }
    std::array<Frame,3> frame_settings(EnvironmentState const&e,float const*light)const{
        // Source response coefficients retained; one authoritative phase and L.
        auto noon=evaluate_environment(12,0);
        float strength=(e.sun_intensity+e.moon_intensity)/(noon.sun_intensity+noon.moon_intensity);
        auto key=lighting::key_light(e);
        float const*color=key.color.data();
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
    void hill_height(Hill const& hill,float x,float y,float& h,float& support)const{
        if(std::abs(x-hill.x)>hill.radius_x||std::abs(y-hill.y)>hill.radius_y)return;
        float authored=composed_source_macro(fields[terrain[14]],hill,x,y);
        // The authored field supplies crest and valley character, while the
        // broad topology envelope keeps hills legible as rolling landforms.
        // Compressing source contrast avoids miniature mountain spikes.
        float s=smooth01((hill_support(hill,x,y)+(authored-.5f)*.30f-.06f)/.90f);
        h=std::max(h,2.5f+hill.height*s*(.40f+authored*.56f));support=std::max(support,s);
    }
    template<class Lookup>float height(float x,float y,Lookup lookup,float*support_out=nullptr)const{
        float h=2.5f,support=0;
        for(int r=int(std::floor(y))-1;r<=int(std::floor(y))+1;r++)for(int c=int(std::floor(x))-1;c<=int(std::floor(x))+1;c++){
            Tile tile=lookup(c,r);if(tile.real!=5)continue;Hill hill=composed_hill(tile);
            hill_height(hill,x,y,h,support);
        }
        if(support_out)*support_out=support;return h;
    }
};
} }
