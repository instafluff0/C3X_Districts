// CPU data checks run without a graphics API or licensed source assets.
#include "data.h"
#include "../../../native/source_fidelity/light_frame.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;

void check(bool value,char const*message){if(!value)throw std::runtime_error(message);}
template<class T>void append(std::vector<std::uint8_t>&out,T const&value){
    auto p=reinterpret_cast<std::uint8_t const*>(&value);out.insert(out.end(),p,p+sizeof(value));
}
template<class T>void replace(std::vector<std::uint8_t>&out,std::size_t at,T const&value){
    std::memcpy(out.data()+at,&value,sizeof(value));
}
struct Fixture {
    std::vector<std::uint8_t> pack={'C','3','X','N','A','T','3',0},dds=std::vector<std::uint8_t>(152);
    std::size_t bindings=0,material=0,body=0,recipe=0,surface=0,surface_vertices=0;
    Fixture(){
        for(unsigned count:{1,1,22,25,3,18})append(pack,count);
        append(pack,5u);for(char c:std::string("a.dds"))pack.push_back(std::uint8_t(c));
        bindings=pack.size();for(unsigned i=0;i<54;i++)append(pack,0u);
        material=pack.size();append(pack,Material{});
        body=pack.size();
        for(unsigned i=0;i<22;i++){
            append(pack,0u);append(pack,3u);
            for(unsigned j=0;j<3;j++)append(pack,BodyVertex{{float(j),float(i),0},{0,0,1},{0,0}});
        }
        recipe=pack.size();
        for(unsigned i=0;i<25;i++)append(pack,Recipe{i%22,1,0,i==0?180u:0u,0,0,0,1,0});
        surface=pack.size();
        for(unsigned biome=0;biome<3;biome++)append(pack,SurfaceRecipe{biome,2,.2f,1,.8f,.7f,biome*6,6});
        surface_vertices=pack.size();
        for(unsigned biome=0;biome<3;biome++)for(auto const&vertex:std::array<SurfaceVertex,6>{{
            {-.4f,-.3f,.1f,.2f},{.4f,-.3f,.9f,.2f},{.4f,.3f,.9f,.8f},
            {-.4f,-.3f,.1f,.2f},{.4f,.3f,.9f,.8f},{-.4f,.3f,.1f,.8f}}})append(pack,vertex);
        replace(dds,12,2u);replace(dds,16,2u);replace(dds,128,61u);
        dds[148]=0;dds[149]=64;dds[150]=128;dds[151]=255;
    }
    bool load(NaturalData&data)const{
        std::vector<unsigned> textures;
        return data.load_data(textures,[&](std::string const&path,std::vector<std::uint8_t>&out){
            if(path=="Renderer/packs/NaturalFidelityRuntime/natural.bin")out=pack;
            else if(path=="Renderer/packs/NaturalFidelityRuntime/a.dds")out=dds;
            else return false;
            return true;
        },[](std::vector<std::uint8_t>const&bytes,unsigned&texture){texture=unsigned(bytes.size());return true;});
    }
};

void valid_data(NaturalData const&data,unsigned surface_count=3){
    check(data.bodies.size()==22 && data.recipes.size()==25 && data.surface_recipes.size()==surface_count,"body/recipe count");
    check(data.surface_vertices.size()>=18 && data.surface_vertices.size()%3==0,"surface triangle count");
    check(data.fields[data.terrain[30]].sample(.25f,.25f)>0,"surface detail field");
    auto flat=[](int c,int r){return Tile{c+r,c-r,c,r,2};};
    auto hill=[](int c,int r){return Tile{c+r,c-r,c,r,c==0&&r==0?5:2};};
    bool relief=false;
    for(int y=-8;y<=16;y++)for(int x=-8;x<=16;x++){
        float support=-1;
        check(data.height(x/8.f,y/8.f,flat,&support)==2.5f && support==0,"flat height");
        float height=data.height(x/8.f,y/8.f,hill,&support);
        check(std::isfinite(height) && height>=2.5f && height<70 && support>=0 && support<=1,"hill bounds");
        relief=relief||height>3;
    }
    check(relief,"hill has no relief");
    for(unsigned hour=0;hour<24;hour++){
        auto environment=evaluate_environment(float(hour),0);
        auto light=light_frame(environment);
        auto frames=data.frame_settings(environment,light.data()+8);
        for(auto const&frame:frames){
            std::array<float,24> values;std::memcpy(values.data(),&frame,sizeof(frame));
            for(float value:values)check(std::isfinite(value),"nonfinite frame setting");
            check(frame.sun[0]==light[8] && frame.sun[1]==light[9] && frame.sun[2]==light[10],"light basis");
        }
        check(frames[0].detail[0]==.43f && frames[0].detail[1]==.075f && frames[0].detail[2]==.70f,"terrain detail response");
        auto const&field=data.fields[data.macro[1][0]];
        check(frames[1].detail[0]==field.minimum && frames[1].detail[1]==field.maximum,"mountain height calibration");
        check(frames[2].detail[0]==.09f && frames[2].detail[1]==.82f,"object material response");
        if(hour==12)check(frames[0].sun[3]==2.08f && frames[1].sun[3]==2.18f && frames[2].sun[3]==2.05f,"noon strengths");
    }
}

int main(int argc,char**argv){try{
    static_assert(sizeof(BodyVertex)==32 && sizeof(Material)==36 && sizeof(Recipe)==36 && sizeof(SurfaceRecipe)==32 && sizeof(SurfaceVertex)==16 && sizeof(Frame)==96,"wire layouts");
    Fixture fixture;NaturalData data;
    check(fixture.load(data),"valid fixture rejected");valid_data(data);
    auto const&field=data.fields[0];
    check(field.sample(0,0)==0 && field.sample(.5f,.5f)==1 && field.sample(-.5f,-.5f)==1,"height wrapping");
    check(std::abs(field.sample(.25f,.25f)-447.f/1020.f)<1e-6f,"bilinear height sample");
    unsigned rejected=0;
    auto reject=[&](Fixture const&bad){NaturalData invalid;check(!bad.load(invalid),"malformed input accepted");++rejected;};
    auto bad=fixture;bad.pack[0]='X';reject(bad);
    bad=fixture;replace(bad.pack,8,129u);reject(bad);
    bad=fixture;bad.pack[36]='/';reject(bad);
    bad=fixture;bad.dds.resize(147);reject(bad);
    bad=fixture;replace(bad.dds,16,0u);reject(bad);
    bad=fixture;replace(bad.pack,bad.bindings,1u);reject(bad);
    bad=fixture;replace(bad.pack,bad.material+28,2u);reject(bad);
    bad=fixture;replace(bad.pack,bad.body+4,4u);reject(bad);
    bad=fixture;replace(bad.pack,bad.body+8,std::numeric_limits<float>::quiet_NaN());reject(bad);
    bad=fixture;replace(bad.pack,bad.recipe+4,0.f);reject(bad);
    bad=fixture;replace(bad.pack,bad.recipe+24,8u);reject(bad);
    bad=fixture;replace(bad.pack,bad.recipe+12,179u);reject(bad);
    bad=fixture;replace(bad.pack,bad.surface,3u);reject(bad);
    bad=fixture;replace(bad.pack,bad.surface+4,0.f);reject(bad);
    bad=fixture;replace(bad.pack,bad.surface+12,0u);reject(bad);
    bad=fixture;replace(bad.pack,bad.surface+16,0.f);reject(bad);
    bad=fixture;replace(bad.pack,bad.surface+28,0u);reject(bad);
    bad=fixture;replace(bad.pack,bad.surface+24,19u);reject(bad);
    bad=fixture;replace(bad.pack,bad.surface_vertices,std::numeric_limits<float>::quiet_NaN());reject(bad);
    bad=fixture;replace(bad.pack,bad.surface_vertices+8,1.2f);reject(bad);
    bad=fixture;bad.pack.pop_back();reject(bad);
    bad=fixture;bad.pack.push_back(0);reject(bad);
    std::cout<<"PASS natural CPU data: "<<rejected<<" invalid inputs, height sampling and 24 lighting phases\n";
    if(argc==2){
        NaturalData real;std::vector<std::size_t> textures;
        check(real.load_data(textures,[&](std::string const&path,std::vector<std::uint8_t>&out){
            std::ifstream stream(std::string(argv[1])+"/"+path,std::ios::binary);
            if(!stream)return false;out.assign(std::istreambuf_iterator<char>(stream),{});return true;
        },[](std::vector<std::uint8_t>const&bytes,std::size_t&texture){texture=bytes.size();return true;}),"production payload rejected");
        valid_data(real,35);
        std::cout<<"PASS production natural payload: textures="<<textures.size()<<" materials="<<real.materials.size()<<" bodies="<<real.bodies.size()<<" recipes="<<real.recipes.size()<<" surface="<<real.surface_recipes.size()<<"\n";
    }
    return 0;
}catch(std::exception const&error){std::cerr<<error.what()<<"\n";return 1;}}
