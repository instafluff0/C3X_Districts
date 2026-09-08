#pragma once
#include "kernels.h"
#include "coast_join.h"
#include "river_corridor.h"
// Generic natural payload. Source-specific names and recipes are compiled offline.
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
struct Natural {
    std::vector<ID3D11ShaderResourceView*> textures;
    std::vector<HeightField> fields;
    std::vector<Material> materials;
    std::vector<Body> bodies;
    std::vector<Recipe> recipes;
    unsigned terrain[22]={},mountain[13]={},macro[5][2]={};
    ID3D11VertexShader* vs[3]={};ID3D11PixelShader* ps[3]={};
    ID3D11InputLayout* layout[3]={};ID3D11Buffer* frames[3]={};
    ID3D11DepthStencilState*decal_depth=nullptr;
    struct RiverPage {int c=0,r=0;std::uint64_t used=0;river::Corridor field;};
    std::vector<RiverPage> river_pages;
    profile_v2::WorldTopology const*river_world=nullptr;
    std::uint64_t river_epoch=0;
    std::int64_t river_revision=-1;
    bool ready=false;
    std::string failure;
    template<class T>void drop(T*&p){if(p)p->Release();p=nullptr;}
    void reset(){drop(decal_depth);river_pages.clear();river_world=nullptr;river_revision=-1;for(auto&p:textures)drop(p);textures.clear();fields.clear();materials.clear();bodies.clear();recipes.clear();
        for(int i=0;i<3;i++){drop(vs[i]);drop(ps[i]);drop(layout[i]);drop(frames[i]);}ready=false;}
    ~Natural(){reset();}
    template<class Read,class Upload>
    bool load(ID3D11Device*device,std::string const&root,Read read,Upload upload,
              char const*shader_directory="source_fidelity"){
        if(ready)return true;reset();
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
            if(format==DXGI_FORMAT_R8_UNORM){auto&f=fields[i];std::memcpy(&f.height,bytes.data()+12,4);std::memcpy(&f.width,bytes.data()+16,4);
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
        char const*names[]={"terrain","mountain","objects"};
        for(unsigned i=0;i<3;i++){
            std::string path=root+"/Renderer/native/"+shader_directory+"/"+names[i]+".hlsl";
            failure="shader "+std::string(names[i]);
            std::wstring wide(path.begin(),path.end());ID3DBlob *v=nullptr,*p=nullptr,*error=nullptr;
            HRESULT hr=profile_v2::compile_cached(wide.c_str(),"VSNative","vs_5_0",&v,&error);
            if(error){failure+=static_cast<char const*>(error->GetBufferPointer());OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));drop(error);}
            if(SUCCEEDED(hr))hr=profile_v2::compile_cached(wide.c_str(),"PSFeature","ps_5_0",&p,&error);
            if(error){failure+=static_cast<char const*>(error->GetBufferPointer());OutputDebugStringA(static_cast<char const*>(error->GetBufferPointer()));drop(error);}
            if(SUCCEEDED(hr))hr=device->CreateVertexShader(v->GetBufferPointer(),v->GetBufferSize(),nullptr,&vs[i]);
            if(SUCCEEDED(hr))hr=device->CreatePixelShader(p->GetBufferPointer(),p->GetBufferSize(),nullptr,&ps[i]);
            D3D11_INPUT_ELEMENT_DESC e[]={
                {"POSITION",0,DXGI_FORMAT_R32G32B32_FLOAT,0,0,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",0,DXGI_FORMAT_R32G32B32A32_FLOAT,0,12,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"NORMAL",0,DXGI_FORMAT_R32G32B32_FLOAT,0,28,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",1,DXGI_FORMAT_R32G32_FLOAT,0,40,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",2,DXGI_FORMAT_R32G32B32A32_FLOAT,0,48,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",3,DXGI_FORMAT_R32G32_FLOAT,0,64,D3D11_INPUT_PER_VERTEX_DATA,0},
                {"TEXCOORD",4,DXGI_FORMAT_R32_FLOAT,0,72,D3D11_INPUT_PER_VERTEX_DATA,0}};
            if(SUCCEEDED(hr))hr=device->CreateInputLayout(e,i==0?7:i==1?5:6,v->GetBufferPointer(),v->GetBufferSize(),&layout[i]);
            drop(v);drop(p);D3D11_BUFFER_DESC desc={};desc.ByteWidth=96;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
            if(SUCCEEDED(hr))hr=device->CreateBuffer(&desc,nullptr,&frames[i]);
            if(FAILED(hr))return false;
        }
        D3D11_DEPTH_STENCIL_DESC depth={};depth.DepthEnable=TRUE;depth.DepthWriteMask=D3D11_DEPTH_WRITE_MASK_ZERO;depth.DepthFunc=D3D11_COMPARISON_LESS_EQUAL;
        if(FAILED(device->CreateDepthStencilState(&depth,&decal_depth)))return false;
        ready=true;return true;
    }
    void update_rivers(profile_v2::WorldTopology const&w,std::int64_t revision){
        river_world=&w;
        if(river_revision!=revision){river_pages.clear();river_revision=revision;}
    }
    river::Corridor const& river_page(double x,double y){
        int pc=int(std::floor(x/8)),pr=int(std::floor(y/8));++river_epoch;
        for(auto&p:river_pages)if(p.c==pc && p.r==pr){p.used=river_epoch;return p.field;}
        // Sixteen 8x8 pages, each with a four-cell authoritative support halo.
        // Distant jumps evict LRU fields instead of building the whole map.
        if(river_pages.size()==16){auto it=std::min_element(river_pages.begin(),river_pages.end(),[](auto const&a,auto const&b){return a.used<b.used;});river_pages.erase(it);}
        RiverPage page;page.c=pc;page.r=pr;page.used=river_epoch;
        auto const&w=*river_world;auto dims=w.dimensions();hydro::Field field;
        field.map_width=dims.width;field.map_height=dims.height;field.wraps=dims.wrap_x;
        for(int r=pr*8-4;r<pr*8+12;r++)for(int c=pc*8-4;c<pc*8+12;c++){
            auto bits=w.at(w.index(c,r));if(bits==0xffffffffu)continue;
            int rx=c+r,ry=c-r;if(dims.wrap_x)rx=profile_v2::mod(rx,dims.width);if(dims.wrap_y)ry=profile_v2::mod(ry,dims.height);
            field.tiles[{c,r}]={c,r,rx,ry,int(bits&255),int((bits>>8)&255),unsigned((bits>>16)&255)};
        }
        auto lookup=[&](int c,int r){auto t=w.tile(c,r);int rx=c+r,ry=c-r;
            if(dims.wrap_x)rx=profile_v2::mod(rx,dims.width);if(dims.wrap_y)ry=profile_v2::mod(ry,dims.height);
            return Tile{rx,ry,c,r,t.present?t.real:-1};};
        page.field.build(field,[&](double u,double v){return height(float(u),float(v),lookup);});
        river_pages.push_back(std::move(page));return river_pages.back().field;
    }
    river::Sample river_sample(hydro::P p){return river_page(p.x,p.y).sample(p);}
    bool river_affects(int c,int r){return river_page(c+.5,r+.5).affects(c,r);}
    void update(ID3D11DeviceContext*c,EnvironmentState const&e,float const*light){
        // Source response coefficients retained; one authoritative phase and L.
        auto noon=evaluate_environment(12,0);
        float strength=(e.sun_intensity+e.moon_intensity)/(noon.sun_intensity+noon.moon_intensity);
        float const*color=e.sun_intensity>=e.moon_intensity?e.sun_color:e.moon_color;
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
            c->UpdateSubresource(frames[i],0,nullptr,&f,0,0);
        }
    }
    void bind(ID3D11DeviceContext*c,unsigned provider,unsigned body=0){
        c->VSSetShader(vs[provider],nullptr,0);c->PSSetShader(ps[provider],nullptr,0);
        c->IASetInputLayout(layout[provider]);c->PSSetConstantBuffers(0,1,&frames[provider]);
        ID3D11ShaderResourceView*views[22]={};
        if(provider==0)for(unsigned i=0;i<22;i++)views[i]=textures[terrain[i]];
        if(provider==1)for(unsigned i=0;i<13;i++)views[i]=textures[mountain[i]];
        if(provider==2){auto const&m=materials[bodies[body].material];for(unsigned i=0;i<7;i++)if(m.channels[i]!=0xffffffffu)views[i+3]=textures[m.channels[i]];}
        // t17 is always the shared atlas, never the source specular channel.
        c->PSSetShaderResources(0,17,views);c->PSSetShaderResources(18,4,views+18);
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
