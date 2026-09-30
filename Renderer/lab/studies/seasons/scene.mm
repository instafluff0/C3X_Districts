// Mac-only headless material laboratory. Production geometry/asset providers
// are reused; diagnostic water and one bounded shadow atlas are explicitly Lab.
// No game process, window, staging, pack writes or geometry files are created.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <set>
#include "../../shared/natural/mesh.h"
#include "../../shared/natural/queries.h"
#include "../../shared/color_response.h"
#include "../../../native/source_fidelity/light_frame.h"
#include "../../shared/natural/patterns.h"
#include "../../../native/source_fidelity/cliff_compiler.h"
#include "lab_geometry.h"
using namespace c3x_renderer;
using namespace c3x_renderer::fidelity;
namespace rc=c3x_renderer::render_core;

void require(bool value,std::string const&message){if(!value)throw std::runtime_error(message);}
std::vector<std::uint8_t> read_bytes(std::string const&path){
    std::ifstream in(path,std::ios::binary);require(bool(in),"Missing Lab input: "+path);
    return {std::istreambuf_iterator<char>(in),{}};
}
std::string read_text(std::string const&p){auto b=read_bytes(p);return {b.begin(),b.end()};}
unsigned word(std::vector<std::uint8_t>const&b,unsigned at){unsigned v;require(at+4<=b.size(),"DDS truncated");std::memcpy(&v,b.data()+at,4);return v;}
MTLPixelFormat format(unsigned f){switch(f){
    case 61:return MTLPixelFormatR8Unorm;case 28:return MTLPixelFormatRGBA8Unorm;
    case 29:return MTLPixelFormatRGBA8Unorm_sRGB;case 35:return MTLPixelFormatRG16Unorm;
    case 11:return MTLPixelFormatRGBA16Unorm;
    case 71:return MTLPixelFormatBC1_RGBA;case 72:return MTLPixelFormatBC1_RGBA_sRGB;
    case 77:return MTLPixelFormatBC3_RGBA;case 78:return MTLPixelFormatBC3_RGBA_sRGB;
    case 80:return MTLPixelFormatBC4_RUnorm;case 83:return MTLPixelFormatBC5_RGUnorm;
    case 98:return MTLPixelFormatBC7_RGBAUnorm;case 99:return MTLPixelFormatBC7_RGBAUnorm_sRGB;
    default:throw std::runtime_error("Unsupported source DDS format "+std::to_string(f));}}
id<MTLTexture> dds(id<MTLDevice>dev,std::vector<std::uint8_t>const&b){
    require(b.size()>=148 && !std::memcmp(b.data(),"DDS ",4),"DDS header");
    unsigned w=word(b,16),h=word(b,12),f=word(b,128),mips=std::max(1u,word(b,28));
    require(w && h && w<=8192 && h<=8192 && mips<=14,"DDS dimensions");
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format(f) width:w height:h mipmapped:mips>1];
    d.mipmapLevelCount=mips;d.storageMode=MTLStorageModeShared;d.usage=MTLTextureUsageShaderRead;
    auto t=[dev newTextureWithDescriptor:d];require(t!=nil,"Source texture allocation");
    unsigned block=(f==71||f==72||f==80)?8:16;std::size_t at=148;
    bool compressed=f>=71;
    for(unsigned level=0;level<mips;++level){
        unsigned pitch=compressed?((w+3)/4)*block:w*(f==61?1:f==11?8:4);
        unsigned rows=compressed?(h+3)/4:h;std::size_t n=std::size_t(pitch)*rows;
        require(at+n<=b.size(),"DDS mip payload");
        [t replaceRegion:MTLRegionMake2D(0,0,w,h) mipmapLevel:level withBytes:b.data()+at bytesPerRow:pitch];
        at+=n;w=std::max(1u,w/2);h=std::max(1u,h/2);
    }
    return t;
}
id<MTLTexture> target(id<MTLDevice>dev,MTLPixelFormat f,unsigned w,unsigned h,bool depth=false){
    auto d=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
    d.storageMode=depth?MTLStorageModePrivate:MTLStorageModeShared;
    d.usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
    auto t=[dev newTextureWithDescriptor:d];require(t!=nil,"Lab target allocation");return t;
}
struct Program {id<MTLFunction> vertex,pixel;std::set<unsigned> vb,pb;};
Program program(id<MTLDevice>dev,std::string const&folder){
    Program p;for(auto entry:{"VSMain","PSMain"}){
        auto s=read_text(folder+"/"+entry+".msl");NSError*error=nil;
        auto opts=[MTLCompileOptions new];opts.mathMode=MTLMathModeSafe;opts.languageVersion=MTLLanguageVersion2_2;
        auto lib=[dev newLibraryWithSource:[NSString stringWithUTF8String:s.c_str()] options:opts error:&error];
        require(lib!=nil,error?std::string(error.localizedDescription.UTF8String):"Metal library");
        auto fn=[lib newFunctionWithName:[NSString stringWithUTF8String:entry]];
        require(fn!=nil,"Missing Metal entry");bool vertex=entry[0]=='V';
        (vertex?p.vertex:p.pixel)=fn;auto&ids=vertex?p.vb:p.pb;
        std::regex r("\\[\\[id\\(([0-9]+)\\)\\]\\]");
        for(auto it=std::sregex_iterator(s.begin(),s.end(),r);it!=std::sregex_iterator();++it)ids.insert(unsigned(std::stoul((*it)[1])));
    }return p;
}
struct GPUVertex {
    float position[3],world[4],normal[3],uv[2],material[4],biome[2],coverage,secondary[2];
};
struct ShadowVertex{float world[3],uv[2],coverage;};
struct Group {
    std::string module;unsigned body=~0u,cliff=~0u;
    std::vector<GPUVertex> vertices;std::vector<ShadowVertex> casters;
    std::array<id<MTLTexture>,128> textures{};
    id<MTLBuffer> geometry,shadow_geometry,frame,season,cutout;
    id<MTLRenderPipelineState> pipeline;
    id<MTLBuffer> vertex_arguments,pixel_arguments,shadow_arguments;
};
FeatureBundle lab_bundle(std::string const&path){
    auto bytes=read_bytes(path);require(bytes.size()>24 && !std::memcmp(bytes.data(),"C3XVEG1",7),"Feature payload");
    std::size_t at=8;
    auto take=[&](void*destination,std::size_t n){require(at+n<=bytes.size(),"Feature payload truncated");std::memcpy(destination,bytes.data()+at,n);at+=n;};
    auto integer=[&](){unsigned v;take(&v,4);return v;};
    auto text=[&](){unsigned n=integer();require(n<4096,"Feature string limit");std::string s(n,' ');take(s.data(),n);return s;};
    require(integer()==1,"Feature version");unsigned nt=integer(),na=integer(),ng=integer();
    require(nt<=32 && na<=256 && ng<=16,"Feature bounds");FeatureBundle b;
    for(unsigned i=0;i<nt;i++)b.texture_paths.push_back(text());
    b.assets.resize(na);for(auto&a:b.assets){a.id=text();a.texture_index=integer();unsigned nv=integer(),ni=integer();
        require(nv<=100000 && ni<=300000 && a.texture_index<nt,"Feature mesh bounds");
        a.vertices.resize(nv);take(a.vertices.data(),nv*sizeof(FeatureSourceVertex));a.indices.resize(ni);take(a.indices.data(),ni*4);
        for(auto i:a.indices)require(i<nv,"Feature mesh index");}
    b.groups.resize(ng);for(auto&g:b.groups){g.name=text();unsigned n=integer();require(n<=1024,"Feature recipes");
        g.placements.resize(n);for(auto&p:g.placements){p.asset_index=integer();take(&p.scale,4);take(&p.scale_variation,4);
            p.count=integer();p.min_count=integer();p.priority=integer();p.flags=integer();take(&p.width,4);take(&p.low_end_reduction,4);}}
    require(at==bytes.size(),"Feature payload trailing bytes");return b;
}
struct FieldState{float field[4],u[4],v[4],l[4],shadow[4],depth[4];};
struct ShadowState{float u[4],v[4],l[4],domain[4],depth[4];};
struct Mode{float season[4],role[4],wrap[4],atlas[4],fall[4],flowers[4];};

id<MTLBuffer> buffer(id<MTLDevice>dev,void const*p,std::size_t size){
    auto b=[dev newBufferWithBytes:p length:std::max(std::size_t(16),size) options:MTLResourceStorageModeShared];
    require(b!=nil,"Lab buffer allocation");return b;
}
id<MTLBuffer> arguments(id<MTLDevice>dev,id<MTLFunction>fn,std::set<unsigned>const&ids,
    std::array<id<MTLTexture>,128>const&tex,std::array<id<MTLBuffer>,8>const&cb,id<MTLSamplerState>wrap,id<MTLSamplerState>clamp){
    auto encoder=ids.empty()?nil:[fn newArgumentEncoderWithBufferIndex:0];
    auto b=[dev newBufferWithLength:std::max(NSUInteger(16),encoder.encodedLength) options:MTLResourceStorageModeShared];
    [encoder setArgumentBuffer:b offset:0];
    for(auto slot:ids){
        if(slot<128){require(tex[slot]!=nil,"Missing texture slot "+std::to_string(slot));[encoder setTexture:tex[slot] atIndex:slot];}
        else if(slot<130)[encoder setSamplerState:slot==128?wrap:clamp atIndex:slot];
        else {require(slot<138 && cb[slot-130]!=nil,"Missing constant binding "+std::to_string(slot));[encoder setBuffer:cb[slot-130] offset:0 atIndex:slot];}
    }return b;
}
id<MTLRenderPipelineState> pipeline(id<MTLDevice>dev,Program const&p,std::string const&module,bool shadow){
    auto vd=[MTLVertexDescriptor vertexDescriptor];
    auto attribute=[&](unsigned index,unsigned count,unsigned offset){
        vd.attributes[index].format=MTLVertexFormat(MTLVertexFormatFloat+count-1);
        vd.attributes[index].offset=offset;vd.attributes[index].bufferIndex=30;};
    if(shadow){attribute(0,3,0);attribute(1,2,12);attribute(2,1,20);vd.layouts[30].stride=sizeof(ShadowVertex);}
    else {
        attribute(0,3,offsetof(GPUVertex,position));attribute(1,4,offsetof(GPUVertex,world));
        attribute(2,3,offsetof(GPUVertex,normal));attribute(3,2,offsetof(GPUVertex,uv));
        if(module!="water")attribute(4,4,offsetof(GPUVertex,material));
        if(module=="objects")attribute(5,2,offsetof(GPUVertex,secondary));
        else if(module!="water"){
            attribute(5,2,offsetof(GPUVertex,biome));attribute(6,1,offsetof(GPUVertex,coverage));}
        vd.layouts[30].stride=sizeof(GPUVertex);
    }
    auto d=[MTLRenderPipelineDescriptor new];d.vertexFunction=p.vertex;d.fragmentFunction=p.pixel;
    d.vertexDescriptor=vd;d.colorAttachments[0].pixelFormat=shadow?MTLPixelFormatR32Float:MTLPixelFormatRGBA16Float;
    d.depthAttachmentPixelFormat=MTLPixelFormatDepth32Float;
    if(!shadow){d.colorAttachments[1].pixelFormat=MTLPixelFormatR8Unorm;
        auto c=d.colorAttachments[0];c.blendingEnabled=YES;c.sourceRGBBlendFactor=MTLBlendFactorOne;
        c.destinationRGBBlendFactor=MTLBlendFactorOneMinusSourceAlpha;c.sourceAlphaBlendFactor=MTLBlendFactorOne;
        c.destinationAlphaBlendFactor=MTLBlendFactorOneMinusSourceAlpha;}
    NSError*error=nil;auto out=[dev newRenderPipelineStateWithDescriptor:d error:&error];
    require(out!=nil,error?std::string(error.localizedDescription.UTF8String):"Metal pipeline");return out;
}
GPUVertex vertex(MapVertex const&v,unsigned w,unsigned h,float camera_x,float camera_y){
    GPUVertex g={};float height=v.world_z*112-2.5f;
    float px=w*.5f+(v.world_x+v.world_y-1-camera_x)*64;
    float py=h*.5f+(v.world_x-v.world_y-camera_y)*32-height*(128.f/224*.82f);
    g.position[0]=px/w*2-1;g.position[1]=1-py/h*2;
    g.position[2]=std::clamp(.5f-((v.world_x-v.world_y-camera_y)*32+height*.0016f*h)/16384.f,.001f,.999f);
    float world[]={v.world_x,v.world_y,v.world_z,v.world_valid};std::copy(world,world+4,g.world);
    float normal[]={v.normal_x,v.normal_y,v.normal_z};std::copy(normal,normal+3,g.normal);
    g.uv[0]=v.u;g.uv[1]=v.v;
    float mat[]={v.material_grass,v.material_plains,v.material_desert,v.material_marsh};std::copy(mat,mat+4,g.material);
    g.biome[0]=v.authored_relief_height;g.biome[1]=v.authored_relief_blend;
    g.coverage=v.base_terrain;g.secondary[0]=v.authored_relief_height;g.secondary[1]=v.authored_relief_blend;
    return g;
}
void append(Group&g,std::vector<MapVertex>const&vertices,unsigned w,unsigned h,float cx,float cy){
    for(auto const&v:vertices){auto a=vertex(v,w,h,cx,cy);g.vertices.push_back(a);
        float coverage=g.module=="mountain"?std::clamp(v.base_terrain-42,0.f,1.f):
            g.module=="objects"?1.f:std::clamp(v.base_terrain+10,0.f,1.f);
        g.casters.push_back({{a.world[0],a.world[1],a.world[2]},{a.uv[0],a.uv[1]},coverage});}
}
std::array<double,3> verify_policy(id<MTLDevice>dev,id<MTLCommandQueue>queue,Program const&program,
                   Group const&base,id<MTLSamplerState>wrap,id<MTLSamplerState>clamp,
                   id<MTLDepthStencilState>depth_state){
    constexpr unsigned size=64;
    auto color=target(dev,MTLPixelFormatRGBA16Float,size,size),valid=target(dev,MTLPixelFormatR8Unorm,size,size);
    auto depth=target(dev,MTLPixelFormatDepth32Float,size,size,true);
    auto pipe=pipeline(dev,program,"water",false);Mode state=*reinterpret_cast<Mode*>(base.season.contents);
    state.wrap[0]=state.wrap[1]=1;state.role[0]=2;state.role[1]=0;
    auto mode=buffer(dev,&state,sizeof(state));float kind[4]={};auto probe=buffer(dev,kind,sizeof(kind));
    std::array<id<MTLBuffer>,8>constants{};constants[5]=mode;constants[6]=probe;
    auto va=arguments(dev,program.vertex,program.vb,base.textures,constants,wrap,clamp);
    auto pa=arguments(dev,program.pixel,program.pb,base.textures,constants,wrap,clamp);
    auto sample=[&](unsigned k,unsigned season,unsigned enabled){
        auto*m=reinterpret_cast<Mode*>(mode.contents);m->season[0]=season;m->season[1]=enabled;
        reinterpret_cast<float*>(probe.contents)[0]=k;
        auto command=[queue commandBuffer];auto p=[MTLRenderPassDescriptor renderPassDescriptor];
        p.colorAttachments[0].texture=color;p.colorAttachments[0].loadAction=MTLLoadActionClear;p.colorAttachments[0].storeAction=MTLStoreActionStore;
        p.colorAttachments[0].clearColor=MTLClearColorMake(0,0,0,0);
        p.colorAttachments[1].texture=valid;p.colorAttachments[1].loadAction=MTLLoadActionClear;p.colorAttachments[1].storeAction=MTLStoreActionStore;
        p.depthAttachment.texture=depth;p.depthAttachment.loadAction=MTLLoadActionClear;p.depthAttachment.storeAction=MTLStoreActionDontCare;p.depthAttachment.clearDepth=1;
        auto e=[command renderCommandEncoderWithDescriptor:p];[e setRenderPipelineState:pipe];[e setDepthStencilState:depth_state];
        [e setVertexBuffer:base.geometry offset:0 atIndex:30];[e setVertexBuffer:va offset:0 atIndex:0];[e setFragmentBuffer:pa offset:0 atIndex:0];
        for(auto t:base.textures)[e useResource:t usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
        [e useResource:mode usage:MTLResourceUsageRead stages:MTLRenderStageFragment];[e useResource:probe usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];[e endEncoding];[command commit];[command waitUntilCompleted];
        require(command.status!=MTLCommandBufferStatusError,"Policy probe GPU failure");
        std::vector<std::uint16_t> pixels(size*size*4);[color getBytes:pixels.data() bytesPerRow:size*8 fromRegion:MTLRegionMake2D(0,0,size,size) mipmapLevel:0];return pixels;
    };
    std::array<double,3> metrics{};
    for(unsigned k:{0u,1u}){
        auto pixels=sample(k,3,1);double error=0;
        for(std::size_t i=0;i<pixels.size();i+=4){float a=labv2::half_float(pixels[i]);
            error=std::max(error,double(std::abs(a-labv2::half_float(pixels[i+1]))));
            error=std::max(error,double(std::abs(a-labv2::half_float(pixels[i+2]))));
            error=std::max(error,double(labv2::half_float(pixels[i+3])));
            if(k==1)metrics[2]=std::max(metrics[2],double(a));}
        metrics[k]=error;
        std::cout<<"POLICY_PROBE kind="<<k<<" wrap_error="<<error<<"\n"<<std::flush;
        require(error<.008,k==0?"World-wrap noise/derivative closure":"World-wrap flowers/color closure");
    }
    require(metrics[2]>.01,"Flower wrap probe must exercise visible blossoms");
    require(sample(3,0,1)==sample(3,1,1),"Evergreen autumn eligibility");
    require(sample(4,0,1)==sample(4,1,1),"Wood protected from autumn tint");
    for(unsigned k:{2u,3u,4u,5u,6u}){
        auto summer=sample(k,0,1);for(unsigned s:{1u,2u,3u})require(summer==sample(k,s,0),"Disabled material policy");
    }
    require(sample(5,0,1)==sample(5,3,1),"No spring flowers in desert");
    require(sample(5,0,1)==sample(5,1,1),"Desert protected from autumn tint");
    require(sample(6,0,1)==sample(6,3,1),"No spring flowers on exposed stone");
    for(unsigned s:{1u,3u})require(sample(7,0,1)==sample(7,s,1),"Fall and spring preserve source normals");
    auto normals=sample(7,2,1);
    for(std::size_t i=0;i<normals.size();i+=4){float length=0;for(unsigned j=0;j<3;j++){float n=labv2::half_float(normals[i+j]);length+=n*n;}
        require(std::isfinite(length) && std::abs(length-1)<.004,"Finite unit snow normal");}
    std::cout<<"PASS_GPU wrap values/derivatives/flowers; evergreen/wood; desert/stone exclusion; finite snow normals\n"<<std::flush;
    return metrics;
}
int main(int argc,char**argv){@autoreleasepool{try{
    require(argc==12,"scene <root> <pack> <csv> <shaders> <out> <width> <height> <camera-x> <camera-y> <roles> <hour>");
    std::string root=argv[1],pack=argv[2],csv=argv[3],shaders=argv[4],out=argv[5];
    unsigned width=unsigned(std::stoul(argv[6])),height=unsigned(std::stoul(argv[7]));
    float cx=std::stof(argv[8]),cy=std::stof(argv[9]);
    float hour=std::stof(argv[11]);
    require(width>=320 && height>=240 && width<=1920 && height<=1200,"Bounded Lab viewport");
    auto dev=MTLCreateSystemDefaultDevice();require(dev!=nil,"Metal unavailable");auto queue=[dev newCommandQueue];
    NaturalWorld natural;std::vector<id<MTLTexture>> source;
    std::cout<<"SEASON_LAB source payload\n"<<std::flush;
    require(natural.load_data(source,[&](std::string const&p,std::vector<std::uint8_t>&b){
        std::string prefix="Renderer/packs/NaturalFidelityRuntime/",path=root+"/"+p;
        if(p.compare(0,prefix.size(),prefix)==0)path=pack+"/natural_runtime/"+p.substr(prefix.size());
        std::ifstream f(path,std::ios::binary);if(!f)return false;b.assign(std::istreambuf_iterator<char>(f),{});return true;
    },[&](auto const&bytes,auto&t){t=dds(dev,bytes);return true;}),"Natural payload rejected: "+natural.failure);
    std::vector<unsigned> roles(natural.bodies.size(),1);
    {std::ifstream f(argv[10]);for(auto&role:roles)require(bool(f>>role) && role<=3,"Semantic foliage roles");}
    std::vector<std::array<id<MTLTexture>,2>> winter_materials(natural.bodies.size());
    {std::ifstream f(out+"/winter-materials.txt");for(auto&channels:winter_materials){std::string p;require(bool(std::getline(f,p)),"Winter material descriptors");
        if(p=="-")continue;auto split=p.find('|');require(split!=std::string::npos,"Winter channel descriptor");
        channels[0]=dds(dev,read_bytes(root+"/"+p.substr(0,split)));channels[1]=dds(dev,read_bytes(root+"/"+p.substr(split+1)));}}
    unsigned flower_enabled=0;{std::ifstream f(out+"/flower-enabled.txt");f>>flower_enabled;}
    float atlas[4],fall[4],flower_grid[4];{std::ifstream f(out+"/policy.txt");
        for(auto data:{atlas,fall,flower_grid})for(unsigned i=0;i<4;i++)require(bool(f>>data[i]),"Generic seasonal recipe");}
    std::ifstream input(csv);require(bool(input),"Terrain CSV");std::string line;std::getline(input,line);
    std::replace(line.begin(),line.end(),',',' ');std::istringstream header(line);std::string magic;rc::World world;unsigned count;
    header>>magic>>world.width>>world.height>>count;require(magic=="C3X_BIQ_TERRAIN_V3","Terrain CSV version");
    unsigned wrap_x=1,wrap_y=0;header>>wrap_x>>wrap_y;
    world.wrap_x=wrap_x!=0;world.wrap_y=wrap_y!=0;std::vector<std::uint32_t> values(count,~0u);
    while(std::getline(input,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);
        int x,y,base,real;unsigned bonus,overlay,river;require(bool(row>>x>>y>>base>>real>>bonus>>overlay>>river),"Terrain CSV record");
        auto i=(std::size_t(y)*world.width+x)/2;require(i<values.size(),"Terrain CSV coordinate");values[i]=base|(real<<8)|(river<<16);}
    require(std::none_of(values.begin(),values.end(),[](auto v){return v==~0u;}),"Complete map required");
    rc::WorldCoast coast;std::cout<<"SEASON_LAB authoritative coast and river fields\n"<<std::flush;
    coast.update(world,values.data(),values.size(),1);natural.update_rivers(coast.world(),1);
    auto observe=[](auto...){return;};rc::ExactPointCache<rc::ShoreSample> scratch;
    auto cliffs=lab_bundle(root+"/Renderer/packs/ShoreNormalized/cliff_runtime.bin");
    std::vector<id<MTLTexture>> cliff_textures;for(auto p:cliffs.texture_paths){std::replace(p.begin(),p.end(),'\\','/');
        cliff_textures.push_back(dds(dev,read_bytes(root+"/Renderer/packs/ShoreNormalized/"+p)));}
    unsigned cliff_start=3+unsigned(natural.bodies.size());
    std::vector<Group> groups(cliff_start+cliffs.assets.size());groups[0].module="water";groups[1].module="terrain";groups[2].module="mountain";
    for(unsigned i=0;i<natural.bodies.size();i++){groups[3+i].module="objects";groups[3+i].body=i;}
    for(unsigned i=0;i<cliffs.assets.size();i++){groups[cliff_start+i].module="objects";groups[cliff_start+i].cliff=i;}
    Group decals;decals.module="terrain";
    float min_raw_x=cx-width/128.f-3,max_raw_x=cx+width/128.f+3;
    float min_raw_y=cy-height/64.f-5,max_raw_y=cy+height/64.f+5;
    unsigned tiles=0;
    std::cout<<"SEASON_LAB production terrain and forest geometry\n"<<std::flush;
    for(int ry=int(std::floor(min_raw_y));ry<=int(std::ceil(max_raw_y));++ry)
        for(int rx=int(std::floor(min_raw_x));rx<=int(std::ceil(max_raw_x));++rx){
        if((rx+ry)&1)continue;int c=(rx+ry)/2,r=(rx-ry)/2;
        auto t=coast.world().tile(c,r);if(!t.present)continue;
        SurfaceQueries queries(coast,scratch,rx,ry,observe,observe,false,&natural);
        auto s=queries.shore(c+.5f,r+.5f);if(s.distance < -1.6)continue;
        auto lookup=[&](int x,int y){return queries.natural_tile(x,y);};
        auto weights=[&](float x,float y){return queries.weights(x,y);};
        auto shore=[&](float x,float y){return queries.shore(x,y);};
        auto river_field=natural.bind_river_page(c+.5,r+.5);
        auto river=[&](float x,float y){return float(river_field.sample({x,y}).distance);};
        auto elevation=[&](float x,float y,float*support=nullptr){
            return queries.height(natural,[](float,float){return 0.f;},x,y,support)+river_channel_cut(river(x,y));};
        auto cancelled=[]{return false;};Tile owner=lookup(c,r);GroundProjection projection{c,r,64,32,128.f/224*.82f,float(height)};
        std::vector<MapVertex>d,m,g;
        require(emit_relief_meshes(natural,t.real,owner,projection,lookup,elevation,shore,river,weights,cancelled,d,m),"Relief emission");
        if(m.empty()){
            HillMaterialFootprint hill={owner.real==5,lookup(c-1,r).real==5,lookup(c+1,r).real==5,lookup(c,r+1).real==5,lookup(c,r-1).real==5};
            auto surface=[&](float u,float v){auto a=ground_surface(projection,u,v,elevation,shore,weights);a.material_desert*=hill(u,v);return a;};
            bool detailed=false;for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++)detailed|=lookup(c+dx,r+dy).real==5;
            detailed|=(s.rocky>.55 && std::abs(s.distance)<1.25) || river(c+.5f,r+.5f)<80;
            require(emit_ground_grid(g,surface,cancelled,detailed?64:16,nullptr,nullptr,false,64),"Ground emission");append(groups[1],g,width,height,cx,cy);
        }else append(groups[2],m,width,height,cx,cy);
        require(emit_surface_decals(natural,owner,projection,lookup,elevation,shore,weights,cancelled,d),"Surface decal emission");
        append(decals,d,width,height,cx,cy);
        {
            std::array<std::vector<MapVertex>,35> trees;
            require(lab_vegetation(natural,owner,projection,lookup,elevation,shore,river,cancelled,trees),"Vegetation emission");
            append(decals,trees[1],width,height,cx,cy);
            for(unsigned i=0;i<natural.bodies.size();i++)append(groups[3+i],trees[3+i],width,height,cx,cy);
        }
        CliffCompileInput ci;ci.tile_x=rx;ci.tile_y=ry;ci.half_w=64;ci.half_h=32;
        ci.relief_projection_scale=128.f/224*.82f;ci.content_view_height=float(height);ci.vertical_basis=150.f/112.f/.82f;
        CliffSurfaces cr;
        compile_cliff_surfaces(world,cliffs,ci,[&](int x,int y){return queries.tile(x,y);},
            [&](int x,int y){return coast.world().index(x,y);},
            [&](double x,double y){return elevation(float(x),float(y))-2.5f;},
            [&](double x,double y){return shore(float(x),float(y)).distance;},
            [&](unsigned i){float h=0;for(auto const&v:cliffs.assets[i].vertices)h=std::max(h,v.position[2]*ci.vertical_basis);return h;},
            [&](int x,int y){return coast.cell(x,y,observe);},
            [&](bool small,unsigned seed){
                auto const&g=cliffs.groups[small?1:0];unsigned total=0;for(auto const&p:g.placements)total+=p.count;
                unsigned n=patterns::feature_hash(seed)%total;
                for(auto const&p:g.placements){if(n<p.count)return rc::CliffRecipe{p.asset_index,p.scale,p.scale_variation};n-=p.count;}
                return rc::CliffRecipe{0,0,0};},cancelled,cr);
        for(unsigned i=0;i<cliffs.assets.size();i++){for(auto&v:cr.vertices[i]){v.material_grass=2;v.material_plains=1;v.material_marsh=1;}
            append(groups[cliff_start+i],cr.vertices[i],width,height,cx,cy);}
        ++tiles;
    }
    groups.insert(groups.begin()+3,std::move(decals));
    // Field texture samples exact production coast/river queries. It is a
    // diagnostic interpolation adapter, not a runtime field replacement.
    FieldState fs={};fs.field[0]=min_raw_x;fs.field[1]=min_raw_y;
    fs.field[2]=1/(max_raw_x-min_raw_x);fs.field[3]=1/(max_raw_y-min_raw_y);
    constexpr unsigned fw=768,fh=768;std::vector<float> surface(fw*fh*4),biomes(fw*fh*4);
    for(unsigned y=0;y<fh;y++)for(unsigned x=0;x<fw;x++){
        float raw_x=min_raw_x+(x+.5f)/(fw*fs.field[2]),raw_y=min_raw_y+(y+.5f)/(fh*fs.field[3]);
        float u=(raw_x+raw_y+1)*.5f,v=(raw_x-raw_y+1)*.5f;
        int c=int(std::floor(u)),r=int(std::floor(v));SurfaceQueries queries(coast,scratch,c+r,c-r,observe,observe,false,&natural);
        auto shore=queries.shore(u,v);auto weights=queries.weights(u,v);auto rd=natural.river_sample({u,v}).distance;
        std::size_t at=(std::size_t(y)*fw+x)*4;surface[at]=shore.distance;surface[at+1]=rd;
        float gx=u-.5f,gy=v-.5f;int ic=int(std::floor(gx)),ir=int(std::floor(gy));float a=rc::smoother(gx-ic),b=rc::smoother(gy-ir);
        for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++)surface[at+2]+=(coast.world().tile(ic+dx,ir+dy).real==4)*
            (dx?a:1-a)*(dy?b:1-b);
        biomes[at]=weights[0];biomes[at+1]=weights[1];biomes[at+2]=weights[2];biomes[at+3]=weights[4];
    }
    auto field_texture=[&](std::vector<float>const&data){auto t=target(dev,MTLPixelFormatRGBA32Float,fw,fh);
        [t replaceRegion:MTLRegionMake2D(0,0,fw,fh) mipmapLevel:0 withBytes:data.data() bytesPerRow:fw*16];return t;};
    auto sf=field_texture(surface),bf=field_texture(biomes);surface.clear();biomes.clear();
    // Base plane is projected from the same canonical world basis.
    for(auto q:std::array<std::array<float,2>,6>{{{{-1,1}},{{1,1}},{{1,-1}},{{-1,1}},{{1,-1}},{{-1,-1}}}}){
        GPUVertex v={};v.position[0]=q[0];v.position[1]=q[1];v.position[2]=.99;
        float raw_x=cx+q[0]*width/128.f,raw_y=cy-q[1]*height/64.f;
        v.world[0]=(raw_x+raw_y+1)*.5f;v.world[1]=(raw_x-raw_y+1)*.5f;v.world[2]=2.5f/112;v.world[3]=1;v.normal[2]=1;
        groups[0].vertices.push_back(v);
    }
    auto light=light_frame(evaluate_environment(hour,0));for(unsigned j=0;j<4;j++){fs.u[j]=light[j];fs.v[j]=light[4+j];fs.l[j]=light[8+j];}
    float low[3]={1e9f,1e9f,1e9f},high[3]={-1e9f,-1e9f,-1e9f};
    for(auto const&g:groups)for(auto const&v:g.casters)for(unsigned a=0;a<3;a++){
        float d=0;for(unsigned k=0;k<3;k++)d+=v.world[k]*light[a*4+k];low[a]=std::min(low[a],d);high[a]=std::max(high[a],d);}
    for(unsigned i=0;i<3;i++){low[i]-=.6f;high[i]+=.6f;}
    constexpr unsigned shadow_size=2048;fs.shadow[0]=low[0];fs.shadow[1]=low[1];fs.shadow[2]=1/(high[0]-low[0]);fs.shadow[3]=1/(high[1]-low[1]);
    fs.depth[0]=low[2];fs.depth[1]=1/(high[2]-low[2]);fs.depth[2]=shadow_size;
    ShadowState ss={};std::copy(fs.u,fs.u+4,ss.u);std::copy(fs.v,fs.v+4,ss.v);std::copy(fs.l,fs.l+4,ss.l);
    std::copy(fs.shadow,fs.shadow+4,ss.domain);ss.depth[0]=low[2];ss.depth[1]=high[2];ss.depth[2]=fs.depth[1];
    auto sm=target(dev,MTLPixelFormatR32Float,shadow_size,shadow_size),sd=target(dev,MTLPixelFormatDepth32Float,shadow_size,shadow_size,true);
    auto shared_frame=buffer(dev,&fs,sizeof(fs)),caster_frame=buffer(dev,&ss,sizeof(ss));
    float q6[20]={};std::copy(light.begin(),light.end(),q6);q6[16]=1;q6[17]=1;auto q6_frame=buffer(dev,q6,sizeof(q6));
    id<MTLSamplerState> samplers[2];for(unsigned i=0;i<2;i++){auto d=[MTLSamplerDescriptor new];d.minFilter=MTLSamplerMinMagFilterLinear;
        d.magFilter=MTLSamplerMinMagFilterLinear;d.mipFilter=MTLSamplerMipFilterLinear;d.maxAnisotropy=8;
        d.sAddressMode=d.tAddressMode=i?MTLSamplerAddressModeClampToEdge:MTLSamplerAddressModeRepeat;d.supportArgumentBuffers=YES;samplers[i]=[dev newSamplerStateWithDescriptor:d];}
    auto depth_descriptor=[MTLDepthStencilDescriptor new];depth_descriptor.depthCompareFunction=MTLCompareFunctionLessEqual;depth_descriptor.depthWriteEnabled=YES;
    auto depth_state=[dev newDepthStencilStateWithDescriptor:depth_descriptor];
    std::uint32_t white=~0u;auto white_tex=target(dev,MTLPixelFormatRGBA8Unorm,1,1);[white_tex replaceRegion:MTLRegionMake2D(0,0,1,1) mipmapLevel:0 withBytes:&white bytesPerRow:4];
    auto optional_dds=[&](std::string const&path){std::ifstream f(path,std::ios::binary);return f?dds(dev,read_bytes(path)):white_tex;};
    auto snow_color=optional_dds(pack+"/textures/snow_base_color.dds");
    auto snow_height=optional_dds(pack+"/textures/snow_height.dds");auto snow_gloss=optional_dds(pack+"/textures/snow_specular.dds");
    auto flowers=dds(dev,read_bytes(out+"/flowers.dds"));
    std::string patch_path=pack+"/textures/water/terrain/snow_decal_base.dds";
    bool patch_enabled=bool(std::ifstream(patch_path,std::ios::binary));auto snow_patches=optional_dds(patch_path);
    std::map<std::string,Program> programs;for(auto name:{"terrain","mountain","objects","water","shadow","probe"})programs[name]=program(dev,shaders+"/"+name);
    auto shadow_pipeline=pipeline(dev,programs["shadow"],"shadow",true);auto frames=natural.frame_settings(evaluate_environment(hour,0),light.data()+8);
    std::size_t vertices=0;
    for(auto&g:groups){if(g.vertices.empty())continue;vertices+=g.vertices.size();
        g.geometry=buffer(dev,g.vertices.data(),g.vertices.size()*sizeof(GPUVertex));
        if(!g.casters.empty())g.shadow_geometry=buffer(dev,g.casters.data(),g.casters.size()*sizeof(ShadowVertex));
        unsigned index=g.module=="mountain"?1:g.module=="objects"?2:0;g.frame=buffer(dev,&frames[index],sizeof(Frame));
        bool winter_art=g.body!=~0u && winter_materials[g.body][0]!=nil;
        Mode mode={{0,1,float(world.width),float(world.height)},{g.body==~0u?0.f:float(roles[g.body]),float(winter_art),float(flower_enabled),float(patch_enabled)},{float(world.wrap_x),float(world.wrap_y),1,1}};
        std::copy(atlas,atlas+4,mode.atlas);std::copy(fall,fall+4,mode.fall);std::copy(flower_grid,flower_grid+4,mode.flowers);
        g.season=buffer(dev,&mode,sizeof(mode));g.textures.fill(white_tex);
        if(g.module=="terrain"){for(unsigned slot=0;slot<31;slot++)g.textures[slot]=source[natural.terrain[slot]];
            for(unsigned slot=0;slot<3;slot++)g.textures[98+slot]=source[natural.floodplain[slot]];
            g.textures[31]=cliff_textures[0];g.textures[32]=source[natural.terrain[16]];}
        if(g.module=="mountain"){
            for(unsigned slot=0;slot<13;slot++)g.textures[slot]=source[natural.mountain[slot]];
            unsigned mappings[][2]={{13,6},{14,7},{15,8},{16,15},{18,16},{19,17},{20,19},{21,20},{22,21},{23,14},{24,3},{25,4},{26,5},{27,9},{28,10},{29,11},{30,30}};
            for(auto m:mappings)g.textures[m[0]]=source[natural.terrain[m[1]]];}
        bool cutout=false;
        if(g.module=="objects" && g.body!=~0u){
            auto const&material=natural.materials[natural.bodies[g.body].material];
            for(unsigned slot=0;slot<3;slot++)g.textures[slot]=source[natural.terrain[slot]];
            for(unsigned slot=0;slot<7;slot++)if(material.channels[slot]!=~0u)g.textures[3+slot]=source[material.channels[slot]];
            cutout=material.channels[6]!=~0u;
        }
        if(g.cliff!=~0u){unsigned ti=cliffs.assets[g.cliff].texture_index;
            g.textures[3]=cliff_textures[ti];g.textures[4]=cliff_textures[ti+1];g.textures[5]=cliff_textures[ti+2];g.textures[7]=cliff_textures[ti+3];}
        if(g.module=="water")g.textures[0]=source[natural.terrain[19]];
        g.textures[90]=snow_color;g.textures[91]=snow_height;g.textures[92]=snow_gloss;
        g.textures[89]=snow_patches;g.textures[93]=flowers;
        if(winter_art){g.textures[105]=winter_materials[g.body][0];g.textures[106]=winter_materials[g.body][1];}
        g.textures[94]=sf;g.textures[95]=bf;g.textures[96]=sm;
        float flags[4]={float(cutout),0,0,0};g.cutout=buffer(dev,flags,sizeof(flags));
        std::array<id<MTLBuffer>,8>constants{};constants[0]=g.frame;constants[2]=q6_frame;constants[3]=shared_frame;constants[5]=g.season;
        auto const&p=programs[g.module];g.pipeline=pipeline(dev,p,g.module,false);
        g.vertex_arguments=arguments(dev,p.vertex,p.vb,g.textures,constants,samplers[0],samplers[1]);
        g.pixel_arguments=arguments(dev,p.pixel,p.pb,g.textures,constants,samplers[0],samplers[1]);
        std::array<id<MTLTexture>,128>shadow_tex{};shadow_tex.fill(white_tex);shadow_tex[0]=g.textures[9];
        constants.fill(nil);constants[0]=caster_frame;constants[1]=g.cutout;
        g.shadow_arguments=arguments(dev,programs["shadow"].pixel,programs["shadow"].pb,shadow_tex,constants,samplers[0],samplers[1]);
        // CPU triangles are disposable once resident. No full-world or pack copies.
        g.vertices.clear();g.vertices.shrink_to_fit();
    }
    auto probes=verify_policy(dev,queue,programs["probe"],groups[0],samplers[0],samplers[1],depth_state);
    std::array<id<MTLTexture>,128>shadow_zero{};shadow_zero.fill(white_tex);std::array<id<MTLBuffer>,8>shadow_constants{};shadow_constants[0]=caster_frame;
    auto shadow_vertex_args=arguments(dev,programs["shadow"].vertex,programs["shadow"].vb,shadow_zero,shadow_constants,samplers[0],samplers[1]);
    auto command=[queue commandBuffer];auto pass=[MTLRenderPassDescriptor renderPassDescriptor];
    pass.colorAttachments[0].texture=sm;pass.colorAttachments[0].loadAction=MTLLoadActionClear;pass.colorAttachments[0].storeAction=MTLStoreActionStore;
    pass.colorAttachments[0].clearColor=MTLClearColorMake(-1e6,0,0,0);pass.depthAttachment.texture=sd;
    pass.depthAttachment.loadAction=MTLLoadActionClear;pass.depthAttachment.storeAction=MTLStoreActionDontCare;pass.depthAttachment.clearDepth=1;
    auto encoder=[command renderCommandEncoderWithDescriptor:pass];[encoder setRenderPipelineState:shadow_pipeline];[encoder setDepthStencilState:depth_state];
    [encoder setVertexBuffer:shadow_vertex_args offset:0 atIndex:0];[encoder useResource:caster_frame usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
    for(auto const&g:groups)if(g.shadow_geometry){[encoder setVertexBuffer:g.shadow_geometry offset:0 atIndex:30];[encoder setFragmentBuffer:g.shadow_arguments offset:0 atIndex:0];
        [encoder useResource:g.cutout usage:MTLResourceUsageRead stages:MTLRenderStageFragment];[encoder useResource:g.textures[9] usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
        [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:g.casters.size()];}
    [encoder endEncoding];[command commit];[command waitUntilCompleted];require(command.status!=MTLCommandBufferStatusError,"Shadow GPU failure");
    constexpr unsigned scale=2;unsigned rw=width*scale,rh=height*scale;
    auto color=target(dev,MTLPixelFormatRGBA16Float,rw,rh),valid=target(dev,MTLPixelFormatR8Unorm,rw,rh),depth=target(dev,MTLPixelFormatDepth32Float,rw,rh,true);
    std::vector<std::uint8_t> summer_pixels;
    std::vector<std::uint16_t> summer_linear;std::vector<std::uint8_t> summer_valid;
    std::array<double,7> gpu_ms{};
    for(unsigned season=0;season<7;season++){
        for(auto&g:groups)if(g.geometry){auto*m=reinterpret_cast<Mode*>(g.season.contents);m->season[0]=season==4?0:season==5?2:season==6?1:season;m->season[1]=season==5?0:1;m->wrap[3]=season==6?0:1;}
        auto cb=[queue commandBuffer];auto p=[MTLRenderPassDescriptor renderPassDescriptor];
        p.colorAttachments[0].texture=color;p.colorAttachments[0].loadAction=MTLLoadActionClear;p.colorAttachments[0].storeAction=MTLStoreActionStore;
        p.colorAttachments[0].clearColor=MTLClearColorMake(0,0,0,0);
        p.colorAttachments[1].texture=valid;p.colorAttachments[1].loadAction=MTLLoadActionClear;p.colorAttachments[1].storeAction=MTLStoreActionStore;
        p.depthAttachment.texture=depth;p.depthAttachment.loadAction=MTLLoadActionClear;p.depthAttachment.storeAction=MTLStoreActionDontCare;p.depthAttachment.clearDepth=1;
        auto e=[cb renderCommandEncoderWithDescriptor:p];[e setDepthStencilState:depth_state];[e setCullMode:MTLCullModeNone];
        for(auto const&g:groups)if(g.geometry){[e setRenderPipelineState:g.pipeline];[e setVertexBuffer:g.geometry offset:0 atIndex:30];
            [e setVertexBuffer:g.vertex_arguments offset:0 atIndex:0];[e setFragmentBuffer:g.pixel_arguments offset:0 atIndex:0];
            for(auto t:g.textures)[e useResource:t usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
            for(auto b:{g.frame,q6_frame,shared_frame,g.season})[e useResource:b usage:MTLResourceUsageRead stages:MTLRenderStageVertex|MTLRenderStageFragment];
            [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:g.geometry.length/sizeof(GPUVertex)];}
        [e endEncoding];[cb commit];[cb waitUntilCompleted];require(cb.status!=MTLCommandBufferStatusError,"Season GPU failure");
        std::vector<std::uint16_t> rgba(std::size_t(rw)*rh*4);std::vector<std::uint8_t> validity(std::size_t(rw)*rh);
        [color getBytes:rgba.data() bytesPerRow:rw*8 fromRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0];
        [valid getBytes:validity.data() bytesPerRow:rw fromRegion:MTLRegionMake2D(0,0,rw,rh) mipmapLevel:0];
        labv2::Packet packet;packet.exposure=1;packet.valid_rect={0,0,width,height};
        auto pixels=labv2::display_pixels(rgba,validity,rw,rh,scale,packet);
        if(season==0){summer_pixels=pixels;summer_linear=rgba;summer_valid=validity;}
        require(validity==summer_valid,"Seasonal material coverage unchanged");
        if(season==4 || season==5){require(pixels==summer_pixels && rgba==summer_linear,season==4?"Summer round-trip parity":"Disabled seasonal parity");}
        else {std::string name=season==6?"fall-hue-blend":"season-"+std::to_string(season);
            std::ofstream file(out+"/"+name+".bgra",std::ios::binary);file.write(reinterpret_cast<char const*>(pixels.data()),pixels.size());require(bool(file),"Lab output write");}
        gpu_ms[season]=(cb.GPUEndTime-cb.GPUStartTime)*1000;
        std::cout<<"SEASON_LAB rendered season="<<season<<" GPU_ms="<<gpu_ms[season]<<"\n"<<std::flush;
    }
    std::cout<<"PASS_GPU summer round-trip and disabled seasons pixel-identical\n";
    std::cout<<"SEASON_LAB tiles="<<tiles<<" vertices="<<vertices<<" GPU_bytes="<<dev.currentAllocatedSize<<"\n";
    std::ofstream metrics(out+"/gpu-checks.json");metrics<<"{\"summer_round_trip_linear_exact\":true,\"disabled_linear_exact\":true,"
        "\"coverage_unchanged\":true,\"evergreen_autumn_unchanged\":true,\"wood_protected\":true,"
        "\"desert_and_stone_spring_excluded\":true,\"snow_normals_finite_unit\":true,\"noise_wrap_max_error\":"<<probes[0]<<
        ",\"flower_wrap_max_error\":"<<probes[1]<<",\"flower_probe_peak_coverage\":"<<probes[2]<<
        ",\"tiles\":"<<tiles<<",\"vertices\":"<<vertices<<",\"peak_reported_gpu_allocation_bytes\":"<<dev.currentAllocatedSize<<",\"gpu_ms\":[";
    for(unsigned i=0;i<gpu_ms.size();i++)metrics<<(i?",":"")<<gpu_ms[i];metrics<<"]}\n";require(bool(metrics),"GPU receipt write");
    return 0;
}catch(std::exception const&e){std::cerr<<e.what()<<"\n";return 1;}}}
