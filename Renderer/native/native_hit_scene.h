#pragma once
// Input geometry accompanies copied native UI commands. A query evaluates one
// UI point; it never waits for the renderer, reads a GPU texture, or rasterizes
// terrain. Map content is an opaque input surface, independent of its lighting.
#include "gpu_image_commands.h"
#include <memory>
#include <array>
#include <unordered_map>
#include <vector>
#include <stdexcept>

namespace c3x_native_hit {
using namespace c3x_gpu_images;
constexpr unsigned opaque_map=0x10000;
struct Node;
using Ref=std::shared_ptr<Node const>;
struct Budget {std::size_t nodes=0,bytes=0;};
struct Node {
    std::shared_ptr<Budget> budget;
    explicit Node(std::shared_ptr<Budget> b):budget(std::move(b)){
        if(budget->nodes>=8192)throw std::runtime_error("native input coverage node budget exceeded");++budget->nodes;
    }
    ~Node(){--budget->nodes;budget->bytes-=pixels.size()*sizeof(unsigned);}
    Command command={};Rect bounds={};Format format=Format::rgb555;
    Ref prior,source,background,program;
    unsigned width=0,height=0,constant=0;bool draw=false;
    std::vector<unsigned> pixels;
};
inline bool inside(Rect r,int x,int y){return x>=r.left&&x<r.right&&y>=r.top&&y<r.bottom;}
inline unsigned sample(Ref const& node,int x,int y,unsigned depth=0){
    if(!node||x<0||y<0||x>=int(node->width)||y>=int(node->height))return 0;
    if(depth>1024)throw std::runtime_error("native input coverage depth exceeded");
    auto read=[&](Ref const& r,int a,int b){return sample(r,a,b,depth+1);};
    if(!node->draw)return node->pixels.empty()?node->constant:node->pixels[std::size_t(y)*node->width+x];
    auto const& c=node->command;
    auto prior=[&]{return read(node->prior,x,y);};
    if(!inside(node->bounds,x,y))return prior();
    int sx=x-c.area.left+c.source_x,sy=y-c.area.top+c.source_y;
    if(c.kind==Kind::fill)return c.color;
    if(c.kind==Kind::quantize)return opaque_map;
    if(c.kind==Kind::copy)return read(node->source,sx,sy);
    if(c.kind==Kind::color_key){auto v=read(node->source,sx,sy);return v==c.color?prior():v;}
    if(c.kind==Kind::invert){auto v=read(node->source,sx,sy);return v==opaque_map?v:v^65535;}
    if(c.kind==Kind::native_sprite){auto v=read(node->source,sx,sy);return v&65536?v&65535:prior();}
    if(c.kind==Kind::native_image){
        auto interval=[](int at,int src,int dst){int last=((2*at+1)*src)/(2*dst);
            return std::pair<int,int>{src<=dst?last:at==0?0:((2*at-1)*src)/(2*dst)+1,last+1};};
        auto xr=interval(x-c.area.left,c.source_width,c.area.right-c.area.left);
        auto yr=interval(y-c.area.top,c.source_height,c.area.bottom-c.area.top);
        unsigned value=65535;
        for(int yy=yr.first;yy<yr.second;++yy)for(int xx=xr.first;xx<xr.second;++xx){
            auto v=read(node->source,c.source_x+xx,c.source_y+yy);
            if(v==opaque_map)return opaque_map;value&=v;
        }
        return value==c.color?prior():value;
    }
    auto green6=node->format==Format::rgb565;
    auto channels=[&](unsigned v){return std::array<unsigned,3>{v&31,(v>>5)&(green6?63u:31u),(v>>(green6?11:10))&31};};
    auto packed=[&](std::array<unsigned,3> v){return v[0]|(v[1]<<5)|(v[2]<<(green6?11:10));};
    if(c.kind==Kind::native_blend){
        auto p=c.color==2?unsigned(c.source_width):read(node->source,sx,sy);
        unsigned weight=c.color==2?unsigned(c.source_height):p>>24;
        if(c.color!=2&&c.color!=4&&weight==255)return prior();
        auto below=read(node->background,x,y);unsigned word=p&65535;
        if(c.color==4)return below==(p>>16)?word:prior();
        if(c.color==0){
            if(!weight)return word;if(below==opaque_map)return opaque_map;
            unsigned b=((word&31)*8+((below&31)*8*weight>>8))>>3;
            unsigned g=(((word>>5)&31)*8+(((below>>5)&31)*8*weight>>8))>>3;
            unsigned r=(((word>>10)&31)*8+(((below>>10)&31)*8*weight>>8))>>3;
            return (b|(g<<5)|(r<<10))&65535;
        }
        auto a=channels(word),b=channels(below);std::array<unsigned,3> q={};
        if(c.color==3){if(below==opaque_map&&weight!=16)return opaque_map;
            for(unsigned i=0;i<3;++i)q[i]=(a[i]*weight+b[i]*(16-weight))>>4;return packed(q);}
        if(c.color==2){if(below==opaque_map&&weight)return opaque_map;
            for(unsigned i=0;i<3;++i){unsigned shift=i==1&&green6?2:3;
                q[i]=(((a[i]<<shift)*(256-weight)>>8)+((b[i]<<shift)*weight>>8))>>shift;}return packed(q);}
        if(c.color==1){
            std::array<unsigned,3> rgb={p&255,(p>>8)&255,(p>>16)&255};
            if(below==opaque_map&&weight)return opaque_map;
            if(!weight){for(unsigned i=0;i<3;++i)b[i]=rgb[i]>>(i==1&&green6?2:3);}
            for(unsigned i=0;i<3;++i){unsigned shift=i==1&&green6?2:3;
                q[i]=((rgb[i]*(255-weight)>>8)+((b[i]<<shift)*(weight+1)>>8))>>shift;}return packed(q);
        }
    }
    if(c.kind==Kind::native_lookup){
        unsigned old=prior(),block=c.color,word=old;
        if(c.program){auto code=read(node->program,sx,sy);
            if(c.color==32){if(code==0xffffffff)return old;if(code&65536)return code&65535;block=code;}
            else {if(code>15)return old;if(!code)return 0;block=code-1;}}
        if(old==0x7c1f)word=read(node->background,x,y);
        if(word==opaque_map)return opaque_map;
        unsigned index=(block<<15)|word;return read(node->source,index&1023,index>>10);
    }
    if(c.kind==Kind::native_text){
        auto below=prior();if(below==opaque_map)return opaque_map;
        auto pixel=read(node->source,sx,sy);auto b=channels(below);unsigned rgb=0;
        for(unsigned i=0;i<3;++i){unsigned shift=i==1&&green6?2:3;
            unsigned level=(b[i]<<shift)|(b[i]>>(i==1&&green6?4:2));
            unsigned index=std::min(level/16,15u),fraction=level-index*16,span=index==15?15:16,id=(pixel>>(i*10))&1023;
            unsigned a=read(node->background,index,id),z=read(node->background,index+1,id);
            rgb|=((a*(span-fraction)+z*fraction+span/2)/span)<<(8*i);
        }
        unsigned result=(rgb>>3&31)|((rgb>>(green6?10:11)&(green6?63:31))<<5)|((rgb>>19&31)<<(green6?11:10));
        if(!green6&&!(pixel&0x40000000))result|=below&32768;return result;
    }
    throw std::runtime_error("unsupported native input coverage command");
}
// Retain only the source regions a command can sample. In particular, a
// partial HUD repaint must retire overwritten history in every canvas involved,
// even when two canvases repeatedly copy their UI back and forth.
using Regions=std::vector<Rect>;
inline bool nonempty(Rect r){return r.left<r.right&&r.top<r.bottom;}
inline void append(Regions& regions,Rect r){if(nonempty(r))regions.push_back(r);}
inline bool overwrite(Kind kind){return kind==Kind::fill||kind==Kind::copy||kind==Kind::quantize||kind==Kind::invert;}
struct RetentionKey {
    Node const* node;Regions regions;
    bool operator==(RetentionKey const& other)const{
        if(node!=other.node||regions.size()!=other.regions.size())return false;
        for(std::size_t i=0;i<regions.size();++i){auto a=regions[i],b=other.regions[i];
            if(a.left!=b.left||a.top!=b.top||a.right!=b.right||a.bottom!=b.bottom)return false;}
        return true;
    }
};
struct RetentionHash {
    std::size_t operator()(RetentionKey const& key)const{
        auto hash=std::hash<Node const*>{}(key.node);
        for(auto r:key.regions)for(auto v:{r.left,r.top,r.right,r.bottom})hash=(hash^unsigned(v))*16777619u;
        return hash;
    }
};
using RetentionCache=std::unordered_map<RetentionKey,Ref,RetentionHash>;
inline Ref retain(Ref const& node,Regions const& needed,RetentionCache& cache,unsigned depth=0){
    if(!node||needed.empty())return {};
    if(!node->draw)return node;
    if(depth>1024)throw std::runtime_error("native input coverage retention depth exceeded");
    RetentionKey key{node.get(),needed};auto found=cache.find(key);if(found!=cache.end())return found->second;
    auto save=[&](Ref value){cache.emplace(std::move(key),value);return value;};
    Regions touched,prior;
    for(auto r:needed){auto hit=intersection(r,node->bounds);append(touched,hit);
        if(!nonempty(hit)||!overwrite(node->command.kind)){append(prior,r);continue;}
        append(prior,{r.left,r.top,r.right,hit.top});append(prior,{r.left,hit.bottom,r.right,r.bottom});
        append(prior,{r.left,hit.top,hit.left,hit.bottom});append(prior,{hit.right,hit.top,r.right,hit.bottom});
    }
    auto before=retain(node->prior,prior,cache,depth+1);
    if(touched.empty())return save(before);
    auto const& c=node->command;Regions source;
    for(auto r:touched){
        if(c.kind==Kind::native_image){
            auto first=[](int at,int src,int dst){return src<=dst?((2*at+1)*src)/(2*dst):at==0?0:((2*at-1)*src)/(2*dst)+1;};
            auto end=[](int at,int src,int dst){return ((2*(at-1)+1)*src)/(2*dst)+1;};
            int w=c.area.right-c.area.left,h=c.area.bottom-c.area.top;
            append(source,{c.source_x+first(r.left-c.area.left,c.source_width,w),c.source_y+first(r.top-c.area.top,c.source_height,h),
                c.source_x+end(r.right-c.area.left,c.source_width,w),c.source_y+end(r.bottom-c.area.top,c.source_height,h)});
        }else append(source,{r.left-c.area.left+c.source_x,r.top-c.area.top+c.source_y,
                            r.right-c.area.left+c.source_x,r.bottom-c.area.top+c.source_y});
    }
    auto from=c.kind==Kind::native_lookup?node->source:retain(node->source,source,cache,depth+1);
    auto background=c.kind==Kind::native_text?node->background:retain(node->background,touched,cache,depth+1);
    auto program=retain(node->program,source,cache,depth+1);
    if(before==node->prior&&from==node->source&&background==node->background&&program==node->program)return save(node);
    auto n=std::make_shared<Node>(node->budget);n->draw=true;n->command=c;n->bounds=node->bounds;
    n->width=node->width;n->height=node->height;n->format=node->format;
    n->prior=std::move(before);n->source=std::move(from);n->background=std::move(background);n->program=std::move(program);return save(n);
}
class Scene {
    struct Image {unsigned width=0,height=0;Format format=Format::rgb555;Ref value;};
    std::unordered_map<Id,Image> images;
    std::shared_ptr<Budget> budget=std::make_shared<Budget>();
public:
    void create(Id id,unsigned w,unsigned h,Format format,unsigned value=0){
        auto n=std::make_shared<Node>(budget);n->width=w;n->height=h;n->format=format;n->constant=value;
        images[id]={w,h,format,n};
    }
    void destroy(Id id){images.erase(id);}
    void upload(Id id,unsigned const* data,std::size_t count){
        auto found=images.find(id);if(found==images.end())return;auto& i=found->second;
        if(count!=std::size_t(i.width)*i.height)throw std::runtime_error("native input coverage upload size");
        if(count*sizeof(unsigned)>96u*1024u*1024u-budget->bytes)throw std::runtime_error("native input coverage byte budget exceeded");
        auto n=std::make_shared<Node>(budget);n->width=i.width;n->height=i.height;n->format=i.format;
        n->pixels.assign(data,data+count);budget->bytes+=count*sizeof(unsigned);i.value=n;
    }
    void submit(Command const& c){
        auto found=images.find(c.destination);if(found==images.end()||found->second.format==Format::bgra32)return;
        auto& i=found->second;auto bounds=intersection(intersection(c.area,c.clip),{0,0,int(i.width),int(i.height)});
        if(bounds.left>=bounds.right||bounds.top>=bounds.bottom)return;
        auto get=[&](Id id)->Ref{auto f=images.find(id);return f==images.end()?Ref{}:f->second.value;};
        if(c.kind==Kind::copy&&c.source==c.destination&&c.source_x==c.area.left&&c.source_y==c.area.top)return;
        bool full=!bounds.left&&!bounds.top&&bounds.right==int(i.width)&&bounds.bottom==int(i.height);
        if(full&&c.kind==Kind::copy&&c.source_x==c.area.left&&c.source_y==c.area.top){
            auto source=get(c.source);if(source&&source->width==i.width&&source->height==i.height){i.value=source;return;}}
        if(full&&(c.kind==Kind::fill||c.kind==Kind::quantize)){
            unsigned value=c.kind==Kind::fill?c.color:opaque_map;
            if(i.value&&!i.value->draw&&i.value->pixels.empty()&&i.value->constant==value)return;
            create(c.destination,i.width,i.height,i.format,value);return;
        }
        auto n=std::make_shared<Node>(budget);n->width=i.width;n->height=i.height;n->format=i.format;
        n->draw=true;n->bounds=bounds;n->command=c;n->source=get(c.source);n->background=get(c.background);n->program=get(c.program);
        if(!overwrite(c.kind)||!full)n->prior=i.value;
        RetentionCache cache;
        i.value=retain(n,{{0,0,int(i.width),int(i.height)}},cache);
    }
    bool pixel(Id id,int x,int y,unsigned& value)const{
        auto f=images.find(id);if(f==images.end()||f->second.format==Format::bgra32)return false;
        value=sample(f->second.value,x,y);return true;
    }
    std::size_t nodes()const{return budget->nodes;}
    std::size_t bytes()const{return budget->bytes;}
};
}
