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
constexpr int tile_size=64;
struct Node;
using Ref=std::shared_ptr<Node const>;
struct Budget {std::size_t nodes=0,bytes=0;};
struct Values {
    std::shared_ptr<Budget> budget;std::vector<unsigned> words;mutable std::uint32_t visit=0;
    Values(std::shared_ptr<Budget> b,std::vector<unsigned> data):budget(std::move(b)),words(std::move(data)){
        if(words.size()*sizeof(unsigned)>96u*1024u*1024u-budget->bytes)throw std::runtime_error("native input coverage byte budget exceeded");
        budget->bytes+=words.size()*sizeof(unsigned);
    }
    ~Values(){budget->bytes-=words.size()*sizeof(unsigned);}
};
struct Node {
    std::shared_ptr<Budget> budget;
    explicit Node(std::shared_ptr<Budget> b):budget(std::move(b)){
        if(budget->nodes>=32768)throw std::runtime_error("native input coverage node budget exceeded");++budget->nodes;
    }
    ~Node(){--budget->nodes;}
    Command command={};Rect bounds={};Format format=Format::rgb555;
    Ref prior,source,background,program;
    int grid_left=0,grid_top=0,grid_columns=0;std::vector<Ref> cells;
    unsigned width=0,height=0,constant=0,depth=0;bool draw=false;
    std::size_t payload_cost=0;
    std::shared_ptr<Values const> pixels;
    // Set on a tile's final history: retaining it again for exactly that
    // tile keeps every reachable value, so later draws reuse it directly
    // instead of walking (up to 24 levels of) its history per tile.
    mutable Rect minimal={};mutable bool minimal_known=false;
    // An image's only upload while it is still that image's current value.
    mutable bool live=false;mutable std::uint32_t visit=0;
};
// Depth and conservative payload follow a node's inputs. Raw pointers: a
// braced list of Refs would copy (and atomically count) every input.
inline void inherit(Node& n){
    for(Node const* ref:{n.prior.get(),n.source.get(),n.background.get(),n.program.get()})if(ref){
        n.depth=std::max(n.depth,ref->depth+1);n.payload_cost=std::min<std::size_t>(96u*1024u*1024u,n.payload_cost+ref->payload_cost);}
}
// Distinct pixel bytes a history pins beyond live uploads, stopping past
// `limit`. Referencing an upload that is still current costs no memory; the
// regional payload bound exists for replaced or destroyed (orphaned) ones.
inline std::size_t orphaned_bytes(Node const* root,std::uint32_t stamp,std::size_t limit){
    std::size_t bytes=0;std::vector<Node const*> pending{root};
    while(!pending.empty()&&bytes<=limit){
        auto node=pending.back();pending.pop_back();
        if(!node||node->visit==stamp)continue;node->visit=stamp;
        if(node->pixels&&!node->live&&node->pixels->visit!=stamp){node->pixels->visit=stamp;bytes+=node->pixels->words.size()*sizeof(unsigned);}
        for(auto const& cell:node->cells)pending.push_back(cell.get());
        for(Node const* ref:{node->prior.get(),node->source.get(),node->background.get(),node->program.get()})pending.push_back(ref);
    }
    return bytes;
}
inline bool inside(Rect r,int x,int y){return x>=r.left&&x<r.right&&y>=r.top&&y<r.bottom;}
inline unsigned sample(Ref const& node,int x,int y,unsigned depth=0){
    if(!node||x<0||y<0||x>=int(node->width)||y>=int(node->height))return 0;
    if(depth>1024)throw std::runtime_error("native input coverage depth exceeded");
    auto read=[&](Ref const& r,int a,int b){return sample(r,a,b,depth+1);};
    if(node->grid_columns){
        int col=x/tile_size-node->grid_left,row=y/tile_size-node->grid_top;
        if(col<0||row<0||col>=node->grid_columns||row>=int(node->cells.size())/node->grid_columns)return 0;
        return read(node->cells[row*node->grid_columns+col],x,y);
    }
    if(!node->draw)return !node->pixels?node->constant:node->pixels->words[std::size_t(y)*node->width+x];
    auto const& c=node->command;
    auto prior=[&]{return read(node->prior,x,y);};
    if(!inside(node->bounds,x,y))return prior();
    if(node->pixels)return node->pixels->words[std::size_t(y-node->bounds.top)*(node->bounds.right-node->bounds.left)+x-node->bounds.left];
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
        // A destination-keyed transfer tests the old destination. On a miss
        // that same sampled value survives; evaluating its history again makes
        // repeated misses branch exponentially before regional compaction.
        if(c.color==4)return below==(p>>16)?word:c.background==c.destination?below:prior();
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
// Prove uniform input coverage from stored regions. This never samples a map
// pixel: map regions contain only the opaque input sentinel. Keyed full-screen
// UI transfers must not build (and periodically evaluate) a history for every
// unchanged map/transparent cell.
inline bool uniform(Ref const& node,Regions const& regions,unsigned& value){
    if(!node||regions.empty())return false;
    bool have=false;
    for(auto r:regions){
        if(!nonempty(r)||r.left<0||r.top<0||r.right>int(node->width)||r.bottom>int(node->height))return false;
        if(node->grid_columns){
            for(int y=r.top/tile_size;y<=(r.bottom-1)/tile_size;++y)
            for(int x=r.left/tile_size;x<=(r.right-1)/tile_size;++x){
                int col=x-node->grid_left,row=y-node->grid_top;
                if(col<0||row<0||col>=node->grid_columns||row>=int(node->cells.size())/node->grid_columns)return false;
                auto part=intersection(r,{x*tile_size,y*tile_size,(x+1)*tile_size,(y+1)*tile_size});unsigned next=0;
                if(!uniform(node->cells[row*node->grid_columns+col],{part},next)||(have&&next!=value))return false;
                value=next;have=true;
            }
        }else{
            if(node->pixels)return false;
            unsigned next=node->constant;
            if(node->draw){
                if(node->command.kind!=Kind::fill||r.left<node->bounds.left||r.top<node->bounds.top||
                   r.right>node->bounds.right||r.bottom>node->bounds.bottom)return false;
                next=node->command.color;
            }
            if(have&&next!=value)return false;value=next;have=true;
        }
    }
    return have;
}
inline Ref retain(Ref const& node,Regions const& needed,RetentionCache& cache,unsigned depth=0){
    if(!node||needed.empty())return {};
    if(node->minimal_known&&needed.size()==1){auto r=needed[0],m=node->minimal;
        if(r.left==m.left&&r.top==m.top&&r.right==m.right&&r.bottom==m.bottom)return node;}
    if(node->grid_columns){
        // Select only the tiles this command reads. A small HUD copy must not
        // keep every unrelated overlay in its source canvas alive.
        int left=int(node->width),top=int(node->height),right=0,bottom=0;
        for(auto r:needed){r=intersection(r,{0,0,int(node->width),int(node->height)});if(!nonempty(r))continue;
            left=std::min(left,r.left/tile_size);top=std::min(top,r.top/tile_size);
            right=std::max(right,(r.right-1)/tile_size+1);bottom=std::max(bottom,(r.bottom-1)/tile_size+1);}
        left=std::max(left,node->grid_left);top=std::max(top,node->grid_top);
        right=std::min(right,node->grid_left+node->grid_columns);
        bottom=std::min(bottom,node->grid_top+int(node->cells.size())/node->grid_columns);
        if(left>=right||top>=bottom)return {};
        if(right-left==1&&bottom-top==1){
            // One tile: the single-cell root below would return its cell.
            Regions clipped;
            for(auto r:needed)append(clipped,intersection(r,{left*tile_size,top*tile_size,(left+1)*tile_size,(top+1)*tile_size}));
            return retain(node->cells[(top-node->grid_top)*node->grid_columns+left-node->grid_left],clipped,cache,depth+1);
        }
        auto root=std::make_shared<Node>(node->budget);root->width=node->width;root->height=node->height;root->format=node->format;
        root->grid_left=left;root->grid_top=top;root->grid_columns=right-left;
        for(int y=top;y<bottom;++y)for(int x=left;x<right;++x){Regions clipped;
            for(auto r:needed)append(clipped,intersection(r,{x*tile_size,y*tile_size,(x+1)*tile_size,(y+1)*tile_size}));
            auto value=retain(node->cells[(y-node->grid_top)*node->grid_columns+x-node->grid_left],clipped,cache,depth+1);
            root->cells.push_back(value);if(value){root->depth=std::max(root->depth,value->depth+1);root->payload_cost=std::min<std::size_t>(96u*1024u*1024u,root->payload_cost+value->payload_cost);}
        }
        return root->cells.size()==1?root->cells[0]:root;
    }
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
    unsigned solid=0;
    bool simple=c.kind==Kind::copy||c.kind==Kind::color_key||c.kind==Kind::invert||c.kind==Kind::native_image;
    if(simple&&uniform(from,source,solid)){
        if((c.kind==Kind::color_key||(c.kind==Kind::native_image&&solid!=opaque_map))&&solid==c.color)return save(before);
        if(c.kind==Kind::invert&&solid!=opaque_map)solid^=65535;
        auto n=std::make_shared<Node>(node->budget);n->draw=true;n->bounds=node->bounds;
        n->command={Kind::fill,c.destination,0,c.area,c.clip,0,0,solid};
        n->width=node->width;n->height=node->height;n->format=node->format;
        bool covered=std::all_of(needed.begin(),needed.end(),[&](Rect r){return r.left>=n->bounds.left&&r.top>=n->bounds.top&&r.right<=n->bounds.right&&r.bottom<=n->bounds.bottom;});
        if(!covered){n->prior=before;if(before){n->depth=before->depth+1;n->payload_cost=before->payload_cost;}}
        return save(n);
    }
    auto background=c.kind==Kind::native_text?node->background:retain(node->background,touched,cache,depth+1);
    auto program=retain(node->program,source,cache,depth+1);
    if(before==node->prior&&from==node->source&&background==node->background&&program==node->program)return save(node);
    auto n=std::make_shared<Node>(node->budget);n->draw=true;n->command=c;n->bounds=node->bounds;
    n->width=node->width;n->height=node->height;n->format=node->format;n->pixels=node->pixels;
    n->prior=std::move(before);n->source=std::move(from);n->background=std::move(background);n->program=std::move(program);
    n->payload_cost=n->pixels?n->pixels->words.size()*sizeof(unsigned):0;
    inherit(*n);
    return save(n);
}
class Scene {
    struct Image {unsigned width=0,height=0;Format format=Format::rgb555;Ref value;unsigned uploads=0;};
    std::unordered_map<Id,Image> images;
    std::shared_ptr<Budget> budget=std::make_shared<Budget>();
    std::uint32_t visits=0;
    std::uint64_t compactions=0,retained_tiles=0;
    static void release(Ref const& value){if(value)value->live=false;}
public:
    void create(Id id,unsigned w,unsigned h,Format format,unsigned value=0){
        auto n=std::make_shared<Node>(budget);n->width=w;n->height=h;n->format=format;n->constant=value;
        auto& image=images[id];release(image.value);image={w,h,format,n};
    }
    void destroy(Id id){auto found=images.find(id);if(found==images.end())return;release(found->second.value);images.erase(found);}
    void upload(Id id,unsigned const* data,std::size_t count){
        auto found=images.find(id);if(found==images.end())return;auto& i=found->second;
        if(count!=std::size_t(i.width)*i.height)throw std::runtime_error("native input coverage upload size");
        if(count*sizeof(unsigned)>96u*1024u*1024u-budget->bytes)throw std::runtime_error("native input coverage byte budget exceeded");
        auto n=std::make_shared<Node>(budget);n->width=i.width;n->height=i.height;n->format=i.format;
        if(i.format==Format::bgra32||count<=tile_size*tile_size){
            n->pixels=std::make_shared<Values>(budget,std::vector<unsigned>(data,data+count));n->payload_cost=count*sizeof(unsigned);
        }else{
            n->grid_columns=(i.width+tile_size-1)/tile_size;
            for(unsigned y=0;y<i.height;y+=tile_size)for(unsigned x=0;x<i.width;x+=tile_size){
                Rect r={int(x),int(y),int(std::min(x+tile_size,i.width)),int(std::min(y+tile_size,i.height))};
                auto cell=std::make_shared<Node>(budget);cell->width=i.width;cell->height=i.height;cell->format=i.format;
                cell->draw=true;cell->bounds=r;cell->command={Kind::fill,id,0,r,r,0,0,data[std::size_t(y)*i.width+x]};
                cell->minimal=r;cell->minimal_known=true; // no history: already its tile's retained form
                std::vector<unsigned> pixels;pixels.reserve((r.right-r.left)*(r.bottom-r.top));
                for(int row=r.top;row<r.bottom;++row)pixels.insert(pixels.end(),data+std::size_t(row)*i.width+r.left,data+std::size_t(row)*i.width+r.right);
                if(!std::all_of(pixels.begin(),pixels.end(),[&](unsigned v){return v==cell->command.color;})){
                    cell->payload_cost=pixels.size()*sizeof(unsigned);cell->pixels=std::make_shared<Values>(budget,std::move(pixels));
                }
                n->payload_cost+=cell->payload_cost;n->cells.push_back(std::move(cell));
            }
            n->depth=1;
        }
        // Re-uploaded sources (sprite preparation, minimap) are never free.
        release(i.value);n->live=n->pixels&&++i.uploads==1;i.value=n;
    }
    void submit(Command const& c){
        auto found=images.find(c.destination);if(found==images.end()||found->second.format==Format::bgra32)return;
        auto& i=found->second;auto bounds=intersection(intersection(c.area,c.clip),{0,0,int(i.width),int(i.height)});
        if(bounds.left>=bounds.right||bounds.top>=bounds.bottom)return;
        auto get=[&](Id id)->Ref{auto f=images.find(id);return f==images.end()?Ref{}:f->second.value;};
        if(c.kind==Kind::copy&&c.source==c.destination&&c.source_x==c.area.left&&c.source_y==c.area.top)return;
        bool full=!bounds.left&&!bounds.top&&bounds.right==int(i.width)&&bounds.bottom==int(i.height);
        if(full&&c.kind==Kind::copy&&c.source_x==c.area.left&&c.source_y==c.area.top){
            auto source=get(c.source);if(source&&source->width==i.width&&source->height==i.height){release(i.value);i.value=source;return;}}
        if(full&&(c.kind==Kind::fill||c.kind==Kind::quantize)){
            unsigned value=c.kind==Kind::fill?c.color:opaque_map;
            if(i.value&&!i.value->draw&&!i.value->grid_columns&&!i.value->pixels&&i.value->constant==value)return;
            create(c.destination,i.width,i.height,i.format,value);return;
        }
        // Capture source versions before changing any destination tiles,
        // including overlapping copies within the same native image.
        auto source=get(c.source),background=get(c.background),program=get(c.program);
        // A large transfer (the per-tick fullscreen unit/HUD canvas onto the
        // screen) is recorded as one deferred node. Eagerly splitting it into
        // every 64-pixel region cost ~15 ms of game-thread allocation per
        // native animation tick. sample() evaluates it directly, and retain()
        // materializes region-local history only where a later draw or query
        // reaches. Bounded depth keeps the region path's compaction authority.
        auto covered=std::int64_t((bounds.right-1)/tile_size-bounds.left/tile_size+1)*
            ((bounds.bottom-1)/tile_size-bounds.top/tile_size+1);
        if(covered>=64&&i.value&&i.value->depth<12){
            auto n=std::make_shared<Node>(budget);n->width=i.width;n->height=i.height;n->format=i.format;
            n->draw=true;n->bounds=bounds;n->command=c;
            n->source=source;n->background=background;n->program=program;n->prior=i.value;
            inherit(*n);
            release(i.value);i.value=std::move(n);return;
        }
        // A full-canvas grid nobody else references (no other canvas or
        // deferred node holds it, and the captured inputs above are not it)
        // is updated in place. Copying its ~650 cell references for every
        // small UI draw dominated this worker under x86 emulation.
        int columns=(int(i.width)+tile_size-1)/tile_size,rows=(int(i.height)+tile_size-1)/tile_size;
        // An aligned keyed transfer (Civ III's per-tick unit/HUD canvas onto
        // the screen) where both tiles already hold their retained history:
        // retain() would return the destination tile unchanged when the source
        // tile is uniformly the key, else one node over those two tiles. Build
        // that result directly instead of ~10 allocations per tile.
        bool keyed=(c.kind==Kind::color_key||(c.kind==Kind::native_image&&c.source_width==c.area.right-c.area.left&&
            c.source_height==c.area.bottom-c.area.top))&&c.source_x==c.area.left&&c.source_y==c.area.top&&!background&&!program&&
            source&&source->width==i.width&&source->height==i.height&&(!source->grid_columns||
            (source->grid_columns==columns&&!source->grid_left&&!source->grid_top&&source->cells.size()==std::size_t(columns)*rows));
        // 0: use retain(); 1: tile unchanged; 2: `made` is retain()'s result.
        RetentionCache cache;
        auto direct=[&](int tx,int ty,Ref const& before,Ref* made)->int{
            Rect tile={tx*tile_size,ty*tile_size,std::min((tx+1)*tile_size,int(i.width)),std::min((ty+1)*tile_size,int(i.height))};
            auto exact=[&](Ref const& r){auto m=r->minimal;return (!r->draw&&!r->grid_columns)||
                (r->minimal_known&&m.left==tile.left&&m.top==tile.top&&m.right==tile.right&&m.bottom==tile.bottom);};
            auto inner=intersection(bounds,tile);unsigned solid=0;
            if(inner.left!=tile.left||inner.top!=tile.top||inner.right!=tile.right||inner.bottom!=tile.bottom||
               !before||!exact(before))return 0;
            // retain() would take the source tile's retained form, alone in
            // its cache because the destination tile returns immediately.
            Ref from=source->grid_columns?source->cells[std::size_t(ty)*columns+tx]:source;
            if(!from)return 0;
            if(!exact(from)){if(!made)return 0;cache.clear();from=retain(source,{tile},cache);if(!from)return 0;}
            bool uniform_source=uniform(from,{tile},solid);
            if(uniform_source&&solid==c.color&&(c.kind==Kind::color_key||solid!=opaque_map))return 1;
            if(!made)return 0;
            auto n=std::make_shared<Node>(budget);n->draw=true;n->bounds=tile;n->width=i.width;n->height=i.height;n->format=i.format;
            if(uniform_source)n->command={Kind::fill,c.destination,0,c.area,c.clip,0,0,solid};
            else{n->command=c;n->prior=before;n->source=from;inherit(*n);}
            *made=std::move(n);return 2;
        };
        auto current=[&](int tx,int ty)->Ref const&{return i.value->grid_columns?i.value->cells[std::size_t(ty)*columns+tx]:i.value;};
        if(keyed&&(!i.value->grid_columns||(i.value->grid_columns==columns&&!i.value->grid_left&&!i.value->grid_top&&
           i.value->cells.size()==std::size_t(columns)*rows))){
            bool all=true;
            for(int ty=bounds.top/tile_size;all&&ty<=(bounds.bottom-1)/tile_size;++ty)
            for(int tx=bounds.left/tile_size;all&&tx<=(bounds.right-1)/tile_size;++tx)all=direct(tx,ty,current(tx,ty),nullptr)==1;
            if(all)return;
        }else keyed=false;
        std::shared_ptr<Node> root;
        if(i.value.use_count()==1&&i.value->grid_columns==columns&&!i.value->grid_left&&!i.value->grid_top&&
           i.value->cells.size()==std::size_t(columns)*rows&&!i.value->draw&&!i.value->pixels){
            root=std::const_pointer_cast<Node>(i.value);root->depth=0;root->payload_cost=0;
        }else{
            root=std::make_shared<Node>(budget);root->width=i.width;root->height=i.height;root->format=i.format;
            root->grid_columns=columns;
            if(i.value->grid_columns)root->cells=i.value->cells;
            else root->cells.assign(std::size_t(columns)*rows,i.value);
        }
        for(int ty=bounds.top/tile_size;ty<=(bounds.bottom-1)/tile_size;++ty)
        for(int tx=bounds.left/tile_size;tx<=(bounds.right-1)/tile_size;++tx){
            Rect tile={tx*tile_size,ty*tile_size,std::min((tx+1)*tile_size,int(i.width)),std::min((ty+1)*tile_size,int(i.height))};
            auto& before=root->cells[ty*root->grid_columns+tx];
            Ref retained;int shortcut=keyed?direct(tx,ty,before,&retained):0;
            if(shortcut==1)continue;
            auto inner=intersection(bounds,tile);
            if(!shortcut&&c.kind==Kind::fill&&!source&&!background&&!program&&inner.left==tile.left&&inner.top==tile.top&&
               inner.right==tile.right&&inner.bottom==tile.bottom){
                // retain() keeps nothing below a fill covering its whole tile.
                auto n=std::make_shared<Node>(budget);n->width=i.width;n->height=i.height;n->format=i.format;
                n->draw=true;n->bounds=tile;n->command=c;retained=std::move(n);shortcut=2;
            }
            if(!shortcut){
                auto n=std::make_shared<Node>(budget);n->width=i.width;n->height=i.height;n->format=i.format;
                n->draw=true;n->bounds=intersection(bounds,tile);n->command=c;
                n->source=source;n->background=background;n->program=program;n->prior=before;
                inherit(*n);
                cache.clear();retained=retain(n,{tile},cache);++retained_tiles;
            }
            // History is bounded independently in each screen region. Compact
            // only that region's input values; never evaluate a full map image
            // or send these values to the renderer/native pixel buffers.
            // A shallow chain can still pin many changing sprite uploads.
            // Bound its payload to two regional value arrays, independently of
            // the depth limit. A single-upload source that is still current
            // (sprite sheet, text raster) costs nothing extra to reference, and
            // compacting every draw from one dominated this worker; it is
            // counted again once replaced or destroyed, and above 16 MB of
            // retained values the conservative cached payload (shared references
            // counted twice) applies alone. Map samples remain the opaque
            // sentinel, with no terrain pixels or GPU work involved.
            auto regional_bytes=std::size_t(tile.right-tile.left)*(tile.bottom-tile.top)*sizeof(unsigned);
            if(retained->depth>=24||(retained->payload_cost>2*regional_bytes&&(budget->bytes>16u*1024u*1024u||
               orphaned_bytes(retained.get(),++visits,2*regional_bytes)>2*regional_bytes))){
                ++compactions;std::vector<unsigned> points;points.reserve(std::size_t(tile.right-tile.left)*(tile.bottom-tile.top));
                for(int y=tile.top;y<tile.bottom;++y)for(int x=tile.left;x<tile.right;++x)points.push_back(sample(retained,x,y));
                auto compact=std::make_shared<Node>(budget);compact->draw=true;compact->command={Kind::fill,c.destination,0,tile,tile};
                compact->bounds=tile;compact->width=i.width;compact->height=i.height;compact->format=i.format;
                if(std::all_of(points.begin(),points.end(),[&](unsigned v){return v==points[0];}))compact->command.color=points[0];
                else {compact->pixels=std::make_shared<Values>(budget,std::move(points));compact->payload_cost=regional_bytes;}
                retained=compact;
            }
            retained->minimal=tile;retained->minimal_known=true;
            before=std::move(retained);
        }
        for(auto const& cell:root->cells)if(cell){root->depth=std::max(root->depth,cell->depth+1);root->payload_cost=std::min<std::size_t>(96u*1024u*1024u,root->payload_cost+cell->payload_cost);}
        release(i.value);i.value=std::move(root);
    }
    bool pixel(Id id,int x,int y,unsigned& value)const{
        auto f=images.find(id);if(f==images.end()||f->second.format==Format::bgra32)return false;
        value=sample(f->second.value,x,y);return true;
    }
    std::size_t nodes()const{return budget->nodes;}
    // Regional value compactions and tiles retained through the general path.
    std::uint64_t compacted_tiles()const{return compactions;}
    std::uint64_t general_tiles()const{return retained_tiles;}
    std::size_t bytes()const{return budget->bytes;}
};
}
