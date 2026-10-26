"""Small native presents must not keep whole screen versions alive.

In diplomacy Civ III redraws its whole form every leader-head tick (~55 a
second): the animated map, a full-screen keyed interface layer, the HUD
icons and the leader frame, almost all identical. It presents only the
leader rectangle. Hovering the trade table's technology links re-uploads
the form panel as each link's highlight turns on and off, presents that
line, and moves Civ III's software cursor, whose restores and 32x32
presents reach every part of the panel.

A partial present keeps the rest of the front on the version it last
showed. The hover pinned eight complete screen versions, each a full-screen
map-dependent expand with its own HUD icons, all re-evaluated every frame:
26 ms frames and native presents blocked 660 ms of every second (user
capture and the diplomacy-hover scenario, October 26).

The production `record`/`commit`/`upload_source` are replayed against a
pixel model. The front must show exactly the pixels of the unchanged path
at every step, while it references a bounded number of versions and the
live graph stays small. The unchanged path is the defect being guarded.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp
from Renderer.native.test_cursor_save_restore import method


def owner(name, methods, uploader):
    return r'''
struct ''' + name + r''' {
 using Direct=::Direct;
 struct Node;
 struct Patch {Rect area;std::shared_ptr<Node> node;unsigned output=0;};
 struct Picture {unsigned width=0,height=0;Format format=Format::bgra32;std::vector<Patch> patches;bool partitioned=true;std::uint64_t version=0;Rect required{};};
 struct Node {Rect area{},recipe_clip{};Command command{};bool selected_world=false,operation=false,dynamic=false,map_dynamic=false,constant=false,view_dependent=false,retired=false,projects_scene=false;
  Picture inputs[6];Id original[6]={};Direct direct;std::shared_ptr<c3x_renderer::ZoomTransition> placement,view;std::shared_ptr<Node> projected;
  Sample sample;std::vector<int> batch;int anchor_x=0,anchor_y=0;int content=0;bool output[2]={};};
 std::map<Id,Picture> images;Picture front;std::uint64_t serial=0,front_revision=0,direct_bytes=0;bool admitted=true;std::size_t nodes=0;
 static bool same_rect(Rect a,Rect b){return a.left==b.left&&a.top==b.top&&a.right==b.right&&a.bottom==b.bottom;}
 static bool same_picture(Picture const& a,Picture const& b,bool require_version=true){
  if(a.width!=b.width||a.height!=b.height||a.format!=b.format||a.partitioned!=b.partitioned||
     (require_version&&a.version!=b.version)||!same_rect(a.required,b.required)||a.patches.size()!=b.patches.size())return false;
  for(std::size_t i=0;i<a.patches.size();++i){auto const& x=a.patches[i];auto const& y=b.patches[i];
   if(x.node!=y.node||x.output!=y.output||!same_rect(x.area,y.area))return false;}return true;}
 Rect intersect(Rect a,Rect b)const{return intersection(a,b);}
 bool empty(Rect r)const{return r.left>=r.right||r.top>=r.bottom;}
 Rect extent(Picture const& p)const{return {0,0,int(p.width),int(p.height)};}
 void reserve(std::uint64_t,char const*){}
 std::shared_ptr<Node> reuse_recipe(std::shared_ptr<Node> const& n){return n;}
 std::uint64_t node_serial=0;
 std::shared_ptr<Node> node(){++nodes;++node_serial;return std::shared_ptr<Node>(new Node,[this](Node* p){delete p;--nodes;});}
 void create(Id id,unsigned w,unsigned h){images[id]={w,h,Format::rgb555,{}};images[id].version=++serial;}
 // An upload snapshot: immutable source pixels, identified by content.
 void source(Id id,ID3D11Texture2D* texture){auto& p=images.at(id);auto n=node();n->area=extent(p);n->content=texture->content;
  n->dynamic=n->map_dynamic=texture->animated;n->output[0]=true;p.patches={{extent(p),n,0}};p.version=++serial;}
 void upload(Id id,ID3D11Texture2D* texture){auto const& p=images.at(id);std::vector<unsigned> pixels(std::size_t(p.width)*p.height);
  for(unsigned y=0;y<p.height;++y)for(unsigned x=0;x<p.width;++x)pixels[std::size_t(y)*p.width+x]=unsigned(content_pixel(texture->content,int(x),int(y),0));
  (void)pixels;''' + uploader + r'''}
''' + methods + '\n};\n'


SIGNATURES = (
    '    Picture read(Id id,Rect region){', '    void replace(Picture& image,',
    '    void write(Picture& image,', '    void record(Command const& c,',
    '    void commit(Id image,Rect area){',
    '    bool pixel_local(Node const& n)const{',
    '    bool same_pixels(Node const& a,Node const& b,Rect r,unsigned depth,unsigned& budget)const{',
    '    bool same_region(Picture const& a,Picture const& b,Rect r,unsigned depth,unsigned& budget)const{',
    '    std::vector<Patch> keep_current(Id image,std::vector<Patch> const& selected){',
    '    void rebase_front(Id image){',
    '    bool cropped(Node const& n)const{',
    '    bool crop_patches(std::vector<Patch>& patches){',
    '    std::shared_ptr<Node> crop(std::shared_ptr<Node> const& n,Rect a){',
    '    std::shared_ptr<Node> rebase_inputs(std::shared_ptr<Node> const& n,unsigned& budget){',
    '    void coalesce(Picture& p){',
    '    void upload_source(Id id,ID3D11Texture2D* texture,unsigned const* pixels,std::size_t count){')
TILES = '    struct UploadTiles {std::uint64_t version=0;std::vector<std::uint64_t> hashes;};\n' \
        '    std::map<Id,UploadTiles> upload_tiles;\n'
CROPS = '    struct CropEntry {std::weak_ptr<Node> whole,crop;};\n' \
        '    std::map<Node const*,CropEntry> crop_cache;\n'
PRELUDE = r'''
#include "Renderer/native/gpu_image_commands.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <stdexcept>
#include <vector>
namespace c3x_renderer {struct ZoomTransition {static constexpr double maximum=3.;};}
using namespace c3x_gpu_images;
struct Compositor {};
struct Direct {bool animated=false;std::uint64_t input_bytes=0;std::function<std::uint64_t(long long,long long)> revision;std::function<bool(Compositor&,Command const&)> draw;};
struct Sample {explicit operator bool()const{return false;}};
struct ID3D11Texture2D {int content=0;bool animated=false;};
'''


def production_methods(test):
    source = (ROOT / 'Renderer/native/retained_composition.h').read_text()
    test.assertIn(TILES, source)
    test.assertIn(CROPS, source)
    return TILES + CROPS + '\n'.join(method(source, s) for s in SIGNATURES)


class FrontRebaseTests(unittest.TestCase):
    def test_partial_presents_do_not_pin_identical_screen_versions(self):
        methods = production_methods(self)
        call = '        rebase_front(image);\n'
        restore = 'replace(target->second,area,keep_current(c.destination,selected));'
        self.assertIn(call, methods)
        self.assertIn(restore, methods)
        general = methods.replace(call, '').replace(restore, 'replace(target->second,area,selected);')
        run_cpp(PRELUDE + r'''
// Upload content: the map moves with time t. The form's highlight states
// differ only inside the highlighted line, like Civ III's panel.
constexpr int line_left=10,line_top=40,line_right=110,line_bottom=56;
int content_pixel(int content,int x,int y,int t){
 if(content>=6&&content<=9&&!(x>=line_left&&x<line_right&&y>=line_top&&y<line_bottom))content=6;
 return content*1000+(content==1?(x*7+y*3+t)%997:(x*5+y*11)%991);
}
''' + owner('Current', methods, 'upload_source(id,texture,pixels.data(),pixels.size());') + owner('General', general, 'source(id,texture);') + r'''
constexpr std::uint32_t key=31775;
// Pixel model: uploads are content-seeded (the map also moves with time t);
// keyed draws show the source unless its pixel is the key.
template<class O> int pixel(typename O::Picture const& p,int x,int y,int t);
template<class O> int node_pixel(typename O::Node const& n,int x,int y,int t){
 auto const& c=n.command;
 if(!n.operation)return content_pixel(n.content,x,y,t);
 if(c.kind==Kind::fill)return int(c.color);
 int sx=x-(c.area.left-c.source_x),sy=y-(c.area.top-c.source_y);
 if(c.kind==Kind::copy)return pixel<O>(n.inputs[1],sx,sy,t);
 if(c.kind==Kind::native_sprite){int lx=x-c.area.left,ly=y-c.area.top;return (lx+ly)%3?7000+lx*8+ly:pixel<O>(n.inputs[0],x,y,t);}
 assert(c.kind==Kind::native_image);
 int s=pixel<O>(n.inputs[1],sx,sy,t);return s%5==0?pixel<O>(n.inputs[0],x,y,t):s;
}
template<class O> int pixel(typename O::Picture const& p,int x,int y,int t){
 for(auto const& q:p.patches)if(x>=q.area.left&&x<q.area.right&&y>=q.area.top&&y<q.area.bottom)return node_pixel<O>(*q.node,x,y,t);
 return -1; // not yet presented
}
template<class O> void reach(typename O::Node const* n,std::set<void const*>& seen){
 if(!seen.insert(n).second)return;for(auto const& input:n->inputs)for(auto const& q:input.patches)reach<O>(q.node.get(),seen);}
template<class O> std::size_t front_nodes(O const& o){std::set<void const*> seen;for(auto const& q:o.front.patches)reach<O>(q.node.get(),seen);return seen.size();}
constexpr int W=320,H=200,S=32;constexpr Id screen=1,save=2,map=3,ui=4,leader=5,icon=6,cursor=7,form=8;
constexpr Rect head={110,10,170,70};
ID3D11Texture2D textures[64];
ID3D11Texture2D* texture(int content,bool animated=false){auto& t=textures[content%64];t.content=content;t.animated=animated;return &t;}
template<class O> void setup(O& o){
 o.create(screen,W,H);o.create(save,S,S);o.create(map,W,H);o.create(ui,W,H);o.create(leader,60,60);o.create(icon,12,10);o.create(cursor,S,S);o.create(form,W,H);
 o.upload(map,texture(1,true));o.upload(ui,texture(2));o.upload(leader,texture(3));o.upload(icon,texture(4));o.upload(cursor,texture(5));o.upload(form,texture(6));
 Command clear={Kind::fill,save,0,{0,0,S,S},{0,0,S,S},0,0,9};o.record(clear);
}
template<class O> void draw(O& o,Id from,Rect area,int sx=0,int sy=0){
 Command c={Kind::native_image,screen,from,area,{0,0,W,H},sx,sy,key,0,0,0,area.right-area.left,area.bottom-area.top};o.record(c);}
template<class O> void copy(O& o,Id from,Id to,Rect area,Rect clip,int sx,int sy){Command c={Kind::copy,to,from,area,clip,sx,sy};o.record(c);}
// One leader-head tick with a cursor move over the panel's links.
template<class O> void tick(O& o,int k,int& px,int& py){
 if(k%8==0)o.upload(leader,texture(10+k/8%40));
 // The hovered link's highlight cycles through four panel states.
 bool highlight=k%6==0;if(highlight)o.upload(form,texture(6+(k/6)%4));
 Command base={Kind::copy,screen,map,{0,0,W,H},{0,0,W,H},0,0};o.record(base); // version selection, no node
 draw(o,ui,{0,0,W,H});
 for(int i=0;i<21;++i)draw(o,icon,{14*(i%7)+200,182+(i/7)*6-12,14*(i%7)+212,182+(i/7)*6-2});
 copy(o,leader,screen,head,{0,0,W,H},0,0);
 // The diplomacy form is keyed over everything drawn this tick.
 draw(o,form,{0,0,W,H});
 // The scenario's hover: 30 moves per loop around the technology links.
 int x=int(std::lround(40+30*std::cos(6.2831853*k/30))),y=int(std::lround(60+17*std::sin(6.2831853*k/30)));
 if(k)copy(o,save,screen,{px,py,px+S,py+S},{0,0,W,H},0,0);
 copy(o,screen,save,{0,0,S,S},{0,0,S,S},x,y);
 Command sprite={Kind::native_sprite,screen,0,{x,y,x+S,y+S},{0,0,W,H}};o.record(sprite);
 o.commit(screen,head);
 if(highlight)o.commit(screen,{line_left,line_top,line_right,line_bottom});
 if(k){o.commit(screen,{px,py,px+S,py+S});}
 o.commit(screen,{x,y,x+S,y+S});
 px=x;py=y;
}
int main(){
 Current current;General general;setup(current);setup(general);int cx=0,cy=0,gx=0,gy=0;
 std::size_t most=0,most_front=0,early=0;
 for(int k=0;k<300;++k){
  tick(current,k,cx,cy);tick(general,k,gx,gy);
  // Opening the form presents it whole.
  if(!k){current.commit(screen,{0,0,W,H});general.commit(screen,{0,0,W,H});}
  most=std::max(most,current.nodes);most_front=std::max(most_front,front_nodes(current));if(k==149)early=most;
  if(k%10==0||k==299)for(int t:{0,7,40})for(int y=0;y<H;++y)for(int x=0;x<W;++x)
   if(pixel<Current>(current.front,x,y,t)!=pixel<General>(general.front,x,y,t)){
    std::fprintf(stderr,"tick %d front pixel %d,%d t=%d differs\n",k,x,y,t);return 1;}
 }
 std::fprintf(stderr,"current nodes=%zu front graph=%zu; general nodes=%zu front graph=%zu\n",most,most_front,general.nodes,front_nodes(general));
 // The unchanged path pins a screen version per cursor move and keeps
 // growing. The rebased front holds a few versions (tile edges just outside
 // a changed line) and stays bounded: no growth after the first half.
 assert(general.nodes>500&&front_nodes(general)>500);
 assert(most<150&&most_front<150&&most==early);
 return 0;
}
''')


    def test_unprovable_versions_cost_only_what_they_show(self):
        # Civ III redraws the panel's technology list from a new temporary
        # image for each hovered link and re-uploads the whole panel every few
        # seconds; neither proves the old screen version identical. Each
        # version is a full-screen keyed expand over the animated map, so the
        # front used to evaluate a whole screen per version it still showed.
        methods = production_methods(self)
        crops = {'auto n=crop(f.node,pixels(v.bounds)<=2*v.pixels?v.bounds:f.area);': 'auto n=f.node;'}
        general = methods
        for call, off in crops.items():
            self.assertTrue(call in methods, call)
            general = general.replace(call, off)
        run_cpp(PRELUDE + r'''
int content_pixel(int content,int x,int y,int t){return content*1000+(content==1?(x*7+y*3+t)%997:(x*5+y*11+content)%991);}
''' + owner('Current', methods, 'upload_source(id,texture,pixels.data(),pixels.size());') +
            owner('General', general, 'upload_source(id,texture,pixels.data(),pixels.size());') + r'''
constexpr std::uint32_t key=31775;
template<class O> int pixel(typename O::Picture const& p,int x,int y,int t);
template<class O> int node_pixel(typename O::Node const& n,int x,int y,int t){
 auto const& c=n.command;
 if(!n.operation)return content_pixel(n.content,x,y,t);
 if(c.kind==Kind::fill)return int(c.color);
 int sx=x-(c.area.left-c.source_x),sy=y-(c.area.top-c.source_y);
 if(c.kind==Kind::copy)return pixel<O>(n.inputs[1],sx,sy,t);
 if(c.kind==Kind::native_sprite){int lx=x-c.area.left,ly=y-c.area.top;return (lx+ly)%3?7000+lx*8+ly:pixel<O>(n.inputs[0],x,y,t);}
 assert(c.kind==Kind::expand);
 int s=pixel<O>(n.inputs[1],sx,sy,t);return s%5==0?pixel<O>(n.inputs[0],x,y,t):s;
}
template<class O> int pixel(typename O::Picture const& p,int x,int y,int t){
 for(auto const& q:p.patches)if(x>=q.area.left&&x<q.area.right&&y>=q.area.top&&y<q.area.bottom)return node_pixel<O>(*q.node,x,y,t);
 return -1;
}
// Pixels of animated operations the front reaches: what a frame evaluates.
template<class O> void reach(typename O::Node const* n,std::set<void const*>& seen,std::uint64_t& cost){
 if(!seen.insert(n).second)return;if(n->operation&&n->dynamic)cost+=std::uint64_t(n->area.right-n->area.left)*(n->area.bottom-n->area.top);
 for(auto const& input:n->inputs)for(auto const& q:input.patches)reach<O>(q.node.get(),seen,cost);}
template<class O> std::uint64_t frame_cost(O const& o){std::set<void const*> seen;std::uint64_t cost=0;for(auto const& q:o.front.patches)reach<O>(q.node.get(),seen,cost);return cost;}
constexpr int W=320,H=200,S=32;constexpr Id screen=1,save=2,map=3,panel=4,cursor=5;
ID3D11Texture2D textures[64];
ID3D11Texture2D* texture(int content,bool animated=false){auto& t=textures[content%64];t.content=content;t.animated=animated;return &t;}
template<class O> void setup(O& o){
 o.create(screen,W,H);o.create(save,S,S);o.create(map,W,H);o.create(panel,W,H);o.create(cursor,S,S);
 o.upload(map,texture(1,true));o.upload(panel,texture(6));o.upload(cursor,texture(5));
 Command clear={Kind::fill,save,0,{0,0,S,S},{0,0,S,S},0,0,9};o.record(clear);
}
template<class O> void copy(O& o,Id from,Id to,Rect area,Rect clip,int sx,int sy){Command c={Kind::copy,to,from,area,clip,sx,sy};o.record(c);}
template<class O> void tick(O& o,int k,int& px,int& py){
 // The hovered link's list redraw, from a new temporary image.
 Rect list=k%2?Rect{20,100,140,150}:Rect{20,40,140,90};Id temporary=100+k;
 o.create(temporary,120,50);o.upload(temporary,texture(20+k%30));copy(o,temporary,panel,list,{0,0,W,H},0,0);
 if(k%7==3)o.upload(panel,texture(6+k%5)); // the whole panel again
 Command base={Kind::copy,screen,map,{0,0,W,H},{0,0,W,H},0,0};o.record(base);
 Command form={Kind::expand,screen,panel,{0,0,W,H},{0,0,W,H},0,0,key};o.record(form);
 int x=int(std::lround(40+30*std::cos(6.2831853*k/30))),y=int(std::lround(60+17*std::sin(6.2831853*k/30)));
 if(k)copy(o,save,screen,{px,py,px+S,py+S},{0,0,W,H},0,0);
 copy(o,screen,save,{0,0,S,S},{0,0,S,S},x,y);
 Command sprite={Kind::native_sprite,screen,0,{x,y,x+S,y+S},{0,0,W,H}};o.record(sprite);
 o.commit(screen,list);
 if(k)o.commit(screen,{px,py,px+S,py+S});
 o.commit(screen,{x,y,x+S,y+S});
 px=x;py=y;
}
int main(){
 Current current;General general;setup(current);setup(general);int cx=0,cy=0,gx=0,gy=0;
 std::uint64_t most=0,most_general=0;
 for(int k=0;k<120;++k){
  tick(current,k,cx,cy);tick(general,k,gx,gy);
  if(!k){current.commit(screen,{0,0,W,H});general.commit(screen,{0,0,W,H});}
  if(k>=2){most=std::max(most,frame_cost(current));most_general=std::max(most_general,frame_cost(general));}
  if(k%5==0||k==119)for(int t:{0,9})for(int y=0;y<H;++y)for(int x=0;x<W;++x)
   if(pixel<Current>(current.front,x,y,t)!=pixel<General>(general.front,x,y,t)){
    std::fprintf(stderr,"tick %d front pixel %d,%d t=%d differs\n",k,x,y,t);return 1;}
 }
 std::fprintf(stderr,"evaluated screens per frame: cropped %.2f, whole versions %.2f; nodes made per tick: cropped %.1f, whole %.1f\n",
  double(most)/(W*H),double(most_general)/(W*H),double(current.node_serial)/120,double(general.node_serial)/120);
 // Whole versions: one full screen per version still shown. Cropped: a
 // node is cut once it shows less than half of itself, so each shown pixel
 // is evaluated at most twice.
 assert(most_general>=4ull*W*H);
 assert(most<2ull*W*H);
 return 0;
}
''')


    def test_sprite_before_image_strips_share_one_crop(self):
        # Civ III's cursor restores leave the sprite's before-image as one-pixel
        # row strips of a full-screen version. Each strip became its own crop:
        # a pass and an output allocation per strip, about 14 per frame while
        # hovering the city screen's Civilopedia links (pedia-hover).
        methods = production_methods(self)
        call = 'changed|=crop_patches(next);'
        self.assertTrue(call in methods, call)
        per_strip = methods.replace(call, 'for(auto& q:next){auto c=crop(q.node,q.area);changed|=c!=q.node;q.node=std::move(c);}')
        run_cpp(PRELUDE + r'''
int content_pixel(int content,int,int,int){return content;}
''' + owner('Current', methods, 'source(id,texture);') + owner('PerStrip', per_strip, 'source(id,texture);') + r'''
template<class O> std::size_t crops(){
 O o;constexpr Id screen=1,panel=2,map=3;Rect full={0,0,320,200},under={100,50,126,81};
 for(Id id:{screen,panel,map})o.create(id,320,200);
 ID3D11Texture2D animated{1,true},still{2,false};o.upload(map,&animated);o.upload(panel,&still);
 Command base={Kind::copy,screen,map,full,full,0,0};o.record(base);
 Command form={Kind::expand,screen,panel,full,full,0,0,31775};o.record(form);
 auto version=o.images.at(screen).patches.front().node;assert(version->operation&&version->dynamic);
 auto sprite=o.node();sprite->operation=true;sprite->dynamic=true;sprite->area=under;
 sprite->command={Kind::native_sprite,screen,0,under,full};sprite->original[0]=screen;
 sprite->inputs[0]={320,200,Format::rgb555,{},true};
 for(int y=under.top;y<under.bottom;++y)sprite->inputs[0].patches.push_back({{under.left,y,under.right,y+1},version,0});
 unsigned budget=8192;auto rebased=o.rebase_inputs(sprite,budget);
 std::set<void const*> nodes;for(auto const& q:rebased->inputs[0].patches){nodes.insert(q.node.get());
  assert(q.node!=version&&q.node->area.left<=q.area.left&&q.node->area.right>=q.area.right&&q.node->area.top<=q.area.top&&q.node->area.bottom>=q.area.bottom);}
 return nodes.size();
}
int main(){
 auto shared=crops<Current>(),strips=crops<PerStrip>();
 std::fprintf(stderr,"crops for one sprite: shared %zu, per strip %zu\n",shared,strips);
 assert(strips==31);assert(shared==1);
 return 0;
}
''')


if __name__ == '__main__':
    unittest.main()
