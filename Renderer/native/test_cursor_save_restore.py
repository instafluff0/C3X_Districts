"""Civ III's software cursor must not chain retained copies.

With a sprite cursor (Civilopedia links in diplomacy, the city screen) Civ
III copies the screen under the cursor into a small save image, draws the
sprite, and on the next move copies the save back before saving the new
spot. A form redraw under the cursor re-saves just that part. Each restore
used to execute as a node reading the previous save, which read the
previous restore: an animated chain that grew by three nodes per mouse
move. Every frame re-evaluated it (1040 nodes, 141 M copied pixels, 218 ms)
and each native present waited for that frame (user capture, October 21).

The production `record` is replayed against a pixel model. Its pictures
must equal the general execution path pixel for pixel at several animation
times, while the restore keeps neither a growing chain nor growing nodes.
"""
import unittest

from Renderer.lab.platform import ROOT
from Renderer.native.native_cpp_test import run_cpp


def method(source, signature):
    begin = source.index(signature)
    # record's default arguments contain braces; the body follows ')'.
    brace = source.index('){', begin) + 1
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


def owner(name, methods):
    return r'''
struct ''' + name + r''' {
 using Direct=::Direct;
 struct Node;
 struct Patch {Rect area;std::shared_ptr<Node> node;unsigned output=0;};
 struct Picture {unsigned width=0,height=0;Format format=Format::bgra32;std::vector<Patch> patches;bool partitioned=true;std::uint64_t version=0;Rect required{};};
 struct Node {Rect area{},recipe_clip{};Command command{};bool operation=false,dynamic=false,map_dynamic=false,constant=false,view_dependent=false,retired=false;
  Picture inputs[6];Id original[6]={};Direct direct;std::shared_ptr<c3x_renderer::ZoomTransition> placement;int anchor_x=0,anchor_y=0;};
 std::map<Id,Picture> images;std::uint64_t serial=0,direct_bytes=0;bool admitted=true;std::size_t nodes=0;
 Rect intersect(Rect a,Rect b)const{return intersection(a,b);}
 bool empty(Rect r)const{return r.left>=r.right||r.top>=r.bottom;}
 Rect extent(Picture const& p)const{return {0,0,int(p.width),int(p.height)};}
 void reserve(std::uint64_t,char const*){}
 std::shared_ptr<Node> reuse_recipe(std::shared_ptr<Node> const& n){return n;}
 std::shared_ptr<Node> node(){++nodes;return std::shared_ptr<Node>(new Node,[this](Node* p){delete p;--nodes;});}
 void create(Id id,unsigned w,unsigned h){images[id]={w,h,Format::rgb555,{}};images[id].version=++serial;}
''' + methods + '\n};\n'


class CursorSaveRestoreTests(unittest.TestCase):
    def test_restore_selects_the_saved_version_without_chaining(self):
        source = (ROOT / 'Renderer/native/retained_composition.h').read_text()
        methods = '\n'.join(method(source, s) for s in (
            '    Picture read(Id id,Rect region){', '    void replace(Picture& image,',
            '    void write(Picture& image,', '    void record(Command const& c,'))
        begin = methods.index("        // Civ III's software cursor")
        end = methods.index("        // JGL's unscaled opaque transfer")
        general = methods[:begin] + methods[end:]
        run_cpp(r'''
#include "Renderer/native/gpu_image_commands.h"
#include <cassert>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
namespace c3x_renderer {struct ZoomTransition {static constexpr double maximum=3.;};}
using namespace c3x_gpu_images;
// Outside the owners: a nested default member initializer cannot serve
// record's `Direct direct={}` default argument.
struct Compositor {};
struct Direct {bool animated=false;std::uint64_t input_bytes=0;std::function<std::uint64_t(long long,long long)> revision;std::function<bool(Compositor&,Command const&)> draw;};
''' + owner('Current', methods) + owner('General', general) + r'''
// Pixel model: the base is the animated map (time t), fills are UI, copies
// read their captured source version, the cursor sprite keys every third pixel.
template<class O> int pixel(typename O::Picture const& p,int x,int y,int t);
template<class O> int node_pixel(typename O::Node const& n,int x,int y,int t){
 auto const& c=n.command;
 if(!n.operation)return x*131+y*17+t;
 if(c.kind==Kind::fill)return int(c.color);
 if(c.kind==Kind::copy)return pixel<O>(n.inputs[1],x-(c.area.left-c.source_x),y-(c.area.top-c.source_y),t);
 assert(c.kind==Kind::native_sprite);int lx=x-c.area.left,ly=y-c.area.top;
 return (lx+ly)%3?7000+lx*8+ly:pixel<O>(n.inputs[0],x,y,t);
}
template<class O> int pixel(typename O::Picture const& p,int x,int y,int t){
 for(auto const& q:p.patches)if(x>=q.area.left&&x<q.area.right&&y>=q.area.top&&y<q.area.bottom)return node_pixel<O>(*q.node,x,y,t);
 assert(!"uncovered pixel");return -1;
}
template<class O> int depth(typename O::Node const* n,std::map<void const*,int>& seen){
 auto found=seen.find(n);if(found!=seen.end())return found->second;int d=0;
 for(auto const& input:n->inputs)for(auto const& q:input.patches)d=std::max(d,depth<O>(q.node.get(),seen));
 return seen[n]=d+1;
}
template<class O> int chain(O const& o,Id id){std::map<void const*,int> seen;int d=0;
 for(auto const& q:o.images.at(id).patches)d=std::max(d,depth<O>(q.node.get(),seen));return d;}
constexpr int W=64,H=48,S=8;constexpr Id screen=1,save=2;
template<class O> void setup(O& o){
 o.create(screen,W,H);o.create(save,S,S);
 auto base=o.node();base->area={0,0,W,H};base->dynamic=base->map_dynamic=true;
 o.images[screen].patches={{{0,0,W,H},base,0}};
 // A new JGL image starts cleared; an edge save leaves part of it unwritten.
 Command clear={Kind::fill,save,0,{0,0,S,S},{0,0,S,S},0,0,1};o.record(clear);
}
template<class O> void copy(O& o,Id from,Id to,Rect area,Rect clip,int sx,int sy){
 Command c={Kind::copy,to,from,area,clip,sx,sy};o.record(c);
}
// One cursor move, in the order of Civ III's cursor update and present.
template<class O> void move(O& o,int k,int& px,int& py){
 int x=30+k%40,y=10+(k/3)%6+(k%17==0?32:0);
 if(k)copy(o,save,screen,{px,py,px+S,py+S},{0,0,W,H},0,0);
 // The adapter clips a save that straddles a screen edge.
 copy(o,screen,save,{0,0,S,S},{0,0,std::min(S,W-x),std::min(S,H-y)},x,y);
 Command sprite={Kind::native_sprite,screen,0,{x,y,x+S,y+S},{0,0,W,H}};o.record(sprite);
 if(k%7==3){
  // A UI element redrawn under the cursor re-saves just that part, then
  // the present draws the sprite again.
  Rect form={26,12,44,40};Command fill={Kind::fill,screen,0,form,{0,0,W,H},0,0,std::uint32_t(500+k)};o.record(fill);
  Rect cut=intersection(intersection(form,{x,y,x+S,y+S}),{0,0,W,H});
  if(cut.left<cut.right&&cut.top<cut.bottom)copy(o,screen,save,{cut.left-x,cut.top-y,cut.right-x,cut.bottom-y},{0,0,S,S},cut.left,cut.top);
  o.record(sprite);
 }
 px=x;py=y;
}
int main(){
 Current current;General general;setup(current);setup(general);int cx=0,cy=0,gx=0,gy=0;
 int largest=0;std::size_t most=0;
 for(int k=0;k<300;++k){
  move(current,k,cx,cy);move(general,k,gx,gy);
  largest=std::max(largest,chain(current,screen));most=std::max(most,current.nodes);
  if(k%25==0||k==299)for(int t:{0,5,40})for(Id id:{screen,save}){auto const& p=current.images.at(id);
   for(int y=0;y<int(p.height);++y)for(int x=0;x<int(p.width);++x)
    if(pixel<Current>(p,x,y,t)!=pixel<General>(general.images.at(id),x,y,t)){
     std::fprintf(stderr,"move %d image %d pixel %d,%d t=%d differs\n",k,int(id),x,y,t);return 1;}}
 }
 std::fprintf(stderr,"current chain=%d nodes=%zu; general chain=%d nodes=%zu\n",largest,most,chain(general,screen),general.nodes);
 // The general path is the defect being guarded: it chains every move.
 assert(chain(general,screen)>100&&general.nodes>300);
 // A restore leaves the sprite over the current fill/base, nothing deeper.
 if(largest>3||most>16){std::fprintf(stderr,"cursor restore chained: depth %d, %zu live nodes\n",largest,most);return 1;}
}
''')


if __name__ == '__main__':
    unittest.main()
