"""Execute the production fragmented-front authority without a D3D device."""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp


def method(source, signature):
    begin = source.index(signature)
    brace = source.index('{', begin)
    depth = 1
    end = brace + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[begin:end]


class RetainedFrontReuseTests(unittest.TestCase):
    def test_static_projection_reuses_only_exact_source_output_and_scale(self):
        source = (Path(__file__).parent / 'retained_composition.h').read_text()
        begin = source.index('            evaluate(original,ticks,frequency,depth+1);\n            auto source_identity=')
        end = source.index('\n        }\n        n->view_scale=scale;', begin)
        run_cpp(r'''
#include "Renderer/native/gpu_image_commands.h"
#include <memory>
#include <vector>
#include <cassert>
using namespace c3x_gpu_images;
struct Tex {};
struct Texture {std::shared_ptr<Tex> p;Tex* Get()const{return p.get();}explicit operator bool()const{return bool(p);}};
struct Node {Rect area{};Texture output[2];std::uint64_t revision=1,seen=0;float view_scale=0;std::vector<std::uint64_t> dependencies;struct {Texture write;}sample_target;};
struct Patch {unsigned output=0;};
struct Layer {unsigned draws=0;void draw(void*,void*,void*,void*,Tex*,Rect,float,float,float,float,float){++draws;}};
struct Replay {Id attach_source_unrecorded(Tex*,Format){return 1;}void* view(Id){return nullptr;}void recycle(Id){}};
struct Context {unsigned clears=0;void ClearUnorderedAccessViewUint(Tex*,unsigned*){++clears;}};
struct Owner {bool compiled_enabled=true;std::uint64_t frame=1,serial=1;void* device=nullptr;Context gpu;Context* context=&gpu;Layer projected_layer;Replay replay;
 static bool same_rect(Rect a,Rect b){return a.left==b.left&&a.top==b.top&&a.right==b.right&&a.bottom==b.bottom;}
 void evaluate(std::shared_ptr<Node> const&,long long,long long,unsigned){}
 void projected_output(Node& n,Rect area){n.area=area;if(!n.output[0])n.output[0].p=std::make_shared<Tex>();n.sample_target.write=n.output[0];}
 std::shared_ptr<Node> run(std::shared_ptr<Node> const& original,std::shared_ptr<Node> const& n,Patch patch,Rect area,float scale,unsigned width=8,unsigned height=8){long long ticks=1,frequency=1000;unsigned depth=1;
''' + source[begin:end] + r'''
 n->view_scale=scale;n->seen=frame;n->revision=++serial;return n;
 }
};
int main(){Owner o;auto original=std::make_shared<Node>(),n=std::make_shared<Node>();original->area={0,0,8,8};original->output[0].p=std::make_shared<Tex>();original->output[1].p=std::make_shared<Tex>();
 Rect area={1,2,6,7};o.run(original,n,{},area,1.25f);auto revision=n->revision;assert(o.projected_layer.draws==1);
 ++o.frame;o.run(original,n,{},area,1.25f);assert(o.projected_layer.draws==1&&n->revision==revision&&n->seen==o.frame);
 ++original->revision;o.run(original,n,{},area,1.25f);assert(o.projected_layer.draws==2&&n->revision!=revision);
 o.run(original,n,{1},area,1.25f);assert(o.projected_layer.draws==3);
 auto old=original->output[1];original->output[1].p=std::make_shared<Tex>();o.run(original,n,{1},area,1.25f);assert(o.projected_layer.draws==4);
 o.run(original,n,{1},area,1.5f);assert(o.projected_layer.draws==5);o.run(original,n,{1},{2,2,7,7},1.5f);assert(o.projected_layer.draws==6);
 o.run(original,n,{1},{2,2,7,7},1.5f,10,8);assert(o.projected_layer.draws==7);o.run(original,n,{1},{2,2,7,7},1.5f,10,10);assert(o.projected_layer.draws==8);
 ++original->area.left;o.run(original,n,{1},{2,2,7,7},1.5f,10,10);assert(o.projected_layer.draws==9);
 o.compiled_enabled=false;o.run(original,n,{1},{2,2,7,7},1.5f);o.run(original,n,{1},{2,2,7,7},1.5f);assert(o.projected_layer.draws==11);
 o.compiled_enabled=true;original->output[1]={};o.run(original,n,{1},{2,2,7,7},1.5f);assert(o.gpu.clears==1);revision=n->revision;
 o.run(original,n,{1},{2,2,7,7},1.5f);assert(o.gpu.clears==1&&n->revision==revision);
}
''')

    def test_exact_changed_fragments_and_optional_lifetime(self):
        source = (Path(__file__).parent / 'retained_composition.h').read_text()
        run_cpp(r'''
#include "Renderer/native/gpu_image_commands.h"
#include <memory>
#include <vector>
#include <map>
#include <array>
#include <chrono>
#include <string>
#include <cassert>
using namespace c3x_gpu_images;
void OutputDebugStringA(char const*){}
struct Tex {unsigned width,height;std::vector<unsigned> pixels;Tex(unsigned w,unsigned h):width(w),height(h),pixels(w*h,0) {}};
struct Texture {std::shared_ptr<Tex> p;Texture()=default;Texture(std::shared_ptr<Tex> v):p(std::move(v)){};
 Tex* Get()const{return p.get();}explicit operator bool()const{return bool(p);}void Reset(){p.reset();}};
struct D3D11_BOX {unsigned left,top,front,right,bottom,back;};
struct Context {unsigned copies=0;void CopySubresourceRegion(Tex* d,unsigned,unsigned x,unsigned y,unsigned,Tex* s,unsigned,D3D11_BOX const* b){
 assert(d&&s&&d!=s&&b->right<=s->width&&b->bottom<=s->height&&x+b->right-b->left<=d->width&&y+b->bottom-b->top<=d->height);
 for(unsigned j=b->top;j<b->bottom;++j)for(unsigned i=b->left;i<b->right;++i)d->pixels[(y+j-b->top)*d->width+x+i-b->left]=s->pixels[j*s->width+i];++copies;}
 void CopyResource(Tex* d,Tex* s){d->pixels=s->pixels;}};
using ID3D11RenderTargetView=Tex;using ID3D11Texture2D=Tex;
struct Storage {using Lease=std::shared_ptr<unsigned>;Lease retain(Tex*){return std::make_shared<unsigned>(1);}};
struct Replay {struct Target {Texture texture;};std::map<Id,Texture> images;std::map<Id,Tex*> borrowed;Id next=0;bool refuse=false;
 Id create(unsigned w,unsigned h,Format,bool){if(refuse)return 0;images[++next]=Texture(std::make_shared<Tex>(w,h));return next;}
 Target release_import_target(Id id){auto texture=images.at(id);images.erase(id);return {texture};}
 Id attach_source_unrecorded(Tex* t,Format){if(refuse)return 0;assert(t);borrowed[++next]=t;return next;}
 bool display(Id id,Tex* target,unsigned width,unsigned height,Rect area,std::array<long long,8>*){if(!target)return false;auto source=borrowed.at(id);assert(width==source->width&&height==source->height);
  for(int y=area.top;y<area.bottom;++y)for(int x=area.left;x<area.right;++x)target->pixels[y*width+x]=source->pixels[y*width+x];return true;}
 void recycle(Id id){borrowed.erase(id);images.erase(id);}};
struct Owner {
 struct Node {Rect area{};Texture output[2];std::uint64_t revision=1;unsigned pins=0;};
 struct Patch {Rect area{};std::shared_ptr<Node> node;unsigned output=0;};
 struct Picture {unsigned width=4,height=4;Format format=Format::bgra32;std::vector<Patch> patches;bool partitioned=true;};
 struct FrontPatch {Rect area{};std::weak_ptr<Node> node;unsigned output=0;std::uint64_t revision=0;};
 struct Work {unsigned operations=0,copies=0,assemblies=0;std::uint64_t copied_pixels=0,assembly_pixels=0;
  double prepare_ms=0,evaluate_ms=0,assemble_ms=0,display_ms=0;std::array<long long,8> display_ticks{};std::string executed;}work;
 // Diagnostics stay off: no held interface frames, no evaluated-operation trace, no GPU timeline.
 bool diag_ui_hold=false,trace_evaluated=false;void render_core_mark(char const*){}
 Picture front;bool compiled_enabled=true;std::uint64_t front_revision=1,used=0,frame=0,drawn_revision=0;double selected_view_scale=1;
 std::vector<std::uint64_t> drawn_dependencies,pending_drawn_dependencies;
 Texture assembled_front;Storage::Lease front_owned,front_physical;unsigned assembled_width=0,assembled_height=0;Format assembled_format=Format::bgra32;
 std::uint64_t assembled_revision=0;std::vector<FrontPatch> assembled_patches;
 static constexpr std::uint64_t resident_budget=256u*1024u*1024u;
 Storage owned_storage,storage;Replay replay;Context gpu;Context* context=&gpu;
 std::uint64_t resident_bytes()const{return used;}
 static bool empty(Rect r){return r.left>=r.right||r.top>=r.bottom;}
 static bool same_rect(Rect a,Rect b){return a.left==b.left&&a.top==b.top&&a.right==b.right&&a.bottom==b.bottom;}
 static Rect intersect(Rect a,Rect b){return intersection(a,b);}
 Rect extent(Picture const& p)const{return {0,0,int(p.width),int(p.height)};}
 bool ready()const{return front.width!=0;}void prepare_front(long long,long long){}void evaluate(std::shared_ptr<Node> const&,long long,long long,unsigned){}
 Id assemble(Picture const&,long long,long long,unsigned,Rect,bool){assert(false);return 0;}
''' + method(source, '    struct Pins {') + ';\n' + method(source, '    void release_front()') + '\n' + method(source, '    bool assemble_front(') + '\n' + method(source, '    int draw(') + r'''
};
int main(){
 Owner o;auto left=std::make_shared<Owner::Node>(),right=std::make_shared<Owner::Node>();
 left->area={0,0,2,4};right->area={2,0,4,4};left->output[0]=Texture(std::make_shared<Tex>(2,4));right->output[0]=Texture(std::make_shared<Tex>(2,4));
 std::fill(left->output[0].p->pixels.begin(),left->output[0].p->pixels.end(),1);std::fill(right->output[0].p->pixels.begin(),right->output[0].p->pixels.end(),2);
 o.front.patches={{left->area,left,0},{right->area,right,0}};Id image=0;Rect damage={};
 assert(o.assemble_front(image,damage)&&o.gpu.copies==2&&o.work.assembly_pixels==16);
 auto canvas=o.assembled_front.Get();for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)assert(canvas->pixels[y*4+x]==(x<2?1u:2u));
 o.work={};std::fill(left->output[0].p->pixels.begin(),left->output[0].p->pixels.end(),3);++left->revision;
 assert(o.assemble_front(image,damage)&&o.assembled_front.Get()==canvas&&o.work.copies==1&&o.work.copied_pixels==8&&o.work.assembly_pixels==8&&Owner::same_rect(damage,left->area));
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)assert(canvas->pixels[y*4+x]==(x<2?3u:2u));
 // An output-plane change cannot reuse an identical revision accidentally.
 right->output[1]=Texture(std::make_shared<Tex>(2,4));std::fill(right->output[1].p->pixels.begin(),right->output[1].p->pixels.end(),4);
 o.front.patches[1].output=1;++o.front_revision;o.work={};assert(o.assemble_front(image,damage)&&o.work.copied_pixels==8);
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)assert(canvas->pixels[y*4+x]==(x<2?3u:4u));
 // Same revision/area but a new owner is a different immutable source.
 auto replacement=std::make_shared<Owner::Node>();replacement->area=left->area;replacement->revision=left->revision;
 replacement->output[0]=Texture(std::make_shared<Tex>(2,4));std::fill(replacement->output[0].p->pixels.begin(),replacement->output[0].p->pixels.end(),5);
 std::weak_ptr<Owner::Node> retired=left;o.front.patches[0].node=replacement;left.reset();assert(retired.expired());
 ++o.front_revision;o.work={};assert(o.assemble_front(image,damage)&&o.work.copied_pixels==8&&canvas->pixels[0]==5);
 // Prior display failure must retry a full display even if assembly is current.
 o.work={};assert(o.assemble_front(image,damage)&&!o.work.copies&&Owner::same_rect(damage,{0,0,4,4}));
 ++o.front_revision;o.work={};assert(o.assemble_front(image,damage)&&!o.work.copied_pixels);
 // Reordering a complete partition changes traversal, not pixel ownership.
 // Any fragment whose index no longer matches is conservatively recopied.
 std::swap(o.front.patches[0],o.front.patches[1]);++o.front_revision;o.work={};
 assert(o.assemble_front(image,damage)&&o.work.copied_pixels==16);
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)assert(canvas->pixels[y*4+x]==(x<2?5u:4u));
 std::swap(o.front.patches[0],o.front.patches[1]);++o.front_revision;assert(o.assemble_front(image,damage));
 // Splitting or merging one fragment leaves the unchanged half resident.
 o.front.patches[1].area={2,0,4,2};o.front.patches.push_back({{2,2,4,4},right,1});
 ++o.front_revision;o.work={};assert(o.assemble_front(image,damage)&&o.work.copied_pixels==8);
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)assert(canvas->pixels[y*4+x]==(x<2?5u:4u));
 o.front.patches.pop_back();o.front.patches[1].area=right->area;
 ++o.front_revision;o.work={};assert(o.assemble_front(image,damage)&&o.work.copied_pixels==8);
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)assert(canvas->pixels[y*4+x]==(x<2?5u:4u));
 // Sparse/nonpartitioned, missing output, and aliased source all fail closed.
 o.front.patches.pop_back();assert(!o.assemble_front(image,damage)&&!o.assembled_front);
 o.front.patches.push_back({right->area,right,1});o.front.partitioned=false;assert(!o.assemble_front(image,damage));o.front.partitioned=true;
 right->output[1].Reset();assert(!o.assemble_front(image,damage));right->output[1]=right->output[0];assert(o.assemble_front(image,damage));
 o.front.patches[0].node->output[0]=o.assembled_front;assert(!o.assemble_front(image,damage)&&!o.assembled_front);
 o.front.patches[0].node->output[0]=Texture(std::make_shared<Tex>(2,4));
 o.used=Owner::resident_budget-63;assert(!o.assemble_front(image,damage)&&!o.assembled_front);o.used=0;
 o.replay.refuse=true;assert(!o.assemble_front(image,damage)&&!o.assembled_front);o.replay.refuse=false;assert(o.assemble_front(image,damage));
 // Existing optional storage is released on source-handle admission failure.
 o.replay.refuse=true;assert(!o.assemble_front(image,damage)&&!o.assembled_front&&o.assembled_patches.empty());
 o.replay.refuse=false;assert(o.assemble_front(image,damage));o.compiled_enabled=false;assert(!o.assemble_front(image,damage)&&!o.assembled_front);
 o.compiled_enabled=true;assert(o.assemble_front(image,damage));auto old=o.assembled_front;
 o.front.format=Format::rgb565;++o.front_revision;assert(o.assemble_front(image,damage)&&o.assembled_front.Get()!=old.Get()&&o.assembled_format==Format::rgb565);
 old=o.assembled_front;o.front.width=2;++o.front_revision;
 replacement->area={0,0,1,4};replacement->output[0]=Texture(std::make_shared<Tex>(1,4));right->area={1,0,2,4};right->output[1]=Texture(std::make_shared<Tex>(1,4));
 o.front.patches={{replacement->area,replacement,0},{right->area,right,1}};
 assert(o.assemble_front(image,damage)&&o.assembled_front.Get()!=old.Get()&&o.assembled_width==2&&o.work.copied_pixels>=8);
 o.release_front();assert(!o.assembled_front&&!o.front_owned&&!o.front_physical&&o.assembled_patches.empty());
 // A failed display's damage must survive a different subsequent change.
 Owner d;auto a=std::make_shared<Owner::Node>(),b=std::make_shared<Owner::Node>();a->area={0,0,2,4};b->area={2,0,4,4};
 a->output[0]=Texture(std::make_shared<Tex>(2,4));b->output[0]=Texture(std::make_shared<Tex>(2,4));
 std::fill(a->output[0].p->pixels.begin(),a->output[0].p->pixels.end(),1);std::fill(b->output[0].p->pixels.begin(),b->output[0].p->pixels.end(),2);
 d.front.patches={{a->area,a,0},{b->area,b,0}};Tex display(4,4),buffer(4,4);
 assert(d.draw(1,1000,&display,&display,&buffer)==1);
 std::fill(a->output[0].p->pixels.begin(),a->output[0].p->pixels.end(),3);++a->revision;
 assert(d.draw(2,1000,nullptr,&display,&buffer)==0);
 std::fill(b->output[0].p->pixels.begin(),b->output[0].p->pixels.end(),4);++b->revision;
 assert(d.draw(3,1000,&display,&display,&buffer)==1);
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)assert(display.pixels[y*4+x]==(x<2?3u:4u));assert(buffer.pixels==display.pixels);
 assert(d.draw(4,1000,&display,&display,&buffer)==2);
 // Flip-model back buffers rotate. The next target does not necessarily hold
 // the preceding complete frame, even when only one source fragment changed.
 Tex rotated(4,4);std::fill(rotated.pixels.begin(),rotated.pixels.end(),99);
 std::fill(a->output[0].p->pixels.begin(),a->output[0].p->pixels.end(),7);++a->revision;
 assert(d.draw(5,1000,&rotated,&rotated,&buffer)==1);
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)
  assert(rotated.pixels[y*4+x]==(x<2?7u:4u));
 assert(buffer.pixels==rotated.pixels);
 // The old target is stale in the other half on its next turn. Both halves
 // must agree with the retained canvas after changing the right fragment.
 std::fill(b->output[0].p->pixels.begin(),b->output[0].p->pixels.end(),8);++b->revision;
 assert(d.draw(6,1000,&display,&display,&buffer)==1);
 for(unsigned y=0;y<4;++y)for(unsigned x=0;x<4;++x)
  assert(display.pixels[y*4+x]==(x<2?7u:8u));
 assert(buffer.pixels==display.pixels);
}
''')


if __name__ == '__main__':
    unittest.main()
