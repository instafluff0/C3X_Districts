"""Execute retained publication provenance with host-side GPU boundary doubles.

Source, commit, graph inspection, and the canonical/projected sampling branches
come from production. The GPU double copies a pixel token so held-image identity
is observable; this fixture does not claim to verify D3D drawing or Present.
"""
from pathlib import Path
import unittest

from Renderer.native.native_cpp_test import run_cpp


def definition(source, signature):
    start = source.index(signature)
    # Default arguments can contain {} before the function's actual body.
    end = source.index("(", start) + 1
    depth = 1
    while depth:
        depth += (source[end] == "(") - (source[end] == ")")
        end += 1
    opening = source.index("{", end)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end] + "\n"


def production_program():
    retained = Path(__file__).with_name("retained_composition.h").read_text()
    session = Path(__file__).with_name("gpu_composition_session.h").read_text()
    sampled = retained[retained.index("    struct SampledImage {"):
                       retained.index("    struct Direct {")]
    graph = retained[retained.index("    struct Node;"):
                     retained.index("    ID3D11Device* device;")]
    canonical = definition(retained, "    void evaluate(")
    # Only source evaluation is needed. Keep its actual guard and all sampled
    # kinds, without stubbing the unrelated native-operation rendering engine.
    canonical = canonical.split("        }else if(n->selected_world){", 1)[0]
    canonical += "        }\n        n->seen=frame;\n    }\n"
    projected = definition(retained, "    std::shared_ptr<Node> evaluate_projected(")
    tail = projected[projected.rindex("        n->view_scale=scale;"):]
    projected = projected.split("        }else if(original->operation&&(original->map_dynamic||retired_underlay)){", 1)[0]
    projected += "        }\n" + tail
    # The projected preamble classifies retired underlays with production
    # projectable(), which records the first unprojected operation.
    refusal = retained[retained.index("    struct Refusal {"):retained.index("    Rect project(")]
    methods = "\n".join(definition(retained, signature) for signature in (
        "    Rect intersect(", "    bool empty(", "    Rect extent(", "    static bool same_rect(",
        "    Picture read(", "    void replace(", "    void write(",
        "    void collect(", "    void prepare(", "    Rect project(",
        "    void projected_output(", "    bool accepting(", "    void create(",
        "    void snapshot(", "    void destroy(", "    void source(",
        "    void commit(", "    void uncommit(",
        "    std::pair<std::uint64_t,std::uint64_t> front_publication(",
    ))
    publish = definition(session, "    bool publish_source(")
    inspect = definition(session, "    std::pair<std::uint64_t,std::uint64_t> visual_publication(")
    # Host Clang rejects a nested defaulted Sample constructor used by a
    # default argument before its enclosing class ends. Hoist the unchanged
    # production types and alias them back into the fixture class.
    return DOUBLE + sampled + r'''
struct RetainedComposition {
    using SampledImage=::SampledImage;using Sample=::Sample;
    struct Placed {Command command;int x=0,y=0;};
    struct Direct {
        bool animated=false;std::uint64_t input_bytes=0;
        std::function<std::uint64_t(long long,long long)> revision;
        std::function<bool(Compositor&,Command const&)> draw;
    };
''' + graph + r'''
    ID3D11Device owned_device;ID3D11DeviceContext owned_context;
    ID3D11Device* device=&owned_device;ID3D11DeviceContext* context=&owned_context;
    Compositor replay;ProjectedLayer projected_layer;
    std::map<Id,Picture> images;Picture front;
    std::uint64_t serial=0,frame=0,front_revision=0,sample_allocations=0,sample_imports=0;
    bool admitted=true;
    // This fixture evaluates publication graphs directly, without compiling
    // the production collect/prepare plans. Keep their invalidation boundary.
    std::vector<int> collect_plan,prepare_plan;
    void invalidate_plan(){}
    // No assembled front exists in this fixture; uncommit still releases it.
    void release_front(){}
    struct {float stretch=1.f;} work; // diagnostic stretch only
    std::shared_ptr<Node> node(){return std::make_shared<Node>();}
    void reserve(std::uint64_t,char const*){}
    void output(Node& n,unsigned i,Texture texture){n.output[i]=std::move(texture);}
    Texture crop(ID3D11Texture2D* texture,Rect area){
        auto id=replay.create(area.right-area.left,area.bottom-area.top,Format::bgra32,false);
        auto result=replay.release_import_target(id);result.texture->pixel=texture->pixel;
        return result.texture;
    }
    void clear(){images.clear();front={};admitted=true;}
    void discard(){images.clear();front={};admitted=false;}
''' + refusal + methods + canonical + projected + r'''
    void canonical_frame(){
        ++frame;
        for(auto const& p:front.patches)collect(p.node,frame,1000,0);
        for(auto const& p:front.patches)prepare(p.node,frame,1000,0);
        for(auto const& p:front.patches)evaluate(p.node,frame,1000,0);
    }
    void projected_frame(float scale){
        ++frame;
        for(auto const& p:front.patches)collect(p.node,frame,1000,0,true);
        for(auto const& p:front.patches)prepare(p.node,frame,1000,0,true,scale);
        for(auto const& p:front.patches)evaluate_projected(p,frame,1000,0,scale,front.width,front.height);
    }
};
struct Session {
    static constexpr unsigned live_image_budget=256u*1024u*1024u;
    Compositor gpu;RetainedComposition layers;
    Id map=0;std::int64_t ticket=0,identity=0;bool map_animation_expected=false;
    unsigned world_width=0,world_height=0;Id world_destination=0,hud_canvas=0,hud_detail=0;
    std::vector<int> hud,fixed_shadows,overlays;
    std::shared_ptr<c3x_renderer::render_core::UnitHudAnchors> unit_anchors;
''' + publish + inspect + r'''
};
using Proof=std::pair<std::uint64_t,std::uint64_t>;
using Sample=RetainedComposition::Sample;
using Sampled=RetainedComposition::SampledImage;
constexpr Rect full{0,0,96,64};
void map_source(RetainedComposition& layers,Id id,ID3D11Texture2D* texture,
                std::uint64_t publication,std::uint64_t generation,Sample sample={}){
    layers.create(id,96,64,Format::bgra32);
    layers.source(id,texture,std::move(sample),true,true,publication,generation);
}
'''


DOUBLE = r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include "Renderer/native/scene_projection.h"
#include "Renderer/native/zoom_transition.h"
#include "Renderer/native/gpu_image_commands.h"
#include "Renderer/native/render_core/unit_hud_anchors.h"
#include <cstring>
using namespace c3x_gpu_images;
struct D3D11_TEXTURE2D_DESC {unsigned Width=96,Height=64;};
struct ID3D11Texture2D;
struct ID3D11ShaderResourceView {ID3D11Texture2D* texture=nullptr;};
struct ID3D11Texture2D {
    unsigned width=96,height=64,pixel=0;
    ID3D11ShaderResourceView view{this};
    explicit ID3D11Texture2D(unsigned value=0):pixel(value){}
    void GetDesc(D3D11_TEXTURE2D_DESC* d){*d={width,height};}
};
struct ID3D11Device {};
struct ID3D11DeviceContext {};
template<class T>struct ComPtr {
    T* value=nullptr;ComPtr()=default;ComPtr(T* v):value(v){}
    T* Get()const{return value;}T* operator->()const{return value;}
    explicit operator bool()const{return value!=nullptr;}
    void Reset(){value=nullptr;}
};
using Texture=ComPtr<ID3D11Texture2D>;
struct CompositionStorage {struct Lease {};};
template<std::size_t N,class... Args>
void sprintf_s(char(&out)[N],char const* format,Args... args){std::snprintf(out,N,format,args...);}
void OutputDebugStringA(char const*){}
struct Compositor {
    struct SpatialSource {};
    struct SpatialPlan {};
    struct ImportTarget {
        Texture texture;ComPtr<ID3D11ShaderResourceView> write;
        unsigned width=0,height=0;
    };
    struct Stats {unsigned long long resident_bytes=0;};
    std::vector<std::unique_ptr<ID3D11Texture2D>> storage;
    std::map<Id,ID3D11Texture2D*> images;Id next=0;
    unsigned allocations=0,imports=0,attachments=0,visits=0;
    bool reject_allocation=false,reject_import=false;
    Stats stats()const{return {};}
    Id create(unsigned width,unsigned height,Format,bool=false){
        if(reject_allocation)return 0;
        auto texture=std::make_unique<ID3D11Texture2D>();texture->width=width;texture->height=height;
        auto id=++next;images[id]=texture.get();storage.push_back(std::move(texture));++allocations;return id;
    }
    Id attach_source(ID3D11Texture2D* texture){++attachments;auto id=++next;images[id]=texture;return id;}
    Id attach_source_unrecorded(ID3D11Texture2D* texture,Format){return attach_source(texture);}
    ImportTarget release_import_target(Id id){auto* t=images.at(id);return {t,&t->view,t->width,t->height};}
    ID3D11ShaderResourceView* view(Id id){return &images.at(id)->view;}
    ID3D11Texture2D* texture(Id id){return images.at(id);}
    bool import_bgra(ImportTarget const& target,ID3D11Texture2D* source,int,int,float=0){
        if(reject_import)return false;++imports;target.texture->pixel=source->pixel;return true;
    }
    bool import_bgra(Id id,ID3D11Texture2D* source,int x,int y){return import_bgra(release_import_target(id),source,x,y);}
    void destroy(Id id){images.erase(id);}
    void recycle(Id id){images.erase(id);}
    template<class F>void visit_images(F f){++visits;for(auto const& p:images)f(p.first,p.second->width,p.second->height,Format::bgra32,p.second);}
};
struct ProjectedLayer {
    unsigned draws=0;
    void draw(ID3D11Device*,ID3D11DeviceContext*,ID3D11ShaderResourceView* source,
              void*,ID3D11ShaderResourceView* target,Rect,float,float,float,float,float){
        ++draws;target->texture->pixel=source->texture->pixel;
    }
};
'''


class RoutePublicationTests(unittest.TestCase):
    def execute(self, body):
        run_cpp(production_program() + "\nint main(){\n" + body + "\n}\n")

    def test_session_publishes_initial_generation_without_sampling(self):
        self.execute(r'''
Session session;ID3D11Texture2D initial(10),other(20);unsigned samples=0,prepares=0;
Sample sample=[&](long long,long long){++samples;return Sampled::bgra(&other,full,0,22);};
sample.source_generation=11;sample.prepare=[&](long long,long long,float){++prepares;};
assert(session.publish_source(&initial,7,0,0,96,64,sample,true));
session.layers.commit(session.map,full);
assert(session.visual_publication()==Proof(7,11));assert(samples==0&&prepares==0);
auto previous=session.map;
assert(!session.publish_source(&other,7,0,0,96,64,sample,true));
assert(session.map==previous&&session.visual_publication()==Proof(7,11));
session.gpu.reject_import=true;
assert(!session.publish_source(&other,8,0,0,96,64,sample,false));
assert(session.map==previous&&session.ticket==7&&session.visual_publication()==Proof(7,11));
session.layers.canonical_frame();
assert(session.visual_publication()==Proof(7,22));assert(samples==1&&prepares==1);
''')

    def test_canonical_sampling_tracks_completed_generation_and_held_front(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10),fresh(20);unsigned samples=0,prepares=0;
auto kind=Sampled::Kind::bgra;std::uint64_t generation=22;
Sample sample=[&](long long,long long){++samples;
    if(kind==Sampled::Kind::bgra)return Sampled::bgra(&fresh,full,0,generation);
    Sampled result;result.kind=kind;return result;};
sample.prepare=[&](long long,long long,float){++prepares;};
map_source(layers,1,&initial,7,11,sample);layers.commit(1,full);
layers.canonical_frame();auto node=layers.front.patches[0].node;
assert(layers.front_publication()==Proof(7,22));assert(node->output[0]->pixel==20);
auto imports=layers.replay.imports;
for(auto pending:{Sampled::Kind::unchanged,Sampled::Kind::held}){
    kind=pending;fresh.pixel=30;generation=33;layers.canonical_frame();
    assert(layers.front_publication()==Proof(7,22));assert(node->output[0]->pixel==20);
    assert(layers.replay.imports==imports&&node->sample);
}
kind=Sampled::Kind::bgra;layers.canonical_frame();
assert(layers.front_publication()==Proof(7,33));assert(node->output[0]->pixel==30);
kind=Sampled::Kind::frozen;layers.canonical_frame();
assert(layers.front_publication()==Proof(7,33));assert(!node->sample&&node->retired);
auto sampled=samples,prepared=prepares;layers.canonical_frame();
assert(samples==sampled&&prepares==prepared&&layers.front_publication()==Proof(7,33));
''')

    def test_projected_hold_and_retirement_keep_previous_projected_generation(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10),fresh(20);unsigned samples=0,prepares=0;
auto kind=Sampled::Kind::bgra;std::uint64_t generation=22;
Sample sample=[&](long long,long long){return Sampled::bgra(&fresh,full,0,99);};
sample.projected=[&](long long,long long,float){++samples;
    if(kind==Sampled::Kind::bgra)return Sampled::bgra(&fresh,full,0,generation);
    Sampled result;result.kind=kind;return result;};
sample.prepare=[&](long long,long long,float){++prepares;};
map_source(layers,1,&initial,7,11,sample);layers.commit(1,full);
layers.projected_frame(1.5f);auto original=layers.front.patches[0].node;
auto projected=original->projected;
assert(layers.front_publication()==Proof(7,22));assert(projected->output[0]->pixel==20);
// The canonical result can advance independently. A held projection still
// displays its own previous image, including after a projection-scale change.
original->source_generation=99;initial.pixel=99;kind=Sampled::Kind::held;
auto draws=layers.projected_layer.draws,imports=layers.replay.imports;
layers.projected_frame(1.5f);
assert(layers.front_publication()==Proof(7,22));assert(projected->output[0]->pixel==20);
assert(layers.projected_layer.draws==draws&&layers.replay.imports==imports&&original->sample);
layers.projected_frame(2.f);
assert(layers.front_publication()==Proof(7,22));assert(projected->output[0]->pixel==20);
assert(layers.projected_layer.draws==draws+1&&layers.replay.imports==imports);
kind=Sampled::Kind::frozen;layers.projected_frame(2.5f);
assert(layers.front_publication()==Proof(7,22));assert(projected->output[0]->pixel==20);
assert(original->retired&&!original->sample);
auto sampled=samples,prepared=prepares;layers.projected_frame(3.f);
assert(layers.front_publication()==Proof(7,22));assert(samples==sampled&&prepares==prepared);
''')

    def test_first_projected_hold_uses_initial_source_and_later_sample_adopts(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10),fresh(20);bool ready=false;unsigned prepares=0;
Sample sample=[](long long,long long){return Sampled{};};
sample.projected=[&](long long,long long,float){return ready?Sampled::bgra(&fresh,full,0,22):Sampled::held();};
sample.prepare=[&](long long,long long,float){++prepares;};
map_source(layers,1,&initial,7,11,sample);layers.commit(1,full);layers.projected_frame(2.f);
auto original=layers.front.patches[0].node;
assert(layers.front_publication()==Proof(7,11));assert(original->projected->output[0]->pixel==10);
assert(original->sample&&!original->retired&&prepares==1);
ready=true;layers.projected_frame(2.f);
assert(layers.front_publication()==Proof(7,22));assert(original->projected->output[0]->pixel==20&&prepares==2);
''')

    def test_first_projected_retirement_preserves_initial_publication(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10);unsigned samples=0,prepares=0;
Sample sample=[](long long,long long){return Sampled{};};
sample.projected=[&](long long,long long,float){++samples;return Sampled::frozen();};
sample.prepare=[&](long long,long long,float){++prepares;};
map_source(layers,1,&initial,7,11,sample);layers.commit(1,full);layers.projected_frame(2.f);
auto original=layers.front.patches[0].node;
assert(layers.front_publication()==Proof(7,11));assert(original->projected->output[0]->pixel==10);
assert(original->retired&&!original->sample&&samples==1&&prepares==1);
layers.projected_frame(3.f);
assert(layers.front_publication()==Proof(7,11));assert(original->projected->output[0]->pixel==10);
assert(samples==1&&prepares==1&&layers.replay.imports==0);
''')

    def test_immutable_source_versions_remain_bound_to_committed_front(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10),fresh(20);
map_source(layers,1,&initial,7,11);layers.commit(1,full);
auto old=layers.front.patches[0].node;
map_source(layers,1,&fresh,8,22);
assert(layers.front_publication()==Proof(7,11));assert(old->output[0].Get()==&initial);
layers.canonical_frame();assert(layers.front_publication()==Proof(7,11));
layers.commit(1,full);assert(layers.front_publication()==Proof(8,22));
layers.destroy(1);assert(layers.front_publication()==Proof(8,22));
assert(layers.front.patches[0].node->output[0].Get()==&fresh);
layers.uncommit();assert(layers.front_publication()==Proof());
''')

    def test_immutable_sample_uses_actual_generation_and_refuses_unknown(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10),fresh(20);std::uint64_t generation=22;
Sample sample=[&](long long,long long){Sampled image{Texture(&fresh)};image.generation=generation;return image;};
map_source(layers,1,&initial,7,11,sample);layers.commit(1,full);layers.canonical_frame();
auto original=layers.front.patches[0].node;
assert(original->output[0].Get()==&fresh&&layers.front_publication()==Proof(7,22));
generation=33;layers.canonical_frame();assert(layers.front_publication()==Proof(7,33));
generation=0;layers.canonical_frame();assert(layers.front_publication()==Proof());
assert(layers.replay.imports==0);
''')

    def test_partial_front_refuses_different_publications_or_generations(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10),fresh(20);
map_source(layers,1,&initial,7,11);layers.commit(1,full);
map_source(layers,2,&fresh,8,22);layers.commit(2,{48,0,96,64});
assert(layers.front.patches.size()==2&&layers.front_publication()==Proof());
layers.commit(1,full);map_source(layers,2,&fresh,7,22);layers.commit(2,{48,0,96,64});
assert(layers.front_publication()==Proof());
layers.commit(1,full);map_source(layers,2,&fresh,7,11);layers.commit(2,{48,0,96,64});
assert(layers.front_publication()==Proof(7,11));
''')

    def test_unknown_map_source_cannot_certify_a_mixed_front(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D known(10),unknown(20);
map_source(layers,1,&known,7,11);layers.commit(1,full);
map_source(layers,2,&unknown,0,0);layers.commit(2,{48,0,96,64});
assert(layers.front_publication()==Proof()&&"unknown map pixels must prevent a route proof");
// Ordinary immutable native UI is untagged and must not hide the known map.
layers.commit(1,full);layers.create(3,96,64,Format::bgra32);
layers.source(3,&unknown,{},true,false);layers.commit(3,{48,0,96,64});
assert(layers.front_publication()==Proof(7,11));
''')

    def test_unknown_sampled_or_projected_generation_refuses_proof(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10),fresh(20);std::uint64_t generation=0;
Sample sample=[&](long long,long long){return Sampled::bgra(&fresh,full,0,generation);};
sample.projected=[&](long long,long long,float){return Sampled::bgra(&fresh,full,0,generation);};
map_source(layers,1,&initial,7,0,sample);layers.commit(1,full);
assert(layers.front_publication()==Proof());
generation=22;layers.canonical_frame();assert(layers.front_publication()==Proof(7,22));
generation=0;layers.projected_frame(2.f);
assert(layers.front_publication()==Proof()&&"unknown projected generation cannot use canonical generation");
generation=33;layers.projected_frame(2.f);assert(layers.front_publication()==Proof(7,33));
''')

    def test_inspection_visits_nested_sources_without_callbacks_or_gpu_work(self):
        self.execute(r'''
RetainedComposition layers;ID3D11Texture2D initial(10);unsigned calls=0;
Sample sample=[&](long long,long long){++calls;throw std::runtime_error("inspection sampled");return Sampled{};};
sample.projected=[&](long long,long long,float){++calls;throw std::runtime_error("inspection projected");return Sampled{};};
sample.prepare=[&](long long,long long,float){++calls;throw std::runtime_error("inspection prepared");};
map_source(layers,1,&initial,7,11,sample);
auto map=layers.images[1].patches[0].node;
auto overlay=layers.node();overlay->area=full;
overlay->inputs[0]=layers.images[1];overlay->batch.resize(1);
overlay->batch[0].inputs[2]=layers.images[1];
overlay->direct.revision=[&](long long,long long){++calls;throw std::runtime_error("inspection revision");return 0;};
layers.create(2,96,64,Format::bgra32);layers.images[2].patches={{full,overlay,0}};layers.commit(2,full);
auto frame=layers.frame,serial=layers.serial,revision=layers.front_revision;
auto allocations=layers.replay.allocations,imports=layers.replay.imports,attachments=layers.replay.attachments;
for(unsigned i=0;i<10;++i)assert(layers.front_publication()==Proof(7,11));
assert(calls==0&&layers.frame==frame&&layers.serial==serial&&layers.front_revision==revision);
assert(layers.replay.allocations==allocations&&layers.replay.imports==imports&&layers.replay.attachments==attachments);
// A projection cached for another frame cannot relabel the canonical front.
map->projected=layers.node();map->projected->map_source=true;
map->projected->publication=8;map->projected->source_generation=22;
map->projected_frame=frame+1;assert(layers.front_publication()==Proof(7,11));
map->projected_frame=frame;assert(layers.front_publication()==Proof(8,22));
assert(calls==0);layers.uncommit();assert(layers.front_publication()==Proof());
''')


if __name__ == "__main__":
    unittest.main()
